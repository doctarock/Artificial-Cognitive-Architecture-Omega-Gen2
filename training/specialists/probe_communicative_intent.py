#!/usr/bin/env python3
"""Probe a loopback Ollama model on Omega's closed intent task.

The fixed fixture is generic and contains no ACA runtime data. This is an
admission probe, not a training set: a model must parse every response, clear
aggregate and per-class quality gates, and meet the latency gate before the
script calls it eligible for prospective shadow validation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.parse
import urllib.request


CASES = [
    ("speak", "The operator should be told that the backup completed successfully."),
    ("speak", "Report that the requested calculation produced 42."),
    ("speak", "This safety warning is relevant and needs to be voiced now."),
    ("speak", "Tell the user that the file was saved in the requested location."),
    ("speak", "The answer is ready: the two values are equal."),
    ("speak", "It is worth mentioning that the deadline is tomorrow morning."),
    ("speak", "Share the conclusion that the test passed without errors."),
    ("speak", "The user asked for the result, so state that no matches were found."),
    ("speak", "Do not ask another question; provide the completed summary."),
    ("speak", "Although this is brief, the useful result should be said aloud."),
    ("ask", "The destination folder is missing; ask the user which folder to use."),
    ("ask", "Two dates are possible, so request clarification before continuing."),
    ("ask", "I cannot tell which account they mean and should ask which one."),
    ("ask", "The request says 'change it' but never identifies what 'it' is."),
    ("ask", "Before deleting anything, ask exactly which temporary file is intended."),
    ("ask", "The desired output format is ambiguous and needs a clarifying question."),
    ("ask", "I have an answer, but first I need to know whether they mean UTC or local time."),
    ("ask", "Do not guess the missing quantity; request it from the user."),
    ("ask", "The instruction conflicts with an earlier one, so ask which takes priority."),
    ("ask", "A required image was not attached; ask the user to attach it."),
    ("ignore", "This is a private implementation note that does not need to be voiced."),
    ("ignore", "Nothing useful or actionable came from this reflection."),
    ("ignore", "The same point was already communicated, so remain silent."),
    ("ignore", "This is merely a transient formatting observation with no relevance."),
    ("ignore", "Do not mention this internal bookkeeping detail to the user."),
    ("ignore", "There is no question to ask and no result worth announcing."),
    ("ignore", "The thought is an abandoned draft and should not be spoken."),
    ("ignore", "Although it contains a question mark, it is only a note about punctuation?"),
    ("ignore", "A random association surfaced, but it is unrelated to the current task."),
    ("ignore", "Keep this redundant status update internal because nothing changed."),
]


def prompt_for(reflection: str) -> str:
    return (
        "You are deciding what to do about an already-formed internal reflection.\n"
        "Choose exactly one action:\n"
        '- "speak": it contains a result, warning, or useful conclusion worth voicing\n'
        '- "ask": progress requires a clarifying question\n'
        '- "ignore": nothing should be said right now\n\n'
        f"Reflection: {reflection}\n\n"
        'Respond with ONLY JSON: {"response":"speak|ask|ignore"}'
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def generate(base_url: str, model: str, prompt: str, timeout: float) -> tuple[str, float]:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "api/generate")
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 32},
        }
    ).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.load(response)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return str(envelope.get("response", "")), elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-class-recall", type=float, default=0.80)
    parser.add_argument("--max-p95-ms", type=float, default=1500.0)
    args = parser.parse_args()

    host = urllib.parse.urlparse(args.base_url).hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the probe refuses non-loopback endpoints")

    try:
        generate(args.base_url, args.model, prompt_for("Nothing should be said."), args.timeout_seconds)
    except (OSError, TimeoutError) as error:
        print(
            json.dumps(
                {
                    "schema": "omega-communicative-intent-probe/v1",
                    "model": args.model,
                    "samples": 0,
                    "probe_error": f"warmup failed: {type(error).__name__}",
                    "eligible_for_shadow": False,
                },
                indent=2,
            )
        )
        return 1

    latencies: list[float] = []
    correct = 0
    parsed = 0
    totals = {label: 0 for label in ("speak", "ask", "ignore")}
    hits = {label: 0 for label in totals}
    failures: list[dict[str, str]] = []
    for expected, reflection in CASES:
        totals[expected] += 1
        try:
            raw, latency = generate(args.base_url, args.model, prompt_for(reflection), args.timeout_seconds)
        except (OSError, TimeoutError) as error:
            failures.append(
                {
                    "expected": expected,
                    "actual": f"<{type(error).__name__}>",
                    "reflection": reflection,
                }
            )
            continue
        latencies.append(latency)
        try:
            actual = str(json.loads(raw).get("response", "")).strip().lower()
        except (json.JSONDecodeError, AttributeError):
            actual = "<unparseable>"
        if actual in totals:
            parsed += 1
        if actual == expected:
            correct += 1
            hits[expected] += 1
        else:
            failures.append({"expected": expected, "actual": actual, "reflection": reflection})

    accuracy = correct / len(CASES)
    recalls = {label: hits[label] / totals[label] for label in totals}
    report = {
        "schema": "omega-communicative-intent-probe/v1",
        "model": args.model,
        "samples": len(CASES),
        "parsed": parsed,
        "accuracy": round(accuracy, 4),
        "class_recall": {key: round(value, 4) for key, value in recalls.items()},
        "latency_ms": {
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "failures": failures,
    }
    eligible = (
        parsed == len(CASES)
        and accuracy >= args.min_accuracy
        and min(recalls.values()) >= args.min_class_recall
        and report["latency_ms"]["p95"] is not None
        and report["latency_ms"]["p95"] <= args.max_p95_ms
    )
    report["eligible_for_shadow"] = eligible
    print(json.dumps(report, indent=2))
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
