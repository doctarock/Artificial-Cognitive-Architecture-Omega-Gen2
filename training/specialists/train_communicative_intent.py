#!/usr/bin/env python3
"""Train a tiny dependency-free Speak/Ask/Ignore softmax specialist.

Training examples are generic authored intent statements. Evaluation uses the
separate fixed admission fixture from probe_communicative_intent.py. The saved
model is shadow-only: held-out intent accuracy does not establish agreement or
utility on ACA's live reflection distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from probe_communicative_intent import CASES as EVAL_CASES


LABELS = ["speak", "ask", "ignore"]
DIMENSIONS = 1024

TRAIN_SEEDS = {
    "speak": [
        "Announce that the operation finished.", "Give the user the final answer.",
        "State the result clearly.", "Share the warning immediately.",
        "Say that the upload succeeded.", "Communicate the useful conclusion.",
        "Let them know the requested work is complete.", "Provide the computed value now.",
        "Voice this important finding.", "Return the finished summary.",
        "Mention the newly discovered deadline.", "Explain that validation passed.",
        "Deliver the answer without another question.", "Report the successful outcome.",
        "Tell them there were no errors.", "Present the result they requested.",
        "This conclusion is relevant enough to say.", "The operator needs to hear this warning.",
        "Respond with the available result.", "Make the completed status known.",
        "The finding should be communicated, not kept internal.", "Provide the concise answer.",
        "Inform the user that the record was created.", "Say the comparison found a match.",
    ],
    "ask": [
        "Request the missing filename.", "Ask which option the user intended.",
        "Seek clarification about the ambiguous date.", "Find out which account they mean.",
        "Pose a clarifying question before proceeding.", "Request the absent attachment.",
        "Ask whether local time or UTC is wanted.", "Do not guess; ask for the quantity.",
        "Clarify what the pronoun refers to.", "Ask which instruction has priority.",
        "Request confirmation of the destructive target.", "Find out the required output format.",
        "Ask the user to choose between the two paths.", "A necessary parameter is missing; request it.",
        "Clarify the destination before saving.", "Request permission details that were omitted.",
        "The scope is unclear, so ask about it.", "Obtain the missing version number.",
        "Ask a question to resolve the conflict.", "Request the image needed to continue.",
        "Clarification is required before an answer is possible.", "Ask which project is in scope.",
        "Find out what 'that' means here.", "Request the exact folder rather than assuming.",
    ],
    "ignore": [
        "Keep this irrelevant implementation note internal.", "Remain silent because nothing changed.",
        "Discard this abandoned draft without speaking.", "This random association is unrelated.",
        "Do not voice the private bookkeeping detail.", "No response or clarification is needed.",
        "The point was already communicated, so say nothing.", "Ignore this transient formatting thought.",
        "This reflection has no useful content.", "Keep the redundant update internal.",
        "Nothing actionable resulted from this thought.", "Do not repeat the answer again.",
        "This internal note is not for the user.", "The observation is irrelevant to the task.",
        "Stay quiet; there is no new information.", "Suppress this obsolete draft.",
        "Neither a statement nor a question is warranted.", "This is merely internal housekeeping.",
        "The association should pass without comment.", "No outward action should follow.",
        "Silence is appropriate for this duplicate thought.", "Leave this low-value note unspoken.",
        "There is nothing worth announcing or asking.", "Ignore the unrelated mental noise.",
    ],
}


def fnv1a(text: str) -> int:
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def feature_strings(text: str) -> list[str]:
    normalized = " ".join(re.findall(r"[a-z0-9']+", text.lower()))
    words = normalized.split()
    features = [f"w:{word}" for word in words]
    features += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    padded = f"  {normalized}  "
    for width in (3, 4, 5):
        features += [f"c:{padded[i:i + width]}" for i in range(len(padded) - width + 1)]
    return features


def vectorize(text: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for feature in feature_strings(text):
        hashed = fnv1a(feature)
        index = hashed % DIMENSIONS
        sign = 1.0 if (hashed >> 63) == 0 else -1.0
        values[index] = values.get(index, 0.0) + sign
    norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
    return {index: value / norm for index, value in values.items()}


def probabilities(weights: list[list[float]], bias: list[float], features: dict[int, float]) -> list[float]:
    logits = [bias[label] + sum(weights[label][index] * value for index, value in features.items()) for label in range(len(LABELS))]
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def evaluate(weights: list[list[float]], bias: list[float]) -> dict[str, object]:
    hits = {label: 0 for label in LABELS}
    totals = {label: 0 for label in LABELS}
    failures = []
    for expected, text in EVAL_CASES:
        totals[expected] += 1
        probs = probabilities(weights, bias, vectorize(text))
        actual = LABELS[max(range(len(LABELS)), key=lambda index: probs[index])]
        if actual == expected:
            hits[expected] += 1
        else:
            failures.append({"expected": expected, "actual": actual, "text": text})
    accuracy = sum(hits.values()) / len(EVAL_CASES)
    recall = {label: hits[label] / totals[label] for label in LABELS}
    return {"samples": len(EVAL_CASES), "accuracy": accuracy, "class_recall": recall, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.4)
    args = parser.parse_args()

    examples = [(LABELS.index(label), vectorize(text)) for label in LABELS for text in TRAIN_SEEDS[label]]
    weights = [[0.0] * DIMENSIONS for _ in LABELS]
    bias = [0.0] * len(LABELS)
    for epoch in range(args.epochs):
        rate = args.learning_rate / (1.0 + 0.002 * epoch)
        for expected, features in examples:
            probs = probabilities(weights, bias, features)
            for label in range(len(LABELS)):
                error = (1.0 if label == expected else 0.0) - probs[label]
                bias[label] += rate * error
                for index, value in features.items():
                    weights[label][index] += rate * (error * value - 0.0001 * weights[label][index])

    metrics = evaluate(weights, bias)
    eligible = (
        metrics["samples"] >= 30
        and metrics["accuracy"] >= 0.90
        and min(metrics["class_recall"].values()) >= 0.80
    )
    report = {**metrics, "eligible_for_shadow": eligible}
    print(json.dumps(report, indent=2))
    if not eligible:
        return 1
    artifact = {
        "schema": "omega-communicative-intent-softmax/v1",
        "status": "shadow_only",
        "labels": LABELS,
        "dimensions": DIMENSIONS,
        "feature_schema": "fnv1a-word-bigram-char3-5-l2/v1",
        "weights": weights,
        "bias": bias,
        "held_out": {"samples": metrics["samples"], "accuracy": metrics["accuracy"], "class_recall": metrics["class_recall"]},
        "limitations": "Generic authored intent benchmark; requires prospective ACA shadow validation before control use.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
