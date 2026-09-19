#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path

OPERATIONS = ["ATTEND", "MAINTAIN", "SWITCH", "SUPPRESS", "IGNORE"]
SOURCES = ["goal", "observation", "memory", "tool", "belief_conflict", "system_event", "unfinished_task"]
REASON_CODES = {
    "ATTEND": "NO_CURRENT_FOCUS",
    "MAINTAIN": "CURRENT_FOCUS_STILL_DOMINANT",
    "SWITCH": "HIGHER_PRIORITY_INTERRUPT",
    "SUPPRESS": "LOW_CONFIDENCE_DISTRACTOR",
    "IGNORE": "NO_COGNITIVE_VALUE",
}

SYSTEM_PROMPT = (
    "You are Omega Attention v0.1. Given a structured cognitive attention input, "
    "select exactly one OCL attention operation. Output only valid compact JSON "
    "with keys: operation, target, confidence, reason_code. Never output prose."
)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def score(candidate):
    return (
        candidate["activation"] * 0.35
        + candidate["salience"] * 0.24
        + candidate["goal_relevance"] * 0.20
        + candidate["novelty"] * 0.11
        + candidate["confidence"] * 0.10
        - candidate["age"] * 0.04
    )


def make_candidate(rng, index, force_low=False):
    source = rng.choice(SOURCES)
    base = 0.15 if force_low else 0.45
    return {
        "id": f"{source}_{index}",
        "source": source,
        "activation": round(clamp(rng.uniform(base, 1.0)), 3),
        "salience": round(clamp(rng.uniform(base, 1.0)), 3),
        "confidence": round(clamp(rng.uniform(0.25 if force_low else 0.55, 1.0)), 3),
        "goal_relevance": round(clamp(rng.uniform(0.0, 1.0)), 3),
        "novelty": round(clamp(rng.uniform(0.0, 1.0)), 3),
        "age": round(rng.uniform(0.0, 4.0), 3),
    }


def label(input_obj):
    candidates = input_obj["candidates"]
    current_focus = input_obj["current_focus"]
    best = max(candidates, key=score) if candidates else None
    if best is None:
        return {
            "operation": "IGNORE",
            "target": None,
            "confidence": 0.8,
            "reason_code": REASON_CODES["IGNORE"],
        }

    best_score = score(best)
    current = next((c for c in candidates if c["id"] == current_focus), None)
    current_score = score(current) if current else None

    if best["confidence"] < 0.35 and best["salience"] < 0.35:
        operation = "SUPPRESS"
        target = best["id"]
    elif current is None:
        operation = "ATTEND"
        target = best["id"]
    elif best["id"] == current["id"] or (current_score is not None and best_score - current_score < 0.12):
        operation = "MAINTAIN"
        target = current["id"]
    elif best_score > (current_score or 0) + 0.12:
        operation = "SWITCH"
        target = best["id"]
    else:
        operation = "IGNORE"
        target = best["id"]

    margin = best_score - (current_score or 0.0) if current else best_score
    confidence = round(clamp(0.62 + abs(margin) * 0.22), 3)
    return {
        "operation": operation,
        "target": target,
        "confidence": confidence,
        "reason_code": REASON_CODES[operation],
    }


def make_example(rng, index):
    count = rng.randint(2, 7)
    force_low = rng.random() < 0.08
    candidates = [make_candidate(rng, i, force_low=force_low) for i in range(count)]

    current_focus = None
    if rng.random() < 0.72:
        current_focus = rng.choice(candidates)["id"]
    if rng.random() < 0.10:
        current_focus = f"missing_focus_{index}"

    input_obj = {
        "protocol": "omega-attention-scenario/v0.1",
        "current_focus": current_focus,
        "working_memory_capacity": rng.choice([4, 5, 7]),
        "candidates": candidates,
    }
    output = label(input_obj)
    user = "Select the next attention operation for this cognitive workspace:\n" + json.dumps(
        input_obj, separators=(",", ":")
    )
    assistant = json.dumps(output, separators=(",", ":"))
    return {
        "id": f"attention_synth_{index}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "input": input_obj,
        "output": output,
        "metadata": {"generator": "omega_attention_v0_heuristic"},
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--eval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out-dir", default="training/attention-v0/data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    train = [make_example(rng, i) for i in range(args.train)]
    eval_rows = [make_example(rng, args.train + i) for i in range(args.eval)]
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    print(f"wrote {len(train)} train rows to {out_dir / 'train.jsonl'}")
    print(f"wrote {len(eval_rows)} eval rows to {out_dir / 'eval.jsonl'}")


if __name__ == "__main__":
    main()
