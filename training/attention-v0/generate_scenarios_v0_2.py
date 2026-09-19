#!/usr/bin/env python3
"""Generate reproducible synthetic v0.2 attention workspaces and labels.

These labels are a policy heuristic, not real-world outcome feedback. Keep the
JSON shape synchronized with crates/aca-engine/src/attention_workspace.rs.
"""

import argparse
import json
import random
import uuid
from pathlib import Path

PROTOCOL = "omega-attention-workspace/v0.2"
SYSTEM_PROMPT = (
    "You are Omega Attention v0.2. Select one attention state-transition "
    "operation for the given cognitive workspace. Return only compact JSON "
    "with operation, target, confidence, and reason_code."
)
KINDS = [
    "observation", "thought", "reflection", "question", "idea", "goal",
    "belief", "intention", "decision", "memory", "hypothesis",
]
REASONS = {
    "ATTEND": "NO_CURRENT_FOCUS",
    "MAINTAIN": "CURRENT_FOCUS_STILL_DOMINANT",
    "SWITCH": "HIGHER_PRIORITY_INTERRUPT",
    "SUPPRESS": "LOW_CONFIDENCE_DISTRACTOR",
    "IGNORE": "NO_COGNITIVE_VALUE",
}


def score(candidate):
    return candidate["activation_total"] + (candidate["surprise"] or 0.0)


def label(workspace):
    candidates = workspace["candidates"]
    threshold = workspace["attention_threshold"]
    eligible = [c for c in candidates if score(c) >= threshold]
    current = next(
        (c for c in eligible if c["id"] == workspace["current_focus"]), None
    )
    if not eligible:
        operation, target = "IGNORE", None
        margin = threshold - max((score(c) for c in candidates), default=threshold)
    else:
        best = max(eligible, key=score)
        alternatives = sorted((score(c) for c in eligible if c["id"] != best["id"]), reverse=True)
        # A rare, compelling but unreliable candidate is the one deliberate
        # departure from plain deterministic rank-and-admit.
        if (
            best["in_working_memory"]
            and best["confidence"] < 0.35
            and score(best) >= max(0.5, threshold + 1.0)
        ):
            operation, target = "SUPPRESS", best["id"]
            margin = (0.35 - best["confidence"]) * 4.0 + max(0.0, score(best) - threshold) * 0.3
        elif current is None:
            operation, target = "ATTEND", best["id"]
            margin = score(best) - (alternatives[0] if alternatives else threshold)
        elif best["id"] == current["id"] or score(best) - score(current) < 0.12:
            operation, target = "MAINTAIN", current["id"]
            margin = score(current) - (alternatives[0] if alternatives else threshold)
        else:
            operation, target = "SWITCH", best["id"]
            margin = score(best) - score(current)
    # This is a heuristic margin proxy, not empirical calibration. Close
    # decisions deliberately fall below the engine's default 0.6 trust gate.
    confidence = round(min(0.95, max(0.5, 0.55 + max(0.0, margin) * 0.08)), 3)
    return {
        "operation": operation,
        "target": target,
        "confidence": confidence,
        "reason_code": REASONS[operation],
    }


def make_example(rng, index):
    count = rng.randint(1, 9)
    threshold = round(rng.uniform(-3.0, 0.5), 3)
    focus_index = rng.randrange(count) if rng.random() < 0.7 else None
    surprise_index = rng.randrange(count) if rng.random() < 0.35 else None
    candidates = []
    for candidate_index in range(count):
        kind = rng.choice(KINDS)
        candidates.append(
            {
                "id": str(uuid.UUID(int=rng.getrandbits(128))),
                "kind": kind,
                "activation_total": round(max(-10.0, min(10.0, rng.gauss(0.0, 3.0))), 3),
                "surprise": round(rng.uniform(0.1, 2.0), 3)
                if candidate_index == surprise_index
                else None,
                "confidence": round(rng.uniform(0.1, 1.0), 3),
                "goal_priority": round(rng.uniform(0.1, 1.0), 3)
                if kind in ("goal", "intention")
                else None,
                "in_working_memory": candidate_index == focus_index
                or rng.random() < 0.15,
                "broadcast_count": rng.randrange(0, 20),
                "age_ms": rng.randrange(0, 86_400_000),
            }
        )
    workspace = {
        "protocol": PROTOCOL,
        "attention_threshold": threshold,
        "working_memory_capacity": rng.choice([3, 4, 5]),
        "current_focus": candidates[focus_index]["id"]
        if focus_index is not None
        else None,
        "candidates": candidates,
    }
    output = label(workspace)
    compact = lambda obj: json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
    return {
        "id": f"attention_v0_2_synth_{index}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user", "content": "Select the next attention operation for this cognitive workspace:\n" + compact(workspace)},
            {"role": "assistant", "content": compact(output)},
        ],
        "input": workspace,
        "output": output,
        "metadata": {"generator": "omega_attention_v0_2_reconstructed_heuristic"},
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n")


def make_balanced_examples(rng, count, start_index):
    """Rejection-sample equal operation classes; retain realistic workspaces."""
    operations = list(REASONS)
    remaining = {
        operation: count // len(operations) + (index < count % len(operations))
        for index, operation in enumerate(operations)
    }
    rows = []
    attempts = 0
    while len(rows) < count:
        if attempts >= max(1000, count * 100):
            raise RuntimeError("unable to balance attention operation classes")
        row = make_example(rng, start_index + attempts)
        operation = row["output"]["operation"]
        if remaining[operation] > 0:
            remaining[operation] -= 1
            rows.append(row)
        attempts += 1
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--eval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--balanced", action="store_true", help="Rejection-sample equal operation classes for a new training iteration")
    parser.add_argument("--train-out", type=Path, default=Path("training/attention-v0/data/train_v0_2.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("training/attention-v0/data/eval_v0_2.jsonl"))
    args = parser.parse_args()
    if args.train < 0 or args.eval < 0:
        parser.error("train and eval counts must be nonnegative")
    rng = random.Random(args.seed)
    if args.balanced:
        train_rows = make_balanced_examples(rng, args.train, 0)
        eval_rows = make_balanced_examples(rng, args.eval, args.train * 100 + 1)
    else:
        train_rows = (make_example(rng, i) for i in range(args.train))
        eval_rows = (make_example(rng, i + args.train) for i in range(args.eval))
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)


if __name__ == "__main__":
    main()
