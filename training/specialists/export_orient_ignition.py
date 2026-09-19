#!/usr/bin/env python3
"""Export local observed ORIENT-to-ignition examples, without object text.

The label is actual same-cycle Working Memory admission, not the deterministic
orienting formula's `fired` flag and not a human success judgment. Refuse
insufficient/missing data rather than manufacturing a usable specialist.
"""

import argparse
from contextlib import closing
import json
import sqlite3
from pathlib import Path

FEATURES = (
    "novelty", "prediction_error", "goal_relevance", "affective_salience",
    "social_relevance", "threat", "urgency",
)


def load_examples(database):
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        admitted_by_cycle = {}
        for cycle_seq, payload_json in connection.execute(
            "SELECT cycle_seq, payload_json FROM cycle_events WHERE phase='broadcast' ORDER BY cycle_seq"
        ):
            payload = json.loads(payload_json)
            admitted_by_cycle.setdefault(cycle_seq, set()).update(
                payload.get("newly_admitted_ids", [])
            )
        rows = []
        for cycle_seq, payload_json in connection.execute(
            "SELECT cycle_seq, payload_json FROM cycle_events WHERE phase='compare' ORDER BY cycle_seq"
        ):
            payload = json.loads(payload_json)
            observation_id = payload.get("observation_id")
            inputs = payload.get("orienting_inputs")
            if not isinstance(observation_id, str) or not isinstance(inputs, dict):
                continue  # legacy event; cannot infer missing numeric fields
            values = [inputs.get(name) for name in FEATURES]
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and 0.0 <= float(value) <= 1.0 for value in values):
                continue
            rows.append({
                "cycle_seq": cycle_seq,
                "observation_id": observation_id,
                "features": [float(value) for value in values],
                "ignited": int(observation_id in admitted_by_cycle.get(cycle_seq, set())),
            })
    return rows


def partition_chronologically(rows, min_rows, min_each_class):
    if len(rows) < min_rows:
        raise ValueError(f"only {len(rows)} keyed real examples; need at least {min_rows}")
    split = max(1, min(len(rows) - 1, len(rows) * 4 // 5))
    train, evaluation = rows[:split], rows[split:]
    for name, partition in (("train", train), ("evaluation", evaluation)):
        counts = {label: sum(row["ignited"] == label for row in partition) for label in (0, 1)}
        if min(counts.values()) < min_each_class:
            raise ValueError(f"{name} has class counts {counts}; need {min_each_class} of each")
    return train, evaluation


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--eval-out", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--min-each-class", type=int, default=20)
    args = parser.parse_args()
    if args.min_rows < 2 or args.min_each_class < 1:
        parser.error("minimum thresholds must be positive")
    if not args.database.is_file():
        parser.error("database does not exist")
    rows = load_examples(args.database)
    try:
        train, evaluation = partition_chronologically(rows, args.min_rows, args.min_each_class)
    except ValueError as error:
        parser.error(str(error))
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, evaluation)
    print(f"local orient-ignition examples: train={len(train)}, eval={len(evaluation)}")


if __name__ == "__main__":
    main()
