#!/usr/bin/env python3
"""Export local host-judged observation outcomes joined to numeric ORIENT inputs.

These are factual labels of *observed* handling, not counterfactual labels of
what would have happened under the opposite ignition decision. A specialist
trained on them remains shadow-only until a prospective outcome comparison.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from export_orient_ignition import load_examples, write_jsonl


def load_labeled_examples(database):
    rows = load_examples(database)
    uri = database.resolve().as_uri() + "?mode=ro"
    labels = {}
    conflicts = set()
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        for (payload_json,) in connection.execute(
            "SELECT payload_json FROM cycle_events WHERE phase='learn' ORDER BY cycle_seq"
        ):
            payload = json.loads(payload_json)
            if payload.get("independent_observation_outcome") is not True:
                continue
            observation_id = payload.get("observation_id")
            successful = payload.get("successful")
            if not isinstance(observation_id, str) or not isinstance(successful, bool):
                continue
            if observation_id in labels and labels[observation_id] != successful:
                conflicts.add(observation_id)  # a corrected/conflicting verdict is unusable
            labels.setdefault(observation_id, successful)
    return [
        {**row, "observed_success": int(labels[row["observation_id"]])}
        for row in rows if row["observation_id"] in labels
        and row["observation_id"] not in conflicts
    ]


def partition_chronologically(rows, min_rows, min_each_cell):
    if len(rows) < min_rows:
        raise ValueError(f"only {len(rows)} independently labeled examples; need at least {min_rows}")
    split = max(1, min(len(rows) - 1, len(rows) * 4 // 5))
    train, evaluation = rows[:split], rows[split:]
    for name, partition in (("train", train), ("evaluation", evaluation)):
        counts = {
            (ignited, successful): sum(
                row["ignited"] == ignited and row["observed_success"] == successful
                for row in partition
            ) for ignited in (0, 1) for successful in (0, 1)
        }
        if min(counts.values()) < min_each_cell:
            raise ValueError(f"{name} has ignition/outcome cell counts {counts}; need {min_each_cell} of each")
    return train, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--eval-out", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--min-each-cell", type=int, default=20)
    args = parser.parse_args()
    if args.min_rows < 2 or args.min_each_cell < 1:
        parser.error("minimum thresholds must be positive")
    if not args.database.is_file():
        parser.error("database does not exist")
    rows = load_labeled_examples(args.database)
    try:
        train, evaluation = partition_chronologically(rows, args.min_rows, args.min_each_cell)
    except ValueError as error:
        parser.error(str(error))
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, evaluation)
    print(f"local independent ORIENT outcome examples: train={len(train)}, eval={len(evaluation)}")


if __name__ == "__main__":
    main()
