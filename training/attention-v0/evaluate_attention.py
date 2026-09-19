#!/usr/bin/env python
import argparse
import json
import uuid
from pathlib import Path

VALID_OPS = {"ATTEND", "MAINTAIN", "SWITCH", "SUPPRESS", "IGNORE"}
TARGET_REQUIRED_OPS = {"ATTEND", "SWITCH", "SUPPRESS"}


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_prediction(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        return {}


def valid_output(obj):
    return (
        isinstance(obj, dict)
        and obj.get("operation") in VALID_OPS
        and "target" in obj
        and (obj["target"] is None or isinstance(obj["target"], str))
        and isinstance(obj.get("confidence"), (float, int))
        and 0.0 <= float(obj["confidence"]) <= 1.0
        and isinstance(obj.get("reason_code"), str)
    )


def resolvable_action(prediction, workspace):
    """Approximate whether Rust can apply the decision to this candidate set."""
    operation = prediction.get("operation")
    if operation not in VALID_OPS:
        return False
    if operation not in TARGET_REQUIRED_OPS:
        return True
    target = prediction.get("target")
    if not isinstance(target, str):
        return False
    if workspace.get("protocol") == "omega-attention-workspace/v0.2":
        try:
            uuid.UUID(target)
        except ValueError:
            return False
    return target in {candidate["id"] for candidate in workspace.get("candidates", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold-field", default="output")
    parser.add_argument("--prediction-field", default="prediction")
    args = parser.parse_args()

    total = 0
    valid = 0
    exact_operation = 0
    exact_target = 0
    resolvable = 0
    for row in load_jsonl(args.predictions):
        total += 1
        gold = row[args.gold_field]
        pred = parse_prediction(row.get(args.prediction_field, row.get("output")))
        if valid_output(pred):
            valid += 1
        if pred.get("operation") == gold.get("operation"):
            exact_operation += 1
        if pred.get("target") == gold.get("target"):
            exact_target += 1
        if valid_output(pred) and resolvable_action(pred, row.get("input", {})):
            resolvable += 1

    if total == 0:
        raise SystemExit("no rows to evaluate")
    print(json.dumps({
        "rows": total,
        "valid_json_rate": round(valid / total, 4),
        "operation_accuracy": round(exact_operation / total, 4),
        "target_accuracy": round(exact_target / total, 4),
        "resolvable_action_rate": round(resolvable / total, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
