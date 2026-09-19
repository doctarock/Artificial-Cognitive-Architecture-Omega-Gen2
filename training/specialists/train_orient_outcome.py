#!/usr/bin/env python3
"""Train a tiny numeric observed-outcome forecaster with fail-closed gates.

This predicts P(success | ORIENT inputs, observed ignition). It is not a
counterfactual attention policy. A saved model is suitable for shadow
forecasting only until prospective evaluation proves intervention utility.
"""

import argparse
import json
import math
from pathlib import Path

from export_orient_ignition import FEATURES

MODEL_FEATURES = FEATURES + ("ignited",)


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            values = raw.get("features")
            label = raw.get("observed_success")
            ignited = raw.get("ignited")
            if (not isinstance(values, list) or len(values) != len(FEATURES)
                    or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                               and math.isfinite(value) for value in values)
                    or label not in (0, 1) or ignited not in (0, 1)):
                raise ValueError(f"invalid row {line_number} in {path}")
            rows.append(([float(value) for value in values] + [float(ignited)], int(label)))
    return rows


def fit(train, steps=2000, learning_rate=0.08, l2=0.001):
    if not train or {label for _, label in train} != {0, 1}:
        raise ValueError("training data must contain both outcome classes")
    width = len(MODEL_FEATURES)
    means = [sum(values[j] for values, _ in train) / len(train) for j in range(width)]
    scales = []
    for j in range(width):
        variance = sum((values[j] - means[j]) ** 2 for values, _ in train) / len(train)
        scales.append(max(math.sqrt(variance), 1e-6))
    normalized = [([(values[j] - means[j]) / scales[j] for j in range(width)], label) for values, label in train]
    counts = {label: sum(y == label for _, y in normalized) for label in (0, 1)}
    class_weights = {label: len(normalized) / (2.0 * counts[label]) for label in (0, 1)}
    weights = [0.0] * width
    bias = 0.0
    for _ in range(steps):
        gradient = [0.0] * width
        bias_gradient = 0.0
        for values, label in normalized:
            score = max(-30.0, min(30.0, bias + sum(w * x for w, x in zip(weights, values))))
            probability = 1.0 / (1.0 + math.exp(-score))
            error = (probability - label) * class_weights[label]
            for j in range(width):
                gradient[j] += error * values[j]
            bias_gradient += error
        for j in range(width):
            weights[j] -= learning_rate * (gradient[j] / len(normalized) + l2 * weights[j])
        bias -= learning_rate * bias_gradient / len(normalized)
    return {"means": means, "scales": scales, "weights": weights, "bias": bias}


def predict(model, values):
    normalized = [(value - mean) / scale for value, mean, scale in zip(values, model["means"], model["scales"])]
    score = max(-30.0, min(30.0, model["bias"] + sum(w * x for w, x in zip(model["weights"], normalized))))
    return 1.0 / (1.0 + math.exp(-score))


def evaluate(model, rows):
    if not rows or {label for _, label in rows} != {0, 1}:
        raise ValueError("evaluation data must contain both outcome classes")
    pairs = [(predict(model, values), label) for values, label in rows]
    recalls = []
    for label in (0, 1):
        subset = [(probability >= 0.5) == bool(label) for probability, actual in pairs if actual == label]
        recalls.append(sum(subset) / len(subset))
    return {
        "sample_count": len(rows),
        "balanced_accuracy": sum(recalls) / 2.0,
        "brier_score": sum((probability - label) ** 2 for probability, label in pairs) / len(pairs),
        "negative_recall": recalls[0],
        "positive_recall": recalls[1],
    }


def train_and_gate(train, evaluation, min_eval_rows, min_balanced_accuracy, max_brier):
    if len(evaluation) < min_eval_rows:
        raise ValueError(f"only {len(evaluation)} evaluation rows; need at least {min_eval_rows}")
    model = fit(train)
    metrics = evaluate(model, evaluation)
    if metrics["balanced_accuracy"] < min_balanced_accuracy or metrics["brier_score"] > max_brier:
        raise ValueError(f"promotion gates failed: {metrics}")
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-eval-rows", type=int, default=200)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.65)
    parser.add_argument("--max-brier", type=float, default=0.22)
    args = parser.parse_args()
    try:
        train = load_rows(args.train)
        evaluation = load_rows(args.eval)
        model, metrics = train_and_gate(train, evaluation, args.min_eval_rows,
                                        args.min_balanced_accuracy, args.max_brier)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    artifact = {
        "schema": "omega-orient-observed-outcome-logistic/v1",
        "status": "shadow_only",
        "features": MODEL_FEATURES,
        **model,
        "held_out_metrics": metrics,
        "limitations": "observational outcomes are not counterfactual attention-policy evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
