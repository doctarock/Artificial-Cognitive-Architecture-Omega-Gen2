#!/usr/bin/env python3
"""Train a gated shadow-only classifier for Omega's default tool intents."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tool_intent_cases import CASES
from train_communicative_intent import DIMENSIONS, vectorize


LABELS = ["none", "current_time", "self_status", "abstraction_status"]

TRAIN_SEEDS = {
    "current_time": [
        "Tell me the time now.", "What date is it today?", "Check the clock.",
        "Report the current timestamp.", "Which weekday is today?", "Give me today's date.",
        "I want to know the present time.", "Read the local clock.", "What hour is it?",
        "Look up the date for me.", "Please state the time.", "What does the calendar say today?",
        "Give the current day and time.", "Check today's date.", "Can you provide the time now?",
        "Use your time capability to tell me the hour.", "Report the date right now.",
        "I need a clock reading.", "What is the day today?", "State the present date.",
        "What time does your clock show?", "Which time reading do you currently have?",
        "Read me the time displayed by your clock.",
    ],
    "self_status": [
        "Tell me what you are thinking.", "Report your cognitive status.",
        "What is in your mind now?", "Describe your current focus.",
        "How do you feel internally?", "What occupies your working memory?",
        "Give an introspective status report.", "Are you thinking about anything?",
        "What has your attention?", "Describe your present mental state.",
        "How busy is your cognition?", "Tell me what is happening inside you.",
        "Report your current awareness.", "What are your active thoughts?",
        "Summarize your internal condition.", "Where is your attention directed?",
        "How is your cognitive system doing?", "What is your mind working on?",
        "Show your working-memory state.", "Provide your self status.",
    ],
    "abstraction_status": [
        "Show the latest abstraction.", "What pattern did you synthesize?",
        "Report your provisional generalization.", "Describe the cross-memory hypothesis.",
        "What common structure have you inferred?", "Tell me your synthesized pattern.",
        "Read the current conceptual abstraction.", "What general lesson emerged?",
        "Show the pattern formed across episodes.", "Report your latest abstraction status.",
        "What hypothesis connects those memories?", "Describe the inferred commonality.",
        "Give me the current synthesis result.", "What broad pattern have you learned?",
        "Tell me the abstraction you are considering.", "Read back the provisional pattern.",
        "What generalization links the examples?", "Show your current abstract summary.",
        "Report the memory synthesis hypothesis.", "What structure did synthesis find?",
    ],
    "none": [
        "The meeting starts at three tomorrow.", "Our status document needs editing.",
        "This fabric uses a striped pattern.", "Cognition is the subject of the paper.",
        "I stored the date in the filename.", "Attention mechanisms are computationally useful.",
        "The abstraction layer depends on the database.", "We bought a new wall clock.",
        "Project status is green.", "The parser performs pattern matching.",
        "Time complexity matters for this algorithm.", "Her mental state changed in the story.",
        "The calendar component has a rendering bug.", "This memory pattern is only an example.",
        "Please revise the paragraph about working memory.", "The word current appears twice.",
        "We can discuss generalization later.", "Status codes are integers.",
        "Clock synchronization is a distributed-systems problem.", "The design uses self monitoring.",
        "The working-memory chapter surveys cognitive theories.",
        "A date column is part of the export format.",
        "The schema contains a current-date property.",
        "This architecture models attention and internal cognition.",
        "We are documenting how the cognitive architecture works.",
        "The report discusses what working memory means.",
        "Date formatting belongs to the serialization layer.",
        "Attention and awareness are concepts in this design.",
    ],
}


def probabilities(weights, bias, features):
    logits = [bias[label] + sum(weights[label][index] * value for index, value in features.items()) for label in range(len(LABELS))]
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def evaluate(weights, bias):
    hits = {label: 0 for label in LABELS}
    totals = {label: 0 for label in LABELS}
    failures = []
    for expected, text in CASES:
        totals[expected] += 1
        probs = probabilities(weights, bias, vectorize(text))
        actual = LABELS[max(range(len(LABELS)), key=lambda index: probs[index])]
        if actual == expected:
            hits[expected] += 1
        else:
            failures.append({"expected": expected, "actual": actual, "text": text})
    return {
        "samples": len(CASES),
        "accuracy": sum(hits.values()) / len(CASES),
        "class_recall": {label: hits[label] / totals[label] for label in LABELS},
        "failures": failures,
    }


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
    eligible = metrics["samples"] >= 40 and metrics["accuracy"] >= 0.90 \
        and min(metrics["class_recall"].values()) >= 0.80
    print(json.dumps({**metrics, "eligible_for_shadow": eligible}, indent=2))
    if not eligible:
        return 1
    artifact = {
        "schema": "omega-tool-intent-softmax/v1",
        "status": "shadow_only",
        "labels": LABELS,
        "dimensions": DIMENSIONS,
        "feature_schema": "fnv1a-word-bigram-char3-5-l2/v1",
        "weights": weights,
        "bias": bias,
        "held_out": {"samples": metrics["samples"], "accuracy": metrics["accuracy"], "class_recall": metrics["class_recall"]},
        "limitations": "Default closed tool set and generic authored benchmark; shadow validation required on real ACA input.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
