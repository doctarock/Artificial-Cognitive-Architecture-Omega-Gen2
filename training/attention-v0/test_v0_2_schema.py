"""Small standard-library checks for the runtime-compatible attention recipe."""

import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_script("generate_scenarios_v0_2")
trainer = load_script("train_unsloth")
inference = load_script("infer_unsloth")
WORKSPACE_KEYS = {
    "protocol", "attention_threshold", "working_memory_capacity",
    "current_focus", "candidates",
}
CANDIDATE_KEYS = {
    "id", "kind", "activation_total", "surprise", "confidence",
    "goal_priority", "in_working_memory", "broadcast_count", "age_ms",
}
RUNTIME_KINDS = {
    "observation", "thought", "reflection", "question", "idea", "goal",
    "belief", "intention", "decision", "memory", "hypothesis",
}


class V02SchemaTests(unittest.TestCase):
    def test_generated_workspaces_and_targets_match_the_runtime_contract(self):
        operations = set()
        for index in range(1000):
            row = generator.make_example(random.Random(index), index)
            workspace = row["input"]
            self.assertEqual(set(workspace), WORKSPACE_KEYS)
            self.assertEqual(workspace["protocol"], generator.PROTOCOL)
            self.assertTrue(1 <= len(workspace["candidates"]) <= 9)
            self.assertLessEqual(
                sum(c["surprise"] is not None for c in workspace["candidates"]), 1
            )
            ids = set()
            for candidate in workspace["candidates"]:
                self.assertEqual(set(candidate), CANDIDATE_KEYS)
                self.assertIn(candidate["kind"], RUNTIME_KINDS)
                self.assertTrue(-10 <= candidate["activation_total"] <= 10)
                ids.add(candidate["id"])
            self.assertIn(workspace["current_focus"], ids | {None})
            self.assertIn(row["output"]["target"], ids | {None})
            operations.add(row["output"]["operation"])
            self.assertIn(generator.PROTOCOL, trainer.format_messages(row))
            self.assertIn(generator.SYSTEM_PROMPT, inference.format_prompt(workspace))
            self.assertIn(generator.PROTOCOL, inference.format_prompt(workspace))
        self.assertEqual(operations, set(generator.REASONS))

    def test_training_preflight_rejects_mixed_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.jsonl"
            v02 = generator.make_example(random.Random(1), 1)
            v01 = {"input": {"protocol": "omega-attention-scenario/v0.1"}}
            path.write_text(json.dumps(v02) + "\n" + json.dumps(v01) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed"):
                trainer.validate_jsonl_protocol(path)
            with self.assertRaisesRegex(ValueError, "expected"):
                trainer.validate_jsonl_protocol(path, "omega-attention-scenario/v0.1")

    def test_balanced_generation_covers_each_operation_and_varies_margin_confidence(self):
        rows = generator.make_balanced_examples(random.Random(3407), 100, 0)
        counts = {
            operation: sum(row["output"]["operation"] == operation for row in rows)
            for operation in generator.REASONS
        }
        self.assertEqual(set(counts.values()), {20})
        confidences = {row["output"]["confidence"] for row in rows}
        self.assertGreater(len(confidences), 10)
        self.assertTrue(all(0.5 <= value <= 0.95 for value in confidences))


if __name__ == "__main__":
    unittest.main()
