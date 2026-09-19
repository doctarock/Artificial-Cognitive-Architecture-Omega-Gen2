import json
import tempfile
import unittest
from pathlib import Path

from train_orient_outcome import load_rows, train_and_gate


def write_rows(path, count, learnable=True):
    with path.open("w", encoding="utf-8") as handle:
        for i in range(count):
            label = i % 2
            signal = 0.9 if label else 0.1
            if not learnable:
                signal = 0.9 if (i * 17 % 7) < 3 else 0.1
            row = {
                "features": [signal, 0.2, 0.3, 0.1, 0.4, 0.0, 0.0],
                "ignited": (i // 2) % 2,
                "observed_success": label,
            }
            handle.write(json.dumps(row) + "\n")


class OutcomeTrainerTests(unittest.TestCase):
    def test_passes_separable_holdout_and_rejects_bad_or_small_data(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.jsonl"
            eval_path = Path(directory) / "eval.jsonl"
            write_rows(train_path, 400)
            write_rows(eval_path, 200)
            train = load_rows(train_path)
            evaluation = load_rows(eval_path)
            _, metrics = train_and_gate(train, evaluation, 200, 0.90, 0.10)
            self.assertGreaterEqual(metrics["balanced_accuracy"], 0.90)
            self.assertLessEqual(metrics["brier_score"], 0.10)
            with self.assertRaisesRegex(ValueError, "need at least"):
                train_and_gate(train, evaluation[:20], 200, 0.65, 0.22)

            write_rows(eval_path, 200, learnable=False)
            noisy_evaluation = load_rows(eval_path)
            with self.assertRaisesRegex(ValueError, "promotion gates failed"):
                train_and_gate(train, noisy_evaluation, 200, 0.90, 0.10)


if __name__ == "__main__":
    unittest.main()
