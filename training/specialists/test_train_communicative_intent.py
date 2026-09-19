import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_communicative_intent.py"


class CommunicativeIntentTrainerTests(unittest.TestCase):
    def test_training_is_reproducible_and_writes_only_a_gated_shadow_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            for output in (first, second):
                completed = subprocess.run(
                    [sys.executable, str(TRAINER), "--output", str(output)],
                    cwd=HERE.parent.parent,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first.read_bytes(),
                (HERE / "outputs" / "communicative_intent.json").read_bytes(),
                "checked artifact must be exactly reproducible from the trainer",
            )
            artifact = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(artifact["schema"], "omega-communicative-intent-softmax/v1")
            self.assertEqual(artifact["status"], "shadow_only")
            self.assertEqual(artifact["labels"], ["speak", "ask", "ignore"])
            self.assertEqual(artifact["dimensions"], 1024)
            self.assertGreaterEqual(artifact["held_out"]["accuracy"], 0.90)
            self.assertTrue(all(value >= 0.80 for value in artifact["held_out"]["class_recall"].values()))


if __name__ == "__main__":
    unittest.main()
