import importlib.util
from contextlib import closing
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exporter", HERE / "export_orient_ignition.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def test_observed_admission_labels_and_no_text_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE cycle_events (cycle_seq INTEGER, phase TEXT, payload_json TEXT)")
                for cycle in range(10):
                    observation_id = f"observation-{cycle}"
                    inputs = {name: 0.1 for name in exporter.FEATURES}
                    connection.execute("INSERT INTO cycle_events VALUES (?,?,?)", (
                        cycle, "compare", json.dumps({
                            "observation_id": observation_id, "orienting_inputs": inputs,
                            "text": "must never leave local DB",
                        }),
                    ))
                    if cycle % 2:
                        connection.execute("INSERT INTO cycle_events VALUES (?,?,?)", (
                            cycle, "broadcast", json.dumps({"newly_admitted_ids": [observation_id]}),
                        ))
                connection.commit()
            rows = exporter.load_examples(path)
            self.assertEqual(len(rows), 10)
            self.assertEqual([row["ignited"] for row in rows], [0, 1] * 5)
            self.assertNotIn("must never leave", json.dumps(rows))
            train, evaluation = exporter.partition_chronologically(rows, 10, 1)
            self.assertEqual((len(train), len(evaluation)), (8, 2))
            self.assertEqual({row["ignited"] for row in evaluation}, {0, 1})
            with self.assertRaisesRegex(ValueError, "need at least"):
                exporter.partition_chronologically(rows, 1000, 20)


if __name__ == "__main__":
    unittest.main()
