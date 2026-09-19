import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from export_orient_ignition import FEATURES
from export_orient_outcomes import load_labeled_examples, partition_chronologically


class IndependentOutcomeExportTests(unittest.TestCase):
    def test_only_independent_nonconflicting_outcomes_join_without_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE cycle_events (cycle_seq INTEGER, phase TEXT, payload_json TEXT)")
                for cycle in range(20):
                    observation_id = f"observation-{cycle}"
                    connection.execute("INSERT INTO cycle_events VALUES (?,?,?)", (
                        cycle, "compare", json.dumps({
                            "observation_id": observation_id,
                            "orienting_inputs": {name: 0.1 for name in FEATURES},
                            "text": "private object text must not be exported",
                        }),
                    ))
                    if cycle % 2:
                        connection.execute("INSERT INTO cycle_events VALUES (?,?,?)", (
                            cycle, "broadcast", json.dumps({"newly_admitted_ids": [observation_id]}),
                        ))
                    connection.execute("INSERT INTO cycle_events VALUES (?,?,?)", (
                        cycle + 100, "learn", json.dumps({
                            "independent_observation_outcome": True,
                            "observation_id": observation_id,
                            "successful": bool(cycle // 2 % 2),
                        }),
                    ))
                connection.commit()
            rows = load_labeled_examples(path)
            self.assertEqual(len(rows), 20)
            self.assertNotIn("private object text", json.dumps(rows))
            train, evaluation = partition_chronologically(rows, 20, 1)
            self.assertEqual((len(train), len(evaluation)), (16, 4))
            self.assertEqual({(row["ignited"], row["observed_success"]) for row in evaluation},
                             {(0, 0), (0, 1), (1, 0), (1, 1)})
            with self.assertRaisesRegex(ValueError, "independently labeled"):
                partition_chronologically(rows, 1000, 20)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("INSERT INTO cycle_events VALUES (?,?,?)", (
                    200, "learn", json.dumps({
                        "independent_observation_outcome": True,
                        "observation_id": "observation-0", "successful": True,
                    }),
                ))
                connection.commit()
            self.assertEqual(len(load_labeled_examples(path)), 19,
                             "contradictory labels for one observation must be excluded")


if __name__ == "__main__":
    unittest.main()
