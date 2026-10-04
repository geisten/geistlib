"""Prevent examples, calibration or reordered duplicates leaking into test."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("decision_dataset", Path(__file__).parents[1] / "tools/decision_dataset.py")
dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dataset)


class DecisionDatasetTests(unittest.TestCase):
    def row(self, name, answer=1):
        return {"subject": "math", "question": name, "choices": ["a", "b", "c", "d"], "answer": answer}

    def source(self):
        return {"dev": [self.row("example")],
                "validation": [self.row(f"v{i}") for i in range(6)],
                "test": [self.row("held out")]}

    def test_fixed_disjoint_splits(self):
        parts, shots, audit = dataset.partition(self.source())
        self.assertEqual(audit["prepared_counts"], {"development": 3, "calibration": 3, "test": 1})
        self.assertEqual(shots["math"][0]["question"], "example")
        before = {s: [r["question_id"] for r in rows] for s, rows in parts.items()}
        source = self.source()
        source["validation"].reverse()
        after, _, _ = dataset.partition(source)
        self.assertEqual(before, {s: [r["question_id"] for r in rows] for s, rows in after.items()})

    def test_overlap_and_reordered_duplicates_require_explicit_policy(self):
        source = self.source()
        source["test"].append(source["validation"][0])
        swapped = {**source["test"][0], "choices": ["d", "c", "b", "a"], "answer": 2}
        source["test"].append(swapped)
        with self.assertRaisesRegex(ValueError, "explicit exclusion"):
            dataset.partition(source)
        parts, _, audit = dataset.partition(source, "exclude")
        self.assertEqual(len(parts["test"]), 1)
        self.assertEqual(audit["excluded_counts"], {"earlier_split_overlap": 1, "within_split_duplicate": 1})

    def test_conflicting_annotations_are_not_silently_scored(self):
        source = self.source()
        source["test"].append(self.row("held out", answer=3))
        parts, _, audit = dataset.partition(source, "exclude")
        self.assertFalse(parts["test"])
        self.assertEqual(audit["excluded_counts"], {"conflicting_annotation": 2})
        source["test"].append(self.row("example", answer=2))
        with self.assertRaisesRegex(ValueError, "few-shot"):
            dataset.partition(source, "exclude")

    def test_rotation_keeps_semantic_gold_and_question_identity(self):
        row = {**self.row("q"), "id": "q", "question_id": "identity"}
        for shift in range(4):
            rotated = dataset.permute(row, shift)
            self.assertEqual(rotated["choices"][rotated["answer"]], "b")
            self.assertEqual(rotated["question_id"], "identity")
            self.assertEqual(dataset.question_key(rotated), dataset.question_key(row))
        with self.assertRaises(ValueError):
            dataset.validate_question({**row, "answer": True})


if __name__ == "__main__":
    unittest.main()
