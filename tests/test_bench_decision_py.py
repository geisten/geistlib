"""Prevent misleading benchmark results from malformed labels or missing trials."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("bench_decision", Path(__file__).parents[1] / "tools/bench_decision.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class DecisionBenchmarkTests(unittest.TestCase):
    def parse(self, case):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(json.dumps(case) + "\n")
            return bench.cases_from_jsonl(path)

    def test_labels_are_single_tokens(self):
        for candidates in ([[1, 2]], [1, 1], [True, 2], [-1, 2], []):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.parse({"prompt_ids": [1], "candidate_ids": candidates})
        with self.assertRaises(ValueError):
            self.parse({"prompt_ids": [1], "candidate_ids": [2], "target_index": 1})
        cases = self.parse({"id": "a", "prompt_ids": [1, 2], "candidate_ids": [3, 4], "target_index": 0})
        self.assertEqual(bench.wire_input(cases), "2 2 1 2 3 4\n")

    def samples(self):
        rows = []
        for trial in range(3):
            for mode in bench.MODES:
                row = {"case_index": 0, "mode": mode, "trial": trial,
                       "phase": "first" if trial == 0 else "warm",
                       "elapsed_ms": trial + 1., "prompt_tokens": 1, "candidate_count": 2,
                       "generated_ids": [3], "best_index": 0,
                       "logits": [2., 1.], "conditional_probabilities": [.75, .25]}
                rows.append(row)
        return rows

    def test_complete_samples_and_quality(self):
        cases = [{"id": "a", "prompt_ids": [1], "candidate_ids": [3, 4], "target_index": 0}]
        rows = self.samples()
        bench.enrich_and_check(rows, cases, 0, 2)
        summary = bench.summarize(rows, cases)
        group = summary["by_case_and_mode"]["a/decision_dense"]
        self.assertEqual(group["p50_ms"], 2.5)
        self.assertAlmostEqual(group["p95_ms"], 2.95)
        self.assertEqual(group["first_token_label_accuracy"], 1.)
        for broken in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                bench.enrich_and_check(broken, cases, 0, 2)
        rows = self.samples()
        rows[1]["logits"] = [9., 1.]
        with self.assertRaisesRegex(ValueError, "disagree"):
            bench.enrich_and_check(rows, cases, 0, 2)

    def test_unlabelled_quality_is_unknown(self):
        cases = [{"id": "a", "prompt_ids": [1], "candidate_ids": [3, 4]}]
        rows = self.samples()
        bench.enrich_and_check(rows, cases, 0, 2)
        self.assertIsNone(bench.summarize(rows, cases)["by_case_and_mode"]["a/decision_dense"]["first_token_label_accuracy"])


if __name__ == "__main__":
    unittest.main()
