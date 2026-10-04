"""Guard against test leakage, ambiguous answer extraction and false quality wins."""
import importlib.util
import math
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("decision_metrics", Path(__file__).parents[1] / "tools/decision_metrics.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class DecisionMetricsTests(unittest.TestCase):
    def rows(self, split="test"):
        return [{"question_id": f"q{i}", "split": split, "variant": 0,
                 "target_index": 0, "logits": [2., 0.], "subject": "math"} for i in range(4)]

    def test_stable_loss_and_no_pseudoreplication(self):
        result = metrics.quality(self.rows())
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["accuracy"], 1.)
        self.assertAlmostEqual(result["log_loss"], math.log1p(math.exp(-2.)))
        row = {**self.rows()[0], "logits": [-1e30, 1e30]}
        self.assertEqual(metrics.quality([row])["log_loss"], 2e30)
        self.assertEqual(metrics.quality([row])["brier_sum_over_classes"], 2.)
        with self.assertRaisesRegex(ValueError, "range overflow"):
            metrics.log_probabilities([-1e308, 1e308])
        for rows in (self.rows() + self.rows()[:1], [{**row, "variant": 1}]):
            with self.assertRaises(ValueError):
                metrics.quality(rows)

    def test_calibration_cannot_use_test_or_overlap(self):
        binding = {"model": "sha", "backend": "metal", "prompt": "sha"}
        with self.assertRaisesRegex(ValueError, "calibration data"):
            metrics.fit_temperature(self.rows(), binding)
        fit = metrics.fit_temperature(self.rows("calibration"), binding)
        self.assertLessEqual(fit["calibration_log_loss_after"], fit["calibration_log_loss_before"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            metrics.apply_calibration(self.rows(), fit, binding)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            metrics.apply_calibration(self.rows(), fit, {**binding, "backend": "cpu_neon"})

    def test_strict_answer_ignores_thinking_and_never_guesses(self):
        self.assertEqual(metrics.parse_answer("A"), 0)
        self.assertEqual(metrics.parse_answer(" Answer: D<|im_end|> "), 3)
        self.assertEqual(metrics.parse_answer("maybe A or B\n</think>\nC", thinking=True), 2)
        for text in ("A or B", "The answer is A", "A\nAnswer: B", "<think>A", "A</think>"):
            self.assertIsNone(metrics.parse_answer(text))
        self.assertIsNone(metrics.parse_answer("thinking about A", thinking=True))
        self.assertIsNone(metrics.parse_answer("x</think>A</think>B", thinking=True))

    def test_small_perfect_agreement_has_uncertainty(self):
        direct = {f"q{i}": True for i in range(12)}
        result = metrics.paired_quality(direct, direct, 2)
        self.assertEqual(result["difference_pp"], 0.)
        self.assertLess(result["interval_pp"][0], -2.)
        self.assertFalse(result["noninferior"])
        n = 12
        _, upper = metrics._exact_binomial_interval(n, 0, .025)
        self.assertAlmostEqual(upper, 1 - .0125 ** (1 / n), places=12)

    def test_binomial_cdf_matches_independent_polynomial(self):
        for n in (1, 5, 20):
            for k in range(n + 1):
                for p in (.01, .3, .7, .99):
                    expected = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                                   for i in range(k + 1))
                    self.assertAlmostEqual(metrics._binomial_cdf(n, k, p), expected, places=12)

    def test_paired_losses_and_validation(self):
        a = {"a": False, "b": True, "c": True}
        b = {"a": True, "b": True, "c": False}
        result = metrics.paired_quality(a, b, 2)
        self.assertEqual((result["gains"], result["losses"]), (1, 1))
        self.assertFalse(result["noninferior"])
        with self.assertRaises(ValueError):
            metrics.paired_quality(a, {"a": True}, 2)
        with self.assertRaises(ValueError):
            metrics.paired_quality(a, b, float("nan"))


if __name__ == "__main__":
    unittest.main()
