"""Contracts which protect optional encoder evaluation from misleading results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decision_dataset as data
import decision_encoders as encoders
import eval_decision_encoders as evaluation


class FakeTokenizer:
    all_special_tokens = ["[MASK]", "[CLS]", "[SEP]"]
    cls_token_id, sep_token_id, mask_token_id = 1, 2, 3

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]


class EncoderContracts(unittest.TestCase):
    def test_disabled_factory_has_no_optional_imports(self):
        script = """import sys; import decision_encoders as e
assert not ({'torch', 'transformers', 'laya'} & set(sys.modules))
try: e.create_backend('modernbert', '/missing')
except RuntimeError: pass
else: raise AssertionError('feature unexpectedly enabled')
assert not ({'torch', 'transformers', 'laya'} & set(sys.modules))
"""
        subprocess.run([sys.executable, "-c", script], cwd=Path(encoders.__file__).parent, check=True)

    def test_nonfinite_or_incomplete_scores_fail(self):
        for values in ([0., float("nan")], [float("inf"), 1.], [0.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                encoders.validate_scores(values, 2)
        self.assertAlmostEqual(sum(encoders.probabilities([1000., 999.])), 1.)

    def test_request_keys_and_controls_are_not_silently_changed(self):
        request = encoders.ChoiceRequest("Text", "Choose", {"A": "one", "B": "two"})
        request.validate()
        with self.assertRaises(ValueError):
            encoders.ChoiceRequest("", "Choose", request.options).validate()
        with self.assertRaises(encoders.InputRejected):
            encoders.validate_literal_text(FakeTokenizer(), ["a literal [MASK] in the evidence"])

    def test_laya_rejects_each_native_truncation_before_forward(self):
        backend = encoders.LayaBackend.__new__(encoders.LayaBackend)
        backend.tokenizer = FakeTokenizer()
        backend.cfg = {"head_max_len": 256}
        backend.max_tokens = 1024
        backend.model = Mock()
        q = {"type": "choice", "instructions": "Choose", "criteria": {"A": "one", "B": "two"}}
        normalized = backend.normalize_question(q)
        options = ["A: one", "B: two"]
        tok = backend.tokenizer
        ids = [1] + tok.encode("choice question: Choose") + [2]
        markers = []
        for option in options:
            markers.append(len(ids))
            ids += [3] + tok.encode(" " + option)
        ids += [2] + tok.encode("state") + [2]
        common = Mock()
        common.QTYPES = {"choice": 0}
        common.render_options.return_value = options
        common.build_sequence.return_value = (ids, markers, {}, {"truncated": False})
        backend.common = common
        prepared = backend.prepare_questions("state", {"q": q})
        self.assertEqual(prepared[0]["question"], normalized)
        for index in (2, markers[0] + 2, len(ids) - 2):
            shortened = ids[:index] + ids[index + 1:]
            common.build_sequence.return_value = (shortened, markers, {}, {"truncated": False})
            with self.subTest(index=index), self.assertRaises(encoders.InputRejected):
                backend.prepare_questions("state", {"q": q})
        backend.model.assert_not_called()

    def test_laya_typed_shapes_and_bounds(self):
        valid = [{"type": "noul", "instructions": "Is it true?"},
                 {"type": "score", "instructions": "Urgency", "criteria": ["low", "high"]}]
        for question in valid:
            self.assertEqual(encoders.LayaBackend.normalize_question(question)["t"], question["type"])
        for question in ({"type": "text", "instructions": "Write"},
                         {"type": "score", "instructions": "Urgency", "criteria": ["only"]},
                         {"type": "noul", "instructions": "?", "criteria": {"maybe": "unknown"}}):
            with self.assertRaises(ValueError):
                encoders.LayaBackend.normalize_question(question)

    def bundle(self, directory, split="development", rotations=1):
        row = {"subject": "subject", "question": "question", "choices": ["one", "two", "three", "four"],
               "answer": 0, "id": "source/0", "split": split, "examples": []}
        row["question_id"] = data.question_key(row)
        records = [{**data.permute(row, rotation), "target_index": data.permute(row, rotation)["answer"]}
                   for rotation in range(rotations)]
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        audit = directory / "split_audit.json"
        audit.write_text(json.dumps({"prepared_counts": {"test": 1}}))
        meta = {"protocol": "geist-decision-encoder-cases-v1", "split": split, "shots_available": 0,
                "rotations": rotations, "questions": 1, "records": rotations,
                "cases_sha256": encoders.sha256(path), "split_audit_sha256": encoders.sha256(audit),
                "per_subject": 0, "subject_limit": 0}
        (directory / "metadata.json").write_text(json.dumps(meta))
        return records, meta

    def test_population_integrity_and_purpose_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows, meta = self.bundle(directory)
            self.assertEqual(len(evaluation.load_cases(directory, "development", "pilot")[1]), 1)
            with self.assertRaises(ValueError):
                evaluation.load_cases(directory, "development", "quality")
            (directory / "development.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                evaluation.load_cases(directory, "development", "pilot")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.bundle(directory, "test")
            with self.assertRaises(ValueError):
                evaluation.load_cases(directory, "test", "pilot")
            self.assertEqual(len(evaluation.load_cases(directory, "test", "quality")[1]), 1)

    def test_rotations_do_not_inflate_quality_and_rejections_count_wrong(self):
        rows = [{"id": "a", "question_id": "q1", "variant": 0, "subject": "s", "split": "development",
                 "target_index": 0},
                {"id": "b", "question_id": "q2", "variant": 0, "subject": "s", "split": "development",
                 "target_index": 1},
                {"id": "rotated", "question_id": "q1", "variant": 1, "subject": "s", "split": "development",
                 "target_index": 3}]
        success = {"status": "scored", "case_id": "a", "choice": "A", "logits": [2., 0., 0., 0.],
                   "request_ms": 2., "forward_ms": 1., "prepare_ms": .5, "input_ids": [1, 2]}
        samples = [{**success, "trial": i} for i in range(2)]
        samples.append({"status": "rejected", "case_id": "b", "reason": "would truncate"})
        samples.append({**success, "case_id": "rotated", "choice": "D", "trial": 1})
        result = evaluation.summarize(samples, rows, 1)
        self.assertEqual((result["questions"], result["correct"], result["scored"], result["rejected"]), (2, 1, 1, 1))
        self.assertEqual(result["accuracy_rejections_count_incorrect"], .5)
        self.assertEqual(result["quality_on_scored_only"]["n"], 1)
        self.assertFalse(result["quality_matched_speedup_established"])

    def test_checkpoint_probabilities_are_distinct_from_raw_logit_metrics(self):
        row = {"id": "a", "question_id": "q", "variant": 0, "subject": "s", "split": "development", "target_index": 0}
        sample = {"status": "scored", "case_id": "a", "choice": "A", "logits": [2., 0., 0., 0.],
                  "temperature": 2., "trial": 1, "request_ms": 2., "forward_ms": 1., "prepare_ms": .5, "input_ids": [1]}
        result = evaluation.summarize([sample], [row], 1)
        self.assertEqual(result["quality_on_scored_only"]["temperature"], 1.)
        scaled = result["quality_checkpoint_scaled_on_scored_only"]
        self.assertEqual(scaled["temperature"], 2.)
        self.assertGreater(scaled["log_loss"], result["quality_on_scored_only"]["log_loss"])

    def test_repeat_prediction_drift_is_exposed(self):
        row = {"id": "a", "question_id": "q", "variant": 0, "subject": "s", "split": "development", "target_index": 0}
        sample = {"status": "scored", "case_id": "a", "choice": "A", "logits": [1., 0., 0., 0.],
                  "trial": 0, "request_ms": 2., "forward_ms": 1., "prepare_ms": .5, "input_ids": [1]}
        changed = {**sample, "trial": 1, "choice": "B", "logits": [0., 1., 0., 0.]}
        result = evaluation.summarize([sample, changed], [row], 1)
        self.assertFalse(result["predictions_stable"])
        self.assertEqual(result["max_repeat_logit_abs_drift"], 1.)


if __name__ == "__main__":
    unittest.main()
