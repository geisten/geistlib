"""Lossless token boundary and tokenizer-process failure contracts, offline."""
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("prepare_decision_eval", TOOLS / "prepare_decision_eval.py")
prepare = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(TOOLS), *sys.path]):
    SPEC.loader.exec_module(prepare)


class FakeTokenizer:
    bos = 9

    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text):
        return self.mapping[text]


class PreparationTests(unittest.TestCase):
    def mapping(self):
        return {"Answer:": [1, 2], "Answer: A": [1, 2, 3], "Answer: B": [1, 2, 4]}

    def test_actual_prefix_plus_one_token_not_standalone_truncation(self):
        good = prepare.token_case(FakeTokenizer(self.mapping()), "Answer:", [" A", " B"])
        self.assertEqual(good["prompt_ids"], [9, 1, 2])
        self.assertEqual(good["candidate_ids"], [3, 4])
        for continuation in ([1, 2, 3, 5], [1, 6, 3], [1, 2]):
            mapping = {**self.mapping(), "Answer: A": continuation}
            with self.assertRaisesRegex(ValueError, "actual prompt boundary"):
                prepare.token_case(FakeTokenizer(mapping), "Answer:", [" A", " B"])
        mapping = {**self.mapping(), "Answer: B": [1, 2, 3]}
        with self.assertRaisesRegex(ValueError, "distinct"):
            prepare.token_case(FakeTokenizer(mapping), "Answer:", [" A", " B"])

    def test_protocol_rejects_text_that_would_be_truncated(self):
        tokenizer = prepare.Tokenizer.__new__(prepare.Tokenizer)
        for text in ("x\0y", "x\ry", "é" * 32768):
            with self.assertRaisesRegex(ValueError, "lossless"):
                tokenizer.encode(text)

    def test_initialization_failure_closes_log_and_process_pipes(self):
        class Log:
            def __init__(self):
                self.stream = io.StringIO()
            def open(self, mode):
                return self.stream
        class Process:
            stdin = io.StringIO()
            stdout = io.StringIO("READY\nOK 1 not-a-number\n")
            def poll(self):
                return 0
        log, process = Log(), Process()
        with patch.object(prepare.subprocess, "Popen", return_value=process):
            with self.assertRaises(ValueError):
                prepare.Tokenizer("binary", "model", log)
        self.assertTrue(log.stream.closed)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        log = Log()
        with patch.object(prepare.subprocess, "Popen", side_effect=OSError("unavailable")):
            with self.assertRaises(OSError):
                prepare.Tokenizer("binary", "model", log)
        self.assertTrue(log.stream.closed)


if __name__ == "__main__":
    unittest.main()
