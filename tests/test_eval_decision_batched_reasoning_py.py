"""Batched reasoning campaign runner (#26): chunked resume and #171-style scoring."""
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).parents[1] / "tools"
with patch.object(sys, "path", [str(TOOLS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("eval_decision_batched_reasoning",
                                                  TOOLS / "eval_decision_batched_reasoning.py")
    runner = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(runner)

# Stand-in for bench_reasoning_batch: wire "cut np ids...". Answers "A" for
# even prompt lengths (with EOS), runs into the cap otherwise.
FAKE_DRIVER = r'''#!/usr/bin/env python3
import json, sys
model, backend, prompt_cap, cap, batch = sys.argv[1:6]
interval, tau = (int(sys.argv[6]), float(sys.argv[7])) if len(sys.argv) > 6 else (0, 0.)
cap = int(cap)
print(json.dumps({"kind": "runtime", "version": "t", "backend": backend, "kv_mode": "fp32",
                  "decode_batch_max": 8, "batch": int(batch), "eos": 7}), flush=True)
last = None
tokens = 0
for case, line in enumerate(sys.stdin):
    ids = [int(x) for x in line.split()]
    cut, np = ids[:2]
    if interval:
        nc, nq = ids[2:4]
        prompt = ids[4:4 + np]
    else:
        prompt = ids[2:2 + np]
    if cut and prompt[:cut] != last:
        last = prompt[:cut]
        print(json.dumps({"kind": "prefix_prepare", "case_index": case, "elapsed_ms": 5.,
                          "cached_tokens": cut, "snapshot_bytes": 100}), flush=True)
    exit_answer, probes = -1, []
    if np % 2 == 0:
        gen, stop, text = [3, 4, 7], "eos", "x</think>A"
    else:
        gen, stop, text = [3] * cap, "limit", "x" * cap
    if interval:
        probes = [{"pos": 0, "ms": 1., "logits": [0., 0., 0., 0.]}]
        confident = stop == "limit"  # the capped ones: a confident probe at `interval`
        probes.append({"pos": interval, "ms": 1., "logits": [0., 5. if confident else 0., 0., 0.]})
        if confident:
            if tau:
                gen, stop, text, exit_answer = gen[:interval], "early_exit", "x" * interval, 1
    tokens += len(gen)
    print(json.dumps({"kind": "sample", "case_index": case, "mode": "generate_long_batched",
                      "prompt_tokens": np, "generated_ids": gen, "generation_stop": stop, "cap": cap,
                      "vocab": 20, "generated_utf8_hex": text.encode().hex(),
                      "generated_text_complete": True, "surface_decode_ms": 0.01, "prefill_ms": 1.,
                      "started_ms": 1. * case, "finished_ms": 1. * case + 10.,
                      "tokens_by_batch_size": [0, 0, len(gen), 0, 0, 0, 0, 0, 0],
                      "exit_answer": exit_answer, "probes": probes}), flush=True)
print(json.dumps({"kind": "summary", "steps": 3, "decode_ms": 100., "tokens": tokens, "wall_ms": 150.,
                  "steps_by_batch_size": [0, 0, 3, 0, 0, 0, 0, 0, 0]}), flush=True)
'''


class NoTemps:
    def __init__(self, path):
        pass

    def close(self):
        pass


class BatchedReasoningRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cases = root / "cases"
        self.cases.mkdir()
        rows = []
        for i in range(5):
            subject, shared = ("law", list(range(1, 11))) if i < 3 else ("math", [50, 51, 52, 53, 54, 55])
            prompt = shared + [100 + i] * (1 + i % 2)  # lengths alternate even / odd
            rows.append({"id": f"q{i}", "question_id": f"h{i}", "subject": subject, "target_index": 0,
                         "profiles": {"chat_reasoning": {"prompt_ids": prompt, "candidate_ids": [3, 4, 5, 6]}}})
        (self.cases / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.model = root / "model.gguf"
        self.model.write_bytes(b"model")
        self.binary = root / "fake_driver"
        self.binary.write_text(FAKE_DRIVER)
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR)
        self.out = root / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **kw):
        values = dict(cases_dir=self.cases, model=self.model, binary=self.binary, out_dir=self.out,
                      split="test", backend="metal", threads=1, align=4, cap=4, batch=8,
                      chunk_questions=2, deadline_minutes=5., cool_max_c=65., cool_timeout_minutes=1.,
                      status=False, probe="198,248069,271", probe_interval=0, exit_tau=0.0,
                      question_ids=None)
        values.update(kw)
        return SimpleNamespace(**values)

    def run_all(self, **kw):
        with patch.object(runner.campaign, "TempLog", NoTemps), \
                patch.object(runner.campaign, "cool_gate", lambda a, d: {"waited": 0}):
            for _ in range(10):
                status = runner.run(self.args(**kw))
                if status["pending"] == 0:
                    return status
        self.fail("campaign did not finish")

    def test_chunked_run_scores_like_171(self):
        status = self.run_all()
        self.assertEqual(status["chunks"], 3)
        s = json.loads((self.out / "report.json").read_text())["summary"]
        self.assertEqual(s["questions"], 5)
        # Even prompt lengths answer A (target 0) with EOS; odd ones hit the cap.
        # Lengths 11, 12, 11 (law) and 8, 7 (math): q1 and q3 answer.
        self.assertEqual(s["capped"], 3)
        self.assertEqual(s["correct"], 2)
        self.assertEqual(s["per_question"]["h0"]["stop"], "limit")
        self.assertTrue(s["per_question"]["h1"]["correct"])
        self.assertEqual(s["prefix_prepares"], 4)  # both subjects span two chunks

    def test_probes_and_early_exit(self):
        self.run_all(probe_interval=2)
        s = json.loads((self.out / "report.json").read_text())["summary"]
        self.assertEqual(s["early_exits"], 0)
        self.assertEqual(s["probes"], 10)  # every question at 0 and 2
        self.assertEqual(len(s["per_question"]["h0"]["probes"]), 2)
        # With tau 0.9 the capped questions stop at token 2 with option B (wrong: target A).
        self.out = self.out.parent / "run-exit"
        self.run_all(probe_interval=2, exit_tau=0.9)
        s = json.loads((self.out / "report.json").read_text())["summary"]
        self.assertEqual(s["early_exits"], 3)
        self.assertEqual(s["capped"], 0)
        self.assertEqual(s["correct"], 2)
        self.assertEqual(s["per_question"]["h0"]["generated_tokens"], 2)

    def test_question_ids_restrict_the_run(self):
        ids = self.out.parent / "ids.json"
        ids.write_text(json.dumps(["h1", "h3"]))
        self.run_all(question_ids=ids)
        s = json.loads((self.out / "report.json").read_text())["summary"]
        self.assertEqual(s["questions"], 2)
        self.assertEqual(s["correct"], 2)

    def test_resume_refuses_a_changed_manifest(self):
        with patch.object(runner.campaign, "TempLog", NoTemps), \
                patch.object(runner.campaign, "cool_gate", lambda a, d: {"waited": 0}):
            runner.run(self.args())
            with self.assertRaises(ValueError):
                runner.run(self.args(batch=4))


if __name__ == "__main__":
    unittest.main()
