"""Shared-prefix campaign runner (#26): per-subject aligned cuts, chunked resume, the report's identity gate."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).parents[1] / "tools"
with patch.object(sys, "path", [str(TOOLS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("eval_decision_prefix", TOOLS / "eval_decision_prefix.py")
    runner = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(runner)

# A stand-in for bench_decision --prefix: wire "cut np nc ids... cands...",
# a prefix_prepare record whenever the cached tokens change, both arms per
# case with logits that depend only on the prompt (FAKE_DRIFT perturbs the
# prefix arm to exercise the mismatch path).
FAKE_DRIVER = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
assert "--prefix" in args
modes = args[args.index("--modes") + 1].split(",")
drift = float(os.environ.get("FAKE_DRIFT", "0"))
print(json.dumps({"kind": "runtime", "backend": args[1], "vocab": 1000, "kv_mode": "fp32", "version": "t"}), flush=True)
cached = None
for case, line in enumerate(sys.stdin):
    ids = [int(x) for x in line.split()]
    cut, np, nc = ids[:3]
    prompt = ids[3:3 + np]
    if cached != prompt[:cut]:
        cached = prompt[:cut]
        print(json.dumps({"kind": "prefix_prepare", "case_index": case, "elapsed_ms": 50., "cached_tokens": cut,
                          "snapshot_bytes": 1000 + cut, "prefill_chunk": int(os.environ.get("FAKE_CHUNK", "4"))}), flush=True)
    base = [float(sum(prompt) % 7), 1., 0., -1.]
    for order, mode in enumerate(modes):
        logits = [base[0] + (drift if mode == "decision_prefix" else 0.)] + base[1:]
        best = max(range(nc), key=lambda i: logits[i])
        s = {"kind": "sample", "case_index": case, "mode": mode, "trial": 0, "phase": "first", "order": order,
             "elapsed_ms": 100. if mode == "decision_dense" else 30., "prompt_tokens": np, "candidate_count": nc,
             "generated_ids": [], "best_index": best, "logits": logits, "conditional_probabilities": [.25] * 4,
             "projected_rows": 1000, "logit_readback_bytes": 4000, "head_ns": 0, "generation_stop": "not_generated",
             "process_peak_rss_bytes": 5000}
        if mode == "decision_prefix":
            s["cached_tokens"] = cut
        print(json.dumps(s), flush=True)
'''


class NoTemps:
    def __init__(self, path):
        pass

    def close(self):
        pass


class PrefixRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cases = root / "cases"
        self.cases.mkdir()
        rows = []
        # Two subjects; "law" shares 10 tokens, "math" shares 6 (cut 9 -> 8 and 5 -> 4 at align 4).
        for i in range(5):
            subject, shared = ("law", list(range(1, 11))) if i < 3 else ("math", [50, 51, 52, 53, 54, 55])
            prompt = shared + [100 + i, 200 + i]
            rows.append({"id": f"q{i}", "subject": subject, "target_index": i % 2,
                         "profiles": {"chat_direct": {"prompt_ids": prompt, "candidate_ids": [3, 4, 5, 6]}}})
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
        values = dict(cases_dir=self.cases, model=self.model, binary=self.binary, out_dir=self.out, split="test",
                      profile="chat_direct", backend="metal", threads=1, align=4, chunk_questions=2,
                      deadline_minutes=5., cool_max_c=65., cool_timeout_minutes=1., status=False,
                      reference_from=None)
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

    def test_cuts_are_aligned_and_leave_a_suffix(self):
        rows = runner.load_rows(self.cases, "test", "chat_direct")
        order, cuts = runner.plan_cuts(rows, "chat_direct", 4)
        self.assertEqual(cuts["law"], {"questions": 3, "shared_tokens": 10, "cut": 8})
        self.assertEqual(cuts["math"], {"questions": 2, "shared_tokens": 6, "cut": 4})
        self.assertEqual(order, [0, 1, 2, 3, 4])

    def test_chunked_run_reports_identity_accuracy_and_cost(self):
        status = self.run_all()
        self.assertEqual(status["chunks"], 3)
        s = json.loads((self.out / "report.json").read_text())["summary"]
        self.assertEqual(s["bit_identical_option_logits"], 5)
        self.assertEqual(s["argmax_identical"], 5)
        self.assertEqual(s["accuracy"]["decision_dense"], s["accuracy"]["decision_prefix"])
        # Chunks [q0 q1] [q2 q3] [q4]: both subjects span two chunks, so each
        # is prepared twice; one prepare per subject counts.
        self.assertEqual(s["prepare_ms"]["measured_prepares"], 4)
        self.assertEqual(s["prepare_ms"]["one_per_subject_sum"], 100.)
        self.assertAlmostEqual(s["end_to_end"]["ratio"], (5 * 30. + 100.) / (5 * 100.))
        self.assertEqual(s["by_subject"]["law"]["cached_tokens"], 8)

    def test_mismatch_is_reported(self):
        with patch.dict(os.environ, {"FAKE_DRIFT": "0.5"}):
            self.run_all()
        s = json.loads((self.out / "report.json").read_text())["summary"]
        self.assertEqual(s["bit_identical_option_logits"], 0)
        self.assertEqual(len(s["mismatches"]), 5)

    def test_engine_chunk_must_match_the_declaration(self):
        with patch.dict(os.environ, {"FAKE_CHUNK": "8"}), self.assertRaises(ValueError):
            self.run_all()

    def test_prefix_only_run_takes_the_reference_run(self):
        self.run_all()
        first = self.out
        self.out = first.parent / "run2"
        with patch.dict(os.environ, {"FAKE_CHUNK": "2"}):
            self.run_all(align=2, reference_from=first)
        s = json.loads((self.out / "report.json").read_text())["summary"]
        self.assertEqual(s["bit_identical_option_logits"], 5)
        self.assertEqual(s["by_subject"]["law"]["cached_tokens"], 8)  # 9 aligned down to 2
        self.assertEqual(s["by_subject"]["math"]["cached_tokens"], 4)
        driver = (self.out / "chunks" / "0000" / "prefix.driver.jsonl").read_text()
        self.assertNotIn("decision_dense", driver)

    def test_reference_run_must_share_the_binding(self):
        self.run_all()
        first = self.out
        self.out = first.parent / "run2"
        with self.assertRaises(ValueError):
            self.run_all(threads=2, reference_from=first)

    def test_subjects_sharing_an_empty_cut_reuse_one_prepare(self):
        # align 16 empties both cuts: one prepare serves both subjects.
        with patch.dict(os.environ, {"FAKE_CHUNK": "16"}):
            self.run_all(align=16, chunk_questions=5)
        s = json.loads((self.out / "report.json").read_text())["summary"]
        self.assertEqual(s["prepare_ms"]["measured_prepares"], 1)
        self.assertTrue(s["by_subject"]["math"]["prefix_reused_from_previous_subject"])
        self.assertEqual(s["by_subject"]["math"]["prepare_ms"], 0.)

    def test_resume_refuses_a_changed_manifest(self):
        with patch.object(runner.campaign, "TempLog", NoTemps), \
                patch.object(runner.campaign, "cool_gate", lambda a, d: {"waited": 0}):
            runner.run(self.args())
            with self.assertRaises(ValueError):
                runner.run(self.args(align=2))


if __name__ == "__main__":
    unittest.main()
