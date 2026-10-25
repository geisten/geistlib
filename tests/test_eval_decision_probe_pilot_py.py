"""Early-exit pilot runner (#236): sampling, chunked resume, transparency, offline analysis."""
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
    SPEC = importlib.util.spec_from_file_location("eval_decision_probe_pilot",
                                                  TOOLS / "eval_decision_probe_pilot.py")
    runner = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(runner)

# Stand-in for bench_reasoning_probe: wire "np nc nq prompt cands probe".
# Every question reasons 6 tokens and answers A (EOS last); probes at 0, 2, 4
# (interval 2) show A growing confident; interval 0 runs without probes.
FAKE_DRIVER = r'''#!/usr/bin/env python3
import json, sys
model, backend, prompt_cap, cap, interval = sys.argv[1:6]
interval = int(interval)
print(json.dumps({"kind": "runtime", "version": "t", "backend": backend, "kv_mode": "fp32", "eos": 7,
                  "interval": interval}), flush=True)
for case, line in enumerate(sys.stdin):
    ids = [int(x) for x in line.split()]
    gen = [3, 3, 3, 3, 3, 7]
    probes = []
    if interval:
        for pos, a in ((0, 0.), (2, 1.), (4, 5.)):
            probes.append({"pos": pos, "ms": 30., "logits": [a, 0., 0., 0.]})
    print(json.dumps({"kind": "sample", "case_index": case, "mode": "generate_long_probed",
                      "prompt_tokens": ids[0], "generated_ids": gen, "generation_stop": "eos", "cap": int(cap),
                      "generated_utf8_hex": "x</think>A".encode().hex(), "generated_text_complete": True,
                      "surface_decode_ms": 0.01, "prefill_ms": 1., "decode_ms": 60., "probes": probes}),
          flush=True)
'''


class NoTemps:
    def __init__(self, path):
        pass

    def close(self):
        pass


class ProbePilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cases = root / "cases"
        self.cases.mkdir()
        rows = [{"id": f"q{i}", "question_id": f"h{i}", "subject": "law" if i % 2 else "math",
                 "target_index": 0 if i < 6 else 1,
                 "profiles": {"chat_reasoning": {"prompt_ids": [1, 2, 3 + i], "candidate_ids": [32, 33, 34, 35]}}}
                for i in range(10)]
        (self.cases / "calibration.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
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
                      split="calibration", sample=8, seed="t", transparency=2, probe="198,248069,271",
                      interval=2, cap=16, vocab=20, backend="metal", threads=1, chunk_questions=3,
                      deadline_minutes=5., cool_max_c=65., cool_timeout_minutes=1., status=False)
        values.update(kw)
        return SimpleNamespace(**values)

    def run_all(self, **kw):
        with patch.object(runner.campaign, "TempLog", NoTemps), \
                patch.object(runner.campaign, "cool_gate", lambda a, d: {"waited": 0}):
            for _ in range(12):
                status = runner.run(self.args(**kw))
                if status["pending"] == 0 and status["transparency_pending"] == 0:
                    return status
        self.fail("pilot did not finish")

    def test_pilot_report(self):
        status = self.run_all()
        self.assertEqual(status["chunks"], 4)  # 8 probed in 3 chunks, then the 2 plain runs
        r = json.loads((self.out / "report.json").read_text())
        self.assertEqual(r["transparency"]["identical"], 2)
        self.assertEqual(r["probe_cost"]["median_ms"], 30.)
        self.assertAlmostEqual(r["probe_cost"]["decode_ms_per_token"], 10.)
        policies = {(p["tau"], p["stable"]): p for p in r["analysis"]["policies"]}
        # tau 0.7: probe at pos 4 (p(A) = e^5/(e^5+3) ~ 0.98) exits; pos 2 (~0.48) does not.
        p = policies[(0.7, False)]
        self.assertEqual(p["early_exits"], 8)
        self.assertEqual(p["generated_tokens"], 8 * 4)
        self.assertAlmostEqual(p["token_ratio"], 4 / 6)
        self.assertEqual(p["accuracy"], r["full_reasoning"]["accuracy"])
        # tau 0.99 never exits.
        self.assertEqual(policies[(0.99, False)]["early_exits"], 0)


if __name__ == "__main__":
    unittest.main()
