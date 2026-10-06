#!/usr/bin/env python3
"""Hermetic checks for benchmark protocol, provenance, and A/B statistics."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


quality = load_module("bench_quality_perf", "tools/bench_quality_perf.py")
apple_ab = load_module("bench_mac_ab", "tools/bench_mac_ab.py")
revision_ab = load_module("bench_revision_ab", "tools/bench_revision_ab.py")
perf_gate = load_module("perf_gate", "benchmark/perf_gate.py")


class PerfRatioGateTest(unittest.TestCase):
    def test_llama_bench_rows_and_ratio(self):
        with tempfile.TemporaryDirectory() as d:
            llama = Path(d) / "llama.json"
            llama.write_text(json.dumps([
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0},
                {"n_prompt": 0, "n_gen": 64, "avg_ts": 20.0}]))
            geist = '{"metadata": {}, "measurement": {"prefill_tps": 130.0, "decode_tps": 18.0}}'
            md, rp, rd = perf_gate.ratio_report(perf_gate.parse(geist), perf_gate.llama_tps(llama))
            self.assertAlmostEqual(rp, 1.3)
            self.assertAlmostEqual(rd, 0.9)
            self.assertIn("**1.30×**", md)


bench_compare = load_module("bench_compare", "benchmark/bench_compare.py")


def _run(commit: str, prefill: float, decode: float) -> dict:
    return {"commit": commit, "model": "m.gguf",
            "rows": [{"seq_len": 32, "prefill_tps": prefill, "decode_tps": decode},
                     {"seq_len": 512, "prefill_tps": prefill, "decode_tps": decode}]}


class BenchCompareTest(unittest.TestCase):
    LIMITS = {"decode_tps": 3.0, "prefill_tps": 5.0}

    def test_noise_passes_and_a_drop_past_the_limit_fails(self):
        base = _run("aaa", 45.0, 15.0)
        _, bad = bench_compare.compare(_run("bbb", 44.0, 14.7), base, self.LIMITS)
        self.assertEqual(bad, [])
        _, bad = bench_compare.compare(_run("bbb", 45.0, 14.4), base, self.LIMITS)
        self.assertEqual(len(bad), 2)  # decode -4 % on both rows
        self.assertIn("decode_tps", bad[0])
        _, bad = bench_compare.compare(_run("bbb", 42.0, 15.0), base, self.LIMITS)
        self.assertEqual(len(bad), 2)  # prefill -6.7 %

    def test_contended_run_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as d:
            base, cur = Path(d) / "base.json", Path(d) / "cur.json"
            base.write_text(json.dumps(_run("aaa", 45.0, 15.0)))
            noisy = _run("bbb", 45.0, 14.0)  # -6.7 % decode, would fail the gate...
            noisy["rows"][1]["spread_pct"] = 15.4  # ...but the 512 row says the box was busy
            cur.write_text(json.dumps(noisy))
            with mock.patch.object(sys, "argv", ["bench_compare", str(cur), "--baseline", str(base)]), \
                 mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GITHUB_STEP_SUMMARY", None)
                self.assertEqual(bench_compare.main(), 3)
            noisy["rows"][1]["spread_pct"] = 1.2
            cur.write_text(json.dumps(noisy))
            with mock.patch.object(sys, "argv", ["bench_compare", str(cur), "--baseline", str(base)]):
                self.assertEqual(bench_compare.main(), 1)  # quiet run, the drop counts

    def test_faster_is_never_a_regression(self):
        lines, bad = bench_compare.compare(_run("bbb", 60.0, 20.0), _run("aaa", 45.0, 15.0), self.LIMITS)
        self.assertEqual(bad, [])
        self.assertIn("+33.3 %", lines[2])


class BenchmarkToolsTest(unittest.TestCase):
    def test_quality_driver_uses_machine_readable_protocol(self):
        protocol = json.loads((ROOT / "benchmark/apple_cpu_protocol.json").read_text())
        for name, workload in quality.SWEEP_WORKLOAD.items():
            expected = protocol["suites"][name]
            self.assertEqual(workload["seq_len"], expected["seq_lens"][0])
            self.assertEqual(workload["decode_n"], expected["decode_n"])
            self.assertEqual(workload["warmup"], expected["warmup"])
            self.assertEqual(workload["repeats"], expected["repeats"])

    def test_apple_documentation_matches_machine_readable_protocol(self):
        protocol = json.loads((ROOT / "benchmark/apple_cpu_protocol.json").read_text())
        documentation = (ROOT / "benchmark/results/APPLE.md").read_text()
        for name, workload in protocol["suites"].items():
            row = (
                f"| `{name}` | {workload['seq_lens'][0]} | {workload['decode_n']} | "
                f"{workload['warmup']} | {workload['repeats']} | "
                f"{workload['aggregation']} |"
            )
            self.assertIn(row, documentation)

    def test_quiet_timeout_is_not_reset_after_failed_cooldown(self):
        loads = iter([(1.0, 0.1), (1.0, 0.1), (3.0, 0.3), (3.0, 0.3)])
        clocks = iter([100.0, 100.0, 131.0, 151.0])
        with mock.patch.object(apple_ab, "load_per_core", side_effect=loads), \
             mock.patch.object(apple_ab.time, "monotonic", side_effect=clocks), \
             mock.patch.object(apple_ab.time, "sleep"):
            with self.assertRaises(TimeoutError):
                apple_ab.wait_for_quiet(0.2, 30, 60, 50)

    def test_provenance_hashes_model_and_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            binary = root / "bench"
            model.write_bytes(b"model-bytes")
            binary.write_bytes(b"binary-bytes")
            metadata = quality.benchmark_metadata(
                binary,
                model,
                {"seq_len": 128, "decode_n": 32, "warmup": 8, "repeats": 3},
                None,
                "diagnostic line",
            )
            self.assertEqual(metadata["model_sha256"], quality.hash_file(model))
            self.assertEqual(metadata["binary_sha256"], quality.hash_file(binary))
            self.assertEqual(metadata["diagnostics"], ["diagnostic line"])

    def test_ttft_is_the_measured_one_when_the_sweep_reports_it(self):
        rec = {"prefill_ms": 100.0, "decode_ms": 400.0, "decode_n": 8}
        self.assertEqual(quality.ttft_ms({**rec, "ttft_ms": 101.5}), 101.5)
        self.assertEqual(quality.ttft_ms(rec), 150.0)

    def test_recorded_suites_do_not_overwrite_each_other(self):
        base = {
            "date": "2026-08-30",
            "model": "model.gguf",
            "host": "host/arm64",
            "os": "Darwin",
            "target": "mac-omp",
            "mode": "release",
            "threads": "default",
            "commit": "0123456789ab",
            "model_sha": "abcdef012345",
            "prefill": "100.0",
            "decode": "20.0",
            "spread": "±1%",
            "ttft": "10",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "APPLE.md"
            quality.update_benchmark_md(result, {**base, "suite": "small"})
            quality.update_benchmark_md(result, {**base, "suite": "detailed"})
            documentation = result.read_text()
            self.assertEqual(documentation.count("| 2026-08-30 |"), 2)
            self.assertIn("| small |", documentation)
            self.assertIn("| detailed |", documentation)

    def test_interleaved_summary_uses_every_ordered_sample(self):
        protocol = {"seq_lens": [128], "decode_n": 64, "repeats": 3}
        variants = [{"label": "base"}, {"label": "candidate"}]

        def run(label, prefill, decode):
            return {
                "variant": label,
                "rows": [{
                    "seq_len": 128,
                    "samples": {
                        "prefill_ms": prefill,
                        "decode_ms": decode,
                        "total_ms": [a + b for a, b in zip(prefill, decode)],
                    },
                }],
            }

        runs = [
            run("base", [1000.0, 1100.0, 900.0], [2000.0, 2200.0, 1800.0]),
            run("candidate", [800.0, 880.0, 720.0], [1600.0, 1760.0, 1440.0]),
        ]
        summary = apple_ab.summarize(runs, variants, protocol, "base")
        candidate = summary["candidate"]["128"]
        self.assertEqual(candidate["prefill"]["samples"], 3)
        self.assertEqual(candidate["decode"]["samples"], 3)
        self.assertAlmostEqual(candidate["prefill"]["vs_baseline_pct"], 25.0)
        self.assertAlmostEqual(candidate["decode"]["vs_baseline_pct"], 25.0)

    def test_resume_requires_an_exact_rotated_schedule_prefix(self):
        variants = [
            {"label": label, "binary": f"/{label}", "binary_sha256": label * 8}
            for label in ("a", "b", "c")
        ]
        protocol = {"cycles": 3}
        metadata = {
            "schema": "geist.benchmark.apple-ab.v1",
            "baseline": "a",
            "model": {"sha256": "model"},
            "protocol": protocol,
            "environment": {"OMP_NUM_THREADS": "default"},
            "variants": variants,
        }
        runs = [
            {"kind": "run", "cycle": 0, "position": 0, "variant": "a"},
            {"kind": "run", "cycle": 0, "position": 1, "variant": "b"},
        ]
        schedule = apple_ab.validate_resume(metadata, runs, metadata, variants, "a")
        self.assertEqual(
            [(cycle, position, variant["label"]) for cycle, position, variant in schedule],
            [(0, 0, "a"), (0, 1, "b"), (0, 2, "c"),
             (1, 0, "b"), (1, 1, "c"), (1, 2, "a"),
             (2, 0, "c"), (2, 1, "a"), (2, 2, "b")],
        )
        runs[1]["variant"] = "c"
        with self.assertRaisesRegex(ValueError, "exact schedule prefix"):
            apple_ab.validate_resume(metadata, runs, metadata, variants, "a")

    def test_resume_rejects_a_changed_binary(self):
        variants = [{"label": "base", "binary": "/base", "binary_sha256": "old"}]
        metadata = {
            "schema": "geist.benchmark.apple-ab.v1",
            "model": {},
            "protocol": {"cycles": 1},
            "environment": {},
            "variants": variants,
        }
        active = [{**variants[0], "binary_sha256": "new"}]
        with self.assertRaisesRegex(ValueError, "variant binaries"):
            apple_ab.validate_resume(metadata, [], metadata, active, "base")


def _revision_runs(times: dict[str, list[float]], seq_len: int = 512) -> list[dict]:
    """One run per variant and cycle; times[label][cycle] is its prefill."""
    return [{"cycle": cycle, "variant": label,
             "rows": [{"seq_len": seq_len, "prefill_ms": value, "decode_ms": 0.0}]}
            for label, series in times.items() for cycle, value in enumerate(series)]


REVISION_METADATA = {
    "baseline": "base",
    "control": True,
    "revisions": [{"label": "base", "ref": "main", "sha": "a" * 40},
                  {"label": "new", "ref": "HEAD", "sha": "b" * 40}],
    "model": {"file": "m.gguf", "sha256": "c" * 64},
    "protocol": {"cycles": 6, "seq_lens": "512", "decode_n": 0, "repeats": 1, "warmup": 16},
    "environment": {"OMP_WAIT_POLICY": "active"},
    "system": {"cpu": "cpu", "cores": 4, "os": "Linux", "thp": "[madvise]"},
}


class RevisionAbTest(unittest.TestCase):
    BASE = [100.0, 104.0, 98.0, 101.0, 99.0, 103.0, 97.0, 102.0]

    def test_revisions_need_unique_labels_and_leave_the_control_free(self):
        self.assertEqual(revision_ab.parse_revs(["base=main", "new=HEAD~1"]),
                         [("base", "main"), ("new", "HEAD~1")])
        for bad in (["base"], ["=main"], ["base="], ["base=a", "base=b"], ["control=main"]):
            with self.assertRaises(ValueError):
                revision_ab.parse_revs(bad)

    def test_the_bench_is_found_where_target_and_mode_put_it(self):
        self.assertEqual(revision_ab.bench_relpath(["CC=gcc-14"], "linux"),
                         Path("bin/linux/release/tests/bench_perf_sweep"))
        self.assertEqual(revision_ab.bench_relpath(["MODE=asan", "TARGET=pi5"], "linux"),
                         Path("bin/pi5/asan/tests/bench_perf_sweep"))

    def test_a_run_counts_the_median_of_its_repeats(self):
        row = {"prefill_ms": 16.3, "samples": {"prefill_ms": [9.0, 10.0, 30.0]}}
        self.assertEqual(revision_ab.run_value(row, "prefill_ms"), 10.0)
        self.assertEqual(revision_ab.run_value({"prefill_ms": 16.3}, "prefill_ms"), 16.3)

    def test_the_interval_ranks_are_the_sign_tests(self):
        # 1 - 2 P(Bin(n, 1/2) < k) >= 95 %: five cycles bound nothing, six to
        # eight only by their extremes, nine and ten by the 2nd, twelve by the
        # 3rd (nine: 1 - 2 * 10 / 512 = 96.1 %; eight: 1 - 2 * 9 / 256 = 93 %).
        ranks = {n: revision_ab.interval_rank(n) for n in (5, 6, 8, 9, 10, 12, 16)}
        self.assertEqual(ranks, {5: None, 6: 1, 8: 1, 9: 2, 10: 2, 12: 3, 16: 4})
        self.assertIsNone(revision_ab.interval_rank(revision_ab.MIN_CYCLES - 1))
        self.assertEqual(revision_ab.interval_rank(revision_ab.MIN_CYCLES), 1)

    def test_noise_stays_noise_and_a_change_is_found_either_way(self):
        jitter = [(a, a * (1.0 + 0.01 * (i % 3 - 1))) for i, a in enumerate(self.BASE)]
        same = revision_ab.paired_change(jitter)
        self.assertEqual(revision_ab.verdict(same), "within noise")
        self.assertEqual((same["low"], same["high"]), (0.99, 1.01))
        gain = revision_ab.paired_change([(a, 0.8 * a) for a in self.BASE])
        self.assertEqual(revision_ab.verdict(gain), "faster")
        self.assertAlmostEqual(gain["ratio"], 0.8)
        self.assertEqual((gain["faster"], gain["cycles"]), (8, 8))
        loss = revision_ab.paired_change([(a, 1.1 * a) for a in self.BASE])
        self.assertEqual(revision_ab.verdict(loss), "slower")

    def test_seven_of_eight_is_noise_and_nine_of_ten_is_not(self):
        def pairs(wins, n):
            return [(100.0, 97.0 if i < wins else 101.0) for i in range(n)]

        self.assertEqual(revision_ab.verdict(revision_ab.paired_change(pairs(7, 8))),
                         "within noise")
        self.assertEqual(revision_ab.verdict(revision_ab.paired_change(pairs(9, 10))), "faster")
        with self.assertRaises(ValueError):
            revision_ab.paired_change(pairs(5, 5))

    def test_the_interval_is_the_ranked_ratios(self):
        ratios = [0.90, 1.08, 0.94, 1.02, 0.96, 1.06, 0.98, 1.00, 0.92, 1.04]
        change = revision_ab.paired_change([(100.0, 100.0 * r) for r in ratios])
        self.assertEqual((change["low"], change["high"]), (0.92, 1.06))  # rank 2 of 10
        self.assertAlmostEqual(change["ratio"], 0.99)

    def test_cycles_pair_by_cycle_not_by_rank(self):
        # Faster in seven cycles, slower in one: noise at eight cycles, though
        # the variant's times sorted against the baseline's all look lower.
        base = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        new = [106.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0]
        change = revision_ab.paired_change(list(zip(base, new)))
        self.assertEqual(change["faster"], 7)
        self.assertEqual(revision_ab.verdict(change), "within noise")

    def test_drift_between_cycles_cancels_within_them(self):
        # The host doubles its speed from one cycle to the next; each cycle's
        # pair still shows the same -5 %, and so does the per-cycle ratio.
        drift = [100.0, 210.0, 90.0, 180.0, 120.0, 240.0, 95.0, 200.0]
        change = revision_ab.paired_change([(a, 0.95 * a) for a in drift])
        self.assertEqual(revision_ab.verdict(change), "faster")
        self.assertAlmostEqual(change["high"], 0.95)

    def test_cycles_are_paired_and_untimed_phases_left_out(self):
        runs = _revision_runs({"base": self.BASE, "new": [0.8 * a for a in self.BASE],
                               "control": self.BASE[1:] + self.BASE[:1]})
        runs = [run for run in runs if not (run["variant"] == "new" and run["cycle"] == 7)]
        rows = revision_ab.summarize(runs, ["base", "new", "control"], "base")
        # decode_ms reads 0.00: no decode rows. Cycle 7 lacks "new": 7 cycles.
        self.assertEqual([(r["metric"], r["variant"]) for r in rows],
                         [("prefill_ms", "new"), ("prefill_ms", "control")])
        self.assertEqual(rows[0]["cycles"], 7)
        self.assertEqual(rows[0]["verdict"], "faster")
        self.assertEqual(rows[1]["verdict"], "within noise")
        short = [run for run in runs if run["cycle"] < revision_ab.MIN_CYCLES - 1]
        self.assertEqual(revision_ab.summarize(short, ["base", "new", "control"], "base"), [])

    def test_the_report_warns_when_the_control_moves(self):
        runs = _revision_runs({"base": self.BASE, "new": self.BASE,
                               "control": [1.1 * a for a in self.BASE]})
        rows = revision_ab.summarize(runs, ["base", "new", "control"], "base")
        report = revision_ab.render_report(REVISION_METADATA, rows)
        self.assertIn("| prefill_ms @ 512 | new |", report)
        self.assertIn("within noise", report)
        self.assertIn("The control moved in 1 of 1 rows", report)
        quiet = revision_ab.render_report(REVISION_METADATA, rows[:1])
        self.assertNotIn("The control moved", quiet)


if __name__ == "__main__":
    unittest.main()
