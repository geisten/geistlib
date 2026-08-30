#!/usr/bin/env python3
"""Hermetic checks for benchmark protocol, provenance, and A/B statistics."""
import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()
