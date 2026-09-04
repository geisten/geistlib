#!/usr/bin/env python3
"""Unit tests for the cross-engine benchmark's normalization and guards."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bench_cross_engine", ROOT / "tools" / "bench_cross_engine.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class CrossEngineBenchmarkTest(unittest.TestCase):
    def test_summary_uses_all_samples_and_median(self) -> None:
        engines = [{"label": "geist"}, {"label": "llama.cpp"}]
        workload = {"seq_lens": [128], "decode_n": 64}
        runs = [
            {
                "engine": "geist",
                "rows": [{"seq_len": 128, "prefill_tps": [110.0, 100.0, 90.0],
                          "decode_tps": [30.0, 20.0, 10.0]}],
            },
            {
                "engine": "llama.cpp",
                "rows": [{"seq_len": 128, "prefill_tps": [50.0, 60.0, 70.0],
                          "decode_tps": [10.0, 15.0, 20.0]}],
            },
        ]

        summary = BENCH.summarize(runs, engines, workload)

        self.assertEqual(summary["geist"]["128"]["prefill"]["samples"], 3)
        self.assertEqual(summary["geist"]["128"]["prefill"]["median_tps"], 100.0)
        self.assertAlmostEqual(summary["geist"]["128"]["prefill"]["vs_llama_pct"],
                               100.0 * (100.0 / 60.0 - 1.0))

    def test_llama_guard_rejects_gpu_or_wrong_commit(self) -> None:
        engine = {"commit": "2d8d612e4c68d380"}
        workload = {"repeats": 3}
        valid = {
            "build_commit": "2d8d612e4",
            "n_gpu_layers": 0,
            "gpu_info": "",
            "backends": "BLAS",
            "no_kv_offload": True,
            "n_threads": 8,
            "poll": 100,
            "cpu_strict": False,
            "cpu_mask": "0x0",
            "samples_ts": [1.0, 2.0, 3.0],
        }
        BENCH.validate_llama_row(valid, engine, workload, 8)
        with self.assertRaises(RuntimeError):
            BENCH.validate_llama_row({**valid, "backends": "BLAS,Metal"}, engine, workload, 8)
        with self.assertRaises(RuntimeError):
            BENCH.validate_llama_row({**valid, "build_commit": "deadbeef"}, engine, workload, 8)

    def test_llama_guard_gpu_mode_requires_full_offload(self) -> None:
        engine = {"commit": "2d8d612e4c68d380"}
        workload = {"repeats": 3, "backend": "gpu", "gpu_offload_layers": 99,
                    "expect_llama_backend": "Metal"}
        valid = {
            "build_commit": "2d8d612e4",
            "n_gpu_layers": 99,
            "gpu_info": "Apple M1 Max",
            "backends": "BLAS,Metal",
            "no_kv_offload": False,
            "n_threads": 4,
            "poll": 100,
            "cpu_strict": False,
            "cpu_mask": "0x0",
            "samples_ts": [1.0, 2.0, 3.0],
        }
        BENCH.validate_llama_row(valid, engine, workload, 4)
        for broken in ({"n_gpu_layers": 0}, {"gpu_info": ""},
                       {"backends": "BLAS"}, {"no_kv_offload": True}):
            with self.assertRaises(RuntimeError):
                BENCH.validate_llama_row({**valid, **broken}, engine, workload, 4)

    def test_resume_requires_an_exact_schedule_prefix(self) -> None:
        engines = [{"label": "geist"}, {"label": "llama.cpp"}]
        current = {
            "schema": "geist.benchmark.cross-engine-run.v1",
            "protocol": {"workload": {"cycles": 1}},
            "protocol_sha256": "protocol",
            "runner": {"sha256": "runner"},
            "model": {"sha256": "model"},
            "engines": engines,
            "system": {"cpu": "fixture"},
        }
        records = [
            {"kind": "metadata", **current},
            {"kind": "run", "cycle": 0, "position": 0, "engine": "geist"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl.partial"
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            saved, runs = BENCH.load_resume(path, current, engines)
            self.assertEqual(saved, current)
            self.assertEqual(len(runs), 1)
            records[1]["engine"] = "llama.cpp"
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            with self.assertRaises(ValueError):
                BENCH.load_resume(path, current, engines)


if __name__ == "__main__":
    unittest.main()
