#!/usr/bin/env python3
"""Unit tests for the cross-engine benchmark's normalization and guards."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bench_cross_engine", ROOT / "tools" / "bench_cross_engine.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
import bench_host as HOST  # noqa: E402  (tools/ is on sys.path now)


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
        # Field values mirror a real llama-bench Metal row at the pinned
        # commit: the backend token is "MTL", not "Metal".
        workload = {"repeats": 3, "backend": "gpu", "gpu_offload_layers": 99,
                    "expect_llama_backend": "MTL"}
        valid = {
            "build_commit": "2d8d612e4",
            "n_gpu_layers": 99,
            "gpu_info": "Apple M1 Max",
            "backends": "MTL,BLAS",
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

    def test_idle_gate_replaces_load_where_a_profile_sets_it(self) -> None:
        load_gate = {"max_load_per_core": 0.2}
        idle_gate = {"max_load_per_core": 0.2, "min_cpu_idle_pct": 80}
        busy_load_idle_cpu = {"load_per_core": 0.5, "cpu_idle_pct": 85.0}
        self.assertFalse(HOST.host_is_quiet(busy_load_idle_cpu, load_gate))
        self.assertTrue(HOST.host_is_quiet(busy_load_idle_cpu, idle_gate))
        self.assertFalse(HOST.host_is_quiet({"load_per_core": 0.0, "cpu_idle_pct": 70.0}, idle_gate))
        self.assertFalse(HOST.host_is_quiet({"load_per_core": 0.0, "cpu_idle_pct": None}, idle_gate),
                         "an unreadable idle share is not quiet")

    def test_repeat_spread_is_the_worst_row_and_metric(self) -> None:
        result = {"rows": [
            {"prefill_tps": [100.0, 101.0, 99.0], "decode_tps": [10.0, 10.0, 10.0]},
            {"prefill_tps": [50.0, 50.0, 50.0], "decode_tps": [20.0, 18.0, 19.0]},
        ]}
        self.assertAlmostEqual(BENCH.repeat_spread_pct(result), 2.0 / 19.0 * 100.0)

    def test_cpu_idle_share_is_readable_here(self) -> None:
        idle = HOST.cpu_idle_pct(1.0)
        self.assertIsNotNone(idle)
        self.assertTrue(0.0 <= idle <= 100.0)


FAKE_MACMON = """#!{python}
# One macmon sample per call: the next GPU,CPU pair from the readings file.
import json, pathlib
path = pathlib.Path({readings!r})
lines = path.read_text().splitlines()
gpu, cpu = map(float, lines[0].split(","))
path.write_text("\\n".join(lines[1:] or lines[:1]))
print(json.dumps({{"gpu_freq_mhz": 389, "temp": {{"gpu_temp_avg": gpu, "cpu_temp_avg": cpu}}}}))
"""


class AppleCoolGateTest(unittest.TestCase):
    """The macmon temperature gate (#17), against a fake macmon."""

    LIMITS = {"max_gpu_c": 65, "max_cpu_c": 70, "fallback_seconds": 45}

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.readings = Path(directory.name) / "readings"
        self.macmon = Path(directory.name) / "macmon"
        self.macmon.write_text(FAKE_MACMON.format(python=sys.executable,
                                                  readings=str(self.readings)))
        self.macmon.chmod(0o755)

    def gate(self, readings: list[str], deadline: float = float("inf"),
             macmon: str | None = "fake") -> tuple[dict, list[float]]:
        self.readings.write_text("\n".join(readings))
        found = str(self.macmon) if macmon else None
        with mock.patch.object(HOST, "find_macmon", return_value=found), \
             mock.patch.object(HOST, "_sleep") as sleep:
            state = HOST.wait_for_apple_cool(self.LIMITS, deadline)
        return state, [call.args[0] for call in sleep.call_args_list]

    def test_macmon_is_found_off_path(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": "/nonexistent"}), \
             mock.patch.object(HOST.shutil, "which", return_value=None) as which:
            self.assertIsNone(HOST.find_macmon())
        self.assertIn("/opt/homebrew/bin", which.call_args.kwargs["path"].split(os.pathsep))

    def test_below_the_limits_starts_at_once(self) -> None:
        state, sleeps = self.gate(["48.24,55.0"])
        self.assertEqual(sleeps, [])
        self.assertEqual(state["cool_gate"], "temperature")
        self.assertEqual((state["gpu_temp_c"], state["cpu_temp_c"]), (48.2, 55.0))

    def test_a_warm_chip_waits_until_both_limits_hold(self) -> None:
        # GPU hot, then GPU cool but CPU hot, then both cool.
        state, sleeps = self.gate(["88.0,60.0", "63.0,75.0", "62.5,66.0"])
        self.assertEqual(sleeps, [HOST.APPLE_TEMP_POLL_SECONDS] * 2)
        self.assertEqual((state["gpu_temp_c"], state["cpu_temp_c"]), (62.5, 66.0))

    def test_without_macmon_the_gate_pauses_and_says_so(self) -> None:
        state, sleeps = self.gate([], macmon=None)
        self.assertEqual(sleeps, [45])
        self.assertEqual(state, {"cool_gate": "time", "cool_wait_seconds": 45,
                                 "gpu_temp_c": None, "cpu_temp_c": None})

    def test_unreadable_macmon_output_falls_back_to_the_pause(self) -> None:
        self.macmon.write_text("#!/bin/sh\necho not-json\n")
        state, sleeps = self.gate(["0,0"])
        self.assertEqual((state["cool_gate"], sleeps), ("time", [45]))

    def test_a_chip_that_stays_hot_fails_as_busy(self) -> None:
        with self.assertRaisesRegex(TimeoutError, "thermal gate"):
            self.gate(["90.0,90.0"], deadline=0.0)

    def test_quiet_gate_records_the_start_temperatures(self) -> None:
        gate = {"max_load_per_core": 0.2, "settle_seconds": 0, "cooldown_seconds": 0,
                "timeout_seconds": 60, "apple_temp_gate": self.LIMITS}
        self.readings.write_text("70.0,60.0\n50.0,60.0")
        with mock.patch.object(HOST, "machine_state",
                               return_value={"load_1m": 0.0, "load_per_core": 0.0}), \
             mock.patch.object(HOST, "find_macmon", return_value=str(self.macmon)), \
             mock.patch.object(HOST.time, "sleep"):
            state = HOST.wait_for_quiet(gate)
        self.assertEqual((state["gpu_temp_c"], state["cpu_temp_c"]), (50.0, 60.0))
        self.assertEqual(state["load_per_core"], 0.0)

    def test_quiet_gate_without_the_entry_reads_no_temperature(self) -> None:
        gate = {"max_load_per_core": 0.2, "settle_seconds": 0, "cooldown_seconds": 0,
                "timeout_seconds": 60}
        with mock.patch.object(HOST, "machine_state",
                               return_value={"load_1m": 0.0, "load_per_core": 0.0}), \
             mock.patch.object(HOST, "apple_temps") as temps, \
             mock.patch.object(HOST.time, "sleep"):
            state = HOST.wait_for_quiet(gate)
        temps.assert_not_called()
        self.assertNotIn("cool_gate", state)

    def test_apple_profiles_enable_the_gate(self) -> None:
        gpu = json.loads((ROOT / "benchmark/cross_engine_gpu_protocol.json").read_text())
        cpu = json.loads((ROOT / "benchmark/cross_engine_cpu_protocol.json").read_text())
        apple = json.loads((ROOT / "benchmark/apple_cpu_protocol.json").read_text())
        metal = gpu["host_profiles"]["apple_m1max_metal"]["host_gate"]["apple_temp_gate"]
        self.assertIn("max_gpu_c", metal)
        self.assertIn("max_cpu_c",
                      cpu["host_profiles"]["apple_m1max"]["host_gate"]["apple_temp_gate"])
        self.assertIn("max_cpu_c", apple["interleaved_ab"]["apple_temp_gate"])
        for name, profile in {**gpu["host_profiles"], **cpu["host_profiles"]}.items():
            if not name.startswith("apple"):
                self.assertNotIn("apple_temp_gate", profile.get("host_gate", {}), name)


if __name__ == "__main__":
    unittest.main()
