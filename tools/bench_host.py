#!/usr/bin/env python3
"""Host plumbing shared by the benchmark runners (bench_cross_engine.py,
bench_mac_ab.py, bench_revision_ab.py, bench_quality_perf.py): provenance
hashes, the quiet-host gate, run schedules and the dispersion statistic.
Stdlib only: the fetch and convert tools import sha256_file from here too."""
from __future__ import annotations

import hashlib
import os
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], timeout: int = 15) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unavailable"


def machine_state() -> dict:
    load = os.getloadavg()[0]
    cores = max(os.cpu_count() or 1, 1)
    return {
        "load_1m": load,
        "load_per_core": load / cores,
        "thermal_status": (command_output(["pmset", "-g", "therm"]).splitlines()
                           if platform.system() == "Darwin"
                           else command_output(["sensors"]).splitlines()),
        "power_source": (command_output(["pmset", "-g", "batt"]).splitlines()
                         if platform.system() == "Darwin" else ["mains (desktop)"]),
        "temperature_c": None,
        "temperature_note": "unavailable to an unprivileged process",
    }


def cpu_idle_pct(seconds: float = 3.0) -> float | None:
    """Share of all cores idle over the next `seconds`, or None where it
    cannot be read. macOS: top's second sample (the first covers the time
    since boot); Linux: two /proc/stat snapshots."""
    if platform.system() == "Darwin":
        out = command_output(["top", "-l", "2", "-n", "0", "-s", str(max(1, round(seconds)))])
        lines = [line for line in out.splitlines() if line.startswith("CPU usage:")]
        if len(lines) < 2:
            return None
        try:
            return float(lines[-1].split(",")[2].split("%")[0])
        except (IndexError, ValueError):
            return None
    try:
        def snap() -> list[int]:
            with open("/proc/stat") as fh:
                return [int(v) for v in fh.readline().split()[1:]]
        a = snap()
        time.sleep(seconds)
        b = snap()
    except (OSError, ValueError):
        return None
    delta = [y - x for x, y in zip(a, b)]
    total = sum(delta)
    return 100.0 * (delta[3] + delta[4]) / total if total > 0 else None


def host_is_quiet(state: dict, gate: dict) -> bool:
    """min_cpu_idle_pct, where a profile sets it, replaces the load-average
    test: macOS on an interactive desktop never shows a load average near the
    Linux hosts' (2026-10-12: load 4-8 at 73 % idle on the M1 Max)."""
    if "min_cpu_idle_pct" in gate:
        return state.get("cpu_idle_pct") is not None and state["cpu_idle_pct"] >= gate["min_cpu_idle_pct"]
    return state["load_per_core"] <= gate["max_load_per_core"]


def read_millic(sensor: str):
    try:
        with open(sensor) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def wait_for_temp(gate: dict) -> None:
    """Block until the host is cool (#364 pi5 cell): passively cooled
    boards throttle mid-measurement otherwise. Inactive on hosts without
    the sensor path."""
    tg = gate.get("temp_gate")
    if not tg:
        return
    millic = read_millic(tg["sensor"])
    if millic is None:
        return
    deadline = time.monotonic() + gate["timeout_seconds"]
    while millic is not None and millic > tg["max_millic"]:
        if time.monotonic() > deadline:
            raise RuntimeError(f"thermal gate: still {millic} millic after timeout")
        print(f"  waiting for cool host: {millic/1000:.1f}C", flush=True)
        time.sleep(30)
        millic = read_millic(tg["sensor"])


# Homebrew's prefixes: a CI runner's service PATH may not include them.
MACMON_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
APPLE_TEMP_POLL_SECONDS = 5


def find_macmon() -> str | None:
    path = os.pathsep.join([os.environ.get("PATH", ""), *MACMON_DIRS])
    return shutil.which("macmon", path=path)


def parse_macmon(line: str) -> dict | None:
    """The GPU and CPU temperature in degrees C from one `macmon pipe` line
    (its averages over the die sensors), or None where the line cannot be
    read."""
    try:
        temp = json.loads(line)["temp"]
        return {"gpu_c": round(float(temp["gpu_temp_avg"]), 1),
                "cpu_c": round(float(temp["cpu_temp_avg"]), 1)}
    except (KeyError, TypeError, ValueError):
        return None


def apple_temps() -> dict | None:
    """The Apple SoC's GPU and CPU temperature in degrees C from one macmon
    sample (its averages over the die sensors; no root needed), or None where
    macmon is missing or its output cannot be read. The sensors refresh about
    once a second: a 200 ms sample right after a short run repeated the
    reading from before it, so the sample spans a full second."""
    macmon = find_macmon()
    if macmon is None:
        return None
    try:
        proc = subprocess.run([macmon, "pipe", "-s", "1", "-i", "1000"],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = proc.stdout.strip().splitlines()
    return parse_macmon(lines[-1]) if lines else None


def apple_too_warm(temps: dict, limits: dict) -> bool:
    return (temps["gpu_c"] > limits.get("max_gpu_c", float("inf"))
            or temps["cpu_c"] > limits.get("max_cpu_c", float("inf")))


# The gate's own waits, patchable apart from time.sleep: subprocess.run(timeout=)
# polls with time.sleep on Linux, so patching that module-wide also records
# the macmon call's internal waits (CI linux-arm64, run 1802).
_sleep = time.sleep


def wait_for_apple_cool(limits: dict, deadline: float) -> dict:
    """Block until the Apple SoC is below the gate's limits (#17): the M1 Max
    GPU holds its top clock only up to ~81-83 C, so a run that starts warm
    measures throttling and the order of the runs, not the engine. Without a
    readable macmon the gate pauses fallback_seconds instead and says so.
    Raises TimeoutError, the busy failure, when the deadline passes first."""
    started = time.monotonic()
    while True:
        temps = apple_temps()
        if temps is None:
            pause = limits["fallback_seconds"]
            print(f"  macmon unavailable: cool-down pause {pause}s", file=sys.stderr, flush=True)
            _sleep(pause)
            return {"cool_gate": "time", "cool_wait_seconds": pause,
                    "gpu_temp_c": None, "cpu_temp_c": None}
        if not apple_too_warm(temps, limits):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"thermal gate: GPU {temps['gpu_c']} C, CPU {temps['cpu_c']} C "
                               f"still above {limits} at timeout")
        print(f"  waiting for cool chip: GPU {temps['gpu_c']} C, CPU {temps['cpu_c']} C",
              file=sys.stderr, flush=True)
        _sleep(APPLE_TEMP_POLL_SECONDS)
    return {"cool_gate": "temperature",
            "cool_wait_seconds": round(time.monotonic() - started, 1),
            "gpu_temp_c": temps["gpu_c"], "cpu_temp_c": temps["cpu_c"]}


def gate_state(gate: dict) -> dict:
    state = machine_state()
    if "min_cpu_idle_pct" in gate:
        state["cpu_idle_pct"] = cpu_idle_pct()
    return state


def wait_for_quiet(gate: dict) -> dict:
    wait_for_temp(gate)
    deadline = time.monotonic() + gate["timeout_seconds"]
    quiet_since: float | None = None
    last_report = float("-inf")
    while True:
        state = gate_state(gate)
        now = time.monotonic()
        if host_is_quiet(state, gate):
            quiet_since = quiet_since or now
            if now - quiet_since >= gate["settle_seconds"]:
                if gate["cooldown_seconds"] > 0:
                    print(f"  cooldown {gate['cooldown_seconds']}s", file=sys.stderr, flush=True)
                    time.sleep(gate["cooldown_seconds"])
                state = gate_state(gate)
                if host_is_quiet(state, gate):
                    if "apple_temp_gate" in gate:
                        state.update(wait_for_apple_cool(gate["apple_temp_gate"], deadline))
                    return state
                quiet_since = None
        else:
            quiet_since = None
        idle = state.get("cpu_idle_pct")
        if now >= deadline:
            raise TimeoutError(
                f"host did not become quiet: load/core={state['load_per_core']:.3f}"
                + (f", idle={idle:.1f} %" if idle is not None else "")
                + f", gate={ {k: v for k, v in gate.items() if k not in ('temp_gate', 'apple_temp_gate')} }"
            )
        if now - last_report >= 60:
            print(
                f"  waiting for quiet host: load {state['load_1m']:.2f} "
                f"({state['load_per_core']:.3f}/core)"
                + (f", {idle:.1f} % idle" if idle is not None else ""),
                file=sys.stderr,
                flush=True,
            )
            last_report = now
        time.sleep(10)


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def build_schedule_alternating(engines: list[dict], cycles: int) -> list[tuple[int, int, dict]]:
    """Even cycles run the given order, odd cycles the reverse (ABBA)."""
    schedule = []
    for cycle in range(cycles):
        order = engines if cycle % 2 == 0 else list(reversed(engines))
        schedule.extend((cycle, position, engine) for position, engine in enumerate(order))
    return schedule


def build_schedule_rotating(variants: list[dict], cycles: int) -> list[tuple[int, int, dict]]:
    """Cycle c starts at variant c mod n: every variant takes every position."""
    schedule = []
    for cycle in range(cycles):
        offset = cycle % len(variants)
        order = variants[offset:] + variants[:offset]
        schedule.extend((cycle, position, variant) for position, variant in enumerate(order))
    return schedule


def perf_sweep_command(binary, gguf, seq_lens, decode_n: int, warmup: int, repeats: int,
                       threads: int | str | None = None) -> list[str]:
    """argv for one bench_perf_sweep run emitting JSONL; seq_lens is a
    comma-separated string or a sequence of ints."""
    if not isinstance(seq_lens, str):
        seq_lens = ",".join(map(str, seq_lens))
    command = [str(binary), "--gguf", str(gguf), "--seq-lens", seq_lens,
               "--decode-n", str(decode_n), "--warmup", str(warmup), "--repeats", str(repeats)]
    if threads is not None:
        command += ["--threads", str(threads)]
    return command + ["--emit-jsonl"]


def run_jsonl(command: list[str], env: dict[str, str] | None = None, label: str = "command",
              cwd: Path | None = None) -> tuple[list[dict], list[str], float]:
    """Run command; return (its stdout JSON rows, stderr lines, wall seconds).
    A nonzero exit raises RuntimeError naming label and the last diagnostic line."""
    started = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True, env=env, cwd=cwd)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        diagnostic = (proc.stderr.splitlines() or proc.stdout.splitlines() or ["no diagnostics"])[-1]
        raise RuntimeError(f"{label} exited {proc.returncode}: {diagnostic}")
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    return rows, proc.stderr.splitlines(), elapsed


def load_partial(path: Path) -> tuple[dict, list[dict]]:
    """The metadata and run records of an unfinished JSONL artifact, for --resume."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or records[0].get("kind") != "metadata":
        raise ValueError("resume artifact does not start with metadata")
    if any(record.get("kind") == "summary" for record in records):
        raise ValueError("resume artifact already contains a completed summary")
    unexpected = [record.get("kind") for record in records[1:]
                  if record.get("kind") not in {"run", "resume"}]
    if unexpected:
        raise ValueError(f"resume artifact contains unexpected records: {unexpected}")
    metadata = {key: value for key, value in records[0].items() if key != "kind"}
    return metadata, [record for record in records[1:] if record.get("kind") == "run"]


def check_schedule_prefix(runs: list[dict], schedule: list[tuple[int, int, dict]],
                          key: str) -> None:
    """Saved runs must be the first len(runs) slots of schedule; run[key]
    names the slot's entry by its label."""
    if len(runs) > len(schedule):
        raise ValueError("resume artifact contains too many runs")
    for run, (cycle, position, entry) in zip(runs, schedule):
        if (run.get("cycle"), run.get("position"), run.get(key)) != (
                cycle, position, entry["label"]):
            raise ValueError("resume runs are not an exact schedule prefix")
