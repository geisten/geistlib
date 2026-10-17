#!/usr/bin/env python3
"""Host plumbing shared by the benchmark runners (bench_cross_engine.py,
bench_mac_ab.py, bench_revision_ab.py, bench_quality_perf.py): provenance
hashes, the quiet-host gate, run schedules and the dispersion statistic."""
from __future__ import annotations

import hashlib
import os
import platform
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
                    return state
                quiet_since = None
        else:
            quiet_since = None
        idle = state.get("cpu_idle_pct")
        if now >= deadline:
            raise TimeoutError(
                f"host did not become quiet: load/core={state['load_per_core']:.3f}"
                + (f", idle={idle:.1f} %" if idle is not None else "")
                + f", gate={ {k: v for k, v in gate.items() if k != 'temp_gate'} }"
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
