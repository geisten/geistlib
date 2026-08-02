#!/usr/bin/env python3
"""tests/test_coverage_gate_py.py — control test for the coverage ratchet.

#185 demands proof that the gate actually fires: feed the gate synthetic
gcovr JSON and baselines, and assert PASS on-baseline, FAIL on a regression
beyond tolerance, FAIL on an empty scope, FAIL on an unset baseline, and
FAIL on a floor violation. Hermetic — no compiler, no gcovr, no network.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

GATE = Path(__file__).resolve().parent.parent / "scripts" / "coverage_gate.py"

fails = 0


def expect(cond: bool, label: str) -> None:
    global fails
    if cond:
        print(f"  ✓ {label}")
    else:
        print(f"  ✗ {label}")
        fails += 1


def gcovr_file(path: str, hit: int, total: int, bhit: int, btotal: int) -> dict:
    lines = []
    for i in range(total):
        line = {"line_number": i + 1, "count": 1 if i < hit else 0, "branches": []}
        if i < btotal:
            line["branches"] = [{"count": 1 if i < bhit else 0}]
        lines.append(line)
    return {"file": path, "lines": lines}


def run_gate(files: list[dict], baselines: dict) -> int:
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "report.json"
        bp = Path(td) / "baselines.json"
        rp.write_text(json.dumps({"files": files}))
        bp.write_text(json.dumps(baselines))
        proc = subprocess.run(
            [sys.executable, str(GATE), str(rp), "--baselines", str(bp)],
            capture_output=True,
            text=True,
        )
        return proc.returncode


BASE = {
    "tolerance_pp": 0.5,
    "subsystems": {"src/engine": {"line": 80, "branch": 60}},
    "floors": {"src/engine": {"line": 35.0}},
    "overall": {"line": 80, "branch": 60},
}

# 80% line (80/100), 60% branch (30/50) — exactly on baseline.
ON_BASELINE = [gcovr_file("src/engine/session.c", 80, 100, 30, 50)]
# 70% line — 10pp under baseline, far beyond the 0.5pp tolerance.
REGRESSED = [gcovr_file("src/engine/session.c", 70, 100, 30, 50)]
# 79.6% line — inside the 0.5pp tolerance band.
IN_TOLERANCE = [gcovr_file("src/engine/session.c", 796, 1000, 300, 500)]

print("coverage-gate control:")
expect(run_gate(ON_BASELINE, BASE) == 0, "on-baseline passes")
expect(run_gate(IN_TOLERANCE, BASE) == 0, "dip inside 0.5pp tolerance passes")
expect(run_gate(REGRESSED, BASE) == 1, "regression beyond tolerance FAILS")
expect(run_gate([], BASE) == 1, "empty scope FAILS (not 100%)")

unset = json.loads(json.dumps(BASE))
unset["subsystems"]["src/engine"]["line"] = None
expect(run_gate(ON_BASELINE, unset) == 1, "unset baseline FAILS and demands a commit")

floor = json.loads(json.dumps(BASE))
floor["subsystems"]["src/engine"] = {"line": 20, "branch": 10}
floor["overall"] = {"line": 20, "branch": 10}
expect(
    run_gate([gcovr_file("src/engine/session.c", 20, 100, 10, 50)], floor) == 1,
    "hard floor fires independently of a satisfied ratchet",
)

if fails:
    print(f"{fails} check(s) failed")
    sys.exit(1)
print("PASS: coverage gate fires when it must and only then")
