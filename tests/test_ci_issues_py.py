#!/usr/bin/env python3
"""tests/test_ci_issues_py.py — scripts/ci-issues.py's plan.

One issue per failing job name: a failure opens one or comments on the open
one, a success closes an open one, skipped and cancelled jobs and the
reporting job itself change nothing. Hermetic: plan() only, no network.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ci_issues", ROOT / "scripts" / "ci-issues.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)

fails = 0


def expect(cond: bool, label: str) -> None:
    global fails
    print(f"  {'✓' if cond else '✗'} {label}")
    fails += not cond


jobs = [
    {"name": "new failure", "conclusion": "failure"},
    {"name": "known failure", "conclusion": "failure"},
    {"name": "fixed", "conclusion": "success"},
    {"name": "green, no issue", "conclusion": "success"},
    {"name": "skipped", "conclusion": "skipped"},
    {"name": "cancelled with issue", "conclusion": "cancelled"},
    {"name": "still running", "status": "running"},
    {"name": "reporter", "conclusion": "failure"},
]
issues = {ci.PREFIX + "known failure": 7, ci.PREFIX + "fixed": 8,
          ci.PREFIX + "cancelled with issue": 9, ci.PREFIX + "reporter": 10}
got = ci.plan(jobs, issues, "reporter")
expect(got == [("open", "new failure", None), ("comment", "known failure", 7), ("close", "fixed", 8)],
       "open / comment / close; skipped, cancelled, running and the reporter itself left alone")
expect(ci.plan([], issues, "reporter") == [], "no jobs, no action")

print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
