#!/usr/bin/env python3
"""scripts/coverage_gate.py — per-subsystem line/branch coverage ratchet (#185).

Consumes a gcovr JSON report (gcovr --json), aggregates it into the
subsystems listed in the baselines file, and gates:

  * every configured subsystem must be present with at least one measured
    file — an empty scope is an error, not 100 %;
  * line and branch coverage per subsystem (and overall) may not fall more
    than `tolerance_pp` percentage points below the versioned baseline;
  * hard floors (e.g. the src/io parser-security gate, >= 35 % line) are
    checked independently of the ratchet and never tolerate a dip;
  * a baseline of null means "not yet measured": the gate prints the
    measured value and fails, so the first run on a new subsystem forces a
    deliberate, reviewed baseline commit rather than silently adopting
    whatever it saw.

Raising a baseline is routine (ratchet); lowering one is a reviewed,
justified edit to benchmark/coverage_baselines.json.

Exit codes: 0 gate passed, 1 gate failed, 2 usage/input error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"coverage_gate: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(2)


def aggregate(report: dict, prefixes: list[str]) -> dict:
    """Sum gcovr per-file counters into {prefix: {lines, lines_hit, branches,
    branches_hit, files}} buckets. A file counts toward the FIRST matching
    prefix, so more specific prefixes must be listed first in the baselines
    file if they ever overlap (they currently don't)."""
    buckets = {
        p: {"lines": 0, "lines_hit": 0, "branches": 0, "branches_hit": 0, "files": 0}
        for p in prefixes
    }
    for f in report.get("files", []):
        fname = f.get("file", "")
        for p in prefixes:
            if fname.startswith(p):
                b = buckets[p]
                b["files"] += 1
                for line in f.get("lines", []):
                    if line.get("gcovr/noncode", False):
                        continue
                    b["lines"] += 1
                    if line.get("count", 0) > 0:
                        b["lines_hit"] += 1
                    for br in line.get("branches", []):
                        b["branches"] += 1
                        if br.get("count", 0) > 0:
                            b["branches_hit"] += 1
                break
    return buckets


def pct(hit: int, total: int) -> float | None:
    return None if total == 0 else 100.0 * hit / total


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="gcovr JSON report (gcovr --json)")
    ap.add_argument("--baselines", required=True, help="benchmark/coverage_baselines.json")
    ap.add_argument("--summary-out", default=os.environ.get("GITHUB_STEP_SUMMARY"),
                    help="markdown summary sink (defaults to $GITHUB_STEP_SUMMARY)")
    args = ap.parse_args()

    report = load_json(args.report)
    cfg = load_json(args.baselines)
    tol = float(cfg.get("tolerance_pp", 0.5))
    subsystems: dict = cfg.get("subsystems", {})
    floors: dict = cfg.get("floors", {})
    if not subsystems:
        print("coverage_gate: baselines file lists no subsystems", file=sys.stderr)
        return 2

    buckets = aggregate(report, list(subsystems.keys()))
    failures: list[str] = []
    rows: list[tuple] = []

    overall = {"lines": 0, "lines_hit": 0, "branches": 0, "branches_hit": 0}
    for sub, base in subsystems.items():
        b = buckets[sub]
        for k in overall:
            overall[k] += b[k]
        line = pct(b["lines_hit"], b["lines"])
        branch = pct(b["branches_hit"], b["branches"])
        if b["files"] == 0:
            failures.append(f"{sub}: no files measured — empty scope is a failure")
            rows.append((sub, 0, line, base.get("line"), branch, base.get("branch")))
            continue
        rows.append((sub, b["files"], line, base.get("line"), branch, base.get("branch")))
        for metric, measured in (("line", line), ("branch", branch)):
            baseline = base.get(metric)
            if baseline is None:
                failures.append(
                    f"{sub}: {metric} baseline unset — measured {fmt(measured)}; "
                    f"commit that value (rounded DOWN to whole percent) to the baselines file")
            elif measured is None:
                failures.append(f"{sub}: no {metric} data in the report")
            elif measured < float(baseline) - tol:
                failures.append(
                    f"{sub}: {metric} coverage {fmt(measured)} fell below baseline "
                    f"{baseline}% - {tol}pp tolerance")

    for sub, fl in floors.items():
        b = buckets.get(sub)
        line = pct(b["lines_hit"], b["lines"]) if b else None
        need = float(fl.get("line", 0))
        if line is None or line < need:
            failures.append(f"{sub}: hard floor {need}% line coverage not met (measured {fmt(line)})")

    o_line = pct(overall["lines_hit"], overall["lines"])
    o_branch = pct(overall["branches_hit"], overall["branches"])
    o_base = cfg.get("overall", {})
    for metric, measured in (("line", o_line), ("branch", o_branch)):
        baseline = o_base.get(metric)
        if baseline is None:
            failures.append(f"overall: {metric} baseline unset — measured {fmt(measured)}")
        elif measured is not None and measured < float(baseline) - tol:
            failures.append(
                f"overall: {metric} coverage {fmt(measured)} fell below baseline "
                f"{baseline}% - {tol}pp tolerance")

    md = ["| Subsystem | Files | Line | Baseline | Branch | Baseline |",
          "| :-- | --: | --: | --: | --: | --: |"]
    for sub, nfiles, line, lbase, branch, bbase in rows:
        md.append(f"| `{sub}` | {nfiles} | {fmt(line)} | {lbase}% | {fmt(branch)} | {bbase}% |")
    md.append(f"| **overall** | | **{fmt(o_line)}** | {o_base.get('line')}% "
              f"| **{fmt(o_branch)}** | {o_base.get('branch')}% |")
    summary = "\n".join(md)
    print(summary)
    if args.summary_out:
        with open(args.summary_out, "a", encoding="utf-8") as fh:
            fh.write("## Coverage gate\n\n" + summary + "\n")

    if failures:
        print("\ncoverage_gate: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\ncoverage_gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
