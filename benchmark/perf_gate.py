#!/usr/bin/env python3
"""perf_gate.py — read a bench_perf_sweep JSON line from stdin and exit non-zero
if prefill/decode tok/s fall below floors.

A COARSE cliff-detector for CI: shared cloud runners are noisy (±~20%), so the
floors are set well below the real number — this catches gross regressions
(scalar fallback, -O0 sneaking in, OpenMP off, a kernel returning early), not
subtle ones. The Raspberry Pi 5 stays the precise arbiter (benchmark/results/PI5.md).

  bench_perf_sweep --gguf M --seq-lens 128 --decode-n 16 ... | \
    python3 benchmark/perf_gate.py --min-prefill 20 --min-decode 4
  python3 benchmark/perf_gate.py --selftest

--record FILE keeps the floors honest about drift the floors cannot see: the
measured line goes to GITHUB_STEP_SUMMARY next to whatever FILE held from
the previous run (CI restores it from the last main run's cache), then FILE
is overwritten with this run's numbers for the next one. No verdict rides
on that comparison — a shared runner's spread is why the floors are coarse.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def parse(line: str) -> tuple[float, float]:
    d = json.loads(line)
    # bench_quality_perf.py wraps the measurement with commit/model/binary
    # provenance. Direct bench_perf_sweep JSON remains accepted for local and
    # Linux CI callers.
    if "measurement" in d:
        d = d["measurement"]
    return float(d["prefill_tps"]), float(d["decode_tps"])


def summary_row(pp: float, tg: float, sha: str, record: Path | None) -> str:
    """Markdown for the step summary: this run, and the recorded previous one."""
    prev = None
    if record and record.is_file():
        try:
            prev = json.loads(record.read_text())
        except (OSError, json.JSONDecodeError):
            prev = None
    lines = ["| perf gate | prefill tok/s | decode tok/s |", "| :-- | --: | --: |"]
    if prev:
        dp = 100.0 * (pp - prev["prefill_tps"]) / prev["prefill_tps"]
        dt = 100.0 * (tg - prev["decode_tps"]) / prev["decode_tps"]
        lines.append(f"| this run `{sha[:7]}` | {pp:.1f} ({dp:+.0f} %) | {tg:.1f} ({dt:+.0f} %) |")
        lines.append(f"| last main `{prev.get('sha', '')[:7]}` | {prev['prefill_tps']:.1f} "
                     f"| {prev['decode_tps']:.1f} |")
    else:
        lines.append(f"| this run `{sha[:7]}` | {pp:.1f} | {tg:.1f} |")
    if record:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"prefill_tps": pp, "decode_tps": tg, "sha": sha}) + "\n")
    return "\n".join(lines) + "\n"


def gate(line: str, min_prefill: float, min_decode: float) -> str | None:
    """Return an error string if below a floor, else None."""
    pp, tg = parse(line)
    print(f"perf: prefill={pp:.1f} tok/s (floor {min_prefill}), "
          f"decode={tg:.1f} tok/s (floor {min_decode})")
    bad = []
    if pp < min_prefill:
        bad.append(f"prefill {pp:.1f} < {min_prefill}")
    if tg < min_decode:
        bad.append(f"decode {tg:.1f} < {min_decode}")
    return "; ".join(bad) if bad else None


def _selftest() -> None:
    ok = '{"prefill_tps": 40.0, "decode_tps": 8.0}'
    assert gate(ok, 20, 4) is None
    assert gate('{"metadata": {}, "measurement": ' + ok + '}', 20, 4) is None
    assert gate('{"prefill_tps": 5.0, "decode_tps": 8.0}', 20, 4) is not None   # prefill cliff
    assert gate('{"prefill_tps": 40.0, "decode_tps": 1.0}', 20, 4) is not None  # decode cliff
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rec = Path(d) / "perf" / "last.json"
        first = summary_row(40.0, 8.0, "aaaaaaa1", rec)
        assert "last main" not in first and rec.is_file()
        second = summary_row(30.0, 8.8, "bbbbbbb2", rec)
        assert "| this run `bbbbbbb` | 30.0 (-25 %) | 8.8 (+10 %) |" in second, second
        assert "| last main `aaaaaaa` | 40.0 | 8.0 |" in second, second
        assert json.loads(rec.read_text())["sha"] == "bbbbbbb2"  # overwritten for the next run
    print("perf_gate selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prefill", type=float, default=0.0)
    ap.add_argument("--min-decode", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", help="JSON file with the previous run's numbers; rewritten with this run's")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()

    line = ""
    for ln in sys.stdin:
        ln = ln.strip()
        if ln.startswith("{"):
            line = ln  # last JSON line wins (one seq-len)
    if not line:
        sys.exit("perf_gate: no JSON line on stdin (did bench_perf_sweep run?)")
    err = gate(line, args.min_prefill, args.min_decode)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary or args.record:
        pp, tg = parse(line)
        md = summary_row(pp, tg, os.environ.get("GITHUB_SHA", ""), Path(args.record) if args.record else None)
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(md + "\n")
        else:
            print(md, end="")
    if err:
        sys.exit(f"PERF REGRESSION: {err}")
    print("perf gate: ok")


if __name__ == "__main__":
    main()
