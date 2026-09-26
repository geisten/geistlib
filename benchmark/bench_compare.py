#!/usr/bin/env python3
"""bench_compare.py — fail when a `make bench` run is slower than the last green one.

    python3 benchmark/bench_compare.py CURRENT_raw.json --baseline last-green.json

Both files are the raw JSON `make bench` writes (tools/bench_reproduce.py).
Every seq_len row is compared on prefill and decode tok/s. The thresholds come
from the Pi 5's own day-to-day noise on an unchanged commit, read off six
consecutive nightly runs: decode moved under 1 %, prefill up to ~3 %. A drop
past 3 % / 5 % is therefore the code, not the weather. perf_gate.py stays the
cliff detector for noisy cloud runners; this one is for a quiet board.

A missing baseline, or one for another model, prints and passes — the caller
then promotes the current run, and the next night has something to compare.

Exit codes: 0 ok (or nothing to compare), 1 regression, 2 usage.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def compare(cur: dict, base: dict, max_drop: dict[str, float]) -> tuple[list[str], list[str]]:
    """Markdown table lines and a list of regressions (empty when none)."""
    base_rows = {r["seq_len"]: r for r in base["rows"]}
    lines = [f"| seq | metric | last green ({base['commit']}) | now ({cur['commit']}) | Δ |",
             "| --: | :-- | --: | --: | --: |"]
    bad = []
    for r in cur["rows"]:
        b = base_rows.get(r["seq_len"])
        if b is None:
            continue
        for metric in ("prefill_tps", "decode_tps"):
            was, now = float(b[metric]), float(r[metric])
            delta = 100.0 * (now - was) / was if was else 0.0
            flag = " **REGRESSION**" if -delta > max_drop[metric] else ""
            lines.append(f"| {r['seq_len']} | {metric} | {was:.2f} | {now:.2f} | {delta:+.1f} %{flag} |")
            if flag:
                bad.append(f"seq {r['seq_len']} {metric}: {was:.2f} -> {now:.2f} ({delta:+.1f} %, "
                           f"limit -{max_drop[metric]:.0f} %)")
    return lines, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("current", help="raw JSON of the run just made")
    ap.add_argument("--baseline", required=True, help="raw JSON of the last green run")
    ap.add_argument("--max-drop-decode", type=float, default=3.0, help="percent")
    ap.add_argument("--max-drop-prefill", type=float, default=5.0, help="percent")
    args = ap.parse_args()

    try:
        cur = json.loads(Path(args.current).read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"bench_compare: cannot read {args.current}: {e}", file=sys.stderr)
        return 2
    try:
        base = json.loads(Path(args.baseline).read_text())
    except (OSError, json.JSONDecodeError):
        print(f"no baseline at {args.baseline} — nothing to compare, this run becomes it")
        return 0
    if base.get("model") != cur.get("model"):
        print(f"baseline is for {base.get('model')}, this run is {cur.get('model')} — not comparable")
        return 0

    lines, bad = compare(cur, base, {"decode_tps": args.max_drop_decode,
                                     "prefill_tps": args.max_drop_prefill})
    report = "\n".join(["## bench vs last green", ""] + lines)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n\n")
    if bad:
        print("PERF REGRESSION vs last green run:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print("bench_compare: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
