#!/usr/bin/env python3
"""perf_ratio_gate.py — gate geist on its RATIO to llama.cpp measured in the same run.

  python3 benchmark/perf_ratio_gate.py --geist geist.jsonl --llama llama.json \
      --min-prefill-ratio 0.6 --min-decode-ratio 0.5

An absolute tok/s floor cannot survive a hosted macOS runner: the Apple guard
saw 23.9, 14.4 and 38.0 prefill tok/s on three consecutive Sundays from the
same code, against a floor of 60. Whatever that runner is on a given day, it
is the same for both engines in one job, so the ratio cancels most of it
(benchmark/METHODOLOGY.md: "the ratio on your own box is the number that
travels"). The floors stay coarse on purpose: a cliff — scalar fallback, -O0,
OpenMP off — halves or worse; the floors sit below drift and above cliffs.

Inputs: geist is the bench_perf_sweep JSON line (optionally wrapped by
bench_quality_perf.py under "measurement"); llama is `llama-bench -o json`.
Exit 0 ok, 1 below a floor, 2 bad input.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def geist_tps(path: Path) -> tuple[float, float]:
    line = ""
    for ln in path.read_text().splitlines():
        if ln.strip().startswith("{"):
            line = ln.strip()  # last JSON line wins
    if not line:
        sys.exit(f"perf_ratio_gate: no JSON line in {path}")
    d = json.loads(line)
    d = d.get("measurement", d)
    return float(d["prefill_tps"]), float(d["decode_tps"])


def llama_tps(path: Path) -> tuple[float, float]:
    rows = json.loads(path.read_text())
    pp = [r["avg_ts"] for r in rows if r.get("n_gen", 0) == 0]
    tg = [r["avg_ts"] for r in rows if r.get("n_prompt", 0) == 0]
    if not pp or not tg:
        sys.exit(f"perf_ratio_gate: {path} lacks a prompt-only or gen-only row")
    return float(pp[-1]), float(tg[-1])


def report(g: tuple[float, float], l: tuple[float, float]) -> tuple[str, float, float]:
    rp, rd = g[0] / l[0], g[1] / l[1]
    md = ("| tok/s | prefill | decode |\n| :-- | --: | --: |\n"
          f"| geist | {g[0]:.1f} | {g[1]:.1f} |\n| llama.cpp | {l[0]:.1f} | {l[1]:.1f} |\n"
          f"| **ratio** | **{rp:.2f}×** | **{rd:.2f}×** |\n")
    return md, rp, rd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geist", required=True, type=Path)
    ap.add_argument("--llama", required=True, type=Path)
    ap.add_argument("--min-prefill-ratio", type=float, default=0.6)
    ap.add_argument("--min-decode-ratio", type=float, default=0.5)
    args = ap.parse_args()

    md, rp, rd = report(geist_tps(args.geist), llama_tps(args.llama))
    print(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## geist vs llama.cpp, same runner, same model\n\n" + md + "\n")
    bad = []
    if rp < args.min_prefill_ratio:
        bad.append(f"prefill ratio {rp:.2f} < {args.min_prefill_ratio}")
    if rd < args.min_decode_ratio:
        bad.append(f"decode ratio {rd:.2f} < {args.min_decode_ratio}")
    if bad:
        print("PERF REGRESSION vs llama.cpp: " + "; ".join(bad), file=sys.stderr)
        return 1
    print("perf ratio gate: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
