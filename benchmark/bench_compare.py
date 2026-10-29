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

A run is judged only if the board was quiet while it ran. The measurement
itself says so: bench_perf_sweep's best/worst spread per row is 0.4-1.5 % on
an idle Pi 5 and ~15 % when another job shares the board (#446). Past
--max-spread the run is reported as not comparable: exit 3, so the workflow neither fails
on it nor promotes it to the new baseline.

A slower host is not a slower geist either. `make bench` measures a
reference engine (bitnet.cpp) on the same board in the same run; when both
runs carry the same engine at the same version, a row is a regression only
if geist dropped past the limit both in tok/s and relative to that engine.
Nightly runs 1905-2182 (2026-10-23..28) read geist decode 3-9 % under the
last green run while bitnet.cpp dropped 2-8 % with them and the ratio held at
1.90-2.01x; an A/B of the two commits outside CI showed no difference.

Exit codes: 0 ok (or nothing to compare), 1 regression, 2 usage,
3 not comparable (contended run).
"""
import argparse
import json
import os
import sys
from pathlib import Path


def reference_shift(cur: dict, base: dict) -> tuple[str, dict[str, float]] | None:
    """The reference engine both runs measured, and how far it moved (now/then
    per metric); None when the runs share no engine at one version."""
    then = {(b.get("engine"), b.get("version")): b for b in base.get("baselines") or []}
    for b in cur.get("baselines") or []:
        old = then.get((b.get("engine"), b.get("version")))
        if old is None or b.get("engine") is None:
            continue
        try:
            shift = {m: float(b[m]) / float(old[m]) for m in ("prefill_tps", "decode_tps")}
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if all(v > 0 for v in shift.values()):
            return b["engine"], shift
    return None


def compare(cur: dict, base: dict, max_drop: dict[str, float],
            ref: tuple[str, dict[str, float]] | None = None) -> tuple[list[str], list[str]]:
    """Markdown table lines and a list of regressions (empty when none). With
    ref (engine, now/then per metric) a drop counts only if geist also lost
    ground against that engine."""
    base_rows = {r["seq_len"]: r for r in base["rows"]}
    head = f"| seq | metric | last green ({base['commit']}) | now ({cur['commit']}) | Δ |"
    rule = "| --: | :-- | --: | --: | --: |"
    if ref is not None:
        head += f" Δ vs {ref[0]} |"
        rule += " --: |"
    lines, bad = [head, rule], []
    for r in cur["rows"]:
        b = base_rows.get(r["seq_len"])
        if b is None:
            continue
        for metric in ("prefill_tps", "decode_tps"):
            was, now = float(b[metric]), float(r[metric])
            delta = 100.0 * (now - was) / was if was else 0.0
            rel = None
            if ref is not None and was:
                rel = 100.0 * (now / ref[1][metric] - was) / was
            slower = -delta > max_drop[metric] and (rel is None or -rel > max_drop[metric])
            flag = " **REGRESSION**" if slower else ""
            line = f"| {r['seq_len']} | {metric} | {was:.2f} | {now:.2f} | {delta:+.1f} %{flag} |"
            if rel is not None:
                line += f" {rel:+.1f} % |"
            lines.append(line)
            if slower:
                bad.append(f"seq {r['seq_len']} {metric}: {was:.2f} -> {now:.2f} ({delta:+.1f} %, "
                           f"limit -{max_drop[metric]:.0f} %"
                           + (f"; {rel:+.1f} % against {ref[0]}" if rel is not None else "") + ")")
    return lines, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("current", help="raw JSON of the run just made")
    ap.add_argument("--baseline", required=True, help="raw JSON of the last green run")
    ap.add_argument("--max-drop-decode", type=float, default=3.0, help="percent")
    ap.add_argument("--max-drop-prefill", type=float, default=5.0, help="percent")
    ap.add_argument("--max-spread", type=float, default=5.0,
                    help="percent; a row spread above this makes the run not comparable")
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
    noisy = [f"seq {r['seq_len']}: ±{float(r.get('spread_pct', 0.0)):.1f} %"
             for r in cur["rows"] if float(r.get("spread_pct", 0.0)) > args.max_spread]
    if noisy:
        msg = (f"bench_compare: run not comparable — spread above {args.max_spread:.0f} % "
               f"says the board was busy ({'; '.join(noisy)}); not judged, not promoted")
        print(msg)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(f"## bench vs last green\n\n{msg}\n\n")
        return 3

    ref = reference_shift(cur, base)
    lines, bad = compare(cur, base, {"decode_tps": args.max_drop_decode,
                                     "prefill_tps": args.max_drop_prefill}, ref)
    note = []
    if ref is not None:
        note = ["", f"Reference engine {ref[0]} on this board, same version in both runs: "
                    f"decode {100.0 * (ref[1]['decode_tps'] - 1):+.1f} %, "
                    f"prefill {100.0 * (ref[1]['prefill_tps'] - 1):+.1f} %. "
                    "A row regresses only if geist also lost ground against it."]
    report = "\n".join(["## bench vs last green", ""] + lines + note)
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
