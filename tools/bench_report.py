#!/usr/bin/env python3
"""bench_report.py — render benchmark result JSON to a chart (matplotlib).

Decouples measuring from drawing: the engines emit JSON (stdlib-only, runs on a
bare Pi), this renders it on a dev box. Grouped bars, one subplot per panel; a
metric value may carry an error bar (CI / best-of-N spread).

Input JSON (one or more files concatenate into panels):
    {"title": "...",
     "panels": [
       {"label": "128-tok prompt + 128 decode",
        "series": {"geist":     {"prefill": 33.5, "decode": 6.7, "total": 11.1},
                   "llama.cpp":  {"prefill": 38.6, "decode": {"value": 6.7, "err": 0.2}}}}
     ]}
A metric value is a number, or {"value": x, "err": e}, or {"value": x, "lo": l, "hi": h}.

Needs matplotlib (a dev-box dep, deliberately not required by the measuring
scripts). Usage:
    python3 tools/bench_report.py results.json -o chart.svg   # .svg or .png
"""
from __future__ import annotations

import argparse
import json
import sys

PALETTE = ["#378ADD", "#1D9E75", "#D85A30", "#9B59B6", "#E0A800"]


def _val_err(v) -> tuple[float, tuple[float, float] | None]:
    """A metric -> (value, (lo_delta, hi_delta) | None)."""
    if isinstance(v, dict):
        x = float(v["value"])
        if "err" in v:
            return x, (float(v["err"]), float(v["err"]))
        if "lo" in v or "hi" in v:
            return x, (x - float(v.get("lo", x)), float(v.get("hi", x)) - x)
        return x, None
    return float(v), None


def render(doc: dict, out, fmt: str | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"   # keep text selectable / SVG small

    panels = doc["panels"]
    metrics: list[str] = []
    for p in panels:                        # union, first-seen order
        for s in p["series"].values():
            for m in s:
                if m not in metrics:
                    metrics.append(m)
    colour = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(metrics)}

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.3 * len(panels)),
                             squeeze=False)
    for ax, p in zip(axes[:, 0], panels):
        series = list(p["series"].items())
        n, k = len(series), len(metrics)
        width = 0.8 / k
        for j, m in enumerate(metrics):
            xs, ys, lo, hi, has_err = [], [], [], [], False
            for i, (_name, vals) in enumerate(series):
                x, err = _val_err(vals[m]) if m in vals else (0.0, None)
                xs.append(i + (j - (k - 1) / 2) * width)
                ys.append(x)
                lo.append(err[0] if err else 0.0)
                hi.append(err[1] if err else 0.0)
                has_err = has_err or err is not None
            bars = ax.bar(xs, ys, width, label=m, color=colour[m],
                          yerr=[lo, hi] if has_err else None, capsize=3,
                          error_kw={"elinewidth": 1})
            ax.bar_label(bars, fmt="%g", fontsize=8, padding=2)
        ax.set_xticks(range(n))
        ax.set_xticklabels([s[0] for s in series])
        ax.set_title(p["label"], fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, ncol=k, frameon=False, loc="upper right")
        ax.margins(y=0.18)
    if doc.get("title"):
        fig.suptitle(doc["title"], fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, format=fmt)
    return len(panels)


def _selfcheck() -> None:
    import io
    buf = io.StringIO()
    doc = {"panels": [{"label": "t", "series": {
        "geist": {"x": 1.0, "y": {"value": 2.0, "err": 0.3}}}}]}
    render(doc, buf, fmt="svg")
    svg = buf.getvalue().lower()     # matplotlib lowercases hex colours
    assert svg.lstrip().startswith("<?xml") or "<svg" in svg
    assert "geist" in svg            # x tick label present
    assert PALETTE[0].lower() in svg and PALETTE[1].lower() in svg
    print("bench_report selfcheck ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="result JSON files ('-' = stdin)")
    ap.add_argument("-o", "--out", help="output chart (.svg/.png); default: stdout SVG")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selfcheck()
    if not args.inputs:
        ap.error("need at least one input JSON (or --selftest)")

    title, panels = None, []
    for f in args.inputs:
        d = json.load(sys.stdin if f == "-" else open(f))
        title = title or d.get("title")
        panels += d["panels"]
    doc = {"title": title, "panels": panels}
    if args.out:
        n = render(doc, args.out)
        print(f"wrote {args.out} ({n} panel(s))")
    else:
        render(doc, sys.stdout, fmt="svg")  # SVG is text; png would need a file


if __name__ == "__main__":
    main()
