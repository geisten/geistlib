#!/usr/bin/env python3
# Render headline_results.json into a horizontal "scoreboard" SVG: geist vs the
# baseline across model/OS, each row its OWN metric + baseline, compared fairly
# as a ratio (geist / baseline). Pure stdlib — no matplotlib, no deps. Numbers
# come straight from the JSON; this only draws them.
#
#   python3 benchmark/chart_headline.py            # -> assets/headline_benchmarks.svg
#   python3 benchmark/chart_headline.py out.svg    # custom path
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "headline_results.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "assets", "headline_benchmarks.svg")

W = 860
HEAD, ROW, FOOT = 110, 66, 44
BX0, BX1 = 350, 800     # bar track (x)
RMAX = 2.4              # ratio axis max
TEAL, TEAL_LITE, GRID = "#0ea5a4", "#c7ede3", "#e5e7eb"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def xr(r):
    return BX0 + r * (BX1 - BX0) / RMAX


def main():
    data = json.load(open(DATA))
    rows = sorted(data["rows"], key=lambda r: r["geist"] / r["baseline"], reverse=True)
    H = HEAD + len(rows) * ROW + FOOT
    plot_top, plot_bot = HEAD - 8, HEAD + len(rows) * ROW

    def g(v):  # tidy number: 144 not 144.0
        return f"{v:g}"

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">')
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" fill="#ffffff" stroke="{GRID}"/>')
    s.append('<text x="40" y="46" font-size="23" font-weight="700" fill="#0f172a">'
             'Headline benchmarks — geist vs the baseline</text>')
    s.append('<text x="40" y="72" font-size="13.5" fill="#475569">'
             'each bar = geist throughput &#247; the baseline engine, on its own metric  ·  '
             '1.0&#215; = parity, longer = faster</text>')

    # x-axis ticks (ratio) + the 1.0x parity reference line
    for r in (0.0, 1.0, 2.0):
        x = xr(r)
        dashed = ' stroke-dasharray="4 4"' if r == 1.0 else ''
        col = "#94a3b8" if r == 1.0 else GRID
        s.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bot}" stroke="{col}"{dashed}/>')
        s.append(f'<text x="{x:.1f}" y="{plot_bot + 18}" font-size="11" fill="#94a3b8" '
                 f'text-anchor="middle">{g(r)}&#215;</text>')
    s.append(f'<text x="{xr(1.0):.1f}" y="{plot_top - 6}" font-size="11" fill="#94a3b8" '
             f'text-anchor="middle">parity</text>')

    for i, row in enumerate(rows):
        ratio = row["geist"] / row["baseline"]
        top = HEAD + i * ROW
        mid = top + ROW / 2
        by = mid - 12
        # left label: model (bold) + "os · metric · G vs B t/s (engine)"
        s.append(f'<text x="40" y="{mid - 3:.1f}" font-size="14" font-weight="600" fill="#0f172a">'
                 f'{esc(row["model"])} <tspan fill="#94a3b8" font-weight="400">· {esc(row["quant"])}</tspan></text>')
        s.append(f'<text x="40" y="{mid + 14:.1f}" font-size="11.5" fill="#64748b">'
                 f'{esc(row["os"])} · {esc(row["metric"])} · {g(row["geist"])} vs {g(row["baseline"])} t/s '
                 f'({esc(row["baseline_engine"])})</text>')
        # two-tone bar: 0..parity light, parity..ratio solid (the win)
        s.append(f'<rect x="{BX0}" y="{by:.1f}" width="{xr(1.0) - BX0:.1f}" height="24" rx="3" fill="{TEAL_LITE}"/>')
        s.append(f'<rect x="{xr(1.0):.1f}" y="{by:.1f}" width="{xr(ratio) - xr(1.0):.1f}" height="24" rx="3" fill="{TEAL}"/>')
        s.append(f'<text x="{xr(ratio) + 9:.1f}" y="{mid + 5:.1f}" font-size="15" font-weight="700" '
                 f'fill="#0e7490">{ratio:.1f}&#215;</text>')

    s.append(f'<text x="40" y="{H - 16}" font-size="11" fill="#94a3b8">'
             'Each row is that model/OS&#8217;s headline metric (decode / prefill / total) vs its own '
             'baseline engine — comparable only as a ratio. Full sweep: benchmark/.</text>')
    s.append('</svg>')

    out = os.path.abspath(OUT)
    open(out, "w").write("\n".join(s) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
