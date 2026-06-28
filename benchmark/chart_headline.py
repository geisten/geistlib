#!/usr/bin/env python3
# Render headline_results.json into a horizontal "scoreboard" SVG: geist vs the
# baseline, GROUPED BY SYSTEM (OS / machine) so the different platforms stand out.
# Each row is its OWN metric + baseline, compared fairly as a ratio (geist /
# baseline). Pure stdlib — no matplotlib, no deps. Numbers come from the JSON.
#
#   python3 benchmark/chart_headline.py            # -> assets/headline_benchmarks.svg
#   python3 benchmark/chart_headline.py out.svg    # custom path
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "headline_results.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "assets", "headline_benchmarks.svg")

# Per-system presentation (display name, machine detail, accent, light tint).
SYSTEMS = {
    "Pi 5":   {"name": "Raspberry Pi 5", "detail": "ARM Cortex-A76 · 4 cores · CPU-only edge board",
               "accent": "#e11d48", "tint": "#fff5f6"},
    "M1 Max": {"name": "Apple M1 Max",   "detail": "Apple Silicon · CPU via Accelerate / AMX · desktop",
               "accent": "#7c3aed", "tint": "#f7f5ff"},
    "AMD 9950X": {"name": "AMD Ryzen 9 9950X", "detail": "Zen 5 · 16C/32T · AVX-512 · desktop",
               "accent": "#2563eb", "tint": "#eff6ff"},
}

W = 880
GX0, GX1 = 32, 848            # group block x-extent
LBL = 56                      # left label x
BX0, BX1 = 412, 820           # bar track x
RMAX = 2.4                    # ratio axis max
HEADER_H, ROW_H, GPAD, GGAP = 42, 56, 12, 16
TEAL, TEAL_LITE = "#0ea5a4", "#c7ede3"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def num(v):
    return f"{v:g}"


def xr(r):
    return BX0 + r * (BX1 - BX0) / RMAX


def main():
    data = json.load(open(DATA))
    # group rows by OS, sort rows within a group + groups by their best ratio
    groups = {}
    for r in data["rows"]:
        groups.setdefault(r["os"], []).append(r)
    for rows in groups.values():
        rows.sort(key=lambda r: r["geist"] / r["baseline"], reverse=True)
    order = sorted(groups, key=lambda o: max(r["geist"] / r["baseline"] for r in groups[o]), reverse=True)

    # layout pass: assign a y to each group
    y = 104
    blocks = []
    for o in order:
        rows = groups[o]
        gh = HEADER_H + len(rows) * ROW_H + GPAD
        blocks.append((o, rows, y, gh))
        y += gh + GGAP
    gtop, gbot = 104, y - GGAP
    H = gbot + 44

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">')
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" fill="#ffffff" stroke="#e5e7eb"/>')
    s.append('<text x="40" y="46" font-size="23" font-weight="700" fill="#0f172a">'
             'Headline benchmarks — geist vs the baseline</text>')
    s.append('<text x="40" y="72" font-size="13.5" fill="#475569">'
             'grouped by system · each bar = geist throughput &#247; the baseline engine, on its own '
             'metric · 1.0&#215; = parity</text>')

    # system group blocks (tint + accent rail + header)
    for o, rows, top, gh in blocks:
        sysm = SYSTEMS.get(o, {"name": o, "detail": "", "accent": "#475569", "tint": "#f8fafc"})
        s.append(f'<rect x="{GX0}" y="{top}" width="{GX1 - GX0}" height="{gh}" rx="12" '
                 f'fill="{sysm["tint"]}" stroke="#eef2f6"/>')
        s.append(f'<rect x="{GX0 + 3}" y="{top + 10}" width="5" height="{gh - 20}" rx="2.5" fill="{sysm["accent"]}"/>')
        # system pill + machine detail
        pill_w = 18 + len(sysm["name"]) * 8.3
        s.append(f'<rect x="{LBL}" y="{top + 12}" width="{pill_w:.0f}" height="23" rx="11.5" fill="{sysm["accent"]}"/>')
        s.append(f'<text x="{LBL + pill_w / 2:.0f}" y="{top + 28}" font-size="13" font-weight="700" '
                 f'fill="#ffffff" text-anchor="middle">{esc(sysm["name"])}</text>')
        s.append(f'<text x="{LBL + pill_w + 12:.0f}" y="{top + 28}" font-size="12" fill="#64748b">'
                 f'{esc(sysm["detail"])}</text>')

    # ratio gridlines + the 1.0x parity reference, spanning all blocks
    for r in (0.0, 1.0, 2.0):
        x = xr(r)
        dash = ' stroke-dasharray="4 4"' if r == 1.0 else ''
        col = "#cbd5e1" if r == 1.0 else "#e8edf2"
        s.append(f'<line x1="{x:.1f}" y1="{gtop}" x2="{x:.1f}" y2="{gbot}" stroke="{col}"{dash}/>')
        s.append(f'<text x="{x:.1f}" y="{gbot + 18}" font-size="11" fill="#94a3b8" '
                 f'text-anchor="middle">{num(r)}&#215;</text>')
    s.append(f'<text x="{xr(1.0):.1f}" y="{gtop - 5}" font-size="11" fill="#94a3b8" '
             f'text-anchor="middle">parity</text>')

    # rows: model · metric label, two-tone bar, ratio badge
    for o, rows, top, gh in blocks:
        for i, row in enumerate(rows):
            ratio = row["geist"] / row["baseline"]
            rtop = top + HEADER_H + i * ROW_H
            mid = rtop + ROW_H / 2 - 4
            s.append(f'<text x="{LBL}" y="{mid - 3:.1f}" font-size="13.5" font-weight="600" fill="#0f172a">'
                     f'{esc(row["model"])} <tspan fill="#94a3b8" font-weight="400">· {esc(row["metric"])}</tspan></text>')
            s.append(f'<text x="{LBL}" y="{mid + 14:.1f}" font-size="11" fill="#64748b">'
                     f'{num(row["geist"])} vs {num(row["baseline"])} t/s · {esc(row["baseline_engine"])}</text>')
            by = mid - 11
            # light = up to parity; solid teal = the margin past parity (none if sub-parity)
            s.append(f'<rect x="{BX0}" y="{by:.1f}" width="{xr(min(ratio, 1.0)) - BX0:.1f}" '
                     f'height="23" rx="3" fill="{TEAL_LITE}"/>')
            if ratio > 1.0:
                s.append(f'<rect x="{xr(1.0):.1f}" y="{by:.1f}" width="{xr(ratio) - xr(1.0):.1f}" '
                         f'height="23" rx="3" fill="{TEAL}"/>')
            disp = round(ratio, 1)
            badge = "#0e7490" if disp >= 1.0 else "#64748b"
            s.append(f'<text x="{xr(ratio) + 9:.1f}" y="{mid + 5:.1f}" font-size="15" font-weight="700" '
                     f'fill="{badge}">{disp:.1f}&#215;</text>')

    s.append(f'<text x="40" y="{H - 15}" font-size="11" fill="#94a3b8">'
             'Each row is that system&#8217;s headline metric (decode / prefill / total) vs its own '
             'baseline engine — comparable only as a ratio. Full sweep: benchmark/.</text>')
    s.append('</svg>')

    out = os.path.abspath(OUT)
    open(out, "w").write("\n".join(s) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
