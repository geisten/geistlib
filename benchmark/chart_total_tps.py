#!/usr/bin/env python3
# Render pi5_results.json into a clean grouped-bar SVG of TOTAL tok/s
# (geist vs llama.cpp), end-to-end, for the README. Pure stdlib — no matplotlib,
# no deps (matches the engine's no-dependency ethos). Numbers come straight from
# the JSON; this only draws them.
#
#   python3 benchmark/chart_total_tps.py            # -> assets/pi5_total_tps.svg
#   python3 benchmark/chart_total_tps.py out.svg    # custom path
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "pi5_results.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "assets", "pi5_total_tps.svg")

# engine -> bar colour. Order = bar order within each group.
ENGINES = [
    ("geist", "geist", "#0ea5a4"),
    ("llama.cpp CPU", "llama.cpp", "#64748b"),
    ("llama.cpp BLAS", "+ OpenBLAS", "#cbd5e1"),
]
# per-panel (group) x-axis captions + a one-line verdict, by index.
GROUPS_META = [
    ("short prompt", "32 prompt + 128 decode", "geist leads end-to-end", "#0ea5a4"),
    ("longer prompt", "128 prompt + 128 decode", "≈ tie (prefill weighs in)", "#64748b"),
]

# canvas + plot box
W, H = 820, 526
L, R, T, B = 80, 780, 120, 410
YMAX, YSTEP = 12, 2
BW, GAP = 54, 14  # bar width, intra-group gap


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ypix(v):
    return B - v * (B - T) / YMAX


def main():
    data = json.load(open(DATA))
    panels = data["panels"]
    n = len(panels)
    span = len(ENGINES) * BW + (len(ENGINES) - 1) * GAP

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">')
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" fill="#ffffff" stroke="#e5e7eb"/>')

    # title + subtitle
    s.append('<text x="40" y="46" font-size="23" font-weight="700" fill="#0f172a">'
             'Total throughput — geist vs llama.cpp</text>')
    s.append('<text x="40" y="72" font-size="13.5" fill="#475569">'
             'Raspberry Pi 5  ·  Gemma 4 E2B-it (Q4_K_M)  ·  4 threads  ·  '
             'end-to-end tokens/s, higher is better</text>')

    # legend (centred row)
    lx = 250
    for key, label, col in ENGINES:
        s.append(f'<rect x="{lx}" y="86" width="13" height="13" rx="2.5" fill="{col}"/>')
        s.append(f'<text x="{lx + 19}" y="97" font-size="13" fill="#334155">{esc(label)}</text>')
        lx += 30 + len(label) * 7.6

    # y gridlines + ticks
    for v in range(0, YMAX + 1, YSTEP):
        y = ypix(v)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" '
                 f'stroke="{"#cbd5e1" if v == 0 else "#eef2f6"}"/>')
        s.append(f'<text x="{L - 12}" y="{y + 4:.1f}" font-size="12" fill="#94a3b8" '
                 f'text-anchor="end">{v}</text>')
    s.append(f'<text transform="translate(28,{(T + B) / 2:.0f}) rotate(-90)" font-size="13" '
             f'fill="#64748b" text-anchor="middle">total tok/s</text>')

    # bars, per group
    for i, panel in enumerate(panels):
        cx = L + (R - L) * (i + 0.5) / n
        x0 = cx - span / 2
        for j, (key, _, col) in enumerate(ENGINES):
            v = panel["series"][key]["total"]
            x = x0 + j * (BW + GAP)
            y = ypix(v)
            s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{BW}" height="{B - y:.1f}" '
                     f'rx="3" fill="{col}"/>')
            bold = ' font-weight="700"' if j == 0 else ''
            lblcol = "#0e7490" if j == 0 else "#475569"
            s.append(f'<text x="{x + BW / 2:.1f}" y="{y - 8:.1f}" font-size="13.5"{bold} '
                     f'fill="{lblcol}" text-anchor="middle">{v:.1f}</text>')
        cap, sub, verdict, vcol = GROUPS_META[i] if i < len(GROUPS_META) else (panel.get("label", ""), "", "", "#64748b")
        s.append(f'<text x="{cx:.1f}" y="{B + 26:.0f}" font-size="13.5" font-weight="600" '
                 f'fill="#334155" text-anchor="middle">{esc(cap)}</text>')
        s.append(f'<text x="{cx:.1f}" y="{B + 44:.0f}" font-size="11.5" fill="#94a3b8" '
                 f'text-anchor="middle">{esc(sub)}</text>')
        s.append(f'<text x="{cx:.1f}" y="{B + 64:.0f}" font-size="12" font-weight="600" '
                 f'fill="{vcol}" text-anchor="middle">{esc(verdict)}</text>')

    # footnote
    s.append(f'<text x="40" y="{H - 16}" font-size="11" fill="#94a3b8">'
             'total = (P+D) / (P/prefill + D/decode), each engine cold-started under a '
             '≤50 °C gate · benchmark/total_tps.py</text>')
    s.append('</svg>')

    out = os.path.abspath(OUT)
    open(out, "w").write("\n".join(s) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
