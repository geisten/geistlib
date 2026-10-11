#!/usr/bin/env python3
"""bench_matrix.py — plan and publish the scheduled cross-engine benchmarks.

    plan [--set nightly|weekly] [--only COLUMN[/MODEL]]
        Print the job matrix (JSON list) for .gitea/workflows/bench-matrix.yml
        from benchmark/bench_matrix.json; weekly includes the nightly models.

    include
        Print the literal matrix include for bench-matrix.yml (Gitea cannot
        expand a matrix from another job's output); tests check it is current.

    snapshot DIR TAG
        Freeze the newest run of every cell under DIR/runs as
        DIR/releases/TAG.json (called by the release workflow).

    publish DIR
        DIR mirrors https://geisten.net/bench/: runs/<column>/<model>/*.jsonl
        (+ .md reports) from bench_cross_engine.py. Writes DIR/index.html and
        DIR/history.json (plus a .txt copy of each report, which httpd
        displays instead of downloading) and prints every cell whose newest run regressed
        against the run before it; exit 1 if any did.

A regression needs both: geist's own median throughput (pp512 or tg64 at
depth 512) fell by more than THRESHOLD, and its ratio to llama.cpp fell too.
A host that was slower that night slows both engines and moves only the
first; a geist change moves both. Runs against a different llama.cpp build
are not compared.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "benchmark" / "bench_matrix.json"
THRESHOLD = 5.0  # percent; two protocol runs on one host drift ~2-4 %
SEQ = "512"
SPARK = 30  # runs in a cell's trend line


def plan(which: str, only: str | None) -> list[dict]:
    matrix = json.loads(MATRIX.read_text())
    jobs = []
    for column, spec in matrix["columns"].items():
        models = spec["nightly"] + (spec["weekly"] if which == "weekly" else [])
        for model in models:
            if only and only not in (column, f"{column}/{model}"):
                continue
            llama = matrix["llama_overrides"].get(model, {})
            jobs.append({"column": column, "model": model, "runner": spec["runner"],
                         "protocol": spec["protocol"], "build": spec["build"],
                         "llama": spec["llama"], "prefix": spec["prefix"],
                         "llama_repo": llama.get("repo", ""), "llama_pin": llama.get("pin", "")})
    return jobs


def include() -> str:
    nightly = {(j["column"], j["model"]) for j in plan("nightly", None)}
    return "\n".join(
        f'          - {{column: {j["column"]}, model: {j["model"]}, runner: {j["runner"]}, '
        f'weekly: {str((j["column"], j["model"]) not in nightly).lower()}}}'
        for j in plan("weekly", None))


def load_run(path: Path) -> dict | None:
    meta = summary = None
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("kind") == "metadata":
            meta = record
        elif record.get("kind") == "summary":
            summary = record["summary"]
    if not meta or not summary or SEQ not in summary.get("geist", {}):
        return None  # an interrupted run has no summary
    geist = summary["geist"][SEQ]
    llama = next(e for e in meta["engines"] if e["label"] == "llama.cpp")
    return {
        "file": path.name,
        "report": path.with_suffix(".txt").name if path.with_suffix(".txt").exists() else None,
        "date": meta["created_utc"][:16].replace("T", " "),
        "commit": meta["runner"]["commit"][:9],
        "llama": llama.get("commit") or llama.get("binary_sha256", "")[:12],
        "pp": geist["prefill"]["median_tps"], "pp_vs": geist["prefill"]["vs_llama_pct"],
        "tg": geist["decode"]["median_tps"], "tg_vs": geist["decode"]["vs_llama_pct"],
    }


def regressions(runs: list[dict]) -> list[str]:
    if len(runs) < 2 or runs[-1]["llama"] != runs[-2]["llama"]:
        return []
    old, new = runs[-2], runs[-1]
    found = []
    for key, name in (("pp", f"pp{SEQ}"), ("tg", f"tg64@{SEQ}")):
        drop = (old[key] - new[key]) / old[key] * 100
        if drop > THRESHOLD and new[f"{key}_vs"] < old[f"{key}_vs"]:
            found.append(f"{name} {old[key]:.1f} -> {new[key]:.1f} t/s (-{drop:.1f} %), "
                         f"vs llama.cpp {old[key + '_vs']:+.0f} -> {new[key + '_vs']:+.0f} % "
                         f"({old['commit']} -> {new['commit']})")
    return found


def spark(values: list[float]) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    pts = " ".join(f"{i * 64 / (len(values) - 1):.1f},{15 - (v - lo) / span * 13:.1f}"
                   for i, v in enumerate(values))
    return (f'<svg class="spark" width="64" height="16" viewBox="0 0 64 16" aria-hidden="true">'
            f'<polyline points="{pts}"/></svg>')


def cell(column: str, model: str, runs: list[dict], prefix: str = "") -> str:
    if not runs:
        return '<td class="na">—</td>'
    last, recent = runs[-1], runs[-SPARK:]
    bad = " bad" if regressions(runs) else ""
    return (f'<td class="c{bad}"><a href="{prefix}runs/{column}/{model}/">'
            f'<span class="x">pp {last["pp_vs"]:+.0f} % · tg {last["tg_vs"]:+.0f} %</span></a>'
            f'<small>{last["pp"]:.0f} / {last["tg"]:.1f} tok/s</small>'
            f'<small>{spark([r["pp"] for r in recent])} {spark([r["tg"] for r in recent])}</small>'
            f'<small>{last["date"][:10]} · <code>{last["commit"]}</code></small></td>')


# Page chrome copied from the webapp repository (partials/header.html and
# footer.html, research link dropped); /css/site.css is the live site's.
HEADER = """<a class="skip-link" href="#main-content">Skip to content</a>
<header class="top">
  <div class="wrap">
    <a class="brand" href="/" aria-label="geisten home"><svg viewBox="125 110 262 315" aria-hidden="true"><path class="mark-shell" d="M256 125C320 125 372 175 372 238V293C372 348 330 395 256 410C182 395 140 348 140 293V238C140 175 192 125 256 125Z"/><path class="mark-inset" d="M256 150C306 150 347 190 347 240V289C347 331 313 365 256 378C199 365 165 331 165 289V240C165 190 206 150 256 150Z"/><path class="mark-seed" d="M256 188C284 188 306 211 306 238C306 271 281 299 256 325C231 299 206 271 206 238C206 211 228 188 256 188Z"/><path class="mark-anchor" d="M204 300C220 320 236 331 256 336C276 331 292 320 308 300M226 346H286"/></svg>geisten</a>
    <nav class="nav" aria-label="Main">
      <a href="/#services">Services</a>
      <a href="/#platform">Platform</a>
      <a href="/#evidence">Evidence</a>
      <a href="/about.html">About</a>
      <a class="cta" href="/#contact">Contact</a>
    </nav>
  </div>
</header>"""
FOOTER = """<footer class="site-footer">
  <div class="wrap">
    <span>© 2024–2026 geisten Holding UG (haftungsbeschränkt) · Kaarst, Germany</span>
    <span class="links"><a href="https://github.com/geisten" rel="noopener noreferrer">GitHub</a><a href="/impressum.html">Legal notice</a><a href="/datenschutz.html">Privacy</a></span>
  </div>
</footer>"""

STYLE = """<style>
.bench th, .bench td { vertical-align: top; }
.bench tr > :first-child { position: sticky; left: 0; background: var(--bg); padding-right: 1rem; }
.bench td.c small { display: block; color: var(--muted); font-size: 0.8rem; }
.bench td.c a { text-decoration: none; }
.bench td.na { color: var(--muted); }
.bench td.bad { box-shadow: inset 3px 0 0 #c4362b; }
.bench td.bad .x { color: #c4362b; }
.spark { vertical-align: middle; }
.spark polyline { fill: none; stroke: currentColor; stroke-width: 1.2; }
</style>"""


def page(title: str, description: str, label: str, heading: str, lede: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <title>{html.escape(title)} · geisten</title>
  <meta name="description" content="{html.escape(description)}"/>
  <link rel="icon" href="/logo/geisten-mark.svg" type="image/svg+xml"/>
  <link rel="stylesheet" href="/css/site.css"/>
  {STYLE}
</head>
<body>
{HEADER}
<main id="main-content">
  <div class="wrap page-head">
    <p class="label">{label}</p>
    <h1>{heading}</h1>
    <p class="lede">{lede}</p>
  </div>
  <section class="section">
    <div class="wrap figure bench">
{body}
    </div>
  </section>
</main>
{FOOTER}
</body>
</html>
"""


def archive_page(column: str, model: str, labels: dict, runs: list[dict]) -> str:
    rows = "\n".join(
        f'<tr><td>{r["date"]}</td><td><code>{r["commit"]}</code></td><td><code>{r["llama"][:9]}</code></td>'
        f'<td class="n">{r["pp"]:.1f}</td><td class="n x">{r["pp_vs"]:+.0f} %</td>'
        f'<td class="n">{r["tg"]:.1f}</td><td class="n x">{r["tg_vs"]:+.0f} %</td>'
        f'<td><a href="{r["report"] or r["file"]}">report</a> · <a href="{r["file"]}">raw</a></td></tr>'
        for r in reversed(runs))
    body = f"""      <table>
        <thead><tr><th>Run (UTC)</th><th>geist</th><th>llama.cpp</th><th class="n">pp{SEQ} tok/s</th><th class="n">vs llama</th><th class="n">tg64 tok/s</th><th class="n">vs llama</th><th>Files</th></tr></thead>
        <tbody>
{rows}
        </tbody>
      </table>
      <p class="note">Every run of this cell, newest first. pp{SEQ}: prefill of {SEQ} tokens; tg64: decode of 64 tokens at depth {SEQ}; median of 12 samples. The raw file holds every sample and the full host, engine and model provenance. <a href="../../../">Back to the matrix</a>.</p>"""
    name = labels["models"].get(model, model)
    host = labels["columns"].get(column, column)
    return page(f"{name} on {host}", f"Benchmark history of geistlib against llama.cpp: {name} on {host}.",
                "Benchmark archive", html.escape(name), f"{html.escape(host)} · {len(runs)} runs", body)


def matrix_table(columns: list[str], models: list[str], labels: dict, history: dict, prefix: str = "") -> str:
    rows = "\n".join(
        f'<tr><td>{html.escape(labels["models"].get(model, model))}</td>'
        + "".join(cell(c, model, history.get(f"{c}/{model}", []), prefix) for c in columns) + "</tr>"
        for model in models)
    head = "".join(f"<th>{html.escape(labels['columns'].get(c, c))}</th>" for c in columns)
    return f"""      <table>
        <thead><tr><th>Model</th>{head}</tr></thead>
        <tbody>
{rows}
        </tbody>
      </table>"""


def snapshot(root: Path, tag: str) -> None:
    """Freeze the newest run of every cell as releases/<tag>.json."""
    cells = {}
    for cell_dir in sorted((root / "runs").glob("*/*")):
        runs = [r for p in sorted(cell_dir.glob("*.jsonl")) if (r := load_run(p))]
        if runs:
            cells[f"{cell_dir.parent.name}/{cell_dir.name}"] = runs[-1]
    (root / "releases").mkdir(exist_ok=True)
    (root / "releases" / f"{tag}.json").write_text(json.dumps(
        {"tag": tag, "created_utc": datetime.now(timezone.utc).isoformat(), "cells": cells}, indent=1) + "\n")


def publish(root: Path) -> int:
    matrix = json.loads(MATRIX.read_text())
    labels = matrix["labels"]
    columns = list(matrix["columns"])
    models = list(dict.fromkeys(m for spec in matrix["columns"].values()
                                for m in spec["nightly"] + spec["weekly"]))
    # httpd serves .md as a download; a .txt copy opens in the browser
    for report in (root / "runs").glob("*/*/*.md"):
        if not report.with_suffix(".txt").exists():
            report.with_suffix(".txt").write_bytes(report.read_bytes())
    history, bad = {}, []
    for column in columns:
        for model in models:
            cell_dir = root / "runs" / column / model
            runs = [r for p in sorted(cell_dir.glob("*.jsonl")) if (r := load_run(p))]
            if runs:
                history[f"{column}/{model}"] = runs
                bad += [f"{column}/{model}: {msg}" for msg in regressions(runs)]
                (cell_dir / "index.html").write_text(archive_page(column, model, labels, runs))
    table = matrix_table(columns, models, labels, history)
    snaps = sorted((json.loads(p.read_text()) for p in (root / "releases").glob("*.json")),
                   key=lambda snap: snap["created_utc"], reverse=True)  # git checkouts share one mtime
    for snap in snaps:
        cells = {k: [v] for k, v in snap["cells"].items()}
        (root / "releases" / snap["tag"]).mkdir(exist_ok=True)
        (root / "releases" / snap["tag"] / "index.html").write_text(page(
            f"Benchmarks at {snap['tag']}", f"geistlib against llama.cpp as measured at release {snap['tag']}.",
            "Benchmarks at a release", f"geistlib {html.escape(snap['tag'])}",
            f"The matrix as it stood when {html.escape(snap['tag'])} was released ({snap['created_utc'][:10]}): "
            "the newest run of every cell at that time, with the geistlib commit it measured.",
            matrix_table(columns, models, labels, cells, prefix="../../")
            + '\n      <p class="note"><a href="../../">Back to the live matrix</a>.</p>'))
    release_links = " · ".join(f'<a href="releases/{s["tag"]}/">{html.escape(s["tag"])}</a>' for s in snaps)
    body = table + f"""
      <p class="note">Each cell: geistlib's speed relative to llama.cpp for prefill (pp{SEQ}) and decode (tg64 at depth {SEQ}), geistlib's own tokens/s, the trend over the last {SPARK} runs, and the date and geistlib commit of the newest run. Ratios compare within a column; tokens/s across columns describe the hardware. A red mark: geistlib slowed by more than {THRESHOLD:.0f} % against the cell's previous run while llama.cpp did not.</p>
      <p class="note">Method: the frozen cross-engine protocol of geistlib (same model file by SHA-256, same shapes and threads for both engines, 4 alternating cycles × 3 samples, median), run on a quiet host by the nightly and weekly <code>bench-matrix</code> workflow. Speed only, not a quality comparison. Click a cell for its archive of every run; all numbers as JSON: <a href="history.json">history.json</a>.</p>""" + (
        f'\n      <p class="note">The matrix at each release: {release_links}.</p>' if release_links else "")
    (root / "index.html").write_text(page(
        "Benchmark matrix", "Nightly geistlib against llama.cpp benchmarks per model and machine, with history.",
        "Benchmarks", "geistlib against llama.cpp",
        "Every model on every machine it fits, measured nightly under one frozen protocol. Each run is archived with its raw samples.",
        body))
    (root / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    for line in bad:
        print(f"REGRESSION {line}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--set", choices=("nightly", "weekly"), default="nightly")
    p.add_argument("--only")
    sub.add_parser("include")
    r = sub.add_parser("snapshot")
    r.add_argument("dir", type=Path)
    r.add_argument("tag")
    q = sub.add_parser("publish")
    q.add_argument("dir", type=Path)
    args = parser.parse_args()
    if args.cmd == "plan":
        print(json.dumps(plan(args.set, args.only)))
        return 0
    if args.cmd == "snapshot":
        snapshot(args.dir, args.tag)
        return 0
    if args.cmd == "include":
        print(include())
        return 0
    return publish(args.dir)


if __name__ == "__main__":
    sys.exit(main())
