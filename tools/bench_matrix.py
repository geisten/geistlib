#!/usr/bin/env python3
"""bench_matrix.py — plan and publish the scheduled cross-engine benchmarks.

    plan [--set nightly|weekly] [--only COLUMN[/MODEL]]
        Print the job matrix (JSON list) for .gitea/workflows/bench-matrix.yml
        from benchmark/bench_matrix.json; weekly includes the nightly models.

    include
        Print the literal matrix include for bench-matrix.yml (Gitea cannot
        expand a matrix from another job's output); tests check it is current.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "benchmark" / "bench_matrix.json"
THRESHOLD = 5.0  # percent; two protocol runs on one host drift ~2-4 %
SEQ = "512"
HISTORY = 30


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
    pts = " ".join(f"{i * 60 / (len(values) - 1):.1f},{14 - (v - lo) / span * 12:.1f}"
                   for i, v in enumerate(values))
    return (f'<svg width="60" height="16" viewBox="0 0 60 16" aria-hidden="true">'
            f'<polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="1.2"/></svg>')


def cell(column: str, model: str, runs: list[dict]) -> str:
    if not runs:
        return "<td class=na>—</td>"
    last = runs[-1]
    href = f"runs/{column}/{model}/{last['report'] or last['file']}"
    bad = " bad" if regressions(runs) else ""
    return (f'<td class="c{bad}"><a href="{html.escape(href)}">'
            f'pp {last["pp_vs"]:+.0f} % · tg {last["tg_vs"]:+.0f} %</a>'
            f'<small>{last["pp"]:.0f} / {last["tg"]:.1f} t/s</small>'
            f'<span class=s>{spark([r["pp"] for r in runs])}{spark([r["tg"] for r in runs])}</span>'
            f'<small>{last["date"]} · {last["commit"]}</small></td>')


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>geist benchmarks</title>
<style>
:root {{ --bg:#fff; --fg:#1b1b1b; --mute:#666; --line:#ddd; --bad:#b3261e; --badbg:#fdecea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#e8e8e8; --mute:#999; --line:#333; --bad:#ff8a80; --badbg:#3a1d1b; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0 auto; padding:24px 16px; max-width:1200px; }}
a {{ color:inherit; }}
.wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; min-width:100%; }}
th, td {{ border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }}
td.c small, td.c .s {{ display:block; color:var(--mute); white-space:nowrap; }}
td.c a {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
td.bad {{ background:var(--badbg); }} td.bad a {{ color:var(--bad); font-weight:600; }}
td.na {{ color:var(--mute); text-align:center; }}
p {{ color:var(--mute); max-width:70ch; }}
</style></head><body>
<h1>geist vs llama.cpp</h1>
<p>Frozen cross-engine protocol (pp{seq}, tg64 at depth {seq}; median of 12 samples) per model and host,
measured on a quiet host by the nightly and weekly runs of geistlib's bench-matrix workflow.
A cell shows geist's speed relative to llama.cpp, geist's own tokens/s (prefill / decode) with their trend
over the last {n} runs, and the run's date and geist commit. Ratios compare within a column;
tokens/s across columns describe hardware. Red: geist slowed by more than {thr:.0f} % against the previous run
while llama.cpp did not. Each cell links its report; every run's numbers: <a href="history.json">history.json</a>.</p>
<div class="wrap"><table>
<tr><th>model</th>{head}</tr>
{rows}
</table></div>
</body></html>
"""


def publish(root: Path) -> int:
    matrix = json.loads(MATRIX.read_text())
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
            runs = [r for p in sorted((root / "runs" / column / model).glob("*.jsonl"))
                    if (r := load_run(p))][-HISTORY:]
            if runs:
                history[f"{column}/{model}"] = runs
                bad += [f"{column}/{model}: {msg}" for msg in regressions(runs)]
    rows = "\n".join(
        f"<tr><th>{html.escape(model)}</th>"
        + "".join(cell(c, model, history.get(f"{c}/{model}", [])) for c in columns) + "</tr>"
        for model in models)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    (root / "index.html").write_text(PAGE.format(seq=SEQ, n=HISTORY, thr=THRESHOLD, head=head, rows=rows))
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
    q = sub.add_parser("publish")
    q.add_argument("dir", type=Path)
    args = parser.parse_args()
    if args.cmd == "plan":
        print(json.dumps(plan(args.set, args.only)))
        return 0
    if args.cmd == "include":
        print(include())
        return 0
    return publish(args.dir)


if __name__ == "__main__":
    sys.exit(main())
