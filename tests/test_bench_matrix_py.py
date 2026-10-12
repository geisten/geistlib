#!/usr/bin/env python3
"""tests/test_bench_matrix_py.py — the scheduled benchmark matrix.

plan: every listed model exists in its column's protocol, weekly is a
superset of nightly, --only narrows, and the workflow's literal matrix
matches the JSON. publish: a geist slowdown that
llama.cpp does not share is a regression; a host-wide slowdown (both engines
slower, ratio unchanged) and a llama.cpp pin change are not. Hermetic.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import bench_matrix as bm  # noqa: E402

fails = 0


def expect(cond: bool, label: str) -> None:
    global fails
    print(f"  {'✓' if cond else '✗'} {label}")
    fails += not cond


nightly, weekly = bm.plan("nightly", None), bm.plan("weekly", None)
protocols = {p: json.loads((ROOT / "benchmark" / f"cross_engine_{p}_protocol.json").read_text())
             for p in ("cpu", "gpu")}
expect(all(j["model"] in protocols[j["protocol"]]["models"]
           and j["column"] in protocols[j["protocol"]]["host_profiles"] for j in weekly),
       "every planned model and column exists in its protocol")
expect({(j["column"], j["model"]) for j in nightly} < {(j["column"], j["model"]) for j in weekly},
       "weekly is a strict superset of nightly")
labels = json.loads(bm.MATRIX.read_text())["labels"]
expect(all(j["column"] in labels["columns"] and j["model"] in labels["models"] for j in weekly),
       "every column and model has a display label")
expect(len({(j["column"], j["model"]) for j in weekly}) == len(weekly), "no duplicate cells")
workflow = (ROOT / ".gitea" / "workflows" / "bench-matrix.yml").read_text()
expect(bm.include() in workflow, "bench-matrix.yml carries the current include (tools/bench_matrix.py include)")
expect([j["model"] for j in bm.plan("weekly", "pi5/qwen3-0.6b-q8")] == ["qwen3-0.6b-q8"], "--only narrows")


def run(path: Path, day: int, pp: float, pp_vs: float, tg: float, tg_vs: float, llama: str = "L1") -> None:
    meta = {"kind": "metadata", "created_utc": f"2026-10-{day:02d}T01:00:00+00:00",
            "runner": {"commit": f"c{day:08d}"},
            "engines": [{"label": "geist"}, {"label": "llama.cpp", "commit": llama}]}
    side = lambda tps, vs: {"median_tps": tps, "vs_llama_pct": vs}  # noqa: E731
    summary = {"kind": "summary", "summary": {"geist": {bm.SEQ: {
        "prefill": side(pp, pp_vs), "decode": side(tg, tg_vs)}}}}
    path.write_text(json.dumps(meta) + "\n" + json.dumps(summary) + "\n")


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    cells = {"amd_9950x/qwen35-0.8b-q8": [(1, 1000, 20, 100, 10), (2, 900, 8, 100, 10)],   # geist only
             "pi5/qwen3-0.6b-q8": [(1, 100, 10, 20, 5), (2, 90, 10, 18, 5)],               # host noise
             "apple_m1max/gemma4-e2b-q4km": [(1, 100, 10, 20, 5)]}
    for key, runs in cells.items():
        d = root / "runs" / key
        d.mkdir(parents=True)
        for day, *v in runs:
            run(d / f"2026-10-{day:02d}.jsonl", day, *v)
    d = root / "runs" / "nvidia_2080ti_vulkan/gemma4-e2b-q4km"
    d.mkdir(parents=True)
    run(d / "a.jsonl", 1, 100, 10, 20, 5)
    run(d / "b.jsonl", 2, 80, 0, 20, 5, llama="L2")                                       # new pin
    (d / "c.jsonl").write_text(json.dumps({"kind": "metadata"}) + "\n")                    # interrupted

    import contextlib, io  # noqa: E401
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = bm.publish(root)
    lines = out.getvalue().splitlines()
    expect(rc == 1 and len(lines) == 1 and "amd_9950x/qwen35-0.8b-q8: pp512" in lines[0],
           "only the geist-only slowdown is a regression")
    page = (root / "index.html").read_text()
    expect('class="c bad"' in page and page.count("<svg") >= 2, "index marks the regressed cell, draws trends")
    expect('href="/css/site.css"' in page, "index uses the site's stylesheet")
    archive = (root / "runs" / "amd_9950x/qwen35-0.8b-q8" / "index.html").read_text()
    expect(archive.count("<tr><td>2026-10-") == 2, "each cell gets an archive page with every run")
    bm.snapshot(root, "v9.9.9")
    with contextlib.redirect_stdout(io.StringIO()):
        bm.publish(root)
    snap = json.loads((root / "releases" / "v9.9.9.json").read_text())
    expect(snap["cells"]["amd_9950x/qwen35-0.8b-q8"]["pp"] == 900, "a release snapshot freezes each cell's newest run")
    expect('href="releases/v9.9.9/"' in (root / "index.html").read_text()
           and (root / "releases" / "v9.9.9" / "index.html").exists(), "the matrix links the release page")
    history = json.loads((root / "history.json").read_text())
    expect(len(history["nvidia_2080ti_vulkan/gemma4-e2b-q4km"]) == 2, "an interrupted run is skipped")

print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
