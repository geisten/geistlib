#!/usr/bin/env python3
"""bench_quality_perf.py — reproducible perf/quality benchmark harness for geist.

Driven by the `make bench-small / bench-detailed / bench-quality-* /
bench-compare-ref` targets. The goal is *reproducibility*: every recorded
number is tagged with the host, OS, target, mode, thread count, and model so
results from different machines never silently overwrite each other.

Perf suites (`small`, `detailed`) are fully implemented: they run the C
`bench_session_throughput` binary against a GGUF and record prefill/decode
tok/s into benchmark/BENCHMARK.md, keeping the best run per (model, host, os, target,
mode, threads) key.

Quality suites (`quality-small`, `quality-detailed`) and `compare-ref` require
a reference toolchain (HF tokenizer + datasets, and/or a llama.cpp build) that
is out of scope for a hermetic `make` invocation. They print setup guidance and
exit cleanly rather than failing the build. See benchmark/BENCHMARKING.md.

Usage (normally invoked via the Makefile):
    python3 tools/bench_quality_perf.py --suite small \\
        --target mac-omp --mode release \\
        --bin-dir bin/mac-omp/release/tests --out-dir ~/bench-geistlib/quality_perf \\
        --benchmark-md benchmark/BENCHMARK.md --record

Environment:
    BENCH_GGUF      Path to the model GGUF (falls back to GEIST_GGUF_PATH).
    BENCH_THREADS   OMP thread count (sets OMP_NUM_THREADS for the child).
    BENCH_REF_GGUF  Reference GGUF for compare-ref (quality suites).
    BENCH_REF_BIN   Reference binary (e.g. llama-bench) for compare-ref.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PERF_SUITES = {"small", "detailed"}
QUALITY_SUITES = {"quality-small", "quality-detailed", "compare-ref"}

# tok/s lines from bench_session_throughput, e.g.
#   prefill (200 tok):    2560.6 ms  =  12.80 ms/tok  =    78.1 tok/s
TPS_RE = re.compile(r"^\s*(prefill|decode)\b.*?=\s*([\d.]+)\s*tok/s", re.MULTILINE)


def host_id() -> str:
    """Stable-ish host label so different machines don't clobber each other."""
    return f"{platform.node()}/{platform.machine()}"


def os_id() -> str:
    return f"{platform.system()} {platform.release()}"


def resolve_gguf() -> str | None:
    g = os.environ.get("BENCH_GGUF") or os.environ.get("GEIST_GGUF_PATH")
    if g and Path(g).is_file():
        return g
    return None


def run_throughput(bin_dir: Path, gguf: str, threads: str | None) -> dict[str, float]:
    """Run bench_session_throughput once; return {'prefill': tps, 'decode': tps}."""
    exe = bin_dir / "bench_session_throughput"
    if not exe.is_file():
        sys.exit(f"bench: missing {exe} — run `make bench` to build the bench binaries first")

    env = dict(os.environ)
    env["GEIST_GGUF_PATH"] = gguf
    env.setdefault("OMP_WAIT_POLICY", "active")
    if threads:
        env["OMP_NUM_THREADS"] = threads

    proc = subprocess.run([str(exe), gguf], env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode not in (0, None):
        sys.stderr.write(out)
        sys.exit(f"bench: bench_session_throughput exited {proc.returncode}")

    found = {m.group(1): float(m.group(2)) for m in TPS_RE.finditer(out)}
    if "prefill" not in found or "decode" not in found:
        sys.stderr.write(out)
        sys.exit("bench: could not parse tok/s from bench_session_throughput output")
    return found


def perf_suite(args: argparse.Namespace) -> None:
    gguf = resolve_gguf()
    if gguf is None:
        print("bench: no model found (set BENCH_GGUF or GEIST_GGUF_PATH, or run "
              "`make fetch-model`). Skipping perf suite.")
        return

    threads = os.environ.get("BENCH_THREADS") or None
    # `detailed` averages more runs to shrink variance; `small` is a quick check.
    n_runs = 5 if args.suite == "detailed" else 2
    best = {"prefill": 0.0, "decode": 0.0}
    for i in range(n_runs):
        r = run_throughput(Path(args.bin_dir), gguf, threads)
        print(f"  run {i + 1}/{n_runs}: prefill {r['prefill']:.1f} tok/s, "
              f"decode {r['decode']:.1f} tok/s")
        best["prefill"] = max(best["prefill"], r["prefill"])
        best["decode"] = max(best["decode"], r["decode"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": Path(gguf).name,
        "host": host_id(),
        "os": os_id(),
        "target": args.target,
        "mode": args.mode,
        "threads": threads or "default",
        "prefill": f"{best['prefill']:.1f}",
        "decode": f"{best['decode']:.1f}",
    }
    print(f"\nbest: prefill {row['prefill']} tok/s | decode {row['decode']} tok/s "
          f"({row['host']}, {args.target}/{args.mode}, threads={row['threads']})")

    if args.record and args.benchmark_md:
        update_benchmark_md(Path(args.benchmark_md), row)
        print(f"recorded to {args.benchmark_md}")


# Marker block in benchmark/BENCHMARK.md that this script owns. Hand-written prose above
# the marker is preserved; only the auto-recorded table below it is rewritten.
MARKER = "<!-- BENCH:AUTO -->"

# Column layout of the auto-recorded table — one source of truth for both the
# header and every positional access below (no magic indices elsewhere).
COLUMNS = ["Date", "Model", "Host", "OS", "Target/Mode", "Threads",
           "Prefill tok/s", "Decode tok/s"]
(COL_DATE, COL_MODEL, COL_HOST, COL_OS, COL_TARGET_MODE,
 COL_THREADS, COL_PREFILL, COL_DECODE) = range(len(COLUMNS))
# Columns identifying a unique config (date + tok/s excluded).
KEY_COLS = (COL_MODEL, COL_HOST, COL_OS, COL_TARGET_MODE, COL_THREADS)

TABLE_HEADER = (
    "| " + " | ".join(COLUMNS) + " |\n"
    "| " + " | ".join(":---" if i < COL_THREADS else ":---:"
                      for i in range(len(COLUMNS))) + " |"
)


def _row_key(cells: list[str]) -> tuple:
    return tuple(cells[i] for i in KEY_COLS)


def update_benchmark_md(path: Path, row: dict) -> None:
    """Insert/replace this run's row, keeping the best decode tok/s per key."""
    new_cells = [row["date"], row["model"], row["host"], row["os"],
                 f"{row['target']}/{row['mode']}", row["threads"],
                 row["prefill"], row["decode"]]

    existing: dict[tuple, list[str]] = {}
    preamble = ""
    if path.is_file():
        text = path.read_text()
        if MARKER in text:
            preamble = text.split(MARKER)[0]
            for line in text.split(MARKER)[1].splitlines():
                if line.strip().startswith("|") and "tok/s" not in line and ":---" not in line:
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    if len(cells) == len(COLUMNS):
                        existing[_row_key(cells)] = cells
        else:
            preamble = text.rstrip() + "\n\n"

    key = _row_key(new_cells)
    prev = existing.get(key)
    # Keep whichever run had the higher decode throughput for this key.
    if prev is None or float(new_cells[COL_DECODE]) >= float(prev[COL_DECODE]):
        existing[key] = new_cells

    if not preamble:
        preamble = ("# geist Benchmarks (auto-recorded)\n\n"
                    "Rows below are appended by `make bench-small` / `bench-detailed`. "
                    "Each (model, host, os, target/mode, threads) key keeps its best "
                    "decode run. See [BENCHMARKING.md](BENCHMARKING.md) for methodology.\n\n")

    rows = sorted(existing.values(),
                  key=lambda c: (c[COL_MODEL], c[COL_HOST], c[COL_TARGET_MODE]))
    body = TABLE_HEADER + "\n" + "\n".join("| " + " | ".join(c) + " |" for c in rows) + "\n"
    path.write_text(f"{preamble}{MARKER}\n\n{body}")


QUALITY_MARKER = "<!-- BENCH:QUALITY -->"
QUALITY_COLS = ["Date", "Model", "Host", "Target/Mode", "MMLU %", "n", "shots"]
# MMLU is quant-determined (same GGUF -> same logits modulo kernel rounding), so
# Host/Target only confirm a build reproduces it; they don't change the number.

MMLU_RE = re.compile(r"MMLU accuracy:\s*([\d.]+)\s*\((\d+)/(\d+)\)")


def _run_mmlu(script: str, extra: list[str], limit: int, shots: int) -> tuple[float, int] | None:
    """Run an eval_mmlu*.py harness; return (accuracy, n) or None on failure."""
    cmd = [sys.executable, script, "--hf", "--shuffle",
           "--limit", str(limit), "--shots", str(shots)] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    m = MMLU_RE.search(out)
    if not m:
        sys.stderr.write(out)
        return None
    return float(m.group(1)), int(m.group(3))


def quality_suite(args: argparse.Namespace) -> None:
    gguf = resolve_gguf()
    if gguf is None:
        print("bench: no model found (set BENCH_GGUF or GEIST_GGUF_PATH). Skipping.")
        return
    # eval_geist lives in the sibling tools/ dir of the bench (tests/) bin dir.
    eval_geist = Path(args.bin_dir).parent / "tools" / "eval_geist"
    if not eval_geist.is_file():
        sys.exit(f"bench: missing {eval_geist} — run `make bin` first")

    here = Path(__file__).parent
    default_limit = "1000" if args.suite == "quality-detailed" else "200"
    limit = int(os.environ.get("BENCH_MMLU_LIMIT", default_limit))
    shots = int(os.environ.get("BENCH_MMLU_SHOTS", "5"))

    print(f"bench: MMLU cloze, {limit} questions, {shots}-shot (needs `pip install datasets`)")
    res = _run_mmlu(str(here / "eval_mmlu.py"),
                    ["--bin", str(eval_geist), "--gguf", gguf], limit, shots)
    if res is None:
        sys.exit("bench: geist MMLU failed (see output above)")
    acc, n = res
    print(f"\ngeist MMLU: {acc:.4f} ({int(acc * n)}/{n}, {shots}-shot)")

    if args.suite == "compare-ref":
        # Reference = a running llama-server on BENCH_REF_URL (same GGUF).
        url = os.environ.get("BENCH_REF_URL", "http://127.0.0.1:8080")
        ref = _run_mmlu(str(here / "eval_mmlu_llama.py"), ["--url", url], limit, shots)
        if ref is None:
            print(f"  reference skipped: no llama-server at {url}. Start one with:")
            print(f"    llama-server -m {gguf} -c 4096")
        else:
            print(f"llama.cpp MMLU: {ref[0]:.4f} (n={ref[1]}, {shots}-shot)  "
                  f"-> geist {('leads' if acc > ref[0] else 'trails' if acc < ref[0] else 'ties')} "
                  f"by {abs(acc - ref[0]) * 100:.1f} pts")

    if args.record and args.benchmark_md:
        row = [datetime.now(timezone.utc).strftime("%Y-%m-%d"), Path(gguf).name,
               host_id(), f"{args.target}/{args.mode}", f"{acc * 100:.1f}", str(n), str(shots)]
        record_quality(Path(args.benchmark_md), row)
        print(f"recorded to {args.benchmark_md}")


def record_quality(path: Path, new_cells: list[str]) -> None:
    """Upsert a quality row under QUALITY_MARKER; key = (model, host, target, n, shots)."""
    key_idx = (1, 2, 3, 5, 6)
    header = ("| " + " | ".join(QUALITY_COLS) + " |\n| "
              + " | ".join(":---" if i < 4 else ":---:" for i in range(len(QUALITY_COLS))) + " |")
    existing: dict[tuple, list[str]] = {}
    preamble = ""
    if path.is_file():
        text = path.read_text()
        if QUALITY_MARKER in text:
            preamble, body = text.split(QUALITY_MARKER, 1)
            for line in body.splitlines():
                if line.strip().startswith("|") and ":---" not in line and "MMLU" not in line:
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    if len(cells) == len(QUALITY_COLS):
                        existing[tuple(cells[i] for i in key_idx)] = cells
        else:
            preamble = text.rstrip() + "\n\n"
    existing[tuple(new_cells[i] for i in key_idx)] = new_cells  # newest wins per key
    rows = sorted(existing.values(), key=lambda c: (c[1], c[3], c[6]))
    body = header + "\n" + "\n".join("| " + " | ".join(c) + " |" for c in rows) + "\n"
    path.write_text(f"{preamble}{QUALITY_MARKER}\n\n{body}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True,
                   choices=sorted(PERF_SUITES | QUALITY_SUITES))
    p.add_argument("--target", default="unknown")
    p.add_argument("--mode", default="release")
    p.add_argument("--bin-dir", default="bin")
    # Run artifacts default OUTSIDE the repo (the Makefile passes the same path)
    # so benchmark output never lands in the working tree.
    p.add_argument("--out-dir",
                   default=str(Path.home() / "bench-geistlib" / "quality_perf"))
    p.add_argument("--benchmark-md", default="benchmark/BENCHMARK.md")
    p.add_argument("--record", action="store_true")
    args = p.parse_args()

    print(f"== geist bench: suite={args.suite} target={args.target} mode={args.mode} ==")
    if args.suite in PERF_SUITES:
        perf_suite(args)
    else:
        quality_suite(args)


if __name__ == "__main__":
    main()
