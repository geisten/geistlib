#!/usr/bin/env python3
"""bench_quality_perf.py — reproducible perf/quality benchmark harness for geist.

Driven by the `make bench-small / bench-detailed / bench-quality-* /
bench-compare-ref` targets. The goal is *reproducibility*: every recorded
number is tagged with the host, OS, target, mode, thread count, and model so
results from different machines never silently overwrite each other.

Perf suites (`small`, `detailed`) are fully implemented: they run the C
`bench_perf_sweep` binary against a GGUF and record prefill/decode tok/s,
derived TTFT and the run-to-run spread into benchmark/results/APPLE.md, keeping
the best run per (model, host, os, target, mode, threads) key.

The sweep owns the measurement protocol — warm-up, repeats, and the
mean/best/worst aggregation — and reports it as JSONL, so this wrapper only
selects the workload and records the result. It deliberately does not
re-implement any statistics.

Quality suites (`quality-small`, `quality-detailed`) and `compare-ref` require
a reference toolchain (HF tokenizer + datasets, and/or a llama.cpp build) that
is out of scope for a hermetic `make` invocation. They print setup guidance and
exit cleanly rather than failing the build. See benchmark/METHODOLOGY.md.

Usage (normally invoked via the Makefile):
    python3 tools/bench_quality_perf.py --suite small \\
        --target mac-omp --mode release \\
        --bin-dir bin/mac-omp/release/tests --out-dir ~/bench-geistlib/quality_perf \\
        --benchmark-md benchmark/results/APPLE.md --record

Environment:
    BENCH_GGUF      Path to the model GGUF (falls back to GEIST_GGUF_PATH).
    BENCH_THREADS   OMP thread count (sets OMP_NUM_THREADS for the child).
    BENCH_REF_GGUF  Reference GGUF for compare-ref (quality suites).
    BENCH_REF_BIN   Reference binary (e.g. llama-bench) for compare-ref.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPLE_PROTOCOL_PATH = ROOT / "benchmark" / "apple_cpu_protocol.json"

PERF_SUITES = {"small", "detailed"}
QUALITY_SUITES = {"quality-small", "quality-detailed", "compare-ref"}

# Machine-readable source of truth shared with APPLE.md and the interleaved A/B
# runner. Changing it is a protocol migration and must update recorded baselines.
_APPLE_PROTOCOL = json.loads(APPLE_PROTOCOL_PATH.read_text())
SWEEP_WORKLOAD = {
    name: {
        "seq_len": config["seq_lens"][0],
        "decode_n": config["decode_n"],
        "warmup": config["warmup"],
        "repeats": config["repeats"],
    }
    for name, config in _APPLE_PROTOCOL["suites"].items()
}

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


def run_sweep(bin_dir: Path, gguf: str, threads: str | None,
              workload: dict[str, int]) -> tuple[dict, str]:
    """Run bench_perf_sweep once and return its JSONL record plus diagnostics.

    The sweep already performs warm-up, repeats and mean/best/worst
    aggregation, and states them in the record (`agg`, `repeats`, `warmup`).
    """
    exe = bin_dir / "bench_perf_sweep"
    if not exe.is_file():
        sys.exit(f"bench: missing {exe} — run `make bench` to build the bench binaries first")

    env = dict(os.environ)
    env["GEIST_GGUF_PATH"] = gguf
    env.setdefault("OMP_WAIT_POLICY", "active")
    if threads:
        env["OMP_NUM_THREADS"] = threads

    cmd = [str(exe), "--gguf", gguf,
           "--seq-lens", str(workload["seq_len"]),
           "--decode-n", str(workload["decode_n"]),
           "--warmup", str(workload["warmup"]),
           "--repeats", str(workload["repeats"]),
           "--emit-jsonl"]
    if threads:
        cmd += ["--threads", threads]

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode not in (0, None):
        sys.stderr.write(out)
        sys.exit(f"bench: bench_perf_sweep exited {proc.returncode}")

    # One JSON object per seq_len; a single seq_len is requested, so take the last.
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line), proc.stderr
            except json.JSONDecodeError:
                break
    sys.stderr.write(out)
    sys.exit("bench: bench_perf_sweep produced no JSONL record")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if proc.returncode != 0:
        return "unavailable"
    return proc.stdout.strip() or "unavailable"


def benchmark_metadata(exe: Path, gguf: Path, workload: dict[str, int],
                       threads: str | None, diagnostics: str) -> dict:
    compiler = shlex.split(os.environ.get("CC", "clang"))
    linker_probe = ["otool", "-L", str(exe)] if sys.platform == "darwin" else ["ldd", str(exe)]
    return {
        "schema": "geist.benchmark.raw.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geist_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "model_path": str(gguf.resolve()),
        "model_sha256": hash_file(gguf),
        "binary_path": str(exe.resolve()),
        "binary_sha256": hash_file(exe),
        "compiler": command_output(compiler + ["--version"]).splitlines()[0],
        "linked_libraries": command_output(linker_probe).splitlines(),
        "host": host_id(),
        "os": os_id(),
        "protocol": {**workload, "aggregation": "mean"},
        "environment": {
            key: os.environ.get(key, "")
            for key in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "KMP_BLOCKTIME",
                        "GEIST_PREFILL_THREADS", "GEIST_DECODE_THREADS")
        },
        "threads": threads or "default",
        "diagnostics": diagnostics.splitlines(),
    }


def spread_pct(rec: dict) -> float:
    """Run-to-run spread of decode throughput, as ±% around the mean.

    Derived from the best/worst decode times the sweep reports. A wide spread
    means the box was not quiet — read the row with that in mind.
    """
    best_ms, worst_ms = rec.get("decode_ms_best"), rec.get("decode_ms_worst")
    mean_tps = rec.get("decode_tps") or 0.0
    if not best_ms or not worst_ms or mean_tps <= 0.0:
        return 0.0
    n = rec["decode_n"]
    tps_best = n * 1000.0 / best_ms      # fastest run -> highest tok/s
    tps_worst = n * 1000.0 / worst_ms
    return (tps_best - tps_worst) / mean_tps * 100.0 / 2.0


def ttft_ms(rec: dict) -> float:
    """Time to first token: prefill plus one decode step.

    Derived, not measured directly — the error is well under one token time.
    This is the ENGINE's TTFT; a request through an agent runtime additionally pays
    tokenization, chat templating and any tool round trip.
    """
    decode_n = rec.get("decode_n") or 1
    return rec["prefill_ms"] + rec["decode_ms"] / decode_n


def perf_suite(args: argparse.Namespace) -> None:
    gguf = resolve_gguf()
    if gguf is None:
        print("bench: no model found (set BENCH_GGUF or GEIST_GGUF_PATH, or run "
              "`make fetch-model`). Skipping perf suite.")
        return

    threads = os.environ.get("BENCH_THREADS") or None
    workload = SWEEP_WORKLOAD[args.suite]
    print(f"  workload: seq_len={workload['seq_len']} decode_n={workload['decode_n']} "
          f"warmup={workload['warmup']} repeats={workload['repeats']}")

    bin_dir = Path(args.bin_dir)
    exe = bin_dir / "bench_perf_sweep"
    rec, diagnostics = run_sweep(bin_dir, gguf, threads, workload)
    spread = spread_pct(rec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the raw record next to the table row — the table is a summary, the
    # JSONL is the evidence (rss, per-phase best/worst, protocol fields).
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    raw = out_dir / f"{timestamp}_{args.target}_{args.suite}.jsonl"
    metadata = benchmark_metadata(exe, Path(gguf), workload, threads, diagnostics)
    raw.write_text(json.dumps({"metadata": metadata, "measurement": rec}, sort_keys=True) + "\n")

    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "suite": args.suite,
        "model": Path(gguf).name,
        "host": host_id(),
        "os": os_id(),
        "target": args.target,
        "mode": args.mode,
        "threads": threads or "default",
        "commit": metadata["geist_commit"][:12],
        "model_sha": metadata["model_sha256"][:12],
        "prefill": f"{rec['prefill_tps']:.1f}",
        "decode": f"{rec['decode_tps']:.1f}",
        "ttft": f"{ttft_ms(rec):.0f}",
        "spread": f"±{spread:.0f}%",
    }
    print(f"\n{rec.get('agg', 'mean')} of {rec.get('repeats', '?')} repeats: "
          f"prefill {row['prefill']} tok/s | decode {row['decode']} tok/s {row['spread']} | "
          f"TTFT {row['ttft']} ms | RSS {rec.get('rss_mb', 0):.0f} MB")
    print(f"  ({row['host']}, {args.target}/{args.mode}, threads={row['threads']})")
    print(f"  raw record: {raw}")

    if args.record and args.benchmark_md:
        update_benchmark_md(Path(args.benchmark_md), row)
        print(f"recorded to {args.benchmark_md}")


# Marker block in benchmark/BENCHMARK.md that this script owns. Hand-written prose above
# the marker is preserved; only the auto-recorded table below it is rewritten.
MARKER = "<!-- BENCH:AUTO -->"

# A recorded row always starts with an ISO date; prose tables in the same file do not.
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _split_at_marker(text: str) -> tuple[str, str | None]:
    """Split into (hand-written prose, auto-table area) at the REAL marker.

    The marker must be alone on its line. The prose above legitimately *mentions*
    the marker inline (in backticks) to explain the convention; splitting on the
    first textual occurrence would treat that sentence as the boundary and
    destroy every line below it on the next --record run.
    """
    lines = text.splitlines(keepends=True)
    idx = None
    for i, line in enumerate(lines):
        if line.strip() == MARKER:
            idx = i
    if idx is None:
        return text, None
    return "".join(lines[:idx]), "".join(lines[idx + 1:])

# Column layout of the auto-recorded table — one source of truth for both the
# header and every positional access below (no magic indices elsewhere).
COLUMNS = ["Date", "Model", "Host", "OS", "Target/Mode", "Threads",
           "Prefill tok/s", "Decode tok/s", "Spread", "TTFT ms", "Commit", "Model SHA256",
           "Suite"]
(COL_DATE, COL_MODEL, COL_HOST, COL_OS, COL_TARGET_MODE,
 COL_THREADS, COL_PREFILL, COL_DECODE, COL_SPREAD, COL_TTFT,
 COL_COMMIT, COL_MODEL_SHA, COL_SUITE) = range(len(COLUMNS))
# Columns identifying a unique config (date + tok/s excluded).
KEY_COLS = (COL_MODEL, COL_HOST, COL_OS, COL_TARGET_MODE, COL_THREADS, COL_SUITE)

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
                 row["prefill"], row["decode"], row["spread"], row["ttft"],
                 row["commit"], row["model_sha"], row["suite"]]

    existing: dict[tuple, list[str]] = {}
    preamble = ""
    if path.is_file():
        text = path.read_text()
        head, tail = _split_at_marker(text)
        if tail is not None:
            preamble = head
            for line in tail.splitlines():
                if line.strip().startswith("|") and "tok/s" not in line and ":---" not in line:
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    # Only OUR rows: the first cell is an ISO date. Hand-written
                    # prose tables may follow the marker and must not be slurped
                    # into the auto table (the old exact-width check hid this).
                    if not cells or not DATE_RE.fullmatch(cells[0]):
                        continue
                    # Rows recorded before Spread/TTFT existed are padded rather
                    # than dropped — an old row is history, not garbage.
                    if len(cells) < len(COLUMNS):
                        cells += ["—"] * (len(COLUMNS) - len(cells))
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
                    "Each (model, host, os, target/mode, threads, suite) key keeps its best "
                    "decode run. See [../METHODOLOGY.md](../METHODOLOGY.md) for methodology.\n\n")

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
