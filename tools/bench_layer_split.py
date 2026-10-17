#!/usr/bin/env python3
"""bench_layer_split.py — measure a model split by layer across a GPU and the CPU (#49).

Runs tests/bench_perf_sweep on one GGUF in several placements, and optionally
llama-bench at the same split points, interleaved over --cycles so slow drift
hits every configuration alike:

  cpu        every layer on the CPU backend (--cpu)
  gpu        every layer and the head on the GPU backend (--gpu); only with
             --gpu-only, for a model that fits the device
  split N    layers 0..N-1 on the GPU, the rest, the embedding and the head on
             the CPU (GEIST_BENCH_SPLIT_*): llama.cpp's -ngl N with the head
             on the CPU, which it is for N <= the layer count
  llama N    llama-bench -ngl N (--llama-bench), its layers N from the end

Per configuration: prefill and decode tok/s (median over cycles, min-max),
peak RSS (/usr/bin/time) and peak device
memory (nvidia-smi polled every 0.2 s, when present). Raw rows go to
--out-dir/raw.jsonl, the table to stdout and --out-dir/report.md.

    python3 tools/bench_layer_split.py --bench bin/linux/release/tests/bench_perf_sweep \
        --gguf qwen3-14b-q8_0.gguf --cpu cpu_x86 --gpu vulkan --splits 0,10,20,30 \
        --llama-bench ~/llama.cpp/build-vulkan/bin/llama-bench --out-dir split-out
Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path


class VramPeak:
    """Peak memory.used of GPU `index` while the block runs (nvidia-smi)."""

    def __init__(self, index: str | None) -> None:
        self.index, self.peak, self._stop = index, None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if self.index is not None:
            cmd += ["-i", self.index]
        while not self._stop.is_set():
            try:
                mb = int(subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.split()[0])
                self.peak = mb if self.peak is None else max(self.peak, mb)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                return
            self._stop.wait(0.2)

    def __enter__(self) -> "VramPeak":
        if shutil.which("nvidia-smi"):
            self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._t.is_alive():
            self._t.join()


def run_child(cmd: list[str], env: dict[str, str]) -> tuple[int, str, str, float]:
    """Run cmd under /usr/bin/time; return (rc, stdout, stderr, peak RSS MB)."""
    darwin = os.uname().sysname == "Darwin"
    timed = ["/usr/bin/time", "-l" if darwin else "-v"] + cmd
    proc = subprocess.run(timed, env=env, capture_output=True, text=True)
    rss_mb = float("nan")
    for line in proc.stderr.splitlines():
        if darwin and line.strip().endswith("maximum resident set size"):
            rss_mb = int(line.split()[0]) / (1024 * 1024)  # bytes
        elif not darwin and "Maximum resident set size" in line:
            rss_mb = int(line.split()[-1]) / 1024  # kbytes
    return proc.returncode, proc.stdout, proc.stderr, rss_mb


def geist_cfg(args: argparse.Namespace, name: str, primary: str, split: int) -> dict:
    env = dict(os.environ, GEIST_BENCH_BACKEND=primary)
    env.pop("GEIST_BENCH_SPLIT_BACKEND", None)
    env.pop("GEIST_BENCH_SPLIT_LAYERS", None)
    if split > 0:
        env.update(GEIST_BENCH_SPLIT_BACKEND=args.gpu, GEIST_BENCH_SPLIT_LAYERS=str(split))
    cmd = [args.bench, "--gguf", args.gguf, "--seq-lens", str(args.prompt), "--decode-n",
           str(args.decode), "--warmup", "1", "--repeats", str(args.repeats), "--emit-jsonl"]
    return {"name": name, "cmd": cmd, "env": env, "kind": "geist"}


def llama_cfg(args: argparse.Namespace, ngl: int) -> dict:
    cmd = [args.llama_bench, "-m", args.gguf, "-ngl", str(ngl), "-p", str(args.prompt),
           "-n", str(args.decode), "-r", str(args.repeats), "-o", "jsonl"]
    return {"name": f"llama.cpp -ngl {ngl}", "cmd": cmd, "env": dict(os.environ), "kind": "llama"}


def parse(cfg: dict, out: str) -> tuple[float, float]:
    """(prefill tok/s, decode tok/s) from one run's stdout."""
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    if cfg["kind"] == "geist":
        r = rows[-1]
        return r["prefill_tps"], r["decode_tps"]
    pp = next(r["avg_ts"] for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0)
    tg = next(r["avg_ts"] for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0)
    return pp, tg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="bench_perf_sweep binary")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--cpu", default="cpu_x86")
    ap.add_argument("--gpu", default="vulkan")
    ap.add_argument("--gpu-only", action="store_true", help="also run every layer on --gpu")
    ap.add_argument("--splits", default="", help="comma-separated N (layers on the GPU)")
    ap.add_argument("--llama-bench", default="", help="llama-bench binary; runs -ngl 0 and each split")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--nvidia-index", default=None)
    ap.add_argument("--out-dir", default="split-out")
    args = ap.parse_args()

    splits = [int(x) for x in args.splits.split(",") if x.strip()]
    cfgs = [geist_cfg(args, f"geistlib cpu ({args.cpu})", args.cpu, 0)]
    if args.gpu_only:
        cfgs.append(geist_cfg(args, f"geistlib gpu ({args.gpu})", args.gpu, 0))
    cfgs += [geist_cfg(args, f"geistlib split {n}", args.cpu, n) for n in splits if n > 0]
    if args.llama_bench:
        cfgs += [llama_cfg(args, n) for n in [0] + [n for n in splits if n > 0]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = (out_dir / "raw.jsonl").open("a")
    results: dict[str, list[dict]] = {c["name"]: [] for c in cfgs}
    for cycle in range(args.cycles):
        for cfg in cfgs:
            with VramPeak(args.nvidia_index) as vram:
                t0 = time.time()
                rc, out, err, rss = run_child(cfg["cmd"], cfg["env"])
            row = {"cycle": cycle, "config": cfg["name"], "rc": rc, "rss_mb_time": rss,
                   "vram_peak_mb": vram.peak, "wall_s": round(time.time() - t0, 1)}
            if rc == 0:
                try:
                    row["prefill_tps"], row["decode_tps"] = parse(cfg, out)
                except (StopIteration, KeyError, IndexError, json.JSONDecodeError) as e:
                    row["error"] = f"parse: {e}"
            else:
                row["error"] = err.strip().splitlines()[-1] if err.strip() else f"rc {rc}"
            geist_rows = [json.loads(line) for line in out.splitlines() if line.startswith('{"seq_len"')]
            if geist_rows:
                row["rss_mb"] = geist_rows[-1].get("rss_mb")
            row["stdout"], row["stderr_tail"] = out, "\n".join(err.splitlines()[-20:])
            raw.write(json.dumps(row) + "\n")
            raw.flush()
            results[cfg["name"]].append(row)
            print(f"[{cycle}] {cfg['name']}: " +
                  (f"pp {row.get('prefill_tps', 0):.1f} tg {row.get('decode_tps', 0):.2f} tok/s"
                   if "error" not in row else row["error"]), flush=True)

    def stat(rows: list[dict], key: str) -> str:
        v = [r[key] for r in rows if key in r]
        return f"{statistics.median(v):.2f} ({min(v):.2f}-{max(v):.2f})" if v else "failed"

    def peak(rows: list[dict], key: str) -> str:
        v = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
        return f"{max(v):.0f}" if v else "-"

    lines = [f"model `{Path(args.gguf).name}`, prompt {args.prompt}, decode {args.decode}, "
             f"{args.cycles} cycles x {args.repeats} repeats; median (min-max) over cycles",
             "", "| configuration | prefill tok/s | decode tok/s | peak RSS MB | peak VRAM MB |",
             "| :-- | --: | --: | --: | --: |"]
    for cfg in cfgs:
        rows = results[cfg["name"]]
        rss_key = "rss_mb_time"
        lines.append(f"| {cfg['name']} | {stat(rows, 'prefill_tps')} | {stat(rows, 'decode_tps')} | "
                     f"{peak(rows, rss_key)} | {peak(rows, 'vram_peak_mb')} |")
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
