#!/usr/bin/env python3
"""metal_profile.py — driver of tools/probes/metal_profile.sh (#590): where
geist's Metal time goes against mlx-lm's, on the model pairs of
tools/probes/mlx_vs_geist_models.json.

    metal_profile.py --cache DIR --gguf-cache DIR --out DIR [--models a,b] [--smoke]

Both engines run the workload of mlx_vs_geist.py (512 prompt tokens of the
prompt file, 128 greedy tokens, one warmup, three repeats per process) with
tools/probes/metal_trace.m loaded, which records every Metal command buffer.
Cells per pair, each behind mlx_vs_geist.py's gate:

    plain        no trace library: the rate the trace must not change
    trace        default settings, traced
    1op          one kernel per command buffer (GEIST_METAL_PIPELINE=1 and
                 _PREFILL=1; MLX_MAX_OPS_PER_BUFFER=1): GPU time per kernel
    geist only:  concurrent encoders, the trace library's
                 (METAL_TRACE_FORCE_CONCURRENT=1..4) and the backend's own
                 (GEIST_METAL_CONCURRENT=1); command-buffer rotation
                 periods; the tensor API off
    gpu state    macmon at 100 ms over a 768-token decode of each engine
    serial against concurrent encoders: time to the first token, and the
                 logits hash of bench_nll (also of origin/main's build)

Writes the traces (gzip), cells.jsonl and summary.md to --out.
"""
import argparse
import gzip
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import mlx_vs_geist as mv  # noqa: E402

# GEIST_METAL_CONCURRENT=0 (the serial encoder, what geist ran before #590)
# is the base of every geist cell: the profile is of that engine, and the
# trace library's own concurrent modes replace the serial encoder.
GEIST_VARIANTS = [
    # name, environment, traced
    ("plain", {}, False),
    ("trace", {}, True),
    ("1op", {"GEIST_METAL_PIPELINE": "1", "GEIST_METAL_PIPELINE_PREFILL": "1"}, True),
    ("concurrent, barrier before every dispatch", {"METAL_TRACE_FORCE_CONCURRENT": "1"}, True),
    ("concurrent, no barrier (wrong output)", {"METAL_TRACE_FORCE_CONCURRENT": "2"}, True),
    ("concurrent, barrier where dependent by bind offset",
     {"METAL_TRACE_FORCE_CONCURRENT": "3"}, True),
    ("concurrent, barrier where dependent by buffer", {"METAL_TRACE_FORCE_CONCURRENT": "4"}, True),
    ("library concurrent encoders", {"GEIST_METAL_CONCURRENT": "1"}, True),
    ("library concurrent encoders, untraced", {"GEIST_METAL_CONCURRENT": "1"}, False),
    ("rotation 48/16", {"GEIST_METAL_PIPELINE": "48", "GEIST_METAL_PIPELINE_PREFILL": "16"}, True),
    ("rotation 192/64", {"GEIST_METAL_PIPELINE": "192", "GEIST_METAL_PIPELINE_PREFILL": "64"}, True),
    ("rotation off", {"GEIST_METAL_PIPELINE": "0"}, True),
    ("tensor API off", {"GEIST_METAL_TENSOR": "0"}, True),
    ("plain again", {}, False),
]
MLX_VARIANTS = [
    ("plain", {}, False),
    ("trace", {}, True),
    ("1op", {"MLX_MAX_OPS_PER_BUFFER": "1"}, True),
    ("plain again", {}, False),
]


def slug(text):
    return "".join(c if c.isalnum() else "-" for c in text).strip("-")


def run(command, env, timeout=1800):
    proc = subprocess.run([str(c) for c in command], capture_output=True, text=True, env=env,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {proc.stderr.strip()[-1500:]}")
    return proc


def gz(path):
    if path.exists():
        with path.open("rb") as src, gzip.open(str(path) + ".gz", "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()


def geist_cell(ctx, model, name, extra, traced, decode_n=mv.DECODE_N, repeats=3):
    env = mv.geist_env("metal")
    env["GEIST_METAL_CONCURRENT"] = "0"
    env.update(extra)
    trace = ctx["out"] / f"{model['key']}.geist.{slug(name)}.jsonl"
    if traced:
        env.update({"DYLD_INSERT_LIBRARIES": str(ctx["lib"]), "METAL_TRACE_OUT": str(trace)})
    proc = run([ctx["geist"], "--gguf", model["gguf"]["path"], "--seq-lens", mv.PROMPT_TOKENS,
                "--decode-n", decode_n, "--warmup", mv.GEIST_WARMUP, "--repeats", repeats,
                "--prompt-file", ROOT / mv.PINS["prompt_file"], "--emit-tokens", "--threads",
                env["GEIST_PREFILL_THREADS"]], env)
    if "[bench] backend: metal" not in proc.stderr:
        raise RuntimeError("geist did not select the metal backend")
    row = json.loads(next(line for line in proc.stdout.splitlines() if line.startswith("{")))
    gz(trace)
    return {"prefill_tps": [mv.PROMPT_TOKENS * 1e3 / ms for ms in row["samples"]["prefill_ms"]],
            "decode_tps": [decode_n * 1e3 / ms for ms in row["samples"]["decode_ms"]],
            "tokens": row.get("tokens", []), "trace": trace.name + ".gz" if traced else None}


def mlx_cell(ctx, model, name, extra, traced, decode_n=mv.DECODE_N, repeats=3):
    command, env = mv.worker(ctx["cache"])
    env.update(extra)
    trace = ctx["out"] / f"{model['key']}.mlx.{slug(name)}.jsonl"
    proc = run([command[0], HERE / "metal_profile_mlx.py", "--trace-lib", ctx["lib"], "--model",
                model["mlx"]["path"], "--prompt-file", ROOT / mv.PINS["prompt_file"], "--out",
                trace if traced else "/dev/null", "--decode-n", decode_n, "--repeats", repeats]
               + ([] if traced else ["--no-hooks"]), env)
    runs = mv.result_of(proc.stdout)["runs"][1:]
    gz(trace)
    return {"prefill_tps": [r["prefill_tps"] for r in runs],
            "decode_tps": [r["decode_tps"] for r in runs],
            "tokens": runs[0]["tokens"], "trace": trace.name + ".gz" if traced else None}


def gpu_state(ctx, model, engine):
    """macmon at 100 ms over a long decode: GPU frequency, active share and
    power while the GPU is mostly busy."""
    macmon = shutil.which("macmon")
    if macmon is None:
        return {"error": "no macmon"}
    log = ctx["out"] / f"{model['key']}.{engine}.macmon.jsonl"
    with log.open("w") as sink:
        sampler = subprocess.Popen([macmon, "pipe", "-i", "100"], stdout=sink,
                                   stderr=subprocess.DEVNULL)
        try:
            time.sleep(1.0)
            cell = (geist_cell if engine == "geist" else mlx_cell)(
                ctx, model, "long", {}, False, decode_n=768, repeats=2)
        finally:
            sampler.terminate()
            sampler.wait()
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
    busy = [r for r in rows if r.get("gpu_usage", [0, 0])[1] > 0.5]
    gz(log)
    if not busy:
        return {"error": "no busy samples", "samples": len(rows)}
    med = statistics.median
    return {"samples": len(rows), "busy_samples": len(busy),
            "gpu_freq_mhz": med(r["gpu_usage"][0] for r in busy),
            "gpu_active": med(r["gpu_usage"][1] for r in busy),
            "gpu_w": med(r["gpu_power"] for r in busy),
            "cpu_w": med(r["cpu_power"] for r in busy),
            "decode_tps": med(cell["decode_tps"])}


def ready_and_identity(ctx, model):
    """Serial against concurrent encoders: the wall time of a process up to
    its first token (one-token prompt, alternating, the first two of each
    dropped) and bench_nll's logits hash over the decode and the prefill
    path, of this build and of --main-tests (origin/main's binaries) when
    given."""
    out = {"ready_ms": {}, "logits": {}}
    gguf = model["gguf"]["path"]
    ready = [ctx["geist"], "--gguf", gguf, "--seq-lens", "1", "--decode-n", "1", "--warmup",
             "0", "--repeats", "1"]
    times = {"0": [], "1": []}
    for i in range(12):
        for flag in ("0", "1") if i % 2 == 0 else ("1", "0"):
            env = mv.geist_env("metal")
            env["GEIST_METAL_CONCURRENT"] = flag
            tic = time.perf_counter()
            run(ready, env)
            times[flag].append((time.perf_counter() - tic) * 1e3)
    for flag, name in (("0", "serial"), ("1", "concurrent")):
        kept = times[flag][2:]
        out["ready_ms"][name] = {"median": statistics.median(kept), "min": min(kept),
                                 "max": max(kept), "n": len(kept)}
    prompt = ROOT / mv.PINS["prompt_file"]
    arms = [("serial", ctx["geist"].parent, "0"), ("concurrent", ctx["geist"].parent, "1")]
    if ctx["main_tests"] is not None:
        arms.append(("origin/main", ctx["main_tests"], "0"))
    for check, extra, args in (("decode", {}, ["96", "0"]),
                               ("prefill", {"GEIST_NLL_PREFILL": "1"}, ["640", "512"])):
        for name, tests, flag in arms:
            env = mv.geist_env("metal")
            env.update(extra)
            env["GEIST_METAL_CONCURRENT"] = flag
            try:
                line = run([tests / "bench_nll", gguf, prompt, *args], env).stdout.strip()
                row = json.loads(line.splitlines()[-1])
                row.pop("tps", None)
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                row = {"error": str(error)[-300:]}
            out["logits"].setdefault(check, {})[name] = row
    out["identical"] = all(
        len({json.dumps(row, sort_keys=True) for row in rows.values()}) == 1
        and "error" not in next(iter(rows.values())) for rows in out["logits"].values())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gguf-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--geist", type=Path,
                        default=ROOT / "bin" / "mac-omp" / "release" / "tests" / "bench_perf_sweep")
    parser.add_argument("--models", default="qwen35-0.8b-8bit,llama32-3b-4bit")
    parser.add_argument("--main-tests", type=Path, default=None,
                        help="bin/.../tests of an origin/main build, for the logits check")
    parser.add_argument("--smoke", action="store_true", help="no gates (method check)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = {"cache": args.cache, "out": args.out, "lib": args.lib, "geist": args.geist,
           "main_tests": args.main_tests}
    mv.ensure_venv(args.cache)
    cells = (args.out / "cells.jsonl").open("a")
    summary = ["## Metal profile, geist against mlx-lm (#590)", ""]
    for key in args.models.split(","):
        model = mv.ensure_model(key, args.cache, args.gguf_cache)
        model["key"] = key
        rows, traces, base_tokens = [], {}, {}
        # geist and mlx-lm cells alternate, so drift of the host hits both.
        geist = [("geist", geist_cell, v) for v in GEIST_VARIANTS]
        mlx = [("mlx", mlx_cell, v) for v in MLX_VARIANTS]
        plan = []
        while geist or mlx:
            plan += geist[:1] + (mlx[:1] if len(mlx) > 1 or not geist[1:] else [])
            mlx = mlx[1:] if len(mlx) > 1 or not geist[1:] else mlx
            geist = geist[1:]
        for engine, cell, (name, extra, traced) in plan:
            gate = mv.gate(args.smoke)
            mv.log(f"{key}: {engine} {name}")
            try:
                got = cell(ctx, model, name, extra, traced)
            except (RuntimeError, subprocess.TimeoutExpired, StopIteration) as error:
                got = {"error": str(error)[-600:]}
            record = {"model": key, "engine": engine, "variant": name, "env": extra,
                      "gate": gate, **got}
            cells.write(json.dumps(record) + "\n")
            cells.flush()
            if "error" in got:
                rows.append(f"| {engine} | {name} | failed: {got['error'][-80:]} | | |")
                continue
            base_tokens.setdefault(engine, got["tokens"])
            same = "same" if got["tokens"] == base_tokens[engine] else "DIFFERENT"
            rows.append(f"| {engine} | {name} | {mv.med_range(got['prefill_tps'], 0)} | "
                        f"{mv.med_range(got['decode_tps'])} | {same} |")
            if got["trace"]:
                traces[f"{engine} {name}"] = args.out / got["trace"]
        summary += [f"### {model['label']}", "",
                    "geist and mlx-lm tokens equal: "
                    f"{base_tokens.get('geist') == base_tokens.get('mlx')}", "",
                    "| engine | cell | prefill tok/s | decode tok/s | tokens against the "
                    "engine's plain cell |", "| :-- | :-- | --: | --: | :-- |", *rows, ""]
        state = {engine: gpu_state(ctx, model, engine) for engine in ("geist", "mlx")}
        cells.write(json.dumps({"model": key, "gpu_state": state}) + "\n")
        cells.flush()
        summary += ["GPU state over a 768-token decode (macmon, samples with the GPU over 50 % "
                    "active): " + json.dumps(state), ""]
        mv.gate(args.smoke)
        check = ready_and_identity(ctx, model)
        cells.write(json.dumps({"model": key, "serial_vs_concurrent": check}) + "\n")
        cells.flush()
        summary += ["Serial against concurrent encoders (GEIST_METAL_CONCURRENT=0/1): process "
                    "start to first token, ms: " + json.dumps(check["ready_ms"]), "",
                    f"bench_nll logits identical across the arms: {check['identical']} "
                    + json.dumps(check["logits"]), ""]
        picked = [f"{label}={path}" for label, path in traces.items()
                  if label.split(" ", 1)[1] in ("trace", "1op", "library concurrent encoders")]
        report = run([sys.executable, HERE / "metal_trace_report.py", *picked], None).stdout
        (args.out / f"{key}.report.md").write_text(report)
        summary += [report, ""]
    (args.out / "summary.md").write_text("\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
