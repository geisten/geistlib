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

--set levers runs the second list instead (GEIST_LEVERS): the settings that
move geist's rates, and then tools/bench_revision_ab.py, --ab-base against
this checkout, on each pair's GGUF. Writes the traces (gzip), cells.jsonl and summary.md to
--out.
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

# What geist ran before #590 is the base of every geist cell: serial encoders
# and a full first command buffer per decode step. The profile is of that
# engine, and the trace library's own concurrent modes replace the serial
# encoder.
SERIAL = {"GEIST_METAL_CONCURRENT": "0", "GEIST_METAL_PIPELINE_FIRST": "0"}
BRANCH = {"GEIST_METAL_CONCURRENT": "1", "GEIST_METAL_PIPELINE_FIRST": "16"}
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
# --set levers: what moves geist's rates, after the profile said where the
# time goes. GEIST_M_MAX changes the rows per prefill pass, and with them
# the low bits of the logits; the others leave the output alone.
GEIST_LEVERS = [
    ("plain", {}, False),
    ("trace", {}, True),
    ("library concurrent encoders", {"GEIST_METAL_CONCURRENT": "1"}, True),
    ("first buffer after 8", {"GEIST_METAL_PIPELINE_FIRST": "8"}, True),
    ("first buffer after 16", {"GEIST_METAL_PIPELINE_FIRST": "16"}, True),
    ("first buffer after 32", {"GEIST_METAL_PIPELINE_FIRST": "32"}, True),
    ("spin instead of wait", {"METAL_TRACE_SPIN_WAIT": "1"}, True),
    ("branch defaults (concurrent, first buffer after 16)", BRANCH, True),
    ("branch defaults, untraced", BRANCH, False),
    ("branch defaults, spin", {**BRANCH, "METAL_TRACE_SPIN_WAIT": "1"}, True),
    ("128 rows per prefill pass", {"GEIST_M_MAX": "128"}, True),
    ("512 rows per prefill pass", {"GEIST_M_MAX": "512"}, True),
    ("512 rows per prefill pass, branch defaults", {**BRANCH, "GEIST_M_MAX": "512"}, True),
    ("plain again", {}, False),
]
MLX_LEVERS = [
    ("plain", {}, False),
    ("trace", {}, True),
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
    env.update(SERIAL)
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
    """The settings before #590 (SERIAL) against the branch's defaults
    (BRANCH): the wall time of a process up to its first token (one-token
    prompt, alternating, the first two of each dropped) and bench_nll's
    logits hash over the decode and the prefill path, of this build and of
    --main-tests (origin/main's binaries) when given."""
    out = {"ready_ms": {}, "logits": {}}
    gguf = model["gguf"]["path"]
    ready = [ctx["geist"], "--gguf", gguf, "--seq-lens", "1", "--decode-n", "1", "--warmup",
             "0", "--repeats", "1"]
    settings = {"serial": SERIAL, "branch defaults": BRANCH}
    times = {name: [] for name in settings}
    for i in range(12):
        for name in list(settings) if i % 2 == 0 else list(settings)[::-1]:
            env = mv.geist_env("metal")
            env.update(settings[name])
            tic = time.perf_counter()
            run(ready, env)
            times[name].append((time.perf_counter() - tic) * 1e3)
    for name in settings:
        kept = times[name][2:]
        out["ready_ms"][name] = {"median": statistics.median(kept), "min": min(kept),
                                 "max": max(kept), "n": len(kept)}
    prompt = ROOT / mv.PINS["prompt_file"]
    arms = [(name, ctx["geist"].parent, flags) for name, flags in settings.items()]
    if ctx["main_tests"] is not None:
        arms.append(("origin/main", ctx["main_tests"], {}))
    for check, extra, args in (("decode", {}, ["96", "0"]),
                               ("prefill", {"GEIST_NLL_PREFILL": "1"}, ["640", "512"])):
        for name, tests, flag in arms:
            env = mv.geist_env("metal")
            env.update(extra)
            env.update(flag)
            try:
                line = run([tests / "bench_nll", gguf, prompt, *args], env).stdout.strip()
                row = json.loads(line.splitlines()[-1])
                row.pop("tps", None)
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                row = {"error": str(error)[-300:]}
            out["logits"].setdefault(check, {})[name] = row
    # The rows per prefill pass: the same logits or not? (Not part of
    # "identical": a different pass size is a different computation.)
    base = out["logits"]["prefill"].get("serial")
    for rows in ("128", "512"):
        env = mv.geist_env("metal")
        env.update(SERIAL)
        env.update({"GEIST_NLL_PREFILL": "1", "GEIST_M_MAX": rows, "GEIST_NLL_M_MAX": rows})
        try:
            line = run([ctx["geist"].parent / "bench_nll", gguf, prompt, "640", "512"],
                       env).stdout.strip()
            row = json.loads(line.splitlines()[-1])
            row.pop("tps", None)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            row = {"error": str(error)[-300:]}
        out.setdefault("rows_per_pass", {})[rows] = {**row, "same_as_256": row == base}
    out["identical"] = all(
        len({json.dumps(row, sort_keys=True) for row in rows.values()}) == 1
        and "error" not in next(iter(rows.values())) for rows in out["logits"].values())
    return out


def revision_ab(ctx, model, base, cycles, smoke):
    """tools/bench_revision_ab.py, `base` against this checkout's HEAD, with
    the cells' workload and environment. Returns its report."""
    out = ctx["out"] / f"ab-{model['key']}"
    env = mv.geist_env("metal")
    command = [sys.executable, ROOT / "tools" / "bench_revision_ab.py", "--rev", f"base={base}",
               "--rev", "new=HEAD", "--gguf", model["gguf"]["path"], "--make-arg",
               "TARGET=mac-omp", "--make-arg", "BACKENDS=cpu_neon cpu_scalar metal",
               "--seq-lens", mv.PROMPT_TOKENS, "--decode-n", mv.DECODE_N, "--warmup",
               mv.GEIST_WARMUP, "--prompt-file", ROOT / mv.PINS["prompt_file"], "--cycles",
               cycles, "--out-dir", out]
    for key in ("OMP_WAIT_POLICY", "OMP_NUM_THREADS", "GEIST_PREFILL_THREADS",
                "GEIST_DECODE_THREADS", "GEIST_TEXT_ONLY", "GEIST_BENCH_BACKEND"):
        command += ["--env", f"{key}={env[key]}"]
    if smoke:
        command += ["--no-cool", "--no-power-gate"]
    try:
        run(command, env, timeout=5400)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        return f"bench_revision_ab failed: {str(error)[-800:]}"
    reports = sorted(out.glob("*/revision_ab.md"))
    return reports[-1].read_text() if reports else "bench_revision_ab wrote no report"


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
    parser.add_argument("--ab-base", default=None,
                        help="revision for the A/B against HEAD (tools/bench_revision_ab.py)")
    parser.add_argument("--ab-cycles", type=int, default=None,
                        help="A/B cycles; default 10 with --set levers, else 0 (no A/B)")
    parser.add_argument("--set", choices=["profile", "levers"], default="profile",
                        help="which cells to run (GEIST_VARIANTS or GEIST_LEVERS)")
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
        levers = args.set == "levers"
        geist = [("geist", geist_cell, v) for v in (GEIST_LEVERS if levers else GEIST_VARIANTS)]
        mlx = [("mlx", mlx_cell, v) for v in (MLX_LEVERS if levers else MLX_VARIANTS)]
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
        cycles = args.ab_cycles if args.ab_cycles is not None else (10 if levers else 0)
        if cycles > 0 and args.ab_base:
            mv.log(f"{key}: A/B {args.ab_base} against HEAD, {cycles} cycles")
            report = revision_ab(ctx, model, args.ab_base, cycles, args.smoke)
            (args.out / f"{key}.revision_ab.md").write_text(report)
            summary += [f"#### A/B, {args.ab_base} against HEAD", "", report, ""]
        mv.gate(args.smoke)
        check = ready_and_identity(ctx, model)
        cells.write(json.dumps({"model": key, "serial_vs_concurrent": check}) + "\n")
        cells.flush()
        summary += ["Settings before #590 against the branch's defaults: process start to first "
                    "token, ms: " + json.dumps(check["ready_ms"]), "",
                    f"bench_nll logits identical across the arms: {check['identical']} "
                    + json.dumps(check["logits"]), "",
                    "bench_nll prefill-path logits at 128 and 512 rows per pass against the "
                    "default 256: " + json.dumps(check.get("rows_per_pass")), ""]
        picked = [f"{label}={path}" for label, path in traces.items()
                  if levers or label.split(" ", 1)[1] in ("trace", "1op",
                                                          "library concurrent encoders")]
        report = run([sys.executable, HERE / "metal_trace_report.py", *picked], None).stdout
        (args.out / f"{key}.report.md").write_text(report)
        summary += [report, ""]
    (args.out / "summary.md").write_text("\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
