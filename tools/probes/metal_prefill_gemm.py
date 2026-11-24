#!/usr/bin/env python3
"""metal_prefill_gemm.py — driver of tools/probes/metal_prefill_gemm.sh (#590).

    metal_prefill_gemm.py --cache DIR --gguf-cache DIR --out DIR --lib DYLIB
                          --main-tests DIR --main-rev SHA [--models a,b] [--cycles N] [--smoke]

Arms per pair of mlx_vs_geist_models.json, interleaved, the order rotated
per cycle, each behind mlx_vs_geist.py's gate:

    main   origin/main's bench_perf_sweep
    new    this checkout's (the tensor-API GEMM of fragments)
    coop   this checkout's with GEIST_METAL_TC_COOP=1 (#314's kernel)
    mlx    mlx-lm (tools/probes/metal_profile_mlx.py, no trace hooks)

Workload as mlx_vs_geist.py: 512 prompt tokens, 128 greedy tokens, one
warmup, three repeats per process. Then tests/bench_nll's logits hash on
the three geist arms, decode path (95 tokens one at a time) and prefill
path (511 tokens in one call, 127 scored steps); where main and another
arm differ, both dump their logits and tools/probes/logits_diff.py measures
the difference. Then the time from process start to the first token of a
one-token prompt, main against new (the tensor unit is compiled in it).
Last per pair: tools/bench_revision_ab.py, main against this checkout's
HEAD (--ab-cycles, 0 for none).
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import metal_profile as mp  # noqa: E402
import mlx_vs_geist as mv  # noqa: E402

ARMS = ["main", "new", "coop", "mlx"]


def geist_run(tests, model, extra):
    env = mv.geist_env("metal")
    env.update(extra)
    proc = mp.run([tests / "bench_perf_sweep", "--gguf", model["gguf"]["path"], "--seq-lens",
                   mv.PROMPT_TOKENS, "--decode-n", mv.DECODE_N, "--warmup", mv.GEIST_WARMUP,
                   "--repeats", 3, "--prompt-file", ROOT / mv.PINS["prompt_file"], "--emit-tokens",
                   "--threads", env["GEIST_PREFILL_THREADS"]], env)
    if "[bench] backend: metal" not in proc.stderr:
        raise RuntimeError("geist did not select the metal backend")
    row = json.loads(next(line for line in proc.stdout.splitlines() if line.startswith("{")))
    return {"prefill_tps": [mv.PROMPT_TOKENS * 1e3 / ms for ms in row["samples"]["prefill_ms"]],
            "decode_tps": [mv.DECODE_N * 1e3 / ms for ms in row["samples"]["decode_ms"]],
            "tokens": row.get("tokens", [])}


def nll(tests, model, extra, args):
    env = mv.geist_env("metal")
    env.update(extra)
    line = mp.run([tests / "bench_nll", model["gguf"]["path"], ROOT / mv.PINS["prompt_file"],
                   *args], env).stdout.strip()
    row = json.loads(line.splitlines()[-1])
    row.pop("tps", None)
    return row


def ready(ctx, model):
    """Process start to the first token of a one-token prompt, ms: main's
    binary against this checkout's, alternating, the first two of each
    dropped. The tensor unit is compiled in that time, and tc_frag_layout
    run."""
    arms = {"main": ctx["main_tests"], "new": ctx["tests"]}
    times = {name: [] for name in arms}
    for i in range(12):
        for name in list(arms) if i % 2 == 0 else list(arms)[::-1]:
            tic = time.perf_counter()
            mp.run([arms[name] / "bench_perf_sweep", "--gguf", model["gguf"]["path"], "--seq-lens",
                    "1", "--decode-n", "1", "--warmup", "0", "--repeats", "1"],
                   mv.geist_env("metal"))
            times[name].append((time.perf_counter() - tic) * 1e3)
    return {name: {"median": round(statistics.median(v[2:]), 2), "min": round(min(v[2:]), 2),
                   "max": round(max(v[2:]), 2), "n": len(v) - 2} for name, v in times.items()}


def identity(ctx, model):
    """bench_nll on the three geist arms; a dump and its difference where
    main and another arm disagree."""
    arms = {"main": (ctx["main_tests"], {}), "new": (ctx["tests"], {}),
            "coop": (ctx["tests"], {"GEIST_METAL_TC_COOP": "1"})}
    out = {}
    for check, extra, args, scored in (("decode", {}, ["96", "0"], 95),
                                       ("prefill", {"GEIST_NLL_PREFILL": "1"}, ["640", "512"],
                                        127)):
        rows = {}
        for name, (tests, flags) in arms.items():
            try:
                rows[name] = nll(tests, model, {**extra, **flags}, args)
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                rows[name] = {"error": str(error)[-300:]}
        out[check] = rows
        for other in ("new", "coop"):
            if rows["main"] == rows[other] or "error" in rows["main"] or "error" in rows[other]:
                continue
            dumps = {}
            for name in ("main", other):
                tests, flags = arms[name]
                dumps[name] = ctx["out"] / f"{model['key']}.{check}.{name}.logits"
                nll(tests, model, {**extra, **flags, "GEIST_NLL_DUMP": str(dumps[name])}, args)
            vocab = dumps["main"].stat().st_size // 4 // scored
            diff = mp.run([sys.executable, HERE / "logits_diff.py", dumps["main"], dumps[other],
                           vocab], None).stdout.strip()
            out[f"{check} main against {other}"] = json.loads(diff)
            for path in dumps.values():
                path.unlink()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gguf-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--main-tests", type=Path, required=True)
    parser.add_argument("--main-rev", default="origin/main")
    parser.add_argument("--models", default="qwen35-0.8b-8bit,llama32-3b-4bit")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--ab-cycles", type=int, default=10,
                        help="cycles of tools/bench_revision_ab.py per pair; 0: none")
    parser.add_argument("--smoke", action="store_true", help="no gates (method check)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = {"cache": args.cache, "out": args.out, "lib": args.lib, "main_tests": args.main_tests,
           "tests": ROOT / "bin" / "mac-omp" / "release" / "tests"}
    mv.ensure_venv(args.cache)
    cells = (args.out / "cells.jsonl").open("a")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    cells.write(json.dumps({"kind": "metadata",
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "geistlib_commit": head, "main": args.main_rev,
                            "models": args.models}) + "\n")
    summary = ["## Tensor-API GEMM of fragments: origin/main, this checkout, GEIST_METAL_TC_COOP=1 "
               "and mlx-lm (#590)", "", f"main `{args.main_rev[:8]}`, new `{head[:8]}`", ""]
    for key in args.models.split(","):
        model = mv.ensure_model(key, args.cache, args.gguf_cache)
        model["key"] = key
        got = {arm: {"prefill_tps": [], "decode_tps": [], "tokens": None} for arm in ARMS}
        for cycle in range(args.cycles):
            for arm in ARMS[cycle % 4:] + ARMS[:cycle % 4]:
                gate = mv.gate(args.smoke)
                mv.log(f"{key}: cycle {cycle} {arm}")
                try:
                    if arm == "mlx":
                        cell = mp.mlx_cell(ctx, model, "plain", {}, False)
                    else:
                        cell = geist_run(ctx["main_tests"] if arm == "main" else ctx["tests"], model,
                                         {"GEIST_METAL_TC_COOP": "1"} if arm == "coop" else {})
                except (RuntimeError, subprocess.TimeoutExpired, StopIteration) as error:
                    cells.write(json.dumps({"model": key, "arm": arm, "cycle": cycle,
                                            "error": str(error)[-600:]}) + "\n")
                    continue
                cells.write(json.dumps({"model": key, "arm": arm, "cycle": cycle, "gate": gate,
                                        "prefill_tps": cell["prefill_tps"],
                                        "decode_tps": cell["decode_tps"],
                                        "tokens": cell["tokens"]}) + "\n")
                cells.flush()
                got[arm]["prefill_tps"] += cell["prefill_tps"]
                got[arm]["decode_tps"] += cell["decode_tps"]
                if got[arm]["tokens"] is None:
                    got[arm]["tokens"] = cell["tokens"]
                elif got[arm]["tokens"] != cell["tokens"]:
                    got[arm]["tokens"] = "varies between cycles"
        base = got["main"]
        summary += [f"### {model['label']}", "",
                    "| arm | prefill tok/s | against main | decode tok/s | against main | "
                    "tokens as main's |", "| :-- | --: | --: | --: | --: | :-- |"]
        for arm in ARMS:
            row = got[arm]
            if not row["prefill_tps"]:
                summary.append(f"| {arm} | failed | | | | |")
                continue
            rel = ["", ""]
            if base["prefill_tps"]:
                rel = ["%+.1f %%" % (100.0 * (statistics.median(row[k]) /
                                              statistics.median(base[k]) - 1.0))
                       for k in ("prefill_tps", "decode_tps")]
            summary.append(f"| {arm} | {mv.med_range(row['prefill_tps'], 0)} | {rel[0]} | "
                           f"{mv.med_range(row['decode_tps'])} | {rel[1]} | "
                           f"{row['tokens'] == base['tokens']} |")
        mv.gate(args.smoke)
        check = identity(ctx, model)
        cells.write(json.dumps({"model": key, "identity": check}) + "\n")
        cells.flush()
        summary += ["", "bench_nll: " + json.dumps(check), ""]
        try:
            start = ready(ctx, model)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            start = {"error": str(error)[-300:]}
        cells.write(json.dumps({"model": key, "ready_ms": start}) + "\n")
        cells.flush()
        summary += ["process start to first token, one-token prompt, ms: " + json.dumps(start), ""]
        if args.ab_cycles > 0:
            mv.log(f"{key}: A/B {args.main_rev} against HEAD, {args.ab_cycles} cycles")
            report = mp.revision_ab(ctx, model, args.main_rev, args.ab_cycles, args.smoke)
            (args.out / f"{key}.revision_ab.md").write_text(report)
            summary += [f"#### A/B, {args.main_rev[:8]} against HEAD", "", report, ""]
        (args.out / "summary.md").write_text("\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
