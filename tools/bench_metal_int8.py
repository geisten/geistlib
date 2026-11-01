#!/usr/bin/env python3
"""Compare block-scaled M6 INT8 with the default Metal 4 FP16 path on identical GGUFs.

Build the four binaries listed in docs/BACKENDS.md first. Run this script
under the Mac host lock. No downloads or external Python dependencies.
The same executable is used for both arms; only GEIST_METAL_INT8 changes.
Outputs raw JSONL, model/binary hashes, thermal/power state and summary JSON.
Quality is a small, self-authored smoke corpus, not a standard perplexity set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import time

from bench_host import command_output, sha256_file, wait_for_apple_cool, wait_for_apple_power
from eval_repl import EvalRepl

ROOT = Path(__file__).resolve().parents[1]
# Fixed completion tasks; the continuation begins with its intended space.
CASES = [
    ("Paris", "France is a country in western Europe. Its largest city and capital is", " Paris, which is known for the Eiffel Tower and the river Seine."),
    ("arithmetic", "A shop has twelve boxes with six pencils in each box. The total number of pencils is", " seventy-two, because twelve multiplied by six equals seventy-two."),
    ("water", "At standard atmospheric pressure, pure water freezes at zero degrees Celsius and boils at", " one hundred degrees Celsius. Pressure changes the boiling point."),
    ("German", "Berlin ist die Hauptstadt von Deutschland. Die Hauptstadt von Frankreich ist", " Paris. Beide Städte sind wichtige politische und kulturelle Zentren in Europa."),
    ("Python", "In Python, a function that returns the square of its argument can be written as:\ndef square(x):\n    return", " x * x\n\nCalling square(5) returns 25."),
    ("JSON", 'A valid JSON object uses double quotes around keys and string values. For example:\n{"name": "Ada", "age":', ' 36, "active": true}'),
    ("sorting", "To sort the numbers 9, 2, 7, and 4 in ascending order, begin with the smallest. The sorted list is", " 2, 4, 7, 9. Each number is greater than the one before it."),
    ("biology", "Plants use sunlight, water, and carbon dioxide to make sugars. This process is called", " photosynthesis, and it releases oxygen into the atmosphere."),
    ("reasoning", "All squares are rectangles, but not all rectangles are squares. A square has four equal sides and", " four right angles. A rectangle does not need to have four equal sides."),
    ("translation", 'The French greeting "Bonjour" is commonly used during the day. In English, it means', ' "Hello" or "Good morning", depending on the context.'),
    ("history", "The first crewed landing on the Moon took place in 1969 during the Apollo 11 mission. The first person to walk on the Moon was", " Neil Armstrong, followed by Buzz Aldrin."),
    ("context", "Mira put a red book on the table and a blue book on the shelf. Later, she picked up the book from the table. The color of the book she picked up was", " red, because the red book was on the table."),
]


CONTEXT = ("Read the following short reference notes, then complete the final sentence. "
           "A library keeps books on numbered shelves and records each loan in a catalogue. "
           "Visitors can ask a librarian to find a title, renew a loan, or reserve a quiet room. "
           "The reading room opens in the morning and closes in the evening. "
           "A community garden beside the library has flowers, vegetables, and several fruit trees. "
           "Volunteers water the plants regularly, repair the paths, and share the harvest. "
           "Weather reports help them plan their work. They keep a notebook with the date, "
           "temperature, rainfall, and tasks completed each day. These notes provide background "
           "only; answer the final prompt using its own facts.\n\n")
CASES = [(name, CONTEXT + prompt, continuation) for name, prompt, continuation in CASES]


def env_for(int8: int) -> dict:
    return dict(os.environ, GEIST_BACKEND="metal", GEIST_BENCH_BACKEND="metal",
                GEIST_DUMP_BACKEND="metal", GEIST_METAL_TENSOR="1",
                GEIST_METAL_INT8=str(int8), GEIST_METAL_INT8_TILE=os.environ.get("GEIST_METAL_INT8_TILE", "64"),
                GEIST_METAL_PROFILE="0", OMP_NUM_THREADS="4")


def escaped(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def quality(binary: Path, model: Path, arm: str, out: Path) -> list[dict]:
    repl = EvalRepl(binary, model, log=out / f"quality-{arm}.err")
    rows = []
    try:
        for name, prompt, continuation in CASES:
            ids = repl.encode(prompt)
            if repl.bos is not None:
                ids.insert(0, repl.bos)
            if len(ids) < 128:
                raise RuntimeError("quality prompt too short to exercise prefill GEMM")
            cont = repl.encode(continuation)
            prefix = f"{len(ids)} " + " ".join(map(str, ids))
            score = repl.command(f"SCORE {prefix} {len(cont)} " + " ".join(map(str, cont)))
            decode = repl.command(f"DECODE {prefix} 24")
            text = " ".join(repl.command("GEN 24 " + escaped(prompt)))
            logprobs = list(map(float, score[1:]))
            if len(logprobs) != len(cont) or not all(math.isfinite(v) for v in logprobs):
                raise RuntimeError("invalid quality scores")
            tokens = list(map(int, decode[1:]))
            if len(tokens) != int(decode[0]) or any(t < 0 for t in tokens):
                raise RuntimeError("invalid generated token count")
            rows.append(dict(case=name, prompt=prompt, continuation=continuation,
                             prompt_ids=ids, continuation_ids=cont, logprobs=logprobs,
                             generated_ids=tokens, generated_text=text))
    finally:
        repl.close()
    log = (out / f"quality-{arm}.err").read_text()
    match = re.search(r"dispatch\.int8_quant_x\s+(\d+) dispatches", log)
    actual = int(match.group(1)) if match else 0
    if (actual > 0) != (arm == "int8"):
        raise RuntimeError("quality arm did not exercise the requested native path")
    for row in rows:
        row["quality_run_int8_dispatches"] = actual
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=GGUF")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bin-dir", type=Path, default=ROOT / "bin/mac-omp/release")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--tile-rows", type=int, choices=(32, 64, 128), default=64)
    ap.add_argument("--tune-only", action="store_true", help="six-cycle tile sweep, no quality run")
    args = ap.parse_args()
    os.environ["GEIST_METAL_INT8_TILE"] = str(args.tile_rows)
    if args.cycles < 6:
        ap.error("at least six interleaved cycles are required")
    models = [tuple(item.split("=", 1)) for item in args.model]
    if len({name for name, _ in models}) != len(models):
        ap.error("model labels must be unique")
    args.out.mkdir(parents=True, exist_ok=True)
    bench = args.bin_dir / "tests/bench_perf_sweep"
    oracle = args.bin_dir / "tests/test_metal_int8_unit"
    evaluate = args.bin_dir / "tools/eval_geist"
    dump = args.bin_dir / "tools/dump_geist_logits"
    for tile in (32, 64, 128):
        subprocess.run([str(oracle)], env=dict(env_for(1), GEIST_METAL_INT8_TILE=str(tile)), check=True)
    metadata = dict(cpu=command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
                    os=command_output(["sw_vers"]), revision=command_output(["git", "rev-parse", "HEAD"]),
                    working_tree_dirty=bool(subprocess.check_output(
                        ["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
                    binary_sha256={str(p.relative_to(args.bin_dir)): sha256_file(p)
                                   for p in (bench, oracle, evaluate, dump)}, cycles=args.cycles,
                    protocol=dict(seq_lens=[128, 512], decode_n=32, warmup=16, repeats=1,
                                  order="AB/BA alternation, same binary", threads=4, int8_tile_rows=args.tile_rows), models={})
    for label, path in models:
        metadata["models"][label] = dict(path=path, sha256=sha256_file(Path(path)))
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.tune_only:
        records = []
        with (args.out / "tuning.jsonl").open("w") as raw:
            for cycle in range(args.cycles):
                for label, path in models:
                    for index in range(3):
                        tile = (32, 64, 128)[(index + cycle) % 3]
                        print(f"tune cycle={cycle} {label} tile={tile}", flush=True)
                        deadline = time.monotonic() + 1200
                        power = wait_for_apple_power(deadline)
                        cool = wait_for_apple_cool(dict(max_gpu_c=65, max_cpu_c=65,
                                                      fallback_seconds=45), deadline)
                        with (args.out / f"tune-{label}-{cycle}-{tile}.err").open("w") as err:
                            proc = subprocess.run([str(bench), "--gguf", path, "--seq-lens", "512",
                                                   "--decode-n", "8", "--warmup", "16", "--repeats", "1"],
                                      env=dict(env_for(1), GEIST_METAL_INT8_TILE=str(tile)),
                                      text=True, stdout=subprocess.PIPE, stderr=err, check=True)
                        for line in proc.stdout.splitlines():
                            row = json.loads(line)
                            if row.get("metal_int8_gemm_dispatches", 0) == 0:
                                raise RuntimeError("tile tuning silently fell back")
                            row.update(model=label, tile_rows=tile, cycle=cycle, power=power, thermal=cool)
                            raw.write(json.dumps(row) + "\n"); raw.flush(); records.append(row)
        summary = [dict(model=label, tile_rows=tile,
                        prefill_tps=statistics.median(r["prefill_tps"] for r in records
                                                     if r["model"] == label and r["tile_rows"] == tile))
                   for label, _ in models for tile in (32, 64, 128)]
        (args.out / "tuning-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return
    runs = []
    with (args.out / "runs.jsonl").open("w") as raw:
        for cycle in range(args.cycles):
            for label, path in models:
                for int8 in ([0, 1] if cycle % 2 == 0 else [1, 0]):
                    arm = "int8" if int8 else "metal4"
                    print(f"cycle {cycle+1}/{args.cycles} {label} {arm}", flush=True)
                    deadline = time.monotonic() + 1200
                    power = wait_for_apple_power(deadline)
                    cool = wait_for_apple_cool(dict(max_gpu_c=65, max_cpu_c=65,
                                                   fallback_seconds=45), deadline)
                    with (args.out / f"perf-{label}-{cycle}-{arm}.err").open("w") as err:
                        proc = subprocess.run([str(bench), "--gguf", path, "--seq-lens", "128,512",
                                               "--decode-n", "32", "--warmup", "16", "--repeats", "1"],
                                              env=env_for(int8), text=True, stdout=subprocess.PIPE,
                                              stderr=err, check=True)
                    for line in proc.stdout.splitlines():
                        row = json.loads(line)
                        if (row.get("metal_int8_quant_dispatches", 0) > 0) != bool(int8) or \
                           (row.get("metal_int8_gemm_dispatches", 0) > 0) != bool(int8):
                            raise RuntimeError("requested INT8 arm silently fell back")
                        row.update(model=label, arm=arm, cycle=cycle, power=power, thermal=cool)
                        raw.write(json.dumps(row) + "\n"); raw.flush(); runs.append(row)
    quality_rows = []
    original_env = dict(os.environ)
    try:
        for label, path in models:
            for int8 in (0, 1):
                arm = "int8" if int8 else "metal4"
                os.environ.update(env_for(int8))
                os.environ["GEIST_METAL_PROFILE"] = "1"
                subdir = args.out / label
                subdir.mkdir(exist_ok=True)
                print(f"quality {label} {arm}", flush=True)
                for row in quality(evaluate, Path(path), arm, subdir):
                    row.update(model=label, arm=arm); quality_rows.append(row)
                # Full-vocabulary logits on three different kinds of prompt.
                for case in (0, 3, 4):
                    prompt = subdir / f"prompt-{case}.txt"
                    prompt.write_text(CASES[case][1])
                    with (subdir / f"logits-{case}-{arm}.err").open("w") as err:
                        subprocess.run([str(dump), path, str(prompt), str(subdir / f"ids-{case}-{arm}.bin"),
                                        str(subdir / f"logits-{case}-{arm}.bin")], check=True, stderr=err)
    finally:
        os.environ.clear(); os.environ.update(original_env)
    (args.out / "quality.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in quality_rows))
    summary = dict(performance=[], quality=[], logits=[])
    for label, _ in models:
        for seq in (128, 512):
            arms = {arm: [r for r in runs if r["model"] == label and r["seq_len"] == seq and r["arm"] == arm]
                    for arm in ("metal4", "int8")}
            row = dict(model=label, seq_len=seq)
            for arm, records in arms.items():
                row[arm] = {metric: statistics.median(r[metric] for r in records)
                            for metric in ("prefill_tps", "decode_tps", "rss_mb", "backend_allocated_mb")}
            ratios = [b["prefill_tps"] / a["prefill_tps"]
                      for a, b in zip(arms["metal4"], arms["int8"])]
            row["paired_prefill_speedup"] = statistics.median(ratios)
            row["paired_prefill_speedup_range"] = [min(ratios), max(ratios)]
            summary["performance"].append(row)
        q = {arm: [r for r in quality_rows if r["model"] == label and r["arm"] == arm]
             for arm in ("metal4", "int8")}
        totals = {arm: -sum(sum(r["logprobs"]) for r in records) / sum(len(r["logprobs"]) for r in records)
                  for arm, records in q.items()}
        for a, b in zip(q["metal4"], q["int8"]):
            if a["prompt_ids"] != b["prompt_ids"] or a["continuation_ids"] != b["continuation_ids"]:
                raise RuntimeError("quality token IDs differ")
        exact = sum(a["generated_ids"] == b["generated_ids"] for a, b in zip(q["metal4"], q["int8"]))
        first = sum(a["generated_ids"][0] == b["generated_ids"][0] for a, b in zip(q["metal4"], q["int8"]))
        summary["quality"].append(dict(model=label, mean_nll=totals,
                                      continuation_tokens=sum(len(r["logprobs"]) for r in q["metal4"]),
                                      conditional_ppl_ratio=math.exp(totals["int8"]-totals["metal4"]),
                                      exact_generations=exact, first_token_matches=first, cases=len(CASES)))
        for case in (0, 3, 4):
            subdir = args.out / label
            if (subdir / f"ids-{case}-metal4.bin").read_bytes() != (subdir / f"ids-{case}-int8.bin").read_bytes():
                raise RuntimeError("logit comparison used different token IDs")
            comparison = subprocess.check_output(["python3", str(ROOT / "tools/compare_logits.py"),
                           str(subdir / f"logits-{case}-metal4.bin"), str(subdir / f"logits-{case}-int8.bin")], text=True)
            metrics = json.loads(comparison)
            if not all(math.isfinite(v) for v in metrics.values() if isinstance(v, float)):
                raise RuntimeError("non-finite logit comparison")
            summary["logits"].append(dict(model=label, case=CASES[case][0], **metrics))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
