#!/usr/bin/env python3
"""Run the controlled M6 precision benchmark under the external host lock.

Examples:
  make bin/mac-omp/release/tests/bench_metal_precision
  ~/bin/with-macmini-lock python3 tools/bench_metal_precision.py --out /tmp/precision

GEMM-only timings consume already converted operands. Convert+GEMM runs
both stages. Calibrate32 additionally measures blockwise dynamic-range
analysis and packing, separately from the simple cast/pack conversion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from bench_host import command_output, sha256_file, wait_for_apple_cool, wait_for_apple_power

ROOT = Path(__file__).resolve().parents[1]
SHAPES = [(128, 4096, 32), (128, 4096, 4096), (512, 4096, 4096), (512, 8192, 8192)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=ROOT / "bin/mac-omp/release/tests/bench_metal_precision")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=32)
    args = ap.parse_args()
    if not 6 <= args.cycles <= 64 or not 1 <= args.repeats <= 256:
        ap.error("cycles must be 6..64 and repeats 1..256")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = dict(cpu=command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
                    os=command_output(["sw_vers"]), revision=command_output(["git", "rev-parse", "HEAD"]),
                    binary_sha256=sha256_file(args.binary),
                    source_sha256={name:sha256_file(ROOT/name) for name in
                                   ("tests/bench_metal_precision.c", "tests/metal_precision_shaders.h")},
                    shapes=SHAPES,
                    repeats=args.repeats, cycles=args.cycles,
                    input="LCG seed 0x54321, integers -8..7, all formats exact",
                    tile=dict(rows=128, columns=64, simdgroups=8),
                    quality="32 CPU double dot samples per format; same normalized non-integer source; fixed scale 1/16; uniform and outlier fixtures",
                    notes="cache-warm controlled benchmark; not a full-model or peak-memory measurement")
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with (args.out / "runs.jsonl").open("w") as output:
        for m, n, k in SHAPES:
            print(f"precision M={m} N={n} K={k}", flush=True)
            deadline = time.monotonic() + 1200
            power = wait_for_apple_power(deadline)["power"]
            thermal = wait_for_apple_cool(dict(max_cpu_c=65, max_gpu_c=65, fallback_seconds=45), deadline)
            with (args.out / f"{m}-{n}-{k}.err").open("w") as err:
                run = subprocess.run([str(args.binary), str(m), str(n), str(k),
                                      str(args.repeats), str(args.cycles)],
                                     stdout=subprocess.PIPE, stderr=err, text=True, check=True)
            parsed = [json.loads(line) for line in run.stdout.splitlines()]
            # Three phases for FP16; four each for FP8 and INT8.
            if len(parsed) != args.cycles * 11 + 6:
                raise RuntimeError("incomplete precision sweep")
            for row in parsed:
                row.update(power=power, thermal=thermal)
                rows.append(row)
                output.write(json.dumps(row) + "\n")
            output.flush()
    summary = []
    for shape in SHAPES:
        for mode in ("f16", "fp8", "i8"):
            point = dict(zip(("m", "n", "k"), shape), mode=mode)
            for phase in ("convert", "gemm", "combined", "calibrate32"):
                selected = [r for r in rows if (r["m"], r["n"], r["k"]) == shape
                            and r["mode"] == mode and r["phase"] == phase]
                if selected:
                    point[phase] = dict(gpu_us=statistics.median(r["gpu_us"] for r in selected),
                                        wall_us=statistics.median(r["wall_us"] for r in selected),
                                        tflops_or_tops=statistics.median(r["tflops_or_tops"] for r in selected))
                    point["operand_bytes"] = selected[0]["operand_bytes"]
                    point["output_bytes"] = selected[0]["output_bytes"]
            summary.append(point)
    (args.out / "quality.json").write_text(json.dumps([r for r in rows if r["phase"] == "quality"], indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
