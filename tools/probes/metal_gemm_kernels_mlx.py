#!/usr/bin/env python3
"""mlx's quantized GEMM on geist's prefill shapes (#590).

    metal_gemm_kernels_mlx.py [--iters N] [--reps N] BITS:N_IN:N_OUT:ROWS:DTYPE ...

One line per shape: the median over REPS of the time of ITERS
mx.quantized_matmul calls (group size 64, transposed weight, as mlx-lm's
QuantizedLinear makes them) evaluated together, divided by ITERS. Run with
the venv's python of tools/probes/mlx_vs_geist.sh.
"""
import argparse
import statistics
import time

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("shapes", nargs="+")
    args = parser.parse_args()
    for spec in args.shapes:
        bits, n_in, n_out, rows, dtype = spec.split(":")
        bits, n_in, n_out, rows = int(bits), int(n_in), int(n_out), int(rows)
        dt = getattr(mx, dtype)
        w, scales, biases = mx.quantize(mx.random.normal((n_out, n_in)) * 0.02, 64, bits)
        scales, biases = scales.astype(dt), biases.astype(dt)
        x = mx.random.normal((1, rows, n_in)).astype(dt)
        mx.eval(w, scales, biases, x)
        samples = []
        for rep in range(-1, args.reps):
            ys = [mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64,
                                      bits=bits) for _ in range(args.iters)]
            t0 = time.perf_counter()
            mx.eval(ys)
            if rep >= 0:
                samples.append((time.perf_counter() - t0) / args.iters)
            del ys
        t = statistics.median(samples)
        print(f"mlx{bits} n_in={n_in} n_out={n_out} m={rows} {dtype} {t * 1e3:.3f} ms "
              f"{2.0 * rows * n_in * n_out / t * 1e-12:.2f} TFLOP/s "
              f"(min {min(samples) * 1e3:.3f} max {max(samples) * 1e3:.3f})", flush=True)


if __name__ == "__main__":
    main()
