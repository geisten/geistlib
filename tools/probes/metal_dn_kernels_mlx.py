#!/usr/bin/env python3
"""mlx-lm's gated delta kernel on Qwen3.5's DeltaNet shapes (#590).

    metal_dn_kernels_mlx.py [--iters N] [--reps N] HK:HV:DK:DV:T ...

ITERS calls chained through the state (as the layers of a prefill are not,
but one evaluation holds them all), median over REPS, ms per call.
"""
import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("shapes", nargs="+")
    args = parser.parse_args()
    for spec in args.shapes:
        hk, hv, dk, dv, t = (int(x) for x in spec.split(":"))
        dt = mx.bfloat16
        q = (mx.random.normal((1, t, hk, dk)) * 0.1).astype(dt)
        k = (mx.random.normal((1, t, hk, dk)) * 0.1).astype(dt)
        v = mx.random.normal((1, t, hv, dv)).astype(dt)
        g = mx.exp(-mx.abs(mx.random.normal((1, t, hv)))).astype(mx.float32)
        beta = mx.sigmoid(mx.random.normal((1, t, hv))).astype(mx.float32)
        state0 = mx.zeros((1, hv, dv, dk), dtype=mx.float32)
        mx.eval(q, k, v, g, beta, state0)
        samples = []
        for rep in range(-1, args.reps):
            ys, state = [], state0
            for _ in range(args.iters):
                y, state = gated_delta_kernel(q, k, v, g, beta, state)
                ys.append(y)
            t0 = time.perf_counter()
            mx.eval(ys, state)
            if rep >= 0:
                samples.append((time.perf_counter() - t0) / args.iters)
        m = statistics.median(samples)
        print(f"mlx-gdn heads={hk}/{hv} d={dk}/{dv} m={t} {m * 1e3:.3f} ms/call "
              f"(min {min(samples) * 1e3:.3f} max {max(samples) * 1e3:.3f})", flush=True)


if __name__ == "__main__":
    main()
