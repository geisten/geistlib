#!/usr/bin/env python3
"""geist_bench.py — apples-to-apples speed benchmark vs llama-bench.

Measures prefill (pp512) and text-generation (tg128) tokens/sec for a given
GGUF, and runs the equivalent llama-bench for comparison.

Run:
    python geist_bench.py gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
"""
import argparse, subprocess, tempfile, time
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
TB_GREEDY = ROOT_DIR / "test_greedy_kv_gguf"

def bench_geist(gguf_path, n_prompt, n_decode):
    """Return (prefill_tps, decode_tps). Hijacks test_greedy_kv_gguf:
       Run with prompt of length n_prompt, generate n_decode tokens.
       Total time T includes prefill + decode + model load.
       To separate them, we run twice: once with n_decode=0 (prefill only,
       almost), once with n_decode>0. Difference yields decode time."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Random prompt token IDs (we don't care about content, just speed)
        ids = np.random.randint(low=2000, high=200000, size=n_prompt, dtype=np.int32)
        ip = tmp / "i.bin"; ip.write_bytes(ids.tobytes())

        # Run 1: prefill + 1 decode (we always need at least 1 decode for the binary to exit)
        op1 = tmp / "o1.bin"
        t0 = time.time()
        subprocess.run([str(TB_GREEDY), str(gguf_path), str(ip), "1", str(op1)],
                       capture_output=True, check=True, timeout=600)
        t_prefill_1 = time.time() - t0

        # Run 2: prefill + n_decode tokens
        op2 = tmp / "o2.bin"
        t0 = time.time()
        subprocess.run([str(TB_GREEDY), str(gguf_path), str(ip), str(n_decode), str(op2)],
                       capture_output=True, check=True, timeout=1200)
        t_total = time.time() - t0

    # Decode-only time = total - prefill_overhead (which includes load + 1 decode)
    # The 1-decode run = load + prefill + 1 decode_step
    # The N-decode run = load + prefill + N decode_steps
    # Difference = (N-1) decode_steps
    t_decode_per_step = (t_total - t_prefill_1) / max(n_decode - 1, 1)
    decode_tps = 1.0 / t_decode_per_step if t_decode_per_step > 0 else 0.0

    # Prefill: t_prefill_1 includes load (significant) + n_prompt-prefill + 1 decode
    # Run a model-load-only benchmark to subtract — easier: assume model load
    # takes ~constant time, estimate by running with n_prompt=1.
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ids1 = np.array([2], dtype=np.int32); ip = tmp / "i.bin"; ip.write_bytes(ids1.tobytes())
        op = tmp / "o.bin"
        subprocess.run([str(TB_GREEDY), str(gguf_path), str(ip), "1", str(op)],
                       capture_output=True, check=True, timeout=600)
    t_baseline = time.time() - t0
    t_prefill_only = t_prefill_1 - t_baseline   # n_prompt-1 extra tokens prefilled
    prefill_tps = (n_prompt - 1) / t_prefill_only if t_prefill_only > 0 else 0.0

    return prefill_tps, decode_tps, t_total


def bench_llamacpp(gguf_path, n_prompt, n_decode):
    """Use llama-bench for the same model."""
    out = subprocess.run(
        ["llama-bench", "-m", str(gguf_path), "-p", str(n_prompt), "-n", str(n_decode),
         "-t", "1", "-ngl", "0"],
        capture_output=True, text=True, check=True, timeout=600)
    pp = tg = None
    for line in out.stdout.splitlines():
        if "pp" in line and "+-" in line:
            parts = line.split("|")
            try:
                t = float(parts[-2].strip().split("±")[0].strip())
                pp = t
            except Exception:
                pass
        if "tg" in line and "+-" in line:
            parts = line.split("|")
            try:
                t = float(parts[-2].strip().split("±")[0].strip())
                tg = t
            except Exception:
                pass
    return pp, tg, out.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--n-prompt", type=int, default=64,
                        help="Tokens for prefill benchmark (small to fit current naive O(N²) attn)")
    parser.add_argument("--n-decode", type=int, default=8,
                        help="Tokens for decode benchmark")
    args = parser.parse_args()

    print(f"Model: {args.gguf_path}")
    print(f"Bench config: prefill={args.n_prompt} tokens, decode={args.n_decode} tokens, threads=1, CPU-only")
    print()

    print("=== geist (our implementation) ===")
    prefill_tps, decode_tps, t_total = bench_geist(args.gguf_path, args.n_prompt, args.n_decode)
    print(f"  prefill (pp{args.n_prompt}): {prefill_tps:>7.2f} tokens/sec")
    print(f"  decode  (tg{args.n_decode}):  {decode_tps:>7.2f} tokens/sec")
    print(f"  total run time: {t_total:.1f}s")

    print("\n=== llama.cpp (llama-bench) ===")
    pp_llama, tg_llama, raw = bench_llamacpp(args.gguf_path, args.n_prompt, args.n_decode)
    print(f"  prefill (pp{args.n_prompt}): {pp_llama:>7.2f} tokens/sec" if pp_llama else "  prefill: <parse failed>")
    print(f"  decode  (tg{args.n_decode}):  {tg_llama:>7.2f} tokens/sec" if tg_llama else "  decode: <parse failed>")

    if pp_llama and prefill_tps:
        print(f"\nRatio prefill (geist/llama): {prefill_tps/pp_llama:.2f}× — {'FASTER' if prefill_tps>pp_llama else 'slower'}")
    if tg_llama and decode_tps:
        print(f"Ratio decode  (geist/llama): {decode_tps/tg_llama:.2f}× — {'FASTER' if decode_tps>tg_llama else 'slower'}")


if __name__ == "__main__":
    main()
