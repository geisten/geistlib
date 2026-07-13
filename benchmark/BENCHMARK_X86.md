# geist Benchmarks — AMD x86-64 (AVX-512)

geist's native `cpu_x86` backend (AVX-512 / VNNI, runtime-dispatched via `hw_probe`)
vs llama.cpp and Microsoft's bitnet.cpp, on the reference desktop.

- **Machine:** AMD Ryzen 9 9950X (Zen 5, 16C/32T, DDR5-6400), Linux.
- **Threads:** `OMP_NUM_THREADS=16 OMP_WAIT_POLICY=active`, same GGUF for both engines.
- **Baselines:** llama.cpp `b9827` (CPU backend, built from master) · bitnet.cpp =
  [microsoft/BitNet](https://github.com/microsoft/BitNet) `master` (its bundled
  llama.cpp fork). See [`compare_ternary_pi5.sh`](compare_ternary_pi5.sh) for the
  pinned reference-build harness (`LLAMA_REF` / `BITNET_REF`).

## Standard quants vs llama.cpp

| model | metric | geist | llama.cpp | result |
| :-- | :-- | --: | --: | :-- |
| Gemma 4 E2B-it Q4_K_M | prefill | **512** | 495 | **+3.4 %** |
| Gemma 4 E2B-it Q4_K_M | decode  | **48.6** | 44.1 | **+10 %** |
| Llama 3.2 3B Q4_K_M   | prefill | **351** | 346 | +1.4 % |
| Llama 3.2 3B Q4_K_M   | decode  | 34.1 | 34.5 | parity |

geist matches-to-beats llama.cpp on x86 across both models and both phases.

## Ternary BitNet vs bitnet.cpp

BitNet b1.58 2B-4T (`i2_s`), biased-u8 VPDPBUSD ternary GEMV/GEMM + an i8 spec-head
for the F16 tied lm_head:

| metric | geist | bitnet.cpp | result |
| :-- | --: | --: | :-- |
| prefill pp128 | **1093** | 679 | **+61 %** |
| decode  tg128 | **94.9** | 56.5 | **+68 %** |

(geist re-measured 2026-07-13 after #102; the bitnet.cpp reference numbers are
the 2026-06 head-to-head runs — re-run `compare_ternary_pi5.sh` for a fresh
pairing.) The AVX-512 code path is bit-identical to the scalar oracle (Δ=0);
the spec-head greedy output is byte-identical to the exact F16 dense head.

### #102 optimization ledger (9950X, 16T `OMP_WAIT_POLICY=active`, mean-of-5)

| change | prefill pp128 | decode tg128 |
| :-- | --: | --: |
| baseline (main @e950433, re-measured) | 1036.7 | 91.7 |
| THP on >2 MB heap blobs (weight repacks, sketch) | 1044.7 | 92.9 |
| `GEIST_I2S_PAIR` default ON (q+k / gate+up fused decode) | — | 94.6 |
| `prefetchnta` on the x4 decode weight stream (512 B ahead) | — | +2.7 % (A/B 94.5 vs 92.0) |
| prefill token tile `I2S_X4_TT` 2 → 4 | **1093** | **94.9** |

Thread sweep: 16 T (physical cores) optimal for both phases; SMT (32 T) costs
−24 % decode. Measured **dead ends** (kept out, see #102): spec-head for top-k
sampling (stride-4 sketch rank noise beyond rank 1 is enormous — true top-40
rows ranked outside even top-8192 rough candidates; exact recall needs a
full-width phase 1 = the Q8 dense head's bytes), ReLU² block-skip in down_proj
(58 % element sparsity but **0.00 %** all-zero 64-element blocks — zeros never
cluster), `I2S_X4_TT=8` (register spills, 811 t/s).

## Reproduce

```bash
# geist (native cpu_x86 backend, opt-in on x86):
make TARGET=linux BACKENDS="cpu_x86 cpu_scalar"
OMP_NUM_THREADS=16 OMP_WAIT_POLICY=active \
  bin/linux/release/tests/bench_perf_sweep --gguf $M --seq-lens 128 --decode-n 128

# llama.cpp / bitnet.cpp reference (same GGUF, 16 threads):
llama-bench -m $M -p 128 -n 128 -t 16
```

These numbers feed the headline scoreboard — see [`headline_results.json`](headline_results.json)
(AMD 9950X rows) and [`README.md`](README.md).
