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
| prefill pp128 | **884** | 679 | **+30 %** |
| decode  tg128 | **77.9** | 56.5 | **+38 %** |

The AVX-512 code path is bit-identical to the scalar oracle (Δ=0); the spec-head
greedy output is byte-identical to the exact F16 dense head.

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
