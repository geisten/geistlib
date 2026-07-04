# Vulkan backend — baseline & targets (RTX 2080 Ti)

Phase-0 snapshot (2026-07-04): the numbers to beat, measured before any geist
Vulkan code exists. Protocol: pp512 + tg128, E2E total = 640 tok / (t_pp + t_tg).

## Rig

- GPU: NVIDIA RTX 2080 Ti 11 GB (Turing, sm_75), driver 595.71.05, Vulkan 1.4
  - reports: fp16 ✓, KHR_coopmat ✓, warp 32 (ggml prints `int dot: 0` — its
    accelerated-int-dot probe fails on this driver; DP4A still worth testing in ours)
- CPU: AMD Ryzen 9 9950X (16C/32T) — CPU runs use `OMP_NUM_THREADS=16`
  (SMT costs ~40% decode; 32-thread runs are invalid)
- llama.cpp pinned `d0f9d2e` (2026-06-22):
  - `build-vulkan/`: `-DGGML_VULKAN=ON` (needs SPIRV-Headers; installed to `~/.local`)
  - `build-cuda/`: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` (reference only)
- llama-bench: `-p 512 -n 128 -r 8`, `GGML_VK_VISIBLE_DEVICES=0` (else it also
  enumerates the Raphael iGPU)
- geist @ main 5ae89d5, `make BACKENDS="cpu_x86 cpu_scalar"`, bench via
  `tests/bench_session_throughput`

## Gemma 4 E2B-it Q4_K_M (2.88 GiB) — the H2H

| engine | pp512 tok/s | tg128 tok/s | E2E tok/s |
| :-- | --: | --: | --: |
| llama.cpp Vulkan (the bar) | 4682.6 ± 39.7 | 152.6 ± 1.2 | 675 |
| llama.cpp CUDA (reference) | 6697.3 ± 85.7 | 181.8 ± 1.3 | 820 |
| geist cpu_x86 (16T) | 484.2 | 42.2 | 156 |
| **geist vulkan — target ≥ 1.10×** | | | **≥ 743** |

Decode dominates E2E (~88% of wall time). At pp parity the target needs
tg ≥ 170 tok/s — i.e. ~85% of the 2080 Ti's 616 GB/s reading a 3.1 GB model
per token. llama.cpp Vulkan sits at ~50% bandwidth efficiency; that's the room.

## BitNet b1.58 2B-4T I2_S — PR-2 target

llama.cpp cannot run I2_S at all (no ggml type, no Vulkan ternary kernels),
so the bar is geist's own CPU path + bitnet.cpp:

| engine | pp512 tok/s | tg128 tok/s | E2E tok/s |
| :-- | --: | --: | --: |
| geist cpu_x86 (16T, the bar) | 909.3 | 71.1 | 271 |
| bitnet.cpp (prior H2H, same rig) | ~679 | ~56.5 | — |

## Reproduce

```sh
GGML_VK_VISIBLE_DEVICES=0 ~/llama.cpp/build-vulkan/bin/llama-bench \
  -m ~/models/gemma/gemma-4-E2B-it-Q4_K_M.gguf -p 512 -n 128 -r 8
OMP_NUM_THREADS=16 GEIST_GGUF_PATH=~/models/gemma/gemma-4-E2B-it-Q4_K_M.gguf \
  GEIST_BENCH_BACKEND=cpu_x86 GEIST_BENCH_PP=512 GEIST_BENCH_TG=128 \
  bin/linux/release/tests/bench_session_throughput
```

## Progress log

| stage | pp512 | tg128 | E2E | notes |
| :-- | --: | --: | --: | :-- |
| Phase 2 — correctness (sync per-linear dispatch) | 27.3 | 2.2 | 8.2 | Q4_K/Q6_K/F32 linears on GPU, parity bit-exact; everything else CPU; one submit+fence per linear |
