# Backends

The engine binds kernels at load time from whatever backends are compiled in
(`BACKENDS="..."` at build time); `geist_backend_create("auto")` picks the best
one for the host. CPU backends are the product; the GPU backends are
experimental and never required.

## CPU (the product)

| Backend | ISA | Status |
| :-- | :-- | :-- |
| `cpu_neon` | ARM NEON + SDOT (Pi 5, Apple Silicon, any armv8.2+) | default on arm64 |
| `cpu_x86` | AVX-512/VNNI, runtime-dispatched over an x86-64-v3 (AVX2) baseline — one binary, no SIGILL on older CPUs | default on x86-64 (`BACKENDS="cpu_x86 cpu_scalar"`) |
| `cpu_scalar` | portable C, no SIMD | numerical reference; parity is dtype-specific (ternary W2A8 is intentionally not bit-exact to scalar W2A32) |

CI guarantees all three: NEON on arm64 runners, AVX-512/VNNI under Intel SDE
emulation (a silent downgrade fails the build), scalar everywhere. Concurrency
is TSan-gated. Details: [`CI_COVERAGE.md`](CI_COVERAGE.md).

## Metal (Apple GPU, experimental)

Build with `BACKENDS="metal cpu_neon cpu_scalar"`. Simdgroup GEMV/GEMM
kernels cover 13 GGUF dtypes (incl. the IQ4/Q3_K/IQ3_S mixed quants) and a
chunked DeltaNet prefill runs the qwen35 hybrids: the 27B decodes at
**1.41× llama.cpp Metal** while prefill is 1.12× on an M1 Max; gemma4-e2b sits
at 992 pp / 79 tg. Every PR executes the Metal device probe and the
quant linear-parity gate on a real GPU in CI; a weekly smoke generates
end-to-end on a gemma model. Ledger:
[`../benchmark/results/QWEN35.md`](../benchmark/results/QWEN35.md) (current)
and [`../benchmark/results/METAL.md`](../benchmark/results/METAL.md)
(the 2026-07 gemma program).

The GPU reads the weights from the GGUF mapping in place. What it binds stays
wired for `GEIST_METAL_KEEP_ALIVE_S` seconds after the last token (default
180), so the next request does not pay to wire it again; `0` leaves that to
macOS, which unwires about 2 s after the GPU goes idle.

## Vulkan (Linux GPU, experimental)

Build with `BACKENDS="vulkan cpu_x86 cpu_scalar"` — `libvulkan` is dlopen'd at
runtime, no link-time dependency. The first non-Apple GPU path (NVIDIA Turing
tested): quality gate passed (MMLU-200 0.520 vs 0.490 on the CPU path, 14/14
tool-calling) and decode reaches ~86 % of llama.cpp Vulkan (132.3 vs 154 t/s
tg128); prefill is the open front. Every PR executes the registry, buffer and
linear-parity tests on Mesa lavapipe in CI.

The Qwen3.5/3.6/3.8 hybrids (Gated-DeltaNet + attention) run end to end on the
device: the DeltaNet mixer (causal conv + delta rule, two dispatches), partial
RoPE, SwiGLU/gate epilogues and Q4_0 / Q4_1 / Q8_0 / Q5_K / Q6_K / TQ2_0
kernels, with logits bit-identical to `cpu_scalar` on an FP32 KV cache (the CPU
backends default to an INT8 KV cache, Vulkan to F16 — pin `GEIST_KV_INT8=0
GEIST_KV_F16=0` when comparing). The 27B Q4_0 (16 GB) needs a device that
holds it: it runs on a 21 GiB integrated GPU (RADV, `GEIST_VK_DEVICE=1`); an
11 GiB card fails the load with an out-of-memory error — there is no spill to
host memory yet. Weight matrices are read from the GGUF mmap and uploaded once
(`caps.weights_device_copy`), so a model no longer needs its size twice in
memory. Llama-family rows (interleaved RoPE) rotate on the device.

Ternary-Bonsai-2-27B (PQ2_0 + `prism.hadamard`, 7.21 GB) also runs on the device:
PQ2_0 kernels (a tensor-core GEMM and a wide-load matvec), the embedding lookup
and the blockwise Walsh-Hadamard rotation (`fused->hadamard_rotate`) are all on
the GPU. It fits an 11 GiB card; on an RTX 2080 Ti it reaches pp512 ≈ 395 t/s
(≈ 517 with `GEIST_M_MAX=128`) and tg ≈ 36 t/s — 0.93× the PrismML fork's Vulkan
prefill and 1.22× its decode on the same card (the default chunk stays 64 so the
scratch pool fits a 256 MB BAR heap). The tensor-core GEMM accumulates in f16 and
folds into f32 every 64 k like the fork's default; `GEIST_VK_PQ2_F32_ACC=1`
selects f32 accumulation. Devices whose subgroup size is not 32 (RADV) still fall
back to per-row matvecs for prefill (#471). Details and the side-by-side
profile: `benchmark/results/TERNARY.md`. Phase-by-phase lab log:
[`../benchmark/results/VULKAN.md`](../benchmark/results/VULKAN.md).

Work that leaves the GPU is counted per site: a fused op the shaders decline
(the arch then runs it on the host), a host loop over mapped memory, a host
buffer copy, and weights whose dtype or row length has no GPU kernel (Q3_K, a
large F16/BF16 matrix, a row that is not a whole number of blocks), which run
on a host row-dequant path. `GEIST_VK_VERBOSE=1` prints the counters at
destroy, and the first host-path linear prints how many weights and MiB took
that path. `GEIST_VK_STRICT=1` turns each of these into an error naming the
site (a host-path weight is refused at load), so a coverage gap fails loudly
instead of showing up only as a slowdown.

## GPU numbers at a glance

| model | platform | metric | **geistlib** | baseline |
| :-- | :-- | :-- | --: | --: |
| Qwen3.8-27B (Q4_0) | **M1 Max GPU** *(Metal)* | prefill t/s (pp512) | **104.4** | 93.1 *(llama.cpp Metal)* |
| Qwen3.8-27B (Q4_0) | **M1 Max GPU** *(Metal)* | **decode t/s (tg64)** | **11.6** | 8.2 *(llama.cpp Metal)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal)* | prefill t/s (pp512) | 992 | 1540 *(llama.cpp Metal)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal)* | decode t/s (tg64) | 79.3 | 92.8 *(llama.cpp Metal)* |
| Gemma 4 E2B-it (Q4_K_M) | **RTX 2080 Ti** *(Vulkan)* | prefill t/s (pp512) | 1150 | 4639 *(llama.cpp Vulkan)* |
| Gemma 4 E2B-it (Q4_K_M) | **RTX 2080 Ti** *(Vulkan)* | decode t/s (tg128) | 132.3 | 154 *(llama.cpp Vulkan)* |

Sub-parity rows shown too — nothing cherry-picked. CPU numbers and the frozen
methodology: [`../benchmark/README.md`](../benchmark/README.md).
