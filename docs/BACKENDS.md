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

## Vulkan (Linux GPU, experimental)

Build with `BACKENDS="vulkan cpu_x86 cpu_scalar"` — `libvulkan` is dlopen'd at
runtime, no link-time dependency. The first non-Apple GPU path (NVIDIA Turing
tested): quality gate passed (MMLU-200 0.520 vs 0.490 on the CPU path, 14/14
tool-calling) and decode reaches ~86 % of llama.cpp Vulkan (132.3 vs 154 t/s
tg128); prefill is the open front. Every PR executes the registry, buffer and
linear-parity tests on Mesa lavapipe in CI. Phase-by-phase lab log:
[`../benchmark/results/VULKAN.md`](../benchmark/results/VULKAN.md).

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
