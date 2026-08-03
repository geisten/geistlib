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
| `cpu_scalar` | portable C, no SIMD | reference — every SIMD kernel is checked bit-exact against it |

CI guarantees all three: NEON on arm64 runners, AVX-512/VNNI under Intel SDE
emulation (a silent downgrade fails the build), scalar everywhere. Concurrency
is TSan-gated. Details: [`CI_COVERAGE.md`](CI_COVERAGE.md).

## Metal (Apple GPU, experimental)

Build with `BACKENDS="metal cpu_neon cpu_scalar"`. Greedy decode is bit-exact
vs the `cpu_scalar` reference and within ~12 % of llama.cpp Metal (81.2 vs
91.3 t/s decode, Gemma E2B on an M1 Max), holding up at long context. Every PR
executes the Metal device probe and a Q4_K/Q6_K/F32 linear-parity gate on a
real GPU in CI. Kernel notes and the measurement ledger:
[`../benchmark/results/METAL.md`](../benchmark/results/METAL.md).

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
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal)* | prefill t/s (pp512) | 987 | 1542 *(llama.cpp Metal)* |
| Gemma 4 E2B-it (Q4_K_M) | **M1 Max GPU** *(Metal)* | decode t/s (tg64) | 81.2 | 91.3 *(llama.cpp Metal)* |
| Gemma 4 E2B-it (Q4_K_M) | **RTX 2080 Ti** *(Vulkan)* | prefill t/s (pp512) | 1150 | 4639 *(llama.cpp Vulkan)* |
| Gemma 4 E2B-it (Q4_K_M) | **RTX 2080 Ti** *(Vulkan)* | decode t/s (tg128) | 132.3 | 154 *(llama.cpp Vulkan)* |

Sub-parity rows shown too — nothing cherry-picked. CPU numbers and the frozen
methodology: [`../benchmark/README.md`](../benchmark/README.md).
