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
11 GiB card fails the load with an out-of-memory error that names the MiB the
failing allocation needs, the MiB in use and the device limit — there is no
spill to host memory yet (#466). `GEIST_VK_VRAM_BUDGET` (bytes, K/M/G suffix)
lowers that limit, to reproduce a smaller card. Weight matrices are read from the GGUF mmap and uploaded once
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

A batch whose submit fails (device lost, out of memory) is reported at the
next host access: `buffer_map` returns nullptr with `GEIST_E_BACKEND` as the
backend error, and argmax, downloads and host views return `GEIST_E_BACKEND`.
Sizes and offsets that do not fit the shaders' 32-bit indices make the op
return `GEIST_E_INVALID_ARG`, and a weight whose `n_in` would not fit the
192 MB activation ring at the 512-row batch limit fails the load.

## Resident memory per backend

What stays in RAM once a model is loaded: the weights in one or more layouts,
plus what each session allocates. Measure with the process's RSS; on Metal,
`geist_backend_resources_snapshot` reports the GPU side (do not add the two).

**How the weights are held.** `geist_model_load(path)` maps the GGUF read-only
and demand-pages it. `geist_model_load_from_memory` aliases the caller's bytes
the same way, never copying or freeing them; those pages are the caller's and
count once whatever the backend does. On top of that:

| Backend | Weights | Extra copies at load | Knobs |
| :-- | :-- | :-- | :-- |
| `cpu_scalar` | read in place from the mapping | none | — |
| `cpu_x86` | in place, except the layouts on the right | **Q4_K** → Q4_Kx8 (+1× Q4_K bytes) only where the AVX-512 prefill panels run; **Q6_K** → W8A8 (≈ +1.8× Q6_K bytes) only with AVX-512 VNNI; **I2_S** x4 + t5 blobs (≈ +0.45 B/weight) with VNNI; a tied F16 `lm_head` → int8 rows (≈ +1 B/weight); F32 matrices → W8A8 (+1.5 or +3 B/weight) | `GEIST_Q4K_RAW=1`, `GEIST_Q6K_RAW=1`, `GEIST_I2S_T5=0`, `GEIST_Q8_LMHEAD=0` keep the source layout (slower prefill) |
| `cpu_neon` | in place, except the layouts on the right | Q4_0 and PQ2_0 x8 GEMV panels (≈ +1×, any SDOT core); on Apple also Q4_K predecode (≈ +2×), Q6_K x8 for the vocabulary head and Q6_K ntile4 for FFN-down | `GEIST_Q4_0_X8_GEMV=0`, `GEIST_PQ2_0_X8_GEMV=0`, `GEIST_Q4K_PREDECODE=0`, `GEIST_Q6K_X8_GEMV=0`, `GEIST_Q6K_NTILE_PREFILL=0` |
| `metal` | wrapped in place (`newBufferWithBytesNoCopy`) when the range is file-backed or read-only | a **writable** `load_from_memory` buffer is copied into a Metal buffer (2×); `mprotect` it read-only to avoid that. Lookup-only tables (PLE, untied `token_embd`) stay on the host | `GEIST_METAL_KEEP_ALIVE_S` |
| `vulkan` | uploaded once to device memory | small tensors (< 1 MiB) are also copied into a host arena; the mapping pages behind uploaded matrices are released after upload (Linux, `load(path)` only); Llama `attn_q`/`attn_k` keep a host copy for the row permutation | `GEIST_WEIGHT_MMAP`, `GEIST_VK_VRAM_BUDGET` |

The repacked CPU layouts replace the source in the hot path, but the source
pages were read once while repacking. With `load(path)` they are clean file
pages the kernel can evict under pressure, so RSS shows them until it does;
with `load_from_memory` they belong to the caller and stay. All backends also
widen non-F32 norm gammas and small F16/BF16 matrices to F32, and Gemma's
`per_layer_model_proj` is dequantized to F32 (≈ 52 MB).

`GEIST_WEIGHT_MMAP=0` copies every tensor into one backend arena and closes
the mapping instead (the Vulkan default); on the CPU that is a full heap copy.

**Per session.** Each session holds its KV cache, sized from its own
`max_seq_len` (S) and allocated in full at create: per attention layer with
E = S × n_kv_heads × head_dim it is 8·E bytes in FP32, 4·E in F16, about
2·E in INT8 (plus 8·S·n_kv_heads of scales), about E in INT4 and about E/2 in
KIVI. Shared-KV and DeltaNet layers hold none. The CPU backends default to
INT8 (FP32 on Apple), Metal and Vulkan to F16. The model's default session is
sized from the model cap (`geist_model_load_with_opts`, 4096 without options).
The scratch pool grows with the prefill chunk (`GEIST_M_MAX`; 64 on the CPU,
256 on Metal, 128 on Vulkan) and the vocabulary.

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
