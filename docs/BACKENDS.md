# Backends

Backends are compiled in with `BACKENDS="..."` at build time (run `make clean`
when you change the list). `geist_backend_create("auto")` takes the first
compiled backend in preference order — `cpu_neon`, `cpu_x86`, `cpu_scalar`,
then `metal`, `vulkan` — so GPU backends are selected by name or with
`GEIST_BACKEND=<name>`. The CPU backends are the product; the GPU backends are
experimental and never required.

## CPU

| Backend | ISA | Status |
| :-- | :-- | :-- |
| `cpu_neon` | ARM NEON + SDOT (Pi 5, Apple Silicon, any armv8.2+ with dot product) | default on arm64 |
| `cpu_x86` | x86-64-v3 (AVX2) baseline with runtime-dispatched AVX-512/VNNI: one binary, no SIGILL on older CPUs | default on x86-64 |
| `cpu_scalar` | portable C, no SIMD | numerical reference; parity is per dtype (ternary W2A8 is intentionally not bit-exact to scalar W2A32, see [ARCHITECTURE.md](ARCHITECTURE.md#where-the-oracle-stops-ternary)) |

A build for armv8.2 refuses a CPU without dot product (Cortex-A72, Pi 4) at
`geist_backend_create`, naming the missing feature. CI runs NEON on arm64,
AVX-512/VNNI under Intel SDE (a silent downgrade fails), scalar everywhere,
and TSan on concurrency: [CI_COVERAGE.md](CI_COVERAGE.md).

## Metal (Apple GPU, experimental)

Build with `BACKENDS="metal cpu_neon cpu_scalar"`. Simdgroup GEMV/GEMM kernels
cover 13 GGUF dtypes (incl. the IQ4/Q3_K/IQ3_S mixed quants), and a chunked
DeltaNet prefill runs the qwen35 hybrids. On an M1 Max the Qwen3.8-27B decodes
at **1.41× llama.cpp Metal** and prefills at 1.12×; gemma4-e2b reaches 992 pp /
79 tg. CI runs the device probe and quant linear-parity gate on a real GPU for
every PR, and a weekly smoke generates end to end with Gemma. Ledgers:
[`QWEN35.md`](../benchmark/results/QWEN35.md) (current) and
[`METAL.md`](../benchmark/results/METAL.md) (Gemma).

The GPU reads weights from the GGUF mapping in place. What it binds stays
wired for `GEIST_METAL_KEEP_ALIVE_S` seconds after the last token (default
180), so the next request does not pay to wire it again; `0` leaves it to
macOS, which unwires about 2 s after the GPU goes idle.

## Vulkan (Linux GPU, experimental)

Build with `BACKENDS="vulkan cpu_x86 cpu_scalar"`; `libvulkan` is loaded at run
time (no link-time dependency). Tested on NVIDIA Turing: MMLU-200 0.520 vs
0.490 on the CPU path, 14/14 tool-calling, decode ~86 % of llama.cpp Vulkan
(132.3 vs 154 t/s tg128); prefill is the open front. CI runs the registry,
buffer and linear-parity tests on Mesa lavapipe for every PR. Lab log:
[`VULKAN.md`](../benchmark/results/VULKAN.md).

**Models.** The Qwen3.5/3.6/3.8 hybrids run end to end on the device (DeltaNet
mixer, partial RoPE, SwiGLU/gate epilogues, Q4_0 / Q4_1 / Q8_0 / Q5_K / Q6_K /
TQ2_0 kernels), with logits bit-identical to `cpu_scalar` on an FP32 KV cache.
Llama-family RoPE also runs on the device. Ternary-Bonsai-2-27B (PQ2_0 +
`prism.hadamard`, 7.21 GB) runs fully on the GPU and fits an 11 GiB card: on an
RTX 2080 Ti pp512 ≈ 395 t/s (≈ 517 with `GEIST_M_MAX=128`) and tg ≈ 36 t/s,
0.93× the PrismML fork's prefill and 1.22× its decode
([`TERNARY.md`](../benchmark/results/TERNARY.md)). Devices whose subgroup size
is not 32 (RADV) still prefill PQ2_0 with per-row matvecs (#471).

**Memory.** Weight matrices are read from the GGUF mapping and uploaded once.
There is no spill to host memory (#466): a model must fit the device. The 27B
Q4_0 (16 GB) runs on a 21 GiB integrated GPU (RADV, `GEIST_VK_DEVICE=1`); on an
11 GiB card the load fails with an out-of-memory error naming the failing
allocation, the memory in use and the device limit.

**Scratch placement.** A session's activation scratch is one host-visible
pool. It goes into the BAR window (device-local and mappable) while that has
room — 256 MB without resizable BAR — and past it into system memory, where
every GPU op on it crosses the bus (prefill drops 3×; e.g. `GEIST_M_MAX` ≥ 128
on a 27B). By default every slot the host never maps goes into plain
device-local VRAM instead; only `h_a`, `h_b` and the logits rows stay
host-visible (`GEIST_VK_SCRATCH_DEVICE=0` turns this off). The arch does this
only for sessions with no host loop over those slots (dense FP32/F16 KV, PLE
only with the on-device row lookup, no DeltaNet, attention output gate, MTP,
SubLN/projection norms or AWQ scales; a `prism.hadamard` rotation is fine);
the others keep the host-visible pool, and their default chunk shrinks until
it fits the BAR window.
Under it, a CPU fallback that would read a device-local slot fails with
`GEIST_E_BACKEND` instead, so a weight dtype without a Vulkan kernel fails its
first prefill.

**Errors.** A failed submit (device lost, out of memory) is reported at the
next host access: `buffer_map` returns nullptr with `GEIST_E_BACKEND`, and
argmax, downloads and host views return `GEIST_E_BACKEND`. Sizes or offsets
beyond the shaders' 32-bit indices return `GEIST_E_INVALID_ARG`, and a weight
whose `n_in` would not fit the 192 MB activation ring at the 512-row batch
limit fails the load.

| Variable | Effect |
| :-- | :-- |
| `GEIST_VK_DEVICE=<index>` | pick a device (default: first discrete GPU, else device 0) |
| `GEIST_VK_VRAM_BUDGET` | lower the device memory limit (bytes, K/M/G suffix) to reproduce a smaller card |
| `GEIST_VK_PIPELINE_CACHE` | compiled-pipeline cache file (default `$XDG_CACHE_HOME/geist` or `~/.cache/geist`, one file per driver build); `0` turns it off. The NVIDIA driver's own cache is per executable, so without it every new binary spends ~2 s compiling pipelines |
| `GEIST_M_MAX` | prefill chunk rows (Vulkan default 128) |
| `GEIST_VK_PQ2_F32_ACC=1` | f32 instead of f16 accumulation in the PQ2_0 tensor-core GEMM (default folds into f32 every 64 k) |
| `GEIST_VK_SCRATCH_DEVICE=0` | keep the scratch pool host-visible (default: device-local where the arch allows it), see above |
| `GEIST_VK_VERBOSE=1` | print scratch placement and, at destroy, counters for work that left the GPU (declined fused ops, host loops, host copies, host-path weights) |
| `GEIST_VK_STRICT=1` | turn each of those host fallbacks into an error naming the site; a host-path weight is refused at load |
| `GEIST_KV_INT8=0 GEIST_KV_F16=0` | FP32 KV cache, for comparing against `cpu_scalar` (CPU defaults to INT8, Vulkan to F16) |

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
