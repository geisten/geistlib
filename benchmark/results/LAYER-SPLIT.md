# Layer split across a GPU and the CPU (#49)

**Result: kept, for decode.** On the RTX 2080 Ti box, a model that does not
fit the card (Qwen3-14B Q8_0, 15.7 GB, 11 GB VRAM) decodes 1.8× faster with
26 of its 40 layers on the GPU than on the CPU alone, and 3.5× faster than
with every layer on the GPU and the weights past VRAM in host memory. For
prefill that all-GPU placement is 1.4× faster than the split: #49's
decision rule (a split beats both) holds for decode only. We keep the
option because decode is the step a user waits on; for long prompts the
all-GPU placement is the faster one. Decode matches or beats llama.cpp at
the same split.
Prefill is 2.9× behind llama.cpp, and all of that gap is llama.cpp's op
offload: with it turned off (`-nopo 1`), geistlib's split prefill is ahead.
On the M1 Max (unified memory) a split never beats Metal alone, as expected.

## What the split is

`geist_session_opts.split_backend` / `split_layers` (EXPERIMENTAL): layers
0..N-1 run on `split_backend`, the embedding, the remaining layers and the
head on the load backend. Placement is fixed at load. Each part loads and
caches only its own layers; one hidden-state transfer per step and
direction crosses the boundary through host memory. Design and limits:
[ARCHITECTURE.md](../../docs/ARCHITECTURE.md#layer-split-across-two-backends).

## Correctness

| Check | Result |
| :-- | :-- |
| Two CPU backends (cpu_neon + cpu_neon, M1 Max): SmolLM2-360M, Qwen3-0.6B, BitNet b1.58 2B-4T, Llama 3.2 3B; split at 1, L/2, L-1, L; FP32 / INT8 / INT4 KV; prefill across two chunks, 16 greedy tokens, reset and prefill | logits and tokens **byte-identical** to the unsplit model, 48 of 48 (`test_layer_split_int`); also under ASan |
| cpu_neon + Metal, same models | max \|d\| of the logits within what Metal alone shows against the CPU (`gpu only` row) at every split point |
| cpu_x86 + Vulkan (RTX 2080 Ti), Qwen3-14B Q8_0 | same: within the Vulkan-only distance; greedy tokens 16/16 with FP32 and INT8 KV |
| Faults: one backend's map / upload / download fails after K calls, K swept until the prefill completes, on either side | every faulted prefill returns an error status and tears down, release and ASan/UBSan (`test_layer_split_fault_int`) |
| Gemma 4 E2B (per-layer embeddings) | refused at load, GEIST_E_UNSUPPORTED |

GPU kernels are not bit-exact with the CPU by design, so the GPU rows are a
comparison against the same model entirely on the GPU, not a tolerance gate.
Raw lines: `raw/layer-split-2026-10-17/*correctness*`.

## RTX 2080 Ti box: a model that does not fit

- Box: RTX 2080 Ti 11 GB (driver 595.91.07), Ryzen 9 9950X, geist-vulkan
  runner, under `scripts/gpu-lock.sh`.
- geistlib `c1c539e1` (`TARGET=linux BACKENDS="vulkan cpu_x86 cpu_scalar"
  GEMM_PROVIDER=native`), CPU side `cpu_x86`, GPU side `vulkan`.
- llama.cpp `d81235049384534c167caea52b85a694f6103d14`, Vulkan build
  (`scripts/ensure-llama-bench.sh vulkan`), `llama-bench -ngl N`; its N
  layers are counted from the end, the head stays on the CPU for N <= 40.
- Model: Qwen/Qwen3-14B-GGUF @ `530227a7`, `Qwen3-14B-Q8_0.gguf`, sha256
  `a0dfe649137410b7d82f06a209240508e218f32f5b6fd81b69d6932160cfcd9d`.
- Prompt 512, decode 64; 3 interleaved cycles × 3 repeats; median and
  min-max over cycles; peak RSS from `/usr/bin/time`, peak VRAM from
  `nvidia-smi` polled every 0.2 s. `tools/bench_layer_split.py`, workflow
  `vulkan-split.yml`, Gitea run 1305.

| configuration | prefill tok/s | decode tok/s | peak RSS MB | peak VRAM MB |
| :-- | --: | --: | --: | --: |
| geistlib CPU (cpu_x86) | 88.1 (81.9-88.3) | 5.18 (5.17-5.18) | 14401 | 365 |
| geistlib split 20 | 133.5 (132.3-133.7) | 7.80 (7.80-7.82) | 7895 | 7409 |
| geistlib split 26 | 168.0 (163.8-168.0) | 9.62 (9.61-9.63) | 5888 | 9432 |
| llama.cpp -ngl 0 | 245.8 (245.8-246.6) | 4.12 (4.12-4.12) | 15369 | 1518 |
| llama.cpp -ngl 20 | 392.1 (392.0-392.7) | 7.20 (7.18-7.21) | 15947 | 7918 |
| llama.cpp -ngl 26 | 485.2 (484.9-486.3) | 9.21 (9.18-9.21) | 15947 | 9939 |
| llama.cpp -ngl 0 -nopo 1 | 64.3 (64.3-64.5) | 5.12 (5.12-5.13) | 15585 | 377 |
| llama.cpp -ngl 20 -nopo 1 | 106.5 (106.3-107.8) | 7.38 (7.37-7.38) | 15947 | 7898 |
| llama.cpp -ngl 26 -nopo 1 | 144.5 (142.3-144.8) | 9.51 (9.48-9.53) | 15947 | 9919 |

Run 1303 (geistlib `c9cab107`, before the region fix below) gave the same
numbers within 1 %: on Vulkan the fix changes nothing measurable.

**Every layer on the GPU.** Vulkan places the weights that do not fit VRAM
in host memory and reads them over the bus (#466). Run 1310 (geistlib
`5cad951e`, same box and protocol, geistlib only) measured it beside the
CPU and the splits. The box ran slower than in run 1305 (CPU prefill 55.6
against 88.1 tok/s); configurations are interleaved within a run, so
compare within this table, not across the two:

| configuration | prefill tok/s | decode tok/s | peak RSS MB | peak VRAM MB |
| :-- | --: | --: | --: | --: |
| geistlib CPU (cpu_x86) | 55.6 (51.7-58.0) | 4.74 (4.05-4.95) | 14402 | 365 |
| geistlib every layer on Vulkan | 170.4 (166.3-175.6) | 2.49 (2.49-2.49) | 5880 | 10520 |
| geistlib split 20 | 92.9 (92.0-95.7) | 7.22 (7.00-7.31) | 7897 | 7409 |
| geistlib split 26 | 121.2 (117.4-121.9) | 8.76 (8.51-8.84) | 5889 | 9432 |

Decode reads every weight once per token, so the weights in host memory
cost a full PCIe read each step: 2.49 tok/s. A prefill chunk reuses each
weight for every row, so streaming them to the GPU pays, as with
llama.cpp's op offload.

**Observations.**

- Split 26 against CPU-only: prefill 168.0 vs 88.1 tok/s, decode 9.62 vs
  5.18 tok/s, min-max ranges far apart.
- Decode: geistlib ahead of llama.cpp at both splits (9.62 vs 9.21 at 26,
  7.80 vs 7.20 at 20) and on the CPU (5.18 vs 4.12).
- Prefill: llama.cpp 2.9× ahead at 26 layers, and 3.8× ahead even with no
  layer on the GPU (`-ngl 0`: 245.8 vs 88.1). With `-nopo 1` llama.cpp
  drops to 64.3 / 106.5 / 144.5 tok/s, below geistlib at every split.
- Peak RSS: llama.cpp ~15.9 GB at every split, geistlib 7.9 / 5.9 GB.
  Both map the GGUF; RSS counts the file pages a process touched, which
  the OS can drop, so this is not a memory requirement. Not investigated:
  presumably llama.cpp touches the pages of its GPU layers when it loads
  them and geistlib's per-tensor device copy lets them go.

**Interpretation.** llama.cpp's op offload (its graph scheduler) streams
the weights of the CPU layers over PCIe and runs their prefill matmuls on
the GPU when the batch is large enough. That is the whole prefill gap; a
layer split placed at load cannot do it, by design (#49 non-goals). Decode
reads every weight once per token, so streaming does not pay there, and
geistlib's split is at least as fast as llama.cpp's.

## M1 Max: unified memory

Llama 3.2 3B Q4_K_M (`llama-3.2-3b-Q4_K_M.gguf`), cpu_neon and Metal,
prompt 512, decode 64, 3 cycles × 3 repeats, under `with-mac-lock`.

| configuration | prefill tok/s | decode tok/s |
| :-- | --: | --: |
| CPU (cpu_neon) | 104.5 (104.4-107.3) | 35.66 (35.57-36.09) |
| Metal, every layer | 879.9 (879.6-917.9) | 81.67 (81.19-83.75) |
| split 7 | 131.7 (130.6-133.3) | 37.30 (36.75-37.32) |
| split 14 | 182.8 (182.4-183.2) | 39.88 (39.51-39.95) |
| split 21 | 293.1 (293.0-294.6) | 47.37 (46.92-47.60) |

A split beats the CPU and never Metal alone; on unified memory the whole
model fits the GPU, so a split has no use there.

**Fixed on the way.** The first measurement had the split stage run its
layers outside its backend's region (`parallel_region_begin`): Metal then
submitted and waited per op, and decode fell the more layers ran on the GPU
(17.9 / 12.1 / 9.1 tok/s at 7 / 14 / 21). The stage now opens its
backend's region for the same phase (`c1c539e1`).
Raw: `raw/layer-split-2026-10-17/m1max-*`.

## Single-backend runs pay nothing

`tools/bench_revision_ab.py`, parent `d4dc20fa` against the branch, plus a
control run of the parent's binary:

- Pi 5 (pi5-ab.yml, run 1304, synthetic Llama 1B Q4_K, 8 cycles): every
  prefill and decode row within noise at a resolution of ±2.0 %.
- M1 Max (Llama 3.2 3B Q4_K_M, 8 cycles, cpu_neon and Metal): every row
  within noise; medians within 2 %, but the run's resolution was coarse
  (±12.5 % CPU, ±48 % Metal).

Raw: `raw/layer-split-2026-10-17/pi5-ab-run1304.md`, `m1max-ab.txt`.

## Limits

- Refused: per-layer embeddings (Gemma 4), shared KV, DeltaNet layers
  (Qwen3.5 to 3.8, Bonsai), MTP blocks, a rotated basis, AWQ; KIVI sessions
  and snapshots on a split model.
- Two backends, a split by layer, placement at load. No op-level
  scheduling, no automatic placement, no multi-GPU.
- The hidden rows cross through host memory, also on unified memory.
