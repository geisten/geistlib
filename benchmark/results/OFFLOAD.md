# Offload measurements and implementation history

These are recorded experiments on the RTX 2080 Ti / Ryzen 9 9950X and
Apple M1 Max, with their original configurations, caveats and raw-run references.
They describe the revisions measured, not current defaults or performance guarantees.
For current behavior and configuration, see [Offload](../../docs/OFFLOAD.md).

## What we already measured

**Layer split, #49** (Qwen3-14B Q8_0, 15.7 GB, RTX 2080 Ti 11 GB + Ryzen 9
9950X, `benchmark/results/LAYER-SPLIT.md`):

| configuration | prefill tok/s | decode tok/s |
| :-- | --: | --: |
| geistlib CPU only | 88.1 | 5.18 |
| geistlib split 26/40 | 168.0 | **9.62** |
| llama.cpp `-ngl 26` (op offload on) | **485.2** | 9.21 |
| llama.cpp `-ngl 26 -nopo 1` (op offload off) | 144.5 | 9.51 |

- Decode is already won by the static split. Decode reads every weight once
  per token, so streaming never pays there (every layer on Vulkan with the
  rest in host memory: 2.49 tok/s).
- Prefill is lost **only** to op offload: llama.cpp streams the CPU layers'
  weights to the GPU when the batch is big enough.
- On unified memory (M1 Max) a split never beats Metal alone. Offload has
  nothing to gain there, and the Pi 5 has no GPU backend. **The target is a
  host with a discrete GPU.**

**Expert usage, #48 phase 0** (`MOE-EXPERT-USAGE.md`, PR #51): held-out
cover of the hottest 18.3 % of experts (Strata's point) is 40 % (Granite 3B,
OLMoE) and **54 %** (Qwen3-30B-A3B, 128 experts). Small MoEs have no hot set;
a static GPU set pays only from about 128 experts per layer on.

## Phase 1: measured (RTX 2080 Ti box)

Qwen3-14B Q8_0 (15.7 GB on an 11 GB card), split 26 (cpu_x86 + Vulkan),
prompt 512, decode 32, 3 interleaved cycles, median, peak RSS, run 1396:

| configuration | prefill tok/s | decode tok/s | peak RSS |
| :-- | --: | --: | --: |
| split alone (chunk 128, cpu_x86's limit) | 125.4 | 8.57 | 6.1 GB |
| split + offload, chunk 128 | 233.0 | 9.06 | 10.8 GB |
| split + offload, chunk 256 | 373.3 | 8.99 | 10.9 GB |
| **split + offload, chunk 512** | **515.3** | **9.07** | 11.0 GB |

- Prefill 4.1× the split's at the same decode. For orientation (other
  runs, the box ran faster then: split 156 in run 1388): every layer on the
  GPU with streaming reached 550 tok/s prefill but decodes at 2.6 (run
  1366); llama.cpp `-ngl 26` with op offload 485 / 9.2 (run 1305).
- **Cost: host memory.** The twins' matrices are a second copy of the CPU
  layers' weights in host memory (GEIST_W_HOST, ≈ 5 GB here); the mmapped
  originals stay for the CPU's decode.
- **Qwen3-14B Q4_K_M, split 20** (run 1410, 2 cycles): split 199.6 / 205.7
  tok/s prefill, offload at chunk 512 570.4 / 575.8; decode 12.4–12.9 vs
  13.2; peak RSS 5.0 vs 9.1 GB.

**The host copy stays (tried, refused by the driver).** Importing the GGUF
mapping into Vulkan (`VK_EXT_external_memory_host`) instead of copying it
would have removed the second copy for the dtypes the GPU reads as stored
(Q4_K, Q5_K, Q4_1; Q8_0, Q4_0, Q6_K, PQ2_0, TQ2_0 are repacked at load and
would also need a repack on the GPU after the stream). On NVIDIA 595.91
the first import of the read-only, `MAP_PRIVATE` mapping already fails in
`vkAllocateMemory` (`VK_ERROR_OUT_OF_DEVICE_MEMORY`), and after that refusal
ordinary host allocations of the same process fail too, so the model does
not load (runs 1400, 1403, 1408); without the import it loads and runs
(run 1410). The code was removed. Left untried: sharing one host copy
between the CPU and Vulkan (the CPU reading Vulkan's cached host buffer
where both use the bytes as stored), and drivers that do import file pages.
- Correctness: two cpu_x86 instances byte-identical with an FP32 cache;
  Vulkan INT8 split 26 within the split's own distance (max|d| 0.201, 16/16
  greedy, run 1393).
- First attempt, a route per linear: no gain (run 1383, below).
- **Default and multi-turn** (run 1417, `GEIST_OFFLOAD_M_MIN` unset → 64,
  CPU steps mirrored into the twins): two cpu_x86 instances byte-identical
  including a second prompt after the decode, 4 chunks on the twins and 16
  mirrored steps per run as expected. Qwen3-14B Q8_0 split 26 offload at
  chunk 512: 507 / 525 tok/s prefill, decode 8.5 / 9.9 against the split's
  9.9 / 8.6 in the same, noisy run (the split's prefill varied 198 / 131
  between its two cycles): the mirroring's decode cost is below this run's
  resolution.

## Phase 0: measured (RTX 2080 Ti box, #78)

`tests/bench_offload_cost`, `.gitea/workflows/offload-cost.yml`, runs 1352,
1362, 1364, 1366 (Gitea Actions, geist-vulkan, under `scripts/gpu-lock.sh`).
RTX 2080 Ti on PCIe 3.0 x16, Ryzen 9 9950X, cpu_x86 with AVX-512 VNNI,
`OMP_NUM_THREADS` unset. Medians of up to 50 repeats.

**One linear, 5120 × 17408 (a Qwen3-14B FFN gate), m = 512, run 1366:**

| step | Q4_0 | Q8_0 |
| :-- | --: | --: |
| CPU (cpu_x86) | 24.1 ms | 62.3 ms |
| GPU compute, weight in VRAM | 3.9 ms | 4.4 ms |
| GPU, weight in host memory, read by the shaders (#466, "v0") | 37.1 ms | 63.1 ms |
| GPU, weight streamed into a VRAM slot, then computed ("v1", measured) | 7.7 ms | 11.8 ms |
| activations up (10.5 MB) | 0.9 ms | 0.9 ms |
| result down by DMA (35.7 MB) | 7.3 ms | 8.0 ms |
| bulk weight upload | 8.6 GB/s | 7.9 GB/s |

- **The GPU computes 6–14× faster than this CPU**; the 9950X is the strong
  side only below ~16 rows.
- **v0 is dead:** shaders reading the weight over the bus are as slow as the
  CPU (37 vs 24 ms, 63 vs 62 ms). Phase 1 skips it.
- **v1 pays from m ≈ 128 for Q8_0 and m ≈ 256–512 for Q4_0/Q4_K** on the
  shapes measured; the streamed copy (≈ 8.6 GB/s) is the bound, not the GPU.
- **Readback was the hidden cost.** Reading 35.7 MB back took 400 ms from a
  BAR-window buffer (~90 MB/s, uncached PCIe reads) and 55 ms from a plain IO
  buffer (write-combined system memory, ~650 MB/s). Host-read buffers now
  take `HOST_CACHED` memory (run 1364): the DMA readback runs at ~5 GB/s.
  The #49 split (`step.c` `run_split_stage`) reads its hidden rows back the same
  way: `transformer_buffer_xfer` maps the stage's `scratch_h_a/h_b`, which on
  Vulkan stay in the BAR window, and memcpys them. At the 24–90 MB/s measured
  that is 0.1–0.4 s per 512-row chunk of Qwen3-14B (10.5 MB), 3–13 % of a
  split-26 prefill chunk; `HOST_CACHED` does not reach BAR memory. A DMA copy
  into a cached IO buffer would fix it there.

**End to end, Qwen3-14B Q8_0 (15.7 GB, 4.7 GB of it spilled to host memory),
every layer on Vulkan, prompt 512, run 1366, tok/s:**

| prefill chunk | spill, read over the bus | **spill, streamed** (`GEIST_VK_STREAM_M_MIN=32`) | cpu_x86 |
| --: | --: | --: | --: |
| 32 | 32.4 | 63.8 | 53.0 |
| 64 | 70.3 | 123.1 | 56.9 |
| 128 | 141.8 | 240.0 | 57.0 |
| 256 | 162.6 | 393.0 | – |
| 512 | 177.2 | **549.9** | – |

Decode stays 2.61 tok/s in both (m = 1 never streams; it reads the spilled
weights over the bus); the layer split of #49 is the decode answer (9.6
tok/s at split 26). For comparison only (other run, 1305): llama.cpp
`-ngl 26` with op offload prefilled at 485 tok/s.

## Implementation and measurement history

**Phase 0: cost model — done** (results above). It also produced the
`HOST_CACHED` readback fix and the in-backend streaming (former phase 2).

**Phase 0b: the planner — built** (`plan_split_layers` in `arch_state.c`,
`GEIST_SPLIT_AUTO`, `test_split_planner_unit`; dense layers only, the
expert part is phase 3). Qwen3-14B Q8_0 at 584 positions, runs 1424 and
1426 (3 interleaved cycles each, median decode tok/s of the split alone):
the planner picks 27 of 40 layers (10438 MiB free, 1216 MiB reserved);
split 24: 7.78 / 7.96, 26: 9.19 / 9.27, **auto (27): 9.73 / 9.59**, 28: 9.49
/ 9.40. The decision rule (within the resolution of the best hand split in
decode) holds. 28 fits as well: the reserve errs on the safe side. With
offload, prefill grows with every layer the GPU holds (fewer to stream):
209–212 / 234 / 246 / 258–260 tok/s at 24 / 26 / 27 / 28, chunk 128.
**Phase 1: the route — built** (phase-dependent placement, above).
Decision rule: keep if split 26 + offload prefills faster than both the split
alone and every layer on the GPU, by more than the run's resolution, with
decode within noise of the split. At chunk 128 (run 1388): 264 against 156
(split) and 240 (every layer on the GPU), decode 10.0 against 9.9: met.
With chunks up to 512 (run 1396): 515 tok/s prefill, decode unchanged.

**Phase 2: overlap — built** (`vk_xq_copy`, `vk_weight_bind` in
`vulkan/ops.c`; on by default, `GEIST_VK_STREAM_ASYNC=0` off). The copies
run on the transfer-only queue family, one weight ahead: each streamed GEMM
is a compute submission of its own that waits on a timeline semaphore for
its slot's copy and signals another when done, and each weight learns which
one the chunk streamed after it and has it copied into the other slot
meanwhile. Before, every copy and its GEMM ran back to back on the compute
queue (the full barrier between them), so a chunk took copy time plus
compute time. Qwen3-14B Q8_0, chunk 512, runs 1517/1518/1523:
every layer on the GPU 551–553 → 654–660 tok/s, split 26 + offload
522–540 → 616–638, decode and peak VRAM unchanged (10684 / 10200 MiB both
ways). The streamed part is now bound by the copy (≈ 8.6 GB/s); more would
need the second copy engine or prefetching deeper than one weight.
Decision rule (beats single-queue streaming by more than the resolution):
met.

**Phase 3: MoE — decode built** (`moe_gpu.c`, `forward/moe.c`,
`fused->linear_t_grouped`). Qwen3-30B-A3B Q4_K_M (18.6 GB, does not fit
11 GB), RTX 2080 Ti + Ryzen 9 9950X, `moe-baseline.yml`.

> **Correction, 2026-10-22: measure on real text.** The decode numbers
> below (runs 1665-1705) are `bench_perf_sweep`'s pseudo-random prompt, as
> llama-bench's are. A mixture of experts routes such tokens unlike text:
> 40 % of the expert rows went to cold experts (run 1712), where the #48
> profile covers 88.5 % of real text's selections, and the few cold experts
> it does pick repeat every token, so their weights sit in the 9950X's
> 64 MB L3. Measured instead on a real prompt — the first 512 tokens of
> `docs/OFFLOAD.md` (`bench_perf_sweep --prompt-file`, llama-server's
> `/completion` with the same token ids), 128 decoded tokens:
>
> | | decode tok/s | prefill tok/s | runs |
> |:--|--:|--:|:--|
> | llama.cpp `--n-cpu-moe 24` (its best split that fits) | 52.7 / 52.8 | 491 / 487 | 1741 |
> | llama.cpp `--n-cpu-moe 32` | 47.3 / 47.6 | 407 / 401 | 1741 |
> | llama.cpp `--n-cpu-moe 48` (all experts on the CPU) | 37.9 / 37.9 | 294 / 301 | 1741 |
> | geist cpu_x86 | 34.8-36.0 | 236-252 | 1740-1743 |
> | geist, model on the CPU, hot experts on the GPU | 42.0-43.0 | 234-251 | 1740-1743 |
> | geist, model on the GPU | **54.5-61.2** | 164-180 | 1742, 1743, 1745, 1746 |
> | geist, model on the GPU, host busy | 2.3-2.5 | 155-171 | 1733, 1734 |
>
> **The model on the GPU decodes fastest**: 12 cycles in four runs at
> 54.5-61.2 tok/s, 3-16 % above llama.cpp's best split, the card at
> 1,980 MHz and ~41 % busy. Run 1746 is the very commit that ran at
> 2.3-2.5 in run 1733: that collapse — its ~3 cold experts per layer took
> 6-7 ms on the CPU instead of 170-200 µs, the card idled at 435 MHz — came
> from the shared host, not the code, and shows the mode's weakness: it
> needs the CPU's cores when the GPU waits for them. Its prefill (164-180
> tok/s) is below the CPU alone. `bench_nll` (11-16 tok/s) understates it:
> its own softmax over the vocabulary after every token idles the card,
> which then drops its clocks. `OPENMP=0` changes little (59.3/60.3 vs
> 57.4/59.7). **The model on the CPU with a GPU split backend** is the
> steady alternative: +19-23 % over the CPU alone in every run and both
> measures, prefill as the CPU's, perplexity 29.96 against 29.90. Prefill
> trails llama.cpp's best split by half in every geist mode — **since #163
> the model on the GPU does not**: the hot rows are gathered and summed on
> the device and a discrete GPU prefills a mixture of experts in chunks of
> 512, 490-498 tok/s (runs 1754/1755). Raw data:
> `benchmark/results/raw/moe-real-text-2026-10-22/`.

- **The model on the GPU.** A mixture-of-experts model loaded on a GPU
  backend runs attention, router, embedding and head there. Its experts are
  resolved on a CPU backend of the state's own (`moe_cpu`: cpu_x86,
  cpu_neon or cpu_scalar), and after the load the GPU gets copies of the
  hottest, in `GEIST_MOE_PROFILE` order (the #48 USAGE lines), until the
  memory the rest left is used: 2,807 of 6,144 experts (8.2 GB), 88.5 % of
  the profile's selections. `GEIST_MOE_HOT=<n>` sets the count.
- **Per layer** the host routes (it has to: the CPU experts need the ids),
  the hot experts run on the GPU as one batch on the calling thread while
  the CPU workers run the cold ones, and the host forms the weighted sum.
  Two waits per layer: the router's batch brings the logits and x (copied
  into cached host memory in that batch), the expert batch its outputs. A
  decode step's hot experts are four dispatches — gate, up and down as one
  grouped matvec each (`matvec_{f32,q4k,q6k}_id`: the single kernels with
  workgroup y picking the weight, read by device address), one silu_mul —
  and take x by device copies. The batch waits 80 µs per layer.
- **The clock was the trap.** With the model on the CPU and only the hot
  experts on the GPU (runs 1665-1676), the card saw short bursts, ~7 %
  busy, and stayed in P8 (300 MHz core, 405 MHz memory): no gain. On the
  GPU with five waits per layer it idled between them, 17 % busy, P8 again:
  3.4 tok/s (run 1679). With two waits it runs at 1.6-2.0 GHz, 36 % busy.

Run 1683 (prompt 512, 128 decoded tokens, 2 interleaved cycles):

| | decode tok/s | prefill tok/s |
|:--|--:|--:|
| geist cpu_x86 | 35.8 / 33.7 | 269 / 266 |
| **geist on Vulkan, hot experts** | **49.2 / 51.8** | 213 / 218 |
| llama.cpp Vulkan `-ot exps=CPU` | 39.5 / 39.7 | 325 / 321 |

**One wait per layer (#161).** Two waits left the card ~30 % busy, at the
driver's threshold: 49-52 tok/s after 128 tokens, 3-5 after 32 (run 1691,
P8/P5). Now the hot experts' batch starts without a wait (`fused->submit`),
the CPU runs the cold ones meanwhile, and the GPU adds the hot outputs onto
the host's cold sum (`fused->rows_weighted_sum`, the sum handed over in host
memory). Runs 1704/1705 (3 cycles): 44.9-51.8 tok/s after 32 tokens,
47.8-51.6 after 128; CPU 34.4-36.8, llama.cpp `-ot exps=CPU` 39.0-39.5;
perplexity 29.99 (CPU 29.90). The card is still only ~35 % busy (P0 most of
the time); routing on the GPU (#162) is the next margin.

**Routing on the GPU (#162).** GPU timestamps showed the card idle for
37-42 us per decode layer between the router and the hot experts (median,
5,400-6,500 boundaries per run): the host's wake-up, routing, recording and
submit. Now the GPU ranks the router's experts itself (`fused->moe_route`,
the host's order: logit descending, id ascending) and runs the experts it
holds by those ids (`fused->linear_t_sel`, a per-layer table of their
device addresses, a cold expert's slot skipped) right after the router, in
the same command buffer; the host waits for an event after the router
(`submit_mark` / `map_marked`), routes too, runs the cold experts and hands
their sum to the GPU as before. A separate submission for the GPU's part
cost ~15 us per layer at the boundary, which the event avoids. 512 decoded
tokens after a 512-token prompt, 3 interleaved cycles, `GEIST_MOE_GPU_ROUTE`
0 / 1:

| decode tok/s | host routes | GPU routes |
|:--|:--|:--|
| Qwen3-30B-A3B Q4_K_M, quiet | 67.8 / 69.2 / 70.7 | **74.0 / 73.6 / 74.4** |
| Qwen3.6-35B-A3B UD-Q4_K_XL, quiet | 81.6 / 76.3 / 82.2 | **86.8 / 87.1 / 87.8** |
| Qwen3-30B-A3B, 16 CPU hogs | 53.8 / 53.3 | 53.6 / 57.9 |
| Qwen3.6-35B-A3B, 16 CPU hogs | 61.7 / 61.2 | **70.7 / 69.4** |

Perplexity of this file (`bench_nll`, 1,023 tokens one at a time): 30.168 /
30.178 and 17.126 / 17.124.

On the pseudo-random prompt decode beats both CPU only (+45 %) and
llama.cpp `-ot exps=CPU` (+30 %); on real text see the correction above. Quality: the perplexity of `docs/OFFLOAD.md`
(1,023 tokens fed one at a time, `bench_nll`) is 29.90 on the CPU and 30.06
on the GPU; greedy tokens are a poor bar for an MoE, whose routing turns
tiny differences into other continuations. **Open: prefill** — a chunk's
experts get ~8 rows each and run one by one (~450 dispatches per layer);
a grouped GEMM is the next step. Raw data:
`benchmark/results/raw/moe-phase3-2026-10-21/`.

**Strata itself on the same box (#167).** `strata-compare.yml`: Strata
built from source (`5c8dd322`, CUDA, its own setup), llama.cpp's
llama-server (CUDA and Vulkan) and geistlib on one model Strata runs,
Qwen3.6-35B-A3B UD-Q4_K_XL (the file Strata pins), the same chat request
of 2,726 prompt tokens of this file, 128 greedy tokens, rounds alternating
the engine order, the servers' own timings. geistlib in GPU mode with the
expert profile from its own router trace (run 1830, 79.2 % cover of the
profile corpus, `tools/moe_usage/trace_usage.py`). Medians of 3 rounds per
run:

| | prefill tok/s | decode tok/s | runs |
|:--|--:|--:|:--|
| Strata, MTP off (`--mtp 0`) | 605 | 62.0 / 62.2 | 1839, 1842 |
| Strata, its MTP draft (`--mtp 2`) | 589 | 62.2 | 1842 |
| llama.cpp CUDA `--n-cpu-moe 26` (24 does not fit) | 470 | 67.8 | 1839 |
| llama.cpp CUDA `--n-cpu-moe 28` | 444 | 66.1 | 1839 |
| llama.cpp CUDA `--n-cpu-moe 40` (every expert on the CPU) | 332 | 56.2 | 1843 |
| llama.cpp Vulkan `--n-cpu-moe 24` | 364 | 52.2 | 1839 |
| llama.cpp Vulkan `--n-cpu-moe 40` | 245 | 44.1 | 1843 |
| **geist Vulkan, GPU mode, profile** | **513 / 518** | **65.9 / 68.0** | 1839, 1842 |
| geist cpu_x86 | 174 | 22.8 | 1843 |

- **Decode:** geistlib is 1.06-1.09× Strata and level with llama.cpp CUDA's
  best split (0.97-1.00×), on Vulkan against their CUDA.
- **Prefill:** 0.85× Strata, 1.1× llama.cpp CUDA, 1.4× llama.cpp Vulkan.
- Strata's MTP draft brought nothing on this request (62.2 either way).
- geistlib's decode scatters more than the others' (57.4-69.0 over six
  rounds; #161).
- The earlier estimate (Strata ~1.4× ahead in decode, from its own docs
  on other cards) is superseded.

Raw data: `benchmark/results/raw/strata-compare-2026-10-23/`.

**Prefill rows padded to 16 (#167, run 1930).** The tensor-core GEMM tiles
need a multiple of 16 rows; a hot expert's share of a chunk rarely is, so
its GEMMs ran on the scalar kernel. Padded (a repeated token, never summed),
with main's cold-expert pool (#161), 4 interleaved rounds, the same request:

| | prefill tok/s | decode tok/s |
|:--|--:|--:|
| Strata, MTP off | 605 | 62.2 |
| **geist GPU mode, profile, padded** | **655** | **71.3** |
| geist GPU mode, profile, not padded | 522 | 71.6 |

The decode collapses of runs 1921-1925 (3.3 tok/s) came on a branch
without #161's fix, at desktop loads up to 30; with it, no round fell.

**Phase 4, only if phase 3 shows the static cover as the limit:** the
adaptive tier, measured against Strata's +16.5 %.

**Batched sessions and the expert budget (#317).** `moe_fit` leaves room
for one session next to the hot experts unless the caller declares more
with `geist_session_opts.batch_sessions` (read at load). Each declared
session reserves its own buffers, and that room comes out of the hot set.
Declare only the sessions actually decoded together; 4-5 is usually a good
value. RTX 2080 Ti, 2,048 positions, aggregate decode tok/s:

| | one session, full hot set | 8 sessions, room for 9 | 8 sessions, 3,000 / 2,000 hot |
|:--|--:|--:|--:|
| Qwen3.6-35B-A3B UD-Q4_K_XL | 88.8 (3,919 hot) | 113 (1,878 hot) | 124.5 |
| Qwen3-30B-A3B Q4_K_M | 74.2 (2,834 hot) | 70.7 (1,052 hot) | 83-87 |

Qwen3-30B-A3B loses more per declared session than Qwen3.6-35B-A3B: its
experts are bigger (2.9 vs 1.8 MB) and a session's KV cache spans 48
layers, so each session pushes more experts to the CPU, whose cold-expert
work every batched row then pays (#238: CPU memory bandwidth bound).

**Decode-sized batch members (`batch_m_max`).** A session's scratch,
DeltaNet slabs and routing buffers scale with its m_max (512 by default for
an MoE on a discrete GPU). Batch members created with a small m_max, and
declared at load with `geist_session_opts.batch_m_max`, cost less, and the
budget gives the difference to the hot experts. RTX 2080 Ti, 9 sessions,
2,048 positions, aggregate decode at 8 sessions:

| members' m_max | Qwen3.6-35B-A3B hot / tok/s | Qwen3-30B-A3B hot / tok/s | prefill of a member (Qwen3.6 / Qwen3-30B) |
|:--|--:|--:|--:|
| 512 (default) | 1,840 / 108.9 | 1,605 / 70.7 | 853 / 801 tok/s |
| 64 | – / 117.5 | – / 79.7 | 432 / 391 tok/s |
| 8 | 2,785 / 121.1 | 2,141 / 83.0 | 191 / 146 tok/s |

A member prefills in chunks of its m_max, so a long prompt on it is slower.
**Recommendation:** `batch_m_max` 64 in general (most of the decode gain at
half the prefill speed); 8 when the members start from a snapshot of a
full-size session (`geist_session_restore`) and rarely prefill themselves,
e.g. #26's shared-prefix questions.

### Cold experts staged for prefill (#239)

**Why.** In GPU mode a prefill chunk's cold experts ran on the CPU, and
they were the critical path: 38 % of a 2,726-token prefill's expert rows
(394 k of 1.03 M, Qwen3.6-35B-A3B, profile run 1830) went to the CPU, and
with them skipped (timing only, outputs wrong) the prefill ran at
1,128-1,131 tok/s instead of 772-799, +43 %. The rows are not spread thin:
75 % of the cold rows belong to experts with more than 16 rows in their
chunk, 55 % to experts with more than 32.

**What.** Each cold expert gets a host copy on the GPU backend at load
(`GEIST_W_HOST`: Vulkan keeps it in pinned host memory), in the order of
the run-time profile's prefill rows when it has them (below), else of the
selection counts, in 4 GiB (or a quarter of the host's memory if less;
`GEIST_MOE_STAGE_HOST` sets it, see the budget below). A prefill step
(`blk_moe_ffn`, `seq > 1`, the model on the GPU) copies the cold experts
with at least `stage_m_min` rows, at most 16 of them, the most rows first,
into VRAM staging slots (`fused->stage_weights`) and runs them in the same
grouped, padded GEMM path as the resident experts, recorded after them;
the CPU keeps the rest. On Vulkan the step's copies are one submission on
the transfer queue that starts before the resident experts' GEMMs are
recorded and overlaps them; the first staged GEMM starts a compute
submission that waits for it on a timeline semaphore, and the next step's
copies wait on the device for the last staged reader. The host never waits
for a copy. Without a transfer queue (lavapipe) the copies go into the
batch. A second set of slots, so that one layer's copies need not wait for
the previous layer's staged GEMMs, measured 931 against 928 tok/s with
one: the next layer's routing waits for this layer anyway. Metal (unified
memory) has no `stage_weights` and nothing changes there; neither does
decode (`seq == 1`) or a batched decode.

**Threshold.** Fixed at load from a cost model measured on the model's own
experts (`moe_gpu_stage_calibrate`, `GEIST_VERBOSE` prints it): the
staging copy rate `bw`, the GPU's time per row and per expert, and the
CPU's time per row with every thread on another cold expert, from memory,
as a step's cold experts run. The CPU is the critical path and the GPU
waits for it, so a moved expert saves its CPU time and costs at most its
copy: `m_min = B / (bw · t_cpu)`, taken at the power of two at or below
(the copy also overlaps the resident GEMMs, so this is an upper bound of
the break-even). Five loads each:

| | copy | expert | GPU per row + per expert | CPU per row (32 threads) | threshold |
|:--|--:|--:|--:|--:|--:|
| Qwen3.6-35B-A3B UD-Q4_K_XL | 12.8-13.0 GB/s | 1.90 MB | 0.47-0.65 + 121-126 µs | 2.74-2.94 µs | 32 (5 of 5) |
| Qwen3-30B-A3B Q4_K_M | 12.2-12.5 GB/s | 3.06 MB | 0.51-0.67 + 102-147 µs | 3.47-3.92 µs | 64 (4 of 5), 32 |

Measured by hand against it (draft code, up to 32 experts per step, 3
cycles, prefill tok/s): Qwen3.6-35B-A3B off 795, 16 rows 904, **32 rows
925**, 64 rows 886; Qwen3-30B-A3B off 723, 16 rows 777, 32 rows 848, **64
rows 843**. The model's pick is the best measured one or within 1 % of it.
Placement is a pure function of the chunk's routing once the threshold is
fixed; a threshold near a power of two can land on either neighbour
between loads (Qwen3-30B-A3B above), as the hot set already depends on the
free VRAM at load. `GEIST_MOE_STAGE_M_MIN=<rows>` pins it, `0` or
`GEIST_MOE_STAGE=0` turns staging off.

**Host budget.** The host copies are pinned memory, so the default does
not take every cold expert: Qwen3.6-35B-A3B has 6,386 of them (11.7 GB),
Qwen3-30B-A3B 3,333 (9.1 GB). The profile's hottest cold experts come first,
and prefill tok/s grows with the budget (2 interleaved cycles each):

| `GEIST_MOE_STAGE_HOST` | off | 1 GiB | 2 GiB | 3 GiB | **4 GiB** | 6 GiB | all |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Qwen3.6-35B-A3B | 789 | 799 | 836 | 828 | **869** | 882 | 930 |
| share of the gain | | 7 % | 33 % | 28 % | **57 %** | 66 % | 100 % |
| Qwen3-30B-A3B | 709 | 744 | 785 | 819 | **825** | 835 | 836 |
| share of the gain | | 28 % | 60 % | 87 % | **91 %** | 99 % | 100 % |

The default is 4 GiB, the knee for Qwen3-30B-A3B. Qwen3.6-35B-A3B's profile
(a decode trace of another corpus) predicts which of its many small cold
experts a prefill stages less well, so it keeps 57 %; the prefill rows of
the run-time profile fix that (next section). VmRSS counts the pinned copies twice (the driver's
and geist's mapping of the same pages; `rssprobe.c` in the raw data: 2,000
MiB of host weights, MemAvailable −1,985 MiB, PSS +2,002 MiB, VmRSS +4,002
MiB), so peak RSS below grows by twice the budget while the host loses it
once.

**Host copies by the prefill profile.** The run-time profile (#165,
`moe_profile.h`) counts every prefill chunk's experts a second time, by
rows: an expert with m rows in a chunk adds m to bucket ⌊log2 m⌋ (ten
buckets, the last from 512 rows). A threshold reads off the rows a prefill
would have staged, the buckets from log2 `m_min` on. The profile cache file
(v2) keeps them beside the USAGE counts, and the threshold the last load
measured with them (a v1 file still loads, with neither). At load the cold
experts are then ranked by their rows at the threshold per pinned byte, ties
by their rows at a quarter of it, then by the selection counts and the index
(`moe_stage_order`); the copies go in that order until the budget is used.
The threshold is `GEIST_MOE_STAGE_M_MIN`, else the profile's, else 32 (the
cost model's pick for both models here; it is measured only after the
copies exist). Without prefill rows (no cache, a fresh one, a v1 file) the
order is the selection counts', as before: the same copies. The resident
experts do not change: they follow the USAGE counts (or
`GEIST_MOE_PROFILE`) as before, and the prefill rows touch only the copies.

The rows a profile needs come with the cache on (`GEIST_MOE_PROFILE_CACHE`):
the first run counts and writes them when the model goes, the next load
uses them. Measured with the cache made by one 2,726-token request (128
decoded) and copied fresh into each run, `GEIST_MOE_PROFILE` as before for
the resident set; SAME is a profile of the measured request itself
(`docs/OFFLOAD.md` of main), CROSS of the first 2,726 tokens of
`docs/ARCHITECTURE.md`. Prefill tok/s, interleaved cycles, 4 GiB unless
"all":

| | off | main (4 GiB) | **SAME** | **CROSS** | all |
|:--|--:|--:|--:|--:|--:|
| Qwen3.6-35B-A3B, 3 cycles | 775 / 792 / 786 | 811 / 858 / 850 | **931 / 926 / 921** | **916 / 913 / 916** | 921 / 893 / 926 |
| median, share of the gain | 786 | 850, 48 % | **926, 104 %** | **916, 96 %** | 921, 100 % |
| staged experts / rows | | 1,701 / 107 k | 2,782 / 195 k | 2,514 / 176 k | 2,782 / 195 k |
| Qwen3-30B-A3B, 5 cycles | 709 / 693 / 702 / 705 / 701 | 820 / 820 / 823 / 797 / 808 | **826 / 845 / 839 / 840 / 842** | **846 / 826 / 842 / 827 / 834** | 846 / 828 / 822 / 823 / 840 |
| median, share of the gain | 702 | 820, 93 % | **840, 110 %** | **834, 105 %** | 828, 100 % |

- With a profile of the request itself 4 GiB stage what every cold expert
  copied (11.7 / 9.1 GB) stages: the same 2,782 / 3,237 experts and rows,
  so the prefill is at the all-copies speed. A profile of another text
  keeps 96 % of Qwen3.6-35B-A3B's gain instead of 48 % (main in the same
  session; 57 % in the budget table above). Qwen3-30B-A3B was at 91-93 %
  already and gains within the noise: SAME and "all" stage the same set,
  so their gap (840 against 828) is the spread.
- Rows at a quarter of the threshold matter after one request: few cold
  experts reached 32 rows in a chunk, the rest of the 4 GiB went by the
  decode counts. Offline (the two caches, 4 GiB, the share of the rows a
  prefill of `docs/OFFLOAD.md` stages at 32 that the copies hold): decode
  counts 51 % / 77 % (Qwen3.6-35B-A3B / Qwen3-30B-A3B), the CROSS rows at
  32 then decode counts 66 % / 85 %, at 32 then at 8 89 % / 94 %; a first
  build with the 32-row key alone measured CROSS at 871 tok/s (60 %).
- Decode unchanged (the same resident experts: 3,917 / 2,833 hot, the same
  cover in every run): medians main 79.5 / 66.3, SAME 77.2 / 66.8, CROSS
  77.7 / 66.7 tok/s, staging off 78.2 / 65.7; the spread within a variant
  is as large (SAME 74.9-78.5; in a first session main 77.9, SAME 79.1).
  A decode step stages nothing, and the copies are not read there.
  NOPROF (the branch, the cache on and empty) staged the same 1,701 experts
  for the same 107,380 rows as main: the fresh start makes the same copies.
- The fresh threshold measurement can still land on 64 for Qwen3-30B-A3B
  (one run in five, every variant), as before.
- Perplexity (`bench_nll`, `GEIST_NLL_PREFILL=1`, the context prefilled in
  512-row chunks, 1,023 tokens scored; tokens 1,024-2,048 and 2,048-3,072
  of `docs/OFFLOAD.md` and of `docs/ARCHITECTURE.md`; main against SAME):
  Qwen3.6-35B-A3B +0.16 / −0.02 / −0.16 / +0.91 % per window, +0.22 % over
  the four; Qwen3-30B-A3B −0.13 / +0.48 / +0.06 / −0.16 %, +0.06 %. The
  +0.91 % window (`docs/ARCHITECTURE.md` 1,024-2,048) is where staging
  itself moves the score: staging off 10.312, main 10.298, every cold
  expert copied 10.373, SAME 10.391 (twice, bit for bit). More rows on
  the GPU's F32 path instead of the CPU's W8A8 flip routing there; #359
  saw the same spread with the default.

**Results with the default** (RTX 2080 Ti + 9950X, the 2,726-token request,
128 decoded tokens, 4 interleaved cycles against staging off on the same
binary, under the desktop lease; one CI coverage job was still running on
one core, so both columns read a little low; Strata from runs
1839/1842/1930):

| | prefill tok/s | decode tok/s | peak VmRSS |
|:--|--:|--:|--:|
| Strata, MTP off (Qwen3.6-35B-A3B, CUDA) | 605 | 62.0-62.2 | |
| Qwen3.6-35B-A3B, staging off | 795 / 797 / 800 / 756 | 77.9 / 76.5 / 77.7 / 77.6 | 15.6 GB |
| **Qwen3.6-35B-A3B, default (4 GiB)** | **856 / 866 / 793 / 867** | 77.2 / 77.9 / 78.3 / 78.6 | 23.3 GB |
| Qwen3-30B-A3B, staging off | 689 / 668 / 697 / 647 | 64.7 / 65.5 / 65.6 / 64.9 | 14.1 GB |
| **Qwen3-30B-A3B, default (4 GiB)** | **810 / 830 / 814 / 809** | 64.9 / 65.5 / 64.7 / 68.2 | 18.5 GB |

- Prefill +8 % (Qwen3.6-35B-A3B, median 861 against 796; 1.42x Strata) and
  +20 % (Qwen3-30B-A3B, 812 against 679). With every cold expert copied
  (`GEIST_MOE_STAGE_HOST` past the model's cold bytes) the earlier
  acceptance run read 921 against main's 799 (+15 %, 1.52x Strata) and
  830 against 695 (+19 %).
- Decode unchanged: medians 78.1 / 77.6 and 65.2 / 65.2.
- Perplexity unchanged (`bench_nll`, `GEIST_NLL_PREFILL=1`: the context
  prefilled in 512-row chunks, the next 1,023 tokens scored; four windows
  per model: tokens 1,024-2,048 and 2,048-3,072 of this file and of
  `docs/ARCHITECTURE.md`): Qwen3.6-35B-A3B −0.28 to +0.11 % per window,
  −0.08 % over the four; Qwen3-30B-A3B −0.36 to +0.49 % per window, +0.03 %
  over the four. The staged experts run in F32 accumulation on the GPU
  instead of the CPU's W8A8; the spread is the routing's sensitivity, not a
  direction.
- **VRAM**: 48 slots (16 experts) of the biggest projection, 864 KiB
  (Qwen3.6-35B-A3B, its Q6_K layers' down) and 1,296 KiB (Qwen3-30B-A3B):
  40.5 and 60.8 MiB, from the planner's 512 MiB headroom, so the hot set is
  the same as with staging off (2,832 hot experts for Qwen3-30B-A3B either
  way). Taken from the hot experts instead (the first version: 23 and 22
  fewer, 0.6-0.7 % of the hot capacity) they cost Qwen3.6-35B-A3B 1.6 %
  decode (77.8 against 79.1 tok/s with the hot count matched, 4 cycles).
  32 experts per step were no faster than 16 (931 against 926 tok/s).
- A step stages ~7-12 experts on average (Qwen3.6-35B-A3B: 1,701 of a
  2,726-token prefill's 240 layer steps with the default, 2,806 with all
  cold experts copied); the copies hide behind the GEMMs.
- The first try, the existing streaming path (#79: each projection copied
  into one of two slots when its GEMM runs), lost: 578 tok/s, the copies
  and GEMMs serialized weight by weight.
- Dense models pay nothing: `bench_revision_ab.py`, synthetic Llama-1B
  Q4_K on the card, base against the branch, 6 cycles: -0.1 to 0.0 %, the
  control ±18.9 %.

Raw data: `benchmark/results/raw/moe-prefill-stage-2026-11-01/`; the host
copies by the prefill profile: `benchmark/results/raw/moe-prefill-profile-2026-11-02/`.

### MTP draft decoding in GPU mode (#246)

Qwen3.6-35B-A3B ships one MTP (nextn) block. `GEIST_MTP=1` loads it, and
`geist_session_decode_speculative` then drafts with it; the target verifies
every draft, so the output is the target's. In GPU mode the block's attention
runs on the GPU and its 256 experts on the CPU (the GPU's expert table covers
the 40 target layers only). Measured on the RTX 2080 Ti + 9950X box with
UD-Q4_K_XL, GPU mode, the run-1830 expert profile, the 2,726-token request of
the sections above, 128 greedy tokens, `bench_perf_sweep --spec K` (draft
length K: the seed and K - 1 drafts per call).

**Correctness.** Greedy output with MTP equals greedy output without, up to
near-ties of the target. Four prompts (this file at 2,726 tokens,
`docs/ARCHITECTURE.md` at 1,024, `src/engine/session.c` at 512, `README.md`
at 256), 128 tokens each, K = 2 and 3: five of the eight runs are identical
over all 128 tokens. The other three part at the reference's closest call
(its top two tokens 0.025, 0.042 and 0.019 logits apart; no closer step in
those runs) and take the runner-up: the verify pass scores its rows with
other kernels than decode_step's single row (`tests/spec_tie.h`, whose bar is
1.5 logits). K = 4 on `docs/ARCHITECTURE.md` parts the same way at a 0.155
gap. MTP off on the branch is token-identical to main on all four prompts.
On Metal (Qwen3.8-27B Q4_0, a dense model with an MTP block; it takes the
per-row DeltaNet rollback below) K = 2 and 3 are identical to MTP off over
128 tokens, and `test_speculative_loop_int` passes with `GEIST_MTP=1`.

**On main MTP decoded at 22-29 tok/s against 79 with it off** (7-9 with a CI
job beside it). Three causes, fixed (single runs, the request above):

1. The draft head mapped the vocabulary's logits row (248k floats in device
   memory behind the BAR, ~30 ms per draft token over the bus) for a host
   argmax. It takes the device argmax now, as the target head does: K = 3
   26.8 → 59.4 tok/s.
2. A verify entered the thread region of the model's backend only, and
   Vulkan has none: the CPU experts ran outside cpu_x86's region. The verify
   enters the region on `moe_cpu` too now, as prefill does, and a draft
   step the decode one: K = 2 8.8 → 65.6 tok/s with a CI job on the box, and
   55.9 → 65.6 for the draft's region alone. The n-gram drafter was hit the
   same way (K = 3: 52.6 → 66.0 tok/s, quiet, with item 3).
3. A partial accept replayed the accepted rows through a prefill whenever the
   model had an MTP block, even with `GEIST_MTP` off (n-gram drafts or a
   draft model on Qwen3.6). The per-row DeltaNet states of #192 serve it now,
   and the MTP block's carry comes from the verify's rows: K = 3 53.0 → 59.4.

A verify's experts also run as decode steps' do (row by row, the batched
decode's path of #245) instead of a prefill chunk's per-expert GEMMs: the
same kernels as decode_step. As fast (K = 2 63.2 → 65.6, K = 3 59.1 → 59.4
tok/s), and the drafts are accepted more often (K = 2: 0.71 → 0.78), the
MTP block reading the target's hidden rows from those kernels.

**Results**, 3 interleaved cycles, the request above (`ab-prompt.jsonl`):

| | decode tok/s | median | vs off | prefill tok/s | tokens/call | acceptance by draft |
|:--|:--|--:|--:|--:|--:|:--|
| main, MTP off | 78.96 / 79.33 / 79.08 | 79.1 | | 862-873 | | |
| branch, MTP off | 79.07 / 80.37 / 79.82 | 79.8 | 1.01 | 856-868 | | |
| main, MTP K = 2 | 28.78 / 22.06 / 28.11 | 28.1 | 0.36 | 618-637 | 2.65 | 0.65 |
| **branch, MTP K = 2** | 64.36 / 66.61 / 66.68 | **66.6** | **0.84** | 634-644 | 2.78 | 0.78 |
| branch, MTP K = 3 | 60.99 / 60.75 / 60.58 | 60.8 | 0.77 | 632-635 | 3.15 | 0.66 / 0.49 |
| main, n-gram K = 3 | 53.80 / 52.64 / 51.67 | 52.6 | 0.67 | 867-878 | 1.78 | 0.16 / 0.14 |
| branch, n-gram K = 3 | 65.51 / 66.00 / 66.28 | 66.0 | 0.83 | 849-875 | 1.78 | 0.16 / 0.14 |

On text the drafts guess better, MTP comes level with MTP off and no further
(`ab-high-acceptance.jsonl`, 3 cycles, medians):

| prompt | MTP off | K = 2 | K = 3 | K = 4 | acceptance at K = 3 |
|:--|--:|--:|--:|--:|:--|
| `docs/ARCHITECTURE.md`, 1,024 tokens | 85.5 | 81.2 (0.95) | 84.9 (0.99) | 84.2 (0.98) | 0.91 / 0.74 |
| `README.md`, 256 tokens | 78.4 | 74.5 (0.95) | 78.5 (1.00) | | 0.91 / 0.80 |

**Why no gain: a verify row costs a decode step.** Per call (wall-clock
phases, the request above, a diagnostic build; `trace/`): a decode step
takes 12.96 ms, 14.49 with the MTP block kept in step. K = 2: the draft 2.3
ms (one MTP step, its CPU experts and head included), the verify of 2 rows
25.7 ms (1.98 steps: layers 21.9, MTP block 1.8, head 1.7), the forward of
the correction token 15.1 ms: 43.1 ms for 2.78 tokens, 15.5 ms a token. K =
3: 4.6 + 32.9 (2.54 steps) + 14.8 = 52.4 ms for 3.15 tokens, 16.6 ms a
token.

- The draft is cheap; the verify is not. Each row routes to its own 8 of
  256 experts, so a verify of k rows moves about k steps' expert work, the
  hot experts' matvecs on the GPU and the cold ones on the CPU, as batched
  decode does (#317: 8 rows at 6.3 steps). The per-expert GEMMs of the
  prefill path cost the same (26.1 against 25.6 ms for 2 rows). A dense
  model's verify reads its weights once for all rows; an MoE's mostly does
  not.
- Break-even at K = 2 and 0.78 acceptance needs a 2-row verify in 18.6 ms
  (1.43 steps). Leaving the correction token unforwarded for the next
  call's verify (llama.cpp's cadence) saves its 15 ms but emits one token
  less per call: 28 ms for 1.78 tokens, 15.7 ms a token, still slower.
- Prefill with MTP on loses 26 % (632-644 against 856-873 tok/s): every
  chunk also runs through the MTP block to keep its cache in step, and its
  experts all stay on the CPU.
- Strata's MTP draft brought nothing on the same request either (62.2 tok/s
  either way, run 1842).

Re-checked after main's GPT-OSS work (#243, which splits a decode step's
CPU experts by rows) was merged in, 2 cycles (`ab-post-merge.jsonl`): MTP off
77.3 / 76.2 on main against 78.0 / 78.1; MTP K = 2 7.3 / 7.2 on main (this
time beside other load) against 62.7 / 62.6, K = 3 63.3 / 63.4 (acceptance
0.77 / 0.51); n-gram K = 3 12.5 / 12.4 against 62.9 / 63.9.

MTP stays opt-in. For an MoE with experts split across GPU and CPU, leave
it off. Raw data: `benchmark/results/raw/moe-mtp-2026-11-02/`.

**A dense model on Metal (#401).** There a verify reads the weights once
for all its rows, so it should be cheap; on Qwen3.8-27B Q4_0 (M1 Max) MTP
nevertheless decoded at a third of MTP off, because 2-7 rows of a Q4_0,
Q4_1, Q8_0 or Q5_K linear took Metal's naive kernels (7-24x a decode row).
With 2- and 3-row matvecs a call of K = 2 costs about 0.1 decode step for the
MTP draft (~8 ms), 1.4 steps for the verify (layers 1.3x, the Q6_K head's
2-row GEMV 2.2x) and one step for the correction token. That pays from an
acceptance of ~0.55: medians of 3 cycles, 128 tokens, MTP off 11.1-11.8
tok/s, K = 2 11.9 (acceptance 0.65, `docs/ARCHITECTURE.md`) and 12.9 (0.74,
`docs/OFFLOAD.md`), K = 3 11.1 / 12.4, K = 4 8.1 / 10.3. The correction
token's own forward is the largest fixed cost left: folding it into the
next verify (llama.cpp's cadence) would need a session state for an
emitted but unforwarded token. Raw data:
`benchmark/results/raw/metal-mtp-27b-2026-11-03/`.

All measurements under `scripts/gpu-lock.sh` (and `with-mac-lock` on the
Mac), interleaved cycles, raw samples kept, `bench_revision_ab.py` parent vs
patch for "single-backend runs pay nothing".
