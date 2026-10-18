# Offload: a run-time choice between placements fixed at load (proposal)

Status: **phase 0 measured; streaming of spilled weights implemented in the
Vulkan backend (opt-in, `GEIST_VK_STREAM_M_MIN`); the CPU/GPU route is not
built yet.** Research branch
`research/offload-scheduler`. Builds on the layer split of #49
(`feat/49-layer-split`, not merged) and the MoE family of #48.

## The question

Can geistlib move work between the CPU and a GPU **at run time**, like
[Strata](https://github.com/Niko1221/Strata) does, without giving up the rule
the whole engine is built on: *decide everything expensive at load time, so
the hot path is branch-light* ([ARCHITECTURE.md](ARCHITECTURE.md))?

## Answer in short

Yes, if "run-time scheduling" means **choosing between paths that were all
bound at load**, and the choice is a pure function of the batch size `m`. No
graph scheduler, no per-op fallback, no migration of weights while decoding.

```
load time:  placement  = where each weight lives          (data, fixed)
            twins      = kernels bound on every backend that may run it
            m_min[w]   = rows from which the GPU path pays (cost model, calibrated)
run time:   route(w,m) = m >= m_min[w] ? gpu : home[w]     (one compare per linear)
```

That one compare covers the two mechanisms with measured or published payoff:

1. **Prefill op offload.** A CPU-resident weight is streamed to the GPU when a
   prefill chunk is large enough. This is all of the measured prefill gap to
   llama.cpp in #49 (2.9×, gone with `-nopo 1`).
2. **MoE expert placement.** Hot experts are resident on the GPU, the CPU
   computes the cold ones in place, concurrently. Strata's core for decode.

Strata's adaptive expert cache, SSD tier and decode-time swaps stay out until
mechanisms 1 and 2 are measured. Their measured gain on Strata's own setup
(+16.5 % decode) is the upper bound to beat.

## What Strata does, and what fits here

From Strata's `docs/DETAILS.md`, `docs/HOW_IT_WORKS.md` (engine 0.1.x, built on
llama.cpp/ggml):

| Strata mechanism | Fits? | Why |
| :-- | :-- | :-- |
| Static profile cache: the hottest experts in VRAM (4,500 of 24,576 serve 50 % of selections) | **yes** | It is a placement fixed at load: #48 phase 2 already plans it |
| CPU computes the experts not on the GPU in place, concurrently with the GPU | **yes** | One more route of the same weight, see below. Needs a non-blocking GPU submit |
| `--pcie-frac`: a share of the missing experts copied to the GPU instead of computed on the CPU | **as a threshold** | The same trade as op offload; we decide it by `m`, not by a fraction |
| Prefill in chunks up to 8,192, borrowing expert-cache VRAM for the buffers | **partly** | Large chunks are what makes streaming pay; borrowing cache slots waits for a cache |
| Next layer's experts streamed over PCIe while this layer's attention runs | **yes (v1)** | Prefetch one layer ahead in a staging ring |
| Adaptive tier: LRU swaps between decode windows on a helper thread | **not yet** | Run-time migration; it changes residency, not arithmetic, but it is the most code. After v1, if the static cover is too low |
| SSD tier (`experts.bin`, `posix_fadvise`) | **no** | The page cache of the mmap does this already (#48 phase 0) |
| Speculative decoding with an MTP draft | separate | Already in geistlib (`mtp.c`, n-gram draft); orthogonal to offload |

## What we already measured

**Layer split, #49** (Qwen3-14B Q8_0, 15.7 GB, RTX 2080 Ti 11 GB + Ryzen 9
9950X, `benchmark/results/LAYER-SPLIT.md` on the branch):

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
  The #49 split (`feat/49-layer-split`) reads its hidden rows back the same
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

## Where the threshold comes from

Streaming a weight of `B` bytes costs `B / bw` once per chunk; running it on
the CPU costs `m · t_cpu` per chunk. The GPU path pays from

```
m_min = (B / bw + t_launch) / (t_cpu - t_gpu)        (t_* = time per row)
```

Qwen3-14B Q8_0 on the 2080 Ti box, one layer: `B` ≈ 0.39 GB, `bw` ≈ 12 GB/s
(PCIe 3.0 x16, to be measured), `t_cpu` ≈ 1/(88 · 40) s = 0.28 ms per row and
layer: **`m_min` ≈ 115 rows.** So decode (m = 1) and speculative verify
(m ≈ 4–16) never stream, a 512-row prefill chunk always does, and llama.cpp's
own fixed batch threshold (32) is in the same range. With the transfer bound,
the 14 streamed layers cap at 12 / 0.39 · 512 / 14 ≈ 1,100 tok/s, well above
the 485 llama.cpp reaches for the whole model.

`bw`, `t_cpu` per dtype and `t_gpu` per dtype are measured once per machine
by the existing calibration driver (`struct geist_tunable`,
`geist_backend_calibrate`) and keyed like every other calibration;
`GEIST_OFFLOAD_M_MIN` overrides for experiments. `m_min` is computed per
weight at load, so the hot path compares two integers.

## Design

### Who decides the placement: one planner at load (missing today)

Today nothing computes a placement from the platform as a whole. The pieces
that exist each decide one thing locally:

| what | where | decides |
| :-- | :-- | :-- |
| ISA, cores, µarch | `src/base/hw_probe.c` | which kernels may be bound; the calibration key |
| CPU quota | `src/base/cpu_limit.c` | the default thread team |
| Measured tunables | `src/base/calibration.c`, `cpu_neon/calibrate.c`, `metal/tuning.c` | kernel knobs per machine, not placement |
| VRAM fill | `vulkan/ops.c` (`vk_weight_fits_vram`, #466) | weights go to VRAM **in load order** until a 1/16 reserve is left, the rest to host memory: greedy, blind to which weight benefits most |
| Prefill chunk | `arch_state.c` (`fast_host_bytes`, `geist_backend_memory_info`), `prefill_tuning.c` | `m_max` from the BAR window, free VRAM and the model size |
| Layer split | #49 `split_layers` | **by hand**, no default |

The planner closes that gap: one pure function at load, from what the probes
already report, to the whole placement.

```
offload_plan_build(model sizes, per-weight decode benefit,
                   backend memory_info (free VRAM, unified?),
                   session KV + scratch bytes (scratch_plan),
                   calibration: bw, t_cpu, t_gpu per dtype)
  -> home[w] for every weight, hot experts, m_min[w]
```

The rule is the same for dense layers and experts: **fill the GPU's free
memory, after the session's KV cache and scratch, with the weights that save
the most decode bytes per byte of VRAM, highest first.** Decode reads each
weight it uses once per token, so the saving of weight `w` is its size times
the probability it is read in a token: 1 for a dense weight, the measured
selection frequency for an expert (#48 profile). Dense layers then all have
the same value per byte, and the rule degenerates to #49's split with
`split_layers = ⌊(free − KV − scratch) / layer bytes⌋`. That replaces the hand
setting and Vulkan's load-order fill. Experts are filled in profile order. On
unified memory (`memory_info.unified_memory`) the planner returns "everything
on the GPU backend" and no twins. Without a GPU it returns today's plan.

Pure function, so it gets a unit test with fake inputs (a small card, a big
card, unified memory, no profile), and `GEIST_VERBOSE` prints the plan it
picked. Overrides (`split_layers`, `GEIST_OFFLOAD_M_MIN`) still win.

### Placement and twins (load time)

The #49 split already gives each layer a home backend. Offload adds one thing
per weight of a CPU layer: a **GPU twin**, the same `geist_weight` resolved on
the GPU backend, whose bytes are *not* resident:

- ~~v0, the shaders read the twin over the bus~~: as slow as the CPU
  (phase 0), dropped.
- **v1, streaming (built, `vk_weight_bind` in `vulkan/ops.c`):** a Vulkan
  weight in host memory is copied by `vkCmdCopyBuffer` into the next of two
  VRAM slots when `linear_t` runs it with `m >= GEIST_VK_STREAM_M_MIN`, and
  the GEMM binds the slot. The sequence's hazard tracking orders copy and
  GEMM. It lives entirely in the backend, so it also speeds up today's
  "every layer on Vulkan, rest spilled" placement (3.1× prefill above). A
  twin is then just a Vulkan weight that did not get VRAM. Open upgrades:
  the copy on the transfer queue one weight ahead (copy and GEMM now share
  the compute queue), and a calibrated `m_min` per dtype instead of the env.

Experts (#48 phase 2) use the same twin: a **hot** expert's twin is
resident in VRAM (from the versioned placement profile), a **cold** expert
has the streamed twin like any CPU-layer weight.

### Route (run time)

The single choke point already exists: every block reaches a weight through
`linear_w_or_legacy` and its pair/triple variants in `forward/linear.c`. The
route is one branch there, before the existing paths:

```c
/* forward/linear.c */
if (w->twin != nullptr && seq >= w->twin->m_min)
    return offload_linear(w->twin, x_buf, y_buf, seq);  /* upload x, GPU linear, download y */
/* ... today's code, unchanged ... */
```

- `offload_linear` uploads the `[seq, n_in]` activations, runs the twin's
  `fused->linear_t` on the GPU backend, downloads `[seq, n_out]`. At m = 512
  and d = 5120 that is ~10 MB each way against ~95 MB of weight, so roughly
  10 % on top of the stream. If it shows in the profile, the upgrade is to
  offload a whole FFN block (one upload, one download; the FFN has no state)
  instead of each linear.
- Attention and the KV cache **never move**: they stay on the layer's home
  backend, so the KV layout, its modes and the snapshot code are untouched.
  Only weight-bound linears are routed; they are stateless.
- The MoE block (`forward/moe.c`) calls the expert kernels directly. It gets
  the same route per active expert: hot experts are recorded on the GPU and
  submitted first, the CPU workers take the cold ones, then the block waits
  for the GPU and sums in expert order (the sum order is already fixed, so
  the result does not depend on who ran what).

### What the backends need

| change | where | size |
| :-- | :-- | :-- |
| A weight can carry a twin and its `m_min` | `struct geist_weight` (internal fields, like `linear_rows`) | small |
| Submit without waiting, wait later | Vulkan: `vk_seq_flush` = `QueueSubmit` + `WaitForFences` today; split it in two. Metal: `commit` / `waitUntilCompleted` already separate | small |
| Async copy into the staging ring (v1) | Vulkan transfer queue + a second fence (`xfer_fence` exists, used synchronously) | medium |
| Calibrated `bw`, `t_cpu`, `t_gpu` | one tunable per backend in the existing calibration blob | small |

No new vtable slot for v0: upload, `linear_t` and download exist.

### Correctness and determinism

- The route depends only on `m` and the load-time plan, so a run is
  deterministic for a given prompt and chunking. It is **not** independent of
  the chunk size: a 512-row chunk takes the GPU path, a 64-row chunk the CPU
  path, and GPU kernels are not bit-exact with the CPU by design. That is the
  same trade #49 accepted for the split and llama.cpp makes for op offload.
- Gates: `m_min = SIZE_MAX` (the default until the decision rule holds) is
  byte-identical to today; `m_min = 1` must match "every layer on the GPU"
  within the per-backend tolerance already defined; two CPU backends as home
  and twin must be byte-identical at every threshold (the #49 trick, it tests
  the plumbing without a GPU).
- Fault injection as in `test_layer_split_fault_int`: a failing upload, GPU
  linear or download in a routed linear returns its status and nothing after
  it runs.

## How it fits the architecture

- **Load-time binding kept.** Every kernel a weight can run on is bound at
  load. At run time there is one integer compare per linear call, no
  dispatch on dtype or op.
- **A block knows how, never who.** Blocks and family units do not change; the
  route lives in `linear.c` and `moe.c`, both already shared mechanics.
- **The CPU stays the product.** Without a GPU backend nothing changes; with
  one, offload is EXPERIMENTAL, off by default, opt-in through
  `geist_session_opts` next to #49's `split_backend`.
- **Dependency rules.** The arch reaches the GPU only through the twin's
  backend vtable; no backend includes another.
- **Measured, not claimed.** Every phase has a decision rule; a negative result
  is documented and the code removed, as #49 prescribes.

## Not built, and when it would be

| left out | add when |
| :-- | :-- |
| Graph scheduler, per-op CPU fallback (ggml_backend_sched) | never; it contradicts load-time binding, and the measured gap needs only weight-bound linears |
| Adaptive expert cache (LRU swaps between decode windows) | the static profile's held-out cover is the bottleneck on a 128+ expert model **and** swap traffic between windows fits the bus |
| SSD tier, own read-ahead | the page cache measurably fails (major faults per token) on a model that does not fit RAM |
| Multi-GPU, tensor/pipeline parallel | out of the target set (hardware people already own) |
| Offload on unified memory (Metal) | never: one memory, nothing to move; #49 showed Metal alone wins |
| FFN-block granularity instead of per linear | activation transfers show up in the v0 profile |

## Plan

**Phase 0: cost model — done** (results above). It also produced the
`HOST_CACHED` readback fix and the in-backend streaming (former phase 2).

**Phase 0b: the planner** (on top of #49, no offload yet). `offload_plan_build`
with its unit test; it sets `split_layers` when the caller leaves it 0 and a
split backend is given. Decision rule: on the 2080 Ti box its pick for
Qwen3-14B Q8_0 is within the run's resolution of the best hand split in
decode.

**Phase 1: the route** (on top of #49). Twins as Vulkan weights (resident
where VRAM is left, else spilled and streamed), the branch in `linear.c`,
the threshold from the env.
Decision rule: keep if split 26 + offload prefills faster than both the split
alone and every layer on the GPU, by more than the run's resolution, with
decode within noise of the split (decode must not take the route).

**Phase 2: overlap.** The copy on the transfer queue, one weight ahead.
Decision rule: keep if it beats single-queue streaming by more than the
resolution.

**Phase 3: MoE.** Qwen3-30B-A3B Q4_K_M (18.6 GB, does not fit 11 GB):
hot experts resident from the #48 profile, cold on the CPU in parallel with
the GPU (split submit/wait), prefill experts streamed above `m_min`.
Decision rule: decode beats both CPU only and llama.cpp `-ot exps=CPU`
(the comparison `strata-ref.yml` sets up) on the same box.

**Phase 4, only if phase 3 shows the static cover as the limit:** the
adaptive tier, measured against Strata's +16.5 %.

All measurements under `scripts/gpu-lock.sh` (and `with-mac-lock` on the
Mac), interleaved cycles, raw samples kept, `bench_revision_ab.py` parent vs
patch for "single-backend runs pay nothing".

## Open questions

- Streaming is off by default until a parent-vs-patch A/B (vulkan-ab.yml)
  shows the readback change and the slot path cost nothing where they do
  not apply, and the GPU parity suite passes on the real card.
- Pinned host memory for DMA: the GGUF mapping is pageable. Staging needs
  either a pinned bounce buffer (one extra memcpy, CPU-bound) or importing
  the mapping (`VK_EXT_external_memory_host`, alignment limits).
- MoE concurrency: the CPU workers and the Vulkan host thread share cores; the
  OpenMP region (`parallel_region_begin`) has to leave one thread for submit.
