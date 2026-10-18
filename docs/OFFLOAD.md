# Offload: a run-time choice between placements fixed at load (proposal)

Status: **proposal, nothing implemented.** Research branch
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

### Placement and twins (load time)

The #49 split already gives each layer a home backend. Offload adds one thing
per weight of a CPU layer: a **GPU twin**, the same `geist_weight` resolved on
the GPU backend, whose bytes are *not* resident:

- **v0, no new transfer code:** the twin is a Vulkan weight in host memory,
  read by the shaders over the bus, exactly as Vulkan does today for weights
  past VRAM (#466). Measured there: every layer on Vulkan with ~5 GiB in host
  memory prefills at 170 tok/s against 121 for the split in the same run. It
  proves the route and the threshold before any streaming code exists.
- **v1, a staging ring:** two VRAM slots of the largest layer's size. The
  twin's bytes are copied there by DMA (transfer queue) one layer ahead, so
  layer `l+1` uploads while layer `l` computes. One bulk copy instead of
  every workgroup reading tiles over the bus; this is what llama.cpp and
  Strata do.

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

**Phase 0: cost model, no engine code.** Measure `bw` (pinned host → VRAM,
host-memory reads by a shader), and `t_cpu` / `t_gpu` per row for Q4_0, Q4_K,
Q8_0 linears on the 2080 Ti box; compute `m_min` and compare it with the knee
in a `GEIST_M_MAX` sweep of "every layer on Vulkan, rest spilled".

**Phase 1: v0 route** (on top of #49). Twins as host-memory Vulkan weights,
the branch in `linear.c`, the threshold from the env.
Decision rule: keep if split 26 + offload prefills faster than both the split
alone and every layer on the GPU, by more than the run's resolution, with
decode within noise of the split (decode must not take the route).

**Phase 2: v1 staging ring.** DMA prefetch one layer ahead.
Target: ≥ 0.8× llama.cpp `-ngl 26` prefill (485 tok/s) on the same GGUF.
Decision rule: keep if it beats v0 by more than the resolution.

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

- PCIe generation and link width on the 2080 Ti box; `bw` decides `m_min`
  linearly.
- Whether Vulkan's host-memory (v0) path reaches enough of `bw` with
  m = 512 for phase 1 to show the effect, or whether phase 2 is needed first.
- Pinned host memory for DMA: the GGUF mapping is pageable. Staging needs
  either a pinned bounce buffer (one extra memcpy, CPU-bound) or importing
  the mapping (`VK_EXT_external_memory_host`, alignment limits).
- MoE concurrency: the CPU workers and the Vulkan host thread share cores; the
  OpenMP region (`parallel_region_begin`) has to leave one thread for submit.
