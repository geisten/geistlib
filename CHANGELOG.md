# Changelog

All notable changes to geist are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches 1.0. While in 0.x, `EXPERIMENTAL`-tagged API may change in any
minor release.

## [Unreleased]

### Added
- **Per-machine calibration, PR 1 of 3** (`@stability EXPERIMENTAL`):
  measured tuning instead of two hard-coded reference points. New API
  `geist_backend_calibrate` / `_apply_calibration` / `_calibration_key`
  — the library measures and validates; consumers own persistence
  entirely (no filesystem access in the library). Backends export
  tunables via a descriptor slot; the driver adds median-of-three
  noise defense, an opaque key (µarch fingerprint from MIDR/sysctl ×
  tunable-set hash × per-backend generation), atomic apply, and
  `GEIST_E_STALE_CALIBRATION`. Precedence per knob: env override ??
  calibration ?? seed. cpu_neon ships five sondes
  (`q5k/q8_0/q4_01/iq4xs_native_mn`, `qk_sgemm_threshold`) that time
  the RESOLVER-INSTALLED paths on throwaway backend instances — a
  first raw-kernel draft confidently mis-calibrated the threshold
  until a real-model referee falsified it; the fidelity contract is
  now documented in the sonde file. The `geist-calibrate` example
  tool and the CI both-branches parity matrix follow.

### Fixed
- **Probe-and-bind, batch 3 — `buffer_copy` stops swallowing device errors**
  (#352): the five per-token/per-layer sites (`step.c` ×3, `head.c`,
  `kv_store.c`) tested `v->buffer_copy != nullptr` *and* fell back to a host
  `memcpy` when the copy **failed** — `kv_store.c` said so outright ("fall
  through to the host path on failure"). `buffer_copy` exists only on the
  batched-submit backends (Metal, Vulkan), whose whole reason for having it
  is that mapping a buffer forces a pipeline flush; a failure there is a
  device error, and it was becoming a silent slow path. The capability is
  bound at plan build now and failures propagate.
- **Probe-and-bind, batch 2 — the two `fused->*` sites that were the hazard**
  (#352): the post-attention `rmsnorm_add` and the PLE gate's `gelu_tanh_mul`
  both had the `op == nullptr || op(...) != GEIST_OK` shape, re-running the
  host path over an output the fused kernel may already have written. Each got
  its own probe rather than reusing an existing bit: `geist_fusion_query`
  carries shapes and the layer's weight pointers, so the FFN-probed
  `fuse_rmsnorm_add` does not answer for the attention post-norm even where a
  backend happens to accept any geometry today. The other nine `fused->*`
  checks are deliberately left: they implement a documented three-way protocol
  (absent / `GEIST_E_UNSUPPORTED`, which is side-effect-free by contract /
  real error, propagated) that is per-weight routing rather than capability
  detection — see the triage table on the issue.
- **A model with a squared-ReLU FFN crashed on Metal** (#352): `layer_ffn.c`
  calls `prims->relu_squared` unconditionally, and the Metal backend declares
  that slot `nullptr` — a null function-pointer call, once per layer. Reachable
  for any GGUF whose `activation` key says `squared_relu` / `relu2` /
  `gated_squared_relu` (the key is read regardless of architecture) on a
  backend that lacks the primitive. It is refused at plan build now, naming
  the backend and the cause, instead of dereferencing null in the hot path.
  The bitnet-b1.58 model in this repo does not reach it — Metal rejects its
  I2_S weights in the linear first — so the crash is shown by construction,
  not by a run.
- **`geist_model_load` replaced a specific diagnosis with a guess** (#352):
  `state_create` returns `void *`, so a failure carries no status upward and
  the caller reported `GEIST_E_IO` "file missing or malformed" for every
  cause, overwriting whatever the arch layer had put in the create-time error
  slot. It now keeps a message a lower layer already set.
- **Probe-and-bind for the optional primitives, batch 1** (#352): the seven
  `prims->op != nullptr` tests in per-token and per-layer paths are bound once
  in the exec plan. Four had the shape
  `if (op == nullptr || op(...) != GEIST_OK) { host loop }`, which conflates
  "absent" with "failed" and re-ran work the device op may already have done —
  `layer.c` carries a comment about exactly that hazard for `add` while the
  `scale_f32` sites had the same shape and no such guard. Bound paths now call
  the primitive unconditionally and propagate its status, per the
  `geist_backend.h` contract that a successful probe means the op must
  succeed. `transformer_layer_scale_output` returns a status instead of `void`
  for the same reason. Behaviour is unchanged on both arms (verified against
  the unmodified tree on Metal and cpu_neon).
- **Allocation-free linear-kernel contract, batch 3 — the x86 side** (#336,
  closes #353): the five cpu_x86 prefill paths allocated their activation
  scratch on every call. `kernel_i2s.c` and `kernel_i2s_avx512_vnni.c` used
  raw `malloc` — outside `heap.h`, and 16-byte aligned on x86-64 for buffers
  that feed AVX-512 loads; `linear_q4k.c`, `linear_q6k.c` and `linear_f32q.c`
  used `heap_alloc_aligned`, three or four per call, per projection, per
  layer, per chunk. All five now take the per-thread cpu_x86 workspace, which
  grows to a high-water mark and is reused. The I2S GEMMs gained `_pre`
  variants on caller-owned scratch (the shape the NEON side already has), and
  `i2s_gemm_avx512_vnni` takes its activation-permute buffer from the caller
  instead of allocating one per GEMM. `linear_f32q`'s scratch-failure path
  used to `return` with `y` untouched — a silently wrong answer; it now falls
  back to the M=1 kernel like its siblings. New `test_x86_kernel_no_alloc_unit`
  asserts a zero `heap_alloc_count()` delta across 200 decode calls plus every
  prefill shape, and runs under Intel SDE so the VNNI paths are what is gated.
  **Throughput is unmeasured and declared so:** there is no x86 hardware in
  this setup (bench board is a Cortex-A76, dev machine is Apple silicon), and
  numbers from an emulator would be meaningless. The case here is the
  contract, as on the ARM side, where the equivalent change measured neutral
  except where allocations sat inside a parallel region.

  **Update (2026-09-01), measured on the AMD 9950X reference host:** paired
  A/B against the immediate parent commit, two alternating rounds each,
  mean-of-5 per round, quiesced host (`bench_perf_sweep --seq-lens 128
  --decode-n 128 --repeats 5 --threads 16`). BitNet 2B-4T (I2_S, the
  `kernel_i2s_avx512_vnni.c` GEMM this batch touched most): prefill pp128
  1077.4 → **1112.4 (+3.3 %)**, decode tg128 122.5 → 123.2 (+0.5 %, noise).
  Gemma 4 E2B-it Q4_K_M (the `linear_q4k`/`linear_q6k`/`linear_f32q` paths):
  prefill 506.8 → 504.5 (−0.5 %), decode 49.3 → 49.3 (parity) — both within
  the ±1.5 % noise band established in
  [`benchmark/results/X86.md`](benchmark/results/X86.md). No regression on
  either model; the I2S prefill GEMM gets a small, plausible win from
  dropping its per-call `malloc`/`free` pair, matching the ARM-side finding
  that this contract change is neutral-to-positive, never negative.

### Added
- **`GEIST_LOG_KERNELS=1`** (#327): the cpu_neon resolver counts its own
  decisions per catalog row and prints them at backend destroy. "Which kernels
  does this model actually run" was being answered by reading the GGUF's dtype
  histogram and reasoning about the table — which is how a Q4_0-only change
  came to be suspected of moving a Q4_K model's throughput. The reference model
  reports `q4_K 182, q6_K 24, f32 71` and no Q4_0 row at all.
- **Apple perf guard band in CI** (#327): the coarse cliff detector
  (`benchmark/perf_gate.py`) ran only on the Linux leg. It now runs on the
  macOS leg too, against the qwen3.5-0.8B Q8_0 that leg already caches — the
  3.1 GB gemma4 reference is a download the mac runner deliberately skips.
  Floors are ~40 % of the worst of four observed runs (prefill 127.5 / 108.7 /
  132.3 / 75.1, decode 59.2 / 46.7 / 58.7 / 25.5 — a 1.8× spread on the shared
  runner). Calibrating from one sample, then from the lower of two, produced a
  gate that failed an unrelated PR; a cliff detector is for multiples, not
  percentages.

### Fixed
- **APPLE.md described a benchmark protocol that no longer existed** (#327):
  it documented pp200/tg50 and best-of-2 / best-of-5, while
  `tools/bench_quality_perf.py` runs 128/32 (warm-up 8, 3 repeats) and 512/64
  (warm-up 64, 10 repeats) and reports the **mean** with best/worst alongside.
  It also documented two environment variables, `GEIST_BENCH_PP` and
  `GEIST_BENCH_TG`, that are not in the source at all. The prose now follows
  the driver, names it as the single source of truth, and adds two things the
  file was missing: why `BENCH_THREADS` must stay unset on Apple silicon
  (measured — 8 threads hold a 1.3 % spread, 6 threads 15 %, 4 threads 29 %),
  and how to A/B two commits on a desktop that cannot be quiesced.
  The tok/s drop reported in the issue does not reproduce: a paired,
  order-alternating series of 8 pairs puts `c1e74ad` against today's main at a
  median of −1.1 % prefill and −1.0 % decode, with both signs represented
  (sign test p = 0.73) — inside a noise band of roughly ±5 %. The gap to the
  June comparison table stays unexplained, but it is not the code in that
  range, and not the OS either (that row's `Darwin 25.5.0` is still this
  host's kernel release).
- **Sampler: bounded selection instead of a full vocabulary sort** (#331):
  `top_k`/`top_p` built an (logit, index) pair array — on the heap for any
  vocabulary above 1024 — and `qsort`ed all of it per token, then capped
  `top_k` at 8192 without telling anyone. Both now select through a
  size-k min-heap held in the sampler workspace: O(n log k) instead of
  O(n log n), zero allocations after `geist_sampler_workspace_init`, and
  every `top_k` in [1, n_vocab] keeps the semantics it asked for. Ties
  break by lowest index (`qsort`'s comparator called every NaN equal, so
  the old order was not even well defined). Measured on M1 Max, vocab
  262144: `top_k=40` 23.9 ms → 181 us/token, `top_p=0.9` 24.7 ms →
  1.01 ms/token; end-to-end sampled decode on gemma4-e2b 20.5 → 41.1 tok/s.
  The one regime that loses is `top_k` at a quarter of the vocabulary or
  more (`top_k=40000` at 151936: 13.2 → 15.8 ms) — where the old path was
  silently sampling from the top 8192 instead.
- **Plain temperature sampling no longer allocates per token** (#331):
  `geist_sampler_temperature` heap-allocates a `n_vocab` probability buffer
  for every vocabulary above 8192, and the decode head called it directly
  whenever `temperature > 0` without a top-k/top-p filter — a 1 MB
  malloc/free per token on gemma4. New `geist_sampler_temperature_ws`
  samples out of the workspace, and the session now sizes that workspace
  for every non-greedy mode (about 4 MB at vocab 262144; greedy still skips
  it). Throughput is unchanged — measured 967 vs 962 us/token at vocab
  262144, the allocation is lost in the softmax — so this is the
  allocation-free contract, not a speedup.

### Added
- **Batched Metal embed lookup** (#345): new fused slot
  `embedding_lookup_scaled_rows` — chunk token ids travel via
  `setBytes`, one dispatch embeds the whole prefill chunk instead of
  one tiny dispatch per token (pp512: 513 → 3 dispatches,
  bit-identical logits).
- **Fused silu_mul** (#347): the SwiGLU FFN epilogue in one pass
  (every qwen35 model) — plan-bound via `GEIST_FUSED_SILU_MUL`,
  greedy-generation-gated (register-vs-memory rounding, top-5 stable).
  Together with #340/#345 the #322 program lands the 4B Metal prefill
  at 626 → 734 t/s; the 27B UD re-baselines at 94 pp (was 91.6 —
  protocol noise; `GEIST_M_MAX` 64/256 is a wash at 27B shapes).

## [0.10.7] — 2026-08-30

### Fixed
- **Allocation-free linear-kernel contract, batch 2** (#336): the rest of the
  NEON resolver table. Twelve wrappers still discarded `be` and called
  quant.h's allocating convenience entry points; they now quantize into the
  per-thread workspace and call the `_pre` variants. Four `_pre` variants
  existed but had never been declared in `quant.h`, so no caller could reach
  them. Four prefill kernels had none at all — Q4_0, Q4_1, IQ4_XS and the
  Q4_0 x8 prefill — and were split into a `_pre` on caller-owned scratch plus
  a thin allocating wrapper, matching the seventeen that already had that
  shape.
- **IQ2_S / IQ3_S prefill allocated inside an OpenMP loop** (#336):
  `linear_iq2s_w2a8_prefill_pre` and its IQ3_S twin allocated `accs` once per
  output row and `row_acc` once per block per row, *inside* `omp parallel
  for`, and dereferenced both unchecked. A 2048-row layer meant ~16k
  malloc/free pairs contending on the allocator in a single linear call.
  `m` is bounded by `GEIST_QUANT_M_CAP`, so both are stack arrays now, and
  the `m` guard the sibling kernels already carried was added.

  Measured on a quiesced Pi 5 (Cortex-A76 ×4, mean-of-10, reversed-order
  control, every run gated below 56 °C): **iq2s prefill +36.6 % at m=32 and
  +27.3 % at m=8, iq3s prefill +19.5 % at m=8 and +8.7 % at m=32**, against a
  null-control band of +0.06 % (−1.14…+0.40 %) over 13 deliberately untouched
  kernel/shape pairs. On the resolver path `linear_m1` gains +1.07 % (Q4_0),
  +0.49 % (IQ4_XS), +0.32 % (Q4_1), +0.28 % (Q3_K), +0.25 % (Q8_0);
  end-to-end on a Q4_0 model that is +0.32 % decode and +0.32 % prefill.

  After this batch no cpu_neon resolver path allocates per call. What remains
  in the kernels are the `static _Thread_local` high-water buffers AGENT.md
  §3 allows, and the thin convenience wrappers themselves.

### Changed
- `test_kernel_no_alloc_unit` covers eight dtypes instead of one and compares
  the whole workspace rather than the activation scratch alone, so a dtype
  whose M>1 path is rebound to the dequant trampoline is held to the same
  standard.

## [0.10.6] — 2026-08-30

### Fixed
- **Allocation-free linear-kernel contract, batch 1** (#336): `geist_weight.h`
  documents `linear_m1` / `linear_mN` as allocation-free; the NEON resolver
  wrappers discarded their `be` argument and called quant.h's convenience
  entry points, which `malloc` a per-call activation buffer. Two of them —
  `linear_q8_0_decode_w8a8` and `linear_q8_0_w8a8_prefill` — dereferenced
  that allocation **without checking it**, so OOM was a null write rather
  than a degraded answer.

  Q8_0, Q3_K and Q5_K (decode) now quantize into the existing per-thread
  workspace and call the `_pre` kernel variants. No new infrastructure was
  needed: the workspace (lock-free, TLS-cached, grow-on-demand via heap.h)
  and the `_pre` variants both already existed and had simply never been
  connected.

  New `test_kernel_no_alloc_unit` guards it by asserting what is
  observable: 200 decode calls plus every prefill shape up to the
  high-water mark leave the workspace pointer *and* capacity untouched.
  Built against the parent it fails; here it passes.

  Throughput is **not** measured and no claim is made — the host was at
  load 119–335 and the reference model contains neither Q3_K nor Q8_0.
  Numbers belong on the quiesced Pi 5 and will be posted to #336.

  Batch 1 of several, per the ticket. Still allocating per call: IQ2_S,
  IQ3_S, IQ4_XS, Q4_0/Q4_1, Q5_K's M>1 path, and the x86 side.

## [0.10.5] — 2026-08-30

### Changed
- **C23 length-first migration, batch 3** (#328): twenty helpers across the
  audio, vision, KV and attention layers, 122 call sites — the
  `attention_{int4,int8,kivi}_via_buffers` family (whose lengths were
  scattered through up to twenty parameters), `kivi_drain_one_layer`, the
  audio encoder entry points, the vision tower runners, and the small
  elementwise/FWHT/int4-KV helpers.

  `[static n]` applied to five parameters only — those passing all three
  of AGENT.md's tests (single dimension, no defensive null-check, no
  nullptr call site). The rest keep plain pointers because their extents
  are products over runtime dimensions, which batch 2 established gcc will
  not accept as a contract.

  Codegen: vector instruction count **identical in all eight touched
  objects**, spills net −3, +24 bytes total. No function changed shape.

  Two scanner hits were left alone as false positives:
  `i2s_x4_gemv_pair_m1` and `i2s_t5_gemv_pair_m1` already place `n_out0`
  before the `y0` it sizes.

  After this batch, ~57 pointer-before-length declarations remain, 55 of
  them the `linear_*` GEMV/GEMM family in `quant.h` — deliberately
  deferred, see #328.

## [0.10.4] — 2026-08-30

### Added
- **`AGENT.md`** — the coding rules the source already cited. Twenty
  comments across sixteen files deferred to "AGENT.md" (the heap.h
  allocation rule, the hot-path no-allocation rule, "no silent
  truncation", "explicit validation over assertions") and the file did not
  exist. It now does, leading with parameter order, and records what
  `[static len]` measurably buys so nobody has to guess. `CLAUDE.md` and
  an `AGENTS.md` symlink make it discoverable to tooling.

### Changed
- **C23 length-first migration, batch 2** (#328): the 13 shared fp32
  kernels in `gemma4_kernels.h` (rmsnorm, rope, both attention variants,
  the elementwise family, `linear_fp32`, the bf16 helpers) and the ten
  public vtable members that took an array before its length —
  `encode_pcm`, `stream_push/poll/end`, `encode_image/video`,
  `ffn_geglu_q4q6_mN`, `embedding_lookup_scaled_rows`, and the
  `geist_kernel_linear_mN_fn` / `_pair_mN_fn` kernel typedefs. ~340 call
  sites.

  `[static len]` is stated where an extent is expressible and deliberately
  omitted where it would be a lie — three separate reasons, all recorded
  in AGENT.md: a nullable parameter (`linear_fp32`'s `bias`,
  `rmsnorm_fp32`'s `weight`, nullptr at 84 and 12 call sites); a parameter
  the callee defensively null-checks (the encoder vtable entry points,
  which gcc reports as `-Wnonnull-compare`); and a bound that is a product
  over runtime dimensions that may be zero (`linear_fp32`/`rmsnorm_fp32`'s
  x and y, which gcc reports as "region of size 0"). Nothing gained
  `restrict` — these kernels document "y may alias x".

  **ABI:** `geist_arch.h`, `geist_backend.h` and `geist_weight.h` are all
  `@stability EXPERIMENTAL` and absent from `docs/API_CONTRACT.md`, so no
  versioned migration is required. Out-of-tree implementors of the encoder
  or backend vtables must reorder their function signatures to match;
  the compiler reports each one as an incompatible function pointer.

  Codegen: no function changed its vector instruction mix and none gained
  a spill; net +76 bytes on a 1.36 MB archive. `[static n]` itself was
  measured to change nothing — one TU compiled three ways gave 2824
  instructions (old order) vs 2819 (new order) vs 2819 (new order with
  `[static n]`).
- **`xmalloc` in `tests/test_helpers.h`** — 113 unchecked `malloc` call
  sites across four logits/layer tests now abort instead of feeding a
  possibly-null buffer into a kernel. Found by gcc once the kernels
  declared their contracts.
- **`cpu_neon/kernel_catalog.h` names the canonical kernel typedefs**
  instead of re-spelling the signature inline, which had silently drifted
  from `geist_weight.h`.

## [0.10.3] — 2026-08-30

### Changed
- **C23 API style, documented and first batch migrated** (#328): the
  contributor guide gains a "C API style" section — lengths before the
  arrays they describe, `T arr[static len]` where the precondition is
  real, `nullptr`/`constexpr`/`[[nodiscard]]`, `restrict` only where
  non-aliasing is proven, subtraction-based cursor bounds, and the rule
  that signature migrations land in bounded, independently benchmarkable
  batches.

  Batch 1 is the 13 `dequant_*_row` codecs, which move to
  `(size_t n_elems, const void *blocks, float out[static n_elems])` — 76
  call sites across 26 files. The `dequant_row_fn` typedef they are
  dispatched through now matches that signature exactly instead of being
  cast to; calling through an incompatible function pointer is undefined
  however compatible the representations look, and the exact typedef
  immediately turned two missed indirect call sites into compile errors.

  Verified as a pure reorder: the optimized disassembly of all 11 dequant
  translation units has an **identical opcode sequence** with only the
  argument registers permuted, `__text` is byte-identical in every one,
  and `libgeist.a` is 88 bytes smaller (the retired casts). Internal
  signatures only — no public ABI change.
- **`NULL` retired from live code**: 81 occurrences across 7 files become
  `nullptr`. The 25 that remain are prose in comments.

## [0.10.2] — 2026-08-30

Bug-fix release: the eight open correctness and hardening tickets against
the GGUF reader, the weight resolvers, the transformer forward path, the
audio streaming lifecycle, and the shared profiler.

### Fixed
- **PLE stage and FFN destination decided once** (#326): the FFN chose
  where to write its residual result on `apply_ple` while the PLE stage
  ran on `apply_ple && per-layer input != NULL`. In between, the result
  went to a scratch buffer nothing then read, and
  `transformer_forward_one_layer` returned `GEIST_OK` with `h_out`
  untouched — 1536 zeros on every Gemma 4 layer reached with a null
  per-layer input. One `ctx->run_ple`, read by both. Prefill and decode
  always pass a per-layer input, so shipped inference was unaffected.
- **I2_S per-tensor scale counted as part of the tensor** (#334): the
  reader sized an I2_S tensor from its packed blocks alone, so a file
  four bytes short passed validation and the kernels read the scale from
  whatever followed. In β weight mode (`GEIST_WEIGHT_MMAP=0`, and all GPU
  backends) only `nbytes` was copied into the arena, so the scale was
  never copied at all and BitNet-2B-4T produced fluent garbage; it now
  matches the mmap path token for token.
- **GGUF cursor bounds by subtraction, and checked size arithmetic**
  (#332): bounds were tested by forming `p + len` for an
  attacker-chosen `len` — undefined before the comparison runs — and
  `skip_value` advanced before validating. Tensor dimensions,
  byte counts, offsets and alignment padding are checked; zero and
  non-power-of-two `general.alignment` are rejected.
- **Weight resolvers validate the source extent** (#325): `struct
  geist_weight` gains `raw_nbytes`, and cpu_neon / cpu_scalar / metal /
  vulkan reject a source shorter than (dtype, n_in, n_out) before
  installing a kernel or repacking. The Q4_K predecode hook was
  repacking whole tensors out of short buffers.
- **Odd rotary `head_dim` rejected** (#329): rotary rotates channel *i*
  against *i + head_dim/2*, so an odd `head_dim` left the last channel
  unpaired, its cos/sin table entry unwritten, and an uninitialized
  stack slot copied into the activation. `head_dim` is model metadata
  (`d_model / n_q_heads` for Llama and BitNet), so a malformed file can
  pick it.
- **Checked allocation and tensor-view arithmetic** (#330): the heap
  array macros multiplied `count * sizeof(type)` before the allocator
  could refuse it; the elementwise ops folded `shape[]` unchecked,
  ignored `ndim > 8`, and never compared the resulting byte range with
  the buffer; image and video geometry reached `height * width * 3` and
  an `int` row stride unbounded.
- **Streaming audio turn is failure-atomic and closeable** (#333):
  `audio_begin` committed `stream_begin` before allocating its scratch,
  and only a successful `audio_end` released either. A session destroyed
  mid-turn leaked the scratch (9.3 MB over two turns, measured) and left
  the encoder stream open for the next session. The encoder vtable gains
  `stream_abort`; destruction is idempotent.
- **Profiler and lazy policy caches no longer race** (#335): the shared
  per-stage counters were incremented with a plain `+=` and the
  registration flag read outside its mutex. Four concurrent sessions with
  profiling on produced 34 TSan data races and an abort; now zero.

### Changed
- **Metal DeltaNet sub-chunking** (#322 step 1): `deltanet_mix` encodes
  the chunk recipe in 64-token sub-chunks internally, so DN models keep
  the backend's `preferred_m_max` (256) instead of the arch-level cap —
  the surrounding GEMMs regain their occupancy (4B pp512 626 → ~726
  tok/s on M1 Max; DN math granularity unchanged, top-3 logits stable
  across a 241-token multi-sub-chunk prompt, gemma4 untouched). New
  backend cap `dn_subchunk`. The remaining 4B gap to llama.cpp Metal
  is GEMM-chain tuning (~500 ms of the wall), tracked in #322.

## [0.10.1] — 2026-08-30

### Added
- **IQ4_XS int8 mN prefill tile kernel** (#321): replaces the
  dequant+SGEMM trampoline for M>1 on non-Accelerate hosts — 4-token
  register tile over the `vqtbl1q` LUT decode, rows L1-resident across
  token groups. Pi 5 prefill: 4B IQ4_XS 8.6 → 15.3 t/s, 0.8B 62.3 →
  86.3; decode and RSS unchanged. Batched path pinned bit-equal to the
  m1 GEMV per token. IQ4_NL prefill keeps the trampoline.

## [0.10.0] — 2026-08-29

### Changed
- **x8 layout v2 — pre-transposed pack, zip-free GEMM, lane-SDOT m1
  GEMV** (#324, closes #319): the pack step stores each 4-row group
  4×4-u32-transposed, the mN GEMM drops its 32 in-kernel zips per
  block, and the m1 decode GEMV migrates to lane-SDOT. Bit-identical
  logits and greedy output (pinned on-board, gcc + clang); Pi 5 4B
  prefill 19.4 → 21.1 t/s, decode 4.2 → 4.36. Cumulative Pi 4B
  prefill across #317/#318/#324: 9.1 → 21.1 t/s (llama.cpp on-board:
  24.8; decode ahead 1.32×). New layout regression test
  `test_q4_0_x8_layout_unit`.
- **Lane-SDOT accumulation in the Q4_0 x8 int8 GEMM**: 4×4 u32 row
  transposes + `vdotq_laneq_s32` replace the per-(row,token) `vaddvq`
  horizontal adds — one accumulator carries four output rows in its
  lanes. Bit-identical logits (pinned on-board); Pi 5 4B prefill
  16.5 → 19.4 t/s on top of the x8 default flip.
- **Q4_0 x8 interleave default-on wherever SDOT exists** (was Mac-only,
  Pi opt-in): the Pi 5 4B A/B shows prefill 9.1 → 16.5 t/s via the #295
  int8 mN GEMM at a net +0.5 GB RSS — the cold Q4_0 mmap pages get
  evicted, so the packed copy never doubles residency.
  `GEIST_Q4_0_X8_GEMV=0` opts out on RAM-tight boards.

### Added
- **NEON IQ4_XS/IQ4_NL W4A8 decode GEMVs** (#314): `vqtbl1q` LUT +
  SDOT kernels replace the dequant trampoline on the m=1 path. Pi 5
  0.8B IQ4_XS decode 4.4 → 18.5 tok/s; Mac CPU 18.5 → 101. Logit-gated
  against the f32 scalar reference on both clang and gcc builds.
- **Metal IQ4 LUT-kernel tuning** (#309): 27B UD-Q4_K_M prefill
  78.7 → 91.6 tok/s; 27B q4_0 104.4 pp / 11.6 tg — ahead of llama.cpp
  Metal on both axes.

### Fixed
- **Prefill chunking used the state m_max instead of the session's**
  (#312): scratch overruns for any session m_max below the state
  default; DeltaNet models now cap chunks at 64 (O(C²) per-chunk cost,
  same cap llama.cpp uses). Also voids the #303 m_max A/B (both arms
  had run identical configs).
- **bench_quality used the Llama chat template for qwen archs** (#315):
  qwen3.5 instruct models degenerated into template-token loops; the
  battery now sends ChatML.
- **Metal simdgroup quant kernels + chunked DeltaNet** (#300–#303, #308):
  llama-class GEMV/GEMM kernels for Q4_0/Q4_1/Q8_0/Q5_K, `_fast` interior
  GEMM variants, a chunked DeltaNet prefill (the serial mixer's
  `mem_device` barriers alone carried 1.4 s per 512 tokens), and a fenced,
  lower-traffic decode mixer. Qwen3.8-27B q4_0 on an M1 Max:
  5.3 → 95.4 tok/s prefill, 2.7 → 12.1 decode (llama.cpp Metal: 93.1/8.2).
  `preferred_m_max` 128 → 256 (+5–6 % prefill, model-independent).
- **IQ4_XS / IQ4_NL loader + Metal kernels for mixed quants** (#304, #308):
  the two formats decode end to end (bit-exact vs gguf-py) with cpu and
  Metal kernels; Q3_K/IQ3_S get Metal kernels alongside. The 27B
  UD-Q4_K_M runs at 78.7 pp / 7.8 tg on Metal (llama decode parity).
- **Weekly gemma4-on-Metal smoke** (#305–#307): probe, kernel parity and
  an end-to-end generation on the cached E2B reference on hosted macOS.

### Fixed
- **gemma4 Metal prefill** (#305, also on the qwen35 branch as #303): the
  PLE fused probe accepted any m while the kernel handles rows==1 only —
  under the strict probe-and-bind contract every gemma4 Metal prefill
  hard-failed at layer 0. Probe now mirrors the kernel gate; the
  fused-probe agreement test pins the contract.
- Vanilla-mac (no libomp) build: unused `tq2_0` mt8 helper vs `-Werror`
  (#306).
- **qwen35 hybrid family** (#281): Qwen3.5/3.6/3.8 dense models load and
  generate. The transformer family generalizes from "attention in every
  layer" to a per-layer token mixer: layers dispatch on
  `GEIST_MIXER_ATTN` vs `GEIST_MIXER_DELTANET` above the attention code,
  which stays mixer-agnostic. The DeltaNet mixer (gated delta rule,
  causal depthwise conv, per-head gated RMSNorm) runs its recurrence as
  host-side scalar f32 over backend-projected activations; its conv +
  delta state lives on the session. Speculative verification checkpoints
  that state lazily and restores/replays an accepted prefix transactionally;
  prefix pinning remains unsupported. Attention
  layers gain the joint query+gate projection with a sigmoid output
  gate and partial NEOX RoPE from `rope.dimension_count`. The gpt2
  pretokenizer accepts `pre = "qwen35"` (qwen2 + \p{M}; the scanner's
  letter approximation already covers combining marks — parity pinned).
  MTP/NextN blocks load into a separate layer array and are never traversed
  by normal autoregressive decode; MoE variants stay rejected.
  Fixture: `make fetch-qwen35-model` (Qwen3.5-0.8B Q8_0, SHA-pinned).
  ponytail: sequential per-token recurrence + scalar kernels — chunked
  prefill and NEON/x86 SIMD are the next phase, tracked in #281.
- **Qwen3 family** (#275): `general.architecture = "qwen3"` loads and
  generates — llama-style GQA stack with per-head QK-norm (new
  `has_qk_norms` config flag, distinct from the Gemma norm bundle) and
  head_dim from `qwen3.attention.key_length` (128 on 0.6B, where
  `d_model/n_heads` would give 64; `q_out > d_model` is now an exercised
  geometry). The GPT-2 BPE encoder gains the qwen2 pretokenizer
  (contractions, single-digit splits, punctuation runs, the
  one-codepoint-prefix and trailing-whitespace rules), selected by
  `tokenizer.ggml.pre == "qwen2"` — token-id parity with the HF
  tokenizer is pinned for 10 fixture cases in `test_qwen3_e2e_int`.
  Special tokens directly after whitespace are no longer shredded (a
  `\n<|im_start|>` in any GPT-2-mode GGUF hit this). Fixture:
  `make fetch-qwen3-model` (Qwen3-0.6B Q8_0, SHA-pinned, CI-cached).

### Fixed
- **Gemma 4 E4B loads** (#258): the gemma4 family populator now derives the
  per-layer geometry from GGUF metadata (`attention.sliding_window_pattern`,
  `shared_kv_layers`, `key_length(_swa)`, scalar-or-array
  `feed_forward_length`, dual RoPE bases) instead of E2B hardcodes, and the
  KV-share sources fall out of the pattern. A GGUF whose geometry is not
  derivable fails with a message naming it instead of a downstream
  weight-wiring error. The audio and vision towers read their soft-token
  width from the checkpoint's `embedding_projection` shape (1536 E2B /
  2560 E4B), and the engine drops a tower whose width doesn't match the
  loaded model's `d_model` — same-family-wrong-variant towers can no longer
  inject garbage (extends the #240 family gate). New `make fetch-e4b-model`
  + a weekly `e4b-smoke` CI gate keep the MODELS.md E4B row executable.

### Changed
- **Audio precision policy is resolved once at encoder create** (#251):
  a single `audio_prec_policy` struct (attn/lconv W8A8, layer limit,
  the Apple FORCE_QUANT decision) is read from env exactly once in
  `audio_prec_policy_resolve`, stored on the encoder, and threaded to
  every layer load — the startup banner prints from the same struct the
  loader consumes, so the diagnostic cannot drift by construction. The
  kernel binding became immutable const tables behind a pure
  `audio_linear_resolve(force_scalar)`; the mutable `g_ops`/`g_bound`
  pair, its locking, and the `audio_linear_rebind` test hook are gone —
  the parity test A/Bs bindings through the pure resolver instead.

### Added
- **Incremental LM injection for the streaming audio turn** (#256 phase
  2): new optional `geist_session_audio_poll` — called from the session
  thread between pushes, it injects the soft tokens the encoder has
  already produced into the LM while the user is still speaking, so
  `audio_end()`'s tail shrinks to the final chunk. Never required for
  correctness (end() covers everything un-polled); greedy-equivalence
  with attach_audio and with the poll-free streaming turn is pinned in
  the session equivalence test. `examples/push_to_talk` polls per VAD
  frame; measured on macOS (paced 2.8 s clip): tail 880 → 678 ms — the
  Pi 5, where injection costs ~1.5 s, is the real beneficiary (#263).

### Added
- **Public streaming audio turn** (#256, `@stability EXPERIMENTAL`):
  `geist_session_audio_begin` / `_push` / `_end`. PCM is pushed while the
  user speaks (push is capture-thread-safe); the encoder overlaps its
  work with the arriving audio, and `end()` pays only the tail. Contract:
  greedy-equivalent to `attach_audio` over the concatenated PCM (pinned
  by `test_audio_stream_session_int` in the audio-smoke job; bit-equality
  is explicitly not promised — the overlapped attention reassociates
  float sums). `examples/push_to_talk` streams per VAD frame and prints
  the measured tail. Known limit, next step in #256: the LM injection of
  the soft tokens is not yet overlapped and now dominates the tail.

### Changed
- **W8A8 attention/LConv is the default** for the audio tower (#238
  rollout): every quality gate is green (soft-token parity, chat e2e 6/6
  voice clips, LibriSpeech WER within noise of the high-precision path)
  and the win is −38 % encode on the Pi 5. `GEIST_AUDIO_ATTN_W8A8=0` /
  `GEIST_AUDIO_LCONV_W8A8=0` opt back out; the parity test now pins
  defaults ≡ fully-quantized explicitly.

### Fixed
- `make TARGET=pi5` builds again: the target was missing the
  `_GNU_SOURCE` feature macro the linux target gained in 0.8.0, so
  `-std=c23` hid `mkstemp` from `test_io_malformed_unit.c` (#230) and the
  build died — unseen, because no CI leg built the pi5 flag set. A new
  build-only `pi5-target-build` job on the arm64 runner now compiles
  `TARGET=pi5` on every push, so the primary deployment target can no
  longer bit-rot. (#244)

### Added
- **W8A8 attention/LConv is now a gated, tested option** (#238): the
  existing `GEIST_AUDIO_ATTN_W8A8` / `GEIST_AUDIO_LCONV_W8A8` opt-ins are
  pinned by `tests/test_audio_attn_w8a8_parity_int.c` — soft-token cosine
  parity vs the shipping precision on a synthetic clip, thresholds
  calibrated from measured drift (mean 1−cos ≈ 3.4e-3), run mandatorily
  in the `audio-smoke` CI job. The three env reads in the weight loader
  are no longer process-latched, so one process can load A/B encoders.
  Defaults unchanged (opt-in stays opt-in per the #238 rollout plan).
  Measured on the Pi 5: attn W8A8 −20 % encode, +lconv −38 % (RTF
  0.33× → 0.21×), chat e2e 6/6 clips green.
- **Live streaming worker verified and pinned** (#235): with
  `GEIST_AUDIO_STREAM=1` the Conformer encodes 48-mel-frame batches while
  PCM still arrives (now with the incremental subsample by default —
  re-convolving from frame 0 per kick was O(T²)); on the Pi 5 the
  post-utterance tail on a 10 s clip drops 3 600 → 202 ms
  (0.36× → 0.02× of audio).
  `tests/test_audio_stream_live_parity_int.c` pins bit-identical output
  vs the monolithic path, mandatory in the `audio-smoke` job.

- **AVX-512 VNNI W8A8 kernel for the audio tower** (#237): x86 hosts with
  VNNI no longer run the scalar path for W8A8-tagged layers — the FFN,
  22–35 % of encode time per the Pi 5 baseline. One catalog entry in
  `audio_linear.c` (VPDPBUSD with unsigned-activation shift and
  weight-row-sum correction), function-level `target` attribute over the
  x86-64-v3 baseline, installed only when the runtime probe confirms
  VNNI and `GEIST_FORCE_ISA` doesn't clamp it away. The SDE CI job now
  proves the binding is actually selected (`GEIST_EXPECT_AUDIO_KERNEL`)
  and that it agrees with the scalar kernel.
- **Audio harness CI smoke** (#234): a new `audio-smoke` job builds and RUNS
  `bench_audio_encode` + `test_audio_latency_e2e` on every push, so the
  harness that produces the Pi 5 audio baseline can no longer rot behind a
  skip. Fixtures are fully reproducible: `make fetch-audio-tower` extracts
  the Gemma 4 audio tower (~590 MB) from the public checkpoint via HTTP
  Range requests instead of downloading the 9.7 GB file
  (`tools/fetch_audio_tower.py`, stdlib-only, output SHA-pinned);
  `audio_test_data/mel_constants.bin` is checked in with its generator
  (`tools/gen_mel_constants.py`); the test clip is synthesized
  (`tools/gen_test_wav.py`) — no voice recording enters the repository.
  `test_audio_latency_e2e` accepts `GEIST_AUDIO_WAV=<path>` to run a single
  caller-provided clip.
- `geist_model_modalities()` (**EXPERIMENTAL**): bitmask of modalities a
  loaded model instance can consume beyond text (`GEIST_MOD_AUDIO`,
  `GEIST_MOD_VISION`, `GEIST_MOD_VIDEO`). Lets a host decide up front
  whether to offer e.g. microphone input, instead of learning it from a
  failing `attach_*` call. The mask mirrors exactly the capability checks
  the attach calls perform; the invariant is pinned by
  `tests/test_model_modalities_int.c`. (#233)

### Changed
- **Audio tower matmul kernels are bound at load time from the hardware
  probe** (#236), like the LLM kernel catalogs — the compile-time
  `#if __ARM_NEON` fork is gone from the forward pass (enforced by
  `scripts/check-audio-dispatch.sh` in CI). Consequences: a binary built
  with SIMD flags binds the portable kernels on a host whose probe lacks
  the feature (no SIGILL on lesser ARM cores); `GEIST_AUDIO_KERNEL=scalar`
  forces the portable kernels; and non-NEON hosts now run true W8A8 for
  W8A8-tagged layers (scalar int8) instead of silently widening to W8A32 —
  the x86 VNNI entry (#237) is now one catalog line away. Binding parity
  is pinned by the hermetic `tests/test_audio_linear_parity_unit.c`
  (probed-best vs forced-scalar vs float64 reference).

### Fixed
- `geist_session_attach_audio` no longer silently truncates audio past
  ~10 s: the soft-token buffer was capped at a hardcoded 256 (≈ 10.2 s at
  ~25 tokens/s), so a 28 s clip paid the full 12 s encode and dropped
  everything past the cap with `GEIST_OK`. The bound now derives from the
  audio length via the encoder vtable's new optional `max_soft_tokens`
  op (single formula home: `audio_soft_bound_from_mel`), a full buffer is
  a loud error instead of a silent drop, and
  `tests/test_audio_long_clip_int.c` pins a 12 s clip to ~300 tokens in
  the `audio-smoke` job. (#247)
- The live streaming worker emitted one soft token fewer than the
  monolithic path for the same audio: the final flush did not mirror the
  monolithic path's extra padded mel frame (n_mel = frames + 1). (#235)
- The Gemma 4 audio/vision towers are no longer attached to other model
  families. Previously the encoder search heuristics (cwd `audio_bench/`,
  directory next to the GGUF) could pick up the tower for e.g. a BitNet
  load; `attach_audio` then injected 1536-dim soft tokens into a 2560-dim
  residual stream — out-of-bounds reads, garbage in the KV cache, returned
  as `GEIST_OK`. Tower load is now gated on `general.architecture ==
  "gemma4"`; other families answer `geist_model_modalities() == 0` and
  refuse `attach_*` with `GEIST_E_NOT_FOUND`. (#240)

## [0.9.0] — 2026-08-15

### Changed
- `geist_model_arch` is **STABLE since 0.9.0** and joins the agent-runtime
  contract (`docs/API_CONTRACT.md`, `examples/agent_contract_smoke.c`). Since
  the agent layer moved out of tree in 0.7.0, chat templating is the consumer's
  job — but the model *family* it selects on can only come from the model file,
  so the key is the engine's to promise. Tag and documentation only; no
  signature, behaviour or ABI change.

## [0.8.2] — 2026-08-03

### Added
- **Minimal REPL and batch mode** in the CLI: no prompt argument on a
  terminal drops into an interactive loop (model stays loaded; every line is
  an independent, memory-less completion — geist applies no chat template);
  a `-` prompt reads prompts line-by-line from stdin.
- `docs/PI5_BITNET.md` — user guide for the self-contained Pi 5 binary:
  tested configuration, measured cold/warm start (14 s / 0.6 s), RAM
  (~1.7 GB peak, file-backed), cooling, install/update/uninstall, common
  errors, model limits and full model/license attribution (BitNet MIT
  notice now also in NOTICE).

### Fixed
- README no longer claims `<1 MB` for the slim CLI — the shipped static
  binary is ~2 MB.

## [0.8.1] — 2026-08-03

### Added
- **Prebuilt CLI binaries** on the release page (linux-arm64, fully static
  musl): `geist-linux-arm64` (<1 MB, runs any GGUF) and
  `geist-bitnet-linux-arm64` (~1.2 GB, the MIT-licensed BitNet b1.58 2B-4T
  folded in via `.incbin` — one file, no model setup; weights alias zero-copy
  from `.rodata` and are demand-paged like an mmap).
- `docs/MODELS.md` and `docs/BACKENDS.md` — the model table and the
  experimental GPU backends, moved out of the README.

### Changed
- **README rebuilt around the Pi 5 promise**: install command, run example and
  the three measured numbers (4 GB Pi, 15–18 decode t/s, ~1.2 GB download)
  first; engine internals, SDK and comparisons after. Benchmark methodology
  and the full CPU table live in `benchmark/README.md`; "Why C?" moved to
  `docs/ARCHITECTURE.md`.
- `examples/simple_generate.c` stops at the model's EOS token and defaults to
  256 new tokens when no count is given (an untemplated completion can run to
  the end of the context without ever emitting EOS); supports being built with
  an embedded model (`-DGEIST_EMBED_MODEL`).

## [0.8.0] — 2026-08-02

### Added

- **The CI now executes the architecture it advertises**, rather than merely
  compiling it. Metal builds on every PR and runs a `cpu_scalar` parity test on
  the hosted runner's real GPU (#202). Vulkan runs the registry, buffer and
  linear-parity tests on Mesa lavapipe, through the full loader→ICD→SPIR-V
  chain (#204). AVX-512/VNNI runs the shipped configuration under Intel SDE,
  with a dispatch-proof test that makes a silent downgrade to a lower tier a
  hard failure (#203). In each case a skip is a failure, so a missing device or
  feature can no longer read as a pass.
- **A pinned Llama fixture.** SmolLM2-360M is content-pinned by SHA-256 and
  mandatory on both Linux architectures, so the Llama path is executed rather
  than skipped (#194).
- **A per-subsystem coverage ratchet** (`MODE=cov`, versioned baselines,
  0.5 pp tolerance) with a control test that proves the gate fires (#195).

### Fixed

- **A data race in `cpu_x86` under concurrent sessions.** The shared activation
  scratch raced exactly like the `cpu_neon` bug `test_multi_session_parallel_int`
  caught; the cure is ported — per-(backend, thread) workspaces with a
  generation-checked TLS cache. A `MODE=tsan` CI leg now runs the multi-session
  tests on x86_64 (#193).

- **`geist_version_components()` reported the wrong version.**
  `GEIST_VERSION_MINOR` still read `6` while `GEIST_VERSION_STRING` said
  `"0.7.0"`, so a consumer version-gating on the numeric components saw 0.6.0.
  `scripts/check-version.sh` compared the string against the README but never
  against the components, so nothing caught it; it does now, and the check is
  verified to fail when the two disagree.
- **`CITATION.cff` was two releases behind** (`0.6.0`), which is what the
  repository's "Cite this repository" button reads. Also covered by
  `check-version.sh` now.
- **`make lib` works on a musl system.** `-std=c23` (not `gnu23`) defines
  `__STRICT_ANSI__`, under which musl hides everything outside ISO C —
  `strdup` included — so a plain `make lib` on Alpine died on an implicit
  declaration. `mk/target-linux.mk` now defines `_GNU_SOURCE` itself. The
  release workflow passed that flag by hand, which is why CI was green while
  the documented build was not; the workflow no longer needs to.
- **Target detection works without bash.** `mk/detect-target.sh` carried a
  `#!/usr/bin/env bash` shebang while using no bash feature at all. On a
  container without bash it produced an empty `TARGET`, and the build died on
  `mk/target-.mk: No such file or directory`. It is `#!/bin/sh` now, and an
  unusable `TARGET` fails with a message that names the available targets.

### Removed

- `install.sh`. It downloaded `geist` / `geist-bitnet` from this repository's
  releases, which 0.7.0 stopped publishing. Installing a runnable artefact is a
  consumer's business, not the engine's.
- The tracked `geist` symlink at the repository root. It pointed at the CLI that
  0.7.0 removed, so a fresh clone got a dangling link.

### Changed

- **Documentation is application-neutral.** Every reference to a specific
  downstream consumer is gone from the README, `ROADMAP.md`, `docs/`, `tests/`
  and `examples/`: the engine documents its own boundary and says nothing about
  who links it.
- **The README quick start runs.** It opened by telling the reader that the
  interesting artefact was in another repository; `make lib && make
  fetch-bench-model && make run` now produces text in three commands, and
  `make bench` is documented as the reproducer for every published ratio.
- Documentation describes a library throughout. `QUICKSTART.md`, `DEPLOY.md`,
  `ROADMAP.md`, `tests/README.md` and the Makefile's default-goal comment still
  told readers to build and deploy a CLI that 0.7.0 removed — including a
  `./geist` symlink that no rule creates any more.
- Corrected the BitNet Pi 5 row in the README's full-numbers table: decode was
  listed as 17.4 vs 8.2 t/s without a context length, while the recorded run
  (`benchmark/reference_runs.json`, 2026-08-01) reads 17.9 vs 9.1 at a 32-token
  prompt and 15.0 at 512. All three lengths are now listed.

## [0.7.0] — 2026-07-29

### Removed

- **The agent layer, and with it the `geist` CLI.** `agent.h`, `agent_main.h`,
  the four `dynamic_*_v1.h` headers, `json_schema_v1.h`, `tools/geist.c`, the
  `.incbin` model-embedding machinery, their 11 tests and both agent examples
  moved out of this repository.
  geistlib is an inference engine: it loads models and produces tokens, and
  has no opinion about whether a model may act.
- **The `geist-*` and `geist-bitnet-*` release assets**, which were built from
  that CLI, and the Homebrew job that packaged it. Publishing a runnable
  runtime is a consumer's job now. Existing releases are
  immutable, so anything pinned to `v0.6.0` keeps resolving.
- **The agent headers from the `libgeist` SDK asset.** Shipping them from both
  repositories would have put the security boundary on two include paths, an
  invitation to audit one and compile the other.

### Changed

- `make run` builds and runs `examples/simple_generate` — the smallest useful
  program against the STABLE core — instead of the removed CLI.
- The packaged-SDK release gate now compiles `agent_contract_smoke.c`, which
  links exactly the symbols the out-of-tree runtime links, and both linux SDK
  smokes pass `-fopenmp`: the shipped `libgeist.a` is an OpenMP build and a
  real consumer cannot link it without that ([#144](https://github.com/geisten/geistlib/issues/144)).

### Notes

What the engine still owes its consumer is unchanged and enforced:
[API_CONTRACT.md](docs/API_CONTRACT.md) plus `make agent-contract-smoke`,
which fails here rather than in the consumer's build if a signature moves.

## [0.6.0] — 2026-07-28

Tool-calling release. Every fix here was found by measuring a real toolset
rather than by reading the code: an enum-constrained home toolset — the shape
Home Assistant actually offers — could not be driven at all, and three separate
defects were responsible. Tool accuracy on a 40-case calibration corpus goes
from **0.000** to **0.967** (Gemma 4 E2B), **0.714** (BitNet b1.58 2B-4T) and
**0.524** (Llama 3.2 1B), with decline respect at 1.000 and zero off-list calls
throughout. Harness and raw data moved out with the agent layer.

### Added — agent-runtime API contract (#134)

`docs/API_CONTRACT.md` states what the stability tags promise across a release
boundary and pins the symbols an out-of-tree agent runtime links.
`geist_session_peek_logits`, `geist_session_pin_prefix` and
`geist_session_tokenize` are promoted from `EXPERIMENTAL` to **STABLE since
0.6.0** — constrained decoding is impossible without them. `peek_logits` gains
an explicit ownership clause so an accelerator backend can satisfy it by
staging device memory, rather than the promotion freezing a CPU-only design.
Two guards keep the document honest: `scripts/check-api-contract.sh` binds it
to the headers, `examples/agent_contract_smoke.c` binds it to the built
library; both run in CI.

### Added — configurable routing confidence (#138)

`GEIST_ROUTE_MIN_MARGIN` sets the router's winner-vs-runner-up margin per
deployment. The default stays 0.35, which suits a toolset of distinct tools; a
home toolset of near-synonyms scores too tightly to clear it, and measured at
0.0 tool accuracy doubled with decline respect and off-list calls unchanged.

### Fixed — enum arguments were validated but never generated (#135)

A literal substring test could not fill an enum-constrained argument, because
identifiers carry separators speech does not: neither `kitchen_light` nor
`light.kitchen` appears in "turn on the kitchen light". Enum values now match
per alphanumeric run. Scalar enums resolve to exactly one value and **fail
closed on a tie** — two devices fitting equally well is where the runtime must
ask rather than pick one; array enums still collect every mentioned value.

The masked call grammar constrained the tool name and each argument key but
left VALUES free, so a model-written call failed schema validation. Such values
are now decoded along their permitted set with the same primitive that already
constrained the tool name: the model still chooses, the grammar now bounds.

### Fixed — the router scored only a name's first token (#140)

`agent_score_names` ranked options by the logit of the **first token** of the
tool name, so names sharing that token scored identically and could not be told
apart. `TurnOn` and `TurnOff` both begin `"Turn"`; worse, a uniformly prefixed
toolset collapsed entirely — Home Assistant's `HassTurnOn`, `HassTurnOff`,
`HassGetState` and `HassSetBrightness` all tokenize to `"H"`, one score for the
whole menu, so forced routing could not select among any of its tools. Tied
groups are now separated by walking one token deeper on the route session,
costing one forward pass per extra token and only when a tie exists.


## [0.5.0] — 2026-07-23

Streaming release: `dynamic-tools-v1` gains additive delta streaming for
time-to-first-token on Assist-class consumers, the project identity settles on
**geistlib** (engine + demo CLI; the binary, C API and brew formula keep the
`geist` name), and a repo-wide dead-code sweep removes ~1k lines with no
behavior change. The v0.4.0 compatibility contract is unchanged — clients that
do not opt into `stream` get byte-identical behavior.

### Added — dynamic-tools-v1 additive streaming (#116)

- `health.result` now advertises capabilities: `"features":["streaming"]`
  (generic mechanism — unknown entries are client-ignored, absence means no
  capabilities, activation only ever via the matching request field).
- Opt-in `"stream": true` on the conversation request (strict boolean) makes
  `--serve` emit `{"type":"conversation.delta","text":"…"}` frames while the
  user-visible answer decodes, followed by the unchanged, normative
  `conversation.result`. Deltas are valid UTF-8 per frame (split multibyte
  token pieces are carried), display-only, and never produced by internal
  decodes. A client that disconnects mid-stream aborts the generation; the
  daemon keeps serving. Clients without `stream` get byte-identical behavior.
- SDK: additive `on_delta`/`on_delta_ctx` hook on `struct geist_agent`
  (nullptr = off); no `libgeist` ABI change. The hook fires from both
  user-visible answer decoders (`agent_generate_turn` and
  `agent_generate_reply`), never from internal routing/masked-call decodes.
  The delta writer drops malformed UTF-8 bytes and withholds deltas until the
  answer clears the degenerate threshold, so streamed text never contradicts
  the normative result; `--serve` also sets `SO_SNDTIMEO` so a stalled reader
  cannot wedge the daemon. Spec: `docs/dynamic-tools-v1.md`
  §Streaming. Tests: `test_dynamic_stream_unit`.

### Changed — geistlib identity + docs

- Project renamed **geist → geistlib** (repo `geisten/geistlib`); the binary
  (`./geist`, `geist-bitnet`), the C API (`geist_*`, `include/geist.h`) and
  `brew install geisten/tap/geist` deliberately keep their names. README
  reframed to what the project is — an inference engine + a small demo CLI,
  CPU-first with experimental Metal/Vulkan GPU backends — and the tool-use /
  smart-home narrative moved to the consumer repos. `docs/ARCHITECTURE.md`
  gains explicit dependency rules and a per-directory reading map.
- `src/backends/metal/backend.c` split: MSL shader sources → `metal_shaders.h`,
  the objc runtime shim → `metal_objc.h` (pure move — the rebuilt object file
  was byte-identical).

### Removed — repo-wide dead-code sweep (−1051 lines, #124)

- Dead files: `vision_siglip/video_pipeline.{c,h}`, x86 `kernel_catalog.{c,h}`,
  `engine/session.h`, `engine/allocator.h`. Dead surface: the `memory_arena` /
  `geist_arena` family, alloc-per-call sampler wrappers, exec_plan's
  session-level mirror (flags now read directly, stages called directly), and
  a dozen zero-caller exports across io/formats/backends. Six copy-pasted
  routing evidence scans collapsed into one helper. No behavior change; all
  kernels and every backend (NEON, x86, Metal, Vulkan) untouched.

## [0.4.0] — 2026-07-13

First release of the host-neutral `dynamic-tools-v1` runtime artifact. A consumer
pins the protocol id `dynamic-tools-v1` + a binary SHA-256 (published as
`SHA256SUMS`) and verifies with the startup health handshake
(`{"type":"health"}` → `…"status":"ready"`). Start forms: `geist --serve SOCKET`
runs the internal (baked-in) model; `geist -m MODEL --serve SOCKET` an external
one. Compatibility contract: `docs/dynamic-tools-v1.md`.

### Changed — engine core slimmed to inference + `--serve`

- The `geist` CLI is now inference-only plus the resident dynamic-tools daemon:
  `geist -m model "prompt"` (instruct/`--raw` generation) and
  `geist -m model --serve <socket>`. The `agent` and `chat` subcommands and the
  interactive REPL are removed — a served request supplies its own toolset over
  `dynamic-tools-v1`, and one-shot/REPL tool use belongs in a consumer that links
  libgeist.
- The concrete reference tools (`doc_search`, `summarize_file`, `list_dir`,
  `web_search`, `web_fetch`, `remember`/`recall` + the `mind` palace) and the
  `stock_movers` demo tool moved **out of the engine** into the
  downstream consumer. geist keeps the
  tool-use **interface** (`agent.h`, `agent_main.h`, the `dynamic_*_v1.h` set),
  which continues to ship in the libgeist SDK. See geist#110.
- Removed the `make bench-tooling` / `bench-agent` quality benches and their CI
  quality gate, the `tests/bench_agent_eval` harness + `tests/data/agent_eval`
  corpus, the per-tool unit/integration tests, and `benchmark/AGENT_EVAL.md`
  (agent-layer reliability is now measured in the consumer). `docs/agent.md` is
  slimmed to the interface.

### Added — host-neutral dynamic tools v1

- `geist agent --serve` accepts an immutable per-request `tools` array and
  performs correlated `tool.call` / `tool.result` round trips over the local
  socket. The server accepts only the dynamic JSON protocol.
- Removed the Home Assistant REST/token client, registry-push and line-protocol
  adapters, unused protocol-v2 stack, and their migration-only tests/config.
- Added a fixed-memory JSON parser and documented Schema-v1 subset with strict
  name, type, required/optional field, enum, array, bound and duplicate-key
  validation. Unsupported keywords fail request compilation.
- Typed forced calls now cover multiple arguments, numbers, booleans, scalar
  enums and enum arrays. Low-confidence routes clarify; invalid/off-list calls
  never cross the host boundary.
- Added global call/retry budgets, correlated cancellation, HA-owned dynamic
  execution, and the independent `make dynamic-example-host` reference build.
- Added deterministic security/HA suites and a real BitNet + Unix-socket + C-host
  end-to-end transcript under `docs/benchmarks/`.
- Added a model-free dynamic-tools-v1 health handshake and a UI-only Home
  Assistant Config/Reconfigure Flow with validated socket and DE/EN errors.
- Added a polling HA Health entity, automatically recovering Repairs, and
  config-entry diagnostics that expose no paths, addresses or HA content.
- Added zero-queue HA request admission, fresh-socket reconnect semantics,
  at-most-one correlated cancellation, and content-free lifecycle logging.
- Added explicit request language and bounded request context plus HA-owned,
  in-memory conversation history with turn/byte/conversation LRU limits.
- Added the Home Assistant app repository and protected-compatible multi-arch
  scaffold with AppArmor, `/data`-only persistence, protocol healthcheck and a
  non-publishing `aarch64`/`amd64` CI build matrix.
- Added an agent-executable Home Assistant implementation plan for phases 0–8;
  HTTP/REST server requirements are explicitly out of scope.

## [0.3.3] — 2026-07-01

### Fixed — release embedded-build download resilience

- The embedded release builds fetch the ~1.1 GB BitNet GGUF 3× per release; on
  v0.3.3 HuggingFace rate-limited (HTTP 429) and the `--retry-delay 2` was too
  short, failing the x86 embedded job. Bumped to `--retry 5 --retry-delay 15
  --retry-all-errors` so a transient 429 is ridden out.

### Added — BitNet b1.58 2B-4T on x86 (AVX-512): beats bitnet.cpp

- The `cpu_x86` backend now runs **BitNet-2B-4T (I2_S ternary)** end-to-end and
  **beats Microsoft's bitnet.cpp on both metrics** (AMD Ryzen 9 9950X, Zen 5,
  16T, same `ggml-model-i2_s.gguf`): prefill **pp128 884 vs 679 t/s (+30 %)**,
  decode **tg128 77.9 vs 56.5 t/s (+38 %)**. It was previously non-functional on
  x86 (`cpu_scalar` had no I2_S linear nor an F16 lm_head path → zero tokens).
- **Ternary kernels** (`kernel_i2s*`): biased-u8 `VPDPBUSD` dot over the packed
  0.25 B/wt stream (Zen 5 has no s8×s8 VNNI, so codes {0,1,2} + a per-token
  sum-correction), unpacked in-register. The prefill GEMM uses an **x4
  row-interleaved layout** (`i2s_to_x4`) — 4 output rows packed at 2-bit
  granularity per byte so one activation load feeds 4 rows (4× fewer act loads),
  the decisive prefill win.
- **lm_head**: ported the NEON-only **speculative i8-sketch head** (`spec_head.c`)
  to x86 (AVX2 sketch dot + F16C finalists) — reads a ~82 MB subsampled sketch
  instead of the 657 MB F16 table, greedy output **byte-identical to the exact
  f16 dense head**. A `Q8` lm_head (`f16_to_q8w`) is the sampling /
  `GEIST_SPEC_HEAD=0` fallback (`GEIST_Q8_LMHEAD=0` forces exact F16C). On x86 the
  spec-head is gated to the F16 lm_head; Gemma/Llama keep their exact dense
  Q-decode.
- Cross-validated byte-identical to bitnet.cpp on the packed path; the x4 /
  spec-head paths verified against the scalar oracle / exact f16
  (`test_i2s_gemv_unit`, `test_q8w_gemv_unit`). `cpu_scalar` gained I2_S / F16 /
  BF16 linear (the unblock + test oracle).
- **`GEIST_I2S_PAIR=1`** (opt-in, default off): fuses the gate+up / q+k decode
  GEMVs into one OMP region sharing a single activation quant (5 OMP
  regions/layer → 3). Measured **neutral on the 9950X** — the ternary GEMVs are
  already bandwidth-bound (~73 GB/s aggregate) under both active and passive OMP
  wait — so it stays off by default; exposed for hosts where the caller measures
  a benefit (low memory bandwidth, high core count, costly thread wakeups).
  Bit-identical to the unfused path (`test_i2s_gemv_unit`).

### Added — prebuilt linux-x86_64 release binary (AVX-512)

- `release.yml` now also builds a **`geist-linux-x86_64.tar.gz`** — a dependency-free
  musl-static binary with the native AVX-512/VNNI backend (`BACKENDS="cpu_x86
  cpu_scalar"`, `GEMM_PROVIDER=native`). Baseline `x86-64-v3` (Haswell / Zen+) with
  AVX-512 kernels runtime-dispatched via `hw_probe`, so the one binary runs on any
  x86-64-v3 CPU. Pair it with a Gemma / Llama / BitNet GGUF. Windows still not shipped.
- …and a **`geist-bitnet-linux-x86_64.tar.gz`** single-file (BitNet 2B-4T baked in),
  now that the x86 I2_S ternary + spec-head kernels make BitNet fast on x86 — same
  one-file, no-model-argument deal as the arm64 / macOS embedded builds.

### Changed — docs reflect the landed x86-64 (AVX-512) backend

- README, `install.sh` and ROADMAP no longer say "x86 / Windows wait on the AVX
  backend" — that backend has landed. x86-64 Linux (AVX-512) now builds from source
  and is competitive with llama.cpp; prebuilt binaries stay ARM64 for now, Windows
  is still pending. Platform badge updated to `ARM64 + x86-64`.

### Added — AMD x86 (AVX-512) benchmarks vs llama.cpp

- README now reports the `cpu_x86` backend's measurements on an **AMD Ryzen 9
  9950X** (Zen 5, 16C/32T): Gemma 4 E2B Q4_K_M prefill **512 vs 495** (+3.4 %) and
  decode **48.6 vs 44.1** (+10 %); Llama 3.2 3B prefill **351 vs 346** (+1.4 %),
  decode 34.1 vs 34.5 (parity) — geist matches-to-beats llama.cpp on x86. Added to
  the headline table and the system-grouped scoreboard chart.

### Changed — clearer benchmark charts (geist vs the baseline)

- Replaced the prefill/decode/total matplotlib chart with two focused, dependency-free
  SVGs (pure stdlib, numbers straight from JSON, no matplotlib):
  - **`headline_benchmarks.svg`** — a scoreboard of geist's throughput as a ratio of
    the baseline across model × OS, each row its own headline metric + baseline
    (BitNet decode 2.1× bitnet.cpp; Gemma prefill 1.5× llama.cpp; Gemma decode/total
    ~1.1× llama.cpp). Generated by [`benchmark/chart_headline.py`](benchmark/chart_headline.py).
  - **`pi5_total_tps.svg`** — total tok/s, geist vs llama.cpp (CPU + OpenBLAS) on a
    Pi 5, honest about the long-prompt tie. Generated by
    [`benchmark/total_tps.py`](benchmark/total_tps.py).

### Changed — generation stops on a sentence by default (no `-n` needed)

- `geist <model> "prompt"` treats the token budget as a **soft target**: it rounds
  up to the next sentence end (capped at 2×) instead of cutting mid-word. A base
  model on a bare completion prompt emits no end token, so the old hard 64-token
  default ended like "…Paris is also known". `-n N` is still an exact **hard** cap.
  So you never *need* to pass `-n` for a clean result.

### Fixed — decode HTML entities in fetched/searched text

- `webfetch_strip_html` now decodes HTML entities (`&amp;` → `&`, `&lt;`/`&gt;`,
  `&quot;`, `&#39;`, `&nbsp;`, numeric `&#NN;`/`&#xHH;`), so `web_search` titles and
  `web_fetch` page text read cleanly instead of showing raw `&amp;`. Bare `&` in
  running text is left untouched. Covered by `test_webfetch_unit`.

## [0.3.1] — 2026-06-28

### Added — one-command install + prebuilt single-file binaries

- **`install.sh`** — `curl -fsSL …/install.sh | sh` detects your platform,
  downloads the single-file `geist-bitnet` (BitNet 2B-4T baked in), and drops it on
  your PATH. One command, one file, nothing else to fetch.
- Releases now ship a `geist-bitnet-<platform>.tar.gz` by default — a self-contained
  binary that runs with **no model argument**. The release workflow's
  `EMBED_MODEL_URL` defaults to the BitNet GGUF; clear it to stop shipping them.
- README "Run it now" is now genuinely copy-paste runnable and split into two clear
  paths: ① the single baked-in binary (no model file), ② the < 1 MB engine + a GGUF
  (one model file runs on every platform). All download links are direct.

### Changed — `geist agent` sensible defaults (force + trace on)

- `geist agent` now **forces the tool call by default** (`GEIST_FORCE_CALL=0` to
  opt out). The bundled models (BitNet 2B-4T, Gemma 4 E2B) aren't tool-trained, so
  without forcing the agent would never run a tool — making it the default means
  `geist agent "<request>"` just works, no env prefix. (`geist chat` never forces.)
- The per-step trace (`· routing → calling → running → observed → answering`) also
  prints by default for `geist agent` — it goes to stderr, so the answer on stdout
  stays clean and piping is unaffected. Silence it with `GEIST_AGENT_TRACE=0`.
  `geist chat` is the opposite: trace stays **opt-in** (`GEIST_AGENT_TRACE=1`) so a
  conversation is quiet by default.

### Added — name the embedded binary (`EMBED_NAME`)

- `make EMBED_MODEL=... EMBED_NAME=geist-bitnet` names the self-contained binary
  distinctly. An embedded binary takes **no model-path argument** (the model is
  baked in), unlike the plain `geist` — giving it its own name avoids the "which
  one needs a model?" confusion. Defaults to `geist` (unchanged for normal builds).

### Added — single-file builds get the agent + chat

- A `make EMBED_MODEL=...` build is no longer text-only: `geist agent <request>`
  and `geist chat` now drive the baked-in model (no model-path argument). One
  self-contained binary generates text *and* runs tools — demoed with BitNet
  b1.58 2B-4T embedded, generating and summarizing a file on a Raspberry Pi 5.
  `geist_agent_main` takes the embedded GGUF bounds; `agent_main_parse_args`
  gained `want_model` to drop the model positional when it is baked in.

### Changed — memory tools are opt-in (`GEIST_MIND_DIR`)

- The agent's default toolset dropped from 7 to 5: `remember`/`recall` are
  included only when a palace is configured (`GEIST_MIND_DIR`). On weak models the
  router scores tool names, and the two memory tools made common requests (e.g.
  "summarize report.md") mis-route to `recall` on some CPU backends (BitNet/NEON).
  Fewer default tools → robust routing across backends. `geist chat`'s
  `/remember`,`/recall` slash commands are unaffected (they call `mind.h` directly).

### Changed — bounded chat context (sliding window)

- Multi-turn `geist chat` now evicts the oldest turns once the transcript passes
  a budget (`agent_compact`): it keeps the protected system prompt and the most
  recent whole turns, down to a target size. This bounds per-turn re-prefill (a
  long chat stays O(n) instead of O(n²)) and replaces the old hard "context full"
  stop — the model forgets the evicted turns. The eviction point is a documented
  hook for folding the dropped span into a running summary later (the summarizer
  already exists). Tunable via `GEIST_AGENT_CTX_BUDGET` / `_TARGET`.

### Added — `geist chat` + memory tools

- The interactive chat is now the **`geist chat`** subcommand, rebuilt on the
  agent engine: `geist_agent_run` gained a `conversation` flag that keeps the
  transcript across turns. It carries the full toolset and the memory palace, and
  inherits the engine's chat-template handling (the old hand-rolled inline framing
  and its stop-marker leak are gone). Removed the `geist_chat` binary.
- The memory palace is now model-callable via two tools (`tools/agent_memory.h`):
  `remember(text)` (title auto-derived from the first line — single-arg so it
  works under a forced call) and `recall(slug)`. "Search my notes" reuses
  `doc_search` over `$GEIST_MIND_DIR`. Both tools are in `geist agent` and `geist
  chat`; when memory is present the notes index is injected so `recall` is usable
  one-shot. The `/remember`, `/recall`, `/notes` slash commands stay as the
  reliable manual path on un-tool-trained models.

### Changed — one agent CLI, folded into `geist`

- The tool-use agent is now the **`geist agent`** subcommand of the main CLI, not
  a separate binary. `geist <model> <prompt>` generates text; `geist agent <model>
  <request>` runs the whitelist-gated tool loop (list_dir, summarize_file,
  doc_search, web_search, web_fetch). Both honour `GEIST_FORCE_CALL=1` and
  `GEIST_AGENT_TRACE=1`. This removes the "which binary?" footgun — `./geist` no
  longer silently ignores the agent env vars.
- Removed the `geist_agent` and `geist_shell` demo binaries (merged into the
  subcommand). The reusable engine `agent_main.h` gained a tool-builder callback
  (so a tool's ctx can reference the loaded model) and now owns the force-call +
  trace env knobs, so every CLI built on it behaves identically.

## [0.3.0] — 2026-06-23

### Added — on-device tool-use agent

- A bounded, whitelist-gated tool-use loop lets a small local model read files and
  search the web **in-process**: `list_dir`, `summarize_file`, `doc_search` (local
  keyword RAG, paragraph-granular + overlap-scored), and `web_fetch` (curl, no
  shell, scheme + host gated). Tool routing and the JSON call structure are forced
  from outside the sampler, so even untrained 2 B models drive the tools reliably.
  Full design and security model in `docs/agent.md`.
- A reusable agent CLI engine (`tools/agent_main.h`) with the `geist_agent`
  reference CLI, plus an interactive chat mode with a file-based memory palace.

### Changed — CI hardening

- New jobs: ASan + UBSan unit tests, a musl/Alpine build (tests what we ship),
  real-model integration + e2e tests, and ccache-cached compilation.

### Changed — speculative output head is now on by default

- The speculative int8-sketch lm_head (below) now defaults **on** for greedy
  decode on an eligible tied head; `GEIST_SPEC_HEAD=0` forces the exact dense
  head. Verified byte-identical greedy output on Gemma 4 (Q6_K head, 256 K
  vocab) and BitNet (F16, 128 K) for ~+5 % Pi 5 decode. Non-greedy sampling,
  ineligible dtypes, and non-NEON/dotprod hosts always fall back to the dense
  head, so the default change is a no-op there.

### Added — speculative int8 output head for BitNet 2B-4T decode (Pi 5)

- `GEIST_SPEC_HEAD=1` enables a speculative lm_head for large **tied F16**
  embeddings (`src/archs/transformer/forward/spec_head.c`). On Microsoft's
  BitNet-b1.58-2B-4T `I2_S` model the F16 lm_head is ~656 MB read per token —
  ~50 % of decode. The spec head keeps a stride-4 int8 **sketch** of the
  embedding (`[vocab, hidden/4]`, ~82 MB), rough-ranks the whole 128 K vocab with
  one SDOT pass, takes the top-512, and computes **exact f16 logits** for only
  those. Greedy output is byte-identical to the dense head (the deciding logits
  are unquantized); opt-in, non-greedy sampling falls back automatically.
- Result on a Raspberry Pi 5 (A76, `tests/bench_perf_sweep`, 2 t, 2.4 GHz):
  BitNet 2B-4T `I2_S` **decode 9.83 → 17.4 tok/s** — ahead of both other engines
  built and run on the same box: Cougar (Rust + `ea` SIMD) **12.3** and bitnet.cpp
  **8.2** (~2×). See `benchmark/TERNARY_BITNET.md` for the same-box three-engine
  comparison and the layer-matmul kernel shapes (4-row, fused gate+up) that were
  tried and reverted as A76 regressions.
- The spec head also covers **block-quantized** tied lm_heads (Q3_K/Q4_K/Q5_K/
  Q6_K/Q8_0). Phase 3 builds a one-row view of the embedding and calls the *same*
  `linear_m1` the dense head uses (W6A8 for Q6_K), so finalist logits are
  **bit-exact** — no f32-dequant approximation. The only approximation is sketch
  recall (which rows become finalists), so `GEIST_SPEC_TOPK` is now vocab-aware
  (512 for ≤200 K, 4096 above) and tunable along with `GEIST_SPEC_STRIDE`. On
  **Gemma 4 E2B** (tied Q6_K, 256 K vocab) greedy is byte-identical to the dense
  head at TOP_K 4096 for **+5 % decode** (6.94 → 7.29 t/s, 4 t; or +14 % if a
  smaller TOP_K is allowed to diverge). Opt-in, greedy only.

## [0.2.1]

### Added — embed a model into the binary (single-file deploy)

- `geist_model_load_from_memory(data, size, be, &model)` (`geist.h`): load a GGUF
  that is already in memory. Weights are aliased zero-copy from the buffer (the
  caller keeps it alive); the GGUF must carry its own tokenizer; text-only. Backed
  by a new internal `gguf_open_memory`.
- `make EMBED_MODEL=path/to/model.gguf` bakes a GGUF into the `geist` CLI via an
  `.incbin` stub (portable ELF + Mach-O), so the engine *and* the model ship as
  one binary; the CLI then takes only a prompt. For small models — the binary
  grows by the model size (build warns past ~1.5 GB).
- This **completes the "Single-file app + model" roadmap item** — deployment can
  be literally one binary. Scoped to small models on purpose: beyond the build
  cost, the default `-mcmodel=small` PC-relative addressing range caps an
  embedded blob (~2 GB on x86-64, ~4 GB on AArch64), and a standalone GGUF is
  already `mmap`-aliased zero-copy at runtime — so for large models the
  separate-file form keeps engine/model decoupling with no RAM penalty. A
  streamed `geist pack` format for large models remains future work.

### Added — per-platform mmap hints for large models

- The weight `mmap` now applies best-effort `madvise` hints. Linux:
  `MADV_HUGEPAGE` (transparent huge pages → fewer TLB misses on the big weight
  tables — a real win on **4 KB-page Linux servers**; `GEIST_NO_HUGEPAGE=1` to
  disable). All platforms: opt-in `MADV_WILLNEED` prefault via
  `GEIST_MMAP_PREFETCH=1` (steadier first-token latency, bigger upfront read).
  Honestly measured: **no effect on the Raspberry Pi 5** — it already uses 16 KB
  base pages and has no THP, so the TLB win is moot there; the lever is for
  4 KB-page Linux. No regression (Pi pp256 unchanged within noise).

### Added — docs
- `docs/QUICKSTART.md`: a two-minute guide to running the CLI and embedding the
  library (complete copy-paste C program, clean EOS stop, single-file build,
  performance knobs), linked from the README.

## [0.2.0]

### Changed — public API split by audience (source-compatibility break)

`geist.h` is now the **minimal surface to run a model** (backend → model →
session → `set_prompt` → `decode_step` → `token_to_str`). Helpers and advanced
features moved to new headers. **No signatures changed**; declarations moved.

- `geist_util.h` *(new)* — chat / advanced apps: special tokens
  (`geist_model_eos_token` / `_bos_token` / `_token_by_text`),
  `geist_session_tokenize` / `_prefill_tokens`, multimodal `attach_audio/image/video`,
  `pin_prefix`, `peek_logits`, `decode_speculative`, telemetry, and
  `geist_backend_supports_op`.
- `geist_types.h` *(new)* — backend-author territory: low-level tensor / op /
  dtype / buffer / quant types (previously in `geist.h`). Pulled in by
  `geist_backend.h` and `geist_weight.h`.

**Migration:** pure text generation needs no change (`#include <geist.h>` still
compiles; `examples/simple_generate.c` and `tools/geist.c` are untouched). Apps
using special tokens / multimodal / speculative / telemetry add
`#include <geist_util.h>`. Backend/tensor-type code adds `#include <geist_types.h>`.

### Added
- `geist_model_eos_token`, `geist_model_bos_token`, `geist_model_token_by_text`
  (`geist_util.h`): special-token ids from GGUF metadata, so a chat app stops on
  `tok == eos` instead of string-matching decoded output.

## [0.1.3]

### Changed
- Linux release artifact is a fully static **musl** binary (was glibc): ~40 %
  smaller download, portable across any aarch64 Linux with no libc dependency.
  Both release binaries are stripped.

## [0.1.2]

### Changed
- Parallelized the O(n²) int8 SDPA prefill core (bit-exact) — flat Pi 5 prefill
  curve (pp1024 +35 %). Benchmark re-measured honestly (Pi thermal-throttling
  artifact corrected; matched cool-start protocol; llama.cpp leads Pi prefill).

## [0.1.1]

### Added
- `geist_gemm` abstraction + BLAS-free native NEON fp32 + vendored FFT; fully
  dependency-free static ARM build; CI release matrix; int8-kernel tuning; CLI.

## [0.1.0]

First public release.

### Added
- C23 inference runtime with a stable C ABI (`include/geist.h`), per-symbol
  `STABLE` / `EXPERIMENTAL` stability tags.
- Backends: `cpu_neon` (Apple Silicon + ARM64, OpenMP-parallel kernels) and
  `cpu_scalar` (portable reference). `cpu_x86` is a policy skeleton.
- Quantization: GGUF `Q4_0/Q8_0`, k-quants `Q3_K/Q4_K/Q5_K/Q6_K`, IQ-quants
  `IQ2_S/IQ3_S`, and ternary `TQ2_0` for 1.58-bit models. Zero-dispatch kernel
  binding: every tensor is bound to a specialized kernel at load time.
- Transformer architecture (Gemma 4 family) with RoPE, GQA attention, KV cache,
  and per-session sampler (greedy / top-k / top-p / temperature).
- KV-cache quantization modes (INT8, KIVI), AWQ scale loading, and an n-gram
  speculative-decode path (all `EXPERIMENTAL`).
- Native multimodal: Conformer audio tower (`attach_audio`) and SigLIP vision
  tower for image/video soft-token prefixes (`attach_image` / `attach_video`).
- Build system with per-target/per-mode segregation (`mac`, `mac-omp`, `pi5`,
  `linux`/generic ARM64), `debug`/`asan`/`perf` modes, and on-demand reference
  model fetch (`make fetch-model`).
- Test suite (exit-code contract, `_unit`/`_int`/`_e2e` tiers) and a
  reproducible perf benchmark harness (`make bench-small`).
- `examples/simple_generate` demonstrating the stable text-generation core.

[Unreleased]: https://github.com/geisten/geistlib/compare/v0.10.6...HEAD
[0.10.6]: https://github.com/geisten/geistlib/compare/v0.10.5...v0.10.6
[0.10.5]: https://github.com/geisten/geistlib/compare/v0.10.4...v0.10.5
[0.10.4]: https://github.com/geisten/geistlib/compare/v0.10.3...v0.10.4
[0.10.3]: https://github.com/geisten/geistlib/compare/v0.10.2...v0.10.3
[0.10.2]: https://github.com/geisten/geistlib/compare/v0.10.1...v0.10.2
[0.3.0]: https://github.com/geisten/geistlib/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/geisten/geistlib/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/geisten/geistlib/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/geisten/geistlib/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/geisten/geistlib/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/geisten/geistlib/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/geisten/geistlib/releases/tag/v0.1.0
