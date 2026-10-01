# Ternary (BitNet b1.58 / TQ2_0) — Pi 5 performance work

**Goal:** geist's ternary decode *and* prefill on a Raspberry Pi 5 (Cortex-A76,
SDOT, **no i8mm**) at or above `MAX(bitnet.cpp, llama.cpp)` on the same model.

**That goal is met on `i2_s` and missed on `TQ2_0`.** Measured 2026-08-01 with
all three engines on `bitnet_b1_58-large` TQ2_0, one `make bench` run, each
engine started under the 56 °C gate:

| | geist | bitnet.cpp | llama.cpp |
| :-- | --: | --: | --: |
| decode, 32-token context | 46.31 | 43.62 — 1.06× | 49.09 — **0.94×** |
| prefill, 512 tokens | 128.08 | 74.81 — 1.71× | 51.59 — 2.48× |

Prefill clears both by a wide margin. Decode appears 6 % behind llama.cpp — but
that row was measured with a default that is **silently wrong on this model**,
and the honest figure is parity. See below.

### The decode row is a defect, not a deficit

`make bench` runs the shipped defaults, which here include the speculative
output head. On this model that head **changes the output**: greedy generation
matches the dense head for 28 tokens and then diverges. Both paths are
deterministic (three identical runs each), so this is a recall miss, not noise —
and the sketch has no fallback, so it never notices.

Measured across configurations, same prompt, 48 tokens:

| configuration | decode t/s | output |
| :-- | --: | :-- |
| default (sketch, stride 4, topk 1024) | 47.15 | **diverges** |
| `GEIST_SPEC_TOPK=4096` | 34.38 | exact |
| `GEIST_SPEC_STRIDE=2` | 45.04 | exact |
| `GEIST_SPEC_HEAD=0` | **48.42** | exact by construction |
| llama.cpp | 49.09 | — |

Among the configurations that produce the right tokens, turning the sketch
**off** is the fastest — so on this model the speculative head is a loss in
every correct setting, and the honest ratio against llama.cpp is **0.99×**,
parity rather than a deficit.

**Root cause: the sketch resolution is `SD = H / stride` with a fixed stride of
4.** The 2B-4T has `H = 2560`, so `SD = 640` and topk 1024 has margin to spare.
This model has `H = 1536`, so `SD = 384` — a proportionally coarser sketch, and
the recall margin verified on the 2B-4T no longer holds. Two independent knobs
confirm it: raising topk to 4096 *or* halving the stride restores exact parity,
and both cost throughput.

`spec_dtype_ok()` gates on dtype and vocabulary size. Neither is the variable
that decides recall.

Recorded in `reference_runs.json`; spread 1.0 %. The row stands as what the
shipped defaults produce — that is what a reproducer would get — with this
section as its reading.

**Status: measured on the Pi 5.** geist decodes the canonical 2B-4T `i2_s` at
**18.05 t/s vs bitnet.cpp's 9.04** (**2.0×**) **at a 32-token prompt** and
**15.13 t/s** (**1.67×**) **at 512 tokens** — both halves from one `make bench`
run on the same board (2026-08-01, frozen protocol, spread 3 %, recorded in
`../reference_runs.json`). Prefill at 512 is 44.0 t/s (1.17×). The one open gap
is a canonical 2B-4T **TQ2_0** GGUF (only `i2_s` ships upstream).

Quote that headline with its context length or not at all — decode falls ~16 %
from a 32-token prompt to a 512-token one, so the same build reads anywhere
between 18.1 and 15.1 t/s depending only on where you measure it.

The pair must stay a pair: geist's half must not be refreshed without
re-running bitnet.cpp beside it in the same `make bench` run. (The previous
headline, 17.4/8.2, suffered exactly that drift and is superseded.)

---

## Verified so far (2026-06)

1. **geist runs a real BitNet ternary model end-to-end.** Previously the TQ2_0 /
   TL1 path was only *synthetically* unit-tested (`tests/test_tl1_parity_unit.c`).
   Confirmed on `gianni-cor/bitnet_b1_58-large-TQ2_0` (0.7 B, 217 MB, real
   ternary weights, `general.architecture = bitnet`): geist loads the arch
   (generic GGUF-driven populator, SubLN + activation detection in
   `arch_family.c`) and the weights, and `bench_perf_sweep` drives the compute
   path to stable numbers.

2. **A76 kernel selection (default).** For TQ2_0 the resolver binds the **SDOT
   `q8a`** path for both decode (`cpu_neon_w_tq2_0_q8a_m1`) and prefill
   (`cpu_neon_w_tq2_0_q8a_mN`). The TL1 LUT path is opt-in: `GEIST_TL1=1`
   (decode) / `GEIST_TL1_PREFILL=1` (prefill). A code comment records that on
   the A76 SDOT prefill already *beats* TL1 (33.6 vs 21.0 t/s seq128, 2B-4T).

3. **The SDOT kernel is already well-tuned** (`kernels/tq2_0.c`): `vdotq_s32`
   with an "unbiased" trick (skips the per-element −1), two accumulators for
   dual-issue, and an `mt4` variant that reuses each weight tile across 4 tokens
   in prefill. No naive low-hanging fruit in the inner loop.

4. **Tokenizer for *older* BitNet models — supported.** `1bitLLM/bitnet_b1_58-*`
   ship a llama **SentencePiece *unigram*** tokenizer (`scores` + `token_type`,
   **no `merges`**). geist handles this via `GGUF_TOK_MODE_UNIGRAM` (the
   llama.cpp merge-by-score algorithm in `src/engine/gguf_tokenizer.c`), so text
   I/O works directly from the GGUF (no `tokenizer.bin` needed): an embedded
   `./geist` completes "The capital of France is Paris…" on `bitnet_b1_58-large`.
   Coherence/quality is then purely a compute (TQ2_0/i2_s) question, not a
   tokenization one.

### Apple reference numbers (NOT the goal hardware — do not transfer to A76)

M1 Max, `large` model, real weights: SDOT decode ~90 tps vs **TL1 decode ~68
tps** — i.e. on this Apple setup TL1 decode is *slower* than SDOT, contradicting
an older "~2× decode" code comment (measurement was noisy: live desktop). Listed
only to flag that the TL1↔SDOT trade-off must be **measured per platform**; it
inverts between Apple and the A76.

---

## Decode against context length (2026-08-01)

The headline number above depends entirely on where it is measured, so here is
the curve. One `bench_perf_sweep` run per column, `--seq-lens 32,128,512
--decode-n 64 --warmup 64 --repeats 10`, mean-of-10 from a cool start (47 °C,
load ~0.0, `geist-home` and `ollama` stopped for the duration):

| prompt tokens | decode t/s | spread | prefill t/s | total t/s |
| --: | --: | --: | --: | --: |
| 32 | **18.28** | ±0.7 % | 49.77 | 23.17 |
| 128 | 17.62 | ±1.7 % | 50.51 | 31.14 |
| 512 | 15.25 | ±0.6 % | 46.45 | 37.85 |

**Decode falls 17 % from a 32-token prompt to a 512-token one; prefill stays
flat.** The weights read per token are identical at every point (see the budget
below), so the difference is attention over the longer KV.

Any decode number from this model must name its context length, or it is
unfalsifiable — the same binary reads 18.3 or 15.2 t/s.

### Where the 17 % used to be 29 %

Before the attention core was parallelized over heads as well as query
positions, the same sweep read:

| prompt tokens | decode t/s (before) | after | Δ |
| --: | --: | --: | --: |
| 32 | 17.96 | 18.28 | +1.8 % |
| 128 | 16.77 | 17.62 | +5.1 % |
| 512 | 12.76 | 15.25 | **+19.5 %** |

Decode passes `n_q == 1`, so a `parallel for` over query positions ran on one
thread while the heads went serially. The tell was that the cost of growing
context did not move with the thread count at all — 22.87 ms/token at 1 thread
against 22.67 at 3, a factor of 1.01, while everything else in decode scaled
1.77×. KV bandwidth could not explain it either: 37.5 MiB per step at ~10 GB/s
is 3.9 ms, not 22.7. `collapse(2)` halved that portion to 10.87 ms/token.

The residual 17 % is what attention still costs after parallelizing. The gain
grows with context, so this curve is worth re-measuring whenever the attention
path changes.

Raw runs: `~/bench-geistlib/stride/2026-08-01_pi5_bitnet-2b4t-i2s_seqlen-sweep*.log`
(`_seqlen-sweep` = before, `_1thread` = the thread-invariance test,
`_head-parallel` = after).

## Per-token byte budget (2B-4T `i2_s`)

Decode on this model is memory-bandwidth bound, so the question "where do the
bytes go?" decides which optimizations can pay at all. Counted **statically from
the GGUF tensor table** (via geist's own reader — gguf-py cannot parse `i2_s`,
type 36), assuming each weight is touched once per token at m=1:

| Read per decode token | MB | share |
| :-- | --: | --: |
| ternary `blk.*` (I2_S) | 497.0 | **86 %** |
| speculative sketch table (V × H/4, int8) | 78.3 | 13 % |
| phase-3 verify, top-1024 rows of the F16 head | 5.0 | 1 % |
| norms (F32, all layers) | 1.7 | <1 % |
| **total** | **582.0** | |

For contrast, the model **on disk** is 1124.8 MB, of which `token_embd.weight`
alone is 626.2 MB (F16, 55.7 %) — it is tied, so a dense head would re-read all
of it every token.

Three things follow, and they bound what is still worth trying:

1. **The output head is done.** 626.2 → 83.3 MB is a **7.5×** cut, and it ships
   on by default (`spec_head.c`, `GEIST_SPEC_HEAD=0` disables). It went from the
   largest single item to a small one. The 78.3 MB sketch table is also 78.3 MB
   of *resident* RAM — a deliberate trade of footprint for bandwidth, which is
   affordable on a 4 GB board only because the F16 table itself stays mmap'd.

2. **86 % of the traffic is ternary weights at ~1.6 bpw** — effectively the
   floor for the format. Going lower means a different format or sparsity, i.e.
   research, not tuning. Any bandwidth idea should be sized against this number
   before it is built.

3. **Weight streaming / prefetch cannot pay here.** geist already keeps the
   tiered residency that disk-streaming engines are built around: mmap-alias is
   the default and leaves lookup-only tables disk-backed (see the storage-mode
   note in `arch_state.c`). What remains prefetchable is a few KB of row lookups
   against ~582 MB of dense per-token traffic. Decode is DRAM-bound, not
   I/O-bound; the lever is fewer bytes, not earlier reads.

The one untested knob is `GEIST_SPEC_STRIDE` (default 4): raising it to 8 halves
the sketch table to 39 MB, about −7 % of per-token traffic. **It must be gated on
token parity, not on t/s.** A coarser sketch loses recall, and a recall miss is
*silent* — there is no per-token fallback to the dense head, so the true argmax
simply never gets computed and the trajectory diverges. `SPEC_TOPK` was already
raised 512 → 1024 for exactly that margin.

Measure it at **short** context. Weight traffic is the same per token at every
context length, so attention dilutes a traffic saving the longer the prompt
gets — a real −7 % would half-vanish at 512 tokens and read as noise.

---

## The scalar oracle does not apply here

Elsewhere `cpu_scalar` is the correctness oracle: other backends must reproduce
its greedy output bit for bit. On ternary weights that check does not hold, and
it is not supposed to — the two paths bind different arithmetic. `cpu_scalar`
dequantizes each row to fp32 and dots against fp32 activations (W2A32);
`cpu_neon` binds `cpu_neon_w_i2_s_q8a_*`, int8 activations against native
ternary weights (**W2A8**).

8-bit activations are part of the BitNet b1.58 definition rather than an
approximation of it, so the faster path is the one computing the intended
scheme and the "reference" is the outlier. Measured on 2B-4T `i2_s`: identical
for 36 tokens, then the 37th lands on a near-tie and they separate. The same
comparison on Q4_K stays bit-identical at 60 tokens, so a divergence outside
ternary is a real bug.

Consequence for benchmarking: **do not use `cpu_scalar` to validate a ternary
kernel change.** Compare against the previous build of the same backend, which
is what the perf work here does. See
[`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) for the general rule.

## Ternary-Bonsai-2-27B on the M1 Max (2026-09-22)

PrismML's ternary Qwen3.8-27B, `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.21 GB,
sha256 `3907dc16…2ec1`): `PQ2_0` projections, embeddings and head, weights
folded into a blockwise Walsh-Hadamard basis (`prism.hadamard.*`). Unlike
BitNet it was not trained with int8 activations — see *Correctness* for why
the A8 kernels are still fine here. Reference engine: PrismML-Eng/llama.cpp
`01ae597`, the only other runtime that reads the format.

### Throughput, pp512 / tg64

| | prefill t/s | decode t/s | RSS |
| :-- | --: | --: | --: |
| geist CPU, first cut (row GEMV, dequant trampoline) | 8.8 | 6.2 | 8.0 GB |
| geist CPU, x8 GEMV + x8 prefill, m_max 64 | ~11 | 9.8 | 14.4 GB |
| geist CPU, same, `GEIST_M_MAX=128` | 14.5 | 9.9 | 14.7 GB |
| geist metal, first cut | 78–82 | 11.2–11.5 (13.0 at 32 ctx) | 2.5 GB |
| **geist metal, table GEMV + parallel DeltaNet decode** | **84–111** | **15.5–17.5** (**19.3–19.5** at 32 ctx) | 2.5 GB |
| PrismML fork, CPU (`-ngl 0`) | 22.4 (see the CPU note below) | 0.45 | |
| PrismML fork, metal (`-ngl 99`), same window as the row above | 110.7 ± 1.7 | 16.65 ± 0.3 (from empty ctx) | |

geist: `bench_perf_sweep --seq-lens 512 --decode-n 64 --warmup 1 --repeats 2-3`
(decode is measured after the 512-token prompt); fork: `llama-bench -p 512
-n 64`, whose tg64 starts from an empty context, so compare it with geist's
32-context figure. The first row ran behind a load < 3 gate; the rest shared
the Mac with other agents' jobs (load 5-8 for the metal rows, up to 30 for
the CPU x8 rows — read those as relative). The fork has no ARM decode kernel
for `PQ2_0` (0.45 ± 0.38 t/s is its generic path), and its CPU prefill likely
leans on llama.cpp's GPU op offload for large batches.

Kernel view, one 27B FFN matrix (17408 × 5120), 8 distinct copies so the
working set is DRAM, not the 48 MB SLC:

| format | bytes | decode m=1 | effective | prefill m=64 |
| :-- | --: | --: | --: | --: |
| F32 | 356 MB | 12.6 ms | 28 GB/s | 15.2 ms |
| F16 | 178 MB | 2.33 ms | 77 GB/s | 12.1 ms |
| PQ2_0 row | 24 MB | 0.56 ms | 42 GB/s | 18.2 ms |
| **PQ2_0 x8** | 24 MB | **0.21 ms** | **110 GB/s** | 16.8 ms |

Decode is byte-bound, so ternary pays off there: 11× F16 and 60× the F32
path on the same matrix. Prefill is compute-bound, F16/F32 run on AMX through
Accelerate SGEMM, and `PQ2_0` currently goes through the same SGEMM after a
dequant pass — no ternary advantage there yet.

### Correctness

- Against the fork (`llama-eval-callback`, prompt "Hello"): the
  inverse-rotated embedding row matches to every printed digit, every layer-0
  intermediate through `ssm_out` and the FFN agrees to < 1 %, final hidden
  within 2 %. Signs, the grouped-value permutation and their order are the
  fork's.
- Chat prompt (`test_bonsai_e2e_int`): prompt ids, next-token top 5 and the
  first 24 greedy tokens equal the fork's on CPU and on Metal. The top logit
  sits ~0.9 below the fork's (25.49 vs 26.37 — fork CPU and Metal agree with
  each other), which is where the two drift apart after token 24.
- A8 vs A32: geist with the `PQ2_0` GEMVs forced onto the fp32 trampoline
  produces the same 64 greedy tokens and the same top-5 as the per-row int8
  kernels (top logit Δ 0.002); `cpu_scalar` agrees with the A32 build to four
  digits. The Hadamard rotation is what makes per-row absmax int8 safe on a
  model trained with full-precision activations.

### Where the time goes

- CPU prefill is AMX SGEMM: the x8 prefill path dequantizes 128-row tiles
  from the x8 copy with NEON and SGEMMs each per thread (901 / 1002 / 1134
  GFLOP/s at m = 64 / 128 / 256 on the FFN matrix, the trampoline did
  628 / 765 / 676). A `sample(1)` profile leaves ~2k of ~50k samples in
  serial code; the rest is SGEMM plus waits where 8 threads share the two
  P-cluster AMX units. The host DeltaNet now sub-chunks, so `GEIST_M_MAX=128`
  costs it nothing; the default stays 64 until a quiet cross-model A/B.
- Metal decode, ~49 ms per token at 32-token context (subtractive profile,
  `GEIST_METAL_PROFILE=1` + `GEIST_SKIP_*`): PQ2_0 GEMVs ~38 ms, DeltaNet
  ~3, Hadamard 0.8, RMS norm 0.47. The norm and the F32 alpha/beta GEMVs
  together cost 3.9 ms until both kernels were widened from 256 to 1024
  threads with a simd_sum reduction (decode 19.29 -> 19.96 t/s at 8 ctx,
  16.55 -> 16.87 at 512, prefill unchanged): one row per threadgroup is
  all the parallelism a decode step has, so they were latency-bound on an
  otherwise idle GPU. A
  dependent dispatch costs ~3 us, so the ~1300 per token are not the
  floor. The GEMV sits within ~10 % of a read-only probe of its own access
  pattern (40 ms; 27 ms without the activation loads).
- An aligned device layout was built, measured and then deleted: regrouping
  each row's blocks by 8 into [8 x half d][8 x 32 codes] = 272 bytes put the
  codes on 8-byte instead of 2-byte boundaries and bought 4 % decode at 32
  ctx (19.4 -> 20.2 t/s), 1.5 % at 512, nothing on prefill — for 13 GB of
  RSS (2.5 -> 15.5), because the source mmap is read-only so the repack
  cannot be in place and reading it makes the file pages resident. The GEMV
  was closer to the bandwidth limit than the alignment suggested. Kept here
  as the record; the three kernels and the second layout are gone.
- The decode GEMV is latency-bound, not bandwidth-bound: the same 27B shapes
  run at 172 GB/s in PQ2_0 but 329 in Q8_0, and 4x the weight bytes only
  doubles the time. A thread reads 128 bytes of x per iteration against
  R*8 bytes of weights, so activations move 15/R times the weight bytes —
  90 MB of (cached) x against 24 MB of weights per ffn gate/up call at
  R = 4. Hence matvec_pq2_n8 (R = 8) for wide projections, and hence a
  smaller weight format would buy little: all 26.87 G weights are strictly
  ternary (the +2 code never occurs), so 5 trits per byte would cut 18 % of
  the weight bytes but only ~6 % of the traffic.

### CPU prefill against the fork

Re-measured head to head at pp256, one window, load < 3 at the start, same
GGUF: **fork 17.35 t/s, geist 13.6** (`GEIST_M_MAX=128`; 11.4 at 64, 11.8
at 256, 11.2 at 512). The pp512 rows above compared each tool at its own
default and overstate the gap — treat 27 % as the number to beat, not 55 %.

The gap is not the GEMM. Our tiled prefill runs the 27B FFN matrix at
1002-1134 GFLOP/s (m = 128-256), which would be ~18-21 t/s if the matmuls
were everything; end to end we get 734 GFLOP/s, so ~30 % goes elsewhere.

What the fork does on CPU, read from its source at 01ae597: PQ2_0 declares
`vec_dot_type = Q8_K` and ARM has no NEON `ggml_vec_dot_pq2_0_q8_K`, so
the scalar generic runs; its tiled PQ2 GEMMs (`tinyBLAS_PQ2_AVX`,
`tinyBLAS_PQ2K_AVX`) and the 4x8 repack GEMM are AVX2/VNNI only. Prefill
therefore goes through the BLAS backend: dequantize the whole weight
matrix to F32 in parallel, then one `cblas_sgemm` (batch >= 32). The
rotation is a matmul tagged `GGML_HINT_SRC0_IS_HADAMARD` that each backend
intercepts with an FWHT; BLAS declines those nodes.

That structure measured *worse* here, twice (see kernels/pq2_0.c): per
-thread tiles let one thread's dequant overlap another's AMX work, a panel
serializes the phases. So the fork's lead comes from somewhere other than
the matmul strategy. `sample` on a prefill run points at threading: ~35 k
samples parked in worker threads and ~11.6 k in OpenMP waits against
~30 k in BLAS itself — we call Accelerate from eight OpenMP threads and it
brings its own pool, while llama.cpp calls it once from one thread.

### Next levers

- Metal decode: the PQ2_0 GEMV is now 38 of the 49 ms and within ~10 % of a
  read-only probe of its access pattern, so the next real step is a
  different decode shape (batching rows, or fewer bytes per weight), not
  another GEMV variant.
- CPU prefill: stop oversubscribing Accelerate (the eight per-thread sgemm
  calls each spin up its pool); the 27 % against the fork is thread
  scheduling, not kernels. Then the DeltaNet share, ~25 % of prefill.
- CPU: the PQ2_0 activation prep and x8 repack are now NEON + threaded
  (decode 6.5 -> 7.4 t/s, load 2.3 s faster on the 27B). Next: a quiet A/B
  of `m_max` 128 as the Mac default; an fp16-AMX (BNNS)
  spike for the prefill GEMM.
- `PTQ1_0` (1.75 bpw): slower to unpack than `PQ2_0` on Apple silicon per
  PrismML; not planned.

## Measurement protocol

Use `benchmark/compare_ternary_pi5.sh` — runs geist (SDOT + TL1), llama.cpp, and
bitnet.cpp on the **same** GGUF / threads, from a **cool** baseline, mean-of-N
after a discarded warm-up, raw outputs saved. See [PI5.md](PI5.md)
for the thermal/quiesce discipline (a stray process halves 4-thread numbers; a
hot board throttles whichever engine runs second).

```sh
MODEL=~/models/bitnet-2b4t-TQ2_0-v2.gguf \
LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench \
BITNET_BENCH=~/BitNet/build/bin/llama-bench \
./benchmark/compare_ternary_pi5.sh
```

Decode is often fastest at **3 threads** (memory-bandwidth-bound), prefill at 4
(compute-bound) — geist auto-selects; sweep `THREADS=3` vs `4` for the references.

---

The phase-by-phase optimization history — measured dead ends, the speculative
lm_head trick, the Cougar/bitnet.cpp head-to-head and the comparison against the
2026 ternary-kernel literature — is a lab log, not reference material. It lives
outside this repo (see the research write-ups).


## Ternary-Bonsai-2-27B on Vulkan (2026-09-26)

Same model as above (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, 7.21 GB, sha256
`3907dc16…`), `vulkan` backend: PQ2_0 matvec and tensor-core GEMM kernels
(struct-of-arrays copy in VRAM, float activations), the PQ2_0 embedding lookup
and `fused->hadamard_rotate` on the device. Everything a token needs stays on
the GPU. RSS is the resident GGUF mapping; the weights are uploaded from it once
and are not duplicated in host memory (`caps.weights_device_copy`).

### Correctness

- Fork goldens (`test_bonsai_e2e_int`, PrismML-Eng/llama.cpp `01ae597`): prompt
  ids, next-token top 5 and the first 16 greedy tokens equal on the RTX 2080 Ti
  and on the RADV iGPU.
- Logits vs `cpu_scalar` (`GEIST_KV_INT8=0 GEIST_KV_F16=0`): the 19-token chat
  prompt (batch not a multiple of 16, register-tiled GEMM) is bit-identical, cos
  1.0000000, max |Δ| 0. A 64-token prompt (tensor-core GEMM active) gives cos
  1.0000000, rms logit error **0.00125**, max |Δ| 0.0069 (logits ≈ 20), identical
  top 5. The tensor-core GEMM accumulates in f16 and folds into f32 every 64 k
  (the fork accumulates in f16 over the whole K); `GEIST_VK_PQ2_F32_ACC=1`
  selects the f32-accumulate variant (rms error 0.00072, bit-equal to
  `cpu_scalar` on the parity test's exact data) at about a quarter
  lower GEMM throughput on this card. Activations are rounded to f16 in either case.

### Throughput (RTX 2080 Ti, `bench_perf_sweep --decode-n 64`, default KV = F16)

| configuration | pp128 | pp512 | tg64 |
| :-- | --: | --: | --: |
| geist Vulkan, default chunk (`GEIST_M_MAX` 64) | 412 | 395 | 36.5 |
| geist Vulkan, `GEIST_M_MAX=128` | **545** | **517** | 36.3 |
| fork, Vulkan (`KHR_coopmat`, `int dot: 0`) | 524 ± 2 | 556 ± 0.2 | 29.7 |
| fork, CUDA (`sm_75`) | 577 ± 23 | 781 ± 1 | **44.4** |
| *M1 Max, metal (table above)* | *84–111* | | *15.5–19.5* |
| *PrismML fork, metal* | *110.7* | | *16.65* |

Fork: PrismML-Eng/llama.cpp `adfffbe` (`prism` branch), `llama-bench -ngl 99
-p 128,512 -n 64 -r 2`, one build with CUDA and one with Vulkan, nothing else
running on the machine. With `GEIST_M_MAX=128` geist is 1.04× / 0.93× the fork's
Vulkan prefill at pp128 / pp512 and 1.22× its decode; against CUDA 0.94× /
0.66× / 0.82×. The default chunk stays 64 because a 128-row scratch pool does
not fit a 256 MB BAR heap on other models (below, #488), so Bonsai on a card
without resizable BAR sets `GEIST_M_MAX=128` (it is not capped for qwen35
hybrids: `caps.dn_subchunk`).

RADV iGPU (Ryzen 9 9950X, 2 CUs, 21 GiB heap): pp64 2.0, tg 1.0 t/s — the fork
gets 7.2 (pp32) and 1.29 on the same device. See *RADV* below.

### How the gap was closed (pp512 / tg on the RTX 2080 Ti)

| step | pp512 | tg |
| :-- | --: | --: |
| register-tiled FP32 GEMM, one-lane-per-block matvec | 38 | 22 |
| tensor-core GEMM (`coopmat`, ternary codes → f16 in shared memory) | 198 | 22 |
| packed k-contiguous shared tiles, arithmetic dequant | 247 | 22 |
| DeltaNet state column in registers (`d_k == 128`) | 316 | 22 |
| 128 × 64 tile, next k-step's loads in flight over the MMAs | 388 | 22 |
| matvec: one lane per row, activations broadcast from shared memory | 381 | **35.5** |
| one scratch pool per model, `GEIST_M_MAX=128` (128 × 64 tile) | 447 | 35.5 |
| 128 × 128 tile (f32 accumulation) | 470 | 35.5 |
| f16 accumulation folded into f32 every 64 k | **517** | 36.3 |

### Where the fork was ahead (side by side, pp512 = 388 t/s vs its 543)

Per-op GPU time of one 512-token pass, fork from `GGML_VK_PERF_LOGGER=1`, geist
from `GEIST_VK_PROFILE=1` (both timestamp queries), at the 388 t/s step:

| | geist | fork |
| :-- | --: | --: |
| PQ2_0 GEMM | 1180 ms | 727 ms |
| DeltaNet recurrence | 98 | 36 (+ 10 `L2_NORM`) |
| attention | 64 | 11 (flash attention) |
| Hadamard | 36 | 20 (three f32 `MUL_MAT`s on the tensor cores) |
| conv, alpha/beta matmul | 23 + 52 | 8 + 37 |
| total | 1508 | 942 |

- **The GEMM was 80 % of the gap.** Same 17408 × 5120 shape: the fork 41.5 TFLOP/s
  at n = 512, ours 22.6 at m = 64 — with a chunk of 64 tokens there are only 136
  workgroups for 68 SMs. Ours at m = 512 measured 32.5, so the rest was the tile
  configuration (fork: 128 × 128 with 64 × 64 per subgroup, `mul_mm.comp`
  `l_warptile_mmq`) and, decisively, **f16 accumulation**: the fork's default
  pipeline is `f16acc` (`coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT`),
  which runs at full rate on GeForce Turing while f32 accumulation runs at half
  rate (≈ 54 vs ≈ 107 TFLOP/s peak). Folding the f16 accumulators into f32 every
  64 k took our GEMM from 30 to 42 TFLOP/s at m = 128 (fork 41.5).
- **Arrays of cooperative matrices indexed in a loop run 4× slower** (not kept in
  registers; 4.2 vs 24.6 TFLOP/s): the accumulators and fragments are unrolled by
  hand in `matmul_pq2_0_cm_body.glsl`. A 64 × 64 quadrant per subgroup with only
  four subgroups was slower than 32 × 64 with eight (too few warps).
- **The chunk size was blocked by the scratch pool**, not the GEMM: the model
  allocated a second, unused default session next to the real one, and two
  180 MiB pools no longer fit the 256 MB BAR heap (the second landed in host
  memory: `silu_mul` 30 → 1461 µs per call). One pool per model fixed that
  (141 → 446 t/s at `GEIST_M_MAX=128`); a device-local pool would remove the BAR
  limit altogether (#488).
- **Decode was activation traffic, not arithmetic.** An isolated experiment
  gave a 505 GB/s pure weight-read roofline for the lane-per-block-half mapping
  but ~155 GB/s once the activations were loaded, and removing the dequant
  arithmetic entirely bought only 7 %. Every lane of a warp needs the same 128
  activations, so a lane per row with the activations staged once per
  workgroup in shared memory (read as broadcasts) reaches 262 GB/s on the
  17408 × 5120 matrix (fork 232–256 GB/s on the same shapes); dequantization
  avoids int → float converts (code bits become the top mantissa bits of 1.0,
  `Σ(c−1)x = 4Σfx − 5Σx`).

### Tensor-core attention (2026-09-27, default since #501's rollout)

`attention_f16_cm.comp`: causal MQA/GQA on `coopmat` instead of one scalar
dot product per thread — one subgroup per 16-query-row block, BR = BC = 16
(the hardware fragment size, so QK^T and P@V are each a single
`coopMatMulAdd` per head_dim/16 tile, no array-of-`coopmat`-in-a-loop).
head_dim == 256 (qwen35/Bonsai's full-attention shape) and
`sliding_window == 0` only; every other shape still runs the scalar
`attention_f16` kernel. Two passes over the causal KV range instead of one
(row max, then exp+accumulate) rather than a streaming rescale: `coopmat`
under `GL_KHR_cooperative_matrix` has no portable per-row scalar multiply on
an opaque accumulator, which streaming softmax needs every tile to rescale
`O`. The fork's flash attention reaches into `GL_NV_cooperative_matrix2`
(`coopMatReduceNV`) for that, an NVIDIA-only extension; this kernel stays on
the portable KHR one so it still *builds* on RADV (RADV reports no
`VK_KHR_cooperative_matrix` at all on this host, so it keeps running the
scalar kernel either way — no regression there, and #471 already tracks its
other coopmat-shader limits).

| | attention only, pp512 | attention only, pp1024 | pp512 total | pp1024 total |
| :-- | --: | --: | --: | --: |
| scalar (`attention_f16`) | 64.8 ms | 245.7 ms | 522 t/s | 492 t/s |
| tensor-core (`attention_f16_cm`) | 28.4 ms | 101.6 ms | 540 t/s | 527 t/s |

~2.3–2.4× faster on the attention op itself at both depths (consistent with
a fixed constant-factor win, not a change in the O(depth²) complexity — the
kernel still computes causal attention exactly, just on tensor cores); +3.5 %
/ +7.1 % end to end, growing with depth because attention's share of the
total grows with it. Verified: `test_backend_vulkan_linear_parity`,
`test_backend_vulkan_ops_unit`, `test_bonsai_e2e_int` (bit-identical greedy
tokens) and `test_qwen35_vulkan_e2e_int`, all green with the flag on and off
on both the RTX 2080 Ti and RADV.

Promoted to the default (`GEIST_VK_ATTN_CM=0` is the escape hatch back to the
scalar kernel) after validating on a second, differently-sized model
(qwen3.5-4B, also head_dim = 256) on both GPUs, plus several misaligned
chunk sizes (`GEIST_M_MAX` 100/65/33/17 — non-multiples of BR/BC = 16, so the
causal-masking boundary logic runs on tiles that straddle it, not just
aligned ones) — all bit-identical against `cpu_scalar`. A single-pass
streaming version — once the per-row rescale problem above has a portable
answer — would close roughly another third of the remaining gap for free
(one QK matmul pass instead of two).

### DeltaNet recurrence: subgroup reductions (2026-09-27, #475 item 4)

`deltanet_delta_f32.comp`'s two per-token reductions (`reduce2`, the l2norm
in `load_qk` and the RMSNorm sum in `epilogue`) were a naive 128-wide
shared-memory tree — 7 barriers per call, 14 per token, on this kernel's
strictly serial per-token critical path (Gated-DeltaNet's state recurrence
can't be parallelized across time within one dispatch). Replaced with
`rmsnorm_f32.comp`'s pattern (#475 item 4): `subgroupAdd` per subgroup, one
tiny shared pass over the (<= 4) subgroup partials — 2 barriers per call,
parametrized by `gl_SubgroupSize` rather than hardcoded to 32, so it stays
correct (not just fast) on RADV's 64-lane subgroups.

pp512 on the RTX 2080 Ti: `dn_delta` 98.8 → 84.3 ms (−15 %); end to end
528 → 541 t/s (pp128 546 → 555, pp1024 515 → 520). Smaller than the
attention win because the barriers were not the dominant cost here — the
256 serial FMA read-modify-writes per token (two passes over the d_k = 128
state column) are. RADV's `dn_delta` is unaffected either way (2355 vs
2381 ms at pp512): its cost there is dominated by something else on that
iGPU, not this reduction. Verified: `test_backend_vulkan_deltanet_unit`,
`test_backend_vulkan_linear_parity`, `test_bonsai_e2e_int` and
`test_qwen35_vulkan_e2e_int` on both GPUs.

A real close of the 98 (now 84) vs 46 ms gap needs the chunked
(GEMM-based) delta rule already used by the CPU host-oracle fallback
(`dn_run_prefill_chunked` in `layer_deltanet.c`) ported to this shader —
replacing the O(seq) sequential state-column walk with batched matmuls, the
same shape of rewrite the tensor-core GEMM and attention kernels went
through. Comparable scope to those; not attempted here.

### Still behind

- The remainder of the DeltaNet recurrence gap (84 vs 46 at the fork — see
  above) and the elementwise ops (#475).
- Attention itself, even tensor-core: 2 QK passes instead of 1 (above), and
  no flash-attention-style KV tiling beyond what causal masking already
  skips.
- Against CUDA: 0.66× at pp512 and 0.82× at decode; the GEMM at m = 512 has
  not been re-measured with the f16 accumulators.
- The default chunk of 64 leaves ~25 % of pp128 / pp512 on the table on hardware
  without resizable BAR (#488).

### RADV (Ryzen 9 9950X iGPU: 2 CUs, DDR5)

Correct but slow: pp 2.0, tg 1.0 t/s (fork 7.2 / 1.29). The matvec measures
15 GB/s; a contiguous pure-read kernel measures 67 GB/s (the device's roofline),
and the same access pattern without any dequantization 31 GB/s — the decode is
about half ALU-bound (2 CUs execute ~3 instructions per weight at the rate of
the whole weight stream) and half access pattern. Prefill is slow because the
tiled GEMMs assume 32-lane subgroups and RADV's is 64, so every batch row runs
as a separate matvec (#471); the fork asks for subgroup size 32 through
`VK_EXT_subgroup_size_control`. An int8 (`dp4a`) matvec would cut the ALU work
about 4× (#467).

### Reproduce

```sh
# geist
make BACKENDS="vulkan cpu_x86 cpu_scalar" bin
GEIST_M_MAX=128 GEIST_VK_DEVICE=0 GEIST_BENCH_BACKEND=vulkan \
  bin/linux/release/tests/bench_perf_sweep --gguf Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --seq-lens 128,512 --decode-n 64 --warmup 16 --repeats 2
GEIST_VK_PROFILE=1 …            # per-pipe GPU time
# fork (needs the SPIRV-Headers on the include path for the Vulkan build)
GGML_VK_VISIBLE_DEVICES=0 GGML_VK_PERF_LOGGER=1 llama-bench -m …gguf -ngl 99 -p 512 -n 0
```

### Paired head-to-head, repo protocol (2026-09-26, after #488/#496)

`bench_cross_engine.py` against `cross_engine_gpu_protocol.json`
(`bonsai2-27b-pq2`) rather than a hand-run `llama-bench`, on the same
host as above. No `GEIST_M_MAX` set: the arch now halves the default
chunk on its own until the scratch pool fits the BAR heap (#488); this
run got the full 128-row pool without the env var.

geist `c374358`; llama.cpp (PrismML fork) `adfffbe4`.

| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |
| --: | --: | --: | --: | --: | --: | --: |
| 128 | 538.95 ± 1.30 | 515.89 ± 5.16 | +4.47% | 36.09 ± 0.06 | 28.98 ± 0.29 | +24.53% |
| 256 | 527.41 ± 1.06 | 533.59 ± 4.49 | -1.16% | 35.87 ± 0.08 | 29.21 ± 0.16 | +22.81% |
| 512 | 508.71 ± 1.05 | 544.11 ± 3.29 | -6.51% | 35.68 ± 0.11 | 29.35 ± 0.13 | +21.59% |
| 1024 | 477.76 ± 0.91 | 540.62 ± 4.83 | -11.63% | 35.17 ± 0.11 | 29.25 ± 0.07 | +20.23% |

Decode leads the fork's Vulkan build by +20…25% across all four depths
(one lane per row, activations broadcast from shared memory, §above).
Prefill is at rough parity at pp128 and falls behind as depth grows
(+4% at 128, -12% at 1024) — the fork's GEMM keeps f16 accumulation over
the whole K and a 128×128 tile at every depth, while ours folds to f32
every 64 k and the 128-row chunk amortizes over more workgroups at
pp128 than at pp1024. Token streams are engine-native synthetic inputs
(compute-shape parity, not logit parity — see *Correctness* above for
that gate).

- protocol: [`cross_engine_gpu_protocol.json`](../cross_engine_gpu_protocol.json),
  host profile `nvidia_2080ti_vulkan`, model key `bonsai2-27b-pq2`
- raw samples:
  [`raw/2026-09-26T220020Z_geist_llama_gpu_bonsai2-27b-pq2_nvidia_2080ti_vulkan.jsonl`](raw/2026-09-26T220020Z_geist_llama_gpu_bonsai2-27b-pq2_nvidia_2080ti_vulkan.jsonl)
- reproduce:
  `python3 tools/bench_cross_engine.py --geist bin/linux/release/tests/bench_perf_sweep
  --llama llama-bench --gguf Ternary-Bonsai-2-27B-PQ2_0.gguf
  --protocol benchmark/cross_engine_gpu_protocol.json
  --host-profile nvidia_2080ti_vulkan --model bonsai2-27b-pq2`

## Ternary-Bonsai-2-27B on x86-64 (2026-09-30, synthetic weights)

Measured on `tools/gen_synth_gguf.py --preset bonsai2-27b-pq2_0`: the real
file's geometry, formats and `prism.hadamard` keys (26.9 G parameters,
7.20 GB) with random ternary weights. Kernel timings do not depend on the
values. The host is an Intel Xeon (Sapphire Rapids) at 2.1 GHz, 4 vCPUs of a
cloud VM with 260 MB of L3, gcc 14, `OMP_WAIT_POLICY=active`, with 41.7 GB/s
read bandwidth on 4 threads.

| | generic path | `PQ2_0` decode GEMV | and prefill GEMM | GEMM on AMX-INT8 | SwiGLU, DeltaNet floor and conv |
| :-- | --: | --: | --: | --: | --: |
| prefill, 64 tokens | 39.1 s, 1.64 t/s | 38.3 s, 1.67 t/s | 7.31 s, 8.75 t/s | 1.58 s, 40.5 t/s | **1.10 s, 58.3 t/s** |
| decode | 7.53 s a token, 0.13 t/s | **0.200 s a token, 5.0 t/s** | the same | the same | the same |
| RSS | 7.1 GB (the mmap'd file, nothing repacked) | 7.1 GB | 7.1 GB | 7.1 GB | 7.1 GB |

### The generic path

cpu_x86 had no `PQ2_0` kernel at first. The format ran through
`linear_generic.c`, which decodes each weight row to fp32 with
`dequant_pq2_0_row`, one element at a time, and dots it with AVX2 FMAs.

The forward profiler (`GEIST_PROFILE_FORWARD=1`) splits the time into:

- the FFN: 74 % of prefill, 70 % of decode;
- the mixers: 26 % and 30 %. They are nearly all projections. The 16
  attention cores took 6 ms of the 39 s prefill, and the DeltaNet recurrence
  and the Hadamard rotation do not show;
- the lm_head: 0.35 s a token.

`perf` (cpu-clock samples) attributes the time as follows:

- decode: 88 % in `dequant_pq2_0_row`, 6 % in the FMA dot;
- prefill: 75 % in the FMA dot (64 activation rows against each decoded
  weight row), 18 % in the decoder.

Decode read 7.2 GB a token in 7.5 s, about 1 GB/s, or 2 % of the bandwidth.
A repacked copy like cpu_neon's x8 layout would not fit next to the model in
this host's 15 GB.

### The decode GEMV

`src/backends/cpu_x86/linear_pq2_0.c` is cpu_neon's W2A8 recipe in AVX2.
The activation is quantized to int8 once per call, with cpu_neon's scale and
rounding, and stored in the order the codes are packed. The raw 2-bit codes
go into `maddubs`, and the +1 bias leaves through each block's activation
sum. The weights are the GGUF bytes, read as they are.

One call on a 17408 × 5120 FFN matrix, 4 threads, 64 distinct copies of it
(1.5 GB, past the L3); the median of 3 alternating rounds:

| | time | weights read |
| :-- | --: | --: |
| generic path | 25.5 ms | 0.93 GB/s |
| W2A8 GEMV, hardware prefetch only | 1.29 ms | 18.4 GB/s |
| **W2A8 GEMV** | **0.556 ms** | **42.6 GB/s** |

With the data in L2, the dot runs at about 12 GB/s per core, so four cores
could take 48 GB/s. From DRAM, the hardware prefetchers set the limit. A
software prefetch 4 KB ahead of the dot brings the kernel to the host's read
bandwidth; 4-8 KB measured best, 2 KB was 15-20 % slower, 32 KB about 10 %.

An AVX-512 VNNI block dot (`vpdpbusd`) is 20 % faster from L2. From DRAM it
gained 2.6 % (median of 8 alternating rounds, 6 of them faster), so it is
not in.

End to end, both builds from scratch (`tools/bench_revision_ab.py`,
6 cycles alternating with a control copy of the baseline; median of the
per-cycle ratios, 95 % interval):

| | generic path | `PQ2_0` decode GEMV | change |
| :-- | --: | --: | --: |
| decode, per token | 7.70 s | 0.200 s | -97.4 % [-97.6, -97.3], 6/6 |
| one-token prefill | 7.63 s | 0.194 s | -97.4 % [-97.7, -97.2], 6/6 |

A decode token now reads the 7.2 GB of weights at about 36 GB/s, 38 times as
fast as before. The control, the baseline's binary run again, stayed within
±6 %.

### The prefill GEMM

Prefill is the same arithmetic as a GEMM, in the same file. Every token's row
is quantized with its own absmax scale, and the blocks of all tokens are laid
out side by side. A group of 4 weight rows then walks the blocks: each
block's codes are extracted once and dotted against every token, into
per-(row, token) accumulators that stay in L1. The four `maddubs` per block
and token are the floor of the loop, so it is compute-bound and needs no
prefetch.

One call at m = 64, 4 threads, the median of 3 alternating rounds:

| matrix | generic path | W2A8 GEMM | |
| :-- | --: | --: | --: |
| 17408 × 5120 (FFN gate, up) | 107.5 ms | 22.7 ms | 4.7× |
| 5120 × 17408 (FFN down) | 204 ms | 20.3 ms | 10× |

The generic path dequantizes each weight row to fp32 once and then dots it
against all 64 tokens: slower still for the down projection, whose
17408-float row does not stay in L1. The GEMM runs at about 1.8 to 2.0 ns per
block and token on each core. That is the AVX2 floor: four `maddubs`, one
`madd`, one convert and one FMA per block and token, all on the same two
ports. Other block orders (rows 2 to 8 per group, one to four tokens per
step, one convert for 8 blocks after a transpose-reduce) measured within
noise of this one or slower.

End to end, both builds from scratch (`tools/bench_revision_ab.py`, 6 cycles
with a control):

| | decode GEMV only | and prefill GEMM | change |
| :-- | --: | --: | --: |
| prefill, 64 tokens | 37.7 s | 7.31 s | -80.5 % [-80.8, -78.9], 6/6 |

On a host with AMX-INT8 the next section's kernel takes over; this one
remains the path everywhere else.

### The prefill GEMM on AMX-INT8

The AVX2 GEMM runs at its floor, four `maddubs` per block, row and token.
This host also has AMX-INT8. One `TDPBSSD` multiplies a 16 × 64 int8 tile by
a 64 × 16 one into 16 × 16 int32 in 16 cycles, which is 16 rows by 16 tokens
over half a block. `src/backends/cpu_x86/kernel_pq2_0_amx.c` runs the same
W2A8 arithmetic on the tiles:

- **A, the weights.** `code - 1` as s8, 16 weight rows per group, in the
  activations' code order. A block's A is extracted once per group and
  serves every token tile. The -1 takes the codes' bias out of the dot, so
  no activation sums are needed.
- **B, the activations.** The int8 rows the AVX2 path quantizes, repacked per
  block into VNNI tiles: a 16 × 16 dword transpose per half block.
- **C, the dots.** One C tile holds a block's exact int32 dots (two
  `TDPBSSD`, one per half). They are scaled into fp32 accumulators,
  `acc += d * C`, and y is the token's factor times acc.

C leaves the tiles through memory, and a vector load of a tile store's bytes
waits until the store commits, after the two `TDPBSSD` it depends on have
retired. `TILEZERO`, the two dots, the store and a vector load of its bytes
took about 140 cycles as a dependent chain. So the kernel post-processes
each C two steps after its store, through a ring of four buffers. C
alternates between two tiles and B between two pairs, the next block's A is
extracted into the other half of a double buffer while this block's steps
run, and the weight rows are prefetched four blocks ahead.

TSC cycles per step (16 rows × 16 tokens × one block) in a microbenchmark of
the tile loop alone:

| | cycles |
| :-- | --: |
| two `TDPBSSD` | 32 |
| with the B loads, the C store and the post right behind it | 75 |
| the post two steps behind, ring of four C buffers | 46 |
| and a new A every 4 steps | 50 |

In the kernel a step costs about 105 cycles on one core with the weights in
L3. Extracting A is about 20 % of that, and loading the B tiles from L2
12-16 %.

Per-thread scratch goes on pages of its own, with a page between threads.
Packed side by side, each core's L2 prefetchers ran on into the neighbour's
accumulators and C buffers, which the neighbour stores to every step, and
the lines bounced between the cores. At m = 128 on 17408 × 5120, in ns per
block, row and token and core:

| per-thread scratch | 2 threads | 4 threads |
| :-- | --: | --: |
| packed, 64-byte aligned | 0.31-0.48 | 0.38-0.52 |
| page-aligned, no gap | 0.31-0.33 | 0.38-0.42 |
| page-aligned, a page between threads | 0.20-0.21 | 0.20-0.30 |

One thread ran at 0.20, and two single-threaded processes did not slow each
other down, which pointed at the shared address space rather than the
hardware.

A pass covers at most 128 tokens (8 token tiles), and the weights stream once
per pass. At m = 256, two passes measured 12 % (17408 × 5120) and 17 %
(5120 × 17408) faster than one, whose packed activations (4.4 MB at
n_in = 17408) no longer fit in L2.

One call at m = 64, 4 threads, 16 distinct copies of the matrix (380 MB, past
the L3), the median of 3 alternating rounds:

| matrix | AVX2 GEMM | AMX GEMM | |
| :-- | --: | --: | --: |
| 17408 × 5120 (FFN gate, up) | 23.6 ms | 2.62 ms | 9.0× |
| 5120 × 17408 (FFN down) | 21.3 ms | 2.66 ms | 8.0× |

End to end, both builds from scratch (`tools/bench_revision_ab.py`, 6 cycles
with a control, which stayed within +0.2 % [-2.1, +1.0]):

| | AVX2 GEMM | AMX GEMM | change |
| :-- | --: | --: | --: |
| prefill, 64 tokens | 7.21 s | 1.58 s | -78.4 % [-79.1, -76.9], 6/6 |

The forward profiler puts the FFN at 0.76 s of a 1.61 s prefill (gate and
up 0.40 s, down 0.19 s; they were 3.0 s and 1.4 s) and the 64 mixers at
0.83 s, of which the 16 attention layers take 0.10 s. In a `perf` profile
of five prefills (and their decode steps) the AMX GEMM has 41 % of the
samples. Next come the DeltaNet recurrence's fp32 GEMMs (OpenBLAS `sgemm`,
12 %), the Hadamard rotation of the activations (`fwht_orthonormal`, 4 %)
and the rest of the DeltaNet chunk code (about 10 %).

`cpu_x86_linear_pq2_0_amx_usable` decides at bind. It checks the dispatcher
tier (so `GEIST_FORCE_ISA=avx2` keeps the AVX2 GEMM), cpuid for AMX-INT8,
AVX-512F and AVX-512BW, and asks Linux for the tile data
(`arch_prctl(ARCH_REQ_XCOMP_PERM)`, Linux 5.16 and later).

### The SwiGLU activation

With the projections on AMX, the FFN's activation became visible. cpu_x86
took cpu_scalar's `silu`: a libm `expf` and a division per element on one
thread (the branch between its two overflow-safe forms keeps gcc from
vectorizing it), and then a separate `mul` pass. The forward profiler put them
at 56 and 27 ms of the 1.6 s prefill.

`src/backends/cpu_x86/elementwise.c` now does both in one pass on the whole
team, eight lanes at a time:

- The exp is Cephes' `expf` in AVX2 (1.26 ulp at worst). A vectorized libm
  `expf` is libmvec, which only glibc has and which runs every lane below
  -87.3 through a scalar slow path. The argument is floored at -87, where
  2^n is still normal.
- A masked tail runs the last lanes through the same instructions, so every
  element takes the same path.
- The fused `silu_mul` is `silu` then `mul` to the bit, as the fused table's
  contract asks. The exec plan now binds it on cpu_x86, and the separate
  `mul` pass is gone.

One call on the FFN gate (m × 17408), 4 threads, best of 20:

| m | cpu_scalar `silu` + `mul` | cpu_x86 `silu_mul` | |
| :-- | --: | --: | --: |
| 1 (decode) | 14.9 µs | 8.8 µs | 1.7× |
| 64 | 1143 µs | 226 µs | 5.1× |

End to end, both builds from scratch (`tools/bench_revision_ab.py`, 10 cycles
with a control):

| | before | after | change |
| :-- | --: | --: | --: |
| prefill, 64 tokens | 1.566 s | 1.507 s | -3.9 % [-12.3, -1.2], 9/10 |

The control stayed within +1.8 % [-5.6, +10.1], and decode moved -0.8 %
[-6.5, +5.2], inside it. The forward profiler puts the activation at 22 ms
of the prefill, with no `mul` stage left.

### Denormals in the DeltaNet chunk

With SiLU off the list, a `perf` profile with call graphs, split by OpenMP
region, put the prefill's thread time (4 threads, per 64-token prefill) at
3.4 s for the AMX projections, 1.25 s for the DeltaNet chunk's head loop,
0.48 s for its conv and gating and 0.36 s for the Hadamard rotation. The
head loop's seven OpenBLAS SGEMMs per head took 414 µs per head chunk. The
same seven calls, with the same shapes and strides, take 91 µs in isolation
and 92 µs with four copies running at once. The difference was denormals:

- The chunk scales by e^γ, γ the gating summed over the chunk. On the
  synthetic model γ ends below -60 in 1488 of a prefill's 2304 head chunks
  and below -87 in 672. 3-5 % of the scaled K and Q (`KCe`, `Qg`, the
  decayed K) were denormal, and 0.4-0.5 % of the masks, each a microcode
  assist. Nothing in the process flushes them: FTZ/DAZ are off.
- With FTZ/DAZ forced on for the whole process (`LD_PRELOAD`), the head loop
  fell to 437 ms and nothing else moved.

`transformer_dn_head_chunk` now takes a decay factor below e^-60 as 0
(`DN_EXP_FLOOR`, the attention kernels' floor), clamps the argument so that
libmvec's `expf` never takes its slow path, and skips the forward
substitution's entries below the floor, which carry e^(γ_i - γ_l). No
denormal is left, and the chunked logits are bit-identical to before.

| | before | after | change |
| :-- | --: | --: | --: |
| head loop, thread time per prefill (`perf`) | 1249 ms | 374 ms | 3.3× |
| prefill, 64 tokens (A/B) | 1.431 s | 1.186 s | -15.9 % [-24.5, -12.4], 10/10 |

The A/B is `tools/bench_revision_ab.py`, both builds from scratch, 10
cycles with a control, which stayed within -7.2 % [-9.4, +5.3]; decode
moved -1.5 % [-10.1, +5.7], inside it. The synthetic gating is steep: with
`ssm_a` = -1 and random `ssm_alpha` rows, every head's γ ends a chunk below
-10. A trained model's heads range from slow to fast forgetting, so its gain
is likely smaller. That is not measured here.

### The DeltaNet conv

After the floor, the `perf` split put the chunk's first parallel region at
427 ms of thread time per prefill. That region runs the causal conv over
the old state and the chunk's rows, silu, the q/k norms and the gating. Its
channel loop chose the source of every tap per element and called silu's
scalar `expf` beside it, so none of it vectorized. Alone, at the model's
sizes (64 × 10240, K = 4), it took 1.93 ms per layer on 4 threads; the norms
and the gating were a few percent of that.

`transformer_dn_conv_silu_row` picks a token's four input rows before the
channel loop, names the taps so that a channel's weights load as one group,
and gives silu a pass of its own, which vectorizes where glibc's libmvec
has a vector `expf`:

| | before | after | change |
| :-- | --: | --: | --: |
| the region alone, per layer | 1.93 ms | 0.24 ms | 7.9× |
| the region in the model, thread time per prefill (`perf`) | 427 ms | 50 ms | 8.5× |
| prefill, 64 tokens (A/B) | 1.201 s | 1.099 s | -11.8 % [-16.4, -3.2], 10/10 |
| the same, a second run | 1.232 s | 1.133 s | -8.6 % [-11.9, -0.1], 9/10 |

Each A/B is `tools/bench_revision_ab.py` with both builds from scratch and
10 cycles with a control. The controls stayed within -0.6 % [-10.2, +12.8]
and +2.9 % [-4.8, +19.1], and decode moved -1.6 % and -2.0 %, inside them.

### The Hadamard rotation

After the conv, the rotation of the activations (`prism.hadamard`, in
1024-float blocks before the projections) was the next region: 324 ms of
thread time per prefill. `fwht_orthonormal` ran its passes of len 1, 2 and
4 scalar: their inner loops of 1, 2 and 4 butterflies are too short for the
compiler to vectorize, and they cost more than the other seven passes of a
1024-float block together. Each now runs as one loop over the block, which
vectorizes. The butterflies and their order are the same, so the result
does not move by a bit: the Vulkan and Metal ports run that order, and the
Vulkan test expects the host's bits.

| | before | after | change |
| :-- | --: | --: | --: |
| `fwht_orthonormal`, 1024-float blocks, one thread | 1.98 ns per float | 0.63 ns | 3.1× |
| the rotation in the model, thread time per prefill (`perf`) | 324 ms | 106 ms | 3.1× |
| prefill, 64 tokens (A/B, 40 cycles pooled) | 1.124 s | 1.095 s | -5.2 % [-6.7, -1.8], 31/40 |

Three A/Bs, each with both builds from scratch and a control
(`tools/bench_revision_ab.py`), gave -6.2 % [-14.5, -0.9] in 9 of 10
cycles, -5.7 % [-10.6, +11.0] in 8 of 10 and, over 20 cycles, -2.9 % [-6.7,
+1.3] in 14 of 20. Each run on its own is close to its resolution. Pooled,
the change per cycle has a median of -5.2 % (95 % bootstrap interval -6.7
to -1.8 %) and 31 of the 40 cycles were faster (one-sided sign test
p = 3e-4), against a control, the baseline's binary again, that read +2.9 %
[-0.4, +4.5] over the same cycles. That is the 55 ms the rotation's thread
time predicts on 4 threads. Decode stayed within noise in every run.

### The layer-output scale

The step after each layer multiplied the hidden state by the layer's
output scale, which only Gemma 4 loads; this model, like every other,
gets 1. On cpu_x86 that was a pass over the 64 × 5120 floats on the
calling thread while the others waited, 10.6 ms per prefill
(`GEIST_PROFILE_PREFILL`). The step returns at once for a scale of 1 now,
which changes no bit on the CPU backends, and the profiler puts it at
0.05 ms. That is 1 % of the prefill, below what an A/B resolves on this
host: two runs with both builds from scratch gave -1.8 % [-5.9, +6.2] in
12 of 20 cycles and +0.4 % [-5.9, +4.8] in 14 of 30.

### RMSNorm and the residual add

Each layer normalizes its input and the FFN's input, and adds the mixer's
and the FFN's output to the residual stream. All four steps were
cpu_scalar's, on the calling thread while the other three waited: 47 ms
of a prefill in `perf`, 21 ms in the norms and 26 ms in the adds. cpu_x86
now spreads the norm's rows and the add's 4 KB chunks over the team, in
AVX2, with the sum of squares in double as before. On one thread the new
kernels are within 10 % of cpu_scalar's; the gain is the team. Below
16384 floats, as for a decode token, the calling thread does them alone.

| | before | after | change |
| :-- | --: | --: | --: |
| rmsnorm, one call at 64 × 5120 (48 copies in turn, past the L2s) | 339 µs | 54 µs | 6.3× |
| add, the same | 218 µs | 49 µs | 4.4× |
| the FFN's norm in the model, per prefill (`GEIST_PROFILE_PREFILL`) | 11.1 ms | 3.0 ms | 3.7× |
| the FFN's residual add, the same | 11.7 ms | 4.2 ms | 2.8× |
| the norms and adds on the calling thread, per prefill (`perf`) | 47 ms | 11 ms | |
| prefill, 64 tokens (A/B, 30 cycles) | 1.145 s | 1.067 s | -5.8 % [-10.7, -3.0], 24/30 |

The profiler runs alternated the two builds three times. Besides the
calling thread's 11 ms, `perf` finds 14 ms on each of the other three
threads, at the same time, so the steps' wall time fell by about 33 ms,
which the A/B's interval holds. The A/B built every revision from
scratch and ran a copy of the baseline as a control, which read -0.3 %
[-5.8, +4.3]; decode moved -1.8 % [-3.1, +1.3].

### What a step of the AMX GEMM costs

After the norms the AMX GEMM is 73 % of the prefill's thread time, 3.4 s
of 4.6 s, 2.86 s of it in the tile loop (`pq2_0_amx_gemm`). A step, 16
weight rows by 16 tokens over one block, has two `TDPBSSD` at 16 cycles
each. On one thread, at 10240 × 5120 and 64 tokens with per-block scales
drawn at random, it takes 113 TSC cycles with the weights in L3 and 119
streamed from DRAM. Each of the kernel's other jobs, left out on its own:

| left out | TSC cycles per step, L3 | DRAM |
| :-- | --: | --: |
| nothing | 113 | 119 |
| extracting the next block's A | 94 | 101 |
| the post (`acc += d × C`) | 91 | 98 |
| the post and the C tile store | 77 | 82 |
| B from L2: every block reads block 0's 8 KB | 90 | 97 |
| all four | 40 | 48 |

The savings nearly add up (19 + 22 + 14 + 23 = 78 against 73): the four
run beside the dots, not under them. B costs its 20 cycles by size: with
the activations of 1, 2, 4, 8 and 40 blocks in turn (8 to 320 KB) a step
took 90-91, 91-94, 99-100, 111-112 and 114-116 cycles; L1 holds 48 KB.

Loop orders and knobs that moved the cost around without lowering it, on
one thread in the same runs (cycles per step, L3):

- loading the next step's B into the other tile pair right after the
  dots: 117 against 108;
- two row groups sharing each B load (no B or C double buffer): 113-115
  against 108-110;
- blocks outermost over chunks of 2, 4 or 8 row groups, so that a block's
  B is read from L1 by all but the first: 104-110 against 107-113. At 4
  threads the chunk of 4 gave -2 to -6 % per call in the median, rounds
  from -12 to +3 %;
- B held in the tiles across 1, 2 or 4 row groups (A reloaded per group):
  109-122 against 110-112;
- two blocks' C folded into one pass over acc, the same FMAs in the same
  order: 115-116 against 107-108;
- software prefetch of the next block's B, 16 or 32 lines a step:
  116-126 against 111-112;
- a branch-free extraction for full groups with one `vcvtph2ps` for the
  16 scales: 125-127 against 110-113;
- the post 1 or 3 steps behind, a ring of 8, weights prefetched 2 or 8
  blocks ahead: none below the shipped 2, 4 and 4;
- at 4 threads, groups handed out dynamically in chunks of 2 to 16
  (tiles configured once per thread): medians within ±5 % of the static
  split.

The hardware is not shared between the threads: four single-threaded
processes, one per vCPU, each ran the GEMM as fast as one alone, and one
process with 4 threads ran it 3.6 times as fast as with 1 (1621 against
5852 µs at 10240 × 5120).

What would take the post and store off most steps is a C tile over more
than one block, which needs consecutive blocks of a row to share their
scale. The synthetic model's do, all of them (`gen_synth_gguf.py` writes
one scale); the real model's are not known here, so the kernel does not
assume it.

### The attention's query, gate and scale

Each of the 16 attention layers projects a query and a gate per head in
one matrix. The arch then copied them apart head by head, scaled the
query by 1/sqrt(256) after its norm and RoPE, and multiplied the
attention's output by sigmoid(gate) with libm's `expf` per element: three
loops on the calling thread while the other three waited. cpu_x86 now
binds the backend ops the arch asks for (`attn_qgate_split`, `scale_f32`,
`sigmoid_mul`), on the team from 16384 floats on, the sigmoid in AVX2
through SiLU's exp. Per prefill, the profiler runs alternating the two
builds three times:

| | before | after |
| :-- | --: | --: |
| calling thread's serial time in the attention blocks (`perf`) | 26.8 ms | 7.8 ms |
| `q_prep` (norms, RoPE, the scale) | 6.8 ms | 3.1 ms |
| `post_core` (the gate) | 5.7 ms | 2.7 ms |
| `qkv` (the projections and the split) | 65.5 ms | 60.8 ms |
| prefill, 64 tokens (A/B, 30 cycles) | 1.142 s | 1.107 s, -2.4 % [-5.3, +0.2], 20/30 |

What is left on the calling thread there is the KV store's append and
RoPE. The A/B, both builds from scratch with a control that read -1.7 %
[-4.5, +3.3], does not resolve the 1 % the profile predicts.

### The DeltaNet's BF16 projections

The 48 DeltaNet layers project alpha and beta from the normed input
through BF16 matrices of 48 rows, cpu_x86's generic linear. It dotted
each dequantized row against one token at a time, two loads per FMA; it
takes 4 rows against 3 tokens now, 7 loads per 12 FMAs (X86.md has the
general case):

| | before | after |
| :-- | --: | --: |
| one call, 48 × 5120, 64 tokens, 4 threads | 227 µs | 153 µs |
| the projections in the model, thread time per prefill (`perf`) | 101 ms | 76 ms |
| prefill, 64 tokens (A/B, 30 cycles) | 1.083 s | 1.081 s, +2.1 % [-2.0, +6.1]: noise |

- reproduce:
  `make gguf_artifacts/synth/bonsai2-27b-pq2_0.gguf`, then
  `GEIST_PROFILE_FORWARD=1 OMP_WAIT_POLICY=active bin/linux/release/tests/bench_perf_sweep
  --gguf gguf_artifacts/synth/bonsai2-27b-pq2_0.gguf --seq-lens 64 --decode-n 1 --warmup 0
  --repeats 1` (and `--seq-lens 1 --decode-n 9` for decode); the A/B:
  `tools/bench_revision_ab.py --rev base=<parent> --rev new=<commit>
  --gguf gguf_artifacts/synth/bonsai2-27b-pq2_0.gguf --seq-lens 1 --decode-n 4
  --warmup 1 --cycles 6 --env OMP_WAIT_POLICY=active`
