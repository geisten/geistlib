# Ternary (BitNet b1.58 / TQ2_0) — Pi 5 performance work

**Goal:** geist's ternary decode *and* prefill on a Raspberry Pi 5 (Cortex-A76,
SDOT, **no i8mm**) at or above `MAX(bitnet.cpp, llama.cpp)` on the same model.

**Status: measured on the Pi 5.** geist decodes the canonical 2B-4T `i2_s` at
**17.4 t/s vs bitnet.cpp's 8.2** (~2×) **at short context**, with prefill and
Cougar/bitnet.cpp head-to-heads built and run on the same board (below). The one
open gap is a canonical 2B-4T **TQ2_0** GGUF (only `i2_s` ships upstream).

Quote that headline with its context length or not at all — decode falls ~29 %
from a 32-token prompt to a 512-token one, so the same build reads anywhere
between 17.9 and 12.8 t/s depending only on where you measure it.

---

## Verified so far (2026-06)

1. **geist runs a real BitNet ternary model end-to-end.** Previously the TQ2_0 /
   TL1 path was only *synthetically* unit-tested (`tests/test_tl1_parity.c`).
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
the curve. One `bench_perf_sweep` run, `--seq-lens 32,128,512 --decode-n 64
--warmup 64 --repeats 10`, mean-of-10 from a cool start (47.7 °C, load 0.00,
`geist-home` and `ollama` stopped for the duration):

| prompt tokens | decode t/s | spread | prefill t/s | total t/s |
| --: | --: | --: | --: | --: |
| 32 | **17.96** | ±0.8 % | 49.58 | 22.81 |
| 128 | 16.77 | ±0.4 % | 50.18 | 30.16 |
| 512 | 12.76 | ±1.3 % | 46.26 | 35.82 |

**Decode falls 29 % from a 32-token prompt to a 512-token one; prefill stays
flat.** That is the KV attention growing with context, not weight bandwidth —
the weights read per token are the same at every point (see the budget below).

Two consequences. Any decode number from this model must name its context
length, or it is unfalsifiable: the same binary reads 17.9 or 12.8 t/s. And an
optimization that shrinks *per-token weight traffic* shows its effect most
clearly at **short** context, where attention does not dilute it.

Raw run: `~/bench-geistlib/stride/2026-08-01_pi5_bitnet-2b4t-i2s_seqlen-sweep.log`.

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

---

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

