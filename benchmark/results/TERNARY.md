# Ternary (BitNet b1.58 / TQ2_0) — Pi 5 performance work

**Goal:** geist's ternary decode *and* prefill on a Raspberry Pi 5 (Cortex-A76,
SDOT, **no i8mm**) at or above `MAX(bitnet.cpp, llama.cpp)` on the same model.

**Status: measured on the Pi 5.** geist decodes the canonical 2B-4T `i2_s` at
**17.4 t/s vs bitnet.cpp's 8.2** (~2×), with prefill and Cougar/bitnet.cpp
head-to-heads built and run on the same board (below). The one open gap is a
canonical 2B-4T **TQ2_0** GGUF (only `i2_s` ships upstream).

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

---

The phase-by-phase optimization history — measured dead ends, the speculative
lm_head trick, the Cougar/bitnet.cpp head-to-head and the comparison against the
2026 ternary-kernel literature — is a lab log, not reference material. It lives
outside this repo (see the research write-ups).

