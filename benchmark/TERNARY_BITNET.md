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
after a discarded warm-up, raw outputs saved. See `benchmark/BENCHMARK_PI5.md`
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

## MEASURED on Pi 5 (2026-06, quiesced & cool, large 728M TQ2_0)

Host `rspdevelop` (`ssh germar@192.168.0.62`), Cortex-A76 ×4, cool start, idle.
geist `make TARGET=pi5`; llama.cpp `acd79d6` OpenBLAS. Same model file.

| metric | geist | llama.cpp | result |
| :-- | --: | --: | :-- |
| prefill pp128 | **~150** | 27.7 | **geist 5.4× faster** |
| decode (3t, low ctx) | 47.0 | 49.6 | llama ~1.05× (≈parity; partly methodology — geist decodes over a longer KV ctx in this harness) |

Decode thread scaling (geist): 1t 23.4 → 2t 42.2 → 3t **46.8** → 4t 43.5 →
saturates at 2–3 threads; ~9 of ~13 GB/s → latency/BW-bound, not compute.

**Verdict vs llama.cpp: geist already meets the goal in aggregate** — dominant
prefill, ≈parity decode.

### Decode kernel is already at parity with llama (source-verified)

llama's ARM `ggml_vec_dot_tq2_0_q8_K` (`ggml-cpu/arch/arm/quants.c`) and geist's
`tq2_0_block_dot_q8a_neon_unbiased` are the **same algorithm**: `vshrq`/`vandq`
to unpack raw {0,1,2} trits, 8× `vdotq_s32` into two accumulators, deferred −1
bias correction. So there is **no free win in the inner loop** — the ~5% is
surrounding overhead, and two cheap levers were ruled out by measurement:

- **software prefetch** in the decode GEMV → no-op (46.8 → 46.7); the A76
  prefetcher already handles the 66-byte stride.
- **`GEIST_PP` spin-pool dispatch** → catastrophic (3.5 t/s; it oversubscribes
  against the active OMP pool).

Consistent with the A76's memory-bound decode: micro-opts don't pan out.
Remaining decode levers are structural (per-GEMV OMP region overhead; fusing
the qkv / gate_up activation-quant), not inner-loop — pursue only if bitnet.cpp
proves a gap worth the risk.

### bitnet.cpp (the other half of MAX)

bitnet.cpp uses its own i2_s/tl1 quant (not TQ2_0), so a fair 3-way needs the
same model in its format. Built on the Pi via `setup_env.py -hr
1bitLLM/bitnet_b1_58-large -q tl1 --use-pretuned` (pretuned TL1 kernel, no
codegen). Two environment snags fixed en route: bitnet.cpp pins `torch~=2.2.1`
(no Py3.13/aarch64 wheel — install unpinned torch), and pip's default temp
overflowed (point `TMPDIR` at the 91 GB `/`). Measured result: its prefill loses
to geist's ~5×, and its **decode** lands at **8.2 t/s** — ~2× behind geist's 17.4
(head-to-head below).

---

## The one open gap

The Pi measurements, the `i2_s` head-to-head, and the reference builds
(`llama.cpp` + `bitnet.cpp` on the board) are **done** — reproduce them with
[`compare_ternary_pi5.sh`](compare_ternary_pi5.sh). What remains is a canonical
**2B-4T TQ2_0 GGUF**: the official `microsoft/bitnet-b1.58-2B-4T-gguf` ships
`i2_s`, not TQ2_0, so a same-format TQ2_0 comparison still needs a convert/quantize
(or the existing `bitnet-2b4t-TQ2_0-v2.gguf` referenced in `mk/target-pi5.mk`).

The measurement loop for any further A76 kernel work: baseline → identify the
behind-metric → targeted kernel change → re-measure cool → keep only if it wins,
record in `BENCHMARK_PI5.md`.

---

## MEASURED on Pi 5 (2026-06): canonical 2B-4T **i2_s** + speculative head

Host `rspdevelop` (`ssh germar@192.168.0.62`), Cortex-A76 ×4, `performance`
governor @2.4 GHz, cool (45 °C), idle. Model: `microsoft/bitnet-b1.58-2B-4T`
`ggml-model-i2_s.gguf` (1.19 GB — all layers I2_S, `token_embd` **F16** and
tied as the lm_head). Built `make TARGET=pi5`; `bench_perf_sweep`, mean-of-5
after a discarded warm-up.

**Reference:** [petlukk/Cougar](https://github.com/petlukk/Cougar) reports
**16.1 tok/s decode** on a Pi 5 for this model (its headline ARM number); built
and run on *this* Pi it reaches **12.3** (see the three-engine table below).

### The bottleneck: a bandwidth-bound decode dominated by the F16 lm_head

Decode is **memory-bandwidth bound**, not compute bound: forcing 1.5→2.4 GHz
moved decode only 9.63→9.92 tok/s, and 2 threads ≈ 4 threads. So the lever is
*bytes read per token*. `GEIST_PROFILE_FORWARD` on the baseline:

```
decode ≈ 100 ms/token   ·   lm_head = 49.9 ms (≈50%)   ·   30 layers ≈ 50 ms
```

The lm_head is one F16 GEMV over the tied `token_embd` — 2560×128256×2 B =
**656 MB read every token**, ~56 % of all traffic. Already a single-pass
fused-convert kernel (`cpu_neon_w_f16_m1`), so the only win is reading less.

### Fix: speculative i8-sketch output head (`GEIST_SPEC_HEAD=1`)

`src/archs/transformer/forward/spec_head.c` (same idea as Cougar's ARM sketch,
but with **exact** final logits). Built once from the F16 lm_head: a
stride-4 subsampled **int8 sketch** (`[VOCAB, HIDDEN/4]` ≈ 82 MB) + one f32
scale/row. Per token: (1) SDOT rough-score the whole vocab against the sketch
(82 MB, parallel); (2) top-512 via a bounded min-heap; (3) compute **exact
f16 logits for those 512 rows only** (~2.6 MB). The argmax-deciding logits are
the unmodified f16 dot products — only *which* rows get scored is approximate.
lm_head drops **49.9 → 9.2 ms/token (5.4×)**; RSS +82 MB.

Greedy output is **byte-identical** to the exact dense head (verified, multiple
prompts) — the true argmax is always inside the 512 candidates. **Default on**
for greedy decode on an eligible tied lm_head; `GEIST_SPEC_HEAD=0` forces the
exact head, and non-greedy sampling falls back automatically.

### Result — geist vs Cougar vs bitnet.cpp (same Pi, same i2_s model)

geist `bench_perf_sweep`, spec head on, 2 threads, @2.4 GHz, mean-of-5 (decode
falls at longer context as O(n) attention grows — orthogonal to the lm_head win,
hits every engine):

| seq_len | prefill t/s | **decode t/s** | end-to-end t/s |
| --: | --: | --: | --: |
| 32  | 46.4 | **17.4** | 22.0 |
| 128 | 48.5 | **16.4** | 29.3 |
| 256 | 47.0 | **15.0** | 33.0 |

Baseline (spec off) decode was **9.83** tok/s → spec on **17.4** (+77 %). The
end-to-end column is `(prompt + 64 decoded) / wall` — it rises with context
because the fast prefill amortizes over more tokens.

**Head-to-head — all three engines built and run on THIS Pi, same model, 2.4 GHz,
greedy** (decode tok/s, each at its best thread count):

| engine | t=2 | t=3 | t=4 | prefill |
| :-- | --: | --: | --: | --: |
| **geist** (spec head) | **17.4** | 17.2 | 17.0 | ~47 |
| Cougar (Rust + `ea` SIMD) | 12.3 | 11.6 | 10.8 | 16.3* |
| bitnet.cpp (LUT) | 8.2 | — | 8.7 | 38 / 64 |

\* Cougar prefill from its own profile/README. Cougar measured here with
`~/workspace/Cougar/target/release/cougar --model … --max-tokens 64 -t N`.

**geist leads Cougar by ~42 % and bitnet.cpp by ~2×** on this box. All three peak
at 2 threads (more threads contend). bitnet.cpp's LUT prefill (64 @ 4t) is the
known LUT-prefill / SDOT-decode trade-off; geist's ~47 prefill matches Cougar's.

> Earlier drafts of this doc inferred (from Cougar's **published** 16.1 t/s, which
> it cites on a different board) that Cougar's `ea`-compiled layer matmuls were
> ~30 % faster than geist's at matched clock. **Directly measuring Cougar on this
> Pi disproves that** — it reaches only 12.3 t/s here, and its own profile shows
> its ternary **FFN matmuls** (gate+up 53 ms + down 28.6 ms = 71 % of 114 ms/tok)
> are the bottleneck, i.e. geist's SDOT FFN kernels are faster on the A76. Both
> engines already make the lm_head cheap with a sketch (Cougar's output stage is
> ~9 %). Lesson: measure the competitor, don't extrapolate its headline number.

### Pushing the layer matmuls further — what did NOT work

After the spec head, decode is dominated by the ternary FFN matmuls (gate+up
≈25.5 ms, down ≈12.6 ms of a ~57 ms token) — the same stage that bottlenecks
Cougar. Per call the existing M=1 kernel already runs at **~54 % of A76 SDOT
peak** (2 threads). Two of Cougar's kernel shapes were ported to try to speed it
up anyway; **both regressed** on the A76 with gcc, and both were reverted:

1. **4-row output blocking** (`i2_s_block_dot_4row`, 4 rows share one
   activation-block load, 1 accumulator/row): **14.8 vs 17.5 t/s** (2t). Decode
   isn't activation-load-bound, and one accumulator per row exposes the ~4-cycle
   SDOT latency that the shipping kernel's *two* accumulators already hide.
2. **Fused gate+up** (`linear_pair_m1`: quantize the shared activation once,
   interleave gate/up into 4 SDOT chains g0,g1,u0,u1): output bit-identical but
   **15.2 vs 17.4 t/s**. The 4 chains + 16 unpacked weight vectors blow the
   register file → spills that cost more than the saved activation loads.

Lesson (consistent with `q4k-prefill-a76-vmlaq-bound`): the **1-row, 2-accumulator
SDOT loop is already at the A76/gcc sweet spot**; extra ILP needs more live
registers than the core has, so it spills. Cougar's edge on the layer matmuls at
matched clock appears to come from its `ea` compiler's register allocation /
scheduling, not from an algorithm we can express better in C. The decode win we
**can** bank is the lm_head (the spec head). Further layer gains would need
either hand-written assembly or fewer unpack ops per SDOT.

### Cougar's algorithm (for reference)

[petlukk/Cougar](https://github.com/petlukk/Cougar) is a ~7K-line Rust inference
engine with SIMD kernels written in its own [`ea`](https://github.com/petlukk/eacompute)
DSL (compiled to `.so`, embedded in the binary). We could not build it (the `ea`
compiler is needed and not distributed), so this is from source reading of the
`kernels/*.ea` and `src/*.rs`:

- **Ternary dot (`bitnet_i2s_arm.ea`).** Identical core to ours: 2-bit trits
  `{0,1,2}` packed 4/byte, ARM `vdot_i32` (SDOT) signed×signed, weights left as
  raw `{0,1,2}` with the −1 bias folded out once via the activation sum. Two
  accumulators per row for dual-issue.
- **Register blocking.** `i2_dot_i8_4row` does 4 output rows per shared
  activation load; `i2_dot_i8_4row_dual` fuses 4 gate + 4 up rows, interleaving
  the chains for ILP. (These are the shapes we ported — they win for Cougar under
  `ea` but lose under gcc on the A76, see above.)
- **FFN.** BitNet-2B-4T uses **squared-ReLU** (`relu(gate)²·up`) and **SubLN**
  sub-norms before O- and down-projection — geist matches both.
- **Speculative output head.** The key decode trick and the one we adopted: a
  stride-4 int8 **sketch** of the embedding (`[vocab, hidden/4]`, ~82 MB) rough-
  scores the whole 128 K vocab, takes the top-512, and finishes those exactly.
  Cougar also stores the embedding as int8 for the exact pass; geist instead
  reads the **f16** rows for the 512 finalists, so its top-k logits carry **no
  embedding-quantization error** — only the candidate *selection* is approximate.
- **Threading.** A custom spin `ThreadPool` with `run`/`run_split3` (split Q/K/V
  or gate/up/down across threads), vs geist's OpenMP.

Published Cougar numbers (its README): Pi 5 BitNet **decode 16.1 t/s**, prefill
16.3, ~62 ms/tok, ~2.0 GB RSS, at ~1.6 GHz (stock-cooler throttling); x86
BitNet decode 18.0 t/s (+22 % vs bitnet.cpp), prefill 47.5. (Directly measured on
this Pi at 2.4 GHz it does 12.3 t/s decode — the published 16.1 did not
reproduce on this board.)

---

## The spec head on Gemma 4 (Q6_K) — does it transfer?

The sketch idea is the one Cougar technique worth porting (the kernel shapes
regressed, above). It is **not** BitNet-specific, so `spec_head.c` was
generalized from F16-only to any block-quantized tied lm_head (Q3_K..Q8_K) by
dequantizing rows. Gemma 4 E2B is the obvious target — a **256 K** vocab (2×
BitNet) tied to a **Q6_K** `token_embd` [1536, 262144].

Measured fraction (`GEIST_PROFILE_FORWARD`, Gemma 4 E2B Q4_K_M, Pi 5):
lm_head = **45 ms/token ≈ 32 % of decode** — substantial, and *dequant*-bound
(Q6_K must be unpacked per token, not just streamed), not pure bandwidth.

**Phase 3 is bit-exact.** The dense Gemma decode kernel is **W6A8**
(`cpu_neon_w_q6k_m1` → `linear_q6k_decode_w6a8`, int8 activation). Rather than an
f32-dequant approximation, phase 3 builds a one-row view of the embedding
(`n_out=1`, `raw = row`) and calls the **same `linear_m1`** — the identical W6A8
kernel — so each finalist logit is bit-for-bit what the dense head computes. No
OMP fork (that kernel runs serially for `n_out=1`), no layering violation. The
plain-Q6_K source layout is the default; the opt-in X8-repack (`GEIST_Q6K_X8_GEMV`)
falls back to an f32-dequant dot.

So, as with F16, the *only* approximation is **which** rows are finalists — and
that is the catch on a **256 K** vocab. With TOP_K = 512 (fine for BitNet's 128 K)
the true argmax sometimes misses the candidate set and greedy diverges; **TOP_K =
4096 makes it byte-identical** to the dense head (verified, 48-token generation;
swept: 512 diverges, ≥4096 exact at any stride). The default is therefore
vocab-aware — 512 for ≤200 K, 4096 above — overridable via `GEIST_SPEC_TOPK`.

Decode (`bench_perf_sweep`, 4 t, mean-3, the bigger TOP_K costs phase-3 time):

| TOP_K | decode t/s | vs off | greedy output |
| --: | --: | --: | :-- |
| off | 6.94 | — | — |
| 512  | 7.89 | +14 % | diverges (approx) |
| 2048 | 7.61 | +10 % | exact |
| **4096** | **7.29** | **+5 %** | **bit-identical** |

So the bit-exact Gemma win is **+5 %** decode (the recall-driven 8× larger
phase-3 eats ~9 pts vs the approximate +14 %). Modest but real and reproducible.
For reference, **llama.cpp** (OpenBLAS) decodes this same Gemma model at **7.23
t/s** (2 t) on this box — so the exact spec head (7.29) nudges geist from ≈parity
to a slight lead; the approximate mode (7.89) leads more clearly.
(The per-finalist `linear_m1` re-quantizes the activation each call — pre-quantizing
once via a backend `_pre` entry would recover some of that; left as future work.)

**Takeaway:** the spec output head is the transferable Cougar idea and helps any
big-vocab model with a heavy lm_head — bit-exact for greedy when TOP_K covers the
argmax. The win scales with the lm_head's decode share *and* with how cheaply it
can be re-ranked: BitNet F16 (~50 % of decode, 656 MB) → **+77 %**; Gemma Q6_K
(~32 %, already 2-bit-ish and dequant-bound, 256 K vocab needing a bigger TOP_K)
→ **+5 % exact** (+14 % if approximate). The kernel-blocking ideas do not transfer
to the A76.

---

## vs. the 2026 ternary-kernel literature (T-SAR, FairyFuse)

Two recent papers push CPU ternary inference further than geist's kernels. Read
honestly, they are **a different *kind* of contribution on a different target** —
neither contests geist's headline numbers, but both are more advanced *as
kernels*.

### The measured picture

| | **geist** | **T-SAR** (arXiv 2511.13676) | **FairyFuse** (arXiv 2604.20913) |
| :-- | :-- | :-- | :-- |
| Kind | integration engine, stock kernels | **full-stack co-design (HW component)** | pure-software kernel |
| Target | **A76/Pi5** (primary), x86, Metal | Ryzen 9950X/7840U, Intel N250 | **x86 AVX-512 only** (Xeon 8558P, 48c) |
| ARM? | **yes** | no | no |
| BitNet tested? | yes (2B-4T) | yes (family) | **no** (own Fairy2i 2-bit, Llama-2-7B) |
| Core idea | `vdotq_s32` SDOT, "unbiased" −1 defer, mt4 tile reuse | in-register LUT (no DRAM TL fetch) → memory→compute-bound | mult-free masked AVX-512 add/sub |
| Headline | **2.1×** decode vs bitnet.cpp (A76); 1.3–1.4× vs bitnet.cpp (9950X) | prefill 8.4–12.4×, decode 4.1–6.4× vs CPU baseline | 1.24× vs Q4_K_M decode |
| HW overhead | 0 | **1.4 % area, 3.2 % power** (synthesized SIMD-ALU mod) | 0 |

geist's own measured ternary numbers, for the ratios above: BitNet 2B-4T `i2_s`
decode **17.4 t/s** vs bitnet.cpp 8.2 on the Pi5; on a 9950X, prefill pp128
**884** vs 679 and decode tg128 **77.9** vs 56.5. (After the #102 x86 round —
THP, fused-pair default, NTA weight prefetch, prefill tile 4 — and the #104
t5 base-3 decode packing (1.6 bpw, pow3 unpack — the TQ1_0 idea, x86-only),
the 9950X numbers are **1098 / 103.1**; see `BENCHMARK_X86.md` for the
per-change ledger and the measured dead ends: top-k-sampling spec head,
ReLU² block-skip.)

### Honest reading

- **T-SAR is not a fair "runs today on a Pi" comparison.** It is a *co-design*:
  the 8–12× assume a modified SIMD ALU (hence the area/power overhead from
  synthesis). You do **not** get those numbers on stock A76 silicon. Its
  cross-platform decode figure (Llama-1.58-**8B** at 128.96 t/s on a 9950X)
  exceeds what a stock DDR5 box can stream for a >1 GB model, which is consistent
  with its converting decode from memory- to compute-bound *on co-designed
  hardware*. Not portable software.
- **But T-SAR names geist's exact wall.** geist's decode is memory-bound on the
  A76 (`~9 of ~13 GB/s`; "micro-opts don't pan out", above). T-SAR's insight —
  keep the ternary LUT in the SIMD register file instead of DRAM — is the
  research answer to that bottleneck. A *software* approximation (in-register
  shuffle-LUT rather than DRAM TL tables) is the only non-structural A76 decode
  lever left. Low priority: geist decode is already at parity with llama.cpp and
  ~2× ahead of bitnet.cpp, so this is a "if a real gap appears" item.
- **FairyFuse is irrelevant to the Pi moat.** AVX-512 only (no ARM), does not
  test BitNet at all (its own Fairy2i 2-bit on Llama-2-7B), and 1.24× vs 4-bit on
  a 48-core server. It touches at most geist's x86 path, and even there with a
  different model/quant. Its mult-free idea geist already approximates (the
  `unbiased` SDOT trick defers the −1 rather than multiplying).

### Where geist stands

- **Weaker on kernel novelty.** As this doc already records, geist's decode inner
  loop is *the same algorithm* as llama.cpp's `ggml_vec_dot_tq2_0_q8_K`; the 2×
  over bitnet.cpp is largely bitnet.cpp's TL1 being slow on the A76, not a better
  geist kernel. T-SAR (register-LUT) and FairyFuse (mult-free) are genuinely more
  advanced *as kernels*.
- **Stronger as a system, on the axis that matters.** For the moat metric —
  **stock ARM, real BitNet 2B-4T, runs today, in a <1 MB embeddable single-codebase
  lib** — neither paper fields a competing number: T-SAR needs new silicon,
  FairyFuse needs AVX-512 and doesn't do ARM or BitNet. geist's claim is
  *portable, embeddable, self-contained*, not *fastest kernel*. Positioning
  should say exactly that (see [../docs/POSITIONING.md](../docs/POSITIONING.md)).

Sources: [T-SAR](https://arxiv.org/abs/2511.13676),
[FairyFuse](https://arxiv.org/pdf/2604.20913),
[bitnet.cpp](https://arxiv.org/html/2502.11880v1),
[BitNet b1.58 2B4T](https://arxiv.org/pdf/2504.12285).
