/*
 * src/archs/transformer/forward/spec_head.c — speculative i8 output head.
 *
 * Layer: ARCHITECTURE (Pi/NEON fast path; no-op elsewhere).
 *
 * Decode on the BitNet-2B-4T i2_s model is memory-bandwidth bound, and the
 * single biggest read per token is the tied F16 lm_head: 2560 x 128256 x 2 B
 * = 656 MB, ~50% of the whole decode step (measured: GEIST_PROFILE_FORWARD).
 *
 * This module cuts that to ~82 MB via a two-stage speculative projection
 * (same idea as petlukk/Cougar's ARM sketch path):
 *
 *   build (once): quantize a STRIDE-subsampled copy of every embedding row to
 *     int8 — a [VOCAB, HIDDEN/STRIDE] "sketch" (~82 MB) plus one f32 scale per
 *     row. Reading the full F16 table happens exactly once, at first decode.
 *
 *   phase 1 (per token): rough-score the whole vocabulary with an int8 SDOT
 *     against the sketch (reads only the sketch, parallel over rows).
 *   phase 2: select the top-K highest rough scores (bounded min-heap).
 *   phase 3: compute high-precision logits for just those K rows.
 *
 * Also handles Gemma 4, whose tied lm_head is Q6_K [1536, 262144] (~32 % of
 * decode): the sketch is built by dequantizing each row. spec_dtype_ok() lists
 * the eligible embedding dtypes.
 *
 * Phase-3 precision and greedy reproducibility:
 *  - F16 lm_head (BitNet 2B-4T): phase 3 is the *same* f16xf32 dot the dense
 *    head uses, so finalist logits are identical.
 *  - source-layout quantized lm_head (Gemma Q6_K): phase 3 builds a one-row
 *    view of the weight and calls the *same* linear_m1 kernel the dense head
 *    uses (W6A8 for Q6_K), so finalist logits are bit-exact too. (Repacked
 *    layouts fall back to an f32-dequant dot — high precision, not bit-exact.)
 *
 * So the only approximation is phase 2 — *which* rows become finalists. The
 * true argmax must be among the top-K; recall grows with the vocab, so TOP_K's
 * default is vocab-aware (512 for ~128 K, 4096 for ~256 K). At those sizes
 * greedy output is byte-identical to the dense head (verified: BitNet 128 K and
 * Gemma 256 K). A too-small TOP_K (e.g. 512 on Gemma) drops the argmax on a
 * near-tie and the continuation diverges — coherent, not reproducible. Tune
 * with GEIST_SPEC_TOPK / GEIST_SPEC_STRIDE.
 *
 * Default ON for greedy decode on an eligible tied lm_head (GEIST_SPEC_HEAD=0
 * forces the exact dense head). Non-greedy sampling, ineligible dtypes, and
 * non-NEON/dotprod hosts always fall back to the dense head — extending the
 * sketch to top-k sampling failed its exactness gate (see the measured-out
 * note at the sampling check below). For source-layout quantized / F16 heads
 * the finalist logits are bit-exact so greedy output is byte-identical;
 * repacked layouts use a high-precision (not bit-exact) phase-3.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../arch_state.h"
#include "../forward.h"

#include <geist.h>
#include <geist_backend.h>

#include "../../../engine/sampler.h"
#include "quant.h"
#include "heap.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* The sketch SDOT + finalist dots need either NEON+dotprod or x86 AVX2(+F16C
 * for the F16 head). Both are gated below; everything else gets the stub. */
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__)
#define SPEC_HEAD_AVAILABLE 1
#endif

/* Sketch resolution, not a divisor. The stride used to be fixed at 4, which
 * ties the sketch's quality to how wide the model happens to be: the same
 * divisor gave the 2B-4T (H=2560) a 640-dim sketch and bitnet_b1_58-large
 * (H=1536) a 384-dim one. The coarser sketch loses the true argmax, and since
 * there is no fallback the loss is silent -- measured 2026-08-01, greedy
 * output matched the dense head for 28 tokens and then diverged, at the
 * shipped defaults. Deriving the stride from a target resolution instead makes
 * the sketch equally good on any width; it reproduces stride 4 for the 2B-4T
 * and stride 2 for the large model, both of which measure exact. */
#define SPEC_SKETCH_DIMS 640

/* How much smaller a sketch row must be than a dense row before the sketch is
 * worth building. This is the whole trade in one number: the sketch replaces
 * reading a dense row per vocabulary entry with reading a sketch row, so if
 * the sketch row is not much smaller there is nothing to win -- and the
 * phase-3 verify and a resident V*SD table still get paid for.
 *
 * Gating on dtype, as this used to, asks the wrong question. What decides the
 * trade is not whether the head is F16 or Q6_K but how wide a dense row is
 * against the sketch replacing it. Measured on a Pi 5, decode against the same
 * build with GEIST_SPEC_HEAD=0:
 *
 *   BitNet 2B-4T        F16  head  5120 B/row  640-dim sketch  8.00x   +82 %
 *   bitnet_b1_58-large  Q6_K head  1260 B/row  768-dim sketch  1.64x   +0.8 %
 *   gemma4-e2b          Q6_K head  1260 B/row  768-dim sketch  1.64x   +0.5 %
 *
 * A quantized head is already compact, so subsampling it to int8 shrinks
 * almost nothing. 3.0 sits between the two measured regimes, and the gap they
 * leave is wide enough that its exact position hardly matters.
 *
 * Declining also hands memory back where the sketch bought nothing: on gemma4
 * its table is 201 MB resident, which is real on a 4 GB board, for +0.5 %. */
#define SPEC_MIN_ROW_SHRINK 3.0

/* 512 was verified byte-identical on the original trajectories, but the
 * margin is thin: a near-tie whose loser the sketch ranks past the cutoff
 * flips greedy (seen live when an unrelated ULP-level kernel change shifted
 * the trajectory, #102 Phase 2). 1024 doubles the recall margin for ~3 % of
 * the head read (phase 3 is 2.6 → 5.2 MB next to the 82 MB sketch). */
#define SPEC_TOPK_DEFAULT 1024

/* (rough score, vocab index) entry of the top-K min-heap. */
struct spec_cand {
    float    s;
    uint32_t idx;
};

/* Eligible with NEON+dotprod or x86 AVX2 (the sketch dot kernel) and a large
 * tied lm_head (F16 on BitNet 2B-4T, Q6_K on Gemma 4 — see spec_dtype_ok). */
#ifdef SPEC_HEAD_AVAILABLE

static size_t spec_env_sz(const char *name, size_t dflt, size_t lo, size_t hi) {
    const char *e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return dflt;
    }
    long v = strtol(e, nullptr, 10);
    if (v < (long) lo) {
        v = (long) lo;
    }
    if (v > (long) hi) {
        v = (long) hi;
    }
    return (size_t) v;
}

static int spec_verify_env(void) {
    /* GEIST_SPEC_VERIFY=1 turns on the certainty bound: every excluded row gets
     * an upper bound on the exact logit it could have had, and the head
     * declines whenever one of them could have beaten the winner, which makes
     * the answer provably the dense head's.
     *
     * Off by default, for a measured reason rather than an assumed one. At the
     * shipped stride the sketch keeps one dimension in four, so the dropped
     * three quarters carry an uncertainty of the same order as the dot product
     * itself -- Cauchy-Schwarz must assume they all conspire. The bound is then
     * almost never satisfiable: on a Pi 5 it fires on nearly every token and
     * decode falls from 18.3 to 9.4 t/s, which is dense-head speed. Correct,
     * and worth nothing at that price.
     *
     * What it is worth is as an instrument. Run a model under it once and the
     * output is either identical to the dense head or it is not -- no sampling,
     * no luck -- which is how the silent divergence on bitnet_b1_58-large would
     * have been caught the day it appeared instead of months later. */
    const char *e = getenv("GEIST_SPEC_VERIFY");
    return (e != nullptr && e[0] == '1') ? 1 : 0;
}

static int spec_head_env(void) {
    /* Default ON for greedy decode + top-k sampling on an eligible tied
     * lm_head: verified byte-identical to the dense head on Gemma 4 (Q6_K,
     * 256 K vocab) and BitNet (F16, 128 K) with the vocab-aware TOP_K.
     * Ineligible modes/dtypes/hosts always fall back to the exact dense head
     * regardless. GEIST_SPEC_HEAD=0 forces it off. Read per call (once per
     * decoded token — noise next to the head itself) so tests can toggle it
     * within one process. */
    const char *e = getenv("GEIST_SPEC_HEAD");
    return (e != nullptr && e[0] == '0') ? 0 : 1;
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

/* Signed int8 dot, n a multiple of 16 in practice (scalar tail otherwise). */
static inline int32_t spec_i8dot(const int8_t *a, const int8_t *b, size_t n) {
    int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
    size_t    k = 0;
    for (; k + 32 <= n; k += 32) {
        acc0 = vdotq_s32(acc0, vld1q_s8(a + k), vld1q_s8(b + k));
        acc1 = vdotq_s32(acc1, vld1q_s8(a + k + 16), vld1q_s8(b + k + 16));
    }
    for (; k + 16 <= n; k += 16) {
        acc0 = vdotq_s32(acc0, vld1q_s8(a + k), vld1q_s8(b + k));
    }
    int32_t s = vaddvq_s32(vaddq_s32(acc0, acc1));
    for (; k < n; k++) {
        s += (int32_t) a[k] * (int32_t) b[k];
    }
    return s;
}

/* Exact f16(weight) x f32(activation) dot — in-register convert, no f32
 * materialization (mirrors cpu_neon_w_f16_m1's inner loop). */
static inline float spec_f16dot(const uint16_t *wr, const float *x, size_t n) {
    const float16_t *w  = (const float16_t *) wr;
    float32x4_t      a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    size_t           k = 0;
    for (; k + 8 <= n; k += 8) {
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vld1_f16(w + k)), vld1q_f32(x + k));
        a1 = vfmaq_f32(a1, vcvt_f32_f16(vld1_f16(w + k + 4)), vld1q_f32(x + k + 4));
    }
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; k < n; k++) {
        s += (float) w[k] * x[k];
    }
    return s;
}

/* Exact f32(weight) x f32(activation) dot — used for the top-K finalists when
 * the embedding is a quantized (block) type that must be dequantized first. */
static inline float spec_f32dot(const float *w, const float *x, size_t n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    size_t      k = 0;
    for (; k + 8 <= n; k += 8) {
        a0 = vfmaq_f32(a0, vld1q_f32(w + k), vld1q_f32(x + k));
        a1 = vfmaq_f32(a1, vld1q_f32(w + k + 4), vld1q_f32(x + k + 4));
    }
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; k < n; k++) {
        s += w[k] * x[k];
    }
    return s;
}

#else /* x86 AVX2 / F16C / FMA (the spec_head.c baseline is x86-64-v3) */

static inline float spec_hsum256(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128       s  = _mm_add_ps(lo, hi);
    s               = _mm_hadd_ps(s, s);
    s               = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* Signed int8 dot via sign-extend to int16 + madd (Zen5 has no s8×s8 VNNI). */
static inline int32_t spec_i8dot(const int8_t *a, const int8_t *b, size_t n) {
    __m256i acc = _mm256_setzero_si256();
    size_t  k   = 0;
    for (; k + 16 <= n; k += 16) {
        const __m256i av = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *) (a + k)));
        const __m256i bv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *) (b + k)));
        acc              = _mm256_add_epi32(acc, _mm256_madd_epi16(av, bv));
    }
    const __m128i lo = _mm256_castsi256_si128(acc);
    const __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i       s  = _mm_add_epi32(lo, hi);
    s                = _mm_hadd_epi32(s, s);
    s                = _mm_hadd_epi32(s, s);
    int32_t r        = _mm_cvtsi128_si32(s);
    for (; k < n; k++) {
        r += (int32_t) a[k] * (int32_t) b[k];
    }
    return r;
}

/* Exact f16(weight) x f32(activation) dot — F16C in-register convert. */
static inline float spec_f16dot(const uint16_t *wr, const float *x, size_t n) {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    size_t k  = 0;
    for (; k + 16 <= n; k += 16) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (wr + k))),
                             _mm256_loadu_ps(x + k),
                             a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (wr + k + 8))),
                             _mm256_loadu_ps(x + k + 8),
                             a1);
    }
    for (; k + 8 <= n; k += 8) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (wr + k))),
                             _mm256_loadu_ps(x + k),
                             a0);
    }
    float s = spec_hsum256(_mm256_add_ps(a0, a1));
    for (; k < n; k++) {
        s += fp16_to_fp32(wr[k]) * x[k];
    }
    return s;
}

static inline float spec_f32dot(const float *w, const float *x, size_t n) {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    size_t k  = 0;
    for (; k + 16 <= n; k += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k + 8), _mm256_loadu_ps(x + k + 8), a1);
    }
    for (; k + 8 <= n; k += 8) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x + k), a0);
    }
    float s = spec_hsum256(_mm256_add_ps(a0, a1));
    for (; k < n; k++) {
        s += w[k] * x[k];
    }
    return s;
}

#endif

/* Embedding dtypes the spec head can read. The tied lm_head is F16 on BitNet
 * 2B-4T and Q6_K on Gemma 4; the block-quantized ones are dequantized one row
 * at a time via geist's gguf_quant row helpers. */
static bool spec_dtype_ok(uint16_t dt) {
    switch (dt) {
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
    case GEIST_DTYPE_F32:
    case GEIST_DTYPE_Q3_K:
    case GEIST_DTYPE_Q4_K:
    case GEIST_DTYPE_Q5_K:
    case GEIST_DTYPE_Q6_K:
    case GEIST_DTYPE_Q8_0:
        return true;
    default:
        return false;
    }
}

/* Byte stride between consecutive embedding rows for `dt` (n elements/row). */
static size_t spec_row_stride(uint16_t dt, size_t n) {
    switch (dt) {
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
        return n * 2;
    case GEIST_DTYPE_F32:
        return n * 4;
    case GEIST_DTYPE_Q3_K:
        return n / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES;
    case GEIST_DTYPE_Q4_K:
        return n / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES;
    case GEIST_DTYPE_Q5_K:
        return n / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES;
    case GEIST_DTYPE_Q6_K:
        return n / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES;
    case GEIST_DTYPE_Q8_0:
        return n / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES;
    default:
        return 0;
    }
}

/* Dequantize one embedding row (`row` points at its first byte) to f32. */
static void spec_row_to_f32(uint16_t dt, const uint8_t *row, size_t n, float *dst) {
    switch (dt) {
    case GEIST_DTYPE_F16: {
        const uint16_t *w = (const uint16_t *) row;
        for (size_t i = 0; i < n; i++) {
            dst[i] = fp16_to_fp32(w[i]);
        }
        break;
    }
    case GEIST_DTYPE_BF16: {
        const uint16_t *w = (const uint16_t *) row;
        for (size_t i = 0; i < n; i++) {
            uint32_t f = (uint32_t) w[i] << 16;
            memcpy(&dst[i], &f, sizeof f);
        }
        break;
    }
    case GEIST_DTYPE_F32:
        memcpy(dst, row, n * sizeof(float));
        break;
    case GEIST_DTYPE_Q3_K:
        dequant_q3_K_row(n, row, dst);
        break;
    case GEIST_DTYPE_Q4_K:
        dequant_q4_K_row(n, row, dst);
        break;
    case GEIST_DTYPE_Q5_K:
        dequant_q5_K_row(n, row, dst);
        break;
    case GEIST_DTYPE_Q6_K:
        dequant_q6_K_row(n, row, dst);
        break;
    case GEIST_DTYPE_Q8_0:
        dequant_q8_0_row(n, row, dst);
        break;
    default:
        memset(dst, 0, n * sizeof(float));
        break;
    }
}

/* Build the sketch table + per-row scales from the tied lm_head (any dtype in
 * spec_dtype_ok). Returns true on success, false if ineligible/OOM. */
static bool spec_head_build(struct transformer_arch_state *st) {
    const struct geist_weight *w = &st->embed_table_w;
    if (w->raw == nullptr || !spec_dtype_ok(w->dtype)) {
        return false;
    }
#if !(defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    /* x86 port: only the F16 tied lm_head (BitNet 2B-4T) is validated
     * byte-identical to the dense head (exact f16 phase-3). The quantized /
     * repacked-layout phase-3 paths (e.g. Gemma Q6_K) are NEON-validated only,
     * so on x86 those heads keep the exact dense decode. */
    if (w->dtype != GEIST_DTYPE_F16) {
        return false;
    }
#endif
    const size_t H = (size_t) w->n_in;
    const size_t V = (size_t) w->n_out;
    if (H != (size_t) st->d_model || V != (size_t) st->vocab_size) {
        return false;
    }
    const size_t row_bytes = spec_row_stride(w->dtype, H);
    if (row_bytes == 0) {
        return false;
    }

    /* Subsample stride and finalist count are tunable: a larger vocabulary /
     * harder ranking (Gemma's 256 K) needs a finer sketch (smaller stride)
     * and/or more finalists for the true argmax to land in the candidate set.
     * Defaults (4, 512) are exact for BitNet's 128 K F16 head. */
    const size_t stride_dflt = H > SPEC_SKETCH_DIMS ? H / SPEC_SKETCH_DIMS : 1;
    size_t       sub         = spec_env_sz("GEIST_SPEC_STRIDE", stride_dflt, 1, H);
    while (sub > 1 && H % sub != 0) {
        sub--;
    } /* require H % sub == 0 */
    /* Default finalist count is vocab-aware: 512 keeps the true argmax for a
     * ~128 K vocab (BitNet, exact); a ~256 K vocab (Gemma) needs ~4096 for the
     * argmax to land in the candidate set. Measured crossover. Override with
     * GEIST_SPEC_TOPK. */
    const size_t topk_dflt = (V >= 200000) ? 4096 : SPEC_TOPK_DEFAULT;
    size_t       topk      = spec_env_sz("GEIST_SPEC_TOPK", topk_dflt, 16, V);
    if (V < 8192) {
        return false;
    }
    const size_t SD = H / sub;

    /* Is a sketch row enough smaller than a dense one to pay for itself? */
    if ((double) row_bytes / (double) SD < SPEC_MIN_ROW_SHRINK) {
        return false;
    }

    int8_t *sketch = heap_alloc_array_aligned(int8_t, V *SD);
    float  *rscale = heap_alloc_array_aligned(float, V);
    /* Row halves of the safety bound: two floats a row -- 1 MB against an
     * 82 MB sketch on the 2B-4T. */
    float *wdrop = heap_alloc_array_aligned(float, V);
    float *wl1   = heap_alloc_array_aligned(float, V);
#ifdef _OPENMP
    const size_t nthr = (size_t) omp_get_max_threads();
#else
    const size_t nthr = 1;
#endif
    float *scratch = heap_alloc_array_aligned(float, nthr *H); /* per-thread */
    if (sketch == nullptr || rscale == nullptr || scratch == nullptr || wdrop == nullptr ||
        wl1 == nullptr) {
        safe_free((void **) &wdrop);
        safe_free((void **) &wl1);
        safe_free((void **) &sketch);
        safe_free((void **) &rscale);
        safe_free((void **) &scratch);
        return false;
    }

    const uint8_t *const Wbase = (const uint8_t *) w->raw;
    const uint16_t       dt    = w->dtype;

    /* Choose which dimensions the sketch keeps, by variance across the
     * vocabulary instead of by a fixed stride. A stride keeps every n-th
     * coordinate whether or not it separates one token from another; the
     * coordinates that do are not spread evenly, so choosing them buys ranking
     * quality at exactly the same width.
     *
     * Variance is estimated from a sample of rows -- a few hundred is plenty
     * for a per-dimension spread, and a full pass here would double the load
     * cost of a table this size for no better answer. */
    uint32_t *dims = heap_alloc_array_aligned(uint32_t, SD);
    float    *dvar = heap_alloc_array_aligned(float, H);
    float    *dsum = heap_alloc_array_aligned(float, H);
    if (dims == nullptr || dvar == nullptr || dsum == nullptr) {
        safe_free((void **) &dims);
        safe_free((void **) &dvar);
        safe_free((void **) &dsum);
        safe_free((void **) &wdrop);
        safe_free((void **) &wl1);
        safe_free((void **) &sketch);
        safe_free((void **) &rscale);
        safe_free((void **) &scratch);
        return false;
    }
    for (size_t i = 0; i < H; i++) {
        dvar[i] = 0.0f;
        dsum[i] = 0.0f;
    }
    const size_t vstep = V > 512 ? V / 512 : 1;
    size_t       vn    = 0;
    for (size_t r = 0; r < V; r += vstep, vn++) {
        /* `scratch` is the per-thread row buffer of the parallel build below;
         * this pass is serial and runs first, so its first H floats are free. */
        spec_row_to_f32(dt, Wbase + r * row_bytes, H, scratch);
        for (size_t i = 0; i < H; i++) {
            dsum[i] += scratch[i];
            dvar[i] += scratch[i] * scratch[i];
        }
    }
    for (size_t i = 0; i < H; i++) {
        const float mean = dsum[i] / (float) vn;
        dvar[i]          = dvar[i] / (float) vn - mean * mean;
    }
    /* Partial selection of the SD highest-variance dimensions: repeatedly take
     * the current maximum. SD is a few hundred against H a few thousand, so
     * this is cheap and needs no sort. */
    for (size_t s = 0; s < SD; s++) {
        size_t best = 0;
        for (size_t i = 1; i < H; i++) {
            if (dvar[i] > dvar[best]) {
                best = i;
            }
        }
        dims[s]    = (uint32_t) best;
        dvar[best] = -1.0f; /* consumed */
    }
    safe_free((void **) &dvar);
    safe_free((void **) &dsum);
    /* Ascending order keeps the gather sequential, which the int8 dot likes. */
    for (size_t a = 0; a + 1 < SD; a++) {
        for (size_t b = a + 1; b < SD; b++) {
            if (dims[b] < dims[a]) {
                const uint32_t tmpd = dims[a];
                dims[a]             = dims[b];
                dims[b]             = tmpd;
            }
        }
    }
    /* Mark kept dimensions so the bound terms can split on membership rather
     * than on the arithmetic the stride used to imply. */
    bool *kept = heap_alloc_array_aligned(bool, H);
    if (kept != nullptr) {
        for (size_t i = 0; i < H; i++) {
            kept[i] = false;
        }
        for (size_t s = 0; s < SD; s++) {
            kept[dims[s]] = true;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < V; r++) {
#ifdef _OPENMP
        float *tmp = scratch + (size_t) omp_get_thread_num() * H;
#else
        float *tmp = scratch;
#endif
        spec_row_to_f32(dt, Wbase + r * row_bytes, H, tmp);
        float amax = 1e-8f;
        for (size_t s = 0; s < SD; s++) {
            const float v = tmp[dims[s]];
            const float a = v < 0.0f ? -v : v;
            if (a > amax) {
                amax = a;
            }
        }
        const float scale = 127.0f / amax;
        int8_t     *sk    = sketch + r * SD;
        for (size_t s = 0; s < SD; s++) {
            int32_t q = (int32_t) lrintf(tmp[dims[s]] * scale);
            if (q > 127) {
                q = 127;
            }
            if (q < -127) {
                q = -127;
            }
            sk[s] = (int8_t) q;
        }
        rscale[r] = amax / 127.0f;
        /* The sketch keeps every sub-th dimension and is blind to the rest, so
         * their combined contribution is what Cauchy-Schwarz has to cover. */
        float sq = 0.0f, l1 = 0.0f;
        for (size_t i = 0; i < H; i++) {
            if (kept != nullptr && kept[i]) {
                l1 += tmp[i] < 0.0f ? -tmp[i] : tmp[i];
            } else {
                sq += tmp[i] * tmp[i];
            }
        }
        wdrop[r] = sqrtf(sq);
        wl1[r]   = l1;
    }
    safe_free((void **) &scratch);

    st->spec_sketch     = sketch;
    st->spec_w_drop     = wdrop;
    st->spec_w_l1       = wl1;
    st->spec_row_scale  = rscale;
    st->spec_sketch_dim = SD;
    st->spec_dims       = dims;
    st->spec_stride     = sub;
    st->spec_topk       = topk;
    return true;
}

/* Eager build at state_create (model-level; sessions only carry scratch).
 * Immutable afterwards, so concurrent sessions read the sketch without
 * coordination. Skipped — spec_state = -1 — when disabled by env, on a
 * batched-submit GPU backend (the sketch is a HOST optimization), or when
 * the head is ineligible. */
void transformer_spec_head_init(struct transformer_arch_state *st) {
    if (spec_head_env() == 0 || st->backend->desc->caps.batched_submit) {
        st->spec_state = -1;
        return;
    }
    st->spec_state = spec_head_build(st) ? 1 : -1;
}

bool transformer_spec_head_try(struct transformer_arch_session *sess, geist_token_t *out_token) {
    struct transformer_arch_state *st = sess->model;
    if (spec_head_env() == 0) {
        return false;
    }
    /* Built eagerly at state_create (transformer_spec_head_init); 0 means
     * init never ran (direct-state caller skipping state_create — treat
     * as inactive), -1 means ineligible or disabled. */
    if (st->spec_state != 1 || sess->spec_x_i8 == nullptr) {
        return false;
    }
    /* A tuned lm_head gain (GEIST_TUNE) scales the dense head's logits; the
     * spec head computes raw dots against the tied embedding and would bypass
     * it. Argmax survives any positive uniform scale, but the trained gain is
     * unconstrained — decline and take the dense head rather than diverge
     * silently at gain <= 0. One load + compare per decoded token. */
    if (st->embed_table_w.gain_slot != nullptr && *st->embed_table_w.gain_slot != 1.0f) {
        return false;
    }
    /* Greedy only. Extending this to top-k sampling was tried and MEASURED
     * OUT (#102 Phase 1): exact sampling needs the true top-k rows inside the
     * candidate set, and the stride-4 sketch's rank noise beyond rank 1 is
     * enormous — on BitNet 2B-4T some true top-40 rows rank outside even the
     * top-8192 rough candidates, so sampled output diverged from the exact
     * dense head at every tested TOP_K. Perfect rank-40 recall required a
     * stride-1 (full-width) int8 phase 1 — the same bytes as the Q8 dense
     * head, i.e. no win. The sketch ranks only the argmax reliably; greedy is
     * where the recall contract holds (byte-identical, verified in
     * tests/test_spec_head_sampling_int.c). */
    if (sess->temperature != 0.0f) {
        return false;
    }

    struct geist_backend            *be     = st->backend;
    const struct geist_backend_vtbl *v      = be->desc->vtbl;
    const size_t                     H      = (size_t) st->d_model;
    const size_t                     V      = (size_t) st->vocab_size;
    const size_t                     SD     = st->spec_sketch_dim;
    const uint32_t *const            dims   = st->spec_dims;
    const size_t                     topk   = st->spec_topk;
    const uint16_t                   dt     = st->embed_table_w.dtype;
    const uint8_t *const             Wbase  = (const uint8_t *) st->embed_table_w.raw;
    const size_t                     stride = spec_row_stride(dt, H);

    const float *h      = (const float *) v->buffer_map(sess->scratch_h_a);
    float       *logits = (float *) v->buffer_map(sess->scratch_logits);
    if (h == nullptr || logits == nullptr) {
        if (h != nullptr) {
            v->buffer_unmap(sess->scratch_h_a);
        }
        if (logits != nullptr) {
            v->buffer_unmap(sess->scratch_logits);
        }
        return false;
    }

    /* Quantize the activation, build the subsampled sketch activation. */
    int8_t *x_i8 = sess->spec_x_i8;
    int8_t *a_sk = sess->spec_act_sketch;
    float   amax = 1e-8f;
    for (size_t i = 0; i < H; i++) {
        const float a = h[i] < 0.0f ? -h[i] : h[i];
        if (a > amax) {
            amax = a;
        }
    }
    const float x_q = 127.0f / amax;
    for (size_t i = 0; i < H; i++) {
        int32_t q = (int32_t) lrintf(h[i] * x_q);
        if (q > 127) {
            q = 127;
        }
        if (q < -127) {
            q = -127;
        }
        x_i8[i] = (int8_t) q;
    }
    for (size_t s = 0; s < SD; s++) {
        a_sk[s] = x_i8[dims[s]];
    }

    /* Phase 1: rough scores over the whole vocab via the i8 sketch. The
     * common x_scale factor is dropped — it does not affect the ranking. */
    float        *rough  = sess->spec_rough;
    const int8_t *sketch = st->spec_sketch;
    const float  *rscale = st->spec_row_scale;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < V; r++) {
        const int32_t d = spec_i8dot(a_sk, sketch + r * SD, SD);
        rough[r]        = (float) d * rscale[r];
    }

    /* Activation halves of the bound: L1 over the dimensions the sketch keeps,
     * L2 over those it drops. One O(H) pass. */
    float h_l1_kept = 0.0f, h_sq_drop = 0.0f;
    for (size_t i = 0; i < H; i++) {
        h_sq_drop += h[i] * h[i];
    }
    for (size_t s = 0; s < SD; s++) {
        const float hv = h[dims[s]];
        h_l1_kept += hv < 0.0f ? -hv : hv;
        h_sq_drop -= hv * hv; /* kept dimensions are not dropped */
    }
    if (h_sq_drop < 0.0f) {
        h_sq_drop = 0.0f; /* rounding */
    }
    const float h_l2_drop = sqrtf(h_sq_drop);

    /* Phase 2: top-K by rough score, via a bounded min-heap of (score, idx).
     * heap[0] is the smallest score currently retained. */
    struct spec_cand *heap = (struct spec_cand *) sess->spec_heap;
    size_t            hn   = 0;
    for (size_t r = 0; r < V; r++) {
        const float sc = rough[r];
        if (hn < topk) {
            /* sift up */
            size_t i    = hn++;
            heap[i].s   = sc;
            heap[i].idx = (uint32_t) r;
            while (i > 0) {
                size_t p = (i - 1) >> 1;
                if (heap[p].s <= heap[i].s) {
                    break;
                }
                struct spec_cand t = heap[p];
                heap[p]            = heap[i];
                heap[i]            = t;
                i                  = p;
            }
        } else if (sc > heap[0].s) {
            heap[0].s   = sc;
            heap[0].idx = (uint32_t) r;
            /* sift down */
            size_t i = 0;
            for (;;) {
                size_t l = 2 * i + 1, rgt = 2 * i + 2, sm = i;
                if (l < topk && heap[l].s < heap[sm].s) {
                    sm = l;
                }
                if (rgt < topk && heap[rgt].s < heap[sm].s) {
                    sm = rgt;
                }
                if (sm == i) {
                    break;
                }
                struct spec_cand t = heap[sm];
                heap[sm]           = heap[i];
                heap[i]            = t;
                i                  = sm;
            }
        }
    }

    /* Phase 3: exact logits for the K candidates only; everything else is -inf
     * so the argmax is taken from the candidate set.
     *   - F16: in-register f16xf32 dot — identical to the dense F16 kernel.
     *   - source-layout quantized (e.g. Gemma Q6_K, plain row-major): build a
     *     one-row view and call the SAME linear_m1 the dense head uses, so the
     *     finalist logits are bit-exact (W6A8 etc.) — greedy matches the dense
     *     path. (No OMP fork: that kernel runs serially for n_out=1.)
     *   - otherwise (repacked layouts): dequantize the row to f32 and dot in
     *     f32 — high precision but not bit-identical to the dense kernel. */
    /* One-row-view + linear_m1 requires a kernel that reads w->raw in source
     * layout. An aux repack (e.g. the x86 Q8 lm_head blob) indexes its aux
     * data by n_out, which a 1-row view breaks — those fall to the local
     * dots below. */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    /* NEON: keep the local F16 dot (mirrors cpu_neon_w_f16_m1's loop, Pi-
     * validated); its kernels run n_out=1 inside an OMP region, which would
     * cost a fork per finalist. Quantized heads keep the one-row-view path. */
    const bool f16_via_kernel = false;
#else
    /* x86: route F16 finalists through the SAME compiled f16_gemv_m1 row
     * loop the dense head uses (it skips the OMP fork for n_out == 1) —
     * greedy byte-identity holds by construction instead of by luck. The
     * previous local spec_f16dot was a second -ffast-math compilation of
     * the same math and drifted a ULP on near-ties (#102 Phase 2). */
    const bool f16_via_kernel = true;
#endif
    const bool exact_kernel = st->embed_table_w.linear_m1 != nullptr &&
                              st->embed_table_w.backend_layout == GEIST_W_LAYOUT_SOURCE &&
                              (st->embed_table_w.flags & GEIST_W_AUX_BACKEND_REPACK) == 0 &&
                              (dt != GEIST_DTYPE_F16 || f16_via_kernel);
    for (size_t i = 0; i < V; i++) {
        logits[i] = -INFINITY;
    }
    for (size_t c = 0; c < hn; c++) {
        const size_t   row = heap[c].idx;
        const uint8_t *rb  = Wbase + row * stride;
        if (dt == GEIST_DTYPE_F16 && !exact_kernel) {
            /* NEON, or an aux-repacked F16 head (x86 Q8 blob): exact f16
             * dot. For the Q8 case the dense head is int8-approximate, so
             * bit-identity to it is neither achievable nor desirable. */
            logits[row] = spec_f16dot((const uint16_t *) rb, h, H);
        } else if (exact_kernel) {
            struct geist_weight rw = st->embed_table_w;
            rw.n_out               = 1;
            rw.raw                 = rb;
            rw.raw_nbytes          = stride; /* this single row, not the table */
            rw.aux_fp32            = nullptr;
            rw.aux_n               = 0;
            float out1             = 0.0f;
            rw.linear_m1(h, &rw, be, &out1);
            logits[row] = out1;
        } else {
            spec_row_to_f32(dt, rb, H, sess->spec_row_f32);
            logits[row] = spec_f32dot(sess->spec_row_f32, h, H);
        }
    }

    *out_token = geist_sampler_argmax(V, logits);

    /* Certainty check. Everything above approximates, in order to choose which
     * rows to score exactly; this decides whether that choice was safe.
     *
     * A rough score differs from the exact logit by three bounded amounts --
     * the dropped dimensions, the weight quantization and the activation
     * quantization -- so every excluded row has an upper bound on what its
     * exact logit could have been. If none reaches the winner's exact logit,
     * no excluded row could have won and the answer is provably the dense
     * head's. If one does, we do not guess: the head declines and head.c
     * recomputes densely.
     *
     * That turns the failure this head used to have -- a silently different
     * token, which took a 28-token trajectory on another model to notice --
     * into an occasional slower step. */
    /* Cutoff from the min-heap: nothing below it made the candidate set. Read
     * before the scan because phase 3 overwrites logits, never rough. */
    const float  cutoff     = hn > 0 ? heap[0].s : -3.4e38f;
    const float  best_exact = logits[*out_token];
    const float  inv_xq     = 1.0f / x_q;
    const float *wdrop      = st->spec_w_drop;
    const float *wl1        = st->spec_w_l1;
    bool         certain    = true;
    if (spec_verify_env() && wdrop != nullptr && wl1 != nullptr) {
        for (size_t r = 0; r < V; r++) {
            /* Excluded rows are those phase 2 left below the heap's cutoff.
             * This asked `logits[r] != -INFINITY` before, which is a correct
             * question only where infinities survive: the Pi builds with plain
             * -ffast-math, whose finite-math assumption lets the compiler fold
             * that test to always-true. Every row then looked like a
             * candidate, the scan examined nothing, and the bound did nothing
             * at all on the one platform it was written for -- while passing
             * on the Mac, which builds -fno-finite-math-only. A cutoff is
             * finite arithmetic and means the same thing on both. */
            if (rough[r] > cutoff) {
                continue; /* a candidate: already scored exactly */
            }
            const float ub = rough[r] * inv_xq + 0.5f * rscale[r] * h_l1_kept +
                             0.5f * wl1[r] * inv_xq + wdrop[r] * h_l2_drop;
            if (ub > best_exact) {
                certain = false;
                break;
            }
        }
    }
    if (!certain) {
        v->buffer_unmap(sess->scratch_logits);
        v->buffer_unmap(sess->scratch_h_a);
        return false; /* head.c recomputes the dense head */
    }

    /* scratch_logits is now SPARSE (-inf off the candidate set) — right for
     * the greedy argmax above, wrong for value consumers. peek_logits checks
     * this flag and lazily recomputes the dense head (scratch_h_a still holds
     * the normalized hidden this path read). */
    sess->logits_sparse = true;

    v->buffer_unmap(sess->scratch_logits);
    v->buffer_unmap(sess->scratch_h_a);
    return true;
}

#else /* no NEON+dotprod / AVX2: spec head unavailable */

void transformer_spec_head_init(struct transformer_arch_state *st) {
    st->spec_state = -1;
}

bool transformer_spec_head_try(struct transformer_arch_session *sess, geist_token_t *out_token) {
    (void) sess;
    (void) out_token;
    return false;
}

#endif

/* ---- Per-session spec scratch ----------------------------------------- *
 * The sketch and its bound terms are model-owned and immutable; everything
 * written per token lives on the session. Sized from the model's built
 * sketch geometry — a no-op when the spec head is inactive. */

bool transformer_spec_session_scratch_alloc(struct transformer_arch_session *sess) {
    struct transformer_arch_state *st = sess->model;
    if (st->spec_state != 1) {
        return true; /* nothing to allocate */
    }
    const size_t H        = (size_t) st->d_model;
    const size_t V        = (size_t) st->vocab_size;
    sess->spec_x_i8       = heap_alloc_array_aligned(int8_t, H);
    sess->spec_act_sketch = heap_alloc_array_aligned(int8_t, st->spec_sketch_dim);
    sess->spec_rough      = heap_alloc_array_aligned(float, V);
    sess->spec_row_f32    = heap_alloc_array_aligned(float, H);
    sess->spec_heap       = heap_alloc_array_aligned(struct spec_cand, st->spec_topk);
    if (sess->spec_x_i8 == nullptr || sess->spec_act_sketch == nullptr ||
        sess->spec_rough == nullptr || sess->spec_row_f32 == nullptr ||
        sess->spec_heap == nullptr) {
        transformer_spec_session_scratch_free(sess);
        return false;
    }
    return true;
}

void transformer_spec_session_scratch_free(struct transformer_arch_session *sess) {
    safe_free((void **) &sess->spec_x_i8);
    safe_free((void **) &sess->spec_act_sketch);
    safe_free((void **) &sess->spec_rough);
    safe_free((void **) &sess->spec_row_f32);
    safe_free((void **) &sess->spec_heap);
}
