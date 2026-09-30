/*
 * src/backends/cpu_x86/linear_pq2_0.c — cpu_x86 PQ2_0 decode: a W2 x A8 GEMV
 * (AVX2).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * PQ2_0 is PrismML's ternary format for Ternary-Bonsai: 128-element blocks
 * of [fp16 d][32 bytes of 2-bit codes], element j at byte j/4, bits
 * 2*(j%4), value (code - 1) * d (reference decoder: dequant_pq2_0_row,
 * src/formats/gguf/pq2_0.c). It ran on linear_generic.c, which decodes every
 * weight row to fp32 one element at a time: 88 % of a Ternary-Bonsai-2-27B
 * decode, about 1 GB/s of weights (benchmark/results/TERNARY.md).
 *
 * This kernel reads the GGUF bytes as they are — no repack; the 27B file is
 * 7.2 GB, a copy would not fit next to it on a 16 GB host — and stays in
 * integers inside a block. The recipe is cpu_neon's (kernels/pq2_0.c):
 *
 *   - x is quantized once per call to int8 with one absmax scale, the
 *     scheme TERNARY.md validated against fp32 activations on the real
 *     model (the Hadamard rotation keeps the outliers down), with the same
 *     rounding as cpu_neon, so both backends see the same int8 values.
 *     They are stored in the order the codes are packed: shift level l of
 *     code byte m is element 4m + l, so
 *
 *         xq[b*128 + 32*l + m] = q(x[b*128 + 4*m + l])
 *
 *     and each shift level of a 32-byte code load meets 32 contiguous
 *     activation bytes;
 *   - the raw codes (0..3, biased by +1) are maddubs' unsigned operand.
 *     |xq| <= 127, so a pair is at most 2*3*127 and the four levels' sum at
 *     most 3048: the int16 sums never saturate and the block's int32 dot is
 *     exact;
 *   - the +1 bias leaves through the block's activation sum, precomputed
 *     once per call into lane 0 of an 8-lane vector per block.
 *
 * Decode streams every weight byte once per token and the dot keeps up with
 * DRAM, so the hardware prefetchers are what limits it: on a 4-vCPU Xeon
 * (Sapphire Rapids class) the loop reads 18 GB/s on its own and 42 GB/s
 * with a software prefetch PREFETCH_BYTES ahead — the host's read
 * bandwidth. Each thread's rows are contiguous (schedule(static)), so the
 * prefetch address runs on across row boundaries. A prefetch never faults,
 * past the end of the weight included.
 *
 * AVX2 is the backend's x86-64-v3 baseline, so this runs on every host
 * cpu_x86 does. Prefill (M>1) stays on linear_generic.c.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_pq2_0.h"

#include "backend_state.h"

#include "checked.h"
#include "linear_ref.h"
#include "quant.h"

#include <geist_backend.h>
#include <geist_types.h>

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

constexpr size_t QK = PQ2_0_BLOCK_ELEMS; /* 128 */
constexpr size_t BB = PQ2_0_BLOCK_BYTES; /* fp16 d, then 32 code bytes */
static_assert(PQ2_0_BLOCK_ELEMS == 128 && PQ2_0_BLOCK_BYTES == 34,
              "a block's codes are one ymm load: four 2-bit levels of 32 elements");

/* Bias vectors: 8 int32 per block, lane 0 = minus the block's xq sum. */
constexpr size_t BIAS_LANES = 8;

/* 4-8 KB measured best on the host above; 2 KB was 15-20 % slower, 32 KB
 * about 10 %. */
constexpr size_t PREFETCH_BYTES = 4096;

/* Within a 16-byte lane of 4 elements x 4 levels: the bytes grouped by
 * level (the same for both lanes). */
alignas(16) static const int8_t BY_LEVEL[16] = {
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

/* x (nb blocks) to int8 in code order, plus the bias vectors. Returns the
 * dequantization factor of the activation, max|x| / 127. The quantization is
 * cpu_neon's: max|x| floored at 1e-5, q = x * 127 / max rounded half away
 * from zero (so |q| <= 127). */
static float quantize_acts(size_t nb, const float *x, int8_t *xq, int32_t *bias) {
    const size_t n    = nb * QK;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256       mx   = _mm256_set1_ps(1e-5f);
    for (size_t i = 0; i < n; i += 8) {
        mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(x + i), absm));
    }
    __m128 m4           = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
    m4                  = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4                  = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    const float max_abs = _mm_cvtss_f32(m4);

    const __m256 scale = _mm256_set1_ps(127.0f / max_abs);
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 sign  = _mm256_castsi256_ps(_mm256_set1_epi32((int32_t) 0x80000000u));
    /* The packs interleave 128-bit lanes; this restores element order. */
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i by_level =
            _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *) BY_LEVEL));
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (size_t b = 0; b < nb; b++) {
        /* r[k]: elements 32k .. 32k+31 of the block, each 16-byte lane
         * holding its 4 code bytes' elements grouped by level (dword l). */
        __m256i r[4];
        for (size_t k = 0; k < 4; k++) {
            const float *xk = x + b * QK + 32 * k;
            __m256i      q[4];
            for (size_t t = 0; t < 4; t++) {
                const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(xk + 8 * t), scale);
                q[t]           = _mm256_cvttps_epi32(
                        _mm256_add_ps(v, _mm256_or_ps(_mm256_and_ps(v, sign), half)));
            }
            const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(q[0], q[1]),
                                                 _mm256_packs_epi32(q[2], q[3]));
            r[k]            = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(p, order), by_level);
        }
        /* 4 x 8 dword transpose: level l's dwords from every lane, in code
         * byte order. */
        const __m256i t01l  = _mm256_unpacklo_epi32(r[0], r[1]);
        const __m256i t01h  = _mm256_unpackhi_epi32(r[0], r[1]);
        const __m256i t23l  = _mm256_unpacklo_epi32(r[2], r[3]);
        const __m256i t23h  = _mm256_unpackhi_epi32(r[2], r[3]);
        const __m256i lv[4] = {_mm256_unpacklo_epi64(t01l, t23l),
                               _mm256_unpackhi_epi64(t01l, t23l),
                               _mm256_unpacklo_epi64(t01h, t23h),
                               _mm256_unpackhi_epi64(t01h, t23h)};
        __m256i       s16   = _mm256_setzero_si256();
        for (size_t l = 0; l < 4; l++) {
            const __m256i v = _mm256_permutevar8x32_epi32(lv[l], order);
            _mm256_storeu_si256((__m256i *) (xq + b * QK + 32 * l), v);
            s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(ones8, v));
        }
        const __m256i s32 = _mm256_madd_epi16(s16, ones16);
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s32), _mm256_extracti128_si256(s32, 1));
        s4         = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4         = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        _mm256_store_si256((__m256i *) (bias + b * BIAS_LANES),
                           _mm256_setr_epi32(-_mm_cvtsi128_si32(s4), 0, 0, 0, 0, 0, 0, 0));
    }
    return max_abs / 127.0f;
}

/* One block: sum over its 128 elements of code * xq, minus the xq sum, as
 * 8 int32 lanes. */
static inline __m256i block_dot(const uint8_t *qs, const int8_t *xb, const int32_t *bias) {
    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i v   = _mm256_loadu_si256((const __m256i *) qs);
    const __m256i c0  = _mm256_and_si256(v, m3);
    const __m256i c1  = _mm256_and_si256(_mm256_srli_epi16(v, 2), m3);
    const __m256i c2  = _mm256_and_si256(_mm256_srli_epi16(v, 4), m3);
    const __m256i c3  = _mm256_and_si256(_mm256_srli_epi16(v, 6), m3);
    const __m256i p01 = _mm256_add_epi16(
            _mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *) xb)),
            _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *) (xb + 32))));
    const __m256i p23 = _mm256_add_epi16(
            _mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i *) (xb + 64))),
            _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i *) (xb + 96))));
    return _mm256_add_epi32(_mm256_madd_epi16(_mm256_add_epi16(p01, p23), _mm256_set1_epi16(1)),
                            _mm256_load_si256((const __m256i *) bias));
}

/* The block's scale d in all 8 lanes. */
static inline __m256 block_scale(const uint8_t *blk) {
    int16_t d;
    memcpy(&d, blk, sizeof d);
    return _mm256_cvtph_ps(_mm_set1_epi16(d));
}

/* One weight row of nb blocks against the quantized activation. Two
 * accumulators (even / odd block) so the FMA chain is not latency-bound;
 * fixed order, so a row's result does not depend on the thread count. */
static inline float row_dot(size_t nb, const uint8_t *w, const int8_t *xq, const int32_t *bias) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t b    = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *k0 = w + b * BB;
        const uint8_t *k1 = k0 + BB;
        _mm_prefetch((const char *) (k0 + PREFETCH_BYTES), _MM_HINT_T0);
        _mm_prefetch((const char *) (k1 + PREFETCH_BYTES), _MM_HINT_T0);
        acc0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(block_dot(k0 + 2, xq + b * QK, bias + b * BIAS_LANES)),
                block_scale(k0),
                acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(block_dot(
                                       k1 + 2, xq + (b + 1) * QK, bias + (b + 1) * BIAS_LANES)),
                               block_scale(k1),
                               acc1);
    }
    if (b < nb) {
        const uint8_t *k0 = w + b * BB;
        acc0              = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(block_dot(k0 + 2, xq + b * QK, bias + b * BIAS_LANES)),
                block_scale(k0),
                acc0);
    }
    const __m256 s  = _mm256_add_ps(acc0, acc1);
    __m128       s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4              = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4              = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    return _mm_cvtss_f32(s4);
}

static void cpu_x86_linear_pq2_0_m1(const float               *x,
                                    const struct geist_weight *w,
                                    struct geist_backend      *be,
                                    float                     *y) {
    const size_t              n_in       = (size_t) w->n_in;
    const size_t              n_out      = (size_t) w->n_out;
    const size_t              nb         = n_in / QK;
    size_t                    bias_bytes = 0;
    struct cpu_x86_workspace *ws         = nullptr;
    if (be != nullptr && be->state != nullptr &&
        !ckd_mul(&bias_bytes, nb, BIAS_LANES * sizeof(int32_t))) {
        ws = cpu_x86_ws_acquire_mN((struct cpu_x86_state *) be->state, n_in, bias_bytes, 0, 0);
    }
    if (ws == nullptr) {
        geist_linear_ref(1, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    int8_t        *xq   = ws->mN_acts;
    int32_t       *bias = ws->mN_sum_a;
    const float    inv  = quantize_acts(nb, x, xq, bias);
    const uint8_t *raw  = (const uint8_t *) w->raw;
    const size_t   rb   = nb * BB;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < n_out; j++) {
        y[j] = row_dot(nb, raw + j * rb, xq, bias) * inv;
    }
}

bool cpu_x86_linear_pq2_0_bind_m1(struct geist_weight *w) {
    if (w == nullptr || w->dtype != GEIST_DTYPE_PQ2_0 || w->n_in <= 0 ||
        (size_t) w->n_in % QK != 0) {
        return false;
    }
    w->linear_m1 = cpu_x86_linear_pq2_0_m1;
    return true;
}
