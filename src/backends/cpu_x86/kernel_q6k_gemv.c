/*
 * src/backends/cpu_x86/kernel_q6k_gemv.c — native Q6_K decode GEMV.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Decode (M=1) reading the ORIGINAL Q6_K weights (block_q6_K_t, ~0.82 B/wt)
 * instead of the W8A8 predecode (1.5 B/wt). Q6_K decode (ffn_down, lm_head)
 * is bandwidth-bound, so halving the weight traffic is the lever
 * (docs/LINUX_X86_PERF_PROFILE.md).
 *
 * The per-row dot is a faithful port of llama.cpp's AVX2
 * ggml_vec_dot_q6_K_q8_K (ggml/src/ggml-cpu/arch/x86/quants.c): unpack the
 * 6-bit weights, VPMADDUBSW against the int8 activation, apply the int8
 * sub-block scales via VPMADD, fold the uniform -32 offset through the
 * activation block-sums, and multiply by the fp32 super-block scale once.
 * Original code Copyright (c) 2023-2025 The ggml authors, MIT-licensed.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_q6k_gemv.h"

#include "quant.h" /* fp16_to_fp32 */
#include "quant_blocks.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* q8_K activation row: int8 quants + per-16 block sums + super-block scale. */
struct q8k_act {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};

/* Quantize one fp32 activation super-block group to q8_K (matches the format
 * the dot expects: d = amax/127, qs = round(x/d), bsums summed per 16). */
static void quantize_q8k_act(size_t n_super, const float *x, struct q8k_act *out) {
    for (size_t s = 0; s < n_super; s++) {
        const float *xs   = x + s * 256;
        float        amax = 0.0f;
        for (size_t k = 0; k < 256; k++) {
            const float ax = fabsf(xs[k]);
            if (ax > amax)
                amax = ax;
        }
        const float d  = amax / 127.0f;
        const float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        out[s].d       = d;
        for (size_t g = 0; g < 16; g++) {
            int32_t sum = 0;
            for (size_t i = 0; i < 16; i++) {
                const int v           = (int) lrintf(xs[g * 16 + i] * id);
                out[s].qs[g * 16 + i] = (int8_t) v;
                sum += v;
            }
            out[s].bsums[g] = (int16_t) sum;
        }
    }
}

static inline __m128i scale_shuffle(int i) {
    static const uint8_t k[128] = {
            0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,
            2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
            5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,
            8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10,
            11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
            13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    return _mm_loadu_si128((const __m128i *) k + i);
}

static inline float hsum_ps_avx(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128       s  = _mm_add_ps(lo, hi);
    s               = _mm_hadd_ps(s, s);
    s               = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* One output row's dot: sum over n_super Q6_K super-blocks. */
static float dot_q6k_q8k(size_t n_super, const struct block_q6_K_t *x, const struct q8k_act *y) {
    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256        acc = _mm256_setzero_ps();

    for (size_t i = 0; i < n_super; i++) {
        const float    d  = y[i].d * fp16_to_fp32(x[i].d);
        const uint8_t *q4 = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *q8 = y[i].qs;

        const __m256i q8sums   = _mm256_loadu_si256((const __m256i *) y[i].bsums);
        const __m128i scales   = _mm_loadu_si128((const __m128i *) x[i].scales);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), 5);

        __m256i sumi = _mm256_setzero_si256();
        int     is   = 0;
        for (int j = 0; j < 2; j++) { /* QK_K/128 = 2 */
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *) q4);
            q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *) q4);
            q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *) qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 =
                    _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 =
                    _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8((char) -64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *) q8);
            q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8);
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) q8);
            q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *) q8);
            q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i s0 = _mm_shuffle_epi8(scales, scale_shuffle(is + 0));
            const __m128i s1 = _mm_shuffle_epi8(scales, scale_shuffle(is + 1));
            const __m128i s2 = _mm_shuffle_epi8(scales, scale_shuffle(is + 2));
            const __m128i s3 = _mm_shuffle_epi8(scales, scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }
        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc  = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_ps_avx(acc);
}

void q6k_gemv_m1(size_t N, size_t K, const float *x, const uint8_t *q6k_raw, float y[static N]) {
    const size_t n_super = K / 256;
    if (n_super == 0 || n_super > 64) {
        return; /* caller falls back. */
    }
    struct q8k_act a[64];
    quantize_q8k_act(n_super, x, a);

    const size_t row_bytes = n_super * sizeof(struct block_q6_K_t);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < N; r++) {
        const struct block_q6_K_t *xr = (const struct block_q6_K_t *) (q6k_raw + r * row_bytes);
        y[r]                          = dot_q6k_q8k(n_super, xr, a);
    }
}
