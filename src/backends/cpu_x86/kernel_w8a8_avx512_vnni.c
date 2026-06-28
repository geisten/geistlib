/*
 * src/backends/cpu_x86/kernel_w8a8_avx512_vnni.c — AVX-512+VNNI W8A8 dot.
 *
 * Compiled with -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni;
 * the dispatcher only calls this when cpuid confirms VNNI support.
 *
 * Two 16-element blocks per 256-bit VPDPBUSD (32 bytes weights + 32
 * acts). 8 int32 lanes; lanes 0-3 = block 0, lanes 4-7 = block 1.
 * Q6_K's 16-element sub-block granularity guarantees n_blocks is even
 * (16 sub-blocks per super-block × n_super); a 16-byte single-block
 * tail handles other callers.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_w8a8.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

[[nodiscard]] float w8a8_dot_avx512_vnni(
        size_t        n_blocks,
        const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
        const float   w_scales[static n_blocks],
        const float   w_offsets[static n_blocks],
        const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_blocks],
        float         scale_x) {
    float  acc = 0.0f;
    size_t b   = 0;

    /* 2 blocks (32 elements) per VPDPBUSD. */
    for (; b + 2 <= n_blocks; b += 2) {
        const __m256i u_w = _mm256_loadu_si256(
                (const __m256i *) (weights + b * W8A8_BLOCK_ELEMS));
        const __m256i s_a = _mm256_loadu_si256(
                (const __m256i *) (acts + b * W8A8_BLOCK_ELEMS));
        const __m256i dot8 =
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), u_w, s_a);

        /* lanes 0-3: block 0; lanes 4-7: block 1. Reduce each half. */
        const __m128i d_lo = _mm256_castsi256_si128(dot8);
        const __m128i d_hi = _mm256_extracti128_si256(dot8, 1);
        __m128i       sum0 = _mm_hadd_epi32(d_lo, d_lo);
        sum0               = _mm_hadd_epi32(sum0, sum0);
        __m128i       sum1 = _mm_hadd_epi32(d_hi, d_hi);
        sum1               = _mm_hadd_epi32(sum1, sum1);
        const int32_t d_b0 = _mm_cvtsi128_si32(sum0);
        const int32_t d_b1 = _mm_cvtsi128_si32(sum1);

        acc += w_scales[b]     * (float) d_b0 -
               w_offsets[b]    * (float) sum_a_per_block[b];
        acc += w_scales[b + 1] * (float) d_b1 -
               w_offsets[b + 1] * (float) sum_a_per_block[b + 1];
    }

    /* Tail: one 16-elem block. */
    if (b < n_blocks) {
        const __m128i u_w128 =
                _mm_loadu_si128((const __m128i *) (weights + b * W8A8_BLOCK_ELEMS));
        const __m128i s_a128 =
                _mm_loadu_si128((const __m128i *) (acts + b * W8A8_BLOCK_ELEMS));
        const __m128i dot4 =
                _mm_dpbusd_epi32(_mm_setzero_si128(), u_w128, s_a128);
        __m128i       s = _mm_hadd_epi32(dot4, dot4);
        s               = _mm_hadd_epi32(s, s);
        const int32_t d_b = _mm_cvtsi128_si32(s);
        acc += w_scales[b] * (float) d_b -
               w_offsets[b] * (float) sum_a_per_block[b];
    }

    return scale_x * acc;
}

/* Reduce a 256-bit dpbusd result into its two per-16-block partial sums:
 * lanes 0-3 → block b, lanes 4-7 → block b+1. */
static inline void reduce_dot8(__m256i dot8, int32_t *d0, int32_t *d1) {
    const __m128i lo  = _mm256_castsi256_si128(dot8);
    const __m128i hi  = _mm256_extracti128_si256(dot8, 1);
    __m128i       s0  = _mm_hadd_epi32(lo, lo);
    s0                = _mm_hadd_epi32(s0, s0);
    __m128i       s1  = _mm_hadd_epi32(hi, hi);
    s1                = _mm_hadd_epi32(s1, s1);
    *d0 = _mm_cvtsi128_si32(s0);
    *d1 = _mm_cvtsi128_si32(s1);
}

/* JT = number of tokens whose accumulators are kept live in the inner loop.
 * 4 independent fp32 chains per weight 2-block load — enough ILP to hide
 * the fp-add latency on Zen 5 while the weight stays resident in L1.
 * ponytail: JT=4 + per-block hadd reductions; the reductions are the
 * ceiling. If this still trails llama.cpp, the upgrade is a lane-parallel
 * W8A8 weight repack (à la block_q4_Kx8) that produces 8 cells per
 * VPDPBUSD and defers reduction — see docs/LINUX_X86_PERF_PROFILE.md. */
#define W8A8_JT 4

void w8a8_gemm_avx512_vnni(
        size_t        n_tokens,
        size_t        n_rows,
        size_t        n_blocks,
        const uint8_t weights[static n_rows * n_blocks * W8A8_BLOCK_ELEMS],
        const float   w_scales[static n_rows * n_blocks],
        const float   w_offsets[static n_rows * n_blocks],
        const int8_t  acts[static n_tokens * n_blocks * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_tokens * n_blocks],
        const float   scale_x[static n_tokens],
        float         out[static n_tokens * n_rows]) {
    const size_t row_bytes = n_blocks * W8A8_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_rows; r++) {
        const uint8_t *w_row = weights + r * row_bytes;
        const float   *ws    = w_scales + r * n_blocks;
        const float   *wo    = w_offsets + r * n_blocks;

        for (size_t j0 = 0; j0 < n_tokens; j0 += W8A8_JT) {
            const size_t jt = (n_tokens - j0 < W8A8_JT) ? (n_tokens - j0) : W8A8_JT;
            float        acc[W8A8_JT] = {0.0f};

            size_t b = 0;
            for (; b + 2 <= n_blocks; b += 2) {
                /* Weight 2-block: loaded once, reused across the JT tokens. */
                const __m256i w2 = _mm256_loadu_si256(
                        (const __m256i *) (w_row + b * W8A8_BLOCK_ELEMS));
                const float ws0 = ws[b], ws1 = ws[b + 1];
                const float wo0 = wo[b], wo1 = wo[b + 1];
                for (size_t jj = 0; jj < jt; jj++) {
                    const int8_t  *a_row = acts + (j0 + jj) * row_bytes;
                    const int32_t *sa    = sum_a_per_block + (j0 + jj) * n_blocks;
                    const __m256i  a2    = _mm256_loadu_si256(
                            (const __m256i *) (a_row + b * W8A8_BLOCK_ELEMS));
                    const __m256i dot8 =
                            _mm256_dpbusd_epi32(_mm256_setzero_si256(), w2, a2);
                    int32_t d0, d1;
                    reduce_dot8(dot8, &d0, &d1);
                    acc[jj] += ws0 * (float) d0 - wo0 * (float) sa[b];
                    acc[jj] += ws1 * (float) d1 - wo1 * (float) sa[b + 1];
                }
            }
            /* Odd-block tail (Q6_K never hits this — n_blocks is always even). */
            if (b < n_blocks) {
                const __m128i w1 = _mm_loadu_si128(
                        (const __m128i *) (w_row + b * W8A8_BLOCK_ELEMS));
                const float ws0 = ws[b], wo0 = wo[b];
                for (size_t jj = 0; jj < jt; jj++) {
                    const int8_t  *a_row = acts + (j0 + jj) * row_bytes;
                    const int32_t *sa    = sum_a_per_block + (j0 + jj) * n_blocks;
                    const __m128i  a1    = _mm_loadu_si128(
                            (const __m128i *) (a_row + b * W8A8_BLOCK_ELEMS));
                    const __m128i dot4 =
                            _mm_dpbusd_epi32(_mm_setzero_si128(), w1, a1);
                    __m128i s = _mm_hadd_epi32(dot4, dot4);
                    s         = _mm_hadd_epi32(s, s);
                    acc[jj] += ws0 * (float) _mm_cvtsi128_si32(s) -
                               wo0 * (float) sa[b];
                }
            }

            for (size_t jj = 0; jj < jt; jj++) {
                out[(j0 + jj) * n_rows + r] = scale_x[j0 + jj] * acc[jj];
            }
        }
    }
}

/* Lane-parallel W8x8 GEMM. One VPDPBUSD lands W8X8_NROWS=8 output rows in
 * the 8 int32 lanes (no per-row hadd); the per-block scale/offset apply as
 * one 8-wide fp32 FMA for all 8 rows. JT tokens kept live per group so the
 * 8-row weight stripe is reused across the token tile. See kernel_w8a8.h
 * for the interleaved layout. */
#define W8X8_JT 4

void w8x8_gemm(
        size_t        n_tokens,
        size_t        n_rows,
        size_t        n_blocks,
        const uint8_t qs[static n_rows * n_blocks * W8A8_BLOCK_ELEMS],
        const float   scales[static n_rows * n_blocks],
        const float   offsets[static n_rows * n_blocks],
        const int8_t  acts[static n_tokens * n_blocks * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_tokens * n_blocks],
        const float   scale_x[static n_tokens],
        float         out[static n_tokens * n_rows]) {
    const size_t n_in        = n_blocks * W8A8_BLOCK_ELEMS;
    const size_t qs_per_grp  = n_in * W8X8_NROWS;     /* bytes per 8-row group */
    const size_t sc_per_grp  = n_blocks * W8X8_NROWS; /* floats per group */
    const size_t NG          = n_rows / W8X8_NROWS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t g = 0; g < NG; g++) {
        const uint8_t *qs_g = qs + g * qs_per_grp;
        const float   *sc_g = scales + g * sc_per_grp;
        const float   *of_g = offsets + g * sc_per_grp;

        for (size_t j0 = 0; j0 < n_tokens; j0 += W8X8_JT) {
            const size_t jt = (n_tokens - j0 < W8X8_JT) ? (n_tokens - j0) : W8X8_JT;
            __m256       acc[W8X8_JT];
            for (size_t jj = 0; jj < jt; jj++) acc[jj] = _mm256_setzero_ps();

            for (size_t b = 0; b < n_blocks; b++) {
                /* 8-row weight stripe for block b: 4 stripes × 32 bytes. */
                const uint8_t *wb = qs_g + b * (W8A8_BLOCK_ELEMS * W8X8_NROWS);
                const __m256   sc8 = _mm256_loadu_ps(sc_g + b * W8X8_NROWS);
                const __m256   of8 = _mm256_loadu_ps(of_g + b * W8X8_NROWS);
                const __m256i  w0  = _mm256_loadu_si256((const __m256i *) (wb + 0));
                const __m256i  w1  = _mm256_loadu_si256((const __m256i *) (wb + 32));
                const __m256i  w2  = _mm256_loadu_si256((const __m256i *) (wb + 64));
                const __m256i  w3  = _mm256_loadu_si256((const __m256i *) (wb + 96));

                for (size_t jj = 0; jj < jt; jj++) {
                    const int8_t *a = acts + (j0 + jj) * n_in + b * W8A8_BLOCK_ELEMS;
                    /* Broadcast each 4-element act stripe to all 8 lanes. */
                    int32_t a4[4];
                    __builtin_memcpy(a4, a, 16);
                    __m256i iacc = _mm256_dpbusd_epi32(
                            _mm256_setzero_si256(), w0, _mm256_set1_epi32(a4[0]));
                    iacc = _mm256_dpbusd_epi32(iacc, w1, _mm256_set1_epi32(a4[1]));
                    iacc = _mm256_dpbusd_epi32(iacc, w2, _mm256_set1_epi32(a4[2]));
                    iacc = _mm256_dpbusd_epi32(iacc, w3, _mm256_set1_epi32(a4[3]));
                    /* lane r = row r's int dot for block b. */
                    const __m256 dotf = _mm256_cvtepi32_ps(iacc);
                    const __m256 sa_b = _mm256_set1_ps(
                            (float) sum_a_per_block[(j0 + jj) * n_blocks + b]);
                    acc[jj] = _mm256_fmadd_ps(sc8, dotf, acc[jj]);
                    acc[jj] = _mm256_fnmadd_ps(of8, sa_b, acc[jj]);
                }
            }

            for (size_t jj = 0; jj < jt; jj++) {
                const __m256 y = _mm256_mul_ps(acc[jj], _mm256_set1_ps(scale_x[j0 + jj]));
                _mm256_storeu_ps(out + (j0 + jj) * n_rows + g * W8X8_NROWS, y);
            }
        }
    }
}

/* 512-bit / 16-row variant of w8x8_gemm: one VPDPBUSD lands 16 output rows
 * in the 16 int32 lanes. ~1.5× w8x8 on Zen 5's full-width AVX-512 datapath. */
#define W8X16_JT 4

void w8x16_gemm(
        size_t        n_tokens,
        size_t        n_rows,
        size_t        n_blocks,
        const uint8_t qs[static n_rows * n_blocks * W8A8_BLOCK_ELEMS],
        const float   scales[static n_rows * n_blocks],
        const float   offsets[static n_rows * n_blocks],
        const int8_t  acts[static n_tokens * n_blocks * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_tokens * n_blocks],
        const float   scale_x[static n_tokens],
        float         out[static n_tokens * n_rows]) {
    const size_t n_in       = n_blocks * W8A8_BLOCK_ELEMS;
    const size_t qs_per_grp = n_in * W8X16_NROWS;     /* bytes per 16-row group */
    const size_t sc_per_grp = n_blocks * W8X16_NROWS; /* floats per group */
    const size_t NG         = n_rows / W8X16_NROWS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t g = 0; g < NG; g++) {
        const uint8_t *qs_g = qs + g * qs_per_grp;
        const float   *sc_g = scales + g * sc_per_grp;
        const float   *of_g = offsets + g * sc_per_grp;

        for (size_t j0 = 0; j0 < n_tokens; j0 += W8X16_JT) {
            const size_t jt = (n_tokens - j0 < W8X16_JT) ? (n_tokens - j0) : W8X16_JT;
            __m512       acc[W8X16_JT];
            for (size_t jj = 0; jj < jt; jj++) acc[jj] = _mm512_setzero_ps();

            for (size_t b = 0; b < n_blocks; b++) {
                /* 16-row weight stripes for block b: 4 stripes × 64 bytes. */
                const uint8_t *wb  = qs_g + b * (W8A8_BLOCK_ELEMS * W8X16_NROWS);
                const __m512   sc16 = _mm512_loadu_ps(sc_g + b * W8X16_NROWS);
                const __m512   of16 = _mm512_loadu_ps(of_g + b * W8X16_NROWS);
                const __m512i  w0  = _mm512_loadu_si512((const void *) (wb + 0));
                const __m512i  w1  = _mm512_loadu_si512((const void *) (wb + 64));
                const __m512i  w2  = _mm512_loadu_si512((const void *) (wb + 128));
                const __m512i  w3  = _mm512_loadu_si512((const void *) (wb + 192));

                for (size_t jj = 0; jj < jt; jj++) {
                    const int8_t *a = acts + (j0 + jj) * n_in + b * W8A8_BLOCK_ELEMS;
                    int32_t a4[4];
                    __builtin_memcpy(a4, a, 16);
                    __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w0,
                                                    _mm512_set1_epi32(a4[0]));
                    d = _mm512_dpbusd_epi32(d, w1, _mm512_set1_epi32(a4[1]));
                    d = _mm512_dpbusd_epi32(d, w2, _mm512_set1_epi32(a4[2]));
                    d = _mm512_dpbusd_epi32(d, w3, _mm512_set1_epi32(a4[3]));
                    const __m512 df   = _mm512_cvtepi32_ps(d);
                    const __m512 sa_b = _mm512_set1_ps(
                            (float) sum_a_per_block[(j0 + jj) * n_blocks + b]);
                    acc[jj] = _mm512_fmadd_ps(sc16, df, acc[jj]);
                    acc[jj] = _mm512_fnmadd_ps(of16, sa_b, acc[jj]);
                }
            }

            for (size_t jj = 0; jj < jt; jj++) {
                const __m512 y = _mm512_mul_ps(acc[jj], _mm512_set1_ps(scale_x[j0 + jj]));
                _mm512_storeu_ps(out + (j0 + jj) * n_rows + g * W8X16_NROWS, y);
            }
        }
    }
}

