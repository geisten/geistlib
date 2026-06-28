/*
 * src/backends/cpu_x86/kernel_q4kx8_gemm_avx512.c — AVX2 Q4_Kx8 GEMV.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * AVX2 lane-parallel inner kernel ported from llama.cpp's
 * ggml_gemv_q4_K_8x8_q8_K (ggml/src/ggml-cpu/arch/x86/repack.cpp:1464).
 * Produces 8 output cells per output tile via lane-parallel VPMADDUBSW
 * + VPMADD_EPI16 + scalemask byte-shuffle — the 8× cells-per-instruction
 * win identified by the perf profile (docs/LINUX_X86_PERF_PROFILE.md).
 *
 * Original code Copyright (c) 2023-2025 The ggml authors, MIT-licensed.
 * Adapted to geist's struct conventions + wrapped in OMP m-parallel.
 *
 * Key bug from the first port attempt and fix: scales_0 / scales_1 in
 * the original are __m128i then shuffle_epi8(scalemask) then
 * cvtepu8_epi16. The scalemask = {7,7,3,3,6,6,2,2,5,5,1,1,4,4,0,0}
 * duplicates each scale byte so the madd_epi16(iacc_0, scales_0) gets
 * the correct per-cell scale pair-multiply (both int16 lanes of one
 * int32 result come from the same cell's two int16 partial sums and
 * are both multiplied by THE SAME scale value). My initial port used
 * MM256_SET_M128I (duplicate the whole low-128) which scrambled the
 * scale-to-cell alignment. The mins_01 construction also relies on
 * mins_and_scales_* being __m128i (16 bytes total = [s0..s7, m0..m7]).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_q4kx8_gemm.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* ---- Q8_K per-row format ---- */

struct q8k_row {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};

__attribute__((unused)) static void quantize_q8k_row(size_t         n_super,
                                                     const float    x[static n_super * 256],
                                                     struct q8k_row out[static n_super]) {
    float amax = 0.0f;
    for (size_t s = 0; s < n_super; s++) {
        for (size_t k = 0; k < 256; k++) {
            const float a  = x[s * 256 + k];
            const float ax = a < 0.0f ? -a : a;
            if (ax > amax) {
                amax = ax;
            }
        }
    }
    const float scale = (amax == 0.0f) ? 1.0f : (amax / 127.0f);
    const float inv   = 1.0f / scale;
    for (size_t s = 0; s < n_super; s++) {
        out[s].d = scale;
        for (size_t k = 0; k < 256; k++) {
            const float v = x[s * 256 + k] * inv;
            const int   q = (int) (v < 0.0f ? v - 0.5f : v + 0.5f);
            out[s].qs[k]  = (int8_t) (q < -128 ? -128 : (q > 127 ? 127 : q));
        }
        for (int g = 0; g < 16; g++) {
            int32_t s_acc = 0;
            for (int kk = 0; kk < 16; kk++) {
                s_acc += out[s].qs[g * 16 + kk];
            }
            out[s].bsums[g] = (int16_t) s_acc;
        }
    }
}

/* ---- AVX2 GEMV: one m-row × 8-cell tile ---- */

__attribute__((target("avx2,avx,f16c,fma"))) static __m256
q4kx8_gemv_one_row_tile(size_t                     n_super,
                        const struct block_q4_Kx8 *W,
                        const struct q8k_row      *X,
                        __m256                    *acc_min_rows_out) {
    static const uint32_t kmask1 = 0x3f3f3f3fu;
    static const uint32_t kmask2 = 0x0f0f0f0fu;
    static const uint32_t kmask3 = 0x03030303u;

    const __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    const __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    const __m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    const __m256i m4b       = _mm256_set1_epi8(0x0F);

    __m256 acc_row      = _mm256_setzero_ps();
    __m256 acc_min_rows = _mm256_setzero_ps();

    for (size_t b = 0; b < n_super; b++) {
        const struct block_q4_Kx8 *bp = &W[b];
        const struct q8k_row      *ap = &X[b];

        const __m256 row_scale_f32 = _mm256_set1_ps(ap->d);

        /* Aligned stack copies because the source struct is packed. */
        uint16_t d_buf[8], dmin_buf[8];
        memcpy(d_buf, bp->d, sizeof(d_buf));
        memcpy(dmin_buf, bp->dmin, sizeof(dmin_buf));
        const __m128i d_raw         = _mm_loadu_si128((const __m128i *) d_buf);
        const __m128i dmin_raw      = _mm_loadu_si128((const __m128i *) dmin_buf);
        const __m256  col_scale_f32 = _mm256_cvtph_ps(_mm_shuffle_epi8(d_raw, deltamask));
        const __m256  col_dmin_f32  = _mm256_cvtph_ps(dmin_raw);

        __m256i iacc_b     = _mm256_setzero_si256();
        __m256i iacc_min_b = _mm256_setzero_si256();

        const __m256i q8sums = _mm256_loadu_si256((const __m256i *) ap->bsums);
        __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(q8sums),
                                                            _mm256_extracti128_si256(q8sums, 1)));
        q8s         = _mm256_permute2f128_si256(q8s, q8s, 0);

        for (int sb = 0; sb < 4; sb++) {
            const uint8_t *qbase   = bp->qs + sb * 256;
            const __m256i  r0123_0 = _mm256_loadu_si256((const __m256i *) (qbase + 0));
            const __m256i  r4567_0 = _mm256_loadu_si256((const __m256i *) (qbase + 32));
            const __m256i  r0123_1 = _mm256_loadu_si256((const __m256i *) (qbase + 64));
            const __m256i  r4567_1 = _mm256_loadu_si256((const __m256i *) (qbase + 96));
            const __m256i  r0123_2 = _mm256_loadu_si256((const __m256i *) (qbase + 128));
            const __m256i  r4567_2 = _mm256_loadu_si256((const __m256i *) (qbase + 160));
            const __m256i  r0123_3 = _mm256_loadu_si256((const __m256i *) (qbase + 192));
            const __m256i  r4567_3 = _mm256_loadu_si256((const __m256i *) (qbase + 224));

            const __m256i r0123_00 = _mm256_and_si256(r0123_0, m4b);
            const __m256i r4567_00 = _mm256_and_si256(r4567_0, m4b);
            const __m256i r0123_01 = _mm256_and_si256(r0123_1, m4b);
            const __m256i r4567_01 = _mm256_and_si256(r4567_1, m4b);
            const __m256i r0123_02 = _mm256_and_si256(r0123_2, m4b);
            const __m256i r4567_02 = _mm256_and_si256(r4567_2, m4b);
            const __m256i r0123_03 = _mm256_and_si256(r0123_3, m4b);
            const __m256i r4567_03 = _mm256_and_si256(r4567_3, m4b);

            const __m256i r0123_10 = _mm256_and_si256(_mm256_srli_epi16(r0123_0, 4), m4b);
            const __m256i r4567_10 = _mm256_and_si256(_mm256_srli_epi16(r4567_0, 4), m4b);
            const __m256i r0123_11 = _mm256_and_si256(_mm256_srli_epi16(r0123_1, 4), m4b);
            const __m256i r4567_11 = _mm256_and_si256(_mm256_srli_epi16(r4567_1, 4), m4b);
            const __m256i r0123_12 = _mm256_and_si256(_mm256_srli_epi16(r0123_2, 4), m4b);
            const __m256i r4567_12 = _mm256_and_si256(_mm256_srli_epi16(r4567_2, 4), m4b);
            const __m256i r0123_13 = _mm256_and_si256(_mm256_srli_epi16(r0123_3, 4), m4b);
            const __m256i r4567_13 = _mm256_and_si256(_mm256_srli_epi16(r4567_3, 4), m4b);

            /* Decode scales descriptor. */
            uint32_t utmp_0[4], utmp_1[4];
            memcpy(utmp_0, bp->scales + 24 * sb, 12);
            utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
            const uint32_t uaux_0 = utmp_0[1] & kmask1;
            utmp_0[1]             = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
            utmp_0[2]             = uaux_0;
            utmp_0[0] &= kmask1;

            memcpy(utmp_1, bp->scales + 12 + sb * 24, 12);
            utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
            const uint32_t uaux_1 = utmp_1[1] & kmask1;
            utmp_1[1]             = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
            utmp_1[2]             = uaux_1;
            utmp_1[0] &= kmask1;

            /* mins_and_scales as __m128i — 16 bytes: [s0..s7, m0..m7]. */
            const __m128i mins_and_scales_0 =
                    _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
            const __m128i mins_and_scales_1 =
                    _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);

            /* scales_0 / scales_1: scalemask-rearranged so per-cell scale is
             * duplicated to match iacc_0/iacc_1's [s_c, s_c] pair-pattern. */
            const __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
            const __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
            const __m256i scales_0           = _mm256_cvtepu8_epi16(scales_rearrange_0);
            const __m256i scales_1           = _mm256_cvtepu8_epi16(scales_rearrange_1);

            /* mins_01: cell c's mins for sub-blocks 2*sb and 2*sb+1 paired
             * at lanes (2c, 2c+1). */
            const __m256i mins_01 = _mm256_cvtepu8_epi16(
                    _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                                      _mm_shuffle_epi32(mins_and_scales_1, 78)));

            const int8_t *a_base = ap->qs + sb * 64;
            __m256i       lhs_00 =
                    _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) (a_base + 0)));
            __m256i lhs_01 =
                    _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) (a_base + 16)));
            __m256i lhs_10 =
                    _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) (a_base + 32)));
            __m256i lhs_11 =
                    _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) (a_base + 48)));
            lhs_00 = _mm256_permute2f128_si256(lhs_00, lhs_00, 0);
            lhs_01 = _mm256_permute2f128_si256(lhs_01, lhs_01, 0);
            lhs_10 = _mm256_permute2f128_si256(lhs_10, lhs_10, 0);
            lhs_11 = _mm256_permute2f128_si256(lhs_11, lhs_11, 0);

            /* 8 maddubs for sb_lo, 8 for sb_hi. */
            __m256i iacc_0 = _mm256_setzero_si256();
            iacc_0         = _mm256_add_epi16(
                    iacc_0,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(r0123_00, _mm256_shuffle_epi32(r4567_00, 177), 170),
                            _mm256_shuffle_epi32(lhs_00, 0)));
            iacc_0 = _mm256_add_epi16(
                    iacc_0,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(_mm256_shuffle_epi32(r0123_00, 177), r4567_00, 170),
                            _mm256_shuffle_epi32(lhs_00, 85)));
            iacc_0 = _mm256_add_epi16(
                    iacc_0,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(r0123_01, _mm256_shuffle_epi32(r4567_01, 177), 170),
                            _mm256_shuffle_epi32(lhs_00, 170)));
            iacc_0 = _mm256_add_epi16(
                    iacc_0,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(_mm256_shuffle_epi32(r0123_01, 177), r4567_01, 170),
                            _mm256_shuffle_epi32(lhs_00, 255)));
            iacc_0 = _mm256_add_epi16(
                    iacc_0,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(r0123_02, _mm256_shuffle_epi32(r4567_02, 177), 170),
                            _mm256_shuffle_epi32(lhs_01, 0)));
            iacc_0 = _mm256_add_epi16(
                    iacc_0,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(_mm256_shuffle_epi32(r0123_02, 177), r4567_02, 170),
                            _mm256_shuffle_epi32(lhs_01, 85)));
            iacc_0 = _mm256_add_epi16(
                    iacc_0,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(r0123_03, _mm256_shuffle_epi32(r4567_03, 177), 170),
                            _mm256_shuffle_epi32(lhs_01, 170)));
            iacc_0 = _mm256_add_epi16(
                    iacc_0,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(_mm256_shuffle_epi32(r0123_03, 177), r4567_03, 170),
                            _mm256_shuffle_epi32(lhs_01, 255)));
            iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

            __m256i iacc_1 = _mm256_setzero_si256();
            iacc_1         = _mm256_add_epi16(
                    iacc_1,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(r0123_10, _mm256_shuffle_epi32(r4567_10, 177), 170),
                            _mm256_shuffle_epi32(lhs_10, 0)));
            iacc_1 = _mm256_add_epi16(
                    iacc_1,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(_mm256_shuffle_epi32(r0123_10, 177), r4567_10, 170),
                            _mm256_shuffle_epi32(lhs_10, 85)));
            iacc_1 = _mm256_add_epi16(
                    iacc_1,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(r0123_11, _mm256_shuffle_epi32(r4567_11, 177), 170),
                            _mm256_shuffle_epi32(lhs_10, 170)));
            iacc_1 = _mm256_add_epi16(
                    iacc_1,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(_mm256_shuffle_epi32(r0123_11, 177), r4567_11, 170),
                            _mm256_shuffle_epi32(lhs_10, 255)));
            iacc_1 = _mm256_add_epi16(
                    iacc_1,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(r0123_12, _mm256_shuffle_epi32(r4567_12, 177), 170),
                            _mm256_shuffle_epi32(lhs_11, 0)));
            iacc_1 = _mm256_add_epi16(
                    iacc_1,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(_mm256_shuffle_epi32(r0123_12, 177), r4567_12, 170),
                            _mm256_shuffle_epi32(lhs_11, 85)));
            iacc_1 = _mm256_add_epi16(
                    iacc_1,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(r0123_13, _mm256_shuffle_epi32(r4567_13, 177), 170),
                            _mm256_shuffle_epi32(lhs_11, 170)));
            iacc_1 = _mm256_add_epi16(
                    iacc_1,
                    _mm256_maddubs_epi16(
                            _mm256_blend_epi32(_mm256_shuffle_epi32(r0123_13, 177), r4567_13, 170),
                            _mm256_shuffle_epi32(lhs_11, 255)));
            iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

            const __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

            const __m256i q8s_sb      = _mm256_shuffle_epi32(q8s, 0);
            const __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
            q8s                       = _mm256_bsrli_epi128(q8s, 4);

            iacc_b     = _mm256_add_epi32(iacc_b, iacc_sb);
            iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
        }

        acc_row = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
        acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b),
                                       _mm256_mul_ps(col_dmin_f32, row_scale_f32),
                                       acc_min_rows);
    }

    acc_row           = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
    *acc_min_rows_out = acc_min_rows;
    return acc_row;
}

/* ---- Public fallback (used by kernel_q4kx8_gemm_avx512_full.c) ---- */

void q4kx8_gemv_avx2_fallback(size_t                     M,
                              size_t                     N,
                              size_t                     K,
                              const struct block_q8_Kx4 *X,
                              const struct block_q4_Kx8 *W,
                              float                      Y[static M * N]);

void q4kx8_gemv_avx2_fallback(size_t                     M,
                              size_t                     N,
                              size_t                     K,
                              const struct block_q8_Kx4 *X,
                              const struct block_q4_Kx8 *W,
                              float                      Y[static M * N]) {
    const size_t n_super_k = K / 256;
    const size_t N_tiles   = N / 8;
    const size_t M_tiles   = M / 4;

    if (M_tiles == 0 || N == 0 || N % 8 != 0) {
        q4kx8_gemm_scalar(M, N, K, X, W, Y);
        return;
    }

    /* Extract row i from Q8_Kx4 directly into q8k_row format (no fp32
     * round-trip). The Q8_Kx4 stripe layout is: per super-block, 4 sub-
     * blocks of 256 bytes each; per sub-block, 8 stripes of 32 bytes;
     * per stripe, 4 rows of 8 int8s. To extract row i's K-element k:
     *   sb = k / 64, stripe = (k % 64) / 8, pos = k % 8
     *   byte = Q8_Kx4.qs[sb * 256 + stripe * 32 + i * 8 + pos]
     * bsums and d are stored per-row already. */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (size_t mt = 0; mt < M_tiles; mt++) {
        for (size_t i = 0; i < 4; i++) {
            const size_t m = mt * 4 + i;

            struct q8k_row a[64];
            if (n_super_k > 64)
                continue;
            for (size_t s = 0; s < n_super_k; s++) {
                const struct block_q8_Kx4 *Xb = &X[mt * n_super_k + s];
                a[s].d                        = Xb->d[i];
                for (int sb = 0; sb < 4; sb++) {
                    for (int stripe = 0; stripe < 8; stripe++) {
                        const int8_t *src = Xb->qs + sb * 256 + stripe * 32 + i * 8;
                        memcpy(a[s].qs + sb * 64 + stripe * 8, src, 8);
                    }
                }
                for (int g = 0; g < 16; g++) {
                    a[s].bsums[g] = Xb->bsums[i * 16 + g];
                }
            }

            for (size_t nt = 0; nt < N_tiles; nt++) {
                __m256       acc_min;
                const __m256 acc_row =
                        q4kx8_gemv_one_row_tile(n_super_k, &W[nt * n_super_k], a, &acc_min);
                _mm256_storeu_ps(Y + m * N + nt * 8, _mm256_sub_ps(acc_row, acc_min));
            }
        }
    }
}

/* Decode (M=1) GEMV over the compact Q4_Kx8 layout. Quantizes the single
 * fp32 activation row to q8k once, then sweeps N/8 output-cell tiles via the
 * 8-cell lane-parallel inner kernel — 8 outputs land in the 8 fp32 lanes,
 * reduced once per tile (no per-block horizontal sum, unlike the W4A8 GEMV).
 * Reads the same q4kx8 blob the prefill path uses (0.56 B/wt, vs W4A8 0.75),
 * so it is both lower-traffic and lower-compute at decode.
 * Requires N % 8 == 0 and K % 256 == 0 (every Q4_K body matrix). */
void q4kx8_gemv_m1(
        size_t N, size_t K, const float *x, const struct block_q4_Kx8 *W, float y[static N]) {
    const size_t n_super = K / 256;
    const size_t N_tiles = N / 8;
    if (n_super == 0 || n_super > 64 || N % 8 != 0) {
        return; /* caller falls back; Q4_K body shapes never hit this. */
    }

    struct q8k_row a[64];
    quantize_q8k_row(n_super, x, a);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t nt = 0; nt < N_tiles; nt++) {
        __m256       acc_min;
        const __m256 acc_row = q4kx8_gemv_one_row_tile(n_super, &W[nt * n_super], a, &acc_min);
        _mm256_storeu_ps(y + nt * 8, _mm256_sub_ps(acc_row, acc_min));
    }
}
