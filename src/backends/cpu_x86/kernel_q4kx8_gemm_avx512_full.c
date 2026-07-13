/*
 * src/backends/cpu_x86/kernel_q4kx8_gemm_avx512_full.c — AVX-512 16x16 Q4_Kx8 GEMM.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Verbatim port of llama.cpp's AVX-512BW/DQ Q4_K GEMM kernel
 * (ggml_gemm_q4_K_8x8_q8_K, ggml/src/ggml-cpu/arch/x86/repack.cpp:2042-2810,
 * pinned at commit 7c082bc417bbe53210a83df4ba5b49e18ce6193c).
 *
 * Original code Copyright (c) 2023-2025 The ggml authors, MIT-licensed.
 * Adapted to geist's struct conventions + wrapped in OMP outer parallel.
 *
 * The inner per-(sb, rp) block emits ~32 VPMADDUBSWs that produce 16 lanes
 * of 16 partial-sum int16 cells = 256 cells per pass. Combined with the
 * 16-row × 16-cell accumulator panel, one super-block walk amortises the
 * weight bytes across 256 output cells — the cells-per-VPMADDUBSW × tile-
 * amortisation product that closes the IPC gap vs llama.cpp.
 *
 * Public entry: q4kx8_gemm_avx512(). For (M, N) with M >= 16 && N >= 16
 * and both divisible by 16, the bulk runs through this kernel; otherwise
 * we fall through to the existing AVX2 GEMV path
 * (q4kx8_gemv_avx2_fallback in kernel_q4kx8_gemm_avx512.c).
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

/* AVX2 GEMV from kernel_q4kx8_gemm_avx512.c — used to handle the tail and
 * M < 16 / N < 16 cases that the 16x16 panel cannot cover. */
void q4kx8_gemv_avx2_fallback(size_t                     M,
                              size_t                     N,
                              size_t                     K,
                              const struct block_q8_Kx4 *X,
                              const struct block_q4_Kx8 *W,
                              float                      Y[static M * N]);

#define AVX512_TARGET "avx2,avx,f16c,fma,avx512f,avx512bw,avx512dq,avx512vl"

/* Load 16 fp16 values from two 8-fp16 sources into 16 fp32 in a __m512.
 * Mirrors GGML_F32Cx8x2_LOAD in llama.cpp's repack.cpp. */
__attribute__((target(AVX512_TARGET))) static inline __m512 ggml_f32cx8x2_load(const uint16_t *x,
                                                                               const uint16_t *y) {
    return _mm512_cvtph_ps(_mm256_set_m128i(_mm_loadu_si128((const __m128i *) y),
                                            _mm_loadu_si128((const __m128i *) x)));
}

/* Process one 16x16 output tile across all super-blocks. b_ptr_0/b_ptr_1
 * each cover 8 output cells × nb super-blocks; a_ptrs[0..3] each cover
 * 4 m-rows × nb super-blocks. Writes 16 rows × 16 cells fp32 starting at
 * dst (row stride N, in floats).
 *
 * Verbatim port of llama.cpp's 16x16 inner (repack.cpp:2090-2447). */
__attribute__((target(AVX512_TARGET))) static void
q4kx8_gemm16x16_tile_avx512(size_t                     nb,
                            const struct block_q4_Kx8 *b_ptr_0,
                            const struct block_q4_Kx8 *b_ptr_1,
                            const struct block_q8_Kx4 *a_ptrs[4],
                            float                     *dst,
                            size_t                     N) {
    static const uint32_t kmask1 = 0x3f3f3f3fu;
    static const uint32_t kmask2 = 0x0f0f0f0fu;
    static const uint32_t kmask3 = 0x03030303u;

    const __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    const __m512i m4bexpanded   = _mm512_set1_epi8(0x0F);

    __m512 acc_rows[16];
    __m512 acc_min_rows[16];
    for (int i = 0; i < 16; i++) {
        acc_rows[i]     = _mm512_setzero_ps();
        acc_min_rows[i] = _mm512_setzero_ps();
    }

    for (size_t b = 0; b < nb; b++) {
        /* Aligned stack buffers — block_q4_Kx8 is packed. */
        uint16_t d0_buf[8], d1_buf[8], dmin0_buf[8], dmin1_buf[8];
        memcpy(d0_buf, b_ptr_0[b].d, sizeof(d0_buf));
        memcpy(d1_buf, b_ptr_1[b].d, sizeof(d1_buf));
        memcpy(dmin0_buf, b_ptr_0[b].dmin, sizeof(dmin0_buf));
        memcpy(dmin1_buf, b_ptr_1[b].dmin, sizeof(dmin1_buf));
        const __m512 col_scale_f32 = ggml_f32cx8x2_load(d0_buf, d1_buf);
        const __m512 col_dmin_f32  = ggml_f32cx8x2_load(dmin0_buf, dmin1_buf);

        for (int sb = 0; sb < 4; sb++) { /* QK_K / 64 = 4 */
            const uint8_t *qs0 = b_ptr_0[b].qs + sb * 256;
            const uint8_t *qs1 = b_ptr_1[b].qs + sb * 256;

            const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *) (qs0 + 0));
            const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *) (qs0 + 32));
            const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *) (qs0 + 64));
            const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *) (qs0 + 96));
            const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i *) (qs0 + 128));
            const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i *) (qs0 + 160));
            const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i *) (qs0 + 192));
            const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i *) (qs0 + 224));

            const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i *) (qs1 + 0));
            const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i *) (qs1 + 32));
            const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i *) (qs1 + 64));
            const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i *) (qs1 + 96));
            const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i *) (qs1 + 128));
            const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i *) (qs1 + 160));
            const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i *) (qs1 + 192));
            const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i *) (qs1 + 224));

            const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
                    rhs_raw_mat_0123_0,
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
                    240);
            const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
                    rhs_raw_mat_4567_0,
                    240);
            const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
                    rhs_raw_mat_0123_1,
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
                    240);
            const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
                    rhs_raw_mat_4567_1,
                    240);
            const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(
                    rhs_raw_mat_0123_2,
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder),
                    240);
            const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder),
                    rhs_raw_mat_4567_2,
                    240);
            const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(
                    rhs_raw_mat_0123_3,
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder),
                    240);
            const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder),
                    rhs_raw_mat_4567_3,
                    240);

            const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(
                    rhs_raw_mat_89AB_0,
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder),
                    240);
            const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder),
                    rhs_raw_mat_CDEF_0,
                    240);
            const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(
                    rhs_raw_mat_89AB_1,
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder),
                    240);
            const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder),
                    rhs_raw_mat_CDEF_1,
                    240);
            const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(
                    rhs_raw_mat_89AB_2,
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder),
                    240);
            const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder),
                    rhs_raw_mat_CDEF_2,
                    240);
            const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(
                    rhs_raw_mat_89AB_3,
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder),
                    240);
            const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder),
                    rhs_raw_mat_CDEF_3,
                    240);

            const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(
                    _mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
            const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(
                    _mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
            const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(
                    _mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
            const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(
                    _mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);
            const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(
                    _mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
            const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(
                    _mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
            const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(
                    _mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
            const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(
                    _mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

            /* 4-bit -> 8-bit (lo nibbles into _0?, hi into _1?). */
            const __m512i rhs_mat_014589CD_00 =
                    _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded);
            const __m512i rhs_mat_2367ABEF_00 =
                    _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded);
            const __m512i rhs_mat_014589CD_01 =
                    _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded);
            const __m512i rhs_mat_2367ABEF_01 =
                    _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded);
            const __m512i rhs_mat_014589CD_02 =
                    _mm512_and_si512(rhs_raw_mat_014589CD_2, m4bexpanded);
            const __m512i rhs_mat_2367ABEF_02 =
                    _mm512_and_si512(rhs_raw_mat_2367ABEF_2, m4bexpanded);
            const __m512i rhs_mat_014589CD_03 =
                    _mm512_and_si512(rhs_raw_mat_014589CD_3, m4bexpanded);
            const __m512i rhs_mat_2367ABEF_03 =
                    _mm512_and_si512(rhs_raw_mat_2367ABEF_3, m4bexpanded);

            const __m512i rhs_mat_014589CD_10 =
                    _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded);
            const __m512i rhs_mat_2367ABEF_10 =
                    _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded);
            const __m512i rhs_mat_014589CD_11 =
                    _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded);
            const __m512i rhs_mat_2367ABEF_11 =
                    _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded);
            const __m512i rhs_mat_014589CD_12 =
                    _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m4bexpanded);
            const __m512i rhs_mat_2367ABEF_12 =
                    _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m4bexpanded);
            const __m512i rhs_mat_014589CD_13 =
                    _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m4bexpanded);
            const __m512i rhs_mat_2367ABEF_13 =
                    _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m4bexpanded);

            /* Shuffle pattern one (sp1) — 8x cell-broadcast. */
            const __m512i rhs_mat_014589CD_00_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_2367ABEF_00_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_014589CD_01_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_2367ABEF_01_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_014589CD_02_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_2367ABEF_02_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_014589CD_03_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_2367ABEF_03_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_014589CD_10_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_2367ABEF_10_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_014589CD_11_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_2367ABEF_11_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_014589CD_12_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_2367ABEF_12_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_014589CD_13_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM) 136);
            const __m512i rhs_mat_2367ABEF_13_sp1 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM) 136);

            /* Shuffle pattern two (sp2). */
            const __m512i rhs_mat_014589CD_00_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_2367ABEF_00_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_014589CD_01_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_2367ABEF_01_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_014589CD_02_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_2367ABEF_02_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_014589CD_03_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_2367ABEF_03_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_014589CD_10_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_2367ABEF_10_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_014589CD_11_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_2367ABEF_11_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_014589CD_12_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_2367ABEF_12_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_014589CD_13_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM) 221);
            const __m512i rhs_mat_2367ABEF_13_sp2 =
                    _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM) 221);

            /* Decode scales descriptor (12 bytes per sb half × 2 halves). */
            uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

            memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
            utmp_00[3] = ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
            const uint32_t uaux00 = utmp_00[1] & kmask1;
            utmp_00[1]            = (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
            utmp_00[2]            = uaux00;
            utmp_00[0] &= kmask1;

            memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
            utmp_01[3] = ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
            const uint32_t uaux01 = utmp_01[1] & kmask1;
            utmp_01[1]            = (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
            utmp_01[2]            = uaux01;
            utmp_01[0] &= kmask1;

            memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
            utmp_10[3] = ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
            const uint32_t uaux10 = utmp_10[1] & kmask1;
            utmp_10[1]            = (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
            utmp_10[2]            = uaux10;
            utmp_10[0] &= kmask1;

            memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
            utmp_11[3] = ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
            const uint32_t uaux11 = utmp_11[1] & kmask1;
            utmp_11[1]            = (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
            utmp_11[2]            = uaux11;
            utmp_11[0] &= kmask1;

            const __m256i mins_and_scales_0 = _mm256_set_epi32((int) utmp_10[3],
                                                               (int) utmp_10[2],
                                                               (int) utmp_10[1],
                                                               (int) utmp_10[0],
                                                               (int) utmp_00[3],
                                                               (int) utmp_00[2],
                                                               (int) utmp_00[1],
                                                               (int) utmp_00[0]);
            const __m512i scales_0          = _mm512_cvtepu8_epi16(
                    _mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

            const __m256i mins_and_scales_1 = _mm256_set_epi32((int) utmp_11[3],
                                                               (int) utmp_11[2],
                                                               (int) utmp_11[1],
                                                               (int) utmp_11[0],
                                                               (int) utmp_01[3],
                                                               (int) utmp_01[2],
                                                               (int) utmp_01[1],
                                                               (int) utmp_01[0]);
            const __m512i scales_1          = _mm512_cvtepu8_epi16(
                    _mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

            const __m512i mins_01 = _mm512_cvtepu8_epi16(
                    _mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78),
                                         _mm256_shuffle_epi32(mins_and_scales_1, 78)));

            const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM) 68);
            const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM) 238);
            const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM) 68);
            const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM) 238);

            for (int rp = 0; rp < 4; rp++) {
                /* Load 4-row interleaved q8 stripes for this sub-block. */
                const int8_t *aqs = a_ptrs[rp][b].qs + 256 * sb;

                __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i *) (aqs + 0));
                __m256i lhs_mat_ymm_01_00 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                __m256i lhs_mat_ymm_23_00 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i *) (aqs + 32));
                __m256i lhs_mat_ymm_01_01 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                __m256i lhs_mat_ymm_23_01 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256((const __m256i *) (aqs + 64));
                __m256i lhs_mat_ymm_01_02 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
                __m256i lhs_mat_ymm_23_02 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
                __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256((const __m256i *) (aqs + 96));
                __m256i lhs_mat_ymm_01_03 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
                __m256i lhs_mat_ymm_23_03 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
                __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i *) (aqs + 128));
                __m256i lhs_mat_ymm_01_10 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                __m256i lhs_mat_ymm_23_10 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i *) (aqs + 160));
                __m256i lhs_mat_ymm_01_11 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                __m256i lhs_mat_ymm_23_11 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256((const __m256i *) (aqs + 192));
                __m256i lhs_mat_ymm_01_12 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
                __m256i lhs_mat_ymm_23_12 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
                __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256((const __m256i *) (aqs + 224));
                __m256i lhs_mat_ymm_01_13 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
                __m256i lhs_mat_ymm_23_13 =
                        _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

                __m512i lhs_mat_01_00 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                __m512i lhs_mat_23_00 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                __m512i lhs_mat_01_01 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                __m512i lhs_mat_23_01 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
                __m512i lhs_mat_01_02 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
                __m512i lhs_mat_23_02 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
                __m512i lhs_mat_01_03 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
                __m512i lhs_mat_23_03 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

                __m512i lhs_mat_01_10 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                __m512i lhs_mat_23_10 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                __m512i lhs_mat_01_11 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                __m512i lhs_mat_23_11 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
                __m512i lhs_mat_01_12 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
                __m512i lhs_mat_23_12 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
                __m512i lhs_mat_01_13 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
                __m512i lhs_mat_23_13 = _mm512_inserti32x8(
                        _mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

                /* bsums in llama layout: at offset 16*sb we want
                 *   [row0_g(4sb..4sb+3), row1_g(...), row2_g(...), row3_g(...)]
                 * Our bsums are row-major (bsums[r*16 + g]); rearrange. */
                int16_t bsums_buf[16];
                for (int r = 0; r < 4; r++) {
                    for (int gi = 0; gi < 4; gi++) {
                        bsums_buf[r * 4 + gi] = a_ptrs[rp][b].bsums[r * 16 + sb * 4 + gi];
                    }
                }
                __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i *) bsums_buf);
                __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(
                        _mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01),
                                       _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(
                        lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
                __m512i lhs_bsums_hsum_0123_01 =
                        _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01),
                                           lhs_bsums_hsum_ymm_0123_01,
                                           1);

                /* sp1 LHS shuffles. */
                const __m512i lhs_mat_01_00_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_23_00_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_01_01_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_23_01_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_01_02_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_23_02_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_01_03_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_23_03_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_01_10_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_23_10_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_01_11_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_23_11_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_01_12_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_23_12_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_01_13_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM) 160);
                const __m512i lhs_mat_23_13_sp1 =
                        _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM) 160);

                /* sp2 LHS shuffles. */
                const __m512i lhs_mat_01_00_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_23_00_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_01_01_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_23_01_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_01_02_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_23_02_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_01_03_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_23_03_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_01_10_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_23_10_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_01_11_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_23_11_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_01_12_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_23_12_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_01_13_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM) 245);
                const __m512i lhs_mat_23_13_sp2 =
                        _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM) 245);

                /* sp1 madds. */
                __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1,
                                                                      lhs_mat_01_03_sp1),
                                                 _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1,
                                                                      lhs_mat_01_02_sp1)),
                                _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1)),
                        _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
                __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1,
                                                                      lhs_mat_01_03_sp1),
                                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1,
                                                                      lhs_mat_01_02_sp1)),
                                _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1)),
                        _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
                __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1,
                                                                      lhs_mat_23_03_sp1),
                                                 _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1,
                                                                      lhs_mat_23_02_sp1)),
                                _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1)),
                        _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
                __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1,
                                                                      lhs_mat_23_03_sp1),
                                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1,
                                                                      lhs_mat_23_02_sp1)),
                                _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1)),
                        _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
                __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1,
                                                                      lhs_mat_01_13_sp1),
                                                 _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1,
                                                                      lhs_mat_01_12_sp1)),
                                _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1)),
                        _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
                __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1,
                                                                      lhs_mat_01_13_sp1),
                                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1,
                                                                      lhs_mat_01_12_sp1)),
                                _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1)),
                        _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
                __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1,
                                                                      lhs_mat_23_13_sp1),
                                                 _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1,
                                                                      lhs_mat_23_12_sp1)),
                                _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1)),
                        _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
                __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1,
                                                                      lhs_mat_23_13_sp1),
                                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1,
                                                                      lhs_mat_23_12_sp1)),
                                _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1)),
                        _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

                /* sp2 madds. */
                __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2,
                                                                      lhs_mat_01_03_sp2),
                                                 _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2,
                                                                      lhs_mat_01_02_sp2)),
                                _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2)),
                        _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
                __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2,
                                                                      lhs_mat_01_03_sp2),
                                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2,
                                                                      lhs_mat_01_02_sp2)),
                                _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2)),
                        _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
                __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2,
                                                                      lhs_mat_23_03_sp2),
                                                 _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2,
                                                                      lhs_mat_23_02_sp2)),
                                _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2)),
                        _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
                __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2,
                                                                      lhs_mat_23_03_sp2),
                                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2,
                                                                      lhs_mat_23_02_sp2)),
                                _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2)),
                        _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
                __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2,
                                                                      lhs_mat_01_13_sp2),
                                                 _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2,
                                                                      lhs_mat_01_12_sp2)),
                                _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2)),
                        _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
                __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2,
                                                                      lhs_mat_01_13_sp2),
                                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2,
                                                                      lhs_mat_01_12_sp2)),
                                _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2)),
                        _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
                __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2,
                                                                      lhs_mat_23_13_sp2),
                                                 _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2,
                                                                      lhs_mat_23_12_sp2)),
                                _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2)),
                        _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
                __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(
                        _mm512_add_epi16(
                                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2,
                                                                      lhs_mat_23_13_sp2),
                                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2,
                                                                      lhs_mat_23_12_sp2)),
                                _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2)),
                        _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

                __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);
                __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(
                        0xCCCC,
                        iacc_mat_00_0,
                        _mm512_shuffle_epi32(iacc_mat_01_0, (_MM_PERM_ENUM) 78));
                __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(
                        0xCCCC,
                        _mm512_shuffle_epi32(iacc_mat_00_0, (_MM_PERM_ENUM) 78),
                        iacc_mat_01_0);
                __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(
                        0xCCCC,
                        iacc_mat_10_0,
                        _mm512_shuffle_epi32(iacc_mat_11_0, (_MM_PERM_ENUM) 78));
                __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(
                        0xCCCC,
                        _mm512_shuffle_epi32(iacc_mat_10_0, (_MM_PERM_ENUM) 78),
                        iacc_mat_11_0);
                __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(
                        0xCCCC,
                        iacc_mat_00_1,
                        _mm512_shuffle_epi32(iacc_mat_01_1, (_MM_PERM_ENUM) 78));
                __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(
                        0xCCCC,
                        _mm512_shuffle_epi32(iacc_mat_00_1, (_MM_PERM_ENUM) 78),
                        iacc_mat_01_1);
                __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(
                        0xCCCC,
                        iacc_mat_10_1,
                        _mm512_shuffle_epi32(iacc_mat_11_1, (_MM_PERM_ENUM) 78));
                __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(
                        0xCCCC,
                        _mm512_shuffle_epi32(iacc_mat_10_1, (_MM_PERM_ENUM) 78),
                        iacc_mat_11_1);

                __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
                __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
                __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
                __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

                /* row_scale broadcast from 4 fp32 d-values for the 4-rank block. */
                float row_d_buf[4];
                memcpy(row_d_buf, a_ptrs[rp][b].d, sizeof(row_d_buf));
                const __m128 row_scale_f32_sse = _mm_loadu_ps(row_d_buf);
                const __m256 row_scale_f32_ymm =
                        _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                const __m512 row_scale_f32 = _mm512_insertf32x8(
                        _mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                acc_rows[rp * 4 + 0] = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(iacc_row_0),
                        _mm512_mul_ps(col_scale_f32,
                                      _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
                        acc_rows[rp * 4 + 0]);
                acc_rows[rp * 4 + 1] = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(iacc_row_1),
                        _mm512_mul_ps(col_scale_f32,
                                      _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
                        acc_rows[rp * 4 + 1]);
                acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(iacc_row_2),
                        _mm512_mul_ps(col_scale_f32,
                                      _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
                        acc_rows[rp * 4 + 2]);
                acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(iacc_row_3),
                        _mm512_mul_ps(col_scale_f32,
                                      _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
                        acc_rows[rp * 4 + 3]);

                __m512i iacc_row_min_0 = _mm512_madd_epi16(
                        _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM) 0), mins_01);
                __m512i iacc_row_min_1 = _mm512_madd_epi16(
                        _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM) 85), mins_01);
                __m512i iacc_row_min_2 = _mm512_madd_epi16(
                        _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM) 170), mins_01);
                __m512i iacc_row_min_3 = _mm512_madd_epi16(
                        _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM) 255), mins_01);

                acc_min_rows[rp * 4 + 0] = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(iacc_row_min_0),
                        _mm512_mul_ps(col_dmin_f32,
                                      _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
                        acc_min_rows[rp * 4 + 0]);
                acc_min_rows[rp * 4 + 1] = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(iacc_row_min_1),
                        _mm512_mul_ps(col_dmin_f32,
                                      _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
                        acc_min_rows[rp * 4 + 1]);
                acc_min_rows[rp * 4 + 2] = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(iacc_row_min_2),
                        _mm512_mul_ps(col_dmin_f32,
                                      _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
                        acc_min_rows[rp * 4 + 2]);
                acc_min_rows[rp * 4 + 3] = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(iacc_row_min_3),
                        _mm512_mul_ps(col_dmin_f32,
                                      _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
                        acc_min_rows[rp * 4 + 3]);
            }
        }
    }

    for (int i = 0; i < 16; i++) {
        _mm512_storeu_ps(dst + i * N, _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
    }
}

/* ---- Public entry ----
 *
 * Dispatch: handle the M=16-aligned, N=16-aligned bulk through the AVX-512
 * 16x16 panel. Remaining tail rows (M % 16) and tail cells (N % 16) go to
 * the AVX2 GEMV path which is correct for any M (multiple of 4) and any N
 * (multiple of 8).
 *
 * In Gemma 4: n_out is always a multiple of 256, so the N tail never fires
 * for body matrices. For Gemma 4 prefill at seq_len=128/256/512, m is also
 * a multiple of 16 (it equals the chunk size). The tail handler is mainly
 * defensive for smaller batches and the output projection.
 */
void q4kx8_gemm_avx512(size_t                     M,
                       size_t                     N,
                       size_t                     K,
                       const struct block_q8_Kx4 *X,
                       const struct block_q4_Kx8 *W,
                       float                      Y[static M * N]) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    /* The 16x16 tile path below is AVX-512 (target(avx512f,bw,dq,vl)) and has no
     * runtime guard of its own. The engine's kernel catalog is meant to gate it
     * by the CPU probe, but the prefill path (M>=16) reaches here directly — so
     * guard here too: on an x86-64-v3 CPU without AVX-512 (a supported release
     * target) run the AVX2 GEMV for the whole GEMM instead of a SIGILL.
     * __builtin_cpu_supports reads a once-initialised global — negligible cost. */
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw") ||
        !__builtin_cpu_supports("avx512dq") || !__builtin_cpu_supports("avx512vl")) {
        q4kx8_gemv_avx2_fallback(M, N, K, X, W, Y);
        return;
    }
#endif
    const size_t n_super_k = K / 256;

    if (M < 16 || N < 16 || (M % 16) != 0 || (N % 16) != 0) {
        /* No 16x16 panel — let the AVX2 GEMV handle everything. */
        q4kx8_gemv_avx2_fallback(M, N, K, X, W, Y);
        return;
    }

    const size_t M16 = M / 16;
    const size_t N16 = N / 16;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (size_t y16 = 0; y16 < M16; y16++) {
        for (size_t x16 = 0; x16 < N16; x16++) {
            /* 16 m-rows = 4 Q8_Kx4 blocks; 16 n-cells = 2 Q4_Kx8 blocks. */
            const struct block_q8_Kx4 *a_ptrs[4];
            for (int rp = 0; rp < 4; rp++) {
                a_ptrs[rp] = X + (y16 * 4 + rp) * n_super_k;
            }
            const struct block_q4_Kx8 *b_ptr_0 = W + (x16 * 2 + 0) * n_super_k;
            const struct block_q4_Kx8 *b_ptr_1 = W + (x16 * 2 + 1) * n_super_k;
            float                     *dst     = Y + (y16 * 16) * N + (x16 * 16);
            q4kx8_gemm16x16_tile_avx512(n_super_k, b_ptr_0, b_ptr_1, a_ptrs, dst, N);
        }
    }
}
