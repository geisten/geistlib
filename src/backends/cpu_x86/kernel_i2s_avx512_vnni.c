/*
 * src/backends/cpu_x86/kernel_i2s_avx512_vnni.c — I2_S ternary decode GEMV.
 *
 * Layer: BACKEND (cpu_x86). Compiled with -mavx512vnni (see backend mk).
 *
 * Biased-u8 ternary dot via VPDPBUSD (u8×s8). The 2-bit codes {0,1,2} are
 * unpacked in-register straight from the packed 0.25 B/wt weight stream
 * (no f32 predecode — decode is bandwidth-bound on the weight read), and
 * the −1 trit offset is folded out once per row by subtracting the
 * per-token activation sum. See kernel_i2s.h for the algebra.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_i2s.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Permute one token's int8 activations into the VPDPBUSD pairing order:
 * for block b, group g (the 2-bit field at shift 6-2g), the 64-lane vector
 * is [ xq[b*256 + g*32 .. +32) , xq[b*256 + 128 + g*32 .. +32) ] — the two
 * 32-byte halves the codes from qs[0..31] (h=0) and qs[32..63] (h=1) land
 * in. Built once per call, shared read-only across all output rows. */
static void build_acts_perm(size_t n_blocks, const int8_t *xq, int8_t *perm) {
    for (size_t b = 0; b < n_blocks; b++) {
        const int8_t *xb  = xq + b * 256;
        int8_t       *dst = perm + b * 256;
        for (size_t g = 0; g < 4; g++) {
            __builtin_memcpy(dst + g * 64, xb + g * 32, 32);
            __builtin_memcpy(dst + g * 64 + 32, xb + 128 + g * 32, 32);
        }
    }
}

void i2s_gemv_m1_avx512_vnni(size_t        n_out,
                             size_t        n_in,
                             const int8_t *xq,
                             int32_t       sum_a,
                             const uint8_t w_raw[],
                             float         scale,
                             float         y[static n_out]) {
    const size_t n_blocks  = n_in / I2S_BLOCK_ELEMS;
    const size_t row_bytes = n_in / 4;

    int8_t *perm = (int8_t *) __builtin_alloca(n_in);
    build_acts_perm(n_blocks, xq, perm);

    const __m512i m3 = _mm512_set1_epi8(3);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_out; r++) {
        const uint8_t *Wr  = w_raw + r * row_bytes;
        __m512i        acc = _mm512_setzero_si512();
        for (size_t b = 0; b < n_blocks; b++) {
            const __m512i w  = _mm512_loadu_si512((const void *) (Wr + b * I2S_BLOCK_BYTES));
            const int8_t *ab = perm + b * 256;

            const __m512i c0 = _mm512_and_si512(_mm512_srli_epi16(w, 6), m3);
            const __m512i c1 = _mm512_and_si512(_mm512_srli_epi16(w, 4), m3);
            const __m512i c2 = _mm512_and_si512(_mm512_srli_epi16(w, 2), m3);
            const __m512i c3 = _mm512_and_si512(w, m3);

            acc = _mm512_dpbusd_epi32(acc, c0, _mm512_loadu_si512((const void *) (ab + 0)));
            acc = _mm512_dpbusd_epi32(acc, c1, _mm512_loadu_si512((const void *) (ab + 64)));
            acc = _mm512_dpbusd_epi32(acc, c2, _mm512_loadu_si512((const void *) (ab + 128)));
            acc = _mm512_dpbusd_epi32(acc, c3, _mm512_loadu_si512((const void *) (ab + 192)));
        }
        const int32_t dot = _mm512_reduce_add_epi32(acc) - sum_a;
        y[r]              = (float) dot * scale;
    }
}

/* Prefill GEMM. JT tokens share each weight-row load: the 4 unpacked code
 * vectors per block are reused across the token-tile, so the packed weight
 * is read once per (row, block, tile) and the VPDPBUSD throughput is the
 * limiter. y is [m, n_out] row-major. */
#define I2S_JT 4

void i2s_gemm_avx512_vnni(size_t         m,
                          size_t         n_out,
                          size_t         n_in,
                          const int8_t  *xq,
                          const int32_t *sum_a,
                          const float   *scale,
                          const uint8_t  w_raw[],
                          float          y[]) {
    const size_t n_blocks  = n_in / I2S_BLOCK_ELEMS;
    const size_t row_bytes = n_in / 4;

    /* Permute every token's activations once (shared, read-only). */
    int8_t *perm = (int8_t *) malloc(m * n_in);
    if (perm == nullptr) {
        for (size_t i = 0; i < m; i++) {
            i2s_gemv_m1_avx512_vnni(
                    n_out, n_in, xq + i * n_in, sum_a[i], w_raw, scale[i], y + i * n_out);
        }
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < m; i++) {
        build_acts_perm(n_blocks, xq + i * n_in, perm + i * n_in);
    }

    const __m512i m3 = _mm512_set1_epi8(3);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_out; r++) {
        const uint8_t *Wr = w_raw + r * row_bytes;
        for (size_t j0 = 0; j0 < m; j0 += I2S_JT) {
            const size_t jt = (m - j0 < I2S_JT) ? (m - j0) : I2S_JT;
            __m512i      acc[I2S_JT];
            for (size_t jj = 0; jj < I2S_JT; jj++) {
                acc[jj] = _mm512_setzero_si512();
            }
            for (size_t b = 0; b < n_blocks; b++) {
                const __m512i w  = _mm512_loadu_si512((const void *) (Wr + b * I2S_BLOCK_BYTES));
                const __m512i c0 = _mm512_and_si512(_mm512_srli_epi16(w, 6), m3);
                const __m512i c1 = _mm512_and_si512(_mm512_srli_epi16(w, 4), m3);
                const __m512i c2 = _mm512_and_si512(_mm512_srli_epi16(w, 2), m3);
                const __m512i c3 = _mm512_and_si512(w, m3);
                for (size_t jj = 0; jj < jt; jj++) {
                    const int8_t *ab = perm + (j0 + jj) * n_in + b * 256;
                    __m512i       a  = acc[jj];
                    a = _mm512_dpbusd_epi32(a, c0, _mm512_loadu_si512((const void *) (ab + 0)));
                    a = _mm512_dpbusd_epi32(a, c1, _mm512_loadu_si512((const void *) (ab + 64)));
                    a = _mm512_dpbusd_epi32(a, c2, _mm512_loadu_si512((const void *) (ab + 128)));
                    a = _mm512_dpbusd_epi32(a, c3, _mm512_loadu_si512((const void *) (ab + 192)));
                    acc[jj] = a;
                }
            }
            for (size_t jj = 0; jj < jt; jj++) {
                const int32_t dot = _mm512_reduce_add_epi32(acc[jj]) - sum_a[j0 + jj];
                y[(j0 + jj) * n_out + r] = (float) dot * scale[j0 + jj];
            }
        }
    }
    free(perm);
}

/* ===================== x4 row-interleaved kernels ========================= */
/* x4[grp*n_in + c] packs 4 output rows' codes for column c (row 4g+rb at
 * shift 6-2rb). One 64-byte load = 64 columns × 4 rows; one activation load
 * (natural column order, no permute) feeds all 4 rows via 4 independent
 * VPDPBUSDs. n_out%4==0, n_in%64==0. */

void i2s_x4_gemv_m1_avx512_vnni(size_t        n_out,
                                size_t        n_in,
                                const int8_t *xq,
                                int32_t       sum_a,
                                const uint8_t x4[],
                                float         scale,
                                float         y[static n_out]) {
    const size_t  n_groups  = n_out / 4;
    const size_t  n_cblocks = n_in / 64;
    const __m512i m3        = _mm512_set1_epi8(3);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t grp = 0; grp < n_groups; grp++) {
        const uint8_t *Wg = x4 + grp * n_in;
        __m512i        a0 = _mm512_setzero_si512();
        __m512i        a1 = _mm512_setzero_si512();
        __m512i        a2 = _mm512_setzero_si512();
        __m512i        a3 = _mm512_setzero_si512();
        for (size_t cb = 0; cb < n_cblocks; cb++) {
            const __m512i w = _mm512_loadu_si512((const void *) (Wg + cb * 64));
            const __m512i a = _mm512_loadu_si512((const void *) (xq + cb * 64));
            a0 = _mm512_dpbusd_epi32(a0, _mm512_and_si512(_mm512_srli_epi16(w, 6), m3), a);
            a1 = _mm512_dpbusd_epi32(a1, _mm512_and_si512(_mm512_srli_epi16(w, 4), m3), a);
            a2 = _mm512_dpbusd_epi32(a2, _mm512_and_si512(_mm512_srli_epi16(w, 2), m3), a);
            a3 = _mm512_dpbusd_epi32(a3, _mm512_and_si512(w, m3), a);
        }
        y[grp * 4 + 0] = (float) (_mm512_reduce_add_epi32(a0) - sum_a) * scale;
        y[grp * 4 + 1] = (float) (_mm512_reduce_add_epi32(a1) - sum_a) * scale;
        y[grp * 4 + 2] = (float) (_mm512_reduce_add_epi32(a2) - sum_a) * scale;
        y[grp * 4 + 3] = (float) (_mm512_reduce_add_epi32(a3) - sum_a) * scale;
    }
}

#define I2S_X4_TT 2 /* tokens per tile: 4 rows × TT → 4·TT live accumulators */

void i2s_x4_gemm_avx512_vnni(size_t         m,
                             size_t         n_out,
                             size_t         n_in,
                             const int8_t  *xq,
                             const int32_t *sum_a,
                             const float   *scale,
                             const uint8_t  x4[],
                             float          y[]) {
    const size_t  n_groups  = n_out / 4;
    const size_t  n_cblocks = n_in / 64;
    const __m512i m3        = _mm512_set1_epi8(3);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t grp = 0; grp < n_groups; grp++) {
        const uint8_t *Wg = x4 + grp * n_in;
        for (size_t j0 = 0; j0 < m; j0 += I2S_X4_TT) {
            const size_t tt = (m - j0 < I2S_X4_TT) ? (m - j0) : I2S_X4_TT;
            /* acc[row][token] */
            __m512i acc[4][I2S_X4_TT];
            for (size_t rb = 0; rb < 4; rb++) {
                for (size_t jj = 0; jj < I2S_X4_TT; jj++) {
                    acc[rb][jj] = _mm512_setzero_si512();
                }
            }
            for (size_t cb = 0; cb < n_cblocks; cb++) {
                const __m512i w  = _mm512_loadu_si512((const void *) (Wg + cb * 64));
                const __m512i r0 = _mm512_and_si512(_mm512_srli_epi16(w, 6), m3);
                const __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(w, 4), m3);
                const __m512i r2 = _mm512_and_si512(_mm512_srli_epi16(w, 2), m3);
                const __m512i r3 = _mm512_and_si512(w, m3);
                for (size_t jj = 0; jj < tt; jj++) {
                    const __m512i a = _mm512_loadu_si512(
                            (const void *) (xq + (j0 + jj) * n_in + cb * 64));
                    acc[0][jj] = _mm512_dpbusd_epi32(acc[0][jj], r0, a);
                    acc[1][jj] = _mm512_dpbusd_epi32(acc[1][jj], r1, a);
                    acc[2][jj] = _mm512_dpbusd_epi32(acc[2][jj], r2, a);
                    acc[3][jj] = _mm512_dpbusd_epi32(acc[3][jj], r3, a);
                }
            }
            for (size_t jj = 0; jj < tt; jj++) {
                const int32_t sa = sum_a[j0 + jj];
                const float   sc = scale[j0 + jj];
                float        *yr = y + (j0 + jj) * n_out + grp * 4;
                yr[0]            = (float) (_mm512_reduce_add_epi32(acc[0][jj]) - sa) * sc;
                yr[1]            = (float) (_mm512_reduce_add_epi32(acc[1][jj]) - sa) * sc;
                yr[2]            = (float) (_mm512_reduce_add_epi32(acc[2][jj]) - sa) * sc;
                yr[3]            = (float) (_mm512_reduce_add_epi32(acc[3][jj]) - sa) * sc;
            }
        }
    }
}
