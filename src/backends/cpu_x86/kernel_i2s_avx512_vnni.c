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
