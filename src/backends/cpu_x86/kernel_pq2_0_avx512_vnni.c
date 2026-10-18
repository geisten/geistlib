/*
 * src/backends/cpu_x86/kernel_pq2_0_avx512_vnni.c — PQ2_0 x int8 prefill
 * tiles, AVX-512 VNNI (#85).
 *
 * Layer: BACKEND (cpu_x86). Compiled with -mavx512f -mavx512bw -mavx512dq
 * -mavx512vl -mavx512vnni (mk/backend-cpu_x86.mk). linear_pq2_0.c decides,
 * at bind time and outside this TU, whether the host may run it.
 *
 * linear_pq2_0.c's AVX2 M>1 kernel spends four maddubs, an s16 add chain
 * and a madd per block and token in 256-bit registers, with the fp32
 * accumulators in memory. Here a 4-row x 4-token register tile walks one
 * block at a time, as kernel_tq2_0_avx512_vnni.c does for TQ2_0:
 *
 *   - A block's 32 code bytes are broadcast to both halves of a zmm and
 *     shifted right by 2l in the low half and 2l + 2 in the high one,
 *     masked to 2 bits: two zmm hold shift levels (0, 1) and (2, 3), i.e.
 *     64 contiguous activation bytes each (32 * l + k, see the header).
 *     Once per row and block, shared by 4 tokens.
 *   - Two VPDPBUSD (codes unsigned 0..2, activations signed) sum the block
 *     for one (row, token) in int32 lanes, exactly; the chain starts from
 *     minus the activation sum in lane 0, which folds the codes' +1 bias.
 *   - One FMA scales the block by the weight's d; the token's scale is
 *     applied once at the end.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_pq2_0_avx512_vnni.h"
#include "linear_util.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

constexpr size_t QK          = 128; /* PQ2_0_BLOCK_ELEMS */
constexpr size_t TILE_ROWS   = PQ2_0_VNNI_TILE_ROWS;
constexpr size_t TILE_TOKENS = 4;
static_assert(TILE_ROWS == 4, "pq2_0_gemm_rows_avx512_vnni instantiates 4-row tiles");

/* rows x tokens outputs; constant rows / tokens at every call site keep acc
 * in registers. w points at the first row, xq / neg_sum / inv at the first
 * token (block stride m), y at (first token, first row). */
[[gnu::always_inline]] static inline void tile(size_t         rows,
                                               size_t         tokens,
                                               size_t         m,
                                               size_t         nb,
                                               size_t         rb,
                                               size_t         n_out,
                                               const uint8_t *w,
                                               const int8_t  *xq,
                                               const int32_t *neg_sum,
                                               const float   *inv,
                                               float         *y) {
    const __m512i mask = _mm512_set1_epi8(3);
    /* Shift counts per 16-bit lane: 0 / 2 in the low / high half for levels
     * (0, 1), 4 / 6 for (2, 3). */
    const __m512i sh01 = _mm512_inserti64x4(_mm512_set1_epi16(0), _mm256_set1_epi16(2), 1);
    const __m512i sh23 = _mm512_inserti64x4(_mm512_set1_epi16(4), _mm256_set1_epi16(6), 1);
    __m512        acc[TILE_ROWS][TILE_TOKENS];
    for (size_t r = 0; r < rows; r++) {
        for (size_t t = 0; t < tokens; t++) {
            acc[r][t] = _mm512_setzero_ps();
        }
    }
    for (size_t b = 0; b < nb; b++) {
        __m512i bias[TILE_TOKENS];
        for (size_t t = 0; t < tokens; t++) {
            bias[t] = _mm512_zextsi128_si512(_mm_cvtsi32_si128(neg_sum[b * m + t]));
        }
        for (size_t r = 0; r < rows; r++) {
            const uint8_t *blk = w + r * rb + b * 34;
            const __m512i  q =
                    _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *) (blk + 2)));
            const __m512i c01 = _mm512_and_si512(_mm512_srlv_epi16(q, sh01), mask);
            const __m512i c23 = _mm512_and_si512(_mm512_srlv_epi16(q, sh23), mask);
            const __m512  d   = _mm512_set1_ps(_cvtsh_ss((uint16_t) (blk[0] | (blk[1] << 8))));
            for (size_t t = 0; t < tokens; t++) {
                const int8_t *xb = xq + (b * m + t) * QK;
                __m512i       s  = _mm512_dpbusd_epi32(bias[t], c01, _mm512_loadu_si512(xb));
                s                = _mm512_dpbusd_epi32(s, c23, _mm512_loadu_si512(xb + 64));
                acc[r][t]        = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), d, acc[r][t]);
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        for (size_t t = 0; t < tokens; t++) {
            y[t * n_out + r] = hsum512_ps(acc[r][t]) * inv[t];
        }
    }
}

void pq2_0_gemm_rows_avx512_vnni(size_t         m,
                                 size_t         nb,
                                 size_t         rb,
                                 size_t         n_out,
                                 size_t         j0,
                                 size_t         n_rows,
                                 const uint8_t *raw,
                                 const int8_t  *xq,
                                 const int32_t *neg_sum,
                                 const float   *inv,
                                 float         *y) {
#define TILE(R, T, J, TT) \
    tile(R,               \
         T,               \
         m,               \
         nb,              \
         rb,              \
         n_out,           \
         raw + (J) * rb,  \
         xq + (TT) * QK,  \
         neg_sum + (TT),  \
         inv + (TT),      \
         y + (TT) * n_out + (J))
    if (n_rows == TILE_ROWS) {
        size_t t = 0;
        for (; t + TILE_TOKENS <= m; t += TILE_TOKENS) {
            TILE(4, 4, j0, t);
        }
        switch (m - t) {
        case 3:
            TILE(4, 3, j0, t);
            break;
        case 2:
            TILE(4, 2, j0, t);
            break;
        case 1:
            TILE(4, 1, j0, t);
            break;
        default:
            break;
        }
        return;
    }
    /* A last group of fewer than 4 rows: one row at a time. */
    for (size_t j = j0; j < j0 + n_rows; j++) {
        size_t t = 0;
        for (; t + TILE_TOKENS <= m; t += TILE_TOKENS) {
            TILE(1, 4, j, t);
        }
        for (; t < m; t++) {
            TILE(1, 1, j, t);
        }
    }
#undef TILE
}
