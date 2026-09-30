/*
 * src/backends/cpu_x86/kernel_q8_0_avx512_vnni.c — Q8_0 x Q8_0 prefill tiles,
 * AVX-512 VNNI.
 *
 * Layer: BACKEND (cpu_x86). Compiled with -mavx512f -mavx512bw -mavx512dq
 * -mavx512vl -mavx512vnni (mk/backend-cpu_x86.mk). linear_q8_0.c decides,
 * at bind time and outside this TU, whether the host may run it.
 *
 * The AVX2 M>1 kernel in linear_q8_0.c runs one dot per (row, token), so
 * every 32-element block pays for the sign trick, the s16 step and its own
 * scale broadcast, and each weight row is re-read from L1 once per token.
 * Here a 4-row x 4-token register tile walks the blocks two at a time:
 *
 *   - One VPDPBUSD covers a block pair (64 bytes) of one (row, token). The
 *     activations enter as u8 = q_x + 128 (one XOR per token and pair,
 *     shared by 4 rows), the weights as stored (s8). The 128 * sum(w) that
 *     the offset adds is taken off up front: each (row, pair) accumulator
 *     starts at -128 * sum(w), one more VPDPBUSD shared by 4 tokens. Each
 *     int32 lane is then exactly the sum of its four q_w * q_x products;
 *     nothing saturates (VPDPBUSD adds four u8 x s8 products into int32).
 *   - Lanes 0-7 hold block b, lanes 8-15 block b+1, so one FMA scales both
 *     with [d_w,b * d_x,b] x8 | [d_w,b+1 * d_x,b+1] x8.
 *
 * Even blocks accumulate in the low half and odd blocks in the high half,
 * and the reduction adds the halves first: the AVX2 kernel's order (acc0 =
 * even, acc1 = odd, hsum of acc0 + acc1). The int32 block sums are exact in
 * both, so the two kernels produce the same bits for finite activations.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_q8_0_avx512_vnni.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

constexpr size_t QK          = 32;
constexpr size_t TILE_ROWS   = Q8_0_VNNI_TILE_ROWS;
constexpr size_t TILE_TOKENS = 4;
static_assert(sizeof(struct block_q8_0_t) == 2 + QK, "one Q8_0 block: fp16 d + 32 int8");
static_assert(TILE_ROWS == 4, "q8_0_gemm_rows_avx512_vnni instantiates 4-row tiles");

/* Blocks b and b+1 of one weight row (34-byte stride, so two loads). */
static inline __m512i load_w_pair(const struct block_q8_0_t *wb) {
    const __m256i lo = _mm256_loadu_si256((const __m256i *) wb[0].qs);
    const __m256i hi = _mm256_loadu_si256((const __m256i *) wb[1].qs);
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

/* [lo x8 | hi x8] */
static inline __m512 halves(float lo, float hi) {
    return _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(lo)), _mm256_set1_ps(hi), 1);
}

/* Same tree as linear_q8_0.c's dot_row: halves, then 8 -> 4 -> 2 -> 1. */
static inline float reduce(__m512 v) {
    const __m256 s  = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    __m128       s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4              = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4              = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    return _mm_cvtss_f32(s4);
}

/* rows x tokens outputs; rows <= TILE_ROWS and tokens <= TILE_TOKENS are
 * compile-time constants at every call site. always_inline keeps them so:
 * without it gcc 14 merges the 4 x {4,3,2,1} calls into one clone with a
 * runtime token count, which puts xu / dx on the stack and branches in the
 * inner loop. Inlined, the loops unroll and acc lives in registers. w
 * points at the first row, q_x / d_x at the first token, y at (first
 * token, first row). */
[[gnu::always_inline]] static inline void tile(size_t                     rows,
                                               size_t                     tokens,
                                               size_t                     nb,
                                               size_t                     n_out,
                                               const struct block_q8_0_t *w,
                                               const int8_t              *q_x,
                                               const float               *d_x,
                                               float                     *y) {
    const size_t  n_in = nb * QK;
    const __m512i k80  = _mm512_set1_epi8((char) 0x80);
    __m512        acc[TILE_ROWS][TILE_TOKENS];
    for (size_t r = 0; r < rows; r++) {
        for (size_t t = 0; t < tokens; t++) {
            acc[r][t] = _mm512_setzero_ps();
        }
    }
    size_t b = 0;
    for (; b + 2 <= nb; b += 2) {
        __m512i xu[TILE_TOKENS];
        __m512  dx[TILE_TOKENS];
        for (size_t t = 0; t < tokens; t++) {
            xu[t] = _mm512_xor_si512(_mm512_loadu_si512(q_x + t * n_in + b * QK), k80);
            dx[t] = halves(d_x[t * nb + b], d_x[t * nb + b + 1]);
        }
        for (size_t r = 0; r < rows; r++) {
            const struct block_q8_0_t *wb   = w + r * nb + b;
            const __m512i              wq   = load_w_pair(wb);
            const __m512i              bias = _mm512_sub_epi32(
                    _mm512_setzero_si512(), _mm512_dpbusd_epi32(_mm512_setzero_si512(), k80, wq));
            const __m512 dw = halves(_cvtsh_ss(wb[0].d), _cvtsh_ss(wb[1].d));
            for (size_t t = 0; t < tokens; t++) {
                const __m512i s = _mm512_dpbusd_epi32(bias, xu[t], wq);
                acc[r][t] =
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(dw, dx[t]), acc[r][t]);
            }
        }
    }
    if (b < nb) {
        /* Odd block count: the last block in the low half, zeros above
         * (0 * 0 adds nothing to the odd-block half). */
        __m512i xu[TILE_TOKENS];
        __m512  dx[TILE_TOKENS];
        for (size_t t = 0; t < tokens; t++) {
            const __m256i x = _mm256_loadu_si256((const __m256i *) (q_x + t * n_in + b * QK));
            xu[t]           = _mm512_xor_si512(_mm512_zextsi256_si512(x), k80);
            dx[t]           = halves(d_x[t * nb + b], 0.0f);
        }
        for (size_t r = 0; r < rows; r++) {
            const struct block_q8_0_t *wb = w + r * nb + b;
            const __m512i wq = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i *) wb->qs));
            const __m512i bias = _mm512_sub_epi32(
                    _mm512_setzero_si512(), _mm512_dpbusd_epi32(_mm512_setzero_si512(), k80, wq));
            const __m512 dw = halves(_cvtsh_ss(wb->d), 0.0f);
            for (size_t t = 0; t < tokens; t++) {
                const __m512i s = _mm512_dpbusd_epi32(bias, xu[t], wq);
                acc[r][t] =
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(dw, dx[t]), acc[r][t]);
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        for (size_t t = 0; t < tokens; t++) {
            y[t * n_out + r] = reduce(acc[r][t]);
        }
    }
}

void q8_0_gemm_rows_avx512_vnni(size_t                     m,
                                size_t                     nb,
                                size_t                     n_out,
                                size_t                     j0,
                                size_t                     n_rows,
                                const struct block_q8_0_t *w,
                                const int8_t              *q_x,
                                const float               *d_x,
                                float                     *y) {
    const size_t n_in = nb * QK;
    if (n_rows == TILE_ROWS) {
        const struct block_q8_0_t *wr = w + j0 * nb;
        size_t                     t  = 0;
        for (; t + TILE_TOKENS <= m; t += TILE_TOKENS) {
            tile(4, 4, nb, n_out, wr, q_x + t * n_in, d_x + t * nb, y + t * n_out + j0);
        }
        switch (m - t) {
        case 3:
            tile(4, 3, nb, n_out, wr, q_x + t * n_in, d_x + t * nb, y + t * n_out + j0);
            break;
        case 2:
            tile(4, 2, nb, n_out, wr, q_x + t * n_in, d_x + t * nb, y + t * n_out + j0);
            break;
        case 1:
            tile(4, 1, nb, n_out, wr, q_x + t * n_in, d_x + t * nb, y + t * n_out + j0);
            break;
        default:
            break;
        }
        return;
    }
    /* A last group of fewer than 4 rows: one row at a time. */
    for (size_t j = j0; j < j0 + n_rows; j++) {
        const struct block_q8_0_t *wr = w + j * nb;
        size_t                     t  = 0;
        for (; t + TILE_TOKENS <= m; t += TILE_TOKENS) {
            tile(1, 4, nb, n_out, wr, q_x + t * n_in, d_x + t * nb, y + t * n_out + j);
        }
        for (; t < m; t++) {
            tile(1, 1, nb, n_out, wr, q_x + t * n_in, d_x + t * nb, y + t * n_out + j);
        }
    }
}
