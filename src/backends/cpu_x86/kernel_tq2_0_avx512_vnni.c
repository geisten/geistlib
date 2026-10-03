/*
 * src/backends/cpu_x86/kernel_tq2_0_avx512_vnni.c — TQ2_0 x int8 prefill
 * tiles, AVX-512 VNNI.
 *
 * Layer: BACKEND (cpu_x86). Compiled with -mavx512f -mavx512bw -mavx512dq
 * -mavx512vl -mavx512vnni (mk/backend-cpu_x86.mk). linear_tq2_0.c decides,
 * at bind time and outside this TU, whether the host may run it.
 *
 * linear_tq2_0.c's AVX2 M>1 kernel spends eight maddubs and an s16 add per
 * block and token in 256-bit registers. Here a 4-row x 4-token register
 * tile walks one block at a time:
 *
 *   - A block's 2-bit codes v (trit = v - 1) unpack into four zmm, each two
 *     adjacent 32-element chunks: the 32 bytes qs[h * 32 ..] in both halves,
 *     shifted right by 2l in the low half and 2l + 2 in the high one, masked
 *     to 2 bits. That is elements h * 128 + l * 32 .. + 64, the order of the
 *     activations. Once per row, shared by 4 tokens.
 *   - Four VPDPBUSD (codes unsigned 0..2, activations signed) sum the block
 *     for one (row, token) in int32 lanes, exactly; the chain starts from
 *     -S in lane 0, so the lanes add up to P - S as in the AVX2 kernel.
 *   - One FMA scales the block by d_w d_x.
 *
 * The int32 block sums equal the AVX2 kernel's; only the fp32 summation
 * order differs, so the two agree to float rounding.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_tq2_0_avx512_vnni.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

constexpr size_t QK          = 256;
constexpr size_t TILE_ROWS   = TQ2_0_VNNI_TILE_ROWS;
constexpr size_t TILE_TOKENS = 4;
static_assert(sizeof(struct block_tq2_0_t) == 66, "TQ2_0 block: 64 code bytes + fp16 d");
static_assert(TILE_ROWS == 4, "tq2_0_gemm_rows_avx512_vnni instantiates 4-row tiles");

static inline float tq2_scale(const struct block_tq2_0_t *w) {
    return _cvtsh_ss((uint16_t) (w->d[0] | (w->d[1] << 8)));
}

/* Halves, then 8 -> 4 -> 2 -> 1. */
static inline float reduce(__m512 v) {
    const __m256 s  = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    __m128       s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4              = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4              = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    return _mm_cvtss_f32(s4);
}

/* rows x tokens outputs; constant rows / tokens at every call site, as in
 * kernel_q8_0_avx512_vnni.c, keep acc in registers. w points at the first
 * row, q_x / d_x / s_x at the first token, y at (first token, first row). */
[[gnu::always_inline]] static inline void tile(size_t                      rows,
                                               size_t                      tokens,
                                               size_t                      nb,
                                               size_t                      n_out,
                                               const struct block_tq2_0_t *w,
                                               const int8_t               *q_x,
                                               const float                *d_x,
                                               const int32_t              *s_x,
                                               float                      *y) {
    const size_t  n_in = nb * QK;
    const __m512i mask = _mm512_set1_epi8(3);
    /* Shift counts per 16-bit lane: 0 / 2 in the low / high half for chunks
     * (l, l + 1) = (0, 1), 4 / 6 for (2, 3). */
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
        float   dx[TILE_TOKENS];
        for (size_t t = 0; t < tokens; t++) {
            bias[t] = _mm512_zextsi128_si512(_mm_cvtsi32_si128(-s_x[t * nb + b]));
            dx[t]   = d_x[t * nb + b];
        }
        for (size_t r = 0; r < rows; r++) {
            const struct block_tq2_0_t *wb = w + r * nb + b;
            __m512i                     c[4];
            for (size_t h = 0; h < 2; h++) {
                const __m512i q = _mm512_broadcast_i64x4(
                        _mm256_loadu_si256((const __m256i *) (wb->qs + h * 32)));
                c[2 * h]     = _mm512_and_si512(_mm512_srlv_epi16(q, sh01), mask);
                c[2 * h + 1] = _mm512_and_si512(_mm512_srlv_epi16(q, sh23), mask);
            }
            const float dw = tq2_scale(wb);
            for (size_t t = 0; t < tokens; t++) {
                const int8_t *xb = q_x + t * n_in + b * QK;
                __m512i       s  = bias[t];
                for (size_t k = 0; k < 4; k++) {
                    s = _mm512_dpbusd_epi32(s, c[k], _mm512_loadu_si512(xb + k * 64));
                }
                acc[r][t] = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(s), _mm512_set1_ps(dw * dx[t]), acc[r][t]);
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        for (size_t t = 0; t < tokens; t++) {
            y[t * n_out + r] = reduce(acc[r][t]);
        }
    }
}

void tq2_0_gemm_rows_avx512_vnni(size_t                      m,
                                 size_t                      nb,
                                 size_t                      n_out,
                                 size_t                      j0,
                                 size_t                      n_rows,
                                 const struct block_tq2_0_t *w,
                                 const int8_t               *q_x,
                                 const float                *d_x,
                                 const int32_t              *s_x,
                                 float                      *y) {
    const size_t n_in = nb * QK;
#define TILE(R, T, J, TT)   \
    tile(R,                 \
         T,                 \
         nb,                \
         n_out,             \
         w + (J) * nb,      \
         q_x + (TT) * n_in, \
         d_x + (TT) * nb,   \
         s_x + (TT) * nb,   \
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
