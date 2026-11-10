/*
 * src/backends/cpu_x86/kernel_q4_0_avx512_vnni.c — Q4_0 / Q4_1 x Q8_0 prefill
 * tiles, AVX-512 VNNI.
 *
 * Layer: BACKEND (cpu_x86). Compiled with -mavx512f -mavx512bw -mavx512dq
 * -mavx512vl -mavx512vnni (mk/backend-cpu_x86.mk). linear_q4_0.c decides,
 * at bind time and outside this TU, whether the host may run it.
 *
 * linear_q4_0.c's AVX2 M>1 kernel spends a maddubs and a madd per block and
 * token in 256-bit registers. Here a 4-row x 4-token register tile walks the
 * blocks two at a time, as kernel_q8_0_avx512_vnni.c does:
 *
 *   - The nibbles of a block pair unpack into one zmm (lanes 0-1 block b,
 *     lanes 2-3 block b+1, low nibbles before high ones: the element order
 *     of the Q8_0 activations), once per row and shared by 4 tokens.
 *   - One VPDPBUSD covers the pair for one (row, token): the nibbles are the
 *     unsigned operand (0..15), the int8 activations the signed one, so the
 *     int32 lanes are exact block-sum partials with no offset trick.
 *   - Q4_0's -8 S per block starts the accumulator, in int32 lane 0 of each
 *     block half (one vector per token and pair, shared by 4 rows), so the
 *     integer sums are P - 8 S as in the AVX2 kernel. Q4_1's m dx S is a
 *     second FMA on a vector that is dx S in those lanes and zero elsewhere.
 *   - One FMA scales both blocks with [d_w,b dx_b] x8 | [d_w,b+1 dx_b+1] x8.
 *
 * The int32 block sums equal the AVX2 kernel's; only the fp32 summation
 * order differs, so the two agree to float rounding.
 *
 * MXFP4 (GPT-OSS's experts, #374) has tiles of its own at the end of the
 * file, described there.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_q4_0_avx512_vnni.h"
#include "linear_util.h"

#include "quant.h"

#include <immintrin.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

constexpr size_t QK          = 32;
constexpr size_t TILE_ROWS   = Q4_0_VNNI_TILE_ROWS;
constexpr size_t TILE_TOKENS = 4;
static_assert(TILE_ROWS == 4, "the gemm entry points instantiate 4-row tiles");

/* One block format: its byte stride and where d, m and qs sit. */
struct fmt {
    size_t stride;
    size_t qs;
    bool   q4_1; /* w = d q + m; otherwise Q4_0, w = d (q - 8) */
};
static constexpr struct fmt FMT_Q4_0 = {sizeof(struct block_q4_0_t), 2, false};
static constexpr struct fmt FMT_Q4_1 = {sizeof(struct block_q4_1_t), 4, true};

static inline float load_h(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, sizeof h);
    return _cvtsh_ss(h);
}

/* The 32 nibbles of each of two blocks as bytes: [lo b | hi b | lo b+1 |
 * hi b+1], 16 each. hi is nullptr for the odd last block (zeros above). */
static inline __m512i unpack_pair(const uint8_t *lo, const uint8_t *hi) {
    const __m128i b0 = _mm_loadu_si128((const __m128i *) lo);
    const __m128i b1 = hi != nullptr ? _mm_loadu_si128((const __m128i *) hi) : _mm_setzero_si128();
    const __m512i u  = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_set_m128i(b0, b0)), _mm256_set_m128i(b1, b1), 1);
    /* 128-bit lanes 1 and 3 (qwords 2, 3, 6, 7) take the high nibbles. */
    const __m512i v = _mm512_mask_blend_epi64(0xCC, u, _mm512_srli_epi16(u, 4));
    return _mm512_and_si512(v, _mm512_set1_epi8(0x0F));
}

/* [lo x8 | hi x8] */
static inline __m512 halves(float lo, float hi) {
    return _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(lo)), _mm256_set1_ps(hi), 1);
}

/* lane 0 = lo, lane 8 = hi, zeros elsewhere. */
static inline __m512i lanes0_8_epi32(int32_t lo, int32_t hi) {
    return _mm512_inserti32x4(
            _mm512_zextsi128_si512(_mm_cvtsi32_si128(lo)), _mm_cvtsi32_si128(hi), 2);
}
static inline __m512 lanes0_8_ps(float lo, float hi) {
    return _mm512_insertf32x4(_mm512_zextps128_ps512(_mm_set_ss(lo)), _mm_set_ss(hi), 2);
}

/* One block pair (or the odd last block, pair == false) of a rows x tokens
 * tile. always_inline with constant rows / tokens / fmt at every call site,
 * as in kernel_q8_0_avx512_vnni.c, keeps acc in registers. */
[[gnu::always_inline]] static inline void step(size_t         rows,
                                               size_t         tokens,
                                               struct fmt     f,
                                               bool           pair,
                                               size_t         b,
                                               size_t         nb,
                                               const uint8_t *w,
                                               const int8_t  *q_x,
                                               const float   *d_x,
                                               const int32_t *s_x,
                                               __m512         acc[TILE_ROWS][TILE_TOKENS]) {
    const size_t n_in   = nb * QK;
    const size_t row_by = nb * f.stride;
    __m512i      xs[TILE_TOKENS];
    __m512       dx[TILE_TOKENS];
    __m512i      bias[TILE_TOKENS];
    __m512       off[TILE_TOKENS];
    for (size_t t = 0; t < tokens; t++) {
        const int8_t *xt = q_x + t * n_in + b * QK;
        xs[t]            = pair ? _mm512_loadu_si512(xt)
                                : _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i *) xt));
        const float   d0 = d_x[t * nb + b];
        const float   d1 = pair ? d_x[t * nb + b + 1] : 0.0f;
        const int32_t s0 = s_x[t * nb + b];
        const int32_t s1 = pair ? s_x[t * nb + b + 1] : 0;
        dx[t]            = halves(d0, d1);
        if (f.q4_1) {
            off[t] = lanes0_8_ps(d0 * (float) s0, d1 * (float) s1);
        } else {
            bias[t] = lanes0_8_epi32(-8 * s0, -8 * s1);
        }
    }
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *b0 = w + r * row_by + b * f.stride;
        const uint8_t *b1 = pair ? b0 + f.stride : nullptr;
        const __m512i  q  = unpack_pair(b0 + f.qs, pair ? b1 + f.qs : nullptr);
        const __m512   dw = halves(load_h(b0), pair ? load_h(b1) : 0.0f);
        __m512         mw = _mm512_setzero_ps();
        if (f.q4_1) {
            mw = halves(load_h(b0 + 2), pair ? load_h(b1 + 2) : 0.0f);
        }
        for (size_t t = 0; t < tokens; t++) {
            const __m512i s =
                    _mm512_dpbusd_epi32(f.q4_1 ? _mm512_setzero_si512() : bias[t], q, xs[t]);
            acc[r][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(dw, dx[t]), acc[r][t]);
            if (f.q4_1) {
                acc[r][t] = _mm512_fmadd_ps(mw, off[t], acc[r][t]);
            }
        }
    }
}

/* rows x tokens outputs. w points at the first row, q_x / d_x / s_x at the
 * first token, y at (first token, first row). */
[[gnu::always_inline]] static inline void tile(size_t         rows,
                                               size_t         tokens,
                                               struct fmt     f,
                                               size_t         nb,
                                               size_t         n_out,
                                               const uint8_t *w,
                                               const int8_t  *q_x,
                                               const float   *d_x,
                                               const int32_t *s_x,
                                               float         *y) {
    __m512 acc[TILE_ROWS][TILE_TOKENS];
    for (size_t r = 0; r < rows; r++) {
        for (size_t t = 0; t < tokens; t++) {
            acc[r][t] = _mm512_setzero_ps();
        }
    }
    size_t b = 0;
    for (; b + 2 <= nb; b += 2) {
        step(rows, tokens, f, true, b, nb, w, q_x, d_x, s_x, acc);
    }
    if (b < nb) {
        step(rows, tokens, f, false, b, nb, w, q_x, d_x, s_x, acc);
    }
    for (size_t r = 0; r < rows; r++) {
        for (size_t t = 0; t < tokens; t++) {
            y[t * n_out + r] = hsum512_ps(acc[r][t]);
        }
    }
}

[[gnu::always_inline]] static inline void gemm_rows(struct fmt     f,
                                                    size_t         m,
                                                    size_t         nb,
                                                    size_t         n_out,
                                                    size_t         j0,
                                                    size_t         n_rows,
                                                    const uint8_t *w,
                                                    const int8_t  *q_x,
                                                    const float   *d_x,
                                                    const int32_t *s_x,
                                                    float         *y) {
    const size_t n_in   = nb * QK;
    const size_t row_by = nb * f.stride;
/* A macro, not a function (#348): it spells the caller's locals into one call of tile(). */
#define TILE(R, T, J, TT)   \
    tile(R,                 \
         T,                 \
         f,                 \
         nb,                \
         n_out,             \
         w + (J) * row_by,  \
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

void q4_0_gemm_rows_avx512_vnni(size_t                     m,
                                size_t                     nb,
                                size_t                     n_out,
                                size_t                     j0,
                                size_t                     n_rows,
                                const struct block_q4_0_t *w,
                                const int8_t              *q_x,
                                const float               *d_x,
                                const int32_t             *s_x,
                                float                     *y) {
    gemm_rows(FMT_Q4_0, m, nb, n_out, j0, n_rows, (const uint8_t *) w, q_x, d_x, s_x, y);
}

void q4_1_gemm_rows_avx512_vnni(size_t                     m,
                                size_t                     nb,
                                size_t                     n_out,
                                size_t                     j0,
                                size_t                     n_rows,
                                const struct block_q4_1_t *w,
                                const int8_t              *q_x,
                                const float               *d_x,
                                const int32_t             *s_x,
                                float                     *y) {
    gemm_rows(FMT_Q4_1, m, nb, n_out, j0, n_rows, (const uint8_t *) w, q_x, d_x, s_x, y);
}

/*
 * MXFP4 x Q8_0 (#374). The codes index the signed table kvalues_mxfp4
 * (-12..12), so the operands' roles swap: one VPSHUFB gives the signed
 * weights, and the activations are the unsigned operand as x + 128 (the
 * caller stores the Q8_0 codes XOR 0x80). The offset adds 128 sum(w) per
 * 4-element lane, a weight-only term: one VPDPBUSD of 128 against -w per
 * row and block pair gives -128 sum(w), which starts that row's
 * accumulators, so every int32 lane is exactly sum(w x) over its four
 * elements, and a block's lanes add up to the AVX2 kernel's sign-trick
 * block sum. Each lane then takes d_w d_x in fp32 (d_w = mxfp4_e8m0_half(e),
 * a power of two). No activation sums are read.
 *
 * From 4 tokens on, each group of 4 weight rows is unpacked once (codes,
 * -128 sum(w) and the scales, up to MX_PAIRS_MAX block pairs, 48 KB on the
 * stack) and every 4-token tile reads it from L1; the AVX2-era path
 * unpacked the rows again for every 4 tokens. Fewer tokens, or a wider K,
 * unpack in the tile itself. Both paths do the same operations on the same
 * values in the same order per output, so a token's result does not depend
 * on m or on which other tokens share its tile: m = 1 gives the bits of the
 * same token inside any m-row call.
 */

constexpr size_t MX_STRIDE    = sizeof(struct block_mxfp4_t);
constexpr size_t MX_PAIRS_MAX = 64; /* K up to 4096 unpacked once per row group */
static_assert(MX_STRIDE == 17, "MXFP4: the exponent byte, then 16 bytes of nibbles");

/* Up to 4 weight rows of an MXFP4 matrix, unpacked: per block pair the
 * signed codes, -128 sum(w) per 4-element lane, and [d_b x8 | d_b+1 x8]. */
struct mx_rows {
    alignas(64) int8_t q[TILE_ROWS][MX_PAIRS_MAX][64];
    alignas(64) int32_t neg[TILE_ROWS][MX_PAIRS_MAX][16];
    alignas(64) float dw[TILE_ROWS][MX_PAIRS_MAX][16];
};

/* Block pair (b, b + 1) of one weight row, or the odd last block alone
 * (pair == false: code 0, value 0, above it). */
[[gnu::always_inline]] static inline void
mx_unpack(bool pair, const uint8_t *b0, __m512i *q, __m512i *neg, __m512 *dw) {
    const uint8_t *b1    = pair ? b0 + MX_STRIDE : nullptr;
    const __m512i  table = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) kvalues_mxfp4));
    *q                   = _mm512_shuffle_epi8(table, unpack_pair(b0 + 1, pair ? b1 + 1 : nullptr));
    *neg                 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                                               _mm512_set1_epi8((char) 0x80),
                                               _mm512_sub_epi8(_mm512_setzero_si512(), *q));
    *dw                  = halves(mxfp4_e8m0_half(b0[0]), pair ? mxfp4_e8m0_half(b1[0]) : 0.0f);
}

/* rows x tokens outputs, weight rows from w (u == nullptr) or from u's rows
 * [r0, r0 + rows). q_x / d_x point at the first token, y at (first token,
 * first row). */
[[gnu::always_inline]] static inline void mx_tile(size_t                rows,
                                                  size_t                tokens,
                                                  size_t                nb,
                                                  size_t                n_out,
                                                  const uint8_t        *w,
                                                  const struct mx_rows *u,
                                                  size_t                r0,
                                                  const uint8_t        *q_x,
                                                  const float          *d_x,
                                                  float                *y) {
    const size_t n_in   = nb * QK;
    const size_t row_by = nb * MX_STRIDE;
    __m512       acc[TILE_ROWS][TILE_TOKENS];
    for (size_t r = 0; r < rows; r++) {
        for (size_t t = 0; t < tokens; t++) {
            acc[r][t] = _mm512_setzero_ps();
        }
    }
    for (size_t b = 0; b < nb; b += 2) {
        const bool pair = b + 1 < nb;
        __m512i    xs[TILE_TOKENS];
        __m512     dx[TILE_TOKENS];
        for (size_t t = 0; t < tokens; t++) {
            const uint8_t *xt = q_x + t * n_in + b * QK;
            xs[t] = pair ? _mm512_loadu_si512(xt)
                         : _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i *) xt));
            dx[t] = halves(d_x[t * nb + b], pair ? d_x[t * nb + b + 1] : 0.0f);
        }
        for (size_t r = 0; r < rows; r++) {
            __m512i q, neg;
            __m512  dw;
            if (u != nullptr) {
                q   = _mm512_load_si512(u->q[r0 + r][b / 2]);
                neg = _mm512_load_si512(u->neg[r0 + r][b / 2]);
                dw  = _mm512_load_ps(u->dw[r0 + r][b / 2]);
            } else {
                mx_unpack(pair, w + r * row_by + b * MX_STRIDE, &q, &neg, &dw);
            }
            for (size_t t = 0; t < tokens; t++) {
                const __m512i s = _mm512_dpbusd_epi32(neg, xs[t], q);
                acc[r][t] =
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(dw, dx[t]), acc[r][t]);
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        for (size_t t = 0; t < tokens; t++) {
            y[t * n_out + r] = hsum512_ps(acc[r][t]);
        }
    }
}

/* The tiles of rows [j0, j0 + n_rows), weight rows from u where given. */
[[gnu::always_inline]] static inline void mx_tiles(size_t                m,
                                                   size_t                nb,
                                                   size_t                n_out,
                                                   size_t                j0,
                                                   size_t                n_rows,
                                                   const uint8_t        *w0,
                                                   const struct mx_rows *u,
                                                   const uint8_t        *q_x,
                                                   const float          *d_x,
                                                   float                *y) {
    const size_t n_in   = nb * QK;
    const size_t row_by = nb * MX_STRIDE;
/* A macro, not a function (#348): it spells the caller's locals into one call of mx_tile(). */
#define MX_TILE(R, T, J, TT)   \
    mx_tile(R,                 \
            T,                 \
            nb,                \
            n_out,             \
            w0 + (J) * row_by, \
            u,                 \
            J,                 \
            q_x + (TT) * n_in, \
            d_x + (TT) * nb,   \
            y + (TT) * n_out + j0 + (J))
    if (n_rows == TILE_ROWS) {
        size_t t = 0;
        for (; t + TILE_TOKENS <= m; t += TILE_TOKENS) {
            MX_TILE(4, 4, 0, t);
        }
        switch (m - t) {
        case 3:
            MX_TILE(4, 3, 0, t);
            break;
        case 2:
            MX_TILE(4, 2, 0, t);
            break;
        case 1:
            MX_TILE(4, 1, 0, t);
            break;
        default:
            break;
        }
        return;
    }
    /* A last group of fewer than 4 rows: one row at a time. */
    for (size_t r = 0; r < n_rows; r++) {
        size_t t = 0;
        for (; t + TILE_TOKENS <= m; t += TILE_TOKENS) {
            MX_TILE(1, 4, r, t);
        }
        for (; t < m; t++) {
            MX_TILE(1, 1, r, t);
        }
    }
#undef MX_TILE
}

void mxfp4_gemm_rows_avx512_vnni(size_t                      m,
                                 size_t                      nb,
                                 size_t                      n_out,
                                 size_t                      j0,
                                 size_t                      n_rows,
                                 const struct block_mxfp4_t *w,
                                 const uint8_t              *q_x,
                                 const float                *d_x,
                                 float                      *y) {
    const size_t   row_by = nb * MX_STRIDE;
    const uint8_t *w0     = (const uint8_t *) w + j0 * row_by;
    if (m < TILE_TOKENS || (nb + 1) / 2 > MX_PAIRS_MAX) {
        mx_tiles(m, nb, n_out, j0, n_rows, w0, nullptr, q_x, d_x, y);
        return;
    }
    struct mx_rows u;
    for (size_t r = 0; r < n_rows; r++) {
        for (size_t b = 0; b < nb; b += 2) {
            __m512i q, neg;
            __m512  dw;
            mx_unpack(b + 1 < nb, w0 + r * row_by + b * MX_STRIDE, &q, &neg, &dw);
            _mm512_store_si512(u.q[r][b / 2], q);
            _mm512_store_si512(u.neg[r][b / 2], neg);
            _mm512_store_ps(u.dw[r][b / 2], dw);
        }
    }
    mx_tiles(m, nb, n_out, j0, n_rows, w0, &u, q_x, d_x, y);
}
