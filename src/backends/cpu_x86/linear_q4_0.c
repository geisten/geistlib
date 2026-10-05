/*
 * src/backends/cpu_x86/linear_q4_0.c — cpu_x86 native Q4_0 / Q4_1 linear
 * (AVX2).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Q4_0 (and the Q4_1 tensors that ship inside "Q4_0" exports) had no x86
 * kernel: they ran the generic path (linear_generic.c), which dequantizes
 * every weight to fp32 before the dot — 2.8 tok/s prefill on a Q4_0 model
 * where llama.cpp runs 273 (#410). This kernel stays in int8 the same way
 * linear_q8_0.c does: the weights are read straight from the GGUF bytes (no
 * repack, no aux memory), the activations are quantized to Q8_0 blocks once
 * per call, and each block's 32 products are one maddubs + madd on the raw
 * nibbles:
 *
 *   maddubs(q, xq)  — q is the unsigned nibble (0..15), xq the int8
 *                     activation (|xq| <= 127): a pair is at most 3810, so
 *                     the int32 block sum P = sum(q * xq) is exact.
 *
 * Both formats are affine in q, so the offset comes out of the block sum
 * S = sum(xq), which the quantizer computes once per activation block:
 *
 *   Q4_0  w = d (q - 8)   ->  sum(w x) = d dx (P - 8 S)
 *   Q4_1  w = d q + m     ->  sum(w x) = d dx P + m dx S
 *
 * P - 8 S is still an exact int32, so Q4_0 gets exactly the integer dot of
 * the sign-trick formulation, minus its two vpsignb per block.
 *
 * M>1 tiles NR activation rows per pass over a weight row: each weight
 * block is unpacked once and multiplied against NR activation blocks, so
 * the unpack and the weight load are amortized and NR accumulators keep
 * the FMA chain busy. Rows split across OpenMP threads; each output element
 * is a fixed-order reduction, independent of the thread count.
 *
 * On AVX-512 VNNI hosts M>1 binds kernel_q4_0_avx512_vnni.c's 4-row x
 * 4-token register tiles instead (same int32 block sums, twice the vector
 * width, one VPDPBUSD per block pair). M=1 keeps the AVX2 GEMV.
 *
 * IQ4_NL and IQ4_XS share the quantizer and the tiling (#410). Their nibbles
 * index the fixed non-linear table kvalues_iq4nl (one VPSHUFB per block), so
 * w is signed and not affine in q: the block sum is the sign-trick
 * maddubs(|w|, sign(xq, w)), at most 2 * 127 * 127 per pair, exact. IQ4_XS
 * is IQ4_NL with a 6-bit scale per 32-element sub-block, which lines up with
 * the Q8_0 activation blocks. Both run the AVX2 kernels at every tier.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q4_0.h"

#include "backend_state.h"
#include "kernel_q4_0_avx512_vnni.h"
#include "linear_util.h"

#include "linear_ref.h"
#include "quant.h"
#include "quant_blocks.h"

#include <geist_backend.h>
#include <geist_types.h>

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

constexpr size_t QK = Q4_0_BLOCK_ELEMS; /* 32 */
static_assert(Q4_0_BLOCK_ELEMS == 32 && Q4_1_BLOCK_ELEMS == 32,
              "the kernels below unpack one block into one ymm");

/* Activation tile height of the M>1 kernel. */
constexpr size_t NR = 4;

/* One activation row to Q8_0 blocks (d = amax / 127, q rounded to nearest-
 * even, as in linear_q8_0.c) plus each block's integer sum S. */
static void quantize_row_q8_0_sum(size_t nb, const float *x, int8_t *qx, float *dx, int32_t *sx) {
    for (size_t b = 0; b < nb; b++) {
        const __m256i q = quant_block_q8_0(x + b * QK, &dx[b]);
        _mm256_storeu_si256((__m256i *) (qx + b * QK), q);
        sx[b] = hsum_epi32(sum_i8(q));
    }
}

/* A block's 16 bytes as 32 unsigned nibbles in element order: low nibbles
 * are elements 0..15, high nibbles 16..31. */
static inline __m256i unpack_nibbles(const uint8_t qs[static 16]) {
    const __m128i b    = _mm_loadu_si128((const __m128i *) qs);
    const __m128i mask = _mm_set1_epi8(0x0F);
    return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(b, 4), mask), _mm_and_si128(b, mask));
}

/* P = sum(q * xq) of one block, as 8 int32 lanes. */
static inline __m256i block_p(__m256i q, const int8_t *xq) {
    const __m256i x = _mm256_loadu_si256((const __m256i *) xq);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), _mm256_set1_epi16(1));
}

/* acc + d dx (P - 8 S) for one Q4_0 block. -8 S goes into lane 0 of the
 * exact int32 block sum before the conversion. */
static inline __m256
fma_block_q4_0(const struct block_q4_0_t *w, const int8_t *xq, float dx, int32_t sx, __m256 acc) {
    const __m256i p = _mm256_sub_epi32(block_p(unpack_nibbles(w->qs), xq),
                                       _mm256_zextsi128_si256(_mm_cvtsi32_si128(8 * sx)));
    return _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w->d) * dx), _mm256_cvtepi32_ps(p), acc);
}

/* acc + d dx P for one Q4_1 block (the m dx S term is scalar). */
static inline __m256
fma_block_q4_1(const struct block_q4_1_t *w, const int8_t *xq, float dx, __m256 acc) {
    const __m256i p = block_p(unpack_nibbles(w->qs), xq);
    return _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w->d) * dx), _mm256_cvtepi32_ps(p), acc);
}

/* Q4_0, one weight row against one activation row. Two accumulators (even
 * / odd block) so the FMA chain is not latency-bound; fixed order. */
static inline float dot_row_q4_0(size_t                     nb,
                                 const struct block_q4_0_t *w,
                                 const int8_t              *qx,
                                 const float               *dx,
                                 const int32_t             *sx) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t b    = 0;
    for (; b + 2 <= nb; b += 2) {
        acc0 = fma_block_q4_0(&w[b], qx + b * QK, dx[b], sx[b], acc0);
        acc1 = fma_block_q4_0(&w[b + 1], qx + (b + 1) * QK, dx[b + 1], sx[b + 1], acc1);
    }
    if (b < nb) {
        acc0 = fma_block_q4_0(&w[b], qx + b * QK, dx[b], sx[b], acc0);
    }
    return hsum_ps(_mm256_add_ps(acc0, acc1));
}

/* Q4_1: d dx P in the vector accumulators, m dx S in a scalar one. */
static inline float dot_row_q4_1(size_t                     nb,
                                 const struct block_q4_1_t *w,
                                 const int8_t              *qx,
                                 const float               *dx,
                                 const int32_t             *sx) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float  off  = 0.0f;
    size_t b    = 0;
    for (; b + 2 <= nb; b += 2) {
        acc0 = fma_block_q4_1(&w[b], qx + b * QK, dx[b], acc0);
        acc1 = fma_block_q4_1(&w[b + 1], qx + (b + 1) * QK, dx[b + 1], acc1);
        off += _cvtsh_ss(w[b].m) * dx[b] * (float) sx[b] +
               _cvtsh_ss(w[b + 1].m) * dx[b + 1] * (float) sx[b + 1];
    }
    if (b < nb) {
        acc0 = fma_block_q4_1(&w[b], qx + b * QK, dx[b], acc0);
        off += _cvtsh_ss(w[b].m) * dx[b] * (float) sx[b];
    }
    return hsum_ps(_mm256_add_ps(acc0, acc1)) + off;
}

/* NR activation rows against one weight row: each block unpacked once, one
 * accumulator per activation row (NR of them keep the FMA chain busy). The
 * integer block sums are the same as dot_row_*'s; only the fp32 summation
 * order differs, so M=1 and M>1 agree to float rounding. */
static void dot_rows_q4_0(size_t                     nb,
                          size_t                     n_in,
                          const struct block_q4_0_t *w,
                          const int8_t              *qx,
                          const float               *dx,
                          const int32_t             *sx,
                          float                      out[static NR]) {
    __m256 acc[NR];
    for (size_t r = 0; r < NR; r++) {
        acc[r] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < nb; b++) {
        const __m256i q  = unpack_nibbles(w[b].qs);
        const float   dw = _cvtsh_ss(w[b].d);
        for (size_t r = 0; r < NR; r++) {
            __m256i p = block_p(q, qx + r * n_in + b * QK);
            p = _mm256_sub_epi32(p, _mm256_zextsi128_si256(_mm_cvtsi32_si128(8 * sx[r * nb + b])));
            acc[r] = _mm256_fmadd_ps(
                    _mm256_set1_ps(dw * dx[r * nb + b]), _mm256_cvtepi32_ps(p), acc[r]);
        }
    }
    for (size_t r = 0; r < NR; r++) {
        out[r] = hsum_ps(acc[r]);
    }
}

static void dot_rows_q4_1(size_t                     nb,
                          size_t                     n_in,
                          const struct block_q4_1_t *w,
                          const int8_t              *qx,
                          const float               *dx,
                          const int32_t             *sx,
                          float                      out[static NR]) {
    __m256 acc[NR];
    float  off[NR];
    for (size_t r = 0; r < NR; r++) {
        acc[r] = _mm256_setzero_ps();
        off[r] = 0.0f;
    }
    for (size_t b = 0; b < nb; b++) {
        const __m256i q  = unpack_nibbles(w[b].qs);
        const float   dw = _cvtsh_ss(w[b].d);
        const float   mw = _cvtsh_ss(w[b].m);
        for (size_t r = 0; r < NR; r++) {
            const __m256i p = block_p(q, qx + r * n_in + b * QK);
            acc[r]          = _mm256_fmadd_ps(
                    _mm256_set1_ps(dw * dx[r * nb + b]), _mm256_cvtepi32_ps(p), acc[r]);
            off[r] += mw * dx[r * nb + b] * (float) sx[r * nb + b];
        }
    }
    for (size_t r = 0; r < NR; r++) {
        out[r] = hsum_ps(acc[r]) + off[r];
    }
}

/* ggml's kvalues_iq4nl (src/formats/gguf/iq4.c), in both 128-bit lanes. */
static inline __m256i iq4_table(void) {
    const __m128i t =
            _mm_setr_epi8(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
    return _mm256_set_m128i(t, t);
}

/* Block b (32 elements) of an IQ4_NL or IQ4_XS row: its 16 index bytes, and
 * its scale through *dw. */
[[gnu::always_inline]] static inline const uint8_t *
iq4_block(bool xs, const uint8_t *row, size_t b, float *dw) {
    if (!xs) {
        const struct block_iq4_nl_t *blk = (const struct block_iq4_nl_t *) row + b;
        *dw                              = _cvtsh_ss(blk->d);
        return blk->qs;
    }
    const struct block_iq4_xs_t *sb = (const struct block_iq4_xs_t *) row + b / 8;
    const size_t                 k  = b % 8;
    const int                    ls =
            ((sb->scales_l[k / 2] >> (4 * (k & 1))) & 0xF) | (((sb->scales_h >> (2 * k)) & 3) << 4);
    *dw = _cvtsh_ss(sb->d) * (float) (ls - 32);
    return sb->qs + 16 * k;
}

/* P = sum(w * xq) of one block of signed weights, as 8 int32 lanes. */
static inline __m256i block_p_signed(__m256i w, const int8_t *xq) {
    const __m256i x = _mm256_loadu_si256((const __m256i *) xq);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)),
                             _mm256_set1_epi16(1));
}

/* IQ4, one weight row against one activation row: d dx P per block, two
 * accumulators as in dot_row_q4_0. */
[[gnu::always_inline]] static inline float
dot_row_iq4(bool xs, size_t nb, const uint8_t *row, const int8_t *qx, const float *dx) {
    const __m256i table  = iq4_table();
    __m256        acc[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    for (size_t b = 0; b < nb; b++) {
        float          dw;
        const uint8_t *qs = iq4_block(xs, row, b, &dw);
        const __m256i  p =
                block_p_signed(_mm256_shuffle_epi8(table, unpack_nibbles(qs)), qx + b * QK);
        acc[b & 1] = _mm256_fmadd_ps(_mm256_set1_ps(dw * dx[b]), _mm256_cvtepi32_ps(p), acc[b & 1]);
    }
    return hsum_ps(_mm256_add_ps(acc[0], acc[1]));
}

/* NR activation rows against one IQ4 weight row, each block looked up once. */
[[gnu::always_inline]] static inline void dot_rows_iq4(bool           xs,
                                                       size_t         nb,
                                                       size_t         n_in,
                                                       const uint8_t *row,
                                                       const int8_t  *qx,
                                                       const float   *dx,
                                                       float          out[static NR]) {
    const __m256i table = iq4_table();
    __m256        acc[NR];
    for (size_t r = 0; r < NR; r++) {
        acc[r] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < nb; b++) {
        float          dw;
        const uint8_t *qs = iq4_block(xs, row, b, &dw);
        const __m256i  w  = _mm256_shuffle_epi8(table, unpack_nibbles(qs));
        const __m256i  aw = _mm256_sign_epi8(w, w);
        for (size_t r = 0; r < NR; r++) {
            const __m256i x = _mm256_loadu_si256((const __m256i *) (qx + r * n_in + b * QK));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w)),
                                                _mm256_set1_epi16(1));
            acc[r]          = _mm256_fmadd_ps(
                    _mm256_set1_ps(dw * dx[r * nb + b]), _mm256_cvtepi32_ps(p), acc[r]);
        }
    }
    for (size_t r = 0; r < NR; r++) {
        out[r] = hsum_ps(acc[r]);
    }
}

/* One instance per format: the OpenMP regions below are outlined once for
 * both, with xs a runtime value there. */
static float dot_row_iq4_nl(size_t nb, const uint8_t *row, const int8_t *qx, const float *dx) {
    return dot_row_iq4(false, nb, row, qx, dx);
}
static float dot_row_iq4_xs(size_t nb, const uint8_t *row, const int8_t *qx, const float *dx) {
    return dot_row_iq4(true, nb, row, qx, dx);
}
static void dot_rows_iq4_nl(size_t         nb,
                            size_t         n_in,
                            const uint8_t *row,
                            const int8_t  *qx,
                            const float   *dx,
                            float          out[static NR]) {
    dot_rows_iq4(false, nb, n_in, row, qx, dx, out);
}
static void dot_rows_iq4_xs(size_t         nb,
                            size_t         n_in,
                            const uint8_t *row,
                            const int8_t  *qx,
                            const float   *dx,
                            float          out[static NR]) {
    dot_rows_iq4(true, nb, n_in, row, qx, dx, out);
}

static void cpu_x86_linear_q4_0_m1(const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    struct cpu_x86_workspace *ws    = acquire_acts(be, 1, n_in, QK, QK);
    if (ws == nullptr) {
        geist_linear_ref(1, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    quantize_row_q8_0_sum(nb, x, ws->mN_acts, ws->mN_scale, ws->mN_sum_a);
    const int8_t  *qx = ws->mN_acts;
    const float   *dx = ws->mN_scale;
    const int32_t *sx = ws->mN_sum_a;

    if (w->dtype == GEIST_DTYPE_Q4_0) {
        const struct block_q4_0_t *wb = (const struct block_q4_0_t *) w->raw;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (size_t j = 0; j < n_out; j++) {
            y[j] = dot_row_q4_0(nb, wb + j * nb, qx, dx, sx);
        }
    } else {
        const struct block_q4_1_t *wb = (const struct block_q4_1_t *) w->raw;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (size_t j = 0; j < n_out; j++) {
            y[j] = dot_row_q4_1(nb, wb + j * nb, qx, dx, sx);
        }
    }
}

/* M>1: one team quantizes the m rows, then runs the GEMM (implicit barrier
 * between the two worksharing loops). With vnni, the GEMM is the register
 * tiles of kernel_q4_0_avx512_vnni.c, one call per group of
 * Q4_0_VNNI_TILE_ROWS output rows. */
static void linear_mN(bool                       vnni,
                      size_t                     m,
                      const float               *x,
                      const struct geist_weight *w,
                      struct geist_backend      *be,
                      float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    struct cpu_x86_workspace *ws    = acquire_acts(be, m, n_in, QK, QK);
    if (ws == nullptr) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    int8_t      *qx      = ws->mN_acts;
    float       *dx      = ws->mN_scale;
    int32_t     *sx      = ws->mN_sum_a;
    const bool   q4_0    = w->dtype == GEIST_DTYPE_Q4_0;
    const size_t m_til   = m - m % NR;
    const size_t n_tiles = (n_out + Q4_0_VNNI_TILE_ROWS - 1) / Q4_0_VNNI_TILE_ROWS;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < m; i++) {
            quantize_row_q8_0_sum(nb, x + i * n_in, qx + i * n_in, dx + i * nb, sx + i * nb);
        }
        if (vnni) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (size_t g = 0; g < n_tiles; g++) {
                const size_t j0 = g * Q4_0_VNNI_TILE_ROWS;
                const size_t rows =
                        n_out - j0 < Q4_0_VNNI_TILE_ROWS ? n_out - j0 : Q4_0_VNNI_TILE_ROWS;
                if (q4_0) {
                    q4_0_gemm_rows_avx512_vnni(m, nb, n_out, j0, rows, w->raw, qx, dx, sx, y);
                } else {
                    q4_1_gemm_rows_avx512_vnni(m, nb, n_out, j0, rows, w->raw, qx, dx, sx, y);
                }
            }
        } else {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (size_t j = 0; j < n_out; j++) {
                float out[NR];
                if (q4_0) {
                    const struct block_q4_0_t *wr = (const struct block_q4_0_t *) w->raw + j * nb;
                    for (size_t i = 0; i < m_til; i += NR) {
                        dot_rows_q4_0(nb, n_in, wr, qx + i * n_in, dx + i * nb, sx + i * nb, out);
                        for (size_t r = 0; r < NR; r++) {
                            y[(i + r) * n_out + j] = out[r];
                        }
                    }
                    for (size_t i = m_til; i < m; i++) {
                        y[i * n_out + j] =
                                dot_row_q4_0(nb, wr, qx + i * n_in, dx + i * nb, sx + i * nb);
                    }
                } else {
                    const struct block_q4_1_t *wr = (const struct block_q4_1_t *) w->raw + j * nb;
                    for (size_t i = 0; i < m_til; i += NR) {
                        dot_rows_q4_1(nb, n_in, wr, qx + i * n_in, dx + i * nb, sx + i * nb, out);
                        for (size_t r = 0; r < NR; r++) {
                            y[(i + r) * n_out + j] = out[r];
                        }
                    }
                    for (size_t i = m_til; i < m; i++) {
                        y[i * n_out + j] =
                                dot_row_q4_1(nb, wr, qx + i * n_in, dx + i * nb, sx + i * nb);
                    }
                }
            }
        }
    }
}

static void cpu_x86_linear_q4_0_mN(size_t                     m,
                                   const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    linear_mN(false, m, x, w, be, y);
}

static void cpu_x86_linear_q4_0_mN_vnni(size_t                     m,
                                        const float               *x,
                                        const struct geist_weight *w,
                                        struct geist_backend      *be,
                                        float                     *y) {
    linear_mN(true, m, x, w, be, y);
}

/* Bytes of one IQ4 weight row of n_in elements. */
static inline size_t iq4_row_bytes(bool xs, size_t n_in) {
    return xs ? n_in / IQ4_XS_BLOCK_ELEMS * sizeof(struct block_iq4_xs_t)
              : n_in / IQ4_NL_BLOCK_ELEMS * sizeof(struct block_iq4_nl_t);
}

static void cpu_x86_linear_iq4_m1(const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    const bool                xs    = w->dtype == GEIST_DTYPE_IQ4_XS;
    const size_t              row_b = iq4_row_bytes(xs, n_in);
    const uint8_t            *raw   = (const uint8_t *) w->raw;
    struct cpu_x86_workspace *ws    = acquire_acts(be, 1, n_in, QK, QK);
    if (ws == nullptr) {
        geist_linear_ref(1, x, w, y);
        return;
    }
    quantize_row_q8_0_sum(nb, x, ws->mN_acts, ws->mN_scale, ws->mN_sum_a);
    const int8_t *qx = ws->mN_acts;
    const float  *dx = ws->mN_scale;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < n_out; j++) {
        y[j] = xs ? dot_row_iq4_xs(nb, raw + j * row_b, qx, dx)
                  : dot_row_iq4_nl(nb, raw + j * row_b, qx, dx);
    }
}

static void cpu_x86_linear_iq4_mN(size_t                     m,
                                  const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    const bool                xs    = w->dtype == GEIST_DTYPE_IQ4_XS;
    const size_t              row_b = iq4_row_bytes(xs, n_in);
    const uint8_t            *raw   = (const uint8_t *) w->raw;
    struct cpu_x86_workspace *ws    = acquire_acts(be, m, n_in, QK, QK);
    if (ws == nullptr) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    int8_t      *qx    = ws->mN_acts;
    float       *dx    = ws->mN_scale;
    int32_t     *sx    = ws->mN_sum_a;
    const size_t m_til = m - m % NR;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < m; i++) {
            quantize_row_q8_0_sum(nb, x + i * n_in, qx + i * n_in, dx + i * nb, sx + i * nb);
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t j = 0; j < n_out; j++) {
            const uint8_t *wr = raw + j * row_b;
            float          out[NR];
            for (size_t i = 0; i < m_til; i += NR) {
                if (xs) {
                    dot_rows_iq4_xs(nb, n_in, wr, qx + i * n_in, dx + i * nb, out);
                } else {
                    dot_rows_iq4_nl(nb, n_in, wr, qx + i * n_in, dx + i * nb, out);
                }
                for (size_t r = 0; r < NR; r++) {
                    y[(i + r) * n_out + j] = out[r];
                }
            }
            for (size_t i = m_til; i < m; i++) {
                y[i * n_out + j] = xs ? dot_row_iq4_xs(nb, wr, qx + i * n_in, dx + i * nb)
                                      : dot_row_iq4_nl(nb, wr, qx + i * n_in, dx + i * nb);
            }
        }
    }
}

void cpu_x86_linear_q4_0_bind(struct geist_weight *w) {
    w->linear_m1 = cpu_x86_linear_q4_0_m1;
    w->linear_mN = vnni_tiles_usable() ? cpu_x86_linear_q4_0_mN_vnni : cpu_x86_linear_q4_0_mN;
}

void cpu_x86_linear_iq4_bind(struct geist_weight *w) {
    w->linear_m1 = cpu_x86_linear_iq4_m1;
    w->linear_mN = cpu_x86_linear_iq4_mN;
}
