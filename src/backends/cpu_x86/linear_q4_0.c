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
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q4_0.h"

#include "backend_state.h"

#include "checked.h"
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
    const __m256  abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i perm     = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i ones_u8  = _mm256_set1_epi8(1);
    const __m256i ones_16  = _mm256_set1_epi16(1);
    for (size_t b = 0; b < nb; b++) {
        const float *xb = x + b * QK;
        const __m256 v0 = _mm256_loadu_ps(xb);
        const __m256 v1 = _mm256_loadu_ps(xb + 8);
        const __m256 v2 = _mm256_loadu_ps(xb + 16);
        const __m256 v3 = _mm256_loadu_ps(xb + 24);
        __m256       m  = _mm256_max_ps(
                _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask)),
                _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask)));
        __m128 m4        = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
        m4               = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4               = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);

        dx[b]              = amax / 127.0f;
        const __m256 scale = _mm256_set1_ps(amax > 0.0f ? 127.0f / amax : 0.0f);
        __m256i      i0    = _mm256_cvtps_epi32(_mm256_round_ps(
                _mm256_mul_ps(v0, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i      i1    = _mm256_cvtps_epi32(_mm256_round_ps(
                _mm256_mul_ps(v1, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i      i2    = _mm256_cvtps_epi32(_mm256_round_ps(
                _mm256_mul_ps(v2, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i      i3    = _mm256_cvtps_epi32(_mm256_round_ps(
                _mm256_mul_ps(v3, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        i0                 = _mm256_packs_epi32(i0, i1);
        i2                 = _mm256_packs_epi32(i2, i3);
        i0                 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(i0, i2), perm);
        _mm256_storeu_si256((__m256i *) (qx + b * QK), i0);

        /* S: maddubs(1, q) pairs into s16, madd to s32, horizontal sum. */
        const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, i0), ones_16);
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s32), _mm256_extracti128_si256(s32, 1));
        s4         = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1, 0, 3, 2)));
        s4         = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(2, 3, 0, 1)));
        sx[b]      = _mm_cvtsi128_si32(s4);
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

static inline float hsum_ps(__m256 s) {
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4        = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4        = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    return _mm_cvtss_f32(s4);
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

/* The calling thread's workspace with room for m quantized activation rows
 * (int8 values, one fp32 scale and one int32 sum per block), or nullptr. */
static struct cpu_x86_workspace *acquire_acts(struct geist_backend *be, size_t m, size_t n_in) {
    size_t acts_bytes = 0, n_blocks = 0, scale_bytes = 0, sum_bytes = 0;
    if (be == nullptr || be->state == nullptr || ckd_mul(&acts_bytes, m, n_in) ||
        ckd_mul(&n_blocks, m, n_in / QK) || ckd_mul(&scale_bytes, n_blocks, sizeof(float)) ||
        ckd_mul(&sum_bytes, n_blocks, sizeof(int32_t))) {
        return nullptr;
    }
    return cpu_x86_ws_acquire_mN(
            (struct cpu_x86_state *) be->state, acts_bytes, sum_bytes, scale_bytes, 0);
}

static void cpu_x86_linear_q4_0_m1(const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    struct cpu_x86_workspace *ws    = acquire_acts(be, 1, n_in);
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

static void cpu_x86_linear_q4_0_mN(size_t                     m,
                                   const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    struct cpu_x86_workspace *ws    = acquire_acts(be, m, n_in);
    if (ws == nullptr) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    int8_t      *qx    = ws->mN_acts;
    float       *dx    = ws->mN_scale;
    int32_t     *sx    = ws->mN_sum_a;
    const bool   q4_0  = w->dtype == GEIST_DTYPE_Q4_0;
    const size_t m_til = m - m % NR;

    /* One team: quantize the m rows, then the GEMM (implicit barrier
     * between the two worksharing loops). */
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

bool cpu_x86_linear_q4_0_bind(struct geist_weight *w) {
    if (w == nullptr || (w->dtype != GEIST_DTYPE_Q4_0 && w->dtype != GEIST_DTYPE_Q4_1) ||
        w->n_in <= 0 || (size_t) w->n_in % QK != 0) {
        return false;
    }
    w->linear_m1 = cpu_x86_linear_q4_0_m1;
    w->linear_mN = cpu_x86_linear_q4_0_mN;
    return true;
}
