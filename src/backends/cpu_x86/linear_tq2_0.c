/*
 * src/backends/cpu_x86/linear_tq2_0.c — cpu_x86 native TQ2_0 linear (AVX2).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * TQ2_0 (ternary BitNet weights, 2 bits per trit, one fp16 scale per 256)
 * had no x86 kernel: it ran the generic path (linear_generic.c), which
 * decodes every trit to fp32 before the dot — 14 tok/s prefill on
 * bitnet-large where llama.cpp runs 1374 (#410). This kernel never leaves
 * int8. The weights are read straight from the GGUF bytes (no repack, no
 * aux memory); the activations are quantized once per call to int8 with
 * one scale per 256 elements (d = amax / 127, as llama.cpp's Q8_K for this
 * format) plus each block's integer sum S. A weight block's 2-bit codes
 * v in {0, 1, 2} (trit = v - 1) go straight into maddubs:
 *
 *   maddubs(v, xq)  — a pair is at most 2 * 2 * 127 = 508, and the eight
 *                     32-element chunks of a block sum in int16 to at most
 *                     4064: no saturation, P = sum(v * xq) is exact.
 *
 * and sum(trit * x) = d_w * d_x * (P - S), with P - S an exact int32.
 *
 * M>1 tiles NR activation rows per pass over a weight row: each block's
 * codes are unpacked once and multiplied against NR activation blocks.
 * Rows split across OpenMP threads; each output element is a fixed-order
 * reduction, independent of the thread count.
 *
 * On AVX-512 VNNI hosts M>1 binds kernel_tq2_0_avx512_vnni.c's 4-row x
 * 4-token register tiles instead (same int32 block sums, four VPDPBUSD per
 * block and token). M=1 keeps the AVX2 GEMV.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_tq2_0.h"

#include "backend_state.h"
#include "kernel_tq2_0_avx512_vnni.h"
#include "linear_util.h"

#include "linear_ref.h"
#include "quant.h"
#include "quant_blocks.h"

#include <geist_backend.h>
#include <geist_types.h>

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

constexpr size_t QK = TQ2_0_BLOCK_ELEMS; /* 256 */
static_assert(TQ2_0_BLOCK_ELEMS == 256, "the kernels below walk 2 x 4 chunks of 32 per block");

/* Activation tile height of the M>1 kernel. */
constexpr size_t NR = 4;

/* One activation row to int8 blocks of 256: d = amax / 127, q rounded to
 * nearest-even (|q| <= 127), plus each block's integer sum S. */
static void quantize_row_q8_256(size_t nb, const float *x, int8_t *qx, float *dx, int32_t *sx) {
    for (size_t b = 0; b < nb; b++) {
        const float *xb    = x + b * QK;
        const float  amax  = amax_ps(QK, xb);
        dx[b]              = amax / 127.0f;
        const __m256 scale = q8_scale(amax);
        __m256i      s16   = _mm256_setzero_si256();
        for (size_t c = 0; c < QK; c += 32) {
            const __m256i q = quant32(xb + c, scale);
            _mm256_storeu_si256((__m256i *) (qx + b * QK + c), q);
            s16 = _mm256_add_epi16(s16,
                                   _mm256_maddubs_epi16(_mm256_set1_epi8(1), q)); /* |.| <= 2032 */
        }
        sx[b] = hsum_epi32(_mm256_madd_epi16(s16, _mm256_set1_epi16(1)));
    }
}

static inline float tq2_scale(const struct block_tq2_0_t *w) {
    return _cvtsh_ss((uint16_t) (w->d[0] | (w->d[1] << 8)));
}

/* acc + d_w d_x (P - S) for one block. -S goes into lane 0 of the exact
 * int32 block sum before the conversion. */
static inline __m256
fma_block(const struct block_tq2_0_t *w, const int8_t *xq, float dx, int32_t sx, __m256 acc) {
    const __m256i mask = _mm256_set1_epi8(3);
    __m256i       p16  = _mm256_setzero_si256();
    for (size_t h = 0; h < 2; h++) {
        const __m256i q = _mm256_loadu_si256((const __m256i *) (w->qs + h * 32));
        for (int l = 0; l < 4; l++) {
            const __m256i v = _mm256_and_si256(_mm256_srli_epi16(q, 2 * l), mask);
            const __m256i x =
                    _mm256_loadu_si256((const __m256i *) (xq + h * 128 + (size_t) l * 32));
            p16 = _mm256_add_epi16(p16, _mm256_maddubs_epi16(v, x));
        }
    }
    const __m256i p = _mm256_sub_epi32(_mm256_madd_epi16(p16, _mm256_set1_epi16(1)),
                                       _mm256_zextsi128_si256(_mm_cvtsi32_si128(sx)));
    return _mm256_fmadd_ps(_mm256_set1_ps(tq2_scale(w) * dx), _mm256_cvtepi32_ps(p), acc);
}

/* One weight row against one activation row. Two accumulators (even / odd
 * block) so the FMA chain is not latency-bound; fixed order. */
static inline float dot_row(size_t                      nb,
                            const struct block_tq2_0_t *w,
                            const int8_t               *qx,
                            const float                *dx,
                            const int32_t              *sx) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t b    = 0;
    for (; b + 2 <= nb; b += 2) {
        acc0 = fma_block(&w[b], qx + b * QK, dx[b], sx[b], acc0);
        acc1 = fma_block(&w[b + 1], qx + (b + 1) * QK, dx[b + 1], sx[b + 1], acc1);
    }
    if (b < nb) {
        acc0 = fma_block(&w[b], qx + b * QK, dx[b], sx[b], acc0);
    }
    return hsum_ps(_mm256_add_ps(acc0, acc1));
}

/* NR activation rows against one weight row: each chunk of codes unpacked
 * once, one accumulator per activation row. The integer block sums are the
 * same as dot_row's; only the fp32 summation order differs, so M=1 and M>1
 * agree to float rounding. */
static void dot_rows(size_t                      nb,
                     size_t                      n_in,
                     const struct block_tq2_0_t *w,
                     const int8_t               *qx,
                     const float                *dx,
                     const int32_t              *sx,
                     float                       out[static NR]) {
    const __m256i mask = _mm256_set1_epi8(3);
    __m256        acc[NR];
    for (size_t r = 0; r < NR; r++) {
        acc[r] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < nb; b++) {
        __m256i p16[NR];
        for (size_t r = 0; r < NR; r++) {
            p16[r] = _mm256_setzero_si256();
        }
        for (size_t h = 0; h < 2; h++) {
            const __m256i q = _mm256_loadu_si256((const __m256i *) (w[b].qs + h * 32));
            for (int l = 0; l < 4; l++) {
                const __m256i v   = _mm256_and_si256(_mm256_srli_epi16(q, 2 * l), mask);
                const size_t  off = b * QK + h * 128 + (size_t) l * 32;
                for (size_t r = 0; r < NR; r++) {
                    const __m256i x = _mm256_loadu_si256((const __m256i *) (qx + r * n_in + off));
                    p16[r]          = _mm256_add_epi16(p16[r], _mm256_maddubs_epi16(v, x));
                }
            }
        }
        const float dw = tq2_scale(&w[b]);
        for (size_t r = 0; r < NR; r++) {
            const __m256i p =
                    _mm256_sub_epi32(_mm256_madd_epi16(p16[r], _mm256_set1_epi16(1)),
                                     _mm256_zextsi128_si256(_mm_cvtsi32_si128(sx[r * nb + b])));
            acc[r] = _mm256_fmadd_ps(
                    _mm256_set1_ps(dw * dx[r * nb + b]), _mm256_cvtepi32_ps(p), acc[r]);
        }
    }
    for (size_t r = 0; r < NR; r++) {
        out[r] = hsum_ps(acc[r]);
    }
}

static void cpu_x86_linear_tq2_0_m1(const float               *x,
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
    quantize_row_q8_256(nb, x, ws->mN_acts, ws->mN_scale, ws->mN_sum_a);
    const int8_t               *qx = ws->mN_acts;
    const float                *dx = ws->mN_scale;
    const int32_t              *sx = ws->mN_sum_a;
    const struct block_tq2_0_t *wb = (const struct block_tq2_0_t *) w->raw;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < n_out; j++) {
        y[j] = dot_row(nb, wb + j * nb, qx, dx, sx);
    }
}

/* M>1: one team quantizes the m rows, then runs the GEMM (implicit barrier
 * between the two worksharing loops). With vnni, the GEMM is the register
 * tiles of kernel_tq2_0_avx512_vnni.c, one call per group of
 * TQ2_0_VNNI_TILE_ROWS output rows. */
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
    int8_t                     *qx      = ws->mN_acts;
    float                      *dx      = ws->mN_scale;
    int32_t                    *sx      = ws->mN_sum_a;
    const struct block_tq2_0_t *wb      = (const struct block_tq2_0_t *) w->raw;
    const size_t                m_til   = m - m % NR;
    const size_t                n_tiles = (n_out + TQ2_0_VNNI_TILE_ROWS - 1) / TQ2_0_VNNI_TILE_ROWS;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < m; i++) {
            quantize_row_q8_256(nb, x + i * n_in, qx + i * n_in, dx + i * nb, sx + i * nb);
        }
        if (vnni) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (size_t g = 0; g < n_tiles; g++) {
                const size_t j0 = g * TQ2_0_VNNI_TILE_ROWS;
                const size_t rows =
                        n_out - j0 < TQ2_0_VNNI_TILE_ROWS ? n_out - j0 : TQ2_0_VNNI_TILE_ROWS;
                tq2_0_gemm_rows_avx512_vnni(m, nb, n_out, j0, rows, wb, qx, dx, sx, y);
            }
        } else {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (size_t j = 0; j < n_out; j++) {
                const struct block_tq2_0_t *wr = wb + j * nb;
                float                       out[NR];
                for (size_t i = 0; i < m_til; i += NR) {
                    dot_rows(nb, n_in, wr, qx + i * n_in, dx + i * nb, sx + i * nb, out);
                    for (size_t r = 0; r < NR; r++) {
                        y[(i + r) * n_out + j] = out[r];
                    }
                }
                for (size_t i = m_til; i < m; i++) {
                    y[i * n_out + j] = dot_row(nb, wr, qx + i * n_in, dx + i * nb, sx + i * nb);
                }
            }
        }
    }
}

static void cpu_x86_linear_tq2_0_mN(size_t                     m,
                                    const float               *x,
                                    const struct geist_weight *w,
                                    struct geist_backend      *be,
                                    float                     *y) {
    linear_mN(false, m, x, w, be, y);
}

static void cpu_x86_linear_tq2_0_mN_vnni(size_t                     m,
                                         const float               *x,
                                         const struct geist_weight *w,
                                         struct geist_backend      *be,
                                         float                     *y) {
    linear_mN(true, m, x, w, be, y);
}
void cpu_x86_linear_tq2_0_bind(struct geist_weight *w) {
    w->linear_m1 = cpu_x86_linear_tq2_0_m1;
    w->linear_mN = vnni_tiles_usable() ? cpu_x86_linear_tq2_0_mN_vnni : cpu_x86_linear_tq2_0_mN;
}
