/*
 * src/backends/cpu_x86/linear_q8_0.c — cpu_x86 native Q8_0 linear (AVX2).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Q8_0 is the format of the small reference models (Qwen3-0.6B, Qwen3.5-0.8B,
 * SmolLM2-360M). The kernel never leaves int8: the weights are read straight
 * from the GGUF bytes (no repack, no aux memory), the activation row is
 * quantized to Q8_0 blocks once per call (d = amax/127 per 32 elements, the
 * same scheme the reference engines use for Q8_0 x Q8_0), and each block's 32
 * products are one maddubs + madd:
 *
 *   maddubs(|w|, sign(x, w))  — u8 x s8 pairs into s16. |w| <= 128 and
 *                               |x| <= 127, so a pair is at most 32512:
 *                               never saturates, the int32 block sum is exact.
 *
 * then scaled by d_w * d_x in fp32. Rows split across threads; M>1
 * quantizes all m activation rows once (in parallel) and keeps each weight
 * row in L1 while it is dotted against every one of them.
 *
 * AVX2 is the backend's x86-64-v3 baseline, so this runs on every host
 * cpu_x86 does (the AVX2-only ones included). On AVX-512 VNNI hosts M>1
 * binds kernel_q8_0_avx512_vnni.c's register tiles instead (same bits,
 * about twice the throughput); M=1 is bound by memory bandwidth and keeps
 * the AVX2 GEMV, which VNNI does not speed up.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q8_0.h"

#include "backend_state.h"
#include "kernel_q8_0_avx512_vnni.h"
#include "linear_util.h"

#include "linear_ref.h"
#include "par.h"
#include "quant.h"
#include "quant_blocks.h"

#include <geist_backend.h>
#include <geist_types.h>

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

constexpr size_t QK = Q8_0_BLOCK_ELEMS; /* 32 */
static_assert(Q8_0_BLOCK_ELEMS == 32, "the kernels below load one Q8_0 block per ymm");

/* One activation row to Q8_0: per block d = amax / 127 and q = x / d
 * rounded to nearest-even, so |q| <= 127 (the bound the sign trick needs).
 * Same packing as the reference engines' quantize_row_q8_0. */
static void quantize_row_q8_0(size_t nb, const float *x, int8_t *qx, float *dx) {
    for (size_t b = 0; b < nb; b++) {
        _mm256_storeu_si256((__m256i *) (qx + b * QK), quant_block_q8_0(x + b * QK, &dx[b]));
    }
}

/* Exact int32 sum of one block's 32 products, as 8 fp32 lanes. */
static inline __m256 block_products(const int8_t *wq, const int8_t *xq) {
    const __m256i w   = _mm256_loadu_si256((const __m256i *) wq);
    const __m256i x   = _mm256_loadu_si256((const __m256i *) xq);
    const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, _mm256_set1_epi16(1)));
}

/* One weight row against one quantized activation row. Two accumulators
 * (even / odd block) so the FMA chain is not latency-bound; fixed order. */
static inline float
dot_row(size_t nb, const struct block_q8_0_t *w, const int8_t *qx, const float *dx) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t b    = 0;
    for (; b + 2 <= nb; b += 2) {
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w[b].d) * dx[b]),
                               block_products(w[b].qs, qx + b * QK),
                               acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w[b + 1].d) * dx[b + 1]),
                               block_products(w[b + 1].qs, qx + (b + 1) * QK),
                               acc1);
    }
    if (b < nb) {
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w[b].d) * dx[b]),
                               block_products(w[b].qs, qx + b * QK),
                               acc0);
    }
    return hsum_ps(_mm256_add_ps(acc0, acc1));
}

/* One call for geist_par_for: the weight, the activation rows x and their
 * Q8_0 blocks (qx, dx: written by quant_rows, read by the rest), y. */
struct q8_call {
    size_t                     m, n_in, n_out, nb;
    const struct block_q8_0_t *wb;
    const float               *x;
    int8_t                    *qx;
    float                     *dx;
    float                     *y;
};

/* Activation rows [i0, i1) to Q8_0 blocks. */
static void quant_rows(void *ctx, size_t i0, size_t i1) {
    const struct q8_call c = *(const struct q8_call *) ctx;
    for (size_t i = i0; i < i1; i++) {
        quantize_row_q8_0(c.nb, c.x + i * c.n_in, c.qx + i * c.n_in, c.dx + i * c.nb);
    }
}

/* M=1: output rows [j0, j1). */
static void rows_m1(void *ctx, size_t j0, size_t j1) {
    const struct q8_call c = *(const struct q8_call *) ctx;
    for (size_t j = j0; j < j1; j++) {
        c.y[j] = dot_row(c.nb, c.wb + j * c.nb, c.qx, c.dx);
    }
}

static void cpu_x86_linear_q8_0_m1(const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    struct cpu_x86_workspace *ws    = acquire_acts(be, 1, n_in, QK, 0);
    if (ws == nullptr) {
        geist_linear_ref(1, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    quantize_row_q8_0(nb, x, ws->mN_acts, ws->mN_scale);
    struct q8_call c = {.m     = 1,
                        .n_in  = n_in,
                        .n_out = n_out,
                        .nb    = nb,
                        .wb    = (const struct block_q8_0_t *) w->raw,
                        .qx    = ws->mN_acts,
                        .dx    = ws->mN_scale,
                        .y     = y};
    geist_par_for(n_out, rows_m1, &c);
}

/* M>1 with vnni: groups [g0, g1) of Q8_0_VNNI_TILE_ROWS output rows. */
static void tiles_vnni(void *ctx, size_t g0, size_t g1) {
    const struct q8_call c = *(const struct q8_call *) ctx;
    for (size_t g = g0; g < g1; g++) {
        const size_t j0   = g * Q8_0_VNNI_TILE_ROWS;
        const size_t rows = c.n_out - j0 < Q8_0_VNNI_TILE_ROWS ? c.n_out - j0 : Q8_0_VNNI_TILE_ROWS;
        q8_0_gemm_rows_avx512_vnni(c.m, c.nb, c.n_out, j0, rows, c.wb, c.qx, c.dx, c.y);
    }
}

/* M>1 without: output rows [j0, j1), each against every activation row. */
static void rows_mN(void *ctx, size_t j0, size_t j1) {
    const struct q8_call c = *(const struct q8_call *) ctx;
    for (size_t j = j0; j < j1; j++) {
        const struct block_q8_0_t *wr = c.wb + j * c.nb;
        for (size_t i = 0; i < c.m; i++) {
            c.y[i * c.n_out + j] = dot_row(c.nb, wr, c.qx + i * c.n_in, c.dx + i * c.nb);
        }
    }
}

/* M>1: the m rows quantized, then the GEMM, one geist_par_for each. With
 * vnni, the GEMM is the 4-row x 4-token register tiles of
 * kernel_q8_0_avx512_vnni.c, one call per group of Q8_0_VNNI_TILE_ROWS
 * output rows; same bits as the AVX2 loop. */
static void linear_mN(bool                       vnni,
                      size_t                     m,
                      const float               *x,
                      const struct geist_weight *w,
                      struct geist_backend      *be,
                      float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    struct cpu_x86_workspace *ws    = acquire_acts(be, m, n_in, QK, 0);
    if (ws == nullptr) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    struct q8_call c = {.m     = m,
                        .n_in  = n_in,
                        .n_out = n_out,
                        .nb    = n_in / QK,
                        .wb    = (const struct block_q8_0_t *) w->raw,
                        .x     = x,
                        .qx    = ws->mN_acts,
                        .dx    = ws->mN_scale,
                        .y     = y};
    geist_par_for(m, quant_rows, &c);
    if (vnni) {
        geist_par_for((n_out + Q8_0_VNNI_TILE_ROWS - 1) / Q8_0_VNNI_TILE_ROWS, tiles_vnni, &c);
    } else {
        geist_par_for(n_out, rows_mN, &c);
    }
}

static void cpu_x86_linear_q8_0_mN(size_t                     m,
                                   const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    linear_mN(false, m, x, w, be, y);
}

static void cpu_x86_linear_q8_0_mN_vnni(size_t                     m,
                                        const float               *x,
                                        const struct geist_weight *w,
                                        struct geist_backend      *be,
                                        float                     *y) {
    linear_mN(true, m, x, w, be, y);
}
void cpu_x86_linear_q8_0_bind(struct geist_weight *w) {
    w->linear_m1 = cpu_x86_linear_q8_0_m1;
    w->linear_mN = vnni_tiles_usable() ? cpu_x86_linear_q8_0_mN_vnni : cpu_x86_linear_q8_0_mN;
}
