/*
 * src/backends/cpu_x86/kernel_w8a8.c — W8A8 dispatcher + GEMV.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Reuses the W4A8 dispatcher's ISA selection: w8a8_dot picks the same
 * tier the W4A8 path picked (single dispatcher init for both). Compiled
 * at baseline -march=x86-64-v3; the AVX-512 + VNNI variant lives in
 * its own TU with -mavx512vnni.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_w8a8.h"

#include "kernel_w4a8.h" /* shared w4a8_dispatcher_init/current */

#include <stddef.h>
#include <stdint.h>

#include "par.h"

typedef float (*w8a8_dot_fn)(size_t        n_blocks,
                             const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
                             const float   w_scales[static n_blocks],
                             const float   w_offsets[static n_blocks],
                             const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
                             const int32_t sum_a_per_block[static n_blocks],
                             const float   act_scales[static w8a8_act_groups(n_blocks)]);

[[nodiscard]] float w8a8_dot_avx512_vnni(size_t        n_blocks,
                                         const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
                                         const float   w_scales[static n_blocks],
                                         const float   w_offsets[static n_blocks],
                                         const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
                                         const int32_t sum_a_per_block[static n_blocks],
                                         const float act_scales[static w8a8_act_groups(n_blocks)]);

void w8a8_gemm_avx512_vnni(
        size_t        n_tokens,
        size_t        n_rows,
        size_t        n_blocks_per_row,
        const uint8_t weights[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
        const float   w_scales[static n_rows * n_blocks_per_row],
        const float   w_offsets[static n_rows * n_blocks_per_row],
        const int8_t  acts[static n_tokens * n_blocks_per_row * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_tokens * n_blocks_per_row],
        const float   act_scales[static n_tokens * w8a8_act_groups(n_blocks_per_row)],
        float         out[static n_tokens * n_rows]);

static w8a8_dot_fn g_dot8    = nullptr;
static int         g_inited8 = 0;

static void w8a8_dispatch_init(void) {
    if (g_inited8 != 0) {
        return;
    }
    /* The W4A8 dispatcher's tier (cpuid + GEIST_FORCE_ISA): AVX-512 VNNI
     * there means VPDPBUSD here too. */
    const enum w4a8_isa tier = w4a8_dispatcher_init();
    if (tier == W4A8_ISA_AVX512_VNNI || tier == W4A8_ISA_AVX512_BF16) {
        g_dot8 = w8a8_dot_avx512_vnni;
    } else {
        g_dot8 = w8a8_dot_scalar;
    }
    g_inited8 = 1;
}

[[nodiscard]] float w8a8_dot(size_t        n_blocks,
                             const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
                             const float   w_scales[static n_blocks],
                             const float   w_offsets[static n_blocks],
                             const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
                             const int32_t sum_a_per_block[static n_blocks],
                             const float   act_scales[static w8a8_act_groups(n_blocks)]) {
    if (g_inited8 == 0) {
        w8a8_dispatch_init();
    }
    return g_dot8(n_blocks, weights, w_scales, w_offsets, acts, sum_a_per_block, act_scales);
}

[[nodiscard]] int w8a8_isa_is_vnni(void) {
    if (g_inited8 == 0) {
        w8a8_dispatch_init();
    }
    return g_dot8 == w8a8_dot_avx512_vnni;
}

/* One GEMV or GEMM for geist_par_for. */
struct w8a8_rows {
    size_t         n_tokens, n_rows, n_blocks;
    w8a8_dot_fn    dot;
    const uint8_t *weights;
    const float   *w_scales;
    const float   *w_offsets;
    const int8_t  *acts;
    const int32_t *sum_a;
    const float   *act_scales; /* w8a8_act_groups(n_blocks) per token */
    float         *out;
};

static void w8a8_gemv_rows(void *ctx, size_t m0, size_t m1) {
    const struct w8a8_rows c              = *(const struct w8a8_rows *) ctx;
    const size_t           bytes_per_row  = c.n_blocks * W8A8_BLOCK_ELEMS;
    const size_t           scales_per_row = c.n_blocks;
    for (size_t m = m0; m < m1; m++) {
        const uint8_t *w_row = c.weights + m * bytes_per_row;
        const float   *s_row = c.w_scales + m * scales_per_row;
        const float   *o_row = c.w_offsets + m * scales_per_row;
        c.out[m] = c.dot(c.n_blocks, w_row, s_row, o_row, c.acts, c.sum_a, c.act_scales);
    }
}

void w8a8_gemv(size_t        n_rows,
               size_t        n_blocks_per_row,
               const uint8_t weights[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const float   w_scales[static n_rows * n_blocks_per_row],
               const float   w_offsets[static n_rows * n_blocks_per_row],
               const int8_t  acts[static n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const int32_t sum_a_per_block[static n_blocks_per_row],
               const float   act_scales[static w8a8_act_groups(n_blocks_per_row)],
               float         out[static n_rows]) {
    if (g_inited8 == 0) {
        w8a8_dispatch_init();
    }
    struct w8a8_rows c = {.n_tokens   = 1,
                          .n_blocks   = n_blocks_per_row,
                          .dot        = g_dot8,
                          .weights    = weights,
                          .w_scales   = w_scales,
                          .w_offsets  = w_offsets,
                          .acts       = acts,
                          .sum_a      = sum_a_per_block,
                          .act_scales = act_scales,
                          .out        = out};
    geist_par_for(n_rows, w8a8_gemv_rows, &c);
}

/* The scalar GEMM's rows [r0, r1): one dispatched dot per token and row. */
static void w8a8_gemm_rows(void *ctx, size_t r0, size_t r1) {
    const struct w8a8_rows c              = *(const struct w8a8_rows *) ctx;
    const size_t           bytes_per_row  = c.n_blocks * W8A8_BLOCK_ELEMS;
    const size_t           scales_per_row = c.n_blocks;
    for (size_t r = r0; r < r1; r++) {
        const uint8_t *w_row = c.weights + r * bytes_per_row;
        const float   *s_row = c.w_scales + r * scales_per_row;
        const float   *o_row = c.w_offsets + r * scales_per_row;
        for (size_t j = 0; j < c.n_tokens; j++) {
            c.out[j * c.n_rows + r] = c.dot(c.n_blocks,
                                            w_row,
                                            s_row,
                                            o_row,
                                            c.acts + j * bytes_per_row,
                                            c.sum_a + j * scales_per_row,
                                            c.act_scales + j * w8a8_act_groups(c.n_blocks));
        }
    }
}

void w8a8_gemm(size_t        n_tokens,
               size_t        n_rows,
               size_t        n_blocks_per_row,
               const uint8_t weights[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const float   w_scales[static n_rows * n_blocks_per_row],
               const float   w_offsets[static n_rows * n_blocks_per_row],
               const int8_t  acts[static n_tokens * n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const int32_t sum_a_per_block[static n_tokens * n_blocks_per_row],
               const float   act_scales[static n_tokens * w8a8_act_groups(n_blocks_per_row)],
               float         out[static n_tokens * n_rows]) {
    if (g_inited8 == 0) {
        w8a8_dispatch_init();
    }
    if (g_dot8 == w8a8_dot_avx512_vnni) {
        w8a8_gemm_avx512_vnni(n_tokens,
                              n_rows,
                              n_blocks_per_row,
                              weights,
                              w_scales,
                              w_offsets,
                              acts,
                              sum_a_per_block,
                              act_scales,
                              out);
        return;
    }
    /* Scalar fallback: one dispatched dot product per token and row. This path
     * is intentionally untiled because it runs only on x86 without VNNI. */
    struct w8a8_rows c = {.n_tokens   = n_tokens,
                          .n_rows     = n_rows,
                          .n_blocks   = n_blocks_per_row,
                          .dot        = g_dot8,
                          .weights    = weights,
                          .w_scales   = w_scales,
                          .w_offsets  = w_offsets,
                          .acts       = acts,
                          .sum_a      = sum_a_per_block,
                          .act_scales = act_scales,
                          .out        = out};
    geist_par_for(n_rows, w8a8_gemm_rows, &c);
}
