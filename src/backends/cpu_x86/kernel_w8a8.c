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

#if defined(_OPENMP)
#include <omp.h>
#endif

typedef float (*w8a8_dot_fn)(size_t        n_blocks,
                             const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
                             const float   w_scales[static n_blocks],
                             const float   w_offsets[static n_blocks],
                             const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
                             const int32_t sum_a_per_block[static n_blocks],
                             float         scale_x);

[[nodiscard]] float w8a8_dot_avx512_vnni(size_t        n_blocks,
                                         const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
                                         const float   w_scales[static n_blocks],
                                         const float   w_offsets[static n_blocks],
                                         const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
                                         const int32_t sum_a_per_block[static n_blocks],
                                         float         scale_x);

void w8a8_gemm_avx512_vnni(
        size_t        n_tokens,
        size_t        n_rows,
        size_t        n_blocks_per_row,
        const uint8_t weights[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
        const float   w_scales[static n_rows * n_blocks_per_row],
        const float   w_offsets[static n_rows * n_blocks_per_row],
        const int8_t  acts[static n_tokens * n_blocks_per_row * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_tokens * n_blocks_per_row],
        const float   scale_x[static n_tokens],
        float         out[static n_tokens * n_rows]);

static w8a8_dot_fn g_dot8    = nullptr;
static int         g_inited8 = 0;

static void w8a8_dispatch_init(void) {
    if (g_inited8 != 0) {
        return;
    }
    /* The W4A8 dispatcher already encapsulates the cpuid + GEIST_FORCE_ISA
     * selection. We mirror its decision: if W4A8 resolved to AVX-512+VNNI
     * we know VPDPBUSD is available for us too. */
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
                             float         scale_x) {
    if (g_inited8 == 0) {
        w8a8_dispatch_init();
    }
    return g_dot8(n_blocks, weights, w_scales, w_offsets, acts, sum_a_per_block, scale_x);
}

[[nodiscard]] int w8a8_isa_is_vnni(void) {
    if (g_inited8 == 0) {
        w8a8_dispatch_init();
    }
    return g_dot8 == w8a8_dot_avx512_vnni;
}

void w8a8_gemv(size_t        n_rows,
               size_t        n_blocks_per_row,
               const uint8_t weights[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const float   w_scales[static n_rows * n_blocks_per_row],
               const float   w_offsets[static n_rows * n_blocks_per_row],
               const int8_t  acts[static n_blocks_per_row * W8A8_BLOCK_ELEMS],
               const int32_t sum_a_per_block[static n_blocks_per_row],
               float         scale_x,
               float         out[static n_rows]) {
    if (g_inited8 == 0) {
        w8a8_dispatch_init();
    }
    const size_t bytes_per_row  = n_blocks_per_row * W8A8_BLOCK_ELEMS;
    const size_t scales_per_row = n_blocks_per_row;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t m = 0; m < n_rows; m++) {
        const uint8_t *w_row = weights + m * bytes_per_row;
        const float   *s_row = w_scales + m * scales_per_row;
        const float   *o_row = w_offsets + m * scales_per_row;
        out[m] = g_dot8(n_blocks_per_row, w_row, s_row, o_row, acts, sum_a_per_block, scale_x);
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
               const float   scale_x[static n_tokens],
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
                              scale_x,
                              out);
        return;
    }
    /* Scalar fallback: per (token, row) dot via the dispatched scalar kernel.
     * ponytail: no tiling — only reached on non-VNNI x86, where the cpu_x86
     * backend is already a slow path; not worth a second blocked kernel. */
    const size_t bytes_per_row  = n_blocks_per_row * W8A8_BLOCK_ELEMS;
    const size_t scales_per_row = n_blocks_per_row;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_rows; r++) {
        const uint8_t *w_row = weights + r * bytes_per_row;
        const float   *s_row = w_scales + r * scales_per_row;
        const float   *o_row = w_offsets + r * scales_per_row;
        for (size_t j = 0; j < n_tokens; j++) {
            out[j * n_rows + r] = g_dot8(n_blocks_per_row,
                                         w_row,
                                         s_row,
                                         o_row,
                                         acts + j * bytes_per_row,
                                         sum_a_per_block + j * scales_per_row,
                                         scale_x[j]);
        }
    }
}
