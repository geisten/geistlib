/*
 * src/backends/cpu_x86/kernel_bf16_gemm_scalar.c — bf16 SGEMM reference.
 *
 * Pure C23. Operates on the same packed layout as the AVX-512+BF16
 * variant so the cross-ISA test is direct comparison.
 *
 * Contract: kernel_bf16_gemm.h.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_bf16_gemm.h"

#include <stddef.h>
#include <string.h>

void bf16_pack_weights_ntile16(size_t       N,
                               size_t       K,
                               const bf16_t W_flat[static N * K],
                               bf16_t       W_packed[static N * K]) {
    /* Per the layout in kernel_bf16_gemm.h:
     *   W_packed[(j/16) * K * 16 + (k/2) * 32 + (j%16) * 2 + (k%2)]
     *     = W_flat[j * K + k]
     */
    const size_t n_tiles = N / BF16_NTILE;
    for (size_t t = 0; t < n_tiles; t++) {
        for (size_t kp = 0; kp < K / 2; kp++) {
            for (size_t lane = 0; lane < BF16_NTILE; lane++) {
                const size_t j       = t * BF16_NTILE + lane;
                const size_t k0      = 2 * kp;
                const size_t k1      = 2 * kp + 1;
                const size_t p_base  = t * K * BF16_NTILE + kp * 32 + lane * 2;
                W_packed[p_base + 0] = W_flat[j * K + k0];
                W_packed[p_base + 1] = W_flat[j * K + k1];
            }
        }
    }
}

void bf16_gemm_scalar(size_t       M,
                      size_t       N,
                      size_t       K,
                      const float  X[static M * K],
                      const bf16_t W_packed[static N * K],
                      float        Y[static M * N]) {
    const size_t n_tiles = N / BF16_NTILE;
    memset(Y, 0, M * N * sizeof(float));

    /* For each output tile (lane group of 16 cells) and each m-row,
     * sweep K in pairs and accumulate. Mirrors the AVX kernel's loop
     * structure verbatim so a divergence in either is easy to localize. */
    for (size_t t = 0; t < n_tiles; t++) {
        const bf16_t *Wt = W_packed + t * K * BF16_NTILE;
        for (size_t i = 0; i < M; i++) {
            float acc[BF16_NTILE];
            for (size_t l = 0; l < BF16_NTILE; l++) {
                acc[l] = 0.0f;
            }
            for (size_t kp = 0; kp < K / 2; kp++) {
                const float   a0  = X[i * K + 2 * kp];
                const float   a1  = X[i * K + 2 * kp + 1];
                const bf16_t *Wkp = Wt + kp * 32;
                for (size_t l = 0; l < BF16_NTILE; l++) {
                    const float w0 = bf16_to_fp32(Wkp[l * 2 + 0]);
                    const float w1 = bf16_to_fp32(Wkp[l * 2 + 1]);
                    acc[l] += a0 * w0 + a1 * w1;
                }
            }
            for (size_t l = 0; l < BF16_NTILE; l++) {
                Y[i * N + t * BF16_NTILE + l] = acc[l];
            }
        }
    }
}
