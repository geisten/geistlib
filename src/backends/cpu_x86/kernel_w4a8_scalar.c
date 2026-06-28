/*
 * src/backends/cpu_x86/kernel_w4a8_scalar.c — W4A8 scalar reference.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Pure C23, no intrinsics. Serves three roles:
 *   1. Correctness oracle for the AVX-512+VNNI / AVX-512 / AVX2 variants
 *      (cross-ISA-consistency tests, see tests/test_w4a8_kernel_unit.c).
 *   2. Fallback when none of the SIMD tiers is available
 *      (cpu_x86 on pre-Haswell, or under GEIST_FORCE_ISA=scalar).
 *   3. Algorithm documentation (the SIMD variants are harder to read).
 *
 * The contract is in kernel_w4a8.h.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_w4a8.h"

#include <stdint.h>

[[nodiscard]] float
w4a8_dot_scalar(size_t        n_blocks,
                const uint8_t weights[static n_blocks * W4A8_BLOCK_BYTES_WEIGHTS],
                const float   w_scales[static n_blocks],
                const float   w_offsets[static n_blocks],
                const int8_t  acts[static n_blocks * W4A8_BLOCK_ELEMS],
                const int32_t sum_a_per_block[static n_blocks],
                float         scale_x) {
    float acc = 0.0f;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *w_block = weights + b * W4A8_BLOCK_BYTES_WEIGHTS;
        const int8_t  *a_block = acts + b * W4A8_BLOCK_ELEMS;

        int32_t dot_uw_a = 0;
        for (size_t k = 0; k < W4A8_BLOCK_BYTES_WEIGHTS; k++) {
            const uint8_t byte = w_block[k];
            const int32_t lo   = (int32_t) (byte & 0x0Fu); /* unsigned 4-bit */
            const int32_t hi   = (int32_t) ((byte >> 4) & 0x0Fu);
            dot_uw_a += lo * (int32_t) a_block[2 * k + 0];
            dot_uw_a += hi * (int32_t) a_block[2 * k + 1];
        }

        const float block_term =
                w_scales[b] * (float) dot_uw_a - w_offsets[b] * (float) sum_a_per_block[b];
        acc += block_term;
    }
    return scale_x * acc;
}
