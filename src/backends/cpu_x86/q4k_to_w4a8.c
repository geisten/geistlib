/*
 * src/backends/cpu_x86/q4k_to_w4a8.c — Q4_K → W4A8 row predecoder.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Runs once per Q4_K weight at model-load. NOT in the inference hot path,
 * so the implementation is straightforward scalar C with no auto-vectorize
 * hints. Contract: q4k_to_w4a8.h.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "q4k_to_w4a8.h"

#include "quant.h"
#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>

void q4k_to_w4a8_row(size_t        n_in,
                     const uint8_t q4k_row[static(n_in / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES],
                     uint8_t weights[static(n_in / W4A8_BLOCK_ELEMS) * W4A8_BLOCK_BYTES_WEIGHTS],
                     float   w_scales[static n_in / W4A8_BLOCK_ELEMS],
                     float   w_offsets[static n_in / W4A8_BLOCK_ELEMS]) {
    const size_t               n_super = n_in / Q4_K_BLOCK_ELEMS;
    const struct block_q4_K_t *blocks  = (const struct block_q4_K_t *) q4k_row;

    /* Each Q4_K super-block (256 elements) → 8 W4A8 blocks (32 each). */
    for (size_t s = 0; s < n_super; s++) {
        const struct block_q4_K_t *blk  = &blocks[s];
        const float                d    = fp16_to_fp32(blk->d);
        const float                dmin = fp16_to_fp32(blk->dmin);

        /* 8 sub-blocks per super-block, paired (sb0+sb1 share 32 source
         * bytes — sb0 takes low nibbles, sb1 takes high nibbles). */
        for (size_t pair = 0; pair < 4; pair++) {
            const size_t   sb0_idx   = 2 * pair;
            const size_t   sb1_idx   = 2 * pair + 1;
            const uint8_t *q_segment = blk->qs + pair * 32;

            /* Scales / mins for the two sub-blocks of this pair. */
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4((int) sb0_idx, blk->scales, &sc0, &m0);
            get_scale_min_k4((int) sb1_idx, blk->scales, &sc1, &m1);

            /* Global W4A8 block indices for the two sub-blocks. */
            const size_t out_block0 = s * 8 + sb0_idx;
            const size_t out_block1 = s * 8 + sb1_idx;

            w_scales[out_block0]  = d * (float) sc0;
            w_offsets[out_block0] = dmin * (float) m0;
            w_scales[out_block1]  = d * (float) sc1;
            w_offsets[out_block1] = dmin * (float) m1;

            /* Repack nibbles:
             *   Q4_K sub-block 0: 32 elements take the low nibble of
             *     q_segment[l] for l in 0..31. Element index l.
             *   Q4_K sub-block 1: 32 elements take the high nibble of
             *     q_segment[l] for l in 0..31. Element index l.
             *
             *   W4A8 block: byte k holds elements 2k (lo) and 2k+1 (hi)
             *     of THIS block. So sub-block 0's W4A8 byte k is:
             *       lo = q_segment[2k] & 0x0F   (sb0 element 2k)
             *       hi = q_segment[2k+1] & 0x0F (sb0 element 2k+1)
             *     Sub-block 1's W4A8 byte k is:
             *       lo = q_segment[2k] >> 4     (sb1 element 2k)
             *       hi = q_segment[2k+1] >> 4   (sb1 element 2k+1)
             */
            uint8_t *out0_bytes = weights + out_block0 * W4A8_BLOCK_BYTES_WEIGHTS;
            uint8_t *out1_bytes = weights + out_block1 * W4A8_BLOCK_BYTES_WEIGHTS;
            for (size_t k = 0; k < W4A8_BLOCK_BYTES_WEIGHTS; k++) {
                const uint8_t a = q_segment[2 * k + 0];
                const uint8_t b = q_segment[2 * k + 1];
                out0_bytes[k]   = (uint8_t) ((a & 0x0Fu) | ((b & 0x0Fu) << 4));
                out1_bytes[k]   = (uint8_t) (((a >> 4) & 0x0Fu) | (b & 0xF0u));
            }
        }
    }
}
