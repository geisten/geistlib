/*
 * src/backends/cpu_x86/q6k_to_w8a8.c — Q6_K → W8A8 row predecoder.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * One pass over the row at model load. The loop structure mirrors
 * dequant_q6_K_row in src/formats/gguf/q6_K.c byte-for-byte; we just
 * write u_w to memory instead of fp32 y. After the per-super-block
 * pass, w_scales[s*16 + k] and w_offsets[s*16 + k] are filled from
 * blk->scales[k] and the super-block d.
 *
 * The element-to-sub-block mapping is verified in the unit test
 * (test_q6k_to_w8a8_unit): reconstruct y[i] = w_scales[k]*u_w[i] -
 * w_offsets[k] and compare element-wise to dequant_q6_K_row.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "q6k_to_w8a8.h"

#include "quant.h"
#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>

void q6k_to_w8a8_row(size_t        n_in,
                     const uint8_t q6k_row[static(n_in / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES],
                     uint8_t       weights[static(n_in / W8A8_BLOCK_ELEMS) * W8A8_BLOCK_ELEMS],
                     float         w_scales[static n_in / W8A8_BLOCK_ELEMS],
                     float         w_offsets[static n_in / W8A8_BLOCK_ELEMS]) {
    const size_t               n_super = n_in / Q6_K_BLOCK_ELEMS;
    const struct block_q6_K_t *blocks  = (const struct block_q6_K_t *) q6k_row;

    for (size_t s = 0; s < n_super; s++) {
        const struct block_q6_K_t *blk       = &blocks[s];
        const float                d         = fp16_to_fp32(blk->d);
        const uint8_t             *ql        = blk->ql;
        const uint8_t             *qh        = blk->qh;
        uint8_t                   *u_w_super = weights + s * Q6_K_BLOCK_ELEMS;

        /* Two 128-element halves, mirror of dequant_q6_K_row. */
        for (size_t half = 0; half < 2; half++) {
            uint8_t *u_w_half = u_w_super + half * 128;
            for (int l = 0; l < 32; l++) {
                const uint8_t q1 = (uint8_t) ((ql[l + 0] & 0x0Fu) | (((qh[l] >> 0) & 0x03u) << 4));
                const uint8_t q2 = (uint8_t) ((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 0x03u) << 4));
                const uint8_t q3 =
                        (uint8_t) (((ql[l + 0] >> 4) & 0x0Fu) | (((qh[l] >> 4) & 0x03u) << 4));
                const uint8_t q4 =
                        (uint8_t) (((ql[l + 32] >> 4) & 0x0Fu) | (((qh[l] >> 6) & 0x03u) << 4));
                u_w_half[l + 0]  = q1;
                u_w_half[l + 32] = q2;
                u_w_half[l + 64] = q3;
                u_w_half[l + 96] = q4;
            }
            ql += 64;
            qh += 32;
        }

        /* Per-sub-block scale/offset. sub-block k covers u_w_super[k*16
         * .. k*16+15] with effective scale d * blk->scales[k]. */
        for (size_t k = 0; k < 16; k++) {
            const float ws        = d * (float) blk->scales[k];
            w_scales[s * 16 + k]  = ws;
            w_offsets[s * 16 + k] = ws * 32.0f;
        }
    }
}
