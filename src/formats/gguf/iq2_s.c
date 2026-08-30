/*
 * src/formats/gguf/iq2_s.c — IQ2_S block dequantization.
 *
 * Pure file-format decoder: dequant_iq2_s_row (block → fp32).
 *
 * The W2A8 NEON kernels live in src/backends/cpu_neon/kernels/iq2_s.c.
 */
#include "quant_blocks.h"
#include "quant.h"
#include "iq_grids.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static void dequant_iq2_s_block(const struct block_iq2_s_t *blk, float *y) {
    const float    d     = fp16_to_fp32(blk->d);
    const uint8_t *qs    = blk->qs;
    const uint8_t *qh    = blk->qh;
    const uint8_t *signs = qs + 32; /* signs occupy second half of qs region */
    for (int ib32 = 0; ib32 < 8; ib32++) {
        float db[2];
        db[0] = d * (0.5f + (blk->scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (blk->scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const float    dl   = db[l / 2];
            const uint16_t idx  = (uint16_t) qs[l] | ((uint16_t) (qh[ib32] << (8 - 2 * l)) & 0x300);
            const uint8_t *grid = (const uint8_t *) (iq2s_grid + idx);
            for (int j = 0; j < 8; j++) {
                y[j] = dl * (float) grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
}

void dequant_iq2_s_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_iq2_s_t *b  = (const struct block_iq2_s_t *) blocks;
    const size_t                nb = n_elems / 256;
    for (size_t i = 0; i < nb; i++)
        dequant_iq2_s_block(&b[i], out + i * 256);
}
