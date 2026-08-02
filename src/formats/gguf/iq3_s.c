/*
 * src/formats/gguf/iq3_s.c — IQ3_S block dequantization.
 *
 * Pure file-format decoder: dequant_iq3_s_row (block → fp32).
 *
 * W3A8 NEON kernels live in src/backends/cpu_neon/kernels/iq3_s.c.
 */
#include "quant_blocks.h"
#include "quant.h"
#include "iq_grids.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static void dequant_iq3_s_block(const struct block_iq3_s_t *blk, float *y) {
    const float    d     = fp16_to_fp32(blk->d);
    const uint8_t *qs    = blk->qs;
    const uint8_t *qh    = blk->qh;
    const uint8_t *signs = blk->signs;
    /* Each ib32 covers 32 elements. Two halves of 4 super-blocks each share scale[ib]. */
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const int   ib  = ib32 / 2; /* scale index 0..3 */
        const float db1 = d * (1.0f + 2 * ((blk->scales[ib] >> 0) & 0xf));
        const float db2 = d * (1.0f + 2 * ((blk->scales[ib] >> 4) & 0xf));
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid1 =
                    (const uint8_t *) (iq3s_grid +
                                       ((uint32_t) qs[2 * l + 0] |
                                        ((uint32_t) (qh[ib32 + 0] << (8 - 2 * l)) & 256)));
            const uint8_t *grid2 =
                    (const uint8_t *) (iq3s_grid +
                                       ((uint32_t) qs[2 * l + 1] |
                                        ((uint32_t) (qh[ib32 + 0] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; j++) {
                y[j + 0] =
                        db1 * (float) grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                y[j + 4] =
                        db1 * (float) grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid1 =
                    (const uint8_t *) (iq3s_grid +
                                       ((uint32_t) qs[2 * l + 0] |
                                        ((uint32_t) (qh[ib32 + 1] << (8 - 2 * l)) & 256)));
            const uint8_t *grid2 =
                    (const uint8_t *) (iq3s_grid +
                                       ((uint32_t) qs[2 * l + 1] |
                                        ((uint32_t) (qh[ib32 + 1] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; j++) {
                y[j + 0] =
                        db2 * (float) grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                y[j + 4] =
                        db2 * (float) grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
    }
}

void dequant_iq3_s_row(const void *blocks, float *out, size_t n_elems) {
    const struct block_iq3_s_t *b  = (const struct block_iq3_s_t *) blocks;
    const size_t                nb = n_elems / 256;
    for (size_t i = 0; i < nb; i++)
        dequant_iq3_s_block(&b[i], out + i * 256);
}
