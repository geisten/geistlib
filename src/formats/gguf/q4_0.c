/*
 * src/formats/gguf/q4_0.c — Q4_0 block dequantization.
 *
 * Layer: BACKEND.
 */
#include "quant_blocks.h"
#include "quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void dequant_q4_0_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_q4_0_t *b  = (const struct block_q4_0_t *) blocks;
    size_t                     nb = n_elems / Q4_0_BLOCK_ELEMS;
    for (size_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(b[i].d);
        float      *y = out + i * Q4_0_BLOCK_ELEMS;
        for (int j = 0; j < 16; j++) {
            const uint8_t bb = b[i].qs[j];
            const int     lo = (int) (bb & 0x0F) - 8;
            const int     hi = (int) (bb >> 4) - 8;
            y[j]             = d * (float) lo;
            y[j + 16]        = d * (float) hi;
        }
    }
}
