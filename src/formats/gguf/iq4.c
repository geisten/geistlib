/*
 * src/formats/gguf/iq4.c — IQ4_NL and IQ4_XS block dequantization.
 *
 * Pure file-format decoders: dequant_iq4_nl_row / dequant_iq4_xs_row
 * (block → fp32). Both formats store 4-bit indices into the same fixed
 * non-linear 16-value table (ggml's kvalues_iq4nl); IQ4_XS adds 6-bit
 * per-32 sub-scales on a 256-element super-block.
 */
#include "quant.h"
#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>

static const int8_t kvalues_iq4nl[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

void dequant_iq4_nl_row(const void *blocks, float *out, size_t n_elems) {
    const struct block_iq4_nl_t *b  = (const struct block_iq4_nl_t *) blocks;
    const size_t                 nb = n_elems / IQ4_NL_BLOCK_ELEMS;
    for (size_t i = 0; i < nb; i++) {
        const float    d  = fp16_to_fp32(b[i].d);
        const uint8_t *qs = b[i].qs;
        float         *y  = out + i * IQ4_NL_BLOCK_ELEMS;
        for (size_t j = 0; j < 16; j++) {
            y[j]      = d * (float) kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = d * (float) kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

void dequant_iq4_xs_row(const void *blocks, float *out, size_t n_elems) {
    const struct block_iq4_xs_t *b  = (const struct block_iq4_xs_t *) blocks;
    const size_t                 nb = n_elems / IQ4_XS_BLOCK_ELEMS;
    for (size_t i = 0; i < nb; i++) {
        const float    d  = fp16_to_fp32(b[i].d);
        const uint8_t *qs = b[i].qs;
        float         *y  = out + i * IQ4_XS_BLOCK_ELEMS;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            const int   ls = ((b[i].scales_l[ib32 / 2] >> (4 * (ib32 & 1))) & 0xf) |
                             (((b[i].scales_h >> (2 * ib32)) & 3) << 4);
            const float dl = d * (float) (ls - 32);
            for (size_t j = 0; j < 16; j++) {
                y[j]      = dl * (float) kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * (float) kvalues_iq4nl[qs[j] >> 4];
            }
            qs += 16;
            y += 32;
        }
    }
}
