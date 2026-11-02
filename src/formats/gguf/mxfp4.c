/*
 * src/formats/gguf/mxfp4.c — MXFP4 block dequantization.
 *
 * Layer: BACKEND. Reference decoder for the format (GPT-OSS's experts):
 * the scalar reference, cpu_x86's generic linear, the NEON trampolines and
 * the embedding lookup go through it, and the native kernels are tested
 * against it. Mirrors dequantize_row_mxfp4 in ggml-quants.c.
 */
#include "quant_blocks.h"
#include "quant.h"

#include <stdint.h>

/* Twice the E2M1 values, so they are exact int8 (the scale halves them). */
const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

void dequant_mxfp4_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_mxfp4_t *b  = (const struct block_mxfp4_t *) blocks;
    const size_t                nb = n_elems / MXFP4_BLOCK_ELEMS;
    for (size_t i = 0; i < nb; i++) {
        const float    d  = mxfp4_e8m0_half(b[i].e);
        const uint8_t *qs = b[i].qs;
        float         *y  = out + i * MXFP4_BLOCK_ELEMS;
        for (size_t j = 0; j < 16; j++) {
            y[j]      = (float) kvalues_mxfp4[qs[j] & 0x0f] * d;
            y[j + 16] = (float) kvalues_mxfp4[qs[j] >> 4] * d;
        }
    }
}
