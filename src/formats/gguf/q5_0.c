/*
 * src/formats/gguf/q5_0.c — Q5_0 block dequantization.
 *
 * Layer: BACKEND. Reference decoder for the format: no backend has a native
 * Q5_0 kernel, so the scalar reference, cpu_x86's generic linear, the NEON
 * trampolines and the embedding lookup all go through it. Mirrors
 * dequantize_row_q5_0 in ggml-quants.c.
 */
#include "quant_blocks.h"
#include "quant.h"

#include <stdint.h>

void dequant_q5_0_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_q5_0_t *b  = (const struct block_q5_0_t *) blocks;
    const size_t               nb = n_elems / Q5_0_BLOCK_ELEMS;
    for (size_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(b[i].d);
        /* Little-endian: element j's high bit is bit j. */
        const uint32_t qh = (uint32_t) b[i].qh[0] | ((uint32_t) b[i].qh[1] << 8) |
                            ((uint32_t) b[i].qh[2] << 16) | ((uint32_t) b[i].qh[3] << 24);
        float         *y  = out + i * Q5_0_BLOCK_ELEMS;
        for (unsigned j = 0; j < 16; j++) {
            const uint8_t bb = b[i].qs[j];
            const int     lo = (int) ((bb & 0x0Fu) | (((qh >> j) & 1u) << 4)) - 16;
            const int     hi = (int) ((bb >> 4) | (((qh >> (j + 16)) & 1u) << 4)) - 16;
            y[j]             = d * (float) lo;
            y[j + 16]        = d * (float) hi;
        }
    }
}
