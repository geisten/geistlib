/*
 * src/formats/gguf/pq2_0.c — PQ2_0 dequant (PrismML ternary, group 128;
 * the W2A8 decode kernel lives in cpu_neon/kernels/pq2_0.c).
 *
 * Layer: BACKEND. Reference decoder for the format: the scalar backend,
 * the NEON trampolines and the embedding lookup all go through it, and
 * the NEON kernel is tested against it. Mirrors dequantize_row_pq2_0 in
 * PrismML-Eng/llama.cpp ggml-quants.c.
 */
#include "quant.h"

#include <stdint.h>

void dequant_pq2_0_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const uint8_t *b  = (const uint8_t *) blocks;
    const size_t   nb = n_elems / PQ2_0_BLOCK_ELEMS;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *blk = b + i * PQ2_0_BLOCK_BYTES;
        const float    d   = fp16_to_fp32((uint16_t) blk[0] | ((uint16_t) blk[1] << 8));
        const uint8_t *qs  = blk + 2;
        float         *y   = out + i * PQ2_0_BLOCK_ELEMS;
        for (size_t j = 0; j < PQ2_0_BLOCK_ELEMS; j++) {
            const int q = (qs[j / 4] >> (2 * (j % 4))) & 3;
            y[j]        = (float) (q - 1) * d;
        }
    }
}
