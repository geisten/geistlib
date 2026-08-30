/*
 * src/formats/gguf/q6_K.c — Q6_K block dequantization.
 *
 * Pure file-format decoder. W6A8 NEON kernels live in
 * src/backends/cpu_neon/kernels/q6_K.c.
 */
#include "quant_blocks.h"
#include "quant.h"

#include <stddef.h>
#include <stdint.h>

void dequant_q6_K_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_q6_K_t *b  = (const struct block_q6_K_t *) blocks;
    size_t                     nb = n_elems / Q6_K_BLOCK_ELEMS;

    for (size_t i = 0; i < nb; i++) {
        const float    d  = fp16_to_fp32(b[i].d);
        const uint8_t *ql = b[i].ql;
        const uint8_t *qh = b[i].qh;
        const int8_t  *sc = b[i].scales;
        float         *y  = out + i * Q6_K_BLOCK_ELEMS;

        /* Q6_K is processed in two 128-element halves. y, ql, qh, sc all
         * advance after each half — do NOT add the outer offset to y[]. */
        for (size_t half = 0; half < Q6_K_BLOCK_ELEMS; half += 128) {
            (void) half;
            for (int l = 0; l < 32; l++) {
                int    is = l / 16;
                int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * (float) sc[is + 0] * (float) q1;
                y[l + 32] = d * (float) sc[is + 2] * (float) q2;
                y[l + 64] = d * (float) sc[is + 4] * (float) q3;
                y[l + 96] = d * (float) sc[is + 6] * (float) q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
            y += 128;
        }
    }
}
