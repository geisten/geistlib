/*
 * src/formats/gguf/q5_K.c — Q5_K block dequantization.
 *
 * Pure file-format decoder. The W5A8 NEON kernels live in
 * src/backends/cpu_neon/kernels/q5_K.c. struct block_q5_K_t lives in
 * internal.h so both can share it.
 */
#include "quant_blocks.h"
#include "quant.h"

#include <stddef.h>
#include <stdint.h>

void dequant_q5_K_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_q5_K_t *b  = (const struct block_q5_K_t *) blocks;
    const size_t               nb = n_elems / Q5_K_BLOCK_ELEMS;

    for (size_t i = 0; i < nb; i++) {
        const float    d    = fp16_to_fp32(b[i].d);
        const float    dmin = fp16_to_fp32(b[i].dmin);
        const uint8_t *ql   = b[i].qs;
        const uint8_t *qh   = b[i].qh;
        float         *y    = out + i * Q5_K_BLOCK_ELEMS;

        int     is = 0;
        uint8_t u1 = 1, u2 = 2;
        uint8_t sc, mn;
        for (size_t sb = 0; sb < Q5_K_BLOCK_ELEMS; sb += 64) {
            get_scale_min_k4(is + 0, b[i].scales, &sc, &mn);
            const float d1 = d * (float) sc;
            const float m1 = dmin * (float) mn;
            get_scale_min_k4(is + 1, b[i].scales, &sc, &mn);
            const float d2 = d * (float) sc;
            const float m2 = dmin * (float) mn;
            for (int l = 0; l < 32; l++) {
                const int q = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                y[sb + l]   = d1 * (float) q - m1;
            }
            for (int l = 0; l < 32; l++) {
                const int q    = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                y[sb + 32 + l] = d2 * (float) q - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}
