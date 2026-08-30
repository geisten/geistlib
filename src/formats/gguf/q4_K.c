/*
 * src/formats/gguf/q4_K.c — Q4_K block dequantization.
 *
 * Pure file-format decoder: reads on-disk Q4_K super-blocks
 * (144 bytes / 256 elements) and produces FP32 output. Backend-
 * agnostic; the W4A8 NEON kernels that consume the same block
 * layout live in src/backends/cpu_neon/kernels/q4_K.c.
 *
 * The struct block_q4_K_t lives in internal.h so the NEON kernels
 * can also see the layout.
 */
#include "quant_blocks.h"
#include "quant.h"

#include <stddef.h>
#include <stdint.h>

void dequant_q4_K_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_q4_K_t *b  = (const struct block_q4_K_t *) blocks;
    size_t                     nb = n_elems / Q4_K_BLOCK_ELEMS;

    /* Note: gcc -O3 -march=native auto-vectorizes the inner loops well. A
     * hand-rolled NEON path benchmarked 2× slower on Pi 5 (Cortex-A76) due
     * to register pressure / spill — leave the auto-vectorizer to it. */
    for (size_t i = 0; i < nb; i++) {
        const float    d    = fp16_to_fp32(b[i].d);
        const float    dmin = fp16_to_fp32(b[i].dmin);
        const uint8_t *q    = b[i].qs;
        float         *y    = out + i * Q4_K_BLOCK_ELEMS;

        int     is = 0;
        uint8_t sc, m;
        for (size_t sb = 0; sb < Q4_K_BLOCK_ELEMS; sb += 64) {
            get_scale_min_k4(is + 0, b[i].scales, &sc, &m);
            const float d1 = d * (float) sc;
            const float m1 = dmin * (float) m;
            get_scale_min_k4(is + 1, b[i].scales, &sc, &m);
            const float d2 = d * (float) sc;
            const float m2 = dmin * (float) m;
            for (int l = 0; l < 32; l++)
                y[sb + l] = d1 * (float) (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++)
                y[sb + 32 + l] = d2 * (float) (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}
