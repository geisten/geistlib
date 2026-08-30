/*
 * src/formats/gguf/q8_0.c — Q8_0 block dequantization.
 *
 * Pure file-format decoder. W8A8 NEON kernels live in
 * src/backends/cpu_neon/kernels/q8_0.c.
 */
#include "quant_blocks.h"
#include "quant.h"

#include <stddef.h>
#include <stdint.h>

void dequant_q8_0_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_q8_0_t *b  = (const struct block_q8_0_t *) blocks;
    size_t                     nb = n_elems / Q8_0_BLOCK_ELEMS;
    for (size_t i = 0; i < nb; i++) {
        float d = fp16_to_fp32(b[i].d);
        for (int j = 0; j < 32; j++)
            out[i * 32 + j] = d * (float) b[i].qs[j];
    }
}
