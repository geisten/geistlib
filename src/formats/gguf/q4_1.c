/*
 * src/formats/gguf/q4_1.c — Q4_1 dequant (read-only; the type shows up
 * only for a handful of ffn_down tensors in unsloth "Q4_0" exports —
 * the generic dequant trampolines carry it, no dedicated hot path).
 *
 * Layer: BACKEND.
 */
#include "quant_blocks.h"
#include "quant.h"

#include <stdint.h>

struct block_q4_1_t {
    uint16_t d; /* scale (fp16) */
    uint16_t m; /* min   (fp16) */
    uint8_t  qs[16];
} __attribute__((packed));
_Static_assert(sizeof(struct block_q4_1_t) == 20, "struct block_q4_1_t size");

void dequant_q4_1_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_q4_1_t *b  = (const struct block_q4_1_t *) blocks;
    size_t                     nb = n_elems / Q4_1_BLOCK_ELEMS;
    for (size_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(b[i].d);
        const float m = fp16_to_fp32(b[i].m);
        float      *y = out + i * Q4_1_BLOCK_ELEMS;
        for (int j = 0; j < 16; j++) {
            const uint8_t bb = b[i].qs[j];
            y[j]             = d * (float) (bb & 0x0F) + m;
            y[j + 16]        = d * (float) (bb >> 4) + m;
        }
    }
}
