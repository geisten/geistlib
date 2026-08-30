/*
 * src/formats/gguf/tq2_0.c — TQ2_0 dequant (BitNet ternary; W1.58A8 kernels live in
 * cpu_neon/weight_resolve.c).
 *
 * Layer: BACKEND. Extracted from gguf_quant.c during the per-quant
 * format split.
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

struct block_tq2_0_t {
    uint8_t qs[64];
    uint8_t d[2]; /* fp16, little-endian */
};
static_assert(sizeof(struct block_tq2_0_t) == 66, "TQ2_0 block must be 66 bytes");

void dequant_tq2_0_row(size_t n_elems, const void *blocks, float out[static n_elems]) {
    const struct block_tq2_0_t *b  = (const struct block_tq2_0_t *) blocks;
    const size_t                nb = n_elems / 256;
    for (size_t i = 0; i < nb; i++) {
        const uint16_t d_bits = (uint16_t) b[i].d[0] | ((uint16_t) b[i].d[1] << 8);
        const float    d      = fp16_to_fp32(d_bits);
        float         *y      = out + i * 256;
        for (size_t j = 0; j < 64; j += 32) {
            for (size_t l = 0; l < 4; l++) {
                for (size_t m = 0; m < 32; m++) {
                    const int trit = (int) ((b[i].qs[j + m] >> (l * 2)) & 3) - 1;
                    *y++           = d * (float) trit;
                }
            }
        }
    }
}
