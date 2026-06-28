/*
 * src/backends/cpu_x86/q8_kx4.c — Q8_Kx4 activation quantizer.
 *
 * Per-row symmetric int8 quantization with the interleaved 8-byte-stripe
 * layout the lane-parallel GEMM kernel expects. Plus precomputed
 * 16-element-sub-block sums (bsums) so the Q4_K min-term can be applied
 * once per super-block in the inner kernel.
 *
 * Pure C23 (no intrinsics). Adapted from the layout described in
 * llama.cpp's repack.h block_q8_Kx4 + repack.cpp quantization helpers.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "q8_kx4.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

void quantize_q8_Kx4(size_t n_in, const float x_rows[static 4 * n_in], struct block_q8_Kx4 *out) {
    const size_t n_super = n_in / 256;

    /* Per row: find max-abs, derive scale_x = max|x| / 127. */
    float scale_x[4];
    for (int r = 0; r < 4; r++) {
        float amax = 0.0f;
        for (size_t k = 0; k < n_in; k++) {
            const float a = fabsf(x_rows[(size_t) r * n_in + k]);
            if (a > amax) {
                amax = a;
            }
        }
        scale_x[r] = (amax == 0.0f) ? 1.0f : (amax / 127.0f);
    }

    /* Per super-block: quantize + interleave + compute bsums.
     *
     * qs layout per super-block (offset within out[s].qs[]):
     *   - 4 sub-blocks of 64 K-elements each (sb = 0..3)
     *   - Per sb: 256 bytes (4 rows × 64 elements)
     *     organized as 8 stripes × 32 bytes; stripe i bytes:
     *       0..7    row 0 elements (8*i..8*i+7) within the sub-block
     *       8..15   row 1
     *       16..23  row 2
     *       24..31  row 3
     */
    for (size_t s = 0; s < n_super; s++) {
        struct block_q8_Kx4 *b = &out[s];
        for (int r = 0; r < 4; r++) {
            b->d[r] = scale_x[r];
        }

        for (int sb = 0; sb < 4; sb++) {
            uint8_t *sb_dst = (uint8_t *) (b->qs + (size_t) sb * 256);
            for (int stripe = 0; stripe < 8; stripe++) {
                uint8_t *stripe_dst = sb_dst + (size_t) stripe * 32;
                for (int r = 0; r < 4; r++) {
                    int8_t      *row_dst = (int8_t *) (stripe_dst + r * 8);
                    const float *row_src = x_rows + (size_t) r * n_in + s * 256 + (size_t) sb * 64 +
                                           (size_t) stripe * 8;
                    const float  inv     = 1.0f / scale_x[r];
                    for (int i = 0; i < 8; i++) {
                        row_dst[i] = (int8_t) lrintf(row_src[i] * inv);
                    }
                }
            }
        }

        /* bsums[r*16 + g] = sum of 16 quantized elements for row r,
         * sub-block 16-elem group g. There are 16 groups of 16 per row
         * (256 / 16 = 16). */
        for (int r = 0; r < 4; r++) {
            for (int g = 0; g < 16; g++) {
                int32_t sum = 0;
                /* Group g covers K positions g*16 .. g*16+15 within the
                 * super-block. With our 8-byte-stripe layout, K positions
                 * g*16..g*16+15 are split across two 8-byte stripes in the
                 * same sub-block (sb = g / 4). */
                const int sb     = g / 4;
                const int sb_pos = (g % 4) * 16; /* K-offset within sb */
                /* sb_pos covers stripes [sb_pos/8, sb_pos/8 + 2) i.e. two
                 * consecutive 8-byte stripes within sub-block sb. */
                const uint8_t *sb_src = (const uint8_t *) (b->qs + (size_t) sb * 256);
                for (int half = 0; half < 2; half++) {
                    const int     stripe_idx = sb_pos / 8 + half;
                    const int8_t *row_stripe =
                            (const int8_t *) (sb_src + (size_t) stripe_idx * 32 + r * 8);
                    for (int i = 0; i < 8; i++) {
                        sum += row_stripe[i];
                    }
                }
                b->bsums[r * 16 + g] = (int16_t) sum;
            }
        }
    }
}
