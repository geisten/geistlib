/*
 * test_q4k_to_q4kx8_unit — Q4_K → Q4_Kx8 repack basic correctness.
 *
 * Verifies the byte-level repack preserves d/dmin per row and that the
 * 8-byte interleaved qs stripes contain the right source bytes. Full
 * GEMM correctness lands in the next test once the inner kernel exists.
 *
 * Deterministic; runs in <50 ms.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/q4k_to_q4kx8.h"
#include "quant.h"
#include "quant_blocks.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t prng_next(uint32_t *state) {
    uint32_t z = (*state += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static int scenario_d_dmin_roundtrip(void) {
    /* 8 source rows × 1 super-block (256 elements). */
    constexpr size_t    N_SUPER = 1;
    struct block_q4_K_t src[8 * N_SUPER];
    uint32_t            s = 0xCAFEBABEu;
    for (size_t r = 0; r < 8 * N_SUPER; r++) {
        src[r].d    = (uint16_t) (prng_next(&s) & 0xFFFFu);
        src[r].dmin = (uint16_t) (prng_next(&s) & 0xFFFFu);
        for (size_t k = 0; k < 12; k++) {
            src[r].scales[k] = (uint8_t) (prng_next(&s) & 0xFFu);
        }
        for (size_t k = 0; k < 128; k++) {
            src[r].qs[k] = (uint8_t) (prng_next(&s) & 0xFFu);
        }
    }

    struct block_q4_Kx8 out[N_SUPER];
    q4k_to_q4kx8_octet(N_SUPER, (const uint8_t *) src, out);

    /* d / dmin must be preserved per row. */
    for (int r = 0; r < 8; r++) {
        if (out[0].d[r] != src[r].d) {
            fprintf(stderr, "d[%d]: out=%u src=%u\n", r, out[0].d[r], src[r].d);
            return 1;
        }
        if (out[0].dmin[r] != src[r].dmin) {
            fprintf(stderr, "dmin[%d]: out=%u src=%u\n", r, out[0].dmin[r], src[r].dmin);
            return 1;
        }
    }

    /* qs[] interleaved in 8-byte stripes. Stripe i comes from
     * src[i % 8].qs[(i / 8) * 8 .. (i / 8) * 8 + 7]. */
    const int n_stripes = Q4_K_BLOCK_ELEMS * 4 / 8; /* = 128 */
    for (int i = 0; i < n_stripes; i++) {
        const int src_row = i % 8;
        const int src_off = (i / 8) * 8;
        const int dst_off = i * 8;
        for (int b = 0; b < 8; b++) {
            const uint8_t expected = src[src_row].qs[src_off + b];
            const uint8_t got      = out[0].qs[dst_off + b];
            if (expected != got) {
                fprintf(stderr, "qs stripe %d byte %d: out=%u src=%u\n", i, b, got, expected);
                return 1;
            }
        }
    }
    return 0;
}

int main(void) {
    int fails = 0;
    if (scenario_d_dmin_roundtrip() != 0) {
        fputs("scenario_d_dmin_roundtrip FAILED\n", stderr);
        fails++;
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
