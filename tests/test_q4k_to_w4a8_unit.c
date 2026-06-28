/*
 * test_q4k_to_w4a8_unit — Q4_K → W4A8 row predecoder correctness.
 *
 * The reconstruction
 *
 *   y[i] = w_scales[block(i)] * u_w[i] - w_offsets[block(i)]
 *
 * over the predecoded W4A8 row MUST match `dequant_q4_K_row` element-for-
 * element within fp32 single-precision rounding noise. Tested over:
 *   1. A single hand-crafted Q4_K super-block with structured d/dmin
 *      values and a deterministic nibble pattern.
 *   2. Two consecutive super-blocks of PRNG-generated nibbles, scales,
 *      mins (typical real-weight conditions).
 *
 * Deterministic; runs in <50 ms; no model, no fixture files.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/kernel_w4a8.h"
#include "../src/backends/cpu_x86/q4k_to_w4a8.h"
#include "quant.h"
#include "quant_blocks.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* fp32 → fp16 via the standard bit-cast path. Used to build hand-crafted
 * Q4_K super-blocks with chosen d/dmin values. */
static uint16_t fp32_to_fp16(float f) {
    /* IEEE 754 binary32 → binary16, round-to-nearest-even. Sufficient for
     * test fixtures; perf doesn't matter here. */
    union {
        float    f;
        uint32_t u;
    } bits            = {.f = f};
    uint32_t x        = bits.u;
    uint32_t sign     = (x >> 31) & 0x1u;
    int32_t  exp32    = (int32_t) ((x >> 23) & 0xFFu) - 127;
    uint32_t mant32   = x & 0x7FFFFFu;
    uint16_t out_sign = (uint16_t) (sign << 15);

    if (exp32 == 128) {
        /* inf / nan */
        return (uint16_t) (out_sign | 0x7C00u | (mant32 != 0));
    }
    if (exp32 < -14) {
        /* underflow → zero (no subnormal-fp16 handling — fixtures avoid it) */
        return out_sign;
    }
    if (exp32 > 15) {
        return (uint16_t) (out_sign | 0x7C00u); /* overflow → ±inf */
    }
    uint16_t exp16 = (uint16_t) ((exp32 + 15) & 0x1Fu);
    /* Round-to-nearest-even on the dropped 13 mantissa bits. */
    uint32_t mant16    = mant32 >> 13;
    uint32_t round_bit = (mant32 >> 12) & 1u;
    uint32_t sticky    = mant32 & 0xFFFu;
    if (round_bit != 0 && (sticky != 0 || (mant16 & 1u) != 0)) {
        mant16 += 1;
        if (mant16 == 0x400u) {
            mant16 = 0;
            exp16  = (uint16_t) (exp16 + 1u);
        }
    }
    return (uint16_t) (out_sign | (uint16_t) (exp16 << 10) | (uint16_t) mant16);
}

/* SplitMix32 PRNG — deterministic. */
static uint32_t prng_next(uint32_t *state) {
    uint32_t z = (*state += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

/* Pack two 6-bit values (scale + min) into Q4_K's 12-byte scales array per
 * the layout used by get_scale_min_k4. Sub-block index j ∈ [0, 8). */
static void set_scale_min_k4(int j, uint8_t scales[12], uint8_t d_in, uint8_t m_in) {
    /* Inverse of get_scale_min_k4 in src/quant/quant_blocks.h. */
    if (j < 4) {
        scales[j]     = (uint8_t) ((scales[j] & 0xC0u) | (d_in & 0x3Fu));
        scales[j + 4] = (uint8_t) ((scales[j + 4] & 0xC0u) | (m_in & 0x3Fu));
    } else {
        /* d_out = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
         * m_out = (q[j+4] >> 4) | ((q[j  ] >> 6) << 4) */
        const uint8_t d_lo4 = d_in & 0x0Fu;
        const uint8_t d_hi2 = (d_in >> 4) & 0x03u;
        const uint8_t m_lo4 = m_in & 0x0Fu;
        const uint8_t m_hi2 = (m_in >> 4) & 0x03u;
        scales[j + 4]       = (uint8_t) ((scales[j + 4] & 0x00u) | d_lo4 | (m_lo4 << 4));
        scales[j - 4]       = (uint8_t) ((scales[j - 4] & 0x3Fu) | (d_hi2 << 6));
        scales[j]           = (uint8_t) ((scales[j] & 0x3Fu) | (m_hi2 << 6));
    }
}

/* Reconstruct y[i] from the W4A8 predecoded layout. Mirrors the contract
 * documented in kernel_w4a8.h. */
static void reconstruct_from_w4a8_row(size_t        n_in,
                                      const uint8_t weights[static n_in / 2],
                                      const float   w_scales[static n_in / W4A8_BLOCK_ELEMS],
                                      const float   w_offsets[static n_in / W4A8_BLOCK_ELEMS],
                                      float         out[static n_in]) {
    const size_t n_block = n_in / W4A8_BLOCK_ELEMS;
    for (size_t b = 0; b < n_block; b++) {
        const float    s   = w_scales[b];
        const float    o   = w_offsets[b];
        const uint8_t *blk = weights + b * W4A8_BLOCK_BYTES_WEIGHTS;
        for (size_t k = 0; k < W4A8_BLOCK_BYTES_WEIGHTS; k++) {
            const uint8_t byte                    = blk[k];
            const float   lo                      = (float) (byte & 0x0Fu);
            const float   hi                      = (float) ((byte >> 4) & 0x0Fu);
            out[b * W4A8_BLOCK_ELEMS + 2 * k + 0] = s * lo - o;
            out[b * W4A8_BLOCK_ELEMS + 2 * k + 1] = s * hi - o;
        }
    }
}

static int
compare_rows(const char *label, size_t n, const float a[static n], const float b[static n]) {
    const float atol = 1e-5f;
    const float rtol = 1e-5f;
    for (size_t i = 0; i < n; i++) {
        const float diff = fabsf(a[i] - b[i]);
        const float tol  = atol + rtol * fabsf(a[i]);
        if (diff > tol) {
            fprintf(stderr,
                    "%s: mismatch at i=%zu ref=%g w4a8=%g diff=%g tol=%g\n",
                    label,
                    i,
                    (double) a[i],
                    (double) b[i],
                    (double) diff,
                    (double) tol);
            return 1;
        }
    }
    return 0;
}

/* ------------------------- Scenario 1: hand-crafted ----------------------- */

static int scenario_handcrafted_super_block(void) {
    /* One Q4_K super-block (256 elements). */
    struct block_q4_K_t blk = {0};
    blk.d                   = fp32_to_fp16(0.0125f);
    blk.dmin                = fp32_to_fp16(0.075f);
    /* Fill 8 sub-blocks with structured (scale, min) pairs. */
    for (int sb = 0; sb < 8; sb++) {
        set_scale_min_k4(
                sb, blk.scales, (uint8_t) ((sb * 5 + 1) & 0x3Fu), (uint8_t) ((sb * 3 + 2) & 0x3Fu));
    }
    /* Deterministic nibbles: pos-encoded so dequant has visible structure. */
    for (size_t k = 0; k < 128; k++) {
        blk.qs[k] = (uint8_t) ((k & 0x0Fu) | (((k >> 4) & 0x0Fu) << 4));
    }

    float ref[256];
    dequant_q4_K_row(&blk, ref, 256);

    uint8_t w_w4a8[128]; /* 256 / 2 */
    float   w_scales[8]; /* 256 / 32 */
    float   w_offsets[8];
    q4k_to_w4a8_row(256, (const uint8_t *) &blk, w_w4a8, w_scales, w_offsets);

    float recon[256];
    reconstruct_from_w4a8_row(256, w_w4a8, w_scales, w_offsets, recon);

    return compare_rows("handcrafted", 256, ref, recon);
}

/* ------------------------- Scenario 2: random multi-super-block ---------- */

static int scenario_random_multi_super_block(void) {
    /* Two Q4_K super-blocks → 512 elements, 16 W4A8 blocks. */
    constexpr size_t N_SUPER = 2;
    constexpr size_t N_IN    = N_SUPER * 256;

    struct block_q4_K_t blocks[N_SUPER] = {0};
    uint32_t            s               = 0x4242BEEFu;
    for (size_t b = 0; b < N_SUPER; b++) {
        /* d, dmin in a realistic Q4_K_M range. */
        const uint32_t db = prng_next(&s);
        const uint32_t mb = prng_next(&s);
        const float    df = (float) ((db & 0xFFFFu) + 1u) * 0.001f / 65536.0f;
        const float    mf = (float) ((mb & 0xFFFFu) + 1u) * 0.05f / 65536.0f;
        blocks[b].d       = fp32_to_fp16(df * 8.0f); /* ~ 0.001..0.008 */
        blocks[b].dmin    = fp32_to_fp16(mf);        /* ~ 0..0.05 */
        for (int sb = 0; sb < 8; sb++) {
            const uint32_t sc_bits = prng_next(&s);
            const uint32_t mn_bits = prng_next(&s);
            set_scale_min_k4(
                    sb, blocks[b].scales, (uint8_t) (sc_bits & 0x3Fu), (uint8_t) (mn_bits & 0x3Fu));
        }
        for (size_t k = 0; k < 128; k++) {
            const uint32_t bits = prng_next(&s);
            blocks[b].qs[k]     = (uint8_t) (bits & 0xFFu);
        }
    }

    float ref[N_IN];
    dequant_q4_K_row(blocks, ref, N_IN);

    uint8_t w_w4a8[N_IN / 2];
    float   w_scales[N_IN / W4A8_BLOCK_ELEMS];
    float   w_offsets[N_IN / W4A8_BLOCK_ELEMS];
    q4k_to_w4a8_row(N_IN, (const uint8_t *) blocks, w_w4a8, w_scales, w_offsets);

    float recon[N_IN];
    reconstruct_from_w4a8_row(N_IN, w_w4a8, w_scales, w_offsets, recon);

    return compare_rows("random_multi", N_IN, ref, recon);
}

/* --------------------------------- main ---------------------------------- */

int main(void) {
    int fails = 0;
    if (scenario_handcrafted_super_block() != 0) {
        fputs("scenario_handcrafted_super_block FAILED\n", stderr);
        fails++;
    }
    if (scenario_random_multi_super_block() != 0) {
        fputs("scenario_random_multi_super_block FAILED\n", stderr);
        fails++;
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
