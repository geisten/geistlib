/*
 * tests/quant_fixtures.h — random but finite weights in every raw layout
 * geist_linear_ref decodes, for tests that run the linear kernels on them.
 *
 * DTYPES lists each dtype with its block geometry, its quant.h row codec
 * (nullptr for F32, F16, BF16 and I2_S, which a test decodes itself) and
 * the byte offsets of its fp16 scale fields, which make_weight pins to 0.5
 * so no block decodes to inf or NaN. Values come from a fixed xorshift
 * stream: the same calls give the same weights.
 */
#ifndef GEIST_TESTS_QUANT_FIXTURES_H
#define GEIST_TESTS_QUANT_FIXTURES_H

#include <geist_types.h>

#include "quant.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef void (*row_fn)(size_t n, const void *blocks, float *out);

static const struct {
    enum geist_dtype dt;
    const char      *name;
    size_t           blk, bytes; /* elements, bytes per block */
    row_fn           row;        /* nullptr: decoded below */
    int              d[2];       /* byte offsets of fp16 scale fields, -1 none */
} DTYPES[] = {
        {GEIST_DTYPE_F32, "F32", 1, 4, nullptr, {-1, -1}},
        {GEIST_DTYPE_F16, "F16", 1, 2, nullptr, {-1, -1}},
        {GEIST_DTYPE_BF16, "BF16", 1, 2, nullptr, {-1, -1}},
        {GEIST_DTYPE_Q4_0, "Q4_0", Q4_0_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, dequant_q4_0_row, {0, -1}},
        {GEIST_DTYPE_Q4_1, "Q4_1", Q4_1_BLOCK_ELEMS, Q4_1_BLOCK_BYTES, dequant_q4_1_row, {0, 2}},
        {GEIST_DTYPE_Q8_0, "Q8_0", Q8_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, dequant_q8_0_row, {0, -1}},
        {GEIST_DTYPE_Q3_K, "Q3_K", Q3_K_BLOCK_ELEMS, Q3_K_BLOCK_BYTES, dequant_q3_K_row, {108, -1}},
        {GEIST_DTYPE_Q4_K, "Q4_K", Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, dequant_q4_K_row, {0, 2}},
        {GEIST_DTYPE_Q5_K, "Q5_K", Q5_K_BLOCK_ELEMS, Q5_K_BLOCK_BYTES, dequant_q5_K_row, {0, 2}},
        {GEIST_DTYPE_Q6_K, "Q6_K", Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES, dequant_q6_K_row, {208, -1}},
        {GEIST_DTYPE_IQ2_S,
         "IQ2_S",
         IQ2_S_BLOCK_ELEMS,
         IQ2_S_BLOCK_BYTES,
         dequant_iq2_s_row,
         {0, -1}},
        {GEIST_DTYPE_IQ3_S,
         "IQ3_S",
         IQ3_S_BLOCK_ELEMS,
         IQ3_S_BLOCK_BYTES,
         dequant_iq3_s_row,
         {0, -1}},
        {GEIST_DTYPE_IQ4_NL,
         "IQ4_NL",
         IQ4_NL_BLOCK_ELEMS,
         IQ4_NL_BLOCK_BYTES,
         dequant_iq4_nl_row,
         {0, -1}},
        {GEIST_DTYPE_IQ4_XS,
         "IQ4_XS",
         IQ4_XS_BLOCK_ELEMS,
         IQ4_XS_BLOCK_BYTES,
         dequant_iq4_xs_row,
         {0, -1}},
        {GEIST_DTYPE_TQ2_0,
         "TQ2_0",
         TQ2_0_BLOCK_ELEMS,
         TQ2_0_BLOCK_BYTES,
         dequant_tq2_0_row,
         {64, -1}},
        {GEIST_DTYPE_PQ2_0,
         "PQ2_0",
         PQ2_0_BLOCK_ELEMS,
         PQ2_0_BLOCK_BYTES,
         dequant_pq2_0_row,
         {0, -1}},
        {GEIST_DTYPE_I2_S, "I2_S", I2_S_BLOCK_ELEMS, I2_S_BLOCK_BYTES, nullptr, {-1, -1}},
};
enum { N_DTYPES = sizeof DTYPES / sizeof DTYPES[0] };

static uint32_t        rng = 0x9e3779b9u;
static inline uint32_t next_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
static inline float next_f(void) {
    return ((float) (next_u32() >> 8) / (float) (1u << 24) - 0.5f) * 4.0f;
}

/* Random but finite storage for [n_out, n_in] of DTYPES[d]; nullptr on OOM
 * or a shape that is not whole blocks. *bytes gets its size, tail
 * included. The caller frees it. */
static inline uint8_t *make_weight(size_t d, size_t n_in, size_t n_out, size_t *bytes) {
    if (quant_raw_bytes(DTYPES[d].dt, n_in * n_out, bytes)) {
        return nullptr;
    }
    uint8_t *raw = malloc(*bytes);
    if (raw == nullptr) {
        return nullptr;
    }
    const size_t n = n_in * n_out;
    switch (DTYPES[d].dt) {
    case GEIST_DTYPE_F32:
        for (size_t i = 0; i < n; i++) {
            const float f = next_f();
            memcpy(raw + 4 * i, &f, 4);
        }
        return raw;
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
        for (size_t i = 0; i < n; i++) {
            /* A normal half / bfloat16 near 1: sign, exponent 14..15 (half)
             * or 126..127 (bf16), random mantissa. */
            const uint16_t r = (uint16_t) next_u32();
            const uint16_t h = DTYPES[d].dt == GEIST_DTYPE_F16
                                       ? (uint16_t) ((r & 0x83FFu) | 0x3800u)
                                       : (uint16_t) ((r & 0x807Fu) | 0x3F00u);
            raw[2 * i]       = (uint8_t) h;
            raw[2 * i + 1]   = (uint8_t) (h >> 8);
        }
        return raw;
    default:
        break;
    }
    for (size_t i = 0; i < *bytes; i++) {
        raw[i] = (uint8_t) next_u32();
    }
    if (DTYPES[d].dt == GEIST_DTYPE_I2_S) {
        const float scale = 0.75f;
        memcpy(raw + i2_s_scale_offset(n), &scale, sizeof scale);
        return raw;
    }
    for (size_t b = 0; b < n / DTYPES[d].blk; b++) {
        uint8_t *blk = raw + b * DTYPES[d].bytes;
        for (size_t f = 0; f < 2; f++) {
            if (DTYPES[d].d[f] >= 0) {
                blk[DTYPES[d].d[f]]     = 0x00; /* fp16 0.5 */
                blk[DTYPES[d].d[f] + 1] = 0x38;
            }
        }
    }
    return raw;
}

#endif /* GEIST_TESTS_QUANT_FIXTURES_H */
