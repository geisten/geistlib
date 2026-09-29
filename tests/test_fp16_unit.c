/*
 * test_fp16_unit — fp16_to_fp32 (quant.h) on all 65536 half-precision
 * values, against a decode written out here in integers: zeros, subnormals,
 * normals and infinities bit for bit, NaNs as NaNs (hardware converters may
 * quiet a signaling NaN). The function is inline hardware conversion on ARM
 * (__fp16) and on x86 with F16C (_Float16), and a software decode in
 * formats/gguf/common.c elsewhere; this holds each build's to the same
 * table.
 */
#include "test_helpers.h"

#include "quant.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t reference_bits(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t       frac = h & 0x3FFu;
    if (exp == 0x1Fu) {
        return sign | 0x7F800000u | (frac << 13);
    }
    if (exp != 0) {
        return sign | ((exp + 112u) << 23) | (frac << 13);
    }
    if (frac == 0) {
        return sign;
    }
    /* Subnormal: frac * 2^-24. Normalize: shift until bit 10 is set. */
    uint32_t e = 113u; /* biased fp32 exponent of 2^-14 */
    while ((frac & 0x400u) == 0) {
        frac <<= 1;
        e--;
    }
    return sign | (e << 23) | ((frac & 0x3FFu) << 13);
}

int main(void) {
    int fails = 0;
    for (uint32_t i = 0; i <= 0xFFFFu; i++) {
        const uint16_t h    = (uint16_t) i;
        const float    f    = fp16_to_fp32(h);
        const uint32_t want = reference_bits(h);
        uint32_t       got;
        memcpy(&got, &f, sizeof got);
        const bool nan_in = ((h >> 10) & 0x1Fu) == 0x1Fu && (h & 0x3FFu) != 0;
        const bool ok =
                nan_in ? (got & 0x7F800000u) == 0x7F800000u && (got & 0x7FFFFFu) != 0 : got == want;
        if (!ok && fails++ < 8) {
            fprintf(stderr, "FAIL: fp16 0x%04x -> 0x%08x, want 0x%08x\n", h, got, want);
        }
    }
    if (fails != 0) {
        fprintf(stderr, "%d of 65536 values wrong\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: fp16_to_fp32 decodes all 65536 half-precision values\n");
    return GEIST_TEST_PASS;
}
