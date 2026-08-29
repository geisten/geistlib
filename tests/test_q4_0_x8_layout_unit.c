/* Regression coverage for the Q4_0 x8 v2 packed layout used by both
 * lane-SDOT decode and the mN prefill kernel. */
#include "test_helpers.h"

#if defined(GEIST_BACKEND_CPU_NEON)

#include "quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct test_block_q4_0 {
    uint16_t d;
    uint8_t  qs[16];
} __attribute__((packed));
_Static_assert(sizeof(struct test_block_q4_0) == 18, "Q4_0 fixture block size");

static size_t v2_index(size_t row, size_t byte) {
    return (row / 4) * 64 + (byte / 4) * 16 + (row % 4) * 4 + (byte % 4);
}

static int expect_same_f32(const float *actual, const float *expected, size_t n, const char *what) {
    if (memcmp(actual, expected, n * sizeof(*actual)) == 0)
        return 0;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(actual + i, expected + i, sizeof(*actual)) != 0) {
            fprintf(stderr,
                    "FAIL: %s differs at %zu: %.9g != %.9g\n",
                    what,
                    i,
                    actual[i],
                    expected[i]);
            break;
        }
    }
    return 1;
}

int main(void) {
    enum {
        N_IN       = 64,
        N_OUT      = 8,
        N_BLOCKS   = N_IN / 32,
        HEADER_LEN = 16,
        BLOCK_LEN  = 8 * 18,
    };
    struct test_block_q4_0 raw[N_OUT * N_BLOCKS];
    for (size_t row = 0; row < N_OUT; row++) {
        for (size_t block = 0; block < N_BLOCKS; block++) {
            struct test_block_q4_0 *qb = &raw[row * N_BLOCKS + block];
            qb->d                      = 0x3C00u; /* fp16 1.0 */
            for (size_t byte = 0; byte < 16; byte++)
                qb->qs[byte] = (uint8_t) (row * 37 + block * 19 + byte * 11);
        }
    }

    int          fails        = 0;
    const size_t packed_bytes = q4_0_x8_gemv_size_bytes(N_IN, N_OUT);
    uint8_t     *packed       = malloc(packed_bytes);
    fails += geist_expect(packed_bytes == HEADER_LEN + N_BLOCKS * BLOCK_LEN, "packed size");
    fails += geist_expect(packed != nullptr, "packed allocation");
    if (packed == nullptr)
        return GEIST_TEST_ERROR;
    fails += geist_expect(q4_0_x8_gemv_pack(raw, N_IN, N_OUT, packed) == 0, "pack succeeds");

    uint32_t magic = 0;
    memcpy(&magic, packed, sizeof(magic));
    fails += geist_expect(magic == 0x38583431u, "v2 magic is 14X8");
    for (size_t block = 0; block < N_BLOCKS; block++) {
        const uint8_t *packed_qs = packed + HEADER_LEN + block * BLOCK_LEN + 8 * sizeof(uint16_t);
        for (size_t row = 0; row < N_OUT; row++) {
            for (size_t byte = 0; byte < 16; byte++) {
                const uint8_t expected = raw[row * N_BLOCKS + block].qs[byte];
                if (packed_qs[v2_index(row, byte)] != expected) {
                    fprintf(stderr,
                            "FAIL: packed map block=%zu row=%zu byte=%zu: %u != %u\n",
                            block,
                            row,
                            byte,
                            (unsigned) packed_qs[v2_index(row, byte)],
                            (unsigned) expected);
                    fails++;
                }
            }
        }
    }

    float x[5 * N_IN];
    for (size_t i = 0; i < 5 * N_IN; i++)
        x[i] = (float) ((int) (i % 31) - 15) * 0.125f;

    float decode_ref[N_OUT];
    float decode_v2[N_OUT];
    linear_q4_0_decode_w4a8(x, raw, N_IN, N_OUT, decode_ref);
    linear_q4_0_decode_w4a8_x8(x, packed, N_IN, N_OUT, decode_v2);
    fails += expect_same_f32(decode_v2, decode_ref, N_OUT, "v2 decode parity");

    static const size_t m_cases[] = {1, 3, 4, 5};
    for (size_t mi = 0; mi < sizeof(m_cases) / sizeof(m_cases[0]); mi++) {
        const size_t m = m_cases[mi];
        float        prefill_ref[5 * N_OUT];
        float        prefill_v2[5 * N_OUT];
        linear_q4_0_w4a8_prefill(x, m, raw, N_IN, N_OUT, prefill_ref);
        linear_q4_0_w4a8_prefill_x8(x, m, packed, N_IN, N_OUT, prefill_v2);
        fails += expect_same_f32(prefill_v2, prefill_ref, m * N_OUT, "v2 prefill parity");
    }

    uint8_t *old_layout = malloc(packed_bytes);
    fails += geist_expect(old_layout != nullptr, "old-layout allocation");
    if (old_layout != nullptr) {
        memcpy(old_layout, packed, packed_bytes);
        const uint32_t old_magic = 0x38583430u;
        memcpy(old_layout, &old_magic, sizeof(old_magic));
        for (size_t i = 0; i < N_OUT; i++)
            decode_v2[i] = 1234.0f + (float) i;
        linear_q4_0_decode_w4a8_x8(x, old_layout, N_IN, N_OUT, decode_v2);
        for (size_t i = 0; i < N_OUT; i++)
            fails += geist_expect(decode_v2[i] == 1234.0f + (float) i, "v1 layout is rejected");
        free(old_layout);
    }

    fails += geist_expect(q4_0_x8_gemv_size_bytes(0, N_OUT) == 0, "zero input rejected");
    fails += geist_expect(q4_0_x8_gemv_size_bytes(N_IN, N_OUT - 1) == 0, "partial tile rejected");
    fails +=
            geist_expect(q4_0_x8_gemv_pack(raw, N_IN, N_OUT, nullptr) != 0, "null output rejected");

    free(packed);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

#else

int main(void) {
    GEIST_SKIP("cpu_neon backend not built");
}

#endif
