/*
 * test_w4a8_kernel_unit — unit tests for the cpu_x86 W4A8 dot kernel.
 *
 * Phase 1a (current): exercises the API contract + scalar reference + every
 * compiled-in ISA-specific variant.
 *   1. tiny-deterministic: 2 blocks of hand-set values, expected exact.
 *      Includes a non-zero w_offset to verify the offset term contributes.
 *   2. zero-blocks: n_blocks=0 must return 0 cleanly.
 *   3. random-cross-check: 64 blocks of seeded PRNG data, dispatcher
 *      output matches scalar reference to ≤1e-3 (γ tolerance from spec).
 *   4. cross-ISA direct: call each compiled-in variant directly (bypassing
 *      the dispatcher latch) and verify each agrees with the scalar
 *      reference. This is the cross-ISA-consistency test from the spec's
 *      Quality-Gates section.
 *
 * Deterministic; runs in <100 ms; no model, no fixture files.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/kernel_w4a8.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward-declare every ISA-specific variant so the cross-ISA test can call
 * them directly, regardless of which one the dispatcher latched. Symbols
 * that the build did not compile in resolve at link time only if used. */
[[nodiscard]] float
w4a8_dot_avx512_vnni(size_t        n_blocks,
                     const uint8_t weights[static n_blocks * W4A8_BLOCK_BYTES_WEIGHTS],
                     const float   w_scales[static n_blocks],
                     const float   w_offsets[static n_blocks],
                     const int8_t  acts[static n_blocks * W4A8_BLOCK_ELEMS],
                     const int32_t sum_a_per_block[static n_blocks],
                     float         scale_x);

/* Pack two unsigned 4-bit nibbles in [0, 15] into one byte. lo → bits 0..3,
 * hi → bits 4..7. Matches the layout described in kernel_w4a8.h. */
static uint8_t pack_nibbles(unsigned lo, unsigned hi) {
    return (uint8_t) ((lo & 0xFu) | ((hi & 0xFu) << 4));
}

/* Tiny SplitMix32 PRNG — deterministic, no dependency on rand().
 * Returns a uint32 each call. Caller folds it into the range it needs. */
static uint32_t prng_next(uint32_t *state) {
    uint32_t z = (*state += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

/* Compute sum_a per block — what a real caller does once per activation row. */
static void compute_sum_a_per_block(size_t       n_blocks,
                                    const int8_t acts[static n_blocks * W4A8_BLOCK_ELEMS],
                                    int32_t      sum_a[static n_blocks]) {
    for (size_t b = 0; b < n_blocks; b++) {
        int32_t s = 0;
        for (size_t i = 0; i < W4A8_BLOCK_ELEMS; i++) {
            s += (int32_t) acts[b * W4A8_BLOCK_ELEMS + i];
        }
        sum_a[b] = s;
    }
}

/* ------------------------- Scenario 1: tiny deterministic ----------------- */

static int scenario_tiny_deterministic(void) {
    /* Two blocks of 32 weights = 32 bytes total (16 per block).
     *
     * Block 0: u_w = 1 everywhere, a = 2, w_scale = 1.0, w_offset = 0.5.
     *   d_b = 32 * 1 * 2 = 64.
     *   sum_a = 32 * 2 = 64.
     *   block_term = 1.0 * 64 - 0.5 * 64 = 32.
     *
     * Block 1: u_w = 15, a = -1, w_scale = 0.25, w_offset = -3.0.
     *   d_b = 32 * 15 * -1 = -480.
     *   sum_a = 32 * -1 = -32.
     *   block_term = 0.25 * -480 - (-3.0) * (-32) = -120 - 96 = -216.
     *
     * scale_x = 4.0; total = 4 * (32 - 216) = -736.
     */
    constexpr size_t NB = 2;
    uint8_t          weights[NB * W4A8_BLOCK_BYTES_WEIGHTS];
    int8_t           acts[NB * W4A8_BLOCK_ELEMS];
    float            w_scales[NB];
    float            w_offsets[NB];
    int32_t          sum_a[NB];

    for (size_t k = 0; k < W4A8_BLOCK_BYTES_WEIGHTS; k++) {
        weights[k] = pack_nibbles(1, 1);
    }
    for (size_t i = 0; i < W4A8_BLOCK_ELEMS; i++) {
        acts[i] = 2;
    }
    w_scales[0]  = 1.0f;
    w_offsets[0] = 0.5f;

    for (size_t k = 0; k < W4A8_BLOCK_BYTES_WEIGHTS; k++) {
        weights[W4A8_BLOCK_BYTES_WEIGHTS + k] = pack_nibbles(15, 15);
    }
    for (size_t i = 0; i < W4A8_BLOCK_ELEMS; i++) {
        acts[W4A8_BLOCK_ELEMS + i] = -1;
    }
    w_scales[1]  = 0.25f;
    w_offsets[1] = -3.0f;

    compute_sum_a_per_block(NB, acts, sum_a);
    const float scale_x  = 4.0f;
    const float expected = -736.0f;

    const float y_ref = w4a8_dot_scalar(NB, weights, w_scales, w_offsets, acts, sum_a, scale_x);
    if (fabsf(y_ref - expected) > 1e-4f) {
        fprintf(stderr, "tiny: scalar y=%f, expected %f\n", (double) y_ref, (double) expected);
        return 1;
    }

    const float y_disp = w4a8_dot(NB, weights, w_scales, w_offsets, acts, sum_a, scale_x);
    if (fabsf(y_disp - y_ref) > 1e-4f) {
        fprintf(stderr,
                "tiny: dispatcher diverged: scalar=%f dispatched=%f\n",
                (double) y_ref,
                (double) y_disp);
        return 1;
    }
    return 0;
}

/* ------------------------- Scenario 2: zero blocks ------------------------ */

static int scenario_zero_blocks(void) {
    /* n_blocks=0 — the [static n] arrays may be unindexable. Kernel must
     * return scale_x * 0 = 0 without dereferencing. */
    uint8_t weights[1]   = {0};
    int8_t  acts[1]      = {0};
    float   w_scales[1]  = {0.0f};
    float   w_offsets[1] = {0.0f};
    int32_t sum_a[1]     = {0};

    const float y = w4a8_dot(0, weights, w_scales, w_offsets, acts, sum_a, 7.5f);
    if (fabsf(y) > 1e-9f) {
        fprintf(stderr, "zero: expected 0.0, got %f\n", (double) y);
        return 1;
    }
    return 0;
}

/* ------------------------- Scenario 3: random cross-check ----------------- */

static int scenario_random_cross_check(void) {
    constexpr size_t NB      = 64;
    constexpr size_t W_BYTES = NB * W4A8_BLOCK_BYTES_WEIGHTS;
    constexpr size_t A_BYTES = NB * W4A8_BLOCK_ELEMS;

    uint8_t weights[W_BYTES];
    int8_t  acts[A_BYTES];
    float   w_scales[NB];
    float   w_offsets[NB];
    int32_t sum_a[NB];

    uint32_t s = 0xDEADBEEFu;
    for (size_t k = 0; k < W_BYTES; k++) {
        const unsigned lo = prng_next(&s) & 0xFu;
        const unsigned hi = prng_next(&s) & 0xFu;
        weights[k]        = pack_nibbles(lo, hi);
    }
    for (size_t i = 0; i < A_BYTES; i++) {
        acts[i] = (int8_t) ((int) (prng_next(&s) % 255u) - 127);
    }
    for (size_t b = 0; b < NB; b++) {
        const uint32_t s_bits = prng_next(&s);
        const uint32_t o_bits = prng_next(&s);
        const float    s_mag  = (float) ((s_bits & 0xFFFFu) + 1u) / 65536.0f;
        const float    o_mag  = (float) ((o_bits & 0xFFFFu) + 1u) / 65536.0f;
        /* Q4_K-ish range: scale ~ d * sub_scale ≈ 0.05, offset ~ dmin * sub_min
         * also ≈ 0.05. */
        w_scales[b]  = s_mag * 0.05f;
        w_offsets[b] = o_mag * 0.05f - 0.025f; /* span [-0.025, 0.025] */
    }
    compute_sum_a_per_block(NB, acts, sum_a);
    const float scale_x = 0.013f;

    const float y_ref  = w4a8_dot_scalar(NB, weights, w_scales, w_offsets, acts, sum_a, scale_x);
    const float y_disp = w4a8_dot(NB, weights, w_scales, w_offsets, acts, sum_a, scale_x);

    const float atol = 1e-3f;
    const float rtol = 1e-3f;
    const float diff = fabsf(y_disp - y_ref);
    const float tol  = atol + rtol * fabsf(y_ref);
    if (diff > tol) {
        fprintf(stderr,
                "random: dispatcher diverged: scalar=%g dispatched=%g diff=%g tol=%g\n",
                (double) y_ref,
                (double) y_disp,
                (double) diff,
                (double) tol);
        return 1;
    }
    return 0;
}

/* ------------------------- Scenario 4: cross-ISA direct ------------------- */

static int compare_to_scalar(const char *label, float y_isa, float y_ref) {
    const float atol = 1e-3f;
    const float rtol = 1e-3f;
    const float diff = fabsf(y_isa - y_ref);
    const float tol  = atol + rtol * fabsf(y_ref);
    if (diff > tol) {
        fprintf(stderr,
                "cross-ISA: %s diverged: ref=%g isa=%g diff=%g tol=%g\n",
                label,
                (double) y_ref,
                (double) y_isa,
                (double) diff,
                (double) tol);
        return 1;
    }
    return 0;
}

static int scenario_cross_isa_direct(void) {
    constexpr size_t NB      = 64;
    constexpr size_t W_BYTES = NB * W4A8_BLOCK_BYTES_WEIGHTS;
    constexpr size_t A_BYTES = NB * W4A8_BLOCK_ELEMS;

    uint8_t weights[W_BYTES];
    int8_t  acts[A_BYTES];
    float   w_scales[NB];
    float   w_offsets[NB];
    int32_t sum_a[NB];

    /* Different seed from scenario_random_cross_check to exercise a
     * distinct input distribution. */
    uint32_t s = 0xCAFEBABEu;
    for (size_t k = 0; k < W_BYTES; k++) {
        const unsigned lo = prng_next(&s) & 0xFu;
        const unsigned hi = prng_next(&s) & 0xFu;
        weights[k]        = pack_nibbles(lo, hi);
    }
    for (size_t i = 0; i < A_BYTES; i++) {
        acts[i] = (int8_t) ((int) (prng_next(&s) % 255u) - 127);
    }
    for (size_t b = 0; b < NB; b++) {
        const uint32_t s_bits = prng_next(&s);
        const uint32_t o_bits = prng_next(&s);
        const float    s_mag  = (float) ((s_bits & 0xFFFFu) + 1u) / 65536.0f;
        const float    o_mag  = (float) ((o_bits & 0xFFFFu) + 1u) / 65536.0f;
        w_scales[b]           = s_mag * 0.05f;
        w_offsets[b]          = o_mag * 0.05f - 0.025f;
    }
    compute_sum_a_per_block(NB, acts, sum_a);
    const float scale_x = 0.017f;

    const float y_ref = w4a8_dot_scalar(NB, weights, w_scales, w_offsets, acts, sum_a, scale_x);

    int                 fails   = 0;
    const enum w4a8_isa current = w4a8_dispatcher_current();
    if (current == W4A8_ISA_AVX512_VNNI || current == W4A8_ISA_AVX512_BF16) {
        const float y_vnni =
                w4a8_dot_avx512_vnni(NB, weights, w_scales, w_offsets, acts, sum_a, scale_x);
        fails += compare_to_scalar("avx512_vnni", y_vnni, y_ref);
    }
    return fails;
}

/* --------------------------------- main ---------------------------------- */

int main(void) {
    int fails = 0;
    if (scenario_tiny_deterministic() != 0) {
        fputs("scenario_tiny_deterministic FAILED\n", stderr);
        fails++;
    }
    if (scenario_zero_blocks() != 0) {
        fputs("scenario_zero_blocks FAILED\n", stderr);
        fails++;
    }
    if (scenario_random_cross_check() != 0) {
        fputs("scenario_random_cross_check FAILED\n", stderr);
        fails++;
    }
    if (scenario_cross_isa_direct() != 0) {
        fputs("scenario_cross_isa_direct FAILED\n", stderr);
        fails++;
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
