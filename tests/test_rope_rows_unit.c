/*
 * test_rope_rows_unit — the load-time q/k row reorder of NORM-RoPE GGUFs (#464).
 *
 * The forward pass used to permute every q/k head (x[2i], x[2i+1]) ->
 * (x[i], x[i + hd/2]) after the projection, on every layer and token. The
 * loader now reorders the projection's rows once instead. Two properties pin
 * that down, no model needed:
 *   - the byte mapping: row 2i of each head lands at i, row 2i+1 at i + hd/2,
 *     whole rows of any width (a quantized row is opaque bytes);
 *   - the equivalence: projecting with the reordered rows gives exactly, bit
 *     for bit, the projection the old runtime permutation produced.
 */
#define GEIST_INTERNAL_ARCH_LAYER /* weight_load/internal.h is layer-internal */

#include "test_helpers.h"

#include "../src/archs/transformer/weight_load/internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

/* Every byte of a row encodes (row, byte) so a misplaced row cannot pass. */
static uint8_t tag(size_t row, size_t b) {
    return (uint8_t) (row * 31u + b * 7u + 1u);
}

static void check_mapping(size_t n_heads, size_t head_dim, size_t row_bytes) {
    const size_t n_rows = n_heads * head_dim;
    uint8_t     *src    = malloc(n_rows * row_bytes);
    uint8_t     *dst    = malloc(n_rows * row_bytes);
    if (src == nullptr || dst == nullptr) {
        fails++;
        free(src);
        free(dst);
        return;
    }
    for (size_t r = 0; r < n_rows; r++) {
        for (size_t b = 0; b < row_bytes; b++) {
            src[r * row_bytes + b] = tag(r, b);
        }
    }
    permute_rope_rows(n_rows, row_bytes, head_dim, src, dst);

    bool         ok   = true;
    const size_t half = head_dim / 2;
    for (size_t h = 0; h < n_heads && ok; h++) {
        for (size_t i = 0; i < head_dim && ok; i++) {
            const size_t from = h * head_dim + (i < half ? 2 * i : 2 * (i - half) + 1);
            ok = memcmp(dst + (h * head_dim + i) * row_bytes, src + from * row_bytes, row_bytes) ==
                 0;
        }
    }
    char what[96];
    snprintf(what,
             sizeof what,
             "row mapping: %zu heads x hd %zu, %zu-byte rows",
             n_heads,
             head_dim,
             row_bytes);
    fails += geist_expect(ok, what);
    free(src);
    free(dst);
}

/* y = W x, one row at a time in a fixed order: the same arithmetic per output
 * element whichever position the row sits at. The accumulator is volatile
 * because the tree builds with -ffast-math: a vectorised reduction may peel
 * by the row's alignment, and w and its permuted copy are separate
 * allocations, so the order of the sum would depend on malloc, not on the
 * permutation under test. */
static void matvec(size_t n_rows, size_t n_in, const float *w, const float *x, float *y) {
    for (size_t r = 0; r < n_rows; r++) {
        volatile float acc = 0.0f;
        for (size_t k = 0; k < n_in; k++) {
            acc += w[r * n_in + k] * x[k];
        }
        y[r] = acc;
    }
}

static void check_equivalence(size_t n_heads, size_t head_dim, size_t n_in) {
    const size_t n_rows = n_heads * head_dim;
    float       *w      = malloc(n_rows * n_in * sizeof(float));
    float       *w2     = malloc(n_rows * n_in * sizeof(float));
    float       *x      = malloc(n_in * sizeof(float));
    float       *y_old  = malloc(n_rows * sizeof(float));
    float       *y_new  = malloc(n_rows * sizeof(float));
    float       *tmp    = malloc(head_dim * sizeof(float));
    if (w == nullptr || w2 == nullptr || x == nullptr || y_old == nullptr || y_new == nullptr ||
        tmp == nullptr) {
        fails++;
        goto out;
    }
    uint32_t seed = 0x9e3779b9u;
    for (size_t i = 0; i < n_rows * n_in; i++) {
        seed = seed * 1664525u + 1013904223u;
        w[i] = (float) (int32_t) (seed >> 8) / (float) (1 << 23) - 1.0f;
    }
    for (size_t k = 0; k < n_in; k++) {
        seed = seed * 1664525u + 1013904223u;
        x[k] = (float) (int32_t) (seed >> 8) / (float) (1 << 23) - 1.0f;
    }

    /* Before: project, then permute each head the way the forward pass did. */
    matvec(n_rows, n_in, w, x, y_old);
    const size_t half = head_dim / 2;
    for (size_t h = 0; h < n_heads; h++) {
        float *yh = y_old + h * head_dim;
        for (size_t i = 0; i < half; i++) {
            tmp[i]        = yh[2 * i];
            tmp[i + half] = yh[2 * i + 1];
        }
        memcpy(yh, tmp, head_dim * sizeof(float));
    }

    /* After: reorder the rows at load, project. */
    permute_rope_rows(n_rows, n_in * sizeof(float), head_dim, (const uint8_t *) w, (uint8_t *) w2);
    matvec(n_rows, n_in, w2, x, y_new);

    char what[96];
    snprintf(what,
             sizeof what,
             "bit-identical to the runtime permute: %zu heads x hd %zu",
             n_heads,
             head_dim);
    fails += geist_expect(memcmp(y_old, y_new, n_rows * sizeof(float)) == 0, what);
out:
    free(w);
    free(w2);
    free(x);
    free(y_old);
    free(y_new);
    free(tmp);
}

int main(void) {
    check_mapping(1, 2, 1);
    check_mapping(3, 8, 34);    /* Q4_0-sized block rows */
    check_mapping(2, 64, 144);  /* Q4_K super-block rows */
    check_mapping(8, 128, 210); /* Q6_K, llama-3.2-3B q geometry */
    check_equivalence(4, 64, 96);
    check_equivalence(2, 128, 64);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
