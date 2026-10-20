/*
 * test_linear_fp32_rows_unit — linear_fp32 gives a row the same bits
 * whatever number of rows it is computed with.
 *
 * The streaming audio encoder runs the rows of a clip in blocks of up to
 * 12 sub-tokens, its tail often as a single row, and must reproduce the
 * batch encode bit for bit: a 2e-6 shift in one soft token flipped the
 * greedy prediction of test_audio_stream_session_int on Neoverse-N1, where
 * OpenBLAS's sgemv (the old m == 1 path) sums in another order than its
 * sgemm. Accelerate does the same for an M=1 sgemm. So every row count
 * from 1 up is checked against the rows of one 64-row product, at the
 * audio tower's shapes, whatever GEMM provider the build links.
 */
#include "test_helpers.h"

#include "gemma4_kernels.h"
#include "heap.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define M_ALL 64

static uint64_t rng = 0x9E3779B97F4A7C15ull;

static float uniform(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (float) ((double) (rng >> 11) / (double) (1ull << 53) * 2.0 - 1.0);
}

/* 0 when every m in M_LIST reproduces the first m rows of the 64-row run. */
static int check_shape(size_t n_in, size_t n_out, bool with_bias) {
    static const size_t M_LIST[] = {1, 2, 3, 5, 11, 12, 13, 50};
    float              *x        = heap_alloc_array_aligned(float, (size_t) M_ALL *n_in);
    float              *w        = heap_alloc_array_aligned(float, n_out *n_in);
    float              *bias     = heap_alloc_array_aligned(float, n_out);
    float              *ref      = heap_alloc_array_aligned(float, (size_t) M_ALL *n_out);
    float              *y        = heap_alloc_array_aligned(float, (size_t) M_ALL *n_out);
    if (x == nullptr || w == nullptr || bias == nullptr || ref == nullptr || y == nullptr) {
        safe_free((void **) &x);
        safe_free((void **) &w);
        safe_free((void **) &bias);
        safe_free((void **) &ref);
        safe_free((void **) &y);
        return -1;
    }
    for (size_t i = 0; i < (size_t) M_ALL * n_in; i++)
        x[i] = uniform();
    for (size_t i = 0; i < n_out * n_in; i++)
        w[i] = uniform() * 0.05f;
    for (size_t i = 0; i < n_out; i++)
        bias[i] = uniform();
    const float *b = with_bias ? bias : nullptr;

    linear_fp32(M_ALL, n_in, n_out, x, w, b, ref);
    int fails = 0;
    for (size_t k = 0; k < sizeof M_LIST / sizeof M_LIST[0]; k++) {
        const size_t m = M_LIST[k];
        /* Each row from a different offset, so a row is also checked away
         * from the start of the batch. */
        const size_t row0 = M_ALL - m;
        memset(y, 0xff, (size_t) m * n_out * sizeof(float));
        linear_fp32(m, n_in, n_out, x + row0 * n_in, w, b, y);
        if (memcmp(y, ref + row0 * n_out, m * n_out * sizeof(float)) != 0) {
            size_t diff = 0;
            for (size_t i = 0; i < m * n_out; i++)
                diff += memcmp(&y[i], &ref[row0 * n_out + i], sizeof(float)) != 0;
            fprintf(stderr,
                    "FAIL: %zux%zu%s, m=%zu: %zu of %zu outputs differ from the %d-row run\n",
                    n_out,
                    n_in,
                    with_bias ? " +bias" : "",
                    m,
                    diff,
                    m * n_out,
                    M_ALL);
            fails++;
        }
    }
    safe_free((void **) &x);
    safe_free((void **) &w);
    safe_free((void **) &bias);
    safe_free((void **) &ref);
    safe_free((void **) &y);
    return fails;
}

int main(void) {
    /* The audio tower's fp32 linears: in_proj / attention (1024 -> 1024),
     * output_proj (1024 -> 1536, with bias), embed_audio (1536 -> 1536),
     * FFN (1024 -> 4096 -> 1024). */
    static const struct {
        size_t n_in, n_out;
        bool   bias;
    } SHAPES[] = {
            {1024, 1024, false},
            {1024, 1536, true},
            {1536, 1536, false},
            {1024, 4096, false},
            {4096, 1024, false},
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof SHAPES / sizeof SHAPES[0]; i++) {
        const int r = check_shape(SHAPES[i].n_in, SHAPES[i].n_out, SHAPES[i].bias);
        if (r < 0)
            return GEIST_TEST_ERROR;
        fails += r;
    }
    if (fails != 0) {
        printf("FAIL (%d)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: linear_fp32 gives each row the same bits for any row count\n");
    return GEIST_TEST_PASS;
}
