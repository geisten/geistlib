/*
 * test_neon_q4k_pair_unit — cpu_neon's paired Q4_K decode GEMV,
 * linear_q4k_decode_w4a8_pair, gives each of its two outputs the bytes of
 * a single linear_q4k_decode_w4a8 on that weight, whatever the two heights:
 * equal (gate/up), the first taller (Llama's q and k), the second taller,
 * and heights that do not divide among the threads. The pair splits the
 * joint row space of both weights across the threads, so a row of either
 * weight may land on any thread; each row is still one dot over the same
 * quantized input.
 */
#include "test_helpers.h"

#if defined(GEIST_BACKEND_CPU_NEON) && defined(__ARM_FEATURE_DOTPROD)

#include "quant.h"
#include "quant_blocks.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t N_IN = 1024;

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}
static float urand(void) { /* [-1, 1) */
    return (float) (next_u64() >> 40) / (float) (1u << 23) - 1.0f;
}

static struct block_q4_K_t *random_q4k(size_t n_out) {
    const size_t         n_blocks = n_out * (N_IN / 256);
    struct block_q4_K_t *w        = xmalloc(n_blocks * sizeof *w);
    for (size_t b = 0; b < n_blocks; b++) {
        w[b].d    = 0x2400; /* fp16 2^-6 */
        w[b].dmin = 0x2000; /* fp16 2^-7 */
        for (size_t i = 0; i < sizeof w[b].scales; i++) {
            w[b].scales[i] = (uint8_t) next_u64();
        }
        for (size_t i = 0; i < sizeof w[b].qs; i++) {
            w[b].qs[i] = (uint8_t) next_u64();
        }
    }
    return w;
}

static int check_pair(size_t n_out0, size_t n_out1, const float x[static N_IN]) {
    struct block_q4_K_t *w0    = random_q4k(n_out0);
    struct block_q4_K_t *w1    = random_q4k(n_out1);
    float               *y0    = xmalloc(n_out0 * sizeof *y0);
    float               *y1    = xmalloc(n_out1 * sizeof *y1);
    float               *want0 = xmalloc(n_out0 * sizeof *want0);
    float               *want1 = xmalloc(n_out1 * sizeof *want1);
    linear_q4k_decode_w4a8(N_IN, n_out0, x, w0, want0);
    linear_q4k_decode_w4a8(N_IN, n_out1, x, w1, want1);
    linear_q4k_decode_w4a8_pair(N_IN, n_out0, n_out1, x, w0, w1, y0, y1);
    char what[96];
    snprintf(what, sizeof what, "pair %zu + %zu rows: first output as a single", n_out0, n_out1);
    int fails = geist_expect(memcmp(y0, want0, n_out0 * sizeof *y0) == 0, what);
    snprintf(what, sizeof what, "pair %zu + %zu rows: second output as a single", n_out0, n_out1);
    fails += geist_expect(memcmp(y1, want1, n_out1 * sizeof *y1) == 0, what);
    free(w0);
    free(w1);
    free(y0);
    free(y1);
    free(want0);
    free(want1);
    return fails;
}

int main(void) {
    float *x = xmalloc(N_IN * sizeof *x);
    for (size_t i = 0; i < N_IN; i++) {
        x[i] = urand();
    }
    static const size_t HEIGHTS[][2] = {{64, 64}, {192, 64}, {64, 192}, {1, 37}, {101, 3}};
    int                 fails        = 0;
    for (size_t c = 0; c < sizeof HEIGHTS / sizeof HEIGHTS[0]; c++) {
        fails += check_pair(HEIGHTS[c][0], HEIGHTS[c][1], x);
    }
    free(x);
    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: the paired Q4_K decode GEMV matches two single ones bit for bit\n");
    return GEIST_TEST_PASS;
}

#else

int main(void) {
    GEIST_SKIP("cpu_neon backend with dotprod not built");
}

#endif
