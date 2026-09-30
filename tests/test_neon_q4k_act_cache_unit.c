/*
 * test_neon_q4k_act_cache_unit — cpu_neon's Q4_K decode kernel,
 * linear_q4k_decode_w4a8, keeps the int8 quantization of its input from one
 * call to the next, since q/k/v and gate/up read one normed vector in turn.
 * Reusing it is only right when the input is the same: a new vector written
 * into the same buffer must be quantized anew, however many values it
 * shares with the old one.
 *
 * Each case calls the kernel on a buffer, changes one element of it in
 * place, calls it again on the same buffer, and compares that output bit
 * for bit with the kernel's output for the same values in another buffer
 * (which cannot hit the cache). A last case calls twice on unchanged data,
 * where the outputs must match as well.
 */
#include "test_helpers.h"

#if defined(GEIST_BACKEND_CPU_NEON) && defined(__ARM_FEATURE_DOTPROD)

#include "quant.h"
#include "quant_blocks.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t N_IN  = 1024;
constexpr size_t N_OUT = 64;

static uint64_t g_rng = 0x2545F4914F6CDD1Dull;
static uint64_t next_u64(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}
static float urand(void) { /* [-1, 1) */
    return (float) (next_u64() >> 40) / (float) (1u << 23) - 1.0f;
}

int main(void) {
    const size_t         n_blocks = N_OUT * (N_IN / 256);
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
    float *x     = xmalloc(N_IN * sizeof *x);
    float *fresh = xmalloc(N_IN * sizeof *fresh);
    float *y     = xmalloc(N_OUT * sizeof *y);
    float *want  = xmalloc(N_OUT * sizeof *want);
    int    fails = 0;

    /* One element changed at a time, at positions spread over the vector. */
    static const size_t AT[] = {1, 7, N_IN / 3, N_IN / 2 + 5, N_IN - 2};
    for (size_t c = 0; c < sizeof AT / sizeof AT[0]; c++) {
        for (size_t i = 0; i < N_IN; i++) {
            x[i] = urand();
        }
        linear_q4k_decode_w4a8(N_IN, N_OUT, x, w, y);
        x[AT[c]] += 0.75f;
        linear_q4k_decode_w4a8(N_IN, N_OUT, x, w, y);
        memcpy(fresh, x, N_IN * sizeof *x);
        linear_q4k_decode_w4a8(N_IN, N_OUT, fresh, w, want);
        char what[96];
        snprintf(what,
                 sizeof what,
                 "input rewritten in place at %zu: output as for a new buffer",
                 AT[c]);
        fails += geist_expect(memcmp(y, want, N_OUT * sizeof *y) == 0, what);
    }

    /* The same data twice: the reuse the cache exists for. */
    for (size_t i = 0; i < N_IN; i++) {
        x[i] = urand();
    }
    linear_q4k_decode_w4a8(N_IN, N_OUT, x, w, want);
    linear_q4k_decode_w4a8(N_IN, N_OUT, x, w, y);
    fails += geist_expect(memcmp(y, want, N_OUT * sizeof *y) == 0,
                          "unchanged input: the same output again");

    free(w);
    free(x);
    free(fresh);
    free(y);
    free(want);
    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: the Q4_K decode kernel quantizes a rewritten input anew\n");
    return GEIST_TEST_PASS;
}

#else

int main(void) {
    GEIST_SKIP("cpu_neon backend with dotprod not built");
}

#endif
