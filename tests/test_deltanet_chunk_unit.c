/*
 * test_deltanet_chunk_unit — #281 chunked delta rule == sequential, at
 * f32 precision, no fixture.
 *
 * Drives transformer_dn_head_chunk against C repetitions of
 * transformer_dn_head_step on random data and compares all C outputs
 * plus the final state. This is the TIGHT equivalence pin (rel err
 * ~1e-5): the end-to-end oracle (test_deltanet_chunk_int) runs
 * through int8-quantized projections whose legitimate cross-platform
 * jitter (~1.0 logits) swamps subtle kernel bugs — a 10% beta error
 * scores only ~0.6 there, but fails here by orders of magnitude.
 *
 * Shapes cover C = 1 (degenerate chunk), C = 2 (< conv kernel, the
 * tiny-batch edge), an odd C, and C = 64 (the m_max chunk); strides
 * exercise the direct-from-y-buffer row layout.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "src/archs/transformer/forward/internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_C 64
#define MAX_D 64
#define TOL 2e-4f
#define STRIDE 7 /* extra floats between rows to exercise sq/sk/sv */

static unsigned long long rng_state = 0x9E3779B97F4A7C15ull;
static float              frand(void) {
    /* xorshift64*; uniform in [-1, 1) */
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    const unsigned long long r = rng_state * 0x2545F4914F6CDD1Dull;
    return (float) ((double) (r >> 11) / (double) (1ull << 53)) * 2.0f - 1.0f;
}

static int run_case(size_t C, size_t d_k, size_t d_v) {
    const size_t sq = d_k + STRIDE, sk = d_k + STRIDE, sv = d_v + STRIDE;
    const size_t sbg = 3;

    static float Q[MAX_C * (MAX_D + STRIDE)], K[MAX_C * (MAX_D + STRIDE)],
            V[MAX_C * (MAX_D + STRIDE)];
    static float beta[MAX_C * 3], g[MAX_C * 3];
    static float S_seq[MAX_D * MAX_D], S_chunk[MAX_D * MAX_D];
    static float o_seq[MAX_C * MAX_D], o_chunk[MAX_C * MAX_D];
    static float kv_mem[MAX_D], delta[MAX_D];

    for (size_t t = 0; t < C; t++) {
        for (size_t i = 0; i < d_k; i++) {
            Q[t * sq + i] = frand() * 0.5f;
            K[t * sk + i] = frand() * 0.5f;
        }
        for (size_t j = 0; j < d_v; j++)
            V[t * sv + j] = frand();
        beta[t * sbg] = 1.0f / (1.0f + expf(-2.0f * frand())); /* (0,1) */
        g[t * sbg]    = -1.5f * (frand() + 1.0f);              /* [-3, 0) */
    }
    for (size_t i = 0; i < d_k * d_v; i++) {
        S_seq[i]   = frand() * 0.3f;
        S_chunk[i] = S_seq[i];
    }

    /* Sequential reference: C dn_head_step calls. */
    for (size_t t = 0; t < C; t++) {
        transformer_dn_head_step(S_seq,
                                 Q + t * sq,
                                 K + t * sk,
                                 V + t * sv,
                                 expf(g[t * sbg]),
                                 beta[t * sbg],
                                 d_k,
                                 d_v,
                                 kv_mem,
                                 delta,
                                 o_seq + t * d_v);
    }

    /* Chunked path. */
    const size_t ws_f = transformer_dn_chunk_ws_floats(C, d_k, d_v);
    float       *ws   = malloc(ws_f * sizeof(float));
    if (ws == NULL) {
        fprintf(stderr, "FAIL: ws alloc\n");
        return 1;
    }
    transformer_dn_head_chunk(S_chunk, Q, sq, K, sk, V, sv, beta, g, sbg, C, d_k, d_v, o_chunk, ws);
    free(ws);

    /* Compare outputs and final state, relative to the value scale. */
    float scale = 1e-3f;
    for (size_t i = 0; i < C * d_v; i++)
        if (fabsf(o_seq[i]) > scale)
            scale = fabsf(o_seq[i]);
    float mo = 0.0f, ms = 0.0f;
    for (size_t i = 0; i < C * d_v; i++) {
        const float d = fabsf(o_chunk[i] - o_seq[i]);
        if (d > mo)
            mo = d;
    }
    for (size_t i = 0; i < d_k * d_v; i++) {
        const float d = fabsf(S_chunk[i] - S_seq[i]);
        if (d > ms)
            ms = d;
    }
    const float rel_o = mo / scale, rel_s = ms / scale;
    printf("C=%2zu d_k=%2zu d_v=%2zu: rel|do|=%.2e rel|dS|=%.2e\n",
           C,
           d_k,
           d_v,
           (double) rel_o,
           (double) rel_s);
    if (rel_o > TOL || rel_s > TOL) {
        fprintf(stderr, "FAIL: chunk kernel != sequential (tol %.0e)\n", (double) TOL);
        return 1;
    }
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= run_case(1, 8, 8);  /* degenerate single-token chunk */
    rc |= run_case(2, 8, 16); /* below conv-kernel length */
    rc |= run_case(5, 16, 8); /* odd C, d_k > d_v */
    rc |= run_case(37, 32, 32);
    rc |= run_case(64, 32, 64); /* full m_max chunk, d_k != d_v */
    rc |= run_case(64, 64, 64);
    if (rc == 0)
        printf("OK: dn_head_chunk == dn_head_step on all shapes\n");
    return rc;
}
