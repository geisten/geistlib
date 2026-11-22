/*
 * test_kv_shared_prefill_unit - test_kv_shared_prefill_int's cases on a
 * Gemma 4 built in memory (#591), on every CPU backend in the build: a
 * prefill that leaves the KV-shared layers out for its cache rows gives the
 * bits of one that runs every layer for every row (GEIST_KV_SHARED_FULL=1),
 * in the logits, the pending token, the length, the token history and every
 * K/V row, after prefills around every boundary of the plan, decode, a
 * repeat, two calls, truncate, snapshot and restore, a pinned prefix, a
 * verify pass, soft tokens, and with each KV representation.
 *
 * The model has the family's parts at a small size: 7 layers, sliding-window
 * and full attention alternating in threes, the last 3 KV-shared (so 4
 * write a cache, n_cache_layers), per-layer embeddings, GEGLU, F32 weights.
 * No model file: this runs wherever the unit tests run, the x86 kernels
 * included, which the integration test reaches only where a Gemma 4 GGUF
 * is.
 */
#define _POSIX_C_SOURCE 200809L
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "kv_shared_prefill.h"
#include "model_fixtures.h"

int main(void) {
    static const struct mf_gemma4 GEOM       = {.layers  = 7,
                                                .shared  = 3,
                                                .period  = 3,
                                                .d_model = 32,
                                                .heads   = 2,
                                                .hd_full = 16,
                                                .hd_swa  = 8,
                                                .ffn     = 64,
                                                .hpl     = 8,
                                                .window  = 24,
                                                .context = CTX,
                                                .seed    = 11};
    static const char            *BACKENDS[] = {"cpu_neon", "cpu_x86", "cpu_scalar"};
    struct tf_buf                 g          = mf_gemma4_gguf(&GEOM);
    const struct kvs_source       src        = {.data = g.b, .size = g.n};
    int                           fails      = 0;
    size_t                        ran        = 0;
    kvs_ids();
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        fails += kvs_run_backend(&src, BACKENDS[b], &ran);
    }
    free(g.b);
    if (ran == 0) {
        GEIST_SKIP("no CPU backend compiled in");
    }
    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: cache rows leave every bit where it was (%zu backends)\n", ran);
    return GEIST_TEST_PASS;
}
