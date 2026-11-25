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
 * is. Its matrices are small enough for a BLAS's small-matrix path, which
 * rounds a row by the row count: where dense F32 goes through Accelerate or
 * OpenBLAS the pending logits of cpu_neon and cpu_x86 are held to
 * KVS_BLAS_TOL, with geist's own GEMM (CI's x86 legs) and on cpu_scalar to
 * the bit.
 *
 * A second model has the kernels of a Gemma 4 Q4_K_M: 4 layers, the last 2
 * KV-shared, d_model 256, the matrices Q4_K. Those kernels round the
 * activations to 8 bits, so a row that rounded otherwise outside its chunk
 * would not stay within rounding; a few prefills and each KV
 * representation, to the bit on every backend.
 */
#define _POSIX_C_SOURCE 200809L
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "kv_shared_prefill.h"
#include "model_fixtures.h"

int main(void) {
    static const struct mf_gemma4 GEOM = {.layers  = 7,
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
    /* The kernels a Gemma 4 Q4_K_M runs: Q4_K matrices, whose kernels round
     * the activations to 8 bits, so that a row which rounds otherwise in a
     * pass of other rows leaves on another grid point (and with an INT8
     * cache the attention rounds the queries too); the per-layer gate and
     * projection F32, 256 x 64, above the BLAS's small-matrix path. */
    static const struct mf_gemma4 QUANT      = {.layers  = 4,
                                                .shared  = 2,
                                                .period  = 2,
                                                .d_model = 256,
                                                .heads   = 2,
                                                .hd_full = 128,
                                                .hd_swa  = 128,
                                                .ffn     = 512,
                                                .hpl     = 64,
                                                .window  = 24,
                                                .context = CTX,
                                                .seed    = 13,
                                                .q4k     = true};
    static const char            *BACKENDS[] = {"cpu_neon", "cpu_x86", "cpu_scalar"};
    int                           fails      = 0;
    size_t                        ran        = 0;
    kvs_ids();
    /* ponytail: under a sanitizer the F32 model alone. With the Q4_K one
     * CI's ASan leg on Linux arm64 took 462 s for this test against 60 and
     * ran into the job's limit; the release legs run both on every
     * platform, and the rows-kept path is the same code for either. */
#if defined(__SANITIZE_ADDRESS__)
    constexpr int MODELS = 1;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    constexpr int MODELS = 1;
#else
    constexpr int MODELS = 2;
#endif
#else
    constexpr int MODELS = 2;
#endif
    for (int q = 0; q < MODELS; q++) {
        struct tf_buf           g   = mf_gemma4_gguf(q == 0 ? &GEOM : &QUANT);
        const struct kvs_source src = {
                .data = g.b, .size = g.n, .blas_rounds = q == 0, .brief = q == 1};
        printf("%s\n", q == 0 ? "F32 weights" : "Q4_K weights");
        for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
            fails += kvs_run_backend(&src, BACKENDS[b], &ran);
        }
        free(g.b);
    }
    if (ran == 0) {
        GEIST_SKIP("no CPU backend compiled in");
    }
    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: cache rows leave every bit where it was (%zu backend runs)\n", ran);
    return GEIST_TEST_PASS;
}
