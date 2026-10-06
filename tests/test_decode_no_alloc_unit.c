/*
 * test_decode_no_alloc_unit — steady-state decode steps and prefill calls
 * allocate nothing (AGENT.md §3: no heap allocation per token, per layer,
 * per block), checked through the public session API for every KV-cache
 * mode on every CPU backend in the build, and on Metal.
 *
 * heap.h counts successful allocations (heap_alloc_count), and a Metal
 * buffer counts once, through its handle (geist_backend_alloc). The test
 * counts across 16 decode steps and one 32-token prefill after a warm-up
 * that pays for lazily sized workspaces, with greedy and with sampling
 * (temperature, top-k, top-p). The prompt is 32 tokens because Metal's
 * attention takes its fast paths from 32 KV rows on (metal_attention,
 * ops.c), so the measured decode steps run the kernels production runs
 * after 31 tokens, not the fallback. The models are built in
 * memory with F32 weights (model_fixtures.h): a two-layer GQA llama, and a
 * Qwen3.5-style hybrid of three gated-DeltaNet blocks and an attention
 * block. Quantized-weight kernels have their own tests
 * (test_x86_kernel_no_alloc_unit and the cpu_neon sibling).
 *
 * It is a ratchet. Each KV mode has a ceiling of allocations per attention
 * layer per call, and each DeltaNet block one per prefill; above it a
 * check fails, below it the test says so, and the ceiling should be
 * lowered. All are at 0. The session keeps the DeltaNet chunked-prefill
 * staging (dn_run_prefill_chunked), sized by its first chunked prefill, and
 * Metal sizes the FP32 cache's f16 attention staging for the whole cache on
 * first use; the warm-up pays for both.
 */
#include "test_helpers.h"
#include "heap.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

constexpr int    STEPS   = 16;
constexpr size_t PREFILL = 32;

static const struct {
    enum geist_kv_mode kv;
    const char        *name;
    uint64_t           per_layer; /* allocations per layer per call, at most */
} MODES[] = {
        {GEIST_KV_INT8, "INT8", 0},
        {GEIST_KV_INT4, "INT4", 0},
        {GEIST_KV_KIVI, "KIVI", 0},
        {GEIST_KV_FP32, "FP32", 0},
};

/* One the build lacks, or Metal without a device, fails
 * geist_backend_create and is skipped. */
static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar", "metal"};

/* DeltaNet blocks' allocations per prefill call, at most. */
constexpr uint64_t DN_PER_PREFILL = 0;

struct model {
    const char   *name;
    struct tf_buf g;
    uint64_t      attn_layers, dn_blocks;
};

static int check(uint64_t            got,
                 uint64_t            ceiling,
                 const struct model *md,
                 const char         *backend,
                 const char         *mode,
                 bool                sampling,
                 const char         *what) {
    char msg[192];
    snprintf(msg,
             sizeof msg,
             "%s %s KV %s %s: at most %llu allocations in %s (got %llu)",
             backend,
             md->name,
             mode,
             sampling ? "sampling" : "greedy",
             (unsigned long long) ceiling,
             what,
             (unsigned long long) got);
    if (got < ceiling) {
        printf("  note: %s — the ceiling can come down\n", msg);
    }
    return geist_expect(got <= ceiling, msg);
}

static int run_session(const struct model   *md,
                       struct geist_model   *m,
                       struct geist_backend *be,
                       const char           *backend,
                       size_t                mode,
                       bool                  sampling) {
    struct geist_session_opts o = {.kv_mode = MODES[mode].kv, .top_p = 1.0f};
    if (sampling) {
        o.temperature = 0.8f;
        o.top_k       = 40;
        o.top_p       = 0.9f;
        o.random_seed = 7;
    }
    struct geist_session *s = nullptr;
    if (geist_session_create(m, be, &o, &s) != GEIST_OK) {
        fprintf(stderr, "FAIL: %s KV %s: session_create\n", backend, MODES[mode].name);
        return 1;
    }
    geist_token_t prompt[PREFILL];
    for (size_t i = 0; i < PREFILL; i++) {
        prompt[i] = (geist_token_t) (1 + 4 * (i % 8)); /* in both vocabularies */
    }
    geist_token_t tok = 0;
    bool          ok  = geist_session_prefill_tokens(s, PREFILL, prompt) == GEIST_OK;
    for (int i = 0; ok && i < 2; i++) { /* warm-up: lazily sized workspaces */
        ok = geist_session_decode_step(s, &tok) == GEIST_OK;
    }
    const uint64_t a0 = heap_alloc_count();
    for (int i = 0; ok && i < STEPS; i++) {
        ok = geist_session_decode_step(s, &tok) == GEIST_OK;
    }
    const uint64_t a1 = heap_alloc_count();
    ok                = ok && geist_session_prefill_tokens(s, PREFILL, prompt) == GEIST_OK;
    const uint64_t a2 = heap_alloc_count();
    geist_session_destroy(s);
    if (!ok) {
        fprintf(stderr,
                "FAIL: %s KV %s: a decode step or prefill failed\n",
                backend,
                MODES[mode].name);
        return 1;
    }
    const uint64_t per_call = MODES[mode].per_layer * md->attn_layers;
    return check(a1 - a0,
                 per_call * STEPS,
                 md,
                 backend,
                 MODES[mode].name,
                 sampling,
                 "16 decode steps") +
           check(a2 - a1,
                 per_call + DN_PER_PREFILL * md->dn_blocks,
                 md,
                 backend,
                 MODES[mode].name,
                 sampling,
                 "a 32-token prefill");
}

int main(void) {
    struct tf_vocab v        = tf_make_vocab("\xc4\xa0", false); /* the hybrid's vocabulary */
    struct model    models[] = {
            {"llama",
             mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                               .d_model  = 128,
                                               .heads    = 4,
                                               .kv_heads = 2,
                                               .ffn      = 256,
                                               .vocab    = 512,
                                               .context  = 256,
                                               .seed     = 7}),
             2,
             0},
            {"qwen35",
             mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
                                                 .interval   = 4,
                                                 .d_model    = 64,
                                                 .heads      = 4,
                                                 .kv_heads   = 2,
                                                 .head_dim   = 16,
                                                 .rope_dims  = 8,
                                                 .ffn        = 128,
                                                 .dn_k_heads = 2,
                                                 .dn_v_heads = 4,
                                                 .dn_head_k  = 16,
                                                 .dn_head_v  = 16,
                                                 .dn_conv    = 4,
                                                 .seed       = 1,
                                                 .tok        = &v}),
             1,
             3},
    };
    int fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue; /* not in this build */
        }
        ran++;
        for (size_t i = 0; i < sizeof models / sizeof models[0]; i++) {
            struct geist_model *m = nullptr;
            if (geist_model_load_from_memory(models[i].g.b, models[i].g.n, be, &m) != GEIST_OK) {
                fprintf(stderr,
                        "FAIL: %s %s: model load: %s\n",
                        BACKENDS[b],
                        models[i].name,
                        geist_last_create_error());
                fails++;
                continue;
            }
            for (size_t k = 0; k < sizeof MODES / sizeof MODES[0]; k++) {
                fails += run_session(&models[i], m, be, BACKENDS[b], k, false);
                fails += run_session(&models[i], m, be, BACKENDS[b], k, true);
            }
            geist_model_destroy(m);
        }
        geist_backend_destroy(be);
    }
    for (size_t i = 0; i < sizeof models / sizeof models[0]; i++) {
        free(models[i].g.b);
    }
    tf_free_vocab(&v);
    if (ran == 0 && fails == 0) {
        printf("SKIP: none of these backends in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: steady-state decode and prefill stay within their allocation ceilings\n");
    return GEIST_TEST_PASS;
}
