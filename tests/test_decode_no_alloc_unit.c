/*
 * test_decode_no_alloc_unit — steady-state decode steps and prefill calls
 * allocate nothing (AGENT.md §3: no heap allocation per token, per layer,
 * per block), checked through the public session API for every KV-cache
 * mode on every CPU backend in the build.
 *
 * heap.h counts successful allocations (heap_alloc_count), so the test
 * counts across 16 decode steps and one 8-token prefill after a warm-up
 * that pays for lazily sized workspaces, with greedy and with sampling
 * (temperature, top-k, top-p). The model is a two-layer GQA llama with F32
 * weights (model_fixtures.h); quantized-weight kernels have their own tests
 * (test_x86_kernel_no_alloc_unit and the cpu_neon sibling).
 *
 * It is a ratchet. Each KV mode has a ceiling of allocations per layer per
 * call: 0, except the FP32 cache, whose attention allocates a score buffer
 * on every call (cpu_x86 attention.c, the gemma4_kernels.c reference that
 * cpu_scalar runs, cpu_neon transformer_ops.c). Above its ceiling a mode
 * fails; below it the test says so, and the change that got it there
 * lowers the ceiling.
 */
#include "test_helpers.h"
#include "heap.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

constexpr uint32_t LAYERS  = 2;
constexpr int      STEPS   = 16;
constexpr size_t   PREFILL = 8;

static const struct {
    enum geist_kv_mode kv;
    const char        *name;
    uint64_t           per_layer; /* allocations per layer per call, at most */
} MODES[] = {
        {GEIST_KV_INT8, "INT8", 0},
        {GEIST_KV_INT4, "INT4", 0},
        {GEIST_KV_KIVI, "KIVI", 0},
        {GEIST_KV_FP32, "FP32", 1}, /* the score buffer, see above */
};

/* cpu_neon is counted by reading its code, not by running it here: its
 * FP32 attention allocates like the others, once per call. */
static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};

static int check(uint64_t    got,
                 uint64_t    ceiling,
                 const char *backend,
                 const char *mode,
                 bool        sampling,
                 const char *what) {
    char msg[192];
    snprintf(msg,
             sizeof msg,
             "%s KV %s %s: at most %llu allocations in %s (got %llu)",
             backend,
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

static int run_session(struct geist_model   *m,
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
    const geist_token_t prompt[PREFILL] = {1, 5, 9, 13, 17, 21, 25, 29};
    geist_token_t       tok             = 0;
    bool                ok = geist_session_prefill_tokens(s, PREFILL, prompt) == GEIST_OK;
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
    const uint64_t per_call = MODES[mode].per_layer * LAYERS;
    return check(a1 - a0,
                 per_call * STEPS,
                 backend,
                 MODES[mode].name,
                 sampling,
                 "16 decode steps") +
           check(a2 - a1, per_call, backend, MODES[mode].name, sampling, "an 8-token prefill");
}

int main(void) {
    struct tf_buf g     = mf_llama_gguf(&(struct mf_llama) {.layers   = LAYERS,
                                                            .d_model  = 128,
                                                            .heads    = 4,
                                                            .kv_heads = 2,
                                                            .ffn      = 256,
                                                            .vocab    = 512,
                                                            .context  = 256,
                                                            .seed     = 7});
    int           fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue; /* not in this build */
        }
        struct geist_model *m = nullptr;
        if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK) {
            fprintf(stderr, "FAIL: %s: model load: %s\n", BACKENDS[b], geist_last_create_error());
            fails++;
        } else {
            ran++;
            for (size_t k = 0; k < sizeof MODES / sizeof MODES[0]; k++) {
                fails += run_session(m, be, BACKENDS[b], k, false);
                fails += run_session(m, be, BACKENDS[b], k, true);
            }
            geist_model_destroy(m);
        }
        geist_backend_destroy(be);
    }
    free(g.b);
    if (ran == 0 && fails == 0) {
        printf("SKIP: no CPU backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: steady-state decode and prefill stay within their allocation ceilings\n");
    return GEIST_TEST_PASS;
}
