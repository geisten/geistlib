/*
 * test_deltanet_prefill_ws_unit — a chunked DeltaNet prefill does not
 * depend on what earlier calls left in the memory it stages in.
 *
 * dn_run_prefill_chunked (layer_deltanet.c) stages the conv output, the
 * gating scalars, the previous conv state and a delta-rule workspace per
 * OpenMP thread for every chunk of a prefill; the layout moves with the
 * chunk length and the thread count. Two sessions on a Qwen3.5-style
 * hybrid (model_fixtures.h), both with m_max 5, so a 12-token prompt runs
 * as chunks of 5, 5 and 2:
 *   - fresh: prefill the prompt;
 *   - used:  prefill 3 tokens on one thread, reset, prefill 7 on every
 *            thread, reset, prefill the prompt.
 * The logits after the prompt and after each of the greedy decode steps
 * that follow must be the same bits. The decode steps read the recurrent
 * state the prefill left, which the logits after it do not all show.
 * Every CPU backend in the build.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

constexpr size_t M_MAX  = 5;
constexpr size_t PROMPT = 12;
constexpr size_t STEPS  = 4;

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};

static const geist_token_t prompt[PROMPT] = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45};
static const geist_token_t other[7]       = {2, 4, 6, 8, 10, 12, 14};

static void set_threads(int n) {
#if defined(_OPENMP)
    omp_set_num_threads(n);
#else
    (void) n;
#endif
}

/* The logits pending now into dst; false if there are none. */
static bool peek(struct geist_session *s, size_t vocab, float dst[static vocab]) {
    size_t       n = 0;
    const float *p = geist_session_peek_logits(&n, s);
    if (p == nullptr || n != vocab) {
        return false;
    }
    memcpy(dst, p, vocab * sizeof *dst);
    return true;
}

/* Prefill the prompt, then STEPS greedy decode steps; logits gets the
 * logits pending after the prefill and after each step. */
static bool run_prompt(struct geist_session *s,
                       size_t                vocab,
                       float                 logits[static(STEPS + 1) * vocab],
                       geist_token_t         tok[static STEPS]) {
    if (geist_session_prefill_tokens(s, PROMPT, prompt) != GEIST_OK || !peek(s, vocab, logits)) {
        return false;
    }
    for (size_t i = 0; i < STEPS; i++) {
        if (geist_session_decode_step(s, &tok[i]) != GEIST_OK ||
            !peek(s, vocab, logits + (i + 1) * vocab)) {
            return false;
        }
    }
    return true;
}

static int run_backend(const char           *backend,
                       struct geist_model   *m,
                       struct geist_backend *be,
                       size_t                vocab,
                       int                   threads) {
    const struct geist_session_opts o = {.top_p = 1.0f, .m_max = M_MAX};
    float                          *a = calloc((STEPS + 1) * vocab, sizeof *a);
    float                          *b = calloc((STEPS + 1) * vocab, sizeof *b);
    geist_token_t                   ta[STEPS], tb[STEPS];
    struct geist_session           *fresh = nullptr, *used = nullptr;
    bool ok = a != nullptr && b != nullptr && geist_session_create(m, be, &o, &fresh) == GEIST_OK &&
              geist_session_create(m, be, &o, &used) == GEIST_OK;

    set_threads(threads);
    ok = ok && run_prompt(fresh, vocab, a, ta);

    set_threads(1);
    ok = ok && geist_session_prefill_tokens(used, 3, other) == GEIST_OK &&
         geist_session_reset(used) == GEIST_OK;
    set_threads(threads);
    ok = ok && geist_session_prefill_tokens(used, 7, other) == GEIST_OK &&
         geist_session_reset(used) == GEIST_OK && run_prompt(used, vocab, b, tb);

    char msg[160];
    snprintf(msg, sizeof msg, "%s: prefills and decode steps run", backend);
    int fails = geist_expect(ok, msg);
    if (ok) {
        snprintf(msg,
                 sizeof msg,
                 "%s: logits after the prompt and %zu decode steps are the same bits",
                 backend,
                 STEPS);
        fails += geist_expect(memcmp(a, b, (STEPS + 1) * vocab * sizeof *a) == 0, msg);
        snprintf(msg, sizeof msg, "%s: the same greedy tokens", backend);
        fails += geist_expect(memcmp(ta, tb, sizeof ta) == 0, msg);
    }
    geist_session_destroy(fresh);
    geist_session_destroy(used);
    free(a);
    free(b);
    return fails;
}

int main(void) {
    /* The chunked path is what this is about; the flag forcing the
     * sequential one is read at model load. */
    unsetenv("GEIST_DN_SEQ_PREFILL");
#if defined(_OPENMP)
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif
    struct tf_vocab v     = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf   g     = mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
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
                                                                .tok        = &v});
    int             fails = 0, ran = 0;
    for (size_t i = 0; i < sizeof BACKENDS / sizeof BACKENDS[0]; i++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[i], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue; /* not in this build */
        }
        ran++;
        struct geist_model *m = nullptr;
        if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK) {
            fprintf(stderr, "FAIL: %s: model load: %s\n", BACKENDS[i], geist_last_create_error());
            fails++;
        } else {
            fails += run_backend(BACKENDS[i], m, be, v.n_tok, threads);
            geist_model_destroy(m);
        }
        geist_backend_destroy(be);
    }
    set_threads(threads);
    free(g.b);
    tf_free_vocab(&v);
    if (ran == 0) {
        printf("SKIP: no CPU backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails == 0) {
        printf("PASS: a chunked DeltaNet prefill gives the same bits whatever ran before it\n");
    }
    return fails == 0 ? 0 : 1;
}
