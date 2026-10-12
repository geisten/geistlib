/*
 * test_decision_prefix_int — prefix scoring (#5) on a real GGUF
 * (GEIST_GGUF_PATH, backend GEIST_TEST_BACKEND), e.g. Qwen3.5 or Ternary
 * Bonsai 2 (hybrid: gated-DeltaNet conv and recurrent state beside the KV
 * cache) on CPU and Metal.
 *
 * The prefix spans two of the model's own prefill chunks. Each question's
 * candidate logits must be bit-identical to a fresh session that prefills
 * the prefix and then the question; with the cut on a chunk boundary also
 * to scoring the joined prompt in one call. Questions are repeated,
 * interleaved and asked through a second handle sharing the prefix. Times
 * per question with and without the cached prefix are printed.
 */
#include "test_helpers.h"

#include <geist_decision.h>
#include <geist_util.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>

constexpr size_t NCAND = 4;
constexpr size_t NQ    = 3;
constexpr size_t QLEN  = 16;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec * 1e3 + (double) t.tv_nsec / 1e6;
}

int main(void) {
    if (!geist_decision_prefix_available()) {
        GEIST_SKIP("DECISION_CONTEXT=0");
    }
    GEIST_REQUIRE_GGUF(path);
    const char                   *backend = getenv("GEIST_TEST_BACKEND");
    struct geist_backend         *be      = nullptr;
    struct geist_model           *m       = nullptr;
    struct geist_decision        *a = nullptr, *b = nullptr;
    struct geist_decision_prefix *p      = nullptr;
    geist_token_t                *prompt = nullptr;
    int                           fails  = 0;
    constexpr size_t              CAP    = 4096;
    struct geist_decision_opts    o      = {
            .max_prompt_tokens = CAP, .max_candidates = NCAND, .kv_mode = GEIST_KV_FP32};
    if (geist_backend_create(backend != nullptr ? backend : "auto", nullptr, nullptr, &be) !=
                GEIST_OK ||
        geist_model_load(path, be, &m) != GEIST_OK ||
        geist_decision_create(m, be, &o, &a) != GEIST_OK) {
        fprintf(stderr, "prefix setup: %s\n", geist_last_create_error());
        fails++;
        goto done;
    }
    size_t chunk = 0;
    if (geist_decision_prefill_chunk(&chunk, a) != GEIST_OK) {
        fprintf(stderr, "prefix capability: %s\n", geist_decision_errmsg(a));
        fails++;
        goto done;
    }
    const size_t n_prefix = 2 * chunk + QLEN * NQ <= CAP ? 2 * chunk : chunk;
    const size_t vocab    = geist_decision_vocab_size(a);
    prompt                = calloc(n_prefix + QLEN, sizeof *prompt);
    if (prompt == nullptr || vocab < 1000) {
        fails += geist_expect(false, "prompt buffer and a real vocabulary");
        goto done;
    }
    /* Deterministic ordinary token ids, away from the special tokens at
     * both ends of the vocabulary. */
    for (size_t i = 0; i < n_prefix; i++) {
        prompt[i] = (geist_token_t) (100 + (i * 7919) % (vocab / 2));
    }
    geist_token_t q[NQ][QLEN];
    for (size_t k = 0; k < NQ; k++) {
        for (size_t i = 0; i < QLEN; i++) {
            q[k][i] = (geist_token_t) (200 + (k * 104729 + i * 31) % (vocab / 2));
        }
    }
    const geist_token_t cand[NCAND] = {100, 200, 300, 400};
    o.m_max                         = chunk; /* the reference sessions split alike */
    if (geist_decision_create(m, be, &o, &b) != GEIST_OK) {
        fails += geist_expect(false, "second handle creates");
        goto done;
    }
    double t0 = now_ms();
    if (geist_decision_prefix_prepare(a, SIZE_MAX, n_prefix, prompt, &p) != GEIST_OK) {
        fprintf(stderr, "prepare: %s\n", geist_decision_errmsg(a));
        fails++;
        goto done;
    }
    const double                      t_prepare = now_ms() - t0;
    struct geist_decision_prefix_info info      = {0};
    (void) geist_decision_prefix_get_info(p, &info);
    printf("%s, %s: chunk %zu, prefix %zu tokens, snapshot %.1f MiB, prepare %.0f ms\n",
           path,
           geist_backend_name(be),
           chunk,
           n_prefix,
           (double) info.snapshot_bytes / (1 << 20),
           t_prepare);

    float                        first[NQ][NCAND];
    double                       t_prefix = 0, t_full = 0;
    struct geist_decision_result r = {0};
    for (size_t k = 0; k < NQ; k++) {
        /* Reference: one fresh session, prefill(prefix), prefill(question). */
        const struct geist_session_opts so = {
                .max_seq_len = CAP, .kv_mode = GEIST_KV_FP32, .m_max = chunk, .top_p = 1.0f};
        struct geist_session *s   = nullptr;
        size_t                nl  = 0;
        const float          *ref = nullptr;
        if (geist_session_create(m, be, &so, &s) == GEIST_OK &&
            geist_session_prefill_tokens(s, n_prefix, prompt) == GEIST_OK &&
            geist_session_prefill_tokens(s, QLEN, q[k]) == GEIST_OK) {
            ref = geist_session_peek_logits(&nl, s);
        }
        t0 = now_ms();
        const bool scored =
                geist_decision_score_from_prefix(a, p, QLEN, NCAND, q[k], cand, &r) == GEIST_OK;
        t_prefix += now_ms() - t0;
        bool exact = scored && ref != nullptr;
        for (size_t i = 0; exact && i < NCAND; i++) {
            exact       = r.logits[i] == ref[cand[i]];
            first[k][i] = r.logits[i];
        }
        fails += geist_expect(exact, "restored prefix + question is bit-identical to fresh");
        geist_session_destroy(s);

        memcpy(prompt + n_prefix, q[k], sizeof q[k]);
        t0 = now_ms();
        const bool all =
                geist_decision_score(b, n_prefix + QLEN, NCAND, prompt, cand, &r) == GEIST_OK;
        t_full += now_ms() - t0;
        bool joined = all;
        for (size_t i = 0; joined && i < NCAND; i++) {
            joined = r.logits[i] == first[k][i];
        }
        fails += geist_expect(joined, "on the chunk boundary the joined prompt matches exactly");
    }
    /* Interleaved and reordered, through both handles. */
    for (size_t round = 0; round < 2; round++) {
        for (size_t j = 0; j < NQ; j++) {
            const size_t           k = round == 0 ? NQ - 1 - j : (j * 2) % NQ;
            struct geist_decision *d = (j + round) % 2 ? b : a;
            const bool ok = geist_decision_score_from_prefix(d, p, QLEN, NCAND, q[k], cand, &r) ==
                                    GEIST_OK &&
                            memcmp(r.logits, first[k], sizeof first[k]) == 0;
            fails += geist_expect(ok, "a repeated, reordered question gives its first answer");
        }
    }
    printf("per question: %.0f ms from the cached prefix, %.0f ms full prompt (%.1fx)\n",
           t_prefix / NQ,
           t_full / NQ,
           t_prefix > 0 ? t_full / t_prefix : 0.0);
done:
    free(prompt);
    geist_decision_prefix_destroy(p);
    geist_decision_destroy(b);
    geist_decision_destroy(a);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    printf("decision prefix: real model %s\n", fails ? "FAIL" : "PASS");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
