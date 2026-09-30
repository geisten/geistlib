/*
 * test_decode_cadence_unit — what a caller of the session API sees around
 * geist_session_decode_step, whatever the architecture does with the step's
 * forward pass internally: when it runs is not observable, only its result.
 *
 * Checked on two models built in memory (model_fixtures.h), a two-layer
 * llama and a Qwen3.5-style hybrid (three gated-DeltaNet blocks and an
 * attention block, so a recurrent state beside the KV cache), for every CPU
 * backend in the build, every KV-cache mode, greedy and sampling:
 *
 *   - peek_logits between decode steps shows the next position's logits:
 *     new values after every step, the same values whichever steps are
 *     peeked, and (greedy) the next step's token as their argmax. The
 *     tokens do not depend on peeking at all.
 *   - a second turn (prefill after decoding) sees the same cache with or
 *     without a peek at the turn boundary.
 *   - reset: nothing is pending afterwards, and a prefill after it gives the
 *     logits of a fresh session, also back to a pinned prefix; so does a
 *     pin after decode steps. The hybrid refuses to pin a prefix (a reset
 *     cannot return its recurrent state to one).
 *   - a full context: exactly as many tokens as fit, then
 *     GEIST_E_TOO_MANY_TOKENS on every further step, the pending logits
 *     still readable.
 *   - decode_speculative after a decode step starts with the token the next
 *     decode step would have returned, with drafts to verify or without.
 *
 * Logits are compared bit for bit: the fixture's greedy decoding repeats
 * tokens, so equal tokens alone would not show which position the logits
 * belong to.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t VOCAB  = 512; /* most tokens a model here has */
constexpr size_t PROMPT = 8;
constexpr size_t STEPS  = 12;
constexpr size_t TURN2  = 5;
constexpr size_t STEPS2 = 6;
constexpr size_t ROOM   = 5; /* decode steps that fit after the prompt */

static const geist_token_t P1[PROMPT] = {1, 5, 9, 13, 17, 21, 25, 29};
static const geist_token_t P2[TURN2]  = {40, 41, 42, 43, 44};
static const geist_token_t PIN[4]     = {2, 3, 4, 6};

static const struct {
    enum geist_kv_mode kv;
    const char        *name;
} MODES[] = {
        {GEIST_KV_INT8, "INT8"},
        {GEIST_KV_INT4, "INT4"},
        {GEIST_KV_KIVI, "KIVI"},
        {GEIST_KV_FP32, "FP32"},
};

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};

struct run {
    struct geist_model   *m;
    struct geist_backend *be;
    const char           *backend;
    const char           *model;
    size_t                vocab;
    bool                  recurrent; /* DeltaNet layers: no pinned prefix */
    size_t                mode;
    bool                  sampling;
};

/* What a plain prefill of P1 and STEPS decode steps produce. */
struct ref {
    geist_token_t tok[STEPS];
    float         logits[VOCAB]; /* after the prefill */
};

static struct geist_session *open_session(const struct run *r, size_t max_seq_len) {
    struct geist_session_opts o = {
            .kv_mode = MODES[r->mode].kv, .top_p = 1.0f, .max_seq_len = max_seq_len};
    if (r->sampling) {
        o.temperature = 0.8f;
        o.top_k       = 40;
        o.top_p       = 0.9f;
        o.random_seed = 7;
    }
    struct geist_session *s = nullptr;
    if (geist_session_create(r->m, r->be, &o, &s) != GEIST_OK) {
        return nullptr;
    }
    return s;
}

static geist_token_t argmax(size_t n, const float v[static n]) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return (geist_token_t) best;
}

static int expect(const struct run *r, bool cond, const char *what) {
    char msg[192];
    snprintf(msg,
             sizeof msg,
             "%s %s KV %s %s: %s",
             r->backend,
             r->model,
             MODES[r->mode].name,
             r->sampling ? "sampling" : "greedy",
             what);
    return geist_expect(cond, msg);
}

/* n decode steps into out; false if one fails. */
static bool decode(struct geist_session *s, size_t n, geist_token_t out[static n]) {
    for (size_t i = 0; i < n; i++) {
        if (geist_session_decode_step(s, &out[i]) != GEIST_OK) {
            return false;
        }
    }
    return true;
}

/* A copy of the pending logits into dst[r->vocab]; false if none are
 * pending. */
static bool peek(const struct run *r, struct geist_session *s, float *dst) {
    size_t       n = 0;
    const float *p = geist_session_peek_logits(&n, s);
    if (p == nullptr || n != r->vocab) {
        return false;
    }
    memcpy(dst, p, r->vocab * sizeof *dst);
    return true;
}

static bool same(const struct run *r, const float *a, const float *b) {
    return memcmp(a, b, r->vocab * sizeof *a) == 0;
}

/* Tokens and logits with peeks between the decode steps. */
static int check_peeks(const struct run *r, const struct ref *ref) {
    int                   fails = 0;
    float                *lg    = xmalloc((STEPS + 1) * r->vocab * sizeof *lg);
    geist_token_t         got[STEPS];
    struct geist_session *s = open_session(r, 0);
    bool ok = s != nullptr && geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK;
    ok      = ok && peek(r, s, lg);
    for (size_t i = 0; ok && i < STEPS; i++) {
        ok = geist_session_decode_step(s, &got[i]) == GEIST_OK &&
             peek(r, s, lg + (i + 1) * r->vocab);
    }
    geist_session_destroy(s);
    fails += expect(r, ok, "prefill, decode steps and a peek after each");
    if (!ok) {
        free(lg);
        return fails;
    }
    fails += expect(r,
                    memcmp(got, ref->tok, sizeof got) == 0,
                    "a peek after every step leaves the tokens as they are without");
    bool moved = true;
    bool next  = true;
    for (size_t i = 0; i < STEPS; i++) {
        moved = moved && !same(r, lg + i * r->vocab, lg + (i + 1) * r->vocab);
        next  = next && argmax(r->vocab, lg + i * r->vocab) == ref->tok[i];
    }
    fails += expect(r, moved, "every decode step moves the peeked logits");
    if (!r->sampling) {
        fails += expect(r, next, "the logits peeked before a step have its token as argmax");
    }

    /* Peeks at odd steps only: the logits there are the same values. */
    float *one = xmalloc(r->vocab * sizeof *one);
    bool   eq  = true;
    s          = open_session(r, 0);
    ok         = s != nullptr && geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK;
    for (size_t i = 0; ok && i < STEPS; i++) {
        ok = geist_session_decode_step(s, &got[i]) == GEIST_OK;
        if (ok && i % 2 == 1) {
            ok = peek(r, s, one);
            eq = eq && same(r, one, lg + (i + 1) * r->vocab);
        }
    }
    geist_session_destroy(s);
    fails += expect(
            r, ok && memcmp(got, ref->tok, sizeof got) == 0, "tokens with peeks at odd steps");
    fails += expect(r, ok && eq, "logits peeked at odd steps equal those peeked at every step");
    free(one);
    free(lg);
    return fails;
}

/* A second turn after decoding, with and without a peek at the boundary. */
static int check_turns(const struct run *r) {
    geist_token_t t[2][STEPS + STEPS2];
    float        *lg    = xmalloc(3 * r->vocab * sizeof *lg); /* [2]: scratch */
    int           fails = 0;
    for (size_t with_peek = 0; with_peek < 2; with_peek++) {
        struct geist_session *s = open_session(r, 0);
        bool ok = s != nullptr && geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK &&
                  decode(s, STEPS, t[with_peek]);
        if (ok && with_peek) {
            ok = peek(r, s, lg + 2 * r->vocab);
        }
        ok = ok && geist_session_prefill_tokens(s, TURN2, P2) == GEIST_OK &&
             peek(r, s, lg + with_peek * r->vocab) && decode(s, STEPS2, t[with_peek] + STEPS);
        geist_session_destroy(s);
        fails += expect(r, ok, "two turns of prefill and decode");
        if (!ok) {
            free(lg);
            return fails;
        }
    }
    fails += expect(r,
                    same(r, lg, lg + r->vocab),
                    "a peek between turns leaves the logits after the second prefill as they are");
    fails += expect(r, memcmp(t[0], t[1], sizeof t[0]) == 0, "and the tokens of both turns");
    if (!r->sampling) {
        fails += expect(r,
                        argmax(r->vocab, lg) == t[0][STEPS],
                        "the logits after the second prefill have its first token as argmax");
    }
    free(lg);
    return fails;
}

/* Reset after decoding: nothing pending, then a fresh session's logits. */
static int check_reset(const struct run *r, const struct ref *ref) {
    int                   fails = 0;
    geist_token_t         got[STEPS];
    geist_token_t         t  = -1;
    size_t                n  = 1;
    float                *lg = xmalloc(2 * r->vocab * sizeof *lg);
    struct geist_session *s  = open_session(r, 0);
    bool ok = s != nullptr && geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK &&
              decode(s, STEPS, got) && geist_session_reset(s) == GEIST_OK;
    fails += expect(r, ok, "prefill, decode steps and a reset");
    if (ok) {
        fails += expect(r,
                        geist_session_decode_step(s, &t) == GEIST_E_INVALID_STATE && t == -1,
                        "after a reset a decode step has nothing pending");
        fails += expect(r,
                        geist_session_peek_logits(&n, s) == nullptr && n == 0,
                        "after a reset there are no logits to peek");
        ok = geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK && peek(r, s, lg) &&
             decode(s, STEPS, got);
        fails += expect(r, ok, "prefill and decode steps after a reset");
        if (ok) {
            fails += expect(r,
                            same(r, lg, ref->logits),
                            "a prefill after a reset gives the logits of a fresh session");
            if (!r->sampling) {
                fails += expect(
                        r, memcmp(got, ref->tok, sizeof got) == 0, "greedy tokens after a reset");
            }
        }
    }
    geist_session_destroy(s);

    if (r->recurrent) {
        s  = open_session(r, 0);
        ok = s != nullptr &&
             geist_session_pin_prefix(s, sizeof PIN / sizeof PIN[0], PIN) == GEIST_E_UNSUPPORTED &&
             geist_session_pin_prefix(s, 0, PIN) == GEIST_OK;
        geist_session_destroy(s);
        fails += expect(r, ok, "a prefix is refused, pinning nothing is not");
        free(lg);
        return fails;
    }
    geist_token_t first[STEPS];
    s  = open_session(r, 0);
    ok = s != nullptr && geist_session_pin_prefix(s, sizeof PIN / sizeof PIN[0], PIN) == GEIST_OK &&
         geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK && peek(r, s, lg) &&
         decode(s, STEPS, first) && geist_session_reset(s) == GEIST_OK &&
         geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK && peek(r, s, lg + r->vocab) &&
         decode(s, STEPS, got);
    geist_session_destroy(s);
    fails += expect(r, ok, "pin, prefill, decode, reset to the pin, prefill, decode");
    if (!ok) {
        free(lg);
        return fails;
    }
    fails += expect(r,
                    same(r, lg, lg + r->vocab),
                    "a prefill after a reset to a pinned prefix gives the same logits");
    if (!r->sampling) {
        fails += expect(r,
                        memcmp(got, first, sizeof got) == 0,
                        "greedy tokens after a reset to a pinned prefix");
    }

    /* Pinning after decode steps starts over as on a fresh session. */
    s  = open_session(r, 0);
    ok = s != nullptr && geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK &&
         decode(s, STEPS, got) &&
         geist_session_pin_prefix(s, sizeof PIN / sizeof PIN[0], PIN) == GEIST_OK &&
         geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK && peek(r, s, lg + r->vocab);
    geist_session_destroy(s);
    fails += expect(r, ok, "prefill, decode, pin, prefill");
    if (ok) {
        fails += expect(r,
                        same(r, lg, lg + r->vocab),
                        "a pin after decode steps gives the logits of a pin on a fresh session");
    }
    free(lg);
    return fails;
}

/* A context with room for ROOM tokens after the prompt. */
static int check_full(const struct run *r, const struct ref *ref) {
    int                   fails = 0;
    geist_token_t         got[ROOM];
    geist_token_t         t  = -1;
    float                *lg = xmalloc(r->vocab * sizeof *lg);
    struct geist_session *s  = open_session(r, PROMPT + ROOM);
    bool ok = s != nullptr && geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK &&
              decode(s, ROOM, got);
    fails += expect(r, ok, "decode steps up to a full context");
    if (ok) {
        fails += expect(r, memcmp(got, ref->tok, sizeof got) == 0, "the tokens that fit");
        fails += expect(r,
                        geist_session_decode_step(s, &t) == GEIST_E_TOO_MANY_TOKENS && t == -1,
                        "the step past a full context fails and writes no token");
        fails += expect(r,
                        geist_session_decode_step(s, &t) == GEIST_E_TOO_MANY_TOKENS && t == -1,
                        "so does the next one");
        const bool peeked = peek(r, s, lg);
        fails += expect(r, peeked, "the pending logits stay readable in a full context");
        if (peeked && !r->sampling) {
            fails += expect(r,
                            argmax(r->vocab, lg) == ref->tok[ROOM],
                            "their argmax is the token that did not fit");
        }
    }
    geist_session_destroy(s);
    free(lg);
    return fails;
}

/* Speculative decode after a decode step starts where it left off: with
 * the history of the session, where the drafter finds nothing and one
 * decode step stands in, and with a history made up to hold a match for
 * its first guess, whichever token that is, so that verify_forward runs
 * the drafts. */
static int check_speculative(const struct run *r, const struct ref *ref) {
    const geist_token_t a = ref->tok[0], b = ref->tok[1];
    geist_token_t       made[] = {a, a, 60, 61, 62, a, b, 70, 71, 72, a};
    geist_token_t       real[PROMPT + 1];
    memcpy(real, P1, sizeof P1);
    int fails = 0;
    for (int m = 0; m < 2; m++) {
        geist_token_t         out[5] = {0};
        size_t                n_out  = 0;
        const size_t          n_hist = m ? sizeof made / sizeof made[0] : PROMPT + 1;
        struct geist_session *s      = open_session(r, 0);
        bool ok = s != nullptr && geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK &&
                  decode(s, 1, real + PROMPT) &&
                  geist_session_decode_speculative(s, 4, n_hist, m ? made : real, 5, out, &n_out) ==
                          GEIST_OK;
        geist_session_destroy(s);
        fails += expect(r,
                        ok && n_out >= 1 && out[0] == b,
                        m ? "speculative decode with drafts starts with the next step's token"
                          : "speculative decode without drafts starts with the next step's token");
    }
    return fails;
}

static int run_all(const struct run *r) {
    struct ref            ref;
    struct geist_session *s = open_session(r, 0);
    bool ok = s != nullptr && geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK &&
              peek(r, s, ref.logits) && decode(s, STEPS, ref.tok);
    geist_session_destroy(s);
    int fails = expect(r, ok, "prefill and decode steps");
    if (!ok) {
        return fails;
    }
    fails += check_peeks(r, &ref);
    fails += check_turns(r);
    fails += check_reset(r, &ref);
    fails += check_full(r, &ref);
    fails += check_speculative(r, &ref);
    return fails;
}

int main(void) {
    struct tf_vocab v = tf_make_vocab("\xc4\xa0", false); /* the hybrid's vocabulary */
    /* Seeds whose greedy decoding changes token early on. */
    const struct {
        const char   *name;
        struct tf_buf g;
        size_t        vocab;
        bool          recurrent;
    } models[] = {
            {"llama",
             mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                               .d_model  = 128,
                                               .heads    = 4,
                                               .kv_heads = 2,
                                               .ffn      = 256,
                                               .vocab    = VOCAB,
                                               .context  = 256,
                                               .seed     = 14}),
             VOCAB,
             false},
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
             v.n_tok,
             true},
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
                for (int sampling = 0; sampling < 2; sampling++) {
                    const struct run r = {m,
                                          be,
                                          BACKENDS[b],
                                          models[i].name,
                                          models[i].vocab,
                                          models[i].recurrent,
                                          k,
                                          sampling != 0};
                    fails += run_all(&r);
                }
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
        printf("SKIP: no CPU backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: decode steps, peeks, turns, resets and a full context behave as specified\n");
    return GEIST_TEST_PASS;
}
