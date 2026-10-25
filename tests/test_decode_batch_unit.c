/*
 * test_decode_batch_unit — geist_session_decode_batch (#26) on models built in
 * memory (model_fixtures.h), for every backend in the build:
 *
 *   - geist_model_decode_batch_max: positive for the Qwen3.5-style hybrid
 *     where the backend has the fused DeltaNet mixer, 0 for a llama and on
 *     backends without it; n = 2 is then refused and nothing advances.
 *   - n = 1 is decode_step: tokens and logits bit for bit.
 *   - Isolation: a session's tokens and pending logits at every step are
 *     bit-identical whichever other sessions share the batch and whichever
 *     slot it has, greedy and sampling.
 *   - Agreement: batched greedy decode stays close to decode_step (the
 *     batched rows round differently; tokens and a logit bound are checked).
 *   - Refusals change nothing: a session with nothing pending, a repeated
 *     session, a session of another model, a full context, n above the max.
 *   - A member that owes a forward from an earlier decode_step decodes in a
 *     batch too, and every member advances by exactly one position.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t VOCAB = 512; /* most tokens a model here has */
constexpr size_t STEPS = 6;
constexpr size_t NP    = 8; /* prompts */
constexpr size_t CTX   = 64;

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar", "vulkan", "metal"};

/* Eight prompts of different lengths and tokens. */
static size_t prompt(size_t p, geist_token_t ids[static 16]) {
    const size_t n = 4 + p;
    for (size_t i = 0; i < n; i++)
        ids[i] = (geist_token_t) (1 + (p * 37 + i * 11) % 200);
    return n;
}

struct trace {
    geist_token_t tok[STEPS];
    float         logits[STEPS][VOCAB]; /* pending before each step */
};

struct run {
    struct geist_model   *m;
    struct geist_model   *other;
    struct geist_backend *be;
    const char           *backend;
    size_t                vocab;
    bool                  sampling;
};

static int expect(const struct run *r, bool cond, const char *what) {
    if (!cond)
        fprintf(stderr, "FAIL: %s %s: %s\n", r->backend, r->sampling ? "sampling" : "greedy", what);
    return cond ? 0 : 1;
}

static struct geist_session *open_session(const struct run *r, struct geist_model *m, size_t ctx) {
    struct geist_session_opts o = {.kv_mode = GEIST_KV_FP32, .top_p = 1.0f, .max_seq_len = ctx};
    if (r->sampling) {
        o.temperature = 0.8f;
        o.top_k       = 40;
        o.random_seed = 7;
    }
    struct geist_session *s = nullptr;
    return geist_session_create(m, r->be, &o, &s) == GEIST_OK ? s : nullptr;
}

static struct geist_session *open_prompted(const struct run *r, size_t p) {
    struct geist_session *s = open_session(r, r->m, CTX);
    geist_token_t         ids[16];
    const size_t          n = prompt(p, ids);
    if (s != nullptr && geist_session_prefill_tokens(s, n, ids) != GEIST_OK) {
        geist_session_destroy(s);
        return nullptr;
    }
    return s;
}

static bool peek(const struct run *r, struct geist_session *s, float dst[static VOCAB]) {
    size_t       n = 0;
    const float *l = geist_session_peek_logits(&n, s);
    if (l == nullptr || n != r->vocab)
        return false;
    memcpy(dst, l, n * sizeof *l);
    return true;
}

/* decode_step trace of prompt p. */
static bool reference(const struct run *r, size_t p, struct trace *t) {
    struct geist_session *s  = open_prompted(r, p);
    bool                  ok = s != nullptr;
    for (size_t i = 0; ok && i < STEPS; i++)
        ok = peek(r, s, t->logits[i]) && geist_session_decode_step(s, &t->tok[i]) == GEIST_OK;
    geist_session_destroy(s);
    return ok;
}

/* Prompts ps[0..n-1] decoded together; traces of every member. */
static bool
batched(const struct run *r, size_t n, const size_t ps[static n], struct trace t[static n]) {
    struct geist_session *ss[8] = {};
    bool                  ok    = true;
    for (size_t b = 0; b < n; b++)
        ok = ok && (ss[b] = open_prompted(r, ps[b])) != nullptr;
    for (size_t i = 0; ok && i < STEPS; i++) {
        for (size_t b = 0; ok && b < n; b++)
            ok = peek(r, ss[b], t[b].logits[i]);
        geist_token_t out[8];
        ok = ok && geist_session_decode_batch(n, ss, out) == GEIST_OK;
        for (size_t b = 0; ok && b < n; b++)
            t[b].tok[i] = out[b];
    }
    for (size_t b = 0; b < n; b++)
        geist_session_destroy(ss[b]);
    return ok;
}

static bool same(const struct run *r, const struct trace *a, const struct trace *b) {
    for (size_t i = 0; i < STEPS; i++)
        if (a->tok[i] != b->tok[i] ||
            memcmp(a->logits[i], b->logits[i], r->vocab * sizeof(float)) != 0)
            return false;
    return true;
}

static float max_diff(const struct run *r, const struct trace *a, const struct trace *b) {
    float m = 0;
    for (size_t i = 0; i < STEPS; i++)
        for (size_t k = 0; k < r->vocab; k++) {
            const float d = fabsf(a->logits[i][k] - b->logits[i][k]);
            m             = d > m ? d : m;
        }
    return m;
}

/* A refused call must leave every member as it was: same length, same
 * pending logits. */
static int check_refusals(const struct run *r, size_t max) {
    int                   fails = 0;
    struct geist_session *a = open_prompted(r, 0), *b = open_prompted(r, 1);
    struct geist_session *fresh = open_session(r, r->m, CTX);
    struct geist_session *alien = open_session(r, r->other, CTX);
    if (a == nullptr || b == nullptr || fresh == nullptr) {
        fails += expect(r, false, "refusal setup");
        goto done;
    }
    static float before[VOCAB], after[VOCAB];
    fails += expect(r, peek(r, a, before), "peek before");
    const size_t          len_a = geist_session_length(a), len_b = geist_session_length(b);
    geist_token_t         out[9];
    struct geist_session *nothing[2] = {a, fresh}, *twice[2] = {a, a}, *mixed[2] = {a, alien};
    fails += expect(r,
                    geist_session_decode_batch(2, nothing, out) == GEIST_E_INVALID_STATE &&
                            out[0] == -1 && out[1] == -1,
                    "a member with nothing pending is GEIST_E_INVALID_STATE, out -1");
    fails += expect(r,
                    geist_session_decode_batch(2, twice, out) == GEIST_E_INVALID_ARG,
                    "a repeated member is GEIST_E_INVALID_ARG");
    if (alien != nullptr)
        fails += expect(r,
                        geist_session_decode_batch(2, mixed, out) == GEIST_E_INVALID_ARG,
                        "a member of another model is GEIST_E_INVALID_ARG");
    struct geist_session *many[9];
    for (size_t i = 0; i < 9; i++)
        many[i] = i % 2 ? b : a;
    fails += expect(r,
                    geist_session_decode_batch(max + 1, many, out) == GEIST_E_UNSUPPORTED,
                    "n above decode_batch_max is GEIST_E_UNSUPPORTED");
    fails += expect(r,
                    geist_session_decode_batch(0, nothing, out) == GEIST_E_INVALID_ARG,
                    "n = 0 is GEIST_E_INVALID_ARG");
    fails += expect(r,
                    geist_session_length(a) == len_a && geist_session_length(b) == len_b &&
                            peek(r, a, after) &&
                            memcmp(before, after, r->vocab * sizeof(float)) == 0,
                    "refusals leave the members untouched");
    /* A full context: a window that holds the prompt and no more. */
    {
        geist_token_t         ids[16];
        const size_t          n    = prompt(2, ids);
        struct geist_session *full = open_session(r, r->m, n);
        if (full != nullptr && geist_session_prefill_tokens(full, n, ids) == GEIST_OK) {
            struct geist_session *pair[2] = {a, full};
            fails += expect(r,
                            geist_session_decode_batch(2, pair, out) == GEIST_E_TOO_MANY_TOKENS,
                            "a full context is GEIST_E_TOO_MANY_TOKENS");
            fails += expect(r, geist_session_length(a) == len_a, "the full context left a alone");
        }
        geist_session_destroy(full);
    }
done:
    geist_session_destroy(a);
    geist_session_destroy(b);
    geist_session_destroy(fresh);
    geist_session_destroy(alien);
    return fails;
}

/* One member owes a forward from decode_step, one is fresh from prefill. */
static int check_owed(const struct run *r) {
    int                   fails = 0;
    struct geist_session *a = open_prompted(r, 3), *b = open_prompted(r, 4);
    geist_token_t         t = 0, out[2];
    if (a == nullptr || b == nullptr || geist_session_decode_step(a, &t) != GEIST_OK) {
        fails += expect(r, false, "owed setup");
    } else {
        const size_t          la = geist_session_length(a), lb = geist_session_length(b);
        struct geist_session *pair[2] = {a, b};
        fails += expect(r, geist_session_decode_batch(2, pair, out) == GEIST_OK, "owed + fresh");
        fails += expect(r,
                        geist_session_length(a) == la + 1 && geist_session_length(b) == lb + 1,
                        "every member advances by one");
        fails += expect(r,
                        geist_session_decode_step(a, &t) == GEIST_OK,
                        "decode_step continues after a batch");
    }
    geist_session_destroy(a);
    geist_session_destroy(b);
    return fails;
}

static int check_batching(const struct run *r, size_t max) {
    int                 fails = 0;
    static struct trace ref[NP], t1[1], x[8], y[8];
    for (size_t p = 0; p < NP; p++)
        if (!reference(r, p, &ref[p]))
            return expect(r, false, "reference decode");
    /* n = 1 is decode_step. */
    const size_t one[1] = {0};
    fails += expect(r,
                    batched(r, 1, one, t1) && same(r, &t1[0], &ref[0]),
                    "n = 1 is bit-identical to decode_step");
    if (max < 2)
        return fails;
    /* Prompt 0 among 1..n-1, and in the middle among other prompts. */
    const size_t n = max < NP ? max : NP;
    size_t       px[8], py[8];
    for (size_t b = 0; b < n; b++) {
        px[b] = b;
        py[b] = (NP - 1 - b) % NP; /* reversed: other companions where n < NP */
    }
    const size_t slot = n / 2;
    for (size_t b = 0; b < n; b++)
        if (py[b] == 0)
            py[b] = py[slot];
    py[slot] = 0;
    if (!batched(r, n, px, x) || !batched(r, n, py, y))
        return fails + expect(r, false, "batched decode");
    char msg[160];
    snprintf(msg, sizeof msg, "prompt 0 is bit-identical in slot 0 and slot %zu of %zu", slot, n);
    fails += expect(r, same(r, &x[0], &y[slot]), msg);
    /* Each member against its own decode_step: close, not bit-identical. */
    if (!r->sampling) {
        float worst = 0;
        for (size_t b = 0; b < n; b++) {
            const float d = max_diff(r, &x[b], &ref[px[b]]);
            worst         = d > worst ? d : worst;
            fails += expect(r, x[b].tok[0] == ref[px[b]].tok[0], "first token as decode_step");
        }
        printf("  %s: batched (n=%zu) vs decode_step max |logit diff| %.3g\n",
               r->backend,
               n,
               (double) worst);
        fails += expect(r, worst < 1e-2f, "batched logits within 1e-2 of decode_step");
    }
    return fails;
}

int main(void) {
    struct tf_vocab v      = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf   hybrid = mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
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
    struct tf_buf   llama  = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                               .d_model  = 128,
                                                               .heads    = 4,
                                                               .kv_heads = 2,
                                                               .ffn      = 256,
                                                               .vocab    = VOCAB,
                                                               .context  = 256,
                                                               .seed     = 14});
    int             fails = 0, ran = 0, batching = 0;
    for (size_t bi = 0; bi < sizeof BACKENDS / sizeof BACKENDS[0]; bi++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[bi], nullptr, nullptr, &be) != GEIST_OK || be == nullptr)
            continue; /* not in this build */
        ran++;
        struct geist_model *mh = nullptr, *ml = nullptr;
        if (geist_model_load_from_memory(hybrid.b, hybrid.n, be, &mh) != GEIST_OK ||
            geist_model_load_from_memory(llama.b, llama.n, be, &ml) != GEIST_OK) {
            fprintf(stderr, "FAIL: %s: model load: %s\n", BACKENDS[bi], geist_last_create_error());
            fails++;
        } else {
            const size_t max = geist_model_decode_batch_max(mh);
            printf("%s: decode_batch_max hybrid %zu, llama %zu\n",
                   BACKENDS[bi],
                   max,
                   geist_model_decode_batch_max(ml));
            const struct run rl = {ml, mh, be, BACKENDS[bi], VOCAB, false};
            fails += expect(
                    &rl, geist_model_decode_batch_max(ml) == 0, "a llama has no batched decode");
            {
                struct geist_session *a = open_session(&rl, ml, CTX),
                                     *b = open_session(&rl, ml, CTX);
                geist_token_t         ids[16], out[2];
                const size_t          n       = prompt(0, ids);
                struct geist_session *pair[2] = {a, b};
                if (a != nullptr && b != nullptr &&
                    geist_session_prefill_tokens(a, n, ids) == GEIST_OK &&
                    geist_session_prefill_tokens(b, n, ids) == GEIST_OK) {
                    fails += expect(&rl,
                                    geist_session_decode_batch(2, pair, out) ==
                                                    GEIST_E_UNSUPPORTED &&
                                            geist_session_length(a) == n,
                                    "n = 2 without batched decode is refused, nothing advances");
                    fails += expect(&rl,
                                    geist_session_decode_batch(1, pair, out) == GEIST_OK,
                                    "n = 1 works without batched decode");
                }
                geist_session_destroy(a);
                geist_session_destroy(b);
            }
            batching += max >= 2;
            for (int sampling = 0; sampling < 2; sampling++) {
                const struct run r = {mh, ml, be, BACKENDS[bi], v.n_tok, sampling != 0};
                fails += check_batching(&r, max);
                if (max >= 2 && sampling == 0) {
                    fails += check_refusals(&r, max);
                    fails += check_owed(&r);
                }
            }
        }
        geist_model_destroy(mh);
        geist_model_destroy(ml);
        geist_backend_destroy(be);
    }
    free(hybrid.b);
    free(llama.b);
    tf_free_vocab(&v);
    if (ran == 0)
        GEIST_SKIP("no backend in this build");
    if (fails) {
        fprintf(stderr, "test_decode_batch_unit: %d failure(s)\n", fails);
        return 1;
    }
    printf("PASS: decode_batch (%d backend(s), %d batching)\n", ran, batching);
    return 0;
}
