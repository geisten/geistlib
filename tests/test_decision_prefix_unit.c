/*
 * test_decision_prefix_unit — snapshot-backed prefix scoring (#5):
 * geist_decision_prefix_prepare + geist_decision_score_from_prefix.
 *
 * On a llama and a Qwen3.5-style hybrid built in memory (the hybrid carries
 * gated-DeltaNet conv and recurrent state beside its KV cache), for every
 * backend in the build:
 *
 *   - restored prefix + suffix gives candidate logits bit-identical to a
 *     fresh session that prefills the prefix and then the suffix;
 *   - with the prefix on a chunk boundary, also bit-identical to scoring
 *     the joined prompt in one call (the unsplit control); off the
 *     boundary the control is only reported;
 *   - repeated, interleaved and reordered questions, two handles sharing
 *     one prefix, a question after a failed call: each answer equals its
 *     first computation, so no question sees another's state;
 *   - the empty prefix equals plain geist_decision_score;
 *   - capacity, snapshot budget, geometry/model mismatch, non-FP32 cache,
 *     SELECTED_ROWS and bad arguments fail explicitly with cleared outputs;
 *   - warmed scoring allocates nothing (heap_alloc_count is flat);
 *   - ties keep the first maximum on an all-equal-weights model.
 *
 * DECISION_CONTEXT=0 builds check the stubs instead; GEIST_EXPECT_DECISION
 * _CONTEXT (set by make test-decision) makes a stale build a failure.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include "heap.h"

#include <geist_decision.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t CTX   = 48;
constexpr size_t CHUNK = 4;
constexpr size_t NCAND = 4;
constexpr size_t VOCAB = 512;

static const geist_token_t PREFIX[8]   = {1, 5, 9, 13, 17, 21, 25, 29};
static const geist_token_t PREFIX6[6]  = {1, 5, 9, 13, 17, 21}; /* off the chunk boundary */
static const geist_token_t S1[3]       = {40, 41, 42};
static const geist_token_t S2[5]       = {7, 30, 3, 11, 2};
static const geist_token_t CAND[NCAND] = {1, 7, 21, 32};

static struct geist_decision_opts opts(size_t m_max, enum geist_kv_mode kv) {
    return (struct geist_decision_opts) {.mode              = GEIST_DECISION_DENSE,
                                         .max_prompt_tokens = CTX,
                                         .max_candidates    = NCAND,
                                         .kv_mode           = kv,
                                         .m_max             = m_max};
}

static bool cleared(const struct geist_decision_result *r) {
    return r->n_candidates == 0 && r->logits == nullptr && r->probabilities == nullptr &&
           r->best_index == SIZE_MAX;
}

/* Candidate logits of a fresh session: prefill(a), then prefill(b). */
static bool reference(struct geist_model   *m,
                      struct geist_backend *be,
                      size_t                na,
                      const geist_token_t  *a,
                      size_t                nb,
                      const geist_token_t  *b,
                      float                 out[static NCAND]) {
    const struct geist_session_opts so = {
            .max_seq_len = CTX, .kv_mode = GEIST_KV_FP32, .m_max = CHUNK, .top_p = 1.0f};
    struct geist_session *s  = nullptr;
    bool                  ok = geist_session_create(m, be, &so, &s) == GEIST_OK &&
                               (na == 0 || geist_session_prefill_tokens(s, na, a) == GEIST_OK) &&
                               geist_session_prefill_tokens(s, nb, b) == GEIST_OK;
    size_t                n  = 0;
    const float          *l  = ok ? geist_session_peek_logits(&n, s) : nullptr;
    ok                       = l != nullptr;
    for (size_t i = 0; ok && i < NCAND; i++) {
        out[i] = l[CAND[i]];
    }
    geist_session_destroy(s);
    return ok;
}

static bool same(const struct geist_decision_result *r, const float want[static NCAND]) {
    return r->n_candidates == NCAND && memcmp(r->logits, want, sizeof(float) * NCAND) == 0;
}

static bool score_p(struct geist_decision              *d,
                    const struct geist_decision_prefix *p,
                    size_t                              n,
                    const geist_token_t                *s,
                    float                               out[static NCAND]) {
    struct geist_decision_result r = {0};
    if (geist_decision_score_from_prefix(d, p, n, NCAND, s, CAND, &r) != GEIST_OK) {
        return false;
    }
    memcpy(out, r.logits, sizeof(float) * NCAND);
    return true;
}

static int run(struct geist_model *m, struct geist_backend *be, const char *what) {
    int                              fails = 0;
    struct geist_decision           *a = nullptr, *b = nullptr, *c = nullptr, *q = nullptr;
    struct geist_decision_prefix    *p = nullptr, *p6 = nullptr, *pe = nullptr, *pf = nullptr;
    const struct geist_decision_opts o = opts(CHUNK, GEIST_KV_FP32);
    if (geist_decision_create(m, be, &o, &a) != GEIST_OK ||
        geist_decision_create(m, be, &o, &b) != GEIST_OK) {
        fprintf(stderr, "FAIL: %s: decision create: %s\n", what, geist_last_create_error());
        return 1;
    }
    fails +=
            geist_expect(geist_decision_prefix_supported(a), "FP32 DENSE handle supports prefixes");
    size_t chunk = 0;
    fails += geist_expect(geist_decision_prefill_chunk(&chunk, a) == GEIST_OK && chunk == CHUNK,
                          "resolved chunk is the requested m_max");

    /* Prepare, with the budget checked before allocation. */
    if (geist_decision_prefix_prepare(a, SIZE_MAX, 8, PREFIX, &p) != GEIST_OK) {
        fprintf(stderr, "FAIL: %s: prepare: %s\n", what, geist_decision_errmsg(a));
        fails++;
        goto done;
    }
    struct geist_decision_prefix_info info = {0};
    fails += geist_expect(geist_decision_prefix_get_info(p, &info) == GEIST_OK &&
                                  info.cached_tokens == 8 && info.snapshot_bytes > 0 &&
                                  info.prefill_chunk == CHUNK && info.max_prompt_tokens == CTX &&
                                  info.kv_mode == GEIST_KV_FP32,
                          "prefix info reports cached tokens, bytes and geometry");
    struct geist_decision_prefix *tight = (struct geist_decision_prefix *) (uintptr_t) 1;
    fails += geist_expect(geist_decision_prefix_prepare(
                                  a, info.snapshot_bytes - 1, 8, PREFIX, &tight) == GEIST_E_OOM &&
                                  tight == nullptr,
                          "a snapshot over budget fails before allocation");
    fails += geist_expect(
            geist_decision_prefix_prepare(a, info.snapshot_bytes, 8, PREFIX, &tight) == GEIST_OK,
            "a snapshot exactly at budget prepares");
    geist_decision_prefix_destroy(tight);

    /* Bit-identity against a fresh prefill(prefix), prefill(suffix). */
    float                        ref1[NCAND], ref2[NCAND], got[NCAND];
    struct geist_decision_result r = {0};
    if (!reference(m, be, 8, PREFIX, 3, S1, ref1) || !reference(m, be, 8, PREFIX, 5, S2, ref2)) {
        fprintf(stderr, "FAIL: %s: reference session\n", what);
        fails++;
        goto done;
    }
    fails += geist_expect(geist_decision_score_from_prefix(a, p, 3, NCAND, S1, CAND, &r) ==
                                          GEIST_OK &&
                                  same(&r, ref1),
                          "restored prefix + suffix is bit-identical to a fresh prefill");
    fails += geist_expect(score_p(a, p, 5, S2, got) && memcmp(got, ref2, sizeof got) == 0,
                          "a second question is bit-identical too");
    fails += geist_expect(score_p(a, p, 3, S1, got) && memcmp(got, ref1, sizeof got) == 0,
                          "repeating the first question gives its first answer");
    fails += geist_expect(score_p(b, p, 5, S2, got) && memcmp(got, ref2, sizeof got) == 0,
                          "another handle scores from the shared prefix");

    /* Unsplit control: on the chunk boundary, the joined prompt in one call. */
    geist_token_t joined[13];
    memcpy(joined, PREFIX, sizeof PREFIX);
    memcpy(joined + 8, S2, sizeof S2);
    fails += geist_expect(geist_decision_score(b, 13, NCAND, joined, CAND, &r) == GEIST_OK &&
                                  same(&r, ref2),
                          "on the chunk boundary the unsplit prompt matches bit for bit");
    if (geist_decision_prefix_prepare(b, SIZE_MAX, 6, PREFIX6, &p6) == GEIST_OK &&
        score_p(b, p6, 5, S2, got)) {
        float         fresh[NCAND];
        geist_token_t j6[11];
        memcpy(j6, PREFIX6, sizeof PREFIX6);
        memcpy(j6 + 6, S2, sizeof S2);
        const bool split_ok =
                reference(m, be, 6, PREFIX6, 5, S2, fresh) && memcmp(got, fresh, sizeof got) == 0;
        fails += geist_expect(split_ok, "off the boundary: equal to the same split, fresh");
        if (geist_decision_score(b, 11, NCAND, j6, CAND, &r) == GEIST_OK) {
            float max_diff = 0.0f;
            for (size_t i = 0; i < NCAND; i++) {
                const float dv = r.logits[i] - got[i];
                max_diff       = dv > max_diff ? dv : (-dv > max_diff ? -dv : max_diff);
            }
            printf("  %s: off-boundary prefix 6 (chunk %zu) vs unsplit prompt: max |dlogit| %g\n",
                   what,
                   CHUNK,
                   (double) max_diff);
        }
    } else {
        fails += geist_expect(false, "off-boundary prefix prepares and scores");
    }

    /* A failed question leaves nothing behind for the next one. */
    const geist_token_t dup[NCAND] = {1, 7, 7, 32};
    fails += geist_expect(geist_decision_score_from_prefix(a, p, 3, NCAND, S1, dup, &r) ==
                                          GEIST_E_INVALID_ARG &&
                                  cleared(&r),
                          "duplicate candidates fail with a cleared result");
    fails += geist_expect(score_p(a, p, 5, S2, got) && memcmp(got, ref2, sizeof got) == 0,
                          "the next question after a failure is unaffected");

    /* The empty prefix is plain scoring. */
    float plain[NCAND];
    fails += geist_expect(geist_decision_score(b, 3, NCAND, S1, CAND, &r) == GEIST_OK &&
                                  (memcpy(plain, r.logits, sizeof plain), true) &&
                                  geist_decision_prefix_prepare(a, SIZE_MAX, 0, nullptr, &pe) ==
                                          GEIST_OK &&
                                  score_p(a, pe, 3, S1, got) && memcmp(got, plain, sizeof got) == 0,
                          "the empty prefix equals geist_decision_score");

    /* Capacity boundaries. */
    geist_token_t full[CTX];
    for (size_t i = 0; i < CTX; i++) {
        full[i] = (geist_token_t) (i % 50 + 1);
    }
    struct geist_decision_prefix *none = (struct geist_decision_prefix *) (uintptr_t) 1;
    fails += geist_expect(geist_decision_prefix_prepare(a, SIZE_MAX, CTX, full, &none) ==
                                          GEIST_E_TOO_MANY_TOKENS &&
                                  none == nullptr,
                          "a prefix filling the prompt capacity is refused");
    /* One suffix token is the decode path (#271: Metal's per-key decode
     * kernel read past head_dim 16 and returned garbage, at times NaN):
     * bit-identical to a fresh prefill(47), prefill(1) as well. */
    float refcap[NCAND];
    fails += geist_expect(
            geist_decision_prefix_prepare(a, SIZE_MAX, CTX - 1, full, &pf) == GEIST_OK &&
                    score_p(a, pf, 1, S1, got) && reference(m, be, CTX - 1, full, 1, S1, refcap) &&
                    memcmp(got, refcap, sizeof got) == 0,
            "prefix + one suffix token at exact capacity scores");
    fails += geist_expect(geist_decision_score_from_prefix(a, pf, 2, NCAND, S1, CAND, &r) ==
                                          GEIST_E_TOO_MANY_TOKENS &&
                                  cleared(&r),
                          "one token over capacity fails with a cleared result");

    /* Bad arguments. */
    const geist_token_t oov[1] = {(geist_token_t) geist_decision_vocab_size(a)};
    fails += geist_expect(
            geist_decision_score_from_prefix(a, p, 0, NCAND, S1, CAND, &r) == GEIST_E_INVALID_ARG &&
                    geist_decision_score_from_prefix(a, p, 1, NCAND, oov, CAND, &r) ==
                            GEIST_E_INVALID_ARG &&
                    geist_decision_score_from_prefix(a, nullptr, 3, NCAND, S1, CAND, &r) ==
                            GEIST_E_INVALID_ARG &&
                    geist_decision_prefix_prepare(a, SIZE_MAX, 1, oov, &none) ==
                            GEIST_E_INVALID_ARG &&
                    geist_decision_prefix_prepare(a, 0, 8, PREFIX, &none) == GEIST_E_INVALID_ARG &&
                    none == nullptr && cleared(&r),
            "empty suffix, out-of-vocabulary tokens, null prefix, zero budget fail");

    /* Another geometry, another cache, another mode. */
    const struct geist_decision_opts o8 = opts(2 * CHUNK, GEIST_KV_FP32);
    if (geist_decision_create(m, be, &o8, &c) == GEIST_OK) {
        fails += geist_expect(geist_decision_score_from_prefix(c, p, 3, NCAND, S1, CAND, &r) ==
                                              GEIST_E_INVALID_ARG &&
                                      cleared(&r),
                              "a prefix from another chunk geometry is refused");
    }
    const struct geist_decision_opts oq = opts(CHUNK, GEIST_KV_INT8);
    if (geist_decision_create(m, be, &oq, &q) == GEIST_OK) {
        fails +=
                geist_expect(!geist_decision_prefix_supported(q) &&
                                     geist_decision_prefix_prepare(q, SIZE_MAX, 8, PREFIX, &none) ==
                                             GEIST_E_UNSUPPORTED &&
                                     none == nullptr,
                             "an INT8 KV cache is capability-negative and refused");
        geist_decision_destroy(q);
        q = nullptr;
    }
    if (geist_decision_mode_supported(m, GEIST_DECISION_SELECTED_ROWS)) {
        struct geist_decision_opts os = o;
        os.mode                       = GEIST_DECISION_SELECTED_ROWS;
        if (geist_decision_create(m, be, &os, &q) == GEIST_OK) {
            fails += geist_expect(
                    !geist_decision_prefix_supported(q) &&
                            geist_decision_prefix_prepare(q, SIZE_MAX, 8, PREFIX, &none) ==
                                    GEIST_E_UNSUPPORTED,
                    "SELECTED_ROWS is DENSE-only refused");
        }
    }

    /* Warmed scoring allocates nothing. */
    if (score_p(a, p, 3, S1, got)) {
        const uint64_t before = heap_alloc_count();
        bool           ok     = true;
        for (int i = 0; i < 20 && ok; i++) {
            ok = score_p(a, p, i % 2 ? 5 : 3, i % 2 ? S2 : S1, got);
        }
        fails += geist_expect(ok && heap_alloc_count() == before,
                              "repeated warmed scoring makes no heap allocation");
    }

done:
    geist_decision_prefix_destroy(p);
    geist_decision_prefix_destroy(p6);
    geist_decision_prefix_destroy(pe);
    geist_decision_prefix_destroy(pf);
    geist_decision_destroy(q);
    geist_decision_destroy(c);
    geist_decision_destroy(b);
    geist_decision_destroy(a);
    return fails;
}

/* All weights 0.01: every logit is equal, so the first candidate wins. */
static int run_ties(struct geist_model *m, struct geist_backend *be, const char *what) {
    struct geist_decision           *d = nullptr;
    struct geist_decision_prefix    *p = nullptr;
    const struct geist_decision_opts o = opts(CHUNK, GEIST_KV_FP32);
    struct geist_decision_result     r = {0};
    const bool ok    = geist_decision_create(m, be, &o, &d) == GEIST_OK &&
                       geist_decision_prefix_prepare(d, SIZE_MAX, 8, PREFIX, &p) == GEIST_OK &&
                       geist_decision_score_from_prefix(d, p, 3, NCAND, S1, CAND, &r) == GEIST_OK;
    int        fails = geist_expect(ok && r.logits[0] == r.logits[3] && r.best_index == 0, what);
    geist_decision_prefix_destroy(p);
    geist_decision_destroy(d);
    return fails;
}

static int disabled(void) {
    struct geist_decision_prefix     *p     = (struct geist_decision_prefix *) (uintptr_t) 1;
    struct geist_decision_result      r     = {.n_candidates = 9, .best_index = 1};
    struct geist_decision_prefix_info info  = {.cached_tokens = 9};
    size_t                            chunk = 9;
    int fails = geist_expect(!geist_decision_prefix_supported(nullptr), "disabled: not supported");
    fails += geist_expect(geist_decision_prefill_chunk(&chunk, nullptr) != GEIST_OK && chunk == 0,
                          "disabled: chunk query fails and clears");
    fails += geist_expect(geist_decision_prefix_prepare(nullptr, 1, 0, nullptr, &p) ==
                                          GEIST_E_UNSUPPORTED &&
                                  p == nullptr,
                          "disabled: prepare is unsupported and clears");
    fails += geist_expect(geist_decision_score_from_prefix(nullptr, nullptr, 1, 1, S1, CAND, &r) ==
                                          GEIST_E_UNSUPPORTED &&
                                  cleared(&r),
                          "disabled: scoring is unsupported and clears");
    fails += geist_expect(geist_decision_prefix_get_info(nullptr, &info) == GEIST_E_UNSUPPORTED &&
                                  info.cached_tokens == 0,
                          "disabled: info is unsupported and clears");
    geist_decision_prefix_destroy(nullptr);
    return fails;
}

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar", "vulkan", "metal"};

int main(void) {
    const char *expected = getenv("GEIST_EXPECT_DECISION_CONTEXT");
    if (expected != nullptr && geist_decision_prefix_available() != (strcmp(expected, "1") == 0)) {
        fprintf(stderr, "FAIL: linked prefix capability differs from requested build flag\n");
        return GEIST_TEST_FAIL;
    }
    if (!geist_decision_prefix_available()) {
        const int f = disabled();
        printf("test_decision_prefix_unit: DECISION_CONTEXT=0 stubs %s\n", f ? "FAIL" : "PASS");
        return f ? GEIST_TEST_FAIL : 0;
    }
    struct tf_vocab v = tf_make_vocab("\xc4\xa0", false);
    const struct {
        const char   *name;
        struct tf_buf g;
    } models[] = {
            {"llama",
             mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                               .d_model  = 128,
                                               .heads    = 4,
                                               .kv_heads = 2,
                                               .ffn      = 256,
                                               .vocab    = VOCAB,
                                               .context  = 256,
                                               .seed     = 14})},
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
                                                 .tok        = &v})},
            {"llama-ties",
             mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                               .d_model  = 64,
                                               .heads    = 4,
                                               .kv_heads = 2,
                                               .ffn      = 128,
                                               .vocab    = VOCAB,
                                               .context  = 256,
                                               .seed     = 0})},
    };
    int fails = 0, ran = 0;
    for (size_t bi = 0; bi < sizeof BACKENDS / sizeof BACKENDS[0]; bi++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[bi], nullptr, nullptr, &be) != GEIST_OK ||
            be == nullptr) {
            continue; /* not in this build */
        }
        ran++;
        for (size_t i = 0; i < sizeof models / sizeof models[0]; i++) {
            struct geist_model *m = nullptr;
            char                what[64];
            snprintf(what, sizeof what, "%s %s", BACKENDS[bi], models[i].name);
            if (geist_model_load_from_memory(models[i].g.b, models[i].g.n, be, &m) != GEIST_OK) {
                fprintf(stderr, "FAIL: %s: model load: %s\n", what, geist_last_create_error());
                fails++;
                continue;
            }
            fails += i == 2 ? run_ties(m, be, "ties keep the first maximum") : run(m, be, what);
            geist_model_destroy(m);
        }
        geist_backend_destroy(be);
    }
    for (size_t i = 0; i < sizeof models / sizeof models[0]; i++) {
        free(models[i].g.b);
    }
    tf_free_vocab(&v);
    if (ran == 0 && fails == 0) {
        printf("SKIP: no backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    printf("test_decision_prefix_unit: %d backend(s), %s\n", ran, fails ? "FAIL" : "PASS");
    return fails ? GEIST_TEST_FAIL : 0;
}
