/*
 * test_deltanet_reset_unit — a reset marks the Gated-DeltaNet state fresh
 * (dn_fresh) instead of clearing its buffers, so nothing they held before
 * may reach a logit after it.
 *
 * On a Qwen3.5-style fixture (model_fixtures.h: two DeltaNet blocks, then
 * a gated attention block), for each CPU backend in the build:
 *   1. A new session is fresh; a reset leaves the buffers as they were
 *      and marks every DeltaNet layer fresh: no pass over the state.
 *   2. After a reset whose buffers are then filled with garbage, the
 *      logits and the state equal those after a reset that clears the
 *      buffers (the old reset), for prompts of 1 token (the token loop),
 *      2 and 3 (shorter than the conv), 40, 64 (one fresh sub-chunk), 100
 *      and 150 (a fresh sub-chunk, then ones that read S, over two forward
 *      calls), and again after two decode steps.
 *   3. The same for a speculative verify right after the reset, whose
 *      snapshot copies the buffers, truncated to 1 and 2 accepted tokens,
 *      against a prefill of those tokens.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t PROMPT_MAX = 150;
constexpr float  GARBAGE    = 0.75f; /* finite: -ffast-math assumes no NaN */

struct fixture {
    struct geist_session            *s;
    struct transformer_arch_session *a;
    size_t                           vocab;
    size_t                           conv_n, s_n; /* floats per DeltaNet layer */
    size_t                           n_dn;
};

/* f(buffer floats, n, ctx) on every DeltaNet layer's conv state, then S. */
static bool each_state(const struct fixture *fx,
                       void (*f)(float *x, size_t n, size_t k, void *ctx),
                       void *ctx) {
    const struct transformer_arch_state *st = fx->a->model;
    const struct geist_backend_vtbl     *v  = st->backend->desc->vtbl;
    size_t                               k  = 0;
    for (size_t li = 0; li < st->n_layers; li++) {
        if (fx->a->dn_S == nullptr || fx->a->dn_S[li] == nullptr)
            continue;
        struct geist_buffer *bufs[2] = {fx->a->dn_conv_state[li], fx->a->dn_S[li]};
        const size_t         ns[2]   = {fx->conv_n, fx->s_n};
        for (size_t b = 0; b < 2; b++) {
            float *x = (float *) v->buffer_map(bufs[b]);
            if (x == nullptr)
                return false;
            f(x, ns[b], k, ctx);
            k += ns[b];
            v->buffer_unmap(bufs[b]);
        }
    }
    return true;
}

static void put_garbage(float *x, size_t n, size_t k, void *ctx) {
    (void) k;
    (void) ctx;
    for (size_t i = 0; i < n; i++)
        x[i] = GARBAGE + (float) (i % 7);
}

static void put_zero(float *x, size_t n, size_t k, void *ctx) {
    (void) k;
    (void) ctx;
    memset(x, 0, n * sizeof *x);
}

static void get_state(float *x, size_t n, size_t k, void *ctx) {
    memcpy((float *) ctx + k, x, n * sizeof *x);
}

static void count_garbage(float *x, size_t n, size_t k, void *ctx) {
    (void) k;
    for (size_t i = 0; i < n; i++)
        *(size_t *) ctx += x[i] == GARBAGE + (float) (i % 7);
}

/* The reset before dn_fresh: the buffers cleared, nothing marked. */
static bool reset_clearing(struct fixture *fx) {
    if (geist_session_reset(fx->s) != GEIST_OK || !each_state(fx, put_zero, nullptr))
        return false;
    for (size_t li = 0; li < fx->a->model->n_layers; li++)
        fx->a->dn_fresh[li] = false;
    return true;
}

/* A reset, then its buffers filled with what a past conversation left. */
static bool reset_dirty(struct fixture *fx) {
    return geist_session_reset(fx->s) == GEIST_OK && each_state(fx, put_garbage, nullptr);
}

/* Equal as values: the fresh chunk may differ from the clearing reset's
 * in the sign of a zero (layer_deltanet.c), and in nothing else. */
static bool same(size_t n, const float *a, const float *b) {
    for (size_t i = 0; i < n; i++)
        if (!(a[i] == b[i]))
            return false;
    return true;
}

/* Logits (vocab floats) then the whole DeltaNet state into out. */
static bool snapshot(struct fixture *fx, float *out) {
    size_t       n = 0;
    const float *p = geist_session_peek_logits(&n, fx->s);
    if (p == nullptr || n != fx->vocab)
        return false;
    memcpy(out, p, n * sizeof *out);
    return each_state(fx, get_state, out + fx->vocab);
}

static int
check_prompt(struct fixture *fx, const geist_token_t *ids, size_t n, float *ref, float *got) {
    const size_t snap  = fx->vocab + fx->n_dn * (fx->conv_n + fx->s_n);
    int          fails = 0;
    for (int dirty = 0; dirty < 2; dirty++) {
        float *out = dirty ? got : ref;
        bool   ok  = (dirty ? reset_dirty(fx) : reset_clearing(fx)) &&
                     geist_session_prefill_tokens(fx->s, n, ids) == GEIST_OK && snapshot(fx, out);
        for (int j = 0; ok && j < 2; j++) {
            geist_token_t tok = -1;
            ok                = geist_session_decode_step(fx->s, &tok) == GEIST_OK;
        }
        ok = ok && snapshot(fx, out + snap);
        if (!ok) {
            fprintf(stderr, "FAIL: %zu-token prompt did not run (dirty %d)\n", n, dirty);
            return 1;
        }
    }
    char what[96];
    snprintf(what, sizeof what, "%zu-token prompt: logits and state as after a clearing reset", n);
    fails += geist_expect(same(snap, ref, got), what);
    snprintf(what, sizeof what, "%zu-token prompt: the same after two decode steps", n);
    fails += geist_expect(same(snap, ref + snap, got + snap), what);
    return fails;
}

static int check_verify(
        struct fixture *fx, const geist_token_t *ids, size_t accepted, float *ref, float *got) {
    const size_t  snap = fx->vocab + fx->n_dn * (fx->conv_n + fx->s_n);
    geist_token_t out[3];
    const bool    ok_ref = reset_clearing(fx) &&
                           geist_session_prefill_tokens(fx->s, accepted, ids) == GEIST_OK &&
                           snapshot(fx, ref);
    const bool ok_got = reset_dirty(fx) &&
                        transformer_verify_forward(fx->a, 3, ids, out) == GEIST_OK &&
                        transformer_kv_truncate(fx->a, accepted) == GEIST_OK && snapshot(fx, got);
    char       what[96];
    snprintf(what, sizeof what, "verify after a reset, %zu accepted: as their prefill", accepted);
    return geist_expect(ok_ref && ok_got && same(snap, ref, got), what);
}

/* Checks 1 to 3 on a loaded fixture. */
static int check_fixture(struct fixture *fx) {
    const struct transformer_arch_state *st = fx->a->model;
    const size_t key_dim = (size_t) st->config.dn_n_k_heads * st->config.dn_head_k;
    const size_t val_dim = (size_t) st->config.dn_n_v_heads * st->config.dn_head_v;
    fx->conv_n           = (st->config.dn_conv_kernel - 1) * (2 * key_dim + val_dim);
    fx->s_n = (size_t) st->config.dn_n_v_heads * st->config.dn_head_k * st->config.dn_head_v;
    for (size_t li = 0; li < st->n_layers; li++)
        fx->n_dn += fx->a->dn_S != nullptr && fx->a->dn_S[li] != nullptr;
    if (fx->n_dn != 2 || fx->a->dn_fresh == nullptr)
        return geist_expect(false, "two DeltaNet layers on a host mixer that marks them fresh");

    /* 1. A new session starts fresh; a reset is a mark: the garbage
     * stays, every layer is fresh again. */
    bool born = true;
    for (size_t li = 0; li < st->n_layers; li++)
        born &= fx->a->dn_fresh[li] == (fx->a->dn_S[li] != nullptr);
    int        fails = geist_expect(born, "a new session's DeltaNet layers are fresh");
    size_t     kept  = 0;
    const bool ok = each_state(fx, put_garbage, nullptr) &&
                    geist_session_reset(fx->s) == GEIST_OK && each_state(fx, count_garbage, &kept);
    bool       marked = true;
    for (size_t li = 0; li < st->n_layers; li++)
        marked &= fx->a->dn_fresh[li] == (fx->a->dn_S[li] != nullptr);
    fails += geist_expect(ok && kept == fx->n_dn * (fx->conv_n + fx->s_n),
                          "a reset leaves the state buffers untouched");
    fails += geist_expect(marked, "a reset marks exactly the DeltaNet layers fresh");

    geist_token_t ids[PROMPT_MAX];
    for (size_t i = 0; i < PROMPT_MAX; i++)
        ids[i] = (geist_token_t) ((i * 7 + 3) % fx->vocab);
    const size_t snap = fx->vocab + fx->n_dn * (fx->conv_n + fx->s_n);
    float       *ref  = malloc(2 * snap * sizeof *ref);
    float       *got  = malloc(2 * snap * sizeof *got);
    if (ref == nullptr || got == nullptr) {
        fails += geist_expect(false, "snapshot buffers");
    } else {
        /* 2. Prompts through every path of the mixer. */
        const size_t lens[] = {1, 2, 3, 40, 64, 100, 150};
        for (size_t i = 0; i < sizeof lens / sizeof lens[0]; i++)
            fails += check_prompt(fx, ids, lens[i], ref, got);
        bool cleared = true;
        for (size_t li = 0; li < st->n_layers; li++)
            cleared &= !fx->a->dn_fresh[li];
        fails += geist_expect(cleared, "the forward clears the marks");
        /* 3. A speculative snapshot straight after the reset. */
        for (size_t acc = 1; acc <= 2; acc++)
            fails += check_verify(fx, ids + 5, acc, ref, got);
    }
    free(ref);
    free(got);
    return fails;
}

static int run_backend(const char *backend) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend, nullptr, nullptr, &be) != GEIST_OK) {
        printf("skip %s: not in this build\n", backend);
        return 0;
    }
    struct tf_vocab                 v  = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf                   g  = mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 3,
                                                                             .interval   = 3,
                                                                             .d_model    = 256,
                                                                             .heads      = 8,
                                                                             .kv_heads   = 2,
                                                                             .head_dim   = 64,
                                                                             .rope_dims  = 16,
                                                                             .ffn        = 512,
                                                                             .dn_k_heads = 2,
                                                                             .dn_v_heads = 4,
                                                                             .dn_head_k  = 16,
                                                                             .dn_head_v  = 16,
                                                                             .dn_conv    = 4,
                                                                             .seed       = 5,
                                                                             .tok        = &v});
    struct geist_model             *m  = nullptr;
    struct fixture                  fx = {.vocab = v.n_tok};
    int                             fails;
    const struct geist_session_opts o = {.top_p = 1.0f, .m_max = 128, .max_seq_len = 512};
    if (geist_model_load_from_memory(g.b, g.n, be, &m) == GEIST_OK &&
        geist_session_create(m, be, &o, &fx.s) == GEIST_OK) {
        fx.a = geist_session_internal_arch_session(fx.s);
        printf("%s: vocab %zu\n", backend, fx.vocab);
        fails = check_fixture(&fx);
    } else {
        fails = geist_expect(false, "the fixture loads");
    }
    geist_session_destroy(fx.s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    tf_free_vocab(&v);
    return fails;
}

int main(void) {
    int fails = 0;
    fails += run_backend("cpu_x86");
    fails += run_backend("cpu_scalar");
    fails += run_backend("cpu_neon");
    if (fails == 0)
        printf("OK: a reset's stale DeltaNet state never reaches a logit\n");
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
