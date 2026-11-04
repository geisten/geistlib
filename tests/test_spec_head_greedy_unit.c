/*
 * test_spec_head_greedy_unit — greedy decode is the argmax of the full
 * output head by default, except on BitNet (#384).
 *
 * The speculative output head (forward/spec_head.c) scores only the rows an
 * int8 sketch ranks highest, and the sketch can rank the true argmax out:
 * on Llama 3.2 3B it emitted " The" where the full head and llama.cpp pick
 * " Casual". The model family decides its default
 * (transformer_family.spec_head_default): on for both BitNet families, off
 * for the rest; GEIST_SPEC_HEAD=0/1 overrides. On a llama fixture whose
 * tied F16 head the spec head accepts (vocab 8192, width 1280):
 *   - by default the spec head is not built, every greedy token is the
 *     argmax of the step's full logits, and the tokens equal those of
 *     GEIST_SPEC_HEAD=0, over N_STEPS steps on every CPU backend;
 *   - GEIST_SPEC_HEAD=1 still builds it where the host has the kernels
 *     (NEON+dotprod or AVX2), so the fixture does exercise an eligible head;
 *   - with GEIST_SPEC_HEAD=1 and the smallest finalist count the argmax
 *     check does see wrong tokens, so it is able to fail. (With the shipped
 *     1024 finalists the opt-in head already misses 1 of the 32 steps here,
 *     which is what the old default did.)
 */
#define _POSIX_C_SOURCE 200809L /* setenv, unsetenv */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_family.h"
#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__)
#define SPEC_KERNELS 1
#endif

enum { N_STEPS = 32, N_PROMPT = 4 };

struct run {
    geist_token_t tok[N_STEPS];
    int           not_argmax; /* steps whose token is not the full head's argmax */
    int           spec_state; /* the model's spec head: 1 built, else not */
};

/* Greedy-decode N_STEPS tokens; before each step, the full logits' argmax
 * (peek_logits recomputes the dense head after a spec step). */
static bool decode(struct geist_model *m, struct geist_backend *be, struct run *r) {
    const struct geist_session_opts o   = {.temperature = 0.0f, .max_seq_len = 64};
    struct geist_session           *s   = nullptr;
    const geist_token_t             p[] = {1, 17, 300, 4242};
    bool                            ok  = geist_session_create(m, be, &o, &s) == GEIST_OK &&
                                          geist_session_prefill_tokens(s, N_PROMPT, p) == GEIST_OK;
    memset(r, 0, sizeof *r);
    if (ok) {
        const struct transformer_arch_session *a = tf_sess(geist_session_internal_arch_session(s));
        r->spec_state                            = a->model->spec_state;
    }
    for (size_t i = 0; ok && i < N_STEPS; i++) {
        size_t       n  = 0;
        const float *lg = geist_session_peek_logits(&n, s);
        ok              = lg != nullptr && n > 0;
        size_t best     = 0;
        for (size_t v = 1; ok && v < n; v++) {
            if (lg[v] > lg[best]) {
                best = v;
            }
        }
        ok = ok && geist_session_decode_step(s, &r->tok[i]) == GEIST_OK;
        if (ok && (size_t) r->tok[i] != best) {
            r->not_argmax++;
        }
    }
    geist_session_destroy(s);
    return ok;
}

static bool load_and_decode(struct geist_backend *be, const struct tf_buf *g, struct run *r) {
    struct geist_model *m = nullptr;
    const bool          ok =
            geist_model_load_from_memory(g->b, g->n, be, &m) == GEIST_OK && decode(m, be, r);
    geist_model_destroy(m);
    return ok;
}

int main(void) {
    const char *const backends[] = {"cpu_scalar", "cpu_neon", "cpu_x86"};
    int               fails      = 0;
    /* d_model 1280: the sketch keeps 640 dimensions, 4x smaller than an F16
     * row, which the spec head requires; vocab 8192 is its minimum. */
    struct tf_buf g = mf_llama_gguf(&(struct mf_llama) {.layers   = 1,
                                                        .d_model  = 1280,
                                                        .heads    = 10,
                                                        .kv_heads = 2,
                                                        .ffn      = 128,
                                                        .vocab    = 8192,
                                                        .context  = 64,
                                                        .seed     = 384,
                                                        .f16_embd = true});
    unsetenv("GEIST_SPEC_TOPK");
    fails += geist_expect(transformer_family_bitnet_b158.spec_head_default &&
                                  transformer_family_bitnet.spec_head_default,
                          "BitNet keeps the spec head on by default");
    fails += geist_expect(!transformer_family_llama.spec_head_default,
                          "llama defaults to the full head");
    for (size_t b = 0; b < sizeof backends / sizeof backends[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(backends[b], nullptr, nullptr, &be) != GEIST_OK) {
            printf("skip %s: not in this build\n", backends[b]);
            continue;
        }
        struct run dflt, exact, spec, spec16;
        unsetenv("GEIST_SPEC_HEAD");
        bool ok = load_and_decode(be, &g, &dflt);
        setenv("GEIST_SPEC_HEAD", "0", 1);
        ok = ok && load_and_decode(be, &g, &exact);
        setenv("GEIST_SPEC_HEAD", "1", 1);
        ok = ok && load_and_decode(be, &g, &spec);
        setenv("GEIST_SPEC_TOPK", "16", 1);
        ok = ok && load_and_decode(be, &g, &spec16);
        unsetenv("GEIST_SPEC_TOPK");
        unsetenv("GEIST_SPEC_HEAD");
        fails += geist_expect(ok, "fixture loads and decodes");
        if (ok) {
            printf("%s: spec head default=%d opt-in=%d; tokens not the argmax: default %d, "
                   "exact %d, opt-in %d, opt-in topk 16 %d\n",
                   backends[b],
                   dflt.spec_state,
                   spec.spec_state,
                   dflt.not_argmax,
                   exact.not_argmax,
                   spec.not_argmax,
                   spec16.not_argmax);
            fails += geist_expect(dflt.spec_state != 1, "the spec head is off by default");
            fails += geist_expect(dflt.not_argmax == 0, "default greedy is the full argmax");
            fails += geist_expect(exact.not_argmax == 0, "GEIST_SPEC_HEAD=0 is the full argmax");
            fails += geist_expect(memcmp(dflt.tok, exact.tok, sizeof dflt.tok) == 0,
                                  "default tokens equal GEIST_SPEC_HEAD=0");
#ifdef SPEC_KERNELS
            fails += geist_expect(spec.spec_state == 1,
                                  "GEIST_SPEC_HEAD=1 builds the head on this fixture");
            fails += geist_expect(spec16.not_argmax > 0,
                                  "the argmax check sees a 16-finalist sketch miss");
#endif
        }
        geist_backend_destroy(be);
    }
    free(g.b);
    if (fails == 0)
        printf("PASS: greedy is the full head's argmax off BitNet unless GEIST_SPEC_HEAD=1\n");
    return fails == 0 ? 0 : 1;
}
