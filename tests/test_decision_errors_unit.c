/* Engine seam: adversarial architectures and logits without large model files. */
#define GEIST_INTERNAL_ENGINE_LAYER
#include "src/engine/model.h"
#include "heap.h"
#include "test_helpers.h"
#include <geist_arch.h>
#include <geist_decision.h>
#include <float.h>

static size_t dense_calls, peek_calls, selected_calls;

struct mock {
    float             logits[4];
    float             softcap;
    enum geist_status status;
    size_t            calls;
    bool              no_logits;
};
static void *session_alloc(void *state, const struct geist_session_opts *opts) {
    (void) opts;
    struct mock *s = heap_alloc_array_aligned(struct mock, 1);
    if (s != nullptr) {
        *s = *(struct mock *) state;
    }
    return s;
}
static void session_free(void *state, void *s) {
    (void) state;
    safe_free(&s);
}
static void reset(void *s) {
    ((struct mock *) s)->calls = 0;
}
static enum geist_status prefill(void *s, size_t n, const geist_token_t ids[static n]) {
    (void) ids;
    struct mock *m = s;
    dense_calls++;
    m->calls += n;
    return m->status;
}
[[gnu::noinline]] static float mock_softcap(float value, float c) {
    return tanhf(value / c) * c;
}
static const float *peek(size_t *n, void *s) {
    struct mock *m = s;
    peek_calls++;
    if (m->no_logits) {
        *n = 0;
        return nullptr;
    }
    *n = 4;
    if (m->softcap > 0) {
        for (size_t i = 0; i < 4; i++) {
            m->logits[i] = mock_softcap(m->logits[i], m->softcap);
        }
        m->softcap = 0;
    }
    return m->logits;
}
static size_t vocab(const void *s) {
    (void) s;
    return 4;
}
static size_t no_vocab(const void *s) {
    (void) s;
    return 0;
}
static bool rows_supported(const void *state) {
    (void) state;
    return true;
}
static enum geist_status rows_create(void *session, size_t cap, void **out) {
    (void) cap;
    *out = session;
    return GEIST_OK;
}
static void rows_destroy(void *r) {
    (void) r;
}
static enum geist_status rows_prefill(size_t              *projected,
                                      size_t              *readback,
                                      uint64_t            *ns,
                                      void                *r,
                                      size_t               np,
                                      size_t               nc,
                                      const geist_token_t *prompt,
                                      const geist_token_t *ids,
                                      float               *out) {
    (void) prompt;
    struct mock *m = r;
    *projected     = 0;
    *readback      = 0;
    *ns            = 0;
    memset(out, 0, nc * sizeof(float));
    selected_calls++;
    m->calls += np;
    if (m->status != GEIST_OK)
        return m->status;
    if (m->no_logits)
        return GEIST_E_BACKEND;
    const float c = m->softcap;
    for (size_t i = 0; i < nc; i++) {
        float v = m->logits[ids[i]];
        if (c > 0)
            v = mock_softcap(v, c);
        out[i] = v;
    }
    *projected = nc;
    *readback  = nc * sizeof(float);
    *ns        = 1;
    return GEIST_OK;
}
static int score_case_mode(struct geist_model      *model,
                           enum geist_status        want,
                           bool                     softcap,
                           enum geist_decision_mode mode) {
    struct geist_decision_opts opts = {.mode = mode, .max_prompt_tokens = 4, .max_candidates = 4};
    struct geist_decision     *d    = nullptr;
    if (geist_decision_create(model, model->backend, &opts, &d) != GEIST_OK) {
        return geist_expect(false, "mock decision create");
    }
    const geist_token_t          p[] = {1, 2}, c[] = {0, 1, 2};
    struct geist_decision_result out;
    const uint64_t               allocations = heap_alloc_count();
    dense_calls = peek_calls = selected_calls = 0;
    const enum geist_status status            = geist_decision_score(d, 2, 3, p, c, &out);
    int                     fails = geist_expect(status == want, "mock status propagated");
    fails += geist_expect(heap_alloc_count() == allocations, "decision hot path allocates nothing");
    if (mode == GEIST_DECISION_SELECTED_ROWS) {
        fails += geist_expect(selected_calls == 1 && dense_calls == 0 && peek_calls == 0,
                              "selected call bypasses dense projection/peek completely");
    }
    if (want != GEIST_OK) {
        fails += geist_expect(out.logits == nullptr && out.probabilities == nullptr &&
                                      out.n_candidates == 0 && out.best_index == SIZE_MAX,
                              "nonfinite/backend error returns no partial output");
    } else {
        double sum = 0;
        for (size_t i = 0; i < 3; i++) {
            fails += geist_expect(isfinite(out.probabilities[i]),
                                  "extreme finite logits normalize to finite values");
            sum += out.probabilities[i];
        }
        fails += geist_expect(fabs(sum - 1) < 1e-12 && out.best_index == 0,
                              "normalization and first-maximum tie rule");
        if (softcap) {
            struct mock  reference = *(const struct mock *) model->text_decoder.arch_meta;
            size_t       n         = 0;
            const float *expected  = peek(&n, &reference);
            for (size_t i = 0; i < 3; i++) {
                fails += geist_expect(out.logits[i] == expected[i],
                                      "scores use peek_logits softcap values");
            }
        }
    }
    geist_decision_destroy(d);
    return fails;
}

static int score_case(struct geist_model *m, enum geist_status want, bool softcap) {
    return score_case_mode(m, want, softcap, GEIST_DECISION_DENSE) +
           score_case_mode(m, want, softcap, GEIST_DECISION_SELECTED_ROWS);
}

int main(void) {
    if (!geist_decision_available()) {
        printf("decision errors: disabled PASS\n");
        return 0;
    }
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK) {
        GEIST_SKIP("cpu_scalar not built");
    }
    struct mock                         state    = {.logits = {100, -2, -3, 0}, .softcap = 30};
    const struct geist_arch_ops_decoder complete = {.session_alloc           = session_alloc,
                                                    .session_free            = session_free,
                                                    .state_reset             = reset,
                                                    .prefill                 = prefill,
                                                    .peek_logits             = peek,
                                                    .logits_vocab_size       = vocab,
                                                    .decision_rows_supported = rows_supported,
                                                    .decision_rows_create    = rows_create,
                                                    .decision_rows_destroy   = rows_destroy,
                                                    .prefill_rows            = rows_prefill};
    struct geist_arch_ops_decoder       ops      = complete;
    struct geist_model model = {.text_decoder = {.arch_ops = &ops, .arch_meta = &state},
                                .backend      = be};
    int                fails = score_case(&model, GEIST_OK, true);
    state                    = (struct mock) {.logits = {FLT_MAX, -FLT_MAX, FLT_MAX, NAN}};
    fails += score_case(&model, GEIST_OK, false); /* unselected NaN is irrelevant */
    const float invalid[] = {NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < 3; i++) {
        state.logits[1] = invalid[i];
        fails += score_case(&model, GEIST_E_BACKEND, false);
    }
    state = (struct mock) {.no_logits = true};
    fails += score_case(&model, GEIST_E_BACKEND, false);
    state = (struct mock) {.status = GEIST_E_TOO_MANY_TOKENS};
    fails += score_case(&model, GEIST_E_TOO_MANY_TOKENS, false);
    struct geist_decision_opts o = {.max_prompt_tokens = 4, .max_candidates = 4};
    struct geist_decision     *d = nullptr;
    ops.session_alloc            = nullptr;
    fails +=
            geist_expect(!geist_decision_supported(&model) &&
                                 geist_decision_create(&model, be, &o, &d) == GEIST_E_UNSUPPORTED &&
                                 d == nullptr,
                         "legacy shared-state architecture rejected");
    ops             = complete;
    ops.state_reset = nullptr;
    fails += geist_expect(!geist_decision_supported(&model), "no-reset architecture unsupported");
    ops                   = complete;
    ops.logits_vocab_size = no_vocab;
    fails += geist_expect(geist_decision_create(&model, be, &o, &d) == GEIST_E_UNSUPPORTED,
                          "embedding-only logits vocabulary rejected at creation");
    ops              = complete;
    o.mode           = GEIST_DECISION_SELECTED_ROWS;
    ops.prefill_rows = nullptr;
    fails +=
            geist_expect(!geist_decision_mode_supported(&model, o.mode) &&
                                 geist_decision_create(&model, be, &o, &d) == GEIST_E_UNSUPPORTED &&
                                 d == nullptr,
                         "incomplete selected hooks never silently fall back");
    ops              = complete;
    o.mode           = GEIST_DECISION_DENSE;
    o.max_candidates = SIZE_MAX;
    fails += geist_expect(geist_decision_create(&model, be, &o, &d) == GEIST_E_INVALID_ARG &&
                                  d == nullptr,
                          "huge candidate capacity rejected");
    o.max_candidates    = 4;
    o.max_prompt_tokens = SIZE_MAX;
    fails += geist_expect(geist_decision_create(&model, be, &o, &d) == GEIST_E_INVALID_ARG &&
                                  d == nullptr,
                          "prompt byte size overflow rejected");
    o.max_prompt_tokens = 4;
    o.mode              = (enum geist_decision_mode) 99;
    fails += geist_expect(geist_decision_create(&model, be, &o, &d) == GEIST_E_INVALID_ARG,
                          "unknown mode rejected");
    o.mode = GEIST_DECISION_DENSE;
    heap_fail_allocations(true);
    const enum geist_status oom = geist_decision_create(&model, be, &o, &d);
    heap_fail_allocations(false);
    fails += geist_expect(oom == GEIST_E_OOM && d == nullptr,
                          "allocation failure clears create output");
    fails += geist_expect(geist_decision_create(&model, be, &o, nullptr) == GEIST_E_INVALID_ARG,
                          "null create output rejected");
    fails += geist_expect(geist_decision_create(&model, nullptr, &o, &d) == GEIST_E_INVALID_ARG &&
                                  d == nullptr,
                          "null backend rejected");
    struct geist_backend *other = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &other) == GEIST_OK) {
        fails += geist_expect(geist_decision_create(&model, other, &o, &d) == GEIST_E_INVALID_ARG &&
                                      d == nullptr,
                              "different backend instance rejected");
        geist_backend_destroy(other);
    } else {
        fails++;
    }
    geist_backend_destroy(be);
    printf("decision: adversarial logits, capabilities, overflow, OOM %s\n",
           fails ? "FAIL" : "PASS");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
