/* Optional decision scoring stays in ENGINE and dispatches through arch_ops. */
#define GEIST_INTERNAL_ENGINE_LAYER
#include "model.h"
#include "checked.h"
#include "error.h"
#include "heap.h"

#include <geist_arch.h>
#include <geist_decision.h>
#include <geist_util.h>

#include <math.h>
#include <stdalign.h>
#include <stdio.h>
#include <string.h>

#ifndef GEIST_ENABLE_DECISION
#define GEIST_ENABLE_DECISION 0
#endif

static void clear_result(struct geist_decision_result *out) {
    if (out != nullptr) {
        *out = (struct geist_decision_result) {.best_index = SIZE_MAX};
    }
}

#if GEIST_ENABLE_DECISION

/* The public contract rejects non-finite selected logits. Fail compilation
 * rather than silently lose that check in a custom build with fast math. */
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "decision scoring requires -fno-finite-math-only"
#endif

struct geist_decision {
    struct geist_session                *session;
    const struct geist_arch_ops_decoder *ops;
    void                                *readout;
    enum geist_decision_mode             mode;
    size_t         max_prompt, max_candidates, vocab, seen_cap, seen_bytes, dense_readback_bytes;
    geist_token_t *seen;
    float         *logits;
    double        *probabilities;
    char           error[256];
};

[[nodiscard]] static enum geist_status
fail(struct geist_decision *d, enum geist_status status, const char *message) {
    snprintf(d->error, sizeof d->error, "%s", message);
    return status;
}

bool geist_decision_available(void) {
    return true;
}

/* Every mode needs independent sessions, reset and a generative logits
 * vocabulary. */
static bool decision_supported(const struct geist_model *m) {
    if (m == nullptr || m->text_decoder.arch_meta == nullptr) {
        return false;
    }
    const struct geist_arch_ops_decoder *ops = m->text_decoder.arch_ops;
    return ops != nullptr && ops->session_alloc != nullptr && ops->session_free != nullptr &&
           ops->state_reset != nullptr && ops->prefill != nullptr && ops->peek_logits != nullptr &&
           ops->logits_vocab_size != nullptr &&
           ops->logits_vocab_size(m->text_decoder.arch_meta) > 0;
}

bool geist_decision_mode_supported(const struct geist_model *m, enum geist_decision_mode mode) {
    if (!decision_supported(m))
        return false;
    if (mode == GEIST_DECISION_DENSE)
        return true;
    const struct geist_arch_ops_decoder *ops = m->text_decoder.arch_ops;
    return mode == GEIST_DECISION_SELECTED_ROWS && ops->decision_rows_supported != nullptr &&
           ops->decision_rows_create != nullptr && ops->decision_rows_destroy != nullptr &&
           ops->prefill_rows != nullptr && ops->decision_rows_supported(m->text_decoder.arch_meta);
}

[[nodiscard]] enum geist_status geist_decision_create(struct geist_model               *m,
                                                      struct geist_backend             *be,
                                                      const struct geist_decision_opts *opts,
                                                      struct geist_decision           **out) {
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (m == nullptr || be == nullptr || be != m->backend) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG,
                                    "geist_decision_create",
                                    "model/backend is null or backend does not match model");
        return GEIST_E_INVALID_ARG;
    }
    if (!decision_supported(m)) {
        geist_error_set_create_time(GEIST_E_UNSUPPORTED,
                                    "geist_decision_create",
                                    "architecture lacks isolated resettable logits sessions");
        return GEIST_E_UNSUPPORTED;
    }
    const struct geist_decision_opts o = opts != nullptr ? *opts : (struct geist_decision_opts) {0};
    const size_t prompt_cap            = o.max_prompt_tokens != 0 ? o.max_prompt_tokens : 512;
    const size_t candidates            = o.max_candidates != 0 ? o.max_candidates : 16;
    const size_t vocab = m->text_decoder.arch_ops->logits_vocab_size(m->text_decoder.arch_meta);
    size_t       slots, seen_bytes, logits_bytes, probability_bytes, prompt_bytes, vocab_bytes;
    /* Bounds precede allocation and indexing, per AGENT.md. The hash table
     * is <= half full; its empty sentinel is an invalid token (-1). */
    if ((o.mode != GEIST_DECISION_DENSE && o.mode != GEIST_DECISION_SELECTED_ROWS) ||
        o.kv_mode < GEIST_KV_AUTO || o.kv_mode > GEIST_KV_INT4 || candidates > vocab ||
        vocab > INT32_MAX || ckd_mul(&vocab_bytes, vocab, sizeof(float)) ||
        ckd_mul(&prompt_bytes, prompt_cap, sizeof(geist_token_t)) || prompt_bytes > PTRDIFF_MAX ||
        ckd_mul(&slots, candidates, 2) || ckd_mul(&logits_bytes, candidates, sizeof(float)) ||
        ckd_mul(&probability_bytes, candidates, sizeof(double))) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG,
                                    "geist_decision_create",
                                    "invalid mode or decision capacities");
        return GEIST_E_INVALID_ARG;
    }
    if (!geist_decision_mode_supported(m, o.mode)) {
        geist_error_set_create_time(GEIST_E_UNSUPPORTED,
                                    "geist_decision_create",
                                    "selected rows unavailable for this model/backend pair");
        return GEIST_E_UNSUPPORTED;
    }
    size_t seen_cap = 1;
    while (seen_cap < slots) {
        if (ckd_mul(&seen_cap, seen_cap, 2)) {
            geist_error_set_create_time(GEIST_E_INVALID_ARG,
                                        "geist_decision_create",
                                        "candidate table capacity overflow");
            return GEIST_E_INVALID_ARG;
        }
    }
    if (ckd_mul(&seen_bytes, seen_cap, sizeof(geist_token_t))) {
        geist_error_set_create_time(
                GEIST_E_INVALID_ARG, "geist_decision_create", "candidate table byte size overflow");
        return GEIST_E_INVALID_ARG;
    }
    struct geist_decision *d = heap_calloc_array_aligned(struct geist_decision, 1);
    if (d == nullptr) {
        geist_error_set_create_time(
                GEIST_E_OOM, "geist_decision_create", "decision allocation failed");
        return GEIST_E_OOM;
    }
    d->ops                  = m->text_decoder.arch_ops;
    d->mode                 = o.mode;
    d->max_prompt           = prompt_cap;
    d->max_candidates       = candidates;
    d->vocab                = vocab;
    d->dense_readback_bytes = vocab_bytes;
    d->seen_cap             = seen_cap;
    d->seen_bytes           = seen_bytes;
    d->seen                 = heap_alloc_aligned(seen_bytes, alignof(geist_token_t));
    d->logits               = heap_alloc_aligned(logits_bytes, alignof(float));
    d->probabilities        = heap_alloc_aligned(probability_bytes, alignof(double));
    if (d->seen == nullptr || d->logits == nullptr || d->probabilities == nullptr) {
        geist_decision_destroy(d);
        geist_error_set_create_time(
                GEIST_E_OOM, "geist_decision_create", "candidate workspace allocation failed");
        return GEIST_E_OOM;
    }
    const struct geist_session_opts session_opts = {
            .max_seq_len = prompt_cap,
            .top_p       = 1.0f,
            .kv_mode     = o.kv_mode,
            .m_max       = o.m_max,
    };
    const enum geist_status status = geist_session_create(m, be, &session_opts, &d->session);
    if (status != GEIST_OK) {
        geist_decision_destroy(d);
        return status;
    }
    if (d->mode == GEIST_DECISION_SELECTED_ROWS) {
        const enum geist_status rs = d->ops->decision_rows_create(
                geist_session_internal_arch_session(d->session), candidates, &d->readout);
        if (rs != GEIST_OK) {
            geist_decision_destroy(d);
            geist_error_set_create_time(
                    rs, "geist_decision_create", "selected-row workspace failed");
            return rs;
        }
    }
    *out = d;
    return GEIST_OK;
}

void geist_decision_destroy(struct geist_decision *d) {
    if (d == nullptr) {
        return;
    }
    if (d->readout != nullptr)
        d->ops->decision_rows_destroy(d->readout);
    geist_session_destroy(d->session);
    safe_free((void **) &d->seen);
    safe_free((void **) &d->logits);
    safe_free((void **) &d->probabilities);
    safe_free((void **) &d);
}

/* Invalidates borrowed results and empties the KV/SSM state, before every
 * query. */
[[nodiscard]] static enum geist_status decision_reset(struct geist_decision *d) {
    const enum geist_status status = geist_session_reset(d->session);
    return status == GEIST_OK ? GEIST_OK : fail(d, status, geist_session_errmsg(d->session));
}

size_t geist_decision_vocab_size(const struct geist_decision *d) {
    return d != nullptr ? d->vocab : 0;
}

const char *geist_decision_errmsg(const struct geist_decision *d) {
    return d != nullptr ? (d->error[0] != '\0' ? d->error : "(no error)") : "null decision";
}

[[nodiscard]] enum geist_status geist_decision_score(struct geist_decision        *d,
                                                     size_t                        n_prompt,
                                                     size_t                        n_candidates,
                                                     const geist_token_t          *prompt_ids,
                                                     const geist_token_t          *candidate_ids,
                                                     struct geist_decision_result *out) {
    clear_result(out);
    if (d == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    d->error[0] = '\0';
    if (out == nullptr || prompt_ids == nullptr || candidate_ids == nullptr || n_prompt == 0 ||
        n_candidates == 0 || n_candidates > d->max_candidates) {
        return fail(d, GEIST_E_INVALID_ARG, "invalid input or candidate capacity exceeded");
    }
    if (n_prompt > d->max_prompt) {
        return fail(d, GEIST_E_TOO_MANY_TOKENS, "prompt capacity exceeded");
    }
    for (size_t i = 0; i < n_prompt; i++) {
        if (prompt_ids[i] < 0 || (size_t) prompt_ids[i] >= d->vocab) {
            return fail(d, GEIST_E_INVALID_ARG, "prompt token outside logits vocabulary");
        }
    }
    memset(d->seen, 0xff, d->seen_bytes);
    for (size_t i = 0; i < n_candidates; i++) {
        const geist_token_t id = candidate_ids[i];
        if (id < 0 || (size_t) id >= d->vocab) {
            return fail(d, GEIST_E_INVALID_ARG, "candidate token outside logits vocabulary");
        }
        size_t slot = ((uint32_t) id * 0x9E3779B1u) & (d->seen_cap - 1);
        while (d->seen[slot] != -1) {
            if (d->seen[slot] == id) {
                return fail(d, GEIST_E_INVALID_ARG, "duplicate candidate token");
            }
            slot = (slot + 1) & (d->seen_cap - 1);
        }
        d->seen[slot] = id;
    }
    size_t            projected_rows = d->vocab;
    size_t            readback_bytes = d->dense_readback_bytes;
    uint64_t          head_ns        = 0;
    enum geist_status status         = decision_reset(d);
    if (status == GEIST_OK) {
        if (d->mode == GEIST_DECISION_SELECTED_ROWS) {
            status = d->ops->prefill_rows(&projected_rows,
                                          &readback_bytes,
                                          &head_ns,
                                          d->readout,
                                          n_prompt,
                                          n_candidates,
                                          prompt_ids,
                                          candidate_ids,
                                          d->logits);
        } else {
            status = geist_session_prefill_tokens(d->session, n_prompt, prompt_ids);
        }
    }
    if (status != GEIST_OK) {
        return fail(d,
                    status,
                    d->mode == GEIST_DECISION_SELECTED_ROWS ? "selected-row prefill/readout failed"
                                                            : geist_session_errmsg(d->session));
    }
    const float *logits = nullptr;
    if (d->mode == GEIST_DECISION_DENSE) {
        size_t n_logits = 0;
        logits          = geist_session_peek_logits(&n_logits, d->session);
        if (logits == nullptr || n_logits != d->vocab) {
            return fail(
                    d, GEIST_E_BACKEND, "backend did not expose the declared logits vocabulary");
        }
    }
    double maximum = -INFINITY;
    size_t best    = 0;
    for (size_t i = 0; i < n_candidates; i++) {
        const float v = d->mode == GEIST_DECISION_DENSE ? logits[candidate_ids[i]] : d->logits[i];
        if (!isfinite(v)) {
            return fail(d, GEIST_E_BACKEND, "non-finite selected logit");
        }
        d->logits[i] = v;
        if ((double) v > maximum) {
            maximum = v;
            best    = i;
        }
    }
    /* Subtract/accumulate in double: finite extreme float logits cannot
     * overflow the subtraction. There is always one exp(0), so sum >= 1. */
    double sum = 0.0;
    for (size_t i = 0; i < n_candidates; i++) {
        d->probabilities[i] = exp((double) d->logits[i] - maximum);
        sum += d->probabilities[i];
    }
    for (size_t i = 0; i < n_candidates; i++) {
        d->probabilities[i] /= sum;
    }
    *out = (struct geist_decision_result) {
            .n_candidates         = n_candidates,
            .logits               = d->logits,
            .probabilities        = d->probabilities,
            .mode                 = d->mode,
            .projected_rows       = projected_rows,
            .logit_readback_bytes = readback_bytes,
            .head_ns              = head_ns,
            .best_index           = best,
    };
    return GEIST_OK;
}

#else

bool geist_decision_available(void) {
    return false;
}
bool geist_decision_mode_supported(const struct geist_model *m, enum geist_decision_mode mode) {
    (void) m;
    (void) mode;
    return false;
}

[[nodiscard]] enum geist_status geist_decision_create(struct geist_model               *m,
                                                      struct geist_backend             *be,
                                                      const struct geist_decision_opts *opts,
                                                      struct geist_decision           **out) {
    (void) m;
    (void) be;
    (void) opts;
    if (out != nullptr) {
        *out = nullptr;
    }
    geist_error_set_create_time(
            GEIST_E_UNSUPPORTED, "geist_decision_create", "built with DECISION=0");
    return GEIST_E_UNSUPPORTED;
}
void geist_decision_destroy(struct geist_decision *d) {
    (void) d;
}
size_t geist_decision_vocab_size(const struct geist_decision *d) {
    (void) d;
    return 0;
}
const char *geist_decision_errmsg(const struct geist_decision *d) {
    (void) d;
    return "built with DECISION=0";
}
[[nodiscard]] enum geist_status geist_decision_score(struct geist_decision        *d,
                                                     size_t                        n_prompt,
                                                     size_t                        n_candidates,
                                                     const geist_token_t          *prompt_ids,
                                                     const geist_token_t          *candidate_ids,
                                                     struct geist_decision_result *out) {
    (void) d;
    (void) n_prompt;
    (void) n_candidates;
    (void) prompt_ids;
    (void) candidate_ids;
    clear_result(out);
    return GEIST_E_UNSUPPORTED;
}
#endif
