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
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifndef GEIST_ENABLE_DECISION
#define GEIST_ENABLE_DECISION 0
#endif
#ifndef GEIST_ENABLE_DECISION_CONTEXT
#define GEIST_ENABLE_DECISION_CONTEXT 0
#endif
#if GEIST_ENABLE_DECISION_CONTEXT && !GEIST_ENABLE_DECISION
#error "DECISION_CONTEXT=1 requires DECISION=1"
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
    const struct geist_model            *model;
    const struct geist_backend          *backend;
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
    d->model                = m;
    d->backend              = be;
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

[[nodiscard]] static enum geist_status check_tokens(struct geist_decision *d,
                                                    size_t                 n,
                                                    const geist_token_t    ids[static n],
                                                    const char            *why) {
    for (size_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (size_t) ids[i] >= d->vocab) {
            return fail(d, GEIST_E_INVALID_ARG, why);
        }
    }
    return GEIST_OK;
}

/* Candidate IDs in the vocabulary and distinct (open addressing in seen). */
[[nodiscard]] static enum geist_status
check_candidates(struct geist_decision *d,
                 size_t                 n_candidates,
                 const geist_token_t    ids[static n_candidates]) {
    memset(d->seen, 0xff, d->seen_bytes);
    for (size_t i = 0; i < n_candidates; i++) {
        const geist_token_t id = ids[i];
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
    return GEIST_OK;
}

/* Reads the candidates' logits (from the session's DENSE logits, or from
 * d->logits where SELECTED_ROWS already wrote them), normalizes over them
 * and fills *out. */
[[nodiscard]] static enum geist_status
finish(struct geist_decision        *d,
       size_t                        n_candidates,
       const geist_token_t           candidate_ids[static n_candidates],
       size_t                        projected_rows,
       size_t                        readback_bytes,
       uint64_t                      head_ns,
       struct geist_decision_result *out) {
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
    enum geist_status status =
            check_tokens(d, n_prompt, prompt_ids, "prompt token outside logits vocabulary");
    if (status == GEIST_OK) {
        status = check_candidates(d, n_candidates, candidate_ids);
    }
    if (status != GEIST_OK) {
        return status;
    }
    size_t   projected_rows = d->vocab;
    size_t   readback_bytes = d->dense_readback_bytes;
    uint64_t head_ns        = 0;
    status                  = decision_reset(d);
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
    return finish(d, n_candidates, candidate_ids, projected_rows, readback_bytes, head_ns, out);
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

static void clear_info(struct geist_decision_prefix_info *out) {
    if (out != nullptr) {
        *out = (struct geist_decision_prefix_info) {0};
    }
}

#if GEIST_ENABLE_DECISION && GEIST_ENABLE_DECISION_CONTEXT

/* A prepared prefix: the complete session image after the prefix, and what
 * it was taken under. Scoring only reads it. */
struct geist_decision_prefix {
    const struct geist_model   *model;
    const struct geist_backend *backend;
    size_t                      chunk, max_prompt, cached, bytes;
    uint8_t                    *image;
};

bool geist_decision_prefix_available(void) {
    return true;
}

/* GEIST_OK with the resolved chunk when d can keep and restore a prefix:
 * DENSE, snapshot and geometry ops, a plain FP32 KV cache, and a session
 * the snapshot accepts (no KIVI, no MTP drafter). Otherwise *chunk is 0 and
 * *why says why; d is not written, so the const queries stay read-only. */
[[nodiscard]] static enum geist_status
prefix_geometry(const struct geist_decision *d, size_t *chunk, const char **why) {
    *chunk = 0;
    if (d->mode != GEIST_DECISION_DENSE) {
        *why = "prefix scoring supports DENSE mode only";
        return GEIST_E_UNSUPPORTED;
    }
    if (d->ops->prefill_geometry == nullptr || d->ops->snapshot_size == nullptr ||
        d->ops->snapshot == nullptr || d->ops->restore == nullptr) {
        *why = "architecture cannot snapshot its prefill state";
        return GEIST_E_UNSUPPORTED;
    }
    size_t                  resolved = 0;
    bool                    kv_fp32  = false;
    const enum geist_status gs       = d->ops->prefill_geometry(
            &resolved, &kv_fp32, geist_session_internal_arch_session(d->session));
    if (gs != GEIST_OK || resolved == 0) {
        *why = "prefill geometry unavailable";
        return gs != GEIST_OK ? gs : GEIST_E_INTERNAL;
    }
    if (!kv_fp32) {
        *why = "prefix scoring needs a plain FP32 KV cache";
        return GEIST_E_UNSUPPORTED;
    }
    /* Runs the snapshot's own refusals (KIVI, MTP drafter, open audio
     * stream) without copying anything. */
    size_t                  probe = 0;
    const enum geist_status ss    = geist_session_snapshot_size(&probe, d->session);
    if (ss != GEIST_OK) {
        *why = "session state cannot be snapshotted (KIVI, MTP drafter or open audio stream)";
        return ss;
    }
    *chunk = resolved;
    return GEIST_OK;
}

/* prefix_geometry for the mutating entry points: records why in d. */
[[nodiscard]] static enum geist_status checked_geometry(struct geist_decision *d, size_t *chunk) {
    const char             *why = nullptr;
    const enum geist_status gs  = prefix_geometry(d, chunk, &why);
    return gs == GEIST_OK ? GEIST_OK : fail(d, gs, why);
}

bool geist_decision_prefix_supported(const struct geist_decision *d) {
    size_t      chunk = 0;
    const char *why   = nullptr;
    return d != nullptr && prefix_geometry(d, &chunk, &why) == GEIST_OK;
}

[[nodiscard]] enum geist_status geist_decision_prefill_chunk(size_t                      *out_chunk,
                                                             const struct geist_decision *d) {
    if (out_chunk == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_chunk = 0;
    if (d == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const char *why = nullptr;
    return prefix_geometry(d, out_chunk, &why);
}

void geist_decision_prefix_destroy(struct geist_decision_prefix *p) {
    if (p == nullptr) {
        return;
    }
    safe_free((void **) &p->image);
    safe_free((void **) &p);
}

[[nodiscard]] enum geist_status geist_decision_prefix_prepare(struct geist_decision *d,
                                                              size_t max_snapshot_bytes,
                                                              size_t n_prefix,
                                                              const geist_token_t *prefix_ids,
                                                              struct geist_decision_prefix **out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    if (d == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    d->error[0] = '\0';
    if (out == nullptr || max_snapshot_bytes == 0 || (n_prefix > 0 && prefix_ids == nullptr)) {
        return fail(d, GEIST_E_INVALID_ARG, "invalid prefix arguments");
    }
    if (n_prefix >= d->max_prompt) {
        return fail(d, GEIST_E_TOO_MANY_TOKENS, "prefix leaves no room for a suffix");
    }
    if (n_prefix > 0) {
        const enum geist_status ts =
                check_tokens(d, n_prefix, prefix_ids, "prefix token outside logits vocabulary");
        if (ts != GEIST_OK) {
            return ts;
        }
    }
    size_t            chunk  = 0;
    enum geist_status status = checked_geometry(d, &chunk);
    if (status != GEIST_OK) {
        return status;
    }
    /* The empty prefix is the reset state itself: no zero-token prefill. */
    status = decision_reset(d);
    if (status == GEIST_OK && n_prefix > 0) {
        status = geist_session_prefill_tokens(d->session, n_prefix, prefix_ids);
        if (status != GEIST_OK) {
            return fail(d, status, geist_session_errmsg(d->session));
        }
    }
    if (status != GEIST_OK) {
        return status;
    }
    size_t bytes = 0;
    status       = geist_session_snapshot_size(&bytes, d->session);
    if (status != GEIST_OK) {
        return fail(d, status, geist_session_errmsg(d->session));
    }
    if (bytes == 0 || bytes > max_snapshot_bytes) {
        return fail(d, GEIST_E_OOM, "prefix snapshot exceeds max_snapshot_bytes");
    }
    struct geist_decision_prefix *p = heap_calloc_array_aligned(struct geist_decision_prefix, 1);
    if (p == nullptr || (p->image = heap_alloc_aligned(bytes, alignof(max_align_t))) == nullptr) {
        geist_decision_prefix_destroy(p);
        return fail(d, GEIST_E_OOM, "prefix snapshot allocation failed");
    }
    size_t written = 0;
    status         = geist_session_snapshot(&written, bytes, p->image, d->session);
    if (status != GEIST_OK || written != bytes) {
        geist_decision_prefix_destroy(p);
        return fail(d,
                    status != GEIST_OK ? status : GEIST_E_INTERNAL,
                    status != GEIST_OK ? geist_session_errmsg(d->session)
                                       : "snapshot size changed between size and copy");
    }
    p->model      = d->model;
    p->backend    = d->backend;
    p->chunk      = chunk;
    p->max_prompt = d->max_prompt;
    p->cached     = n_prefix;
    p->bytes      = bytes;
    *out          = p;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
geist_decision_score_from_prefix(struct geist_decision              *d,
                                 const struct geist_decision_prefix *p,
                                 size_t                              n_suffix,
                                 size_t                              n_candidates,
                                 const geist_token_t                *suffix_ids,
                                 const geist_token_t                *candidate_ids,
                                 struct geist_decision_result       *out) {
    clear_result(out);
    if (d == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    d->error[0] = '\0';
    if (out == nullptr || p == nullptr || suffix_ids == nullptr || candidate_ids == nullptr ||
        n_suffix == 0 || n_candidates == 0 || n_candidates > d->max_candidates) {
        return fail(d, GEIST_E_INVALID_ARG, "invalid input or candidate capacity exceeded");
    }
    size_t            chunk  = 0;
    enum geist_status status = checked_geometry(d, &chunk);
    if (status != GEIST_OK) {
        return status;
    }
    if (p->model != d->model || p->backend != d->backend || p->chunk != chunk ||
        p->max_prompt != d->max_prompt) {
        return fail(d, GEIST_E_INVALID_ARG, "prefix was prepared under another model or geometry");
    }
    size_t total = 0;
    if (ckd_add(&total, p->cached, n_suffix) || total > d->max_prompt) {
        return fail(d, GEIST_E_TOO_MANY_TOKENS, "prefix plus suffix exceed the prompt capacity");
    }
    status = check_tokens(d, n_suffix, suffix_ids, "suffix token outside logits vocabulary");
    if (status == GEIST_OK) {
        status = check_candidates(d, n_candidates, candidate_ids);
    }
    if (status != GEIST_OK) {
        return status;
    }
    /* The whole image replaces the session's state, pending logits
     * included, so nothing of an earlier question survives. */
    status = geist_session_restore(p->bytes, p->image, d->session);
    if (status == GEIST_OK) {
        status = geist_session_prefill_tokens(d->session, n_suffix, suffix_ids);
    }
    if (status != GEIST_OK) {
        return fail(d, status, geist_session_errmsg(d->session));
    }
    return finish(d, n_candidates, candidate_ids, d->vocab, d->dense_readback_bytes, 0, out);
}

[[nodiscard]] enum geist_status
geist_decision_prefix_get_info(const struct geist_decision_prefix *p,
                               struct geist_decision_prefix_info  *out) {
    clear_info(out);
    if (p == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = (struct geist_decision_prefix_info) {
            .cached_tokens     = p->cached,
            .snapshot_bytes    = p->bytes,
            .prefill_chunk     = p->chunk,
            .max_prompt_tokens = p->max_prompt,
            .kv_mode           = GEIST_KV_FP32,
    };
    return GEIST_OK;
}

#else

bool geist_decision_prefix_available(void) {
    return false;
}
bool geist_decision_prefix_supported(const struct geist_decision *d) {
    (void) d;
    return false;
}
[[nodiscard]] enum geist_status geist_decision_prefill_chunk(size_t                      *out_chunk,
                                                             const struct geist_decision *d) {
    (void) d;
    if (out_chunk != nullptr) {
        *out_chunk = 0;
    }
    return GEIST_E_UNSUPPORTED;
}
[[nodiscard]] enum geist_status geist_decision_prefix_prepare(struct geist_decision *d,
                                                              size_t max_snapshot_bytes,
                                                              size_t n_prefix,
                                                              const geist_token_t *prefix_ids,
                                                              struct geist_decision_prefix **out) {
    (void) d;
    (void) max_snapshot_bytes;
    (void) n_prefix;
    (void) prefix_ids;
    if (out != nullptr) {
        *out = nullptr;
    }
    return GEIST_E_UNSUPPORTED;
}
[[nodiscard]] enum geist_status
geist_decision_score_from_prefix(struct geist_decision              *d,
                                 const struct geist_decision_prefix *p,
                                 size_t                              n_suffix,
                                 size_t                              n_candidates,
                                 const geist_token_t                *suffix_ids,
                                 const geist_token_t                *candidate_ids,
                                 struct geist_decision_result       *out) {
    (void) d;
    (void) p;
    (void) n_suffix;
    (void) n_candidates;
    (void) suffix_ids;
    (void) candidate_ids;
    clear_result(out);
    return GEIST_E_UNSUPPORTED;
}
[[nodiscard]] enum geist_status
geist_decision_prefix_get_info(const struct geist_decision_prefix *p,
                               struct geist_decision_prefix_info  *out) {
    (void) p;
    clear_info(out);
    return GEIST_E_UNSUPPORTED;
}
void geist_decision_prefix_destroy(struct geist_decision_prefix *p) {
    (void) p;
}
#endif
