/*
 * Optional numeric next-token decisions. All declarations are EXPERIMENTAL.
 * Build with DECISION=1; the same symbols remain linkable when disabled.
 */
#pragma once

#include <geist.h>

#ifdef __cplusplus
extern "C" {
#endif

struct geist_decision;

/* @stability EXPERIMENTAL — explicit per-instance execution mode.
 * DENSE uses the model's ordinary prefill and model-conformant peek_logits.
 * SELECTED_ROWS reuses the backbone but projects only requested row tiles.
 * Unsupported model/backend pairs fail explicitly; no automatic fallback.
 * Neither mode supplies a trained classifier head or calibration. */
enum geist_decision_mode { GEIST_DECISION_DENSE = 0, GEIST_DECISION_SELECTED_ROWS = 1 };

/* @stability EXPERIMENTAL — zero initialization selects the defaults.
 * These limits size the private session/workspace at creation. */
struct geist_decision_opts {
    enum geist_decision_mode mode;
    size_t             max_prompt_tokens; /* 0 = 512; must fit the loaded model's sequence cap */
    size_t             max_candidates;    /* 0 = 16; must not exceed the logits vocabulary */
    enum geist_kv_mode kv_mode;
    size_t             m_max; /* 0 = architecture's prefill chunk default */
};

/* @stability EXPERIMENTAL
 * Borrowed arrays in candidate input order. Probabilities are softmax over
 * ONLY the supplied alternatives, at temperature 1. They are neither
 * full-vocabulary probabilities nor calibrated correctness confidence.
 * Ties select the first maximum. Arrays are valid until the next score call
 * (even a failed call) or destruction of THIS handle. */
struct geist_decision_result {
    size_t                   n_candidates;
    const float             *logits;
    const double            *probabilities;
    size_t                   best_index;
    enum geist_decision_mode mode;
    size_t                   projected_rows; /* includes aligned neighbor rows in backend tiles */
    size_t                   logit_readback_bytes; /* row tiles only in SELECTED_ROWS */
    uint64_t head_ns; /* SELECTED_ROWS final-stage elapsed, including pending device work; 0 for
                         DENSE */
};

/* @stability EXPERIMENTAL — build capability; false in DECISION=0 builds. */
[[nodiscard]] bool geist_decision_available(void);

/* @stability EXPERIMENTAL — mode capability for this loaded pair. Every mode
 * requires independent sessions, reset and a generative logits vocabulary:
 * false for embedding-only/legacy archs, unknown modes or feature-off
 * builds. Backend failures can still occur during creation/scoring. */
[[nodiscard]] bool geist_decision_mode_supported(const struct geist_model *m,
                                                 enum geist_decision_mode  mode);

/* @stability EXPERIMENTAL
 * Owns an independent session and candidate workspace, shares model weights.
 * be must be the model's backend. Destroy decisions before model, then backend.
 * Setup/teardown on one model must be serialized with all its operations.
 * Different handles may score concurrently, one thread per handle, under the
 * decoder architecture's session contract. Never call one handle concurrently.
 * No tokenizer, sampler mode or model weights are modified.
 * On failure *out is nullptr; diagnostics use geist_last_create_error().
 * Disabled builds return GEIST_E_UNSUPPORTED (also for score). */
[[nodiscard]] enum geist_status geist_decision_create(struct geist_model               *m,
                                                      struct geist_backend             *be,
                                                      const struct geist_decision_opts *opts,
                                                      struct geist_decision           **out);

/* @stability EXPERIMENTAL — nullptr is a no-op. */
void geist_decision_destroy(struct geist_decision *d);

/* @stability EXPERIMENTAL — loaded logits vocabulary, or 0 for nullptr/off. */
[[nodiscard]] size_t geist_decision_vocab_size(const struct geist_decision *d);

/* @stability EXPERIMENTAL — borrowed diagnostic, valid until score/destroy.
 * nullptr handle returns a static diagnostic. */
const char *geist_decision_errmsg(const struct geist_decision *d);

/* @stability EXPERIMENTAL
 * Score distinct single-token candidate IDs after an independent prompt.
 * Both counts must be nonzero; arrays must contain their indicated counts.
 * IDs outside [0, vocab_size) and duplicates are GEIST_E_INVALID_ARG.
 * Exceeding max_candidates is GEIST_E_INVALID_ARG; exceeding the prompt cap
 * is GEIST_E_TOO_MANY_TOKENS. Multi-token labels have no representation here:
 * applications must tokenize/validate labels and never silently take token 0.
 * Uses raw model logits including its softcap, ignores generation sampling.
 * A non-finite SELECTED logit is GEIST_E_BACKEND, with no partial result.
 * Unselected logits do not enter the conditional normalization.
 * Every failure clears *out (pointers nullptr, count 0, best_index SIZE_MAX).
 * Input/output storage must not overlap this handle's borrowed result arrays.
 * Defensive entry points accept plain pointers, per AGENT.md. */
[[nodiscard]] enum geist_status geist_decision_score(struct geist_decision        *d,
                                                     size_t                        n_prompt,
                                                     size_t                        n_candidates,
                                                     const geist_token_t          *prompt_ids,
                                                     const geist_token_t          *candidate_ids,
                                                     struct geist_decision_result *out);

#ifdef __cplusplus
} /* extern "C" */
#endif
