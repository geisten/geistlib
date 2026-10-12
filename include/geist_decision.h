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

/* ---- Prefix scoring (DECISION_CONTEXT=1) --------------------------------
 *
 * Several questions over one long context: prefill the shared prefix once,
 * keep the session's complete state (geist_session_snapshot: every KV row,
 * each recurrent layer's conv and recurrent state, positions and pending
 * logits), and per question restore it and prefill only the question's
 * suffix. A question never sees another question's state: every score
 * restores the whole image first. DENSE mode only.
 *
 * A prefix is bound to the decision handle's model and backend, a plain
 * FP32 KV cache (create the handle with kv_mode = GEIST_KV_FP32) and the
 * resolved prefill chunk; any handle on the same model with the same
 * max_prompt_tokens, chunk and cache may score from it, others fail with
 * GEIST_E_INVALID_ARG. It owns its snapshot; it borrows nothing from the
 * handle that prepared it, but must be destroyed before the model.
 * Scoring only reads it, so handles may share one prefix concurrently,
 * each handle on its own thread.
 *
 * Bit-identity: restore + suffix gives the logits a fresh session gives for
 * prefill(prefix) followed by prefill(suffix). A single prefill of the
 * joined prompt splits into the same passes only when the prefix length is
 * a multiple of geist_decision_prefill_chunk.
 *
 * Builds without DECISION_CONTEXT=1 keep these symbols: they return
 * GEIST_E_UNSUPPORTED (or false / nothing) and clear their outputs. */

struct geist_decision_prefix;

/* @stability EXPERIMENTAL — read-only diagnostics of a prepared prefix. */
struct geist_decision_prefix_info {
    size_t             cached_tokens;     /* prefix length; 0 = reset state */
    size_t             snapshot_bytes;    /* owned image size */
    size_t             prefill_chunk;     /* resolved tokens per prefill pass */
    size_t             max_prompt_tokens; /* cached + suffix must not exceed */
    enum geist_kv_mode kv_mode;           /* always GEIST_KV_FP32 */
};

/* @stability EXPERIMENTAL — build capability; false unless DECISION=1 and
 * DECISION_CONTEXT=1. */
[[nodiscard]] bool geist_decision_prefix_available(void);

/* @stability EXPERIMENTAL — handle capability: a DENSE handle whose
 * architecture can snapshot, restore and report its prefill geometry, with a
 * plain FP32 KV cache. false for nullptr and feature-off builds. */
[[nodiscard]] bool geist_decision_prefix_supported(const struct geist_decision *d);

/* @stability EXPERIMENTAL — the handle's resolved prefill chunk in
 * *out_chunk (0 on failure). Choose prefix lengths that are multiples of it
 * to cut where a single prefill would. */
[[nodiscard]] enum geist_status geist_decision_prefill_chunk(size_t                      *out_chunk,
                                                             const struct geist_decision *d);

/* @stability EXPERIMENTAL
 * Prefill n_prefix tokens into d's session and keep its complete state as a
 * new prefix. n_prefix may be 0 (the reset state, no prefill runs) and must
 * leave room for a suffix: n_prefix < max_prompt_tokens, else
 * GEIST_E_TOO_MANY_TOKENS. Tokens outside the logits vocabulary are
 * GEIST_E_INVALID_ARG. The snapshot size is checked against
 * max_snapshot_bytes (nonzero) before anything is allocated; a larger image
 * is GEIST_E_OOM with nothing allocated. Unsupported handles (SELECTED_ROWS,
 * a non-FP32 KV cache, KIVI, an enabled MTP drafter, architectures without
 * snapshots) are GEIST_E_UNSUPPORTED. Invalidates d's borrowed result. On
 * failure *out is nullptr and geist_decision_errmsg(d) says why. */
[[nodiscard]] enum geist_status
geist_decision_prefix_prepare(struct geist_decision         *d,
                              size_t                         max_snapshot_bytes,
                              size_t                         n_prefix,
                              const geist_token_t           *prefix_ids,
                              struct geist_decision_prefix **out);

/* @stability EXPERIMENTAL
 * geist_decision_score with the prefix's state restored in place of a
 * reset: prefills the n_suffix (>= 1) tokens after the cached ones and reads
 * the candidates from the resulting DENSE logits, under every rule and with
 * the result contract of geist_decision_score. cached + n_suffix beyond
 * max_prompt_tokens is GEIST_E_TOO_MANY_TOKENS; a prefix bound to another
 * model, backend or geometry is GEIST_E_INVALID_ARG. A failed restore leaves
 * d's session reset; the next call restores again. */
[[nodiscard]] enum geist_status
geist_decision_score_from_prefix(struct geist_decision              *d,
                                 const struct geist_decision_prefix *p,
                                 size_t                              n_suffix,
                                 size_t                              n_candidates,
                                 const geist_token_t                *suffix_ids,
                                 const geist_token_t                *candidate_ids,
                                 struct geist_decision_result       *out);

/* @stability EXPERIMENTAL — fills *out; nullptr inputs and feature-off
 * builds are GEIST_E_INVALID_ARG / GEIST_E_UNSUPPORTED with *out zeroed. */
[[nodiscard]] enum geist_status
geist_decision_prefix_get_info(const struct geist_decision_prefix *p,
                               struct geist_decision_prefix_info  *out);

/* @stability EXPERIMENTAL — frees the prefix and its snapshot; nullptr is a
 * no-op. */
void geist_decision_prefix_destroy(struct geist_decision_prefix *p);

#ifdef __cplusplus
} /* extern "C" */
#endif
