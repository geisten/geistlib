/*
 * src/archs/transformer/forward.h — internal forward-pass helpers,
 * implemented in forward/. Not part of the public ABI.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_FORWARD_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_FORWARD_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/forward.h is internal to the architecture layer."
#endif

#include "arch_state.h"
#include <geist_backend.h>
#include <geist_types.h>

/* Bounds guard: every KV-appending entry point (prefill text/audio,
 * decode, verify) must call this before writing. Positions past
 * sess->max_seq_len have no KV rows and no RoPE table entries — writing
 * there is heap corruption on CPU and garbage/aborted forwards on GPU. */
[[nodiscard]] static inline enum geist_status
transformer_check_kv_room(struct transformer_arch_session *sess, size_t n_new) {
    if (sess->kv_len + n_new <= sess->max_seq_len) {
        return GEIST_OK;
    }
    geist_backend_set_error(sess->model->backend,
                            GEIST_E_TOO_MANY_TOKENS,
                            "transformer: kv_len %zu + %zu tokens exceeds session "
                            "max_seq_len %zu (raise geist_session_opts.max_seq_len)",
                            sess->kv_len,
                            n_new,
                            sess->max_seq_len);
    return GEIST_E_TOO_MANY_TOKENS;
}

/* Bounds guard: token ids arrive from callers (prefill_tokens,
 * pin_prefix) and from draft models with their own vocabulary. An id
 * outside [0, vocab_size) indexes the embed table out of bounds — a
 * wild read, not a garbage token. Check before the pointer moves. */
[[nodiscard]] static inline enum geist_status transformer_check_token_ids(
        struct transformer_arch_session *sess, size_t n, const geist_token_t ids[static n]) {
    const size_t vocab = sess->model->vocab_size;
    for (size_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (size_t) ids[i] >= vocab) {
            geist_backend_set_error(sess->model->backend,
                                    GEIST_E_INVALID_ARG,
                                    "transformer: token id %d at index %zu is outside "
                                    "vocabulary [0, %zu)",
                                    ids[i],
                                    i,
                                    vocab);
            return GEIST_E_INVALID_ARG;
        }
    }
    return GEIST_OK;
}

/* Layer loop: feed `seq` token rows through all st->n_layers
 * layers. Writes into out_h_buf (residual stream). KV slot is at
 * q_position; advance_kv inside transformer_forward_one_layer is
 * the caller's job, this helper iterates and orchestrates. */
[[nodiscard]] enum geist_status transformer_run_all_layers(struct transformer_arch_session *sess,
                                                           size_t               q_position,
                                                           size_t               seq,
                                                           struct geist_buffer *initial_h_buf,
                                                           struct geist_buffer *per_layer_input_buf,
                                                           struct geist_buffer *out_h_buf);

/* Drain the per-session KIVI residual ring across all non-shared layers,
 * if KIVI mode is enabled and residual_count >= R. No-op otherwise. */
[[nodiscard]] enum geist_status transformer_kivi_drain_full(struct transformer_arch_session *sess);

/* Copy the residual rows of a just-pinned prefix (kivi_pin_tail), for
 * transformer_kivi_pin_restore to write back when a reset returns to it.
 * No-op, and no copy kept, outside KIVI or when the prefix fills whole
 * groups. GEIST_E_OOM when the copy cannot be allocated. */
[[nodiscard]] enum geist_status transformer_kivi_pin_save(struct transformer_arch_session *sess);
void                            transformer_kivi_pin_restore(struct transformer_arch_session *sess);

/* Scratch floats the INT8 KV attention uses at most per call, for these
 * head counts and head_dim: the partial results of a decode split into
 * context chunks (forward/attention.c). The session's frame arena is sized
 * for it. */
[[nodiscard]] size_t attention_int8_scratch_floats(size_t n_q_heads, size_t head_dim);

/* Batched PLE precompute: dequant n PLE rows + model_proj(h) + rmsnorm.
 * Output: out_buf [n, PLE_OUT]. Caller ensures n <= st->m_max. */
[[nodiscard]] enum geist_status
compute_per_layer_inputs_batch(struct transformer_arch_session *sess,
                               size_t                           n,
                               const geist_token_t             *ple_ids,
                               struct geist_buffer             *h_buf,
                               struct geist_buffer             *out_buf);

struct transformer_decision_readout;
[[nodiscard]] enum geist_status transformer_head_prepare(struct transformer_arch_session *sess,
                                                         size_t                           row_idx);
[[nodiscard]] enum geist_status transformer_decision_finish(struct transformer_decision_readout *r,
                                                            size_t row_idx);
[[nodiscard]] enum geist_status transformer_prefill_rows(struct transformer_arch_session     *sess,
                                                         size_t                               n,
                                                         const geist_token_t                 *ids,
                                                         struct transformer_decision_readout *r);
[[nodiscard]] bool              transformer_decision_supported(const void *state);
[[nodiscard]] enum geist_status transformer_decision_create(void *session, size_t cap, void **out);
void                            transformer_decision_destroy(void *r);
[[nodiscard]] enum geist_status transformer_decision_prefill(size_t              *rows,
                                                             size_t              *readback_bytes,
                                                             uint64_t            *head_ns,
                                                             void                *r,
                                                             size_t               n_prompt,
                                                             size_t               n_candidates,
                                                             const geist_token_t *prompt,
                                                             const geist_token_t *candidates,
                                                             float               *out);

/* Output head — softcap'd lm_head on a single row of the residual
 * stream. Writes scratch_logits and sets next_token_pending +
 * logits_valid. row_idx selects which row of scratch_h_a/h_b to read
 * from (decode hot path passes 0; prefill last-row variants pass
 * seq-1). n_ctx counts the positions up to and including that row: the
 * repetition penalties (#695) look back over tok_hist[0, n_ctx). */
[[nodiscard]] enum geist_status finalize_logits_one_row(struct transformer_arch_session *sess,
                                                        size_t                           row_idx,
                                                        size_t                           n_ctx,
                                                        geist_token_t                   *out_token);

/* Speculative i8-sketch output head (GEIST_SPEC_HEAD=1). On a large tied F16
 * lm_head it rough-ranks the vocab via an int8 sketch, then computes exact
 * f16 logits for the top-K candidates only — writing scratch_logits and the
 * greedy argmax into *out_token. Returns true if it handled the projection
 * (caller skips the dense lm_head); false to fall back to the exact path
 * (disabled, ineligible weight, non-greedy sampling, or first-build OOM).
 * Reads the normalized hidden from scratch_h_a. */
bool transformer_spec_head_try(struct transformer_arch_session *sess, geist_token_t *out_token);

/* Eager spec-head build (model-level). Called once from state_create;
 * sets st->spec_state to 1 (active) or -1 (ineligible/disabled). */
void transformer_spec_head_init(struct transformer_arch_state *st);

/* Per-session spec scratch lifecycle. alloc is a no-op (returns true)
 * when the model's spec head is inactive. */
bool transformer_spec_session_scratch_alloc(struct transformer_arch_session *sess);
void transformer_spec_session_scratch_free(struct transformer_arch_session *sess);

/* Recompute the dense lm_head from the normalized hidden still in scratch_h_a
 * after the spec fast path left scratch_logits sparse (see logits_sparse).
 * Called lazily by peek_logits; clears logits_sparse on success. */
[[nodiscard]] enum geist_status
transformer_head_dense_recompute(struct transformer_arch_session *sess);

/* Batched variant for verify_forward — runs lm_head on k rows in one
 * batched call, writes k per-position argmaxes into out_tokens. */
[[nodiscard]] enum geist_status
finalize_logits_batch(struct transformer_arch_session *sess, size_t k, geist_token_t *out_tokens);

/* After a batched prefill of `seq` rows, materialize logits for the
 * LAST row only (the one a subsequent decode_step will consume). */
[[nodiscard]] enum geist_status finalize_logits_last_row(struct transformer_arch_session *sess,
                                                         size_t                           seq);

/* Output-head front half: scratch_h_b's first `k` rows, output_norm'd, into
 * scratch_h_a. The lm_head is NOT applied -- see head.c. */
[[nodiscard]] enum geist_status transformer_norm_rows(struct transformer_arch_session *sess,
                                                      size_t                           k);

/* Mean pooling, which unlike last-token spans chunks: accumulate every
 * chunk, then finalize once with the total token count. */
[[nodiscard]] enum geist_status
transformer_embedding_accumulate(struct transformer_arch_session *sess, size_t k, bool first);
[[nodiscard]] enum geist_status finalize_embedding_mean(struct transformer_arch_session *sess,
                                                        size_t                           n_tokens);

/* Embedding models' terminal step: pool + output_norm + L2 normalise into
 * the session's staging row, instead of running the LM head. `seq` is the
 * row count of the final prefill chunk. Sets sess->embedding_valid. */
[[nodiscard]] enum geist_status finalize_embedding_last_row(struct transformer_arch_session *sess,
                                                            size_t                           seq);

/* Dequant ONE row of an arbitrary-dtype tensor into a host fp32 row.
 * Used by the PLE single + batched paths and by the embedding lookup. */
[[nodiscard]] enum geist_status
dequant_one_row(struct geist_backend *be, const struct geist_tensor *t, size_t row_idx, float *dst);

/* Whether the host gathers the rows of a lookup-only table (an untied
 * token_embd, the PLE table) and uploads them, rather than the backend
 * looking them up on the device. A unified-memory device would page the
 * whole table in once it binds it (caps.lookup_tables_on_host, #529); a
 * device-copy backend has no device copy of it, as resolve_weight runs on
 * matrices only, and a copy would cost the table's whole size in device
 * memory for one row per token (caps.weights_device_copy, #468). */
static inline bool transformer_lookup_on_host(const struct geist_backend *be) {
    return be->desc->caps.lookup_tables_on_host || be->desc->caps.weights_device_copy;
}

/* The host gather of a lookup-only table: rows ids[0..n) of `table`, each
 * `row` floats, times `scale`, into dst. Where transformer_lookup_on_host
 * holds, the rows go through the session's lookup_rows and buffer_upload,
 * which a batched backend orders behind the work still reading dst
 * (mapping dst would flush that work once per prefill chunk, #529) and
 * which reaches a dst the host cannot map (a device-local scratch pool,
 * #488). Other backends write through buffer_map. */
[[nodiscard]] enum geist_status transformer_gather_rows(struct transformer_arch_session *sess,
                                                        const struct geist_tensor       *table,
                                                        size_t                           n,
                                                        const geist_token_t  ids[static n],
                                                        size_t               row,
                                                        float                scale,
                                                        struct geist_buffer *dst);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_FORWARD_H */
