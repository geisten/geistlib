/*
 * src/archs/transformer/arch_ops.c — P1.3.c: arch-ops entry points.
 *
 * Layer: ARCHITECTURE.
 *
 * Extracted from arch_state.c. Owns the public-internal entry
 * points that bridge the engine's geist_arch_ops_decoder vtable thunks
 * (in arch.c) to the forward primitives in forward.c:
 *
 *   - transformer_prefill_text_batch  : chunked batched prefill
 *   - transformer_prefill_audio_batch : audio soft-token prefill
 *   - transformer_verify_forward      : speculative-decode batch
 *   - transformer_kv_truncate         : speculative-decode rewind
 *   - transformer_pin_prefix          : KV prefix snapshot
 *   - apply_awq_to_state              : AWQ scale folding (load-time)
 *
 * Each orchestrates the per-token forward pass over a multi-token
 * payload (chunked into st->m_max-token sub-batches), runs PLE +
 * run_all_layers + the appropriate output-head finalizer, and manages
 * KIVI drains across the chunk boundary.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch_state.h"
#include "arch_ops.h"
#include "forward.h"

#include "gemma4_kernels.h"
#include "heap.h"
#include "kivi.h"
#include "ptqtp_awq.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- DeltaNet speculative-state transaction ------------------------- */

static bool deltanet_state_geometry(const struct transformer_arch_session *sess,
                                    size_t                                *n_dn,
                                    size_t                                *conv_n,
                                    size_t                                *s_n) {
    const struct transformer_arch_state *st = sess->model;
    *n_dn                                   = 0;
    for (size_t li = 0; li < st->n_layers; li++) {
        if (st->layers[li].mixer == GEIST_MIXER_DELTANET)
            (*n_dn)++;
    }
    if (*n_dn == 0)
        return false;

    const size_t key_dim   = st->config.dn_n_k_heads * st->config.dn_head_k;
    const size_t value_dim = st->config.dn_n_v_heads * st->config.dn_head_v;
    *conv_n                = (st->config.dn_conv_kernel - 1) * (2 * key_dim + value_dim);
    *s_n                   = st->config.dn_n_v_heads * st->config.dn_head_k * st->config.dn_head_v;
    return true;
}

void transformer_recurrent_txn_commit(struct transformer_arch_session *sess) {
    if (sess != nullptr)
        sess->dn_txn_active = false;
}

static enum geist_status
deltanet_txn_begin(struct transformer_arch_session *sess, size_t k, const geist_token_t *ids) {
    size_t n_dn, conv_n, s_n;
    if (!deltanet_state_geometry(sess, &n_dn, &conv_n, &s_n))
        return GEIST_OK;
    if (conv_n > SIZE_MAX / n_dn || s_n > SIZE_MAX / n_dn ||
        n_dn * conv_n > SIZE_MAX / sizeof(float) || n_dn * s_n > SIZE_MAX / sizeof(float) ||
        sess->m_max > SIZE_MAX / sizeof(geist_token_t)) {
        return GEIST_E_OOM;
    }

    /* A normal append after verify commits the previous transaction. A
     * second verify therefore starts from the current, fully accepted state. */
    transformer_recurrent_txn_commit(sess);
    const size_t conv_count = n_dn * conv_n;
    const size_t S_count    = n_dn * s_n;
    if (sess->dn_txn_conv == nullptr)
        sess->dn_txn_conv = heap_alloc_aligned(conv_count * sizeof(float), 64);
    if (sess->dn_txn_S == nullptr)
        sess->dn_txn_S = heap_alloc_aligned(S_count * sizeof(float), 64);
    if (sess->dn_txn_ids == nullptr)
        sess->dn_txn_ids =
                heap_alloc_aligned(sess->m_max * sizeof(geist_token_t), alignof(geist_token_t));
    if (sess->dn_txn_conv == nullptr || sess->dn_txn_S == nullptr || sess->dn_txn_ids == nullptr) {
        geist_backend_set_error(sess->model->backend,
                                GEIST_E_OOM,
                                "transformer: DeltaNet transaction alloc failed");
        return GEIST_E_OOM;
    }

    size_t                           dn = 0;
    const struct geist_backend_vtbl *v  = sess->model->backend->desc->vtbl;
    for (size_t li = 0; li < sess->model->n_layers; li++) {
        if (sess->model->layers[li].mixer != GEIST_MIXER_DELTANET)
            continue;
        const float *conv = (const float *) v->buffer_map(sess->dn_conv_state[li]);
        const float *S    = (const float *) v->buffer_map(sess->dn_S[li]);
        if (conv == nullptr || S == nullptr) {
            if (conv != nullptr)
                v->buffer_unmap(sess->dn_conv_state[li]);
            if (S != nullptr)
                v->buffer_unmap(sess->dn_S[li]);
            geist_backend_set_error(sess->model->backend,
                                    GEIST_E_BACKEND,
                                    "transformer: DeltaNet transaction map failed");
            return GEIST_E_BACKEND;
        }
        memcpy(sess->dn_txn_conv + dn * conv_n, conv, conv_n * sizeof(float));
        memcpy(sess->dn_txn_S + dn * s_n, S, s_n * sizeof(float));
        v->buffer_unmap(sess->dn_conv_state[li]);
        v->buffer_unmap(sess->dn_S[li]);
        dn++;
    }
    memcpy(sess->dn_txn_ids, ids, k * sizeof(*ids));
    sess->dn_txn_conv_count          = conv_count;
    sess->dn_txn_S_count             = S_count;
    sess->dn_txn_base_kv_len         = sess->kv_len;
    sess->dn_txn_k                   = k;
    sess->dn_txn_kivi_residual_count = sess->kivi_residual_count;
    sess->dn_txn_kivi_drained_count  = sess->kivi_drained_count;
    if (sess->model->n_mtp_layers > 0 && sess->mtp_pending_h != nullptr &&
        sess->mtp_txn_pending_h != nullptr) {
        memcpy(sess->mtp_txn_pending_h, sess->mtp_pending_h, sess->model->d_model * sizeof(float));
        sess->mtp_txn_base_len = sess->mtp_kv_len;
    }
    sess->dn_txn_active = true;
    return GEIST_OK;
}

static enum geist_status deltanet_txn_restore(struct transformer_arch_session *sess) {
    size_t n_dn, conv_n, s_n;
    if (!sess->dn_txn_active || !deltanet_state_geometry(sess, &n_dn, &conv_n, &s_n))
        return GEIST_OK;
    size_t                           dn = 0;
    const struct geist_backend_vtbl *v  = sess->model->backend->desc->vtbl;
    for (size_t li = 0; li < sess->model->n_layers; li++) {
        if (sess->model->layers[li].mixer != GEIST_MIXER_DELTANET)
            continue;
        float *conv = (float *) v->buffer_map(sess->dn_conv_state[li]);
        float *S    = (float *) v->buffer_map(sess->dn_S[li]);
        if (conv == nullptr || S == nullptr) {
            if (conv != nullptr)
                v->buffer_unmap(sess->dn_conv_state[li]);
            if (S != nullptr)
                v->buffer_unmap(sess->dn_S[li]);
            geist_backend_set_error(sess->model->backend,
                                    GEIST_E_BACKEND,
                                    "transformer: DeltaNet transaction restore map failed");
            return GEIST_E_BACKEND;
        }
        memcpy(conv, sess->dn_txn_conv + dn * conv_n, conv_n * sizeof(float));
        memcpy(S, sess->dn_txn_S + dn * s_n, s_n * sizeof(float));
        v->buffer_unmap(sess->dn_conv_state[li]);
        v->buffer_unmap(sess->dn_S[li]);
        dn++;
    }
    sess->kv_len              = sess->dn_txn_base_kv_len;
    sess->kivi_residual_count = sess->dn_txn_kivi_residual_count;
    sess->kivi_drained_count  = sess->dn_txn_kivi_drained_count;
    if (sess->model->n_mtp_layers > 0 && sess->mtp_pending_h != nullptr &&
        sess->mtp_txn_pending_h != nullptr) {
        sess->mtp_kv_len = sess->mtp_txn_base_len;
        memcpy(sess->mtp_pending_h, sess->mtp_txn_pending_h, sess->model->d_model * sizeof(float));
    }
    return GEIST_OK;
}

/* ---- Batched text prefill -------------------------------------------- */

static enum geist_status prefill_text_batch_inner(struct transformer_arch_session *sess,
                                                  size_t                           n,
                                                  const geist_token_t             *ids) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || (n > 0 && ids == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    if (n == 0) {
        return GEIST_OK;
    }
    enum geist_status room = transformer_check_kv_room(sess, n);
    if (room != GEIST_OK) {
        return room;
    }
    const enum geist_status vocab_ok = transformer_check_token_ids(sess, n, ids);
    if (vocab_ok != GEIST_OK) {
        return vocab_ok;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    /* sqrt(d_model) embedding scale is Gemma-3/4-specific; Llama / BitNet
     * don't scale. has_ple gates Gemma family identity. */
    const float embed_scale = st->config.has_embed_scale ? sqrtf((float) st->d_model) : 1.0f;

    /* Chunk by the SESSION cap: every scratch buffer is sized with
     * sess->m_max (arch_state.c), and a session opt below the state's
     * preferred value must shrink the chunks with it — since the metal
     * preferred_m_max moved to 256, chunking by st->m_max overran the
     * scratch of any session created with a smaller m_max (embed failed
     * at row sess->m_max exactly). */
    for (size_t off = 0; off < n; off += sess->m_max) {
        const size_t chunk = (n - off > sess->m_max) ? sess->m_max : (n - off);

        /* 1. Embed all chunk tokens into scratch_h_a [chunk, HIDDEN].
         * Device path first: per-row fused lookup+scale dispatches keep
         * batched GPU backends from flushing the pipeline through a
         * mapped host pointer (and skip the host dequant loop). */
        bool embed_on_device = st->model_fusions.embed_lookup_scaled;
        bool embed_batched   = false;
        if (embed_on_device && fused->embedding_lookup_scaled_rows != nullptr) {
            /* #322: one dispatch for the whole chunk instead of one tiny
             * dispatch per token. Non-OK (row cap, unsupported dtype)
             * falls through to the per-token loop below. */
            struct geist_tensor t_rows = {
                    .buffer = sess->scratch_h_a,
                    .offset = 0,
                    .dtype  = GEIST_DTYPE_F32,
                    .layout = GEIST_LAYOUT_DENSE,
                    .ndim   = 2,
                    .shape  = {(int64_t) chunk, (int64_t) st->d_model, 0, 0, 0, 0, 0, 0},
                    .stride = {(int64_t) st->d_model, 1, 0, 0, 0, 0, 0, 0},
            };
            embed_batched = fused->embedding_lookup_scaled_rows(
                                    be, chunk, &st->embed_table, ids + off, embed_scale, &t_rows) ==
                            GEIST_OK;
        }
        if (embed_on_device && !embed_batched) {
            for (size_t t = 0; t < chunk; t++) {
                struct geist_tensor t_row = {
                        .buffer = sess->scratch_h_a,
                        .offset = t * st->d_model * sizeof(float),
                        .dtype  = GEIST_DTYPE_F32,
                        .layout = GEIST_LAYOUT_DENSE,
                        .ndim   = 1,
                        .shape  = {(int64_t) st->d_model, 0, 0, 0, 0, 0, 0, 0},
                        .stride = {1, 0, 0, 0, 0, 0, 0, 0},
                };
                const enum geist_status es = fused->embedding_lookup_scaled(
                        be, &st->embed_table, ids[off + t], embed_scale, &t_row);
                if (es != GEIST_OK) {
                    return es; /* bound at plan build — failure is a real error */
                }
            }
        }
        if (!embed_on_device) {
            float *h_dst = (float *) v->buffer_map(sess->scratch_h_a);
            for (size_t t = 0; t < chunk; t++) {
                enum geist_status s = dequant_one_row(
                        be, &st->embed_table, (size_t) ids[off + t], h_dst + t * st->d_model);
                if (s != GEIST_OK) {
                    v->buffer_unmap(sess->scratch_h_a);
                    return s;
                }
            }
            if (embed_scale != 1.0f) {
                const size_t n_floats = chunk * st->d_model;
                for (size_t i = 0; i < n_floats; i++) {
                    h_dst[i] *= embed_scale;
                }
            }
            v->buffer_unmap(sess->scratch_h_a);
        }

        /* 2. Batched PLE precompute. P1.5.b: skipped for non-PLE families. */
        enum geist_status    s       = GEIST_OK;
        struct geist_buffer *ple_buf = nullptr;
        if (st->config.has_ple) {
            s = compute_per_layer_inputs_batch(
                    sess, chunk, ids + off, sess->scratch_h_a, sess->scratch_per_layer_input);
            if (s != GEIST_OK) {
                return s;
            }
            ple_buf = sess->scratch_per_layer_input;
        }

        /* 3. Layer loop seq=chunk. */
        const size_t q_pos = sess->kv_len;
        s                  = transformer_run_all_layers(
                sess, q_pos, chunk, sess->scratch_h_a, ple_buf, sess->scratch_h_b);
        if (s != GEIST_OK) {
            return s;
        }
        s = transformer_mtp_sync_target(sess, chunk, ids + off, sess->scratch_h_b);
        if (s != GEIST_OK) {
            return s;
        }

        /* 4. Advance kv_len by chunk. */
        sess->kv_len += chunk;
        if (sess->kv_kivi_enabled) {
            sess->kivi_residual_count += chunk;
            transformer_kivi_drain_full(sess);
        }

        /* 5. On the final chunk, finish the pass. A generative model
         *    computes logits for the last token so ops->decode_step has a
         *    pending prediction; an embedding model pools instead — it has
         *    no LM head to run and no next token to predict. */
        if (off + chunk == n) {
            s = geist_pooling_is_embedding(st->config.pooling)
                        ? finalize_embedding_last_row(sess, chunk)
                        : finalize_logits_last_row(sess, chunk);
            if (s != GEIST_OK) {
                return s;
            }
        }
    }
    return GEIST_OK;
}

enum geist_status transformer_prefill_text_batch(struct transformer_arch_session *sess,
                                                 size_t                           n,
                                                 const geist_token_t             *ids) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || (n > 0 && ids == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    transformer_recurrent_txn_commit(sess);
    /* Prefill is compute-bound; let the backend enter its prefill thread
     * regime (cpu_neon bumps OMP to all cores). Restored after the pass. */
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const int                        region_tok =
            v->parallel_region_begin ? v->parallel_region_begin(be, GEIST_REGION_PREFILL_BATCH) : 0;
    const enum geist_status s = prefill_text_batch_inner(sess, n, ids);
    if (v->parallel_region_end) {
        v->parallel_region_end(be, region_tok);
    }
    return s;
}

/* ---- Speculative-decode verify pass ---------------------------------- *
 *
 * Like prefill_text_batch except it computes a per-position sampled
 * token for ALL k positions (not just the last). Caller uses these to
 * accept/reject the draft and then optionally truncates kv_len to
 * undo verify-pass KV writes past the accept point. */

enum geist_status transformer_verify_forward(struct transformer_arch_session *sess,
                                             size_t                           k,
                                             const geist_token_t             *ids,
                                             geist_token_t                   *out_tokens) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || k == 0 || ids == nullptr || out_tokens == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (k > sess->m_max) {
        /* Spec K should fit in one prefill chunk. Larger requires chunking. */
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status room = transformer_check_kv_room(sess, k);
    if (room != GEIST_OK) {
        return room;
    }
    const enum geist_status vocab_ok = transformer_check_token_ids(sess, k, ids);
    if (vocab_ok != GEIST_OK) {
        return vocab_ok;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const float embed_scale = st->config.has_embed_scale ? sqrtf((float) st->d_model) : 1.0f;

    /* 1. Embed all k tokens into scratch_h_a [k, HIDDEN]. */
    {
        float *h_dst = (float *) v->buffer_map(sess->scratch_h_a);
        for (size_t t = 0; t < k; t++) {
            enum geist_status s =
                    dequant_one_row(be, &st->embed_table, (size_t) ids[t], h_dst + t * st->d_model);
            if (s != GEIST_OK) {
                v->buffer_unmap(sess->scratch_h_a);
                return s;
            }
        }
        if (embed_scale != 1.0f) {
            const size_t n_floats = k * st->d_model;
            for (size_t i = 0; i < n_floats; i++) {
                h_dst[i] *= embed_scale;
            }
        }
        v->buffer_unmap(sess->scratch_h_a);
    }

    /* 2. PLE precompute. P1.5.b: skipped for non-PLE families. */
    enum geist_status    s       = GEIST_OK;
    struct geist_buffer *ple_buf = nullptr;
    if (st->config.has_ple) {
        s = compute_per_layer_inputs_batch(
                sess, k, ids, sess->scratch_h_a, sess->scratch_per_layer_input);
        if (s != GEIST_OK) {
            return s;
        }
        ple_buf = sess->scratch_per_layer_input;
    }

    /* 3. Snapshot recurrent state, then run the layer loop. Attention KV
     * is position-addressed and can be overwritten; DeltaNet is recurrent
     * and must be restored explicitly on a partial accept. */
    s = deltanet_txn_begin(sess, k, ids);
    if (s != GEIST_OK)
        return s;
    const size_t q_pos = sess->kv_len;
    s = transformer_run_all_layers(sess, q_pos, k, sess->scratch_h_a, ple_buf, sess->scratch_h_b);
    if (s != GEIST_OK) {
        deltanet_txn_restore(sess);
        transformer_recurrent_txn_commit(sess);
        return s;
    }
    s = transformer_mtp_sync_target(sess, k, ids, sess->scratch_h_b);
    if (s != GEIST_OK) {
        deltanet_txn_restore(sess);
        transformer_recurrent_txn_commit(sess);
        return s;
    }

    /* 4. Advance kv_len by k — caller may truncate later on reject.
     * KIVI: bump residual_count but do NOT drain (drain commits 2-bit
     * permanently; kv_truncate after a reject must roll back across the
     * residual region). The residual buffer is sized R + m_max so up
     * to m_max tokens can sit past the drain threshold safely. Drain
     * happens on subsequent decode_step (accept) or kv_truncate (reject). */
    sess->kv_len += k;
    if (sess->kv_kivi_enabled) {
        sess->kivi_residual_count += k;
    }
    sess->logits_valid       = false;
    sess->next_token_pending = 0;

    /* 5. Sample one token per row of scratch_h_b. Two paths:
     *  - k=1: single-row lm_head via finalize_logits_one_row.
     *  - k>1: batched lm_head — ONE m=k linear against embed_table
     *    (lights up the M>1 IQ kernels) + per-row softcap+argmax.
     *
     * On Pi 5 the lm_head linear is the dominant cost in verify_forward
     * (262 144-wide projection). The k=1 SGEMV path is competitive only
     * because we don't batch; m=k SGEMM uses the IQ2_S / IQ3_S prefill
     * kernels and amortizes the weight stream over k columns. Greedy-
     * only softcap skip would be additive but the linear is the bulk. */
    if (k == 1) {
        s = finalize_logits_one_row(sess, 0, &out_tokens[0]);
        if (s != GEIST_OK) {
            const enum geist_status restore = deltanet_txn_restore(sess);
            if (restore != GEIST_OK)
                return restore;
            transformer_recurrent_txn_commit(sess);
            return s;
        }
    } else {
        s = finalize_logits_batch(sess, k, out_tokens);
        if (s != GEIST_OK) {
            deltanet_txn_restore(sess);
            transformer_recurrent_txn_commit(sess);
            return s;
        }
    }
    /* The verified batch already computed the exact prediction following its
     * last row. Retain it as the normal pending token; the engine can leave it
     * unconsumed instead of paying a redundant single-token prefill. */
    sess->next_token_pending = out_tokens[k - 1];
    sess->logits_valid       = true;
    return GEIST_OK;
}

enum geist_status transformer_kv_truncate(struct transformer_arch_session *sess, size_t new_len) {
    if (sess == nullptr)
        return GEIST_E_INVALID_ARG;
    if (new_len > sess->kv_len)
        return GEIST_E_INVALID_ARG; /* monotonic shrink only */

    bool keep_pending = false;
    if (sess->dn_txn_active) {
        const size_t base = sess->dn_txn_base_kv_len;
        const size_t k    = sess->dn_txn_k;
        if (new_len < base || new_len > base + k)
            return GEIST_E_INVALID_ARG;
        const size_t accepted = new_len - base;
        if (accepted == k) {
            transformer_recurrent_txn_commit(sess);
            keep_pending = sess->logits_valid;
        } else {
            deltanet_txn_restore(sess);
            transformer_recurrent_txn_commit(sess);
            if (accepted > 0) {
                const enum geist_status rs =
                        transformer_prefill_text_batch(sess, accepted, sess->dn_txn_ids);
                if (rs != GEIST_OK) {
                    /* The same prefix just succeeded as part of verify. If
                     * replay nevertheless fails, restore the known-good base
                     * rather than exposing a partially advanced recurrence. */
                    sess->dn_txn_active = true;
                    deltanet_txn_restore(sess);
                    transformer_recurrent_txn_commit(sess);
                    sess->logits_valid       = false;
                    sess->next_token_pending = 0;
                    return rs;
                }
                /* Replay materialized the exact prediction following the
                 * accepted prefix, which is the correction token the verify
                 * pass returned at accepted-1. */
                keep_pending = sess->logits_valid;
            }
        }
    }
    sess->kv_len = new_len;
    if (sess->kv_kivi_enabled) {
        /* Truncate can't cross a drain boundary backwards (drained groups
         * are 2-bit committed and can't be un-quantized). Speculative
         * K ≤ m_max=64 < R=128, so truncates in practice never reach
         * the drained region. If asked anyway, clamp to drain boundary. */
        const size_t drained = sess->kivi_drained_count;
        if (new_len < drained) {
            sess->kv_len              = drained;
            sess->kivi_residual_count = 0;
        } else {
            sess->kivi_residual_count = new_len - drained;
        }
        /* Truncate may settle residual into commit-safe territory
         * (verify_forward burst → accept → truncate at kv_len_old + a). */
        transformer_kivi_drain_full(sess);
    }
    if (!keep_pending) {
        sess->logits_valid       = false;
        sess->next_token_pending = 0;
    }
    return GEIST_OK;
}

/* ---- Batched audio prefill ------------------------------------------- */

enum geist_status transformer_prefill_audio_batch(struct transformer_arch_session *sess,
                                                  size_t                           n,
                                                  const float                     *soft_tokens) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || (n > 0 && soft_tokens == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    transformer_recurrent_txn_commit(sess);
    if (n == 0) {
        return GEIST_OK;
    }
    enum geist_status room = transformer_check_kv_room(sess, n);
    if (room != GEIST_OK) {
        return room;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    /* Pre-fill a pad-id array for compute_per_layer_inputs_batch.
     * Audio tokens use pad_token_id (0) as PLE row identity. */
    geist_token_t *pad_ids =
            heap_alloc_aligned(sess->m_max * sizeof(geist_token_t), alignof(geist_token_t));
    if (pad_ids == nullptr) {
        return GEIST_E_OOM;
    }
    for (size_t i = 0; i < sess->m_max; i++) {
        pad_ids[i] = 0;
    }

    enum geist_status rc    = GEIST_OK;
    const size_t      m_max = sess->m_max;
    for (size_t off = 0; off < n; off += m_max) {
        const size_t chunk = (n - off > m_max) ? m_max : (n - off);

        /* HF reference, two PLE components at multimodal positions
         * (language_model.forward):
         *
         *   per_layer_inputs = get_per_layer_inputs(llm_input_ids, ...)
         *       → identity component: PLE-table row of pad_token_id (0)
         *   per_layer_inputs = project_per_layer_inputs(inputs_embeds, per_layer_inputs)
         *       → projection component: model_proj of inputs_embeds, which
         *         at audio positions is the SOFT TOKEN (masked_scatter runs
         *         before the language model is called)
         *
         * So the PLE projection must see the soft tokens, not a pad row —
         * feeding it pad rows starves all 30 layers' per-layer signal of
         * audio content (#268: measured as ~10x WER vs llama.cpp on the
         * same Q4 weights while soft tokens were reference-exact).
         *
         *   1. Place soft tokens in scratch_h_a (LM residual stream, raw,
         *      no embed_scale).
         *   2. PLE precompute: identity from pad_ids, projection from the
         *      soft tokens in scratch_h_a.
         *   3. Layer loop over the same scratch_h_a.
         */

        /* 1. Soft tokens into the residual-stream scratch. */
        {
            const size_t bytes = chunk * st->d_model * sizeof(float);
            uint8_t     *dst   = (uint8_t *) v->buffer_map(sess->scratch_h_a);
            memcpy(dst, (const uint8_t *) (soft_tokens + off * st->d_model), bytes);
            v->buffer_unmap(sess->scratch_h_a);
        }

        /* 2. PLE precompute against the soft tokens. */
        struct geist_buffer *ple_buf = nullptr;
        if (st->config.has_ple) {
            rc = compute_per_layer_inputs_batch(
                    sess, chunk, pad_ids, sess->scratch_h_a, sess->scratch_per_layer_input);
            if (rc != GEIST_OK) {
                goto cleanup;
            }
            ple_buf = sess->scratch_per_layer_input;
        }

        /* 3. Layer loop. */
        const size_t q_pos = sess->kv_len;
        rc                 = transformer_run_all_layers(
                sess, q_pos, chunk, sess->scratch_h_a, ple_buf, sess->scratch_h_b);
        if (rc != GEIST_OK) {
            goto cleanup;
        }

        sess->kv_len += chunk;
        if (sess->kv_kivi_enabled) {
            sess->kivi_residual_count += chunk;
            transformer_kivi_drain_full(sess);
        }

        if (off + chunk == n) {
            rc = finalize_logits_last_row(sess, chunk);
            if (rc != GEIST_OK) {
                goto cleanup;
            }
        }
    }

cleanup: {
    void *p = pad_ids;
    safe_free(&p);
}
    return rc;
}

/* ---- Prefix pinning -------------------------------------------------- */

enum geist_status
transformer_pin_prefix(struct transformer_arch_session *sess, size_t n, const geist_token_t *ids) {
    if (sess == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n > 0 && ids == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    /* Truncate to empty so the prefill that follows starts from kv_len=0;
     * this matches lm.c::lm_pin_prefix and gives a clean snapshot point. */
    sess->kv_len             = 0;
    sess->prefix_length      = 0;
    sess->logits_valid       = false;
    sess->next_token_pending = 0;
    if (n == 0) {
        return GEIST_OK;
    }
    enum geist_status rc = transformer_prefill_text_batch(sess, n, ids);
    if (rc != GEIST_OK) {
        return rc;
    }
    sess->prefix_length = sess->kv_len;
    return GEIST_OK;
}

/* ---- AWQ scale application -------------------------------------------- *
 *
 * Folds attn_norm and ffn_norm gamma in-place against per-channel AWQ
 * scales so the runtime sees a single gamma multiply (zero AWQ overhead
 * for the foldable norms). Stores 1/s for the non-foldable o_proj and
 * down_proj inputs as host arrays on the layer struct; the forward path
 * multiplies attn_out / post-GeGLU gate by these before the linear call.
 *
 * Mirrors lm.c::apply_awq_to_layers (lm.c:1079). Names follow the AWQS
 * file convention: blk.{i}.{attn_norm.out, ffn_norm.out, attn.out, ffn.out}.
 *
 * Returns GEIST_OK if AWQ applied (or no-op if path is nullptr).
 * On size mismatch or alloc failure: returns the error WITHOUT undoing
 * already-applied changes; caller treats the state as poisoned. */
enum geist_status apply_awq_to_state(struct transformer_arch_state *st, const char *awq_path) {
    if (awq_path == nullptr)
        return GEIST_OK;

    const char           *err = nullptr;
    struct ptqtp_awq_ctx *awq = ptqtp_awq_open(awq_path, &err);
    if (awq == nullptr) {
        geist_backend_set_error(st->backend,
                                GEIST_E_FILE_NOT_FOUND,
                                "AWQ open(%s): %s",
                                awq_path,
                                err != nullptr ? err : "(no detail)");
        return GEIST_E_FILE_NOT_FOUND;
    }

    const struct geist_backend_vtbl *v = st->backend->desc->vtbl;

    enum geist_status rc = GEIST_OK;
    char              key[64];

    for (int i = 0; (size_t) i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        size_t                            n = 0;
        const float                      *s = nullptr;

        /* attn_norm gamma /= s_attn_norm */
        snprintf(key, sizeof key, "blk.%d.attn_norm.out", i);
        s = ptqtp_awq_get(awq, key, &n);
        if (s != nullptr) {
            if (n != st->d_model) {
                geist_backend_set_error(st->backend,
                                        GEIST_E_INVALID_ARG,
                                        "AWQ %s: n=%zu, expected %d",
                                        key,
                                        n,
                                        (int) st->d_model);
                rc = GEIST_E_INVALID_ARG;
                goto cleanup;
            }
            float *g = (float *) v->buffer_map(L->attn_norm.buffer);
            for (size_t j = 0; j < st->d_model; j++)
                g[j] /= s[j];
            v->buffer_unmap(L->attn_norm.buffer);
        }

        /* ffn_norm gamma /= s_ffn_norm */
        snprintf(key, sizeof key, "blk.%d.ffn_norm.out", i);
        s = ptqtp_awq_get(awq, key, &n);
        if (s != nullptr) {
            if (n != st->d_model) {
                geist_backend_set_error(st->backend,
                                        GEIST_E_INVALID_ARG,
                                        "AWQ %s: n=%zu, expected %d",
                                        key,
                                        n,
                                        (int) st->d_model);
                rc = GEIST_E_INVALID_ARG;
                goto cleanup;
            }
            float *g = (float *) v->buffer_map(L->ffn_norm.buffer);
            for (size_t j = 0; j < st->d_model; j++)
                g[j] /= s[j];
            v->buffer_unmap(L->ffn_norm.buffer);
        }

        /* o_proj input: 1/s applied at runtime to attn_out [q_out]. */
        snprintf(key, sizeof key, "blk.%d.attn.out", i);
        s = ptqtp_awq_get(awq, key, &n);
        if (s != nullptr) {
            if (n != (size_t) L->q_out) {
                geist_backend_set_error(st->backend,
                                        GEIST_E_INVALID_ARG,
                                        "AWQ %s: n=%zu, expected %zu",
                                        key,
                                        n,
                                        (size_t) L->q_out);
                rc = GEIST_E_INVALID_ARG;
                goto cleanup;
            }
            L->o_awq_inv_scale = heap_alloc_aligned(n * sizeof(float), alignof(float));
            if (L->o_awq_inv_scale == nullptr) {
                rc = GEIST_E_OOM;
                goto cleanup;
            }
            for (size_t j = 0; j < n; j++)
                L->o_awq_inv_scale[j] = 1.0f / s[j];
        }

        /* down_proj input: 1/s applied at runtime to post-GeGLU gate [intermediate]. */
        snprintf(key, sizeof key, "blk.%d.ffn.out", i);
        s = ptqtp_awq_get(awq, key, &n);
        if (s != nullptr) {
            if (n != (size_t) L->intermediate) {
                geist_backend_set_error(st->backend,
                                        GEIST_E_INVALID_ARG,
                                        "AWQ %s: n=%zu, expected %zu",
                                        key,
                                        n,
                                        (size_t) L->intermediate);
                rc = GEIST_E_INVALID_ARG;
                goto cleanup;
            }
            L->down_awq_inv_scale = heap_alloc_aligned(n * sizeof(float), alignof(float));
            if (L->down_awq_inv_scale == nullptr) {
                rc = GEIST_E_OOM;
                goto cleanup;
            }
            for (size_t j = 0; j < n; j++)
                L->down_awq_inv_scale[j] = 1.0f / s[j];
        }
    }

cleanup:
    ptqtp_awq_close(awq);
    return rc;
}
