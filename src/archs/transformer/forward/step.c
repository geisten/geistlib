/*
 * src/archs/transformer/forward/step.c — decode step, layer loop,
 * embedding lookup, KIVI drain, and the small session-state lifecycle
 * helpers that hang off forward.c.
 *
 * Layer: ARCHITECTURE.
 *
 * Contains:
 *
 *   transformer_kivi_drain_full   — group-drain residual KV across layers
 *   transformer_kivi_pin_save / _restore — a pinned prefix's residual rows
 *   transformer_run_all_layers    — single- + multi-step layer loop
 *   embed_lookup_and_scale (st.)  — token id → hidden vector + scale
 *   transformer_run_one_step (st.)— one-token forward including head
 *   transformer_decode_step       — public decode driver
 *   transformer_advance_audio_token — public audio-token advance
 *   transformer_state_reset       — public state reset
 *   transformer_state_apply_opts  — public session-opts apply
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../arch_state.h"
#include "../rotation.h"
#include "../forward.h"

#include "checked.h"
#include "heap.h"
#include "quant.h"
#include "gemma4_kernels.h"
#include "kivi.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum geist_status transformer_kivi_drain_full(struct transformer_arch_session *sess) {
    struct transformer_arch_state *st = sess->model;
    if (!sess->kv_kivi_enabled)
        return GEIST_OK;
    if (sess->kivi_residual_count < KIVI_K_GROUP_SIZE)
        return GEIST_OK;
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const size_t                     R  = KIVI_K_GROUP_SIZE;

    while (sess->kivi_residual_count >= R) {
        for (size_t li = 0; li < st->n_layers; li++) {
            if (st->layers[li].is_kv_shared)
                continue;
            const size_t hd       = st->layers[li].head_dim;
            float       *k_res    = (float *) v->buffer_map(sess->k_residual[li]);
            float       *v_res    = (float *) v->buffer_map(sess->v_residual[li]);
            uint8_t     *k_q4     = (uint8_t *) v->buffer_map(sess->k_kivi_q[li]);
            uint8_t     *v_q4     = (uint8_t *) v->buffer_map(sess->v_kivi_q[li]);
            float       *k_scales = (float *) v->buffer_map(sess->k_kivi_scales[li]);
            float       *k_zeros  = (float *) v->buffer_map(sess->k_kivi_zeros[li]);
            float       *v_scales = (float *) v->buffer_map(sess->v_kivi_scales[li]);
            float       *v_zeros  = (float *) v->buffer_map(sess->v_kivi_zeros[li]);
            if (k_res == nullptr || v_res == nullptr || k_q4 == nullptr || v_q4 == nullptr ||
                k_scales == nullptr || k_zeros == nullptr || v_scales == nullptr ||
                v_zeros == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            kivi_drain_one_layer(sess->kivi_drained_count,
                                 sess->kivi_residual_count,
                                 R,
                                 hd,
                                 st->n_kv_heads,
                                 k_res,
                                 v_res,
                                 k_q4,
                                 v_q4,
                                 k_scales,
                                 k_zeros,
                                 v_scales,
                                 v_zeros);
            v->buffer_unmap(sess->k_residual[li]);
            v->buffer_unmap(sess->v_residual[li]);
            v->buffer_unmap(sess->k_kivi_q[li]);
            v->buffer_unmap(sess->v_kivi_q[li]);
            v->buffer_unmap(sess->k_kivi_scales[li]);
            v->buffer_unmap(sess->k_kivi_zeros[li]);
            v->buffer_unmap(sess->v_kivi_scales[li]);
            v->buffer_unmap(sess->v_kivi_zeros[li]);
        }
        sess->kivi_drained_count += R;
        sess->kivi_residual_count -= R;
    }
    return GEIST_OK;
}

/* A pinned prefix that is no whole number of groups keeps its last
 * (prefix_length mod R) rows in the residual ring. The first drain of a
 * later turn packs them into a 2-bit group with that turn's tokens and
 * moves later tokens over their rows, but a reset to the prefix reads them
 * from those rows again. So pinning copies them out, and reset writes them
 * back: the prefix is then the ring and groups it was at pin time. */
[[nodiscard]] static enum geist_status kivi_pin_copy(struct transformer_arch_session *sess,
                                                     bool                             save) {
    const struct transformer_arch_state *st   = sess->model;
    const struct geist_backend_vtbl     *v    = st->backend->desc->vtbl;
    float                               *tail = sess->kivi_pin_tail;
    for (size_t li = 0; li < st->n_layers; li++) {
        if (sess->k_residual[li] == nullptr) {
            continue; /* KV-shared or DeltaNet: no ring */
        }
        const size_t         len = sess->kivi_pin_rows * st->n_kv_heads * st->layers[li].head_dim;
        struct geist_buffer *ring[2] = {sess->k_residual[li], sess->v_residual[li]};
        for (size_t i = 0; i < 2; i++) {
            float *rows = (float *) v->buffer_map(ring[i]);
            if (rows == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            memcpy(save ? tail : rows, save ? rows : tail, len * sizeof *tail);
            v->buffer_unmap(ring[i]);
            tail += len;
        }
    }
    return GEIST_OK;
}

enum geist_status transformer_kivi_pin_save(struct transformer_arch_session *sess) {
    void *old = sess->kivi_pin_tail;
    safe_free(&old);
    sess->kivi_pin_tail = nullptr;
    sess->kivi_pin_rows = 0;
    const size_t rows   = sess->kivi_residual_count; /* < R after the prefill's drains */
    if (!sess->kv_kivi_enabled || rows == 0) {
        return GEIST_OK;
    }
    const struct transformer_arch_state *st = sess->model;
    size_t                               n  = 0;
    for (size_t li = 0; li < st->n_layers; li++) {
        size_t layer = 0;
        if (sess->k_residual[li] != nullptr &&
            (ckd_mul(&layer, rows, 2 * st->n_kv_heads * st->layers[li].head_dim) ||
             ckd_add(&n, n, layer))) {
            return GEIST_E_OOM;
        }
    }
    if (n == 0) {
        return GEIST_OK; /* no layer has a ring */
    }
    sess->kivi_pin_tail = heap_alloc_array_aligned(float, n);
    if (sess->kivi_pin_tail == nullptr) {
        return GEIST_E_OOM;
    }
    sess->kivi_pin_rows = rows;
    return kivi_pin_copy(sess, true);
}

void transformer_kivi_pin_restore(struct transformer_arch_session *sess) {
    /* Reset has put the counters back to the prefix: exactly the rows the
     * pin copied are residual again. Anything else is not that prefix. */
    if (sess->kivi_pin_rows != 0 && sess->kivi_pin_rows == sess->kivi_residual_count) {
        /* Reset has no status: an unmappable ring leaves the rows as they
         * are, and the next drain or attention map reports the backend. */
        (void) kivi_pin_copy(sess, false);
    }
}

/* ---- Layer loop + end-to-end decode_step ------------------------------ */

/* Run all 35 layers for seq tokens starting at q_position.
 *
 *   initial_h_buf:           [seq, HIDDEN]
 *   per_layer_input_buf:     [seq, NUM_LAYERS * HIDDEN_PER_LAYER]
 *                             (concatenated per_layer_inputs for the seq
 *                              tokens; for token t, layer li, the slice
 *                              is at offset t*PLE_OUT + li*HIDDEN_PER_LAYER).
 *   out_h_buf:               [seq, HIDDEN]
 *
 * Internally ping-pongs between scratch_h_a and scratch_h_b. Each layer's
 * PLE input is gathered into scratch_ple_lookup as a contiguous
 * [seq, HIDDEN_PER_LAYER] slab before the forward call. */
[[nodiscard]] enum geist_status transformer_run_all_layers(struct transformer_arch_session *sess,
                                                           size_t               q_position,
                                                           size_t               seq,
                                                           struct geist_buffer *initial_h_buf,
                                                           struct geist_buffer *per_layer_input_buf,
                                                           struct geist_buffer *out_h_buf) {
    struct transformer_arch_state *st = sess->model;

    struct geist_backend             *be                    = st->backend;
    const struct geist_backend_vtbl  *v                     = be->desc->vtbl;
    const struct geist_backend_fused *fused                 = geist_backend_fused_tbl(be);
    const size_t                      row_bytes_h           = st->d_model * sizeof(float);
    const size_t                      row_bytes_ple         = st->hidden_per_layer * sizeof(float);
    const size_t                      row_bytes_per_tok_ple = st->ple_out * sizeof(float);

    /* Seed scratch_h_a with seq rows of HIDDEN. */
    {
        /* Bound once (#352): a device copy that FAILS is a device error, not
         * a missing capability, and must not become a quiet host memcpy. */
        if (st->model_fusions.backend_buffer_copy) {
            const enum geist_status cs =
                    v->buffer_copy(sess->scratch_h_a, 0, initial_h_buf, 0, seq * row_bytes_h);
            if (cs != GEIST_OK) {
                return cs;
            }
        } else {
            const uint8_t *src = (const uint8_t *) v->buffer_map(initial_h_buf);
            uint8_t       *dst = (uint8_t *) v->buffer_map(sess->scratch_h_a);
            if (src == nullptr || dst == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            memcpy(dst, src, seq * row_bytes_h);
            v->buffer_unmap(initial_h_buf);
            v->buffer_unmap(sess->scratch_h_a);
        }
    }

    struct geist_buffer *h_in  = sess->scratch_h_a;
    struct geist_buffer *h_out = sess->scratch_h_b;

    for (size_t li = 0; li < st->n_layers; li++) {
        /* P1.5.b: gather this layer's per_layer_input slices into
         * scratch_ple_lookup, but only when the family actually has
         * PLE (per_layer_input_buf != nullptr). Non-PLE families
         * (Llama / Mistral) pass nullptr through to forward_one_layer
         * which then skips the PLE injection block. */
        struct geist_buffer *layer_ple_buf = nullptr;
        if (per_layer_input_buf != nullptr && fused->linear_t != nullptr &&
            per_layer_input_buf == sess->scratch_per_layer_input) {
            /* Batched GPU backends read the layer's PLE slice directly from
             * the slab as a strided view (see layer_ple.c) — the per-layer
             * gather would be seq*n_layers copy dispatches per chunk. */
            layer_ple_buf = per_layer_input_buf;
        } else if (per_layer_input_buf != nullptr) {
            if (st->model_fusions.backend_buffer_copy) {
                /* Device-side gather: keeps batched GPU backends from
                 * flushing their pipeline for a host memcpy each layer. */
                enum geist_status cs = GEIST_OK;
                for (size_t t = 0; cs == GEIST_OK && t < seq; t++) {
                    cs = v->buffer_copy(sess->scratch_ple_lookup,
                                        t * row_bytes_ple,
                                        per_layer_input_buf,
                                        t * row_bytes_per_tok_ple + li * row_bytes_ple,
                                        row_bytes_ple);
                }
                if (cs != GEIST_OK) {
                    return cs;
                }
            } else {
                const uint8_t *src = (const uint8_t *) v->buffer_map(per_layer_input_buf);
                uint8_t       *dst = (uint8_t *) v->buffer_map(sess->scratch_ple_lookup);
                if (src == nullptr || dst == nullptr) {
                    return GEIST_E_BACKEND; /* the backend said why */
                }
                for (size_t t = 0; t < seq; t++) {
                    memcpy(dst + t * row_bytes_ple,
                           src + t * row_bytes_per_tok_ple + li * row_bytes_ple,
                           row_bytes_ple);
                }
                v->buffer_unmap(per_layer_input_buf);
                v->buffer_unmap(sess->scratch_ple_lookup);
            }
            layer_ple_buf = sess->scratch_ple_lookup;
        }

        enum geist_status s = transformer_forward_one_layer(sess,
                                                            (int) li,
                                                            q_position,
                                                            seq,
                                                            /* advance_kv = */ false,
                                                            h_in,
                                                            layer_ple_buf,
                                                            h_out);
        if (s != GEIST_OK) {
            return s;
        }

        /* Swap. */
        struct geist_buffer *tmp = h_in;
        h_in                     = h_out;
        h_out                    = tmp;
    }

    /* After the loop, h_in is the latest output (post-swap). Copy seq rows
     * to out_h_buf. */
    {
        if (st->model_fusions.backend_buffer_copy) {
            const enum geist_status cs = v->buffer_copy(out_h_buf, 0, h_in, 0, seq * row_bytes_h);
            if (cs != GEIST_OK) {
                return cs;
            }
        } else {
            const uint8_t *src = (const uint8_t *) v->buffer_map(h_in);
            uint8_t       *dst = (uint8_t *) v->buffer_map(out_h_buf);
            if (src == nullptr || dst == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            memcpy(dst, src, seq * row_bytes_h);
            v->buffer_unmap(h_in);
            v->buffer_unmap(out_h_buf);
        }
    }
    return GEIST_OK;
}

/* Dequantize one row of the embed_table (Q-format) into a host-pointer
 * region of a HIDDEN-sized scratch buffer. Gemma 3/4 multiplies the
 * embedding by sqrt(d_model) (mirrors lm.c's compute_token_inputs);
 * Llama / BitNet do NOT scale the embedding. The scale is gated on
 * has_ple because PLE is Gemma-family-exclusive. */
[[nodiscard]] static enum geist_status embed_lookup_and_scale(struct transformer_arch_session *sess,
                                                              geist_token_t        token_id,
                                                              struct geist_buffer *out_h_buf) {
    struct transformer_arch_state *st = sess->model;

    if (token_id < 0 || (size_t) token_id >= (size_t) st->vocab_size) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    /* Device path: fused lookup+scale keeps batched GPU backends from
     * dequantizing through a mapped host pointer every token. Bound at
     * plan build. */
    if (st->model_fusions.embed_lookup_scaled) {
        struct geist_tensor     t_out = view_1d(out_h_buf, st->d_model);
        const float             scale = st->config.has_ple ? sqrtf((float) st->d_model) : 1.0f;
        const enum geist_status es =
                fused->embedding_lookup_scaled(be, &st->embed_table, token_id, scale, &t_out);
        if (es != GEIST_OK) {
            return es;
        }
        /* prism.hadamard: token_embd stores rotated rows. */
        return st->rotation.embed_inverse
                       ? transformer_rotate(st, 1, st->d_model, false, true, out_h_buf, out_h_buf)
                       : GEIST_OK;
    }

    float *dst = (float *) v->buffer_map(out_h_buf);
    if (dst == nullptr) {
        return GEIST_E_BACKEND; /* the backend said why */
    }
    enum geist_status s = dequant_one_row(be, &st->embed_table, (size_t) token_id, dst);
    if (s != GEIST_OK) {
        v->buffer_unmap(out_h_buf);
        return s;
    }
    if (st->config.has_ple) {
        const float scale = sqrtf((float) st->d_model);
        for (size_t i = 0; i < (size_t) st->d_model; i++) {
            dst[i] *= scale;
        }
    }
    v->buffer_unmap(out_h_buf);
    return st->rotation.embed_inverse
                   ? transformer_rotate(st, 1, st->d_model, false, true, out_h_buf, out_h_buf)
                   : GEIST_OK;
}

/* Post-seed step: with scratch_h_a already populated with the residual-
 * stream input (either embedded text + sqrt scale OR raw audio soft-token),
 * run PLE → 35-layer loop → output_norm → lm_head → softcap → argmax,
 * stash the result in next_token_pending, advance kv_len by 1.
 *
 * ple_token_id selects which row of the PLE table is looked up:
 *   - Text  : the input token id (PLE row matches the actual token)
 *   - Audio : 0 (pad_token_id) per HF's masked-scatter semantics
 *
 * out_token receives the greedy argmax. */
[[nodiscard]] static enum geist_status
transformer_run_one_step(struct transformer_arch_session *sess,
                         geist_token_t                    ple_token_id,
                         geist_token_t                   *out_token) {
    struct transformer_arch_state *st = sess->model;

    enum geist_status s = transformer_check_kv_room(sess, 1);
    if (s != GEIST_OK) {
        return s;
    }

    /* 1. PLE precompute for this token using the seeded h. P1.5.b:
     *    family-conditional — non-PLE families skip the precompute and
     *    pass nullptr through to run_all_layers, which then skips the
     *    per-layer gather and the layer body's PLE injection block. */
    struct geist_buffer *ple_buf = nullptr;
    if (st->config.has_ple) {
        s = transformer_compute_per_layer_input(
                sess, ple_token_id, sess->scratch_h_a, sess->scratch_per_layer_input);
        if (s != GEIST_OK) {
            return s;
        }
        ple_buf = sess->scratch_per_layer_input;
    }
    (void) ple_token_id; /* unused when !has_ple */

    /* 2. Layer loop (seq=1 for the single-token path). q_position = current
     *    kv_len; advance after. */
    const size_t q_position = sess->kv_len;
    s                       = transformer_run_all_layers(
            sess, q_position, /* seq = */ 1, sess->scratch_h_a, ple_buf, sess->scratch_h_b);
    if (s != GEIST_OK) {
        return s;
    }
    s = transformer_mtp_sync_target(sess, 1, &ple_token_id, sess->scratch_h_b);
    if (s != GEIST_OK) {
        return s;
    }

    geist_token_t best_id;
    s = finalize_logits_one_row(sess, 0, &best_id);
    if (s != GEIST_OK) {
        return s;
    }

    /* 3. Advance KV, stash prediction. */
    sess->kv_len = q_position + 1;
    if (sess->kv_kivi_enabled) {
        sess->kivi_residual_count += 1;
        s = transformer_kivi_drain_full(sess);
        if (s != GEIST_OK) {
            return s;
        }
    }
    sess->next_token_pending = best_id;
    sess->logits_valid       = true;

    *out_token = best_id;
    return GEIST_OK;
}

enum geist_status transformer_decode_step(struct transformer_arch_session *sess,
                                          geist_token_t                    input_token,
                                          geist_token_t                   *out_token) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    transformer_recurrent_txn_commit(sess);
    /* Decode is memory-bound; let the backend enter its decode thread regime
     * (cpu_neon caps OMP threads). Restored after the step. */
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const int                        region_tok =
            v->parallel_region_begin ? v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP) : 0;
    /* Embed the input token into scratch_h_a, scale by sqrt(HIDDEN). */
    enum geist_status s = embed_lookup_and_scale(sess, input_token, sess->scratch_h_a);
    if (s == GEIST_OK) {
        /* PLE uses the token's actual id (text path). */
        s = transformer_run_one_step(sess, input_token, out_token);
    }
    if (v->parallel_region_end) {
        v->parallel_region_end(be, region_tok);
    }
    return s;
}

enum geist_status transformer_advance_audio_token(struct transformer_arch_session *sess,
                                                  const float                     *h_in_host) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || h_in_host == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    transformer_recurrent_txn_commit(sess);
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    /* Audio soft-tokens enter the residual stream directly (no embed
     * lookup, no sqrt scale). Copy host bytes into scratch_h_a. */
    {
        const size_t bytes = (size_t) st->d_model * sizeof(float);
        uint8_t     *dst   = (uint8_t *) v->buffer_map(sess->scratch_h_a);
        if (dst == nullptr) {
            return GEIST_E_BACKEND; /* the backend said why */
        }
        memcpy(dst, h_in_host, bytes);
        v->buffer_unmap(sess->scratch_h_a);
    }
    /* PLE token-identity is the pad token (0) per HF masked-scatter. */
    geist_token_t out_unused;
    return transformer_run_one_step(sess, 0, &out_unused);
}

void transformer_session_reset(struct transformer_arch_session *sess) {
    if (sess == nullptr) {
        return;
    }
    transformer_recurrent_txn_commit(sess);
    /* Truncate to pinned prefix length (0 if no prefix has been pinned).
     * The KV state up to prefix_length stays valid in the cache buffers;
     * future prefill/decode appends start at kv_len. */
    sess->kv_len = sess->prefix_length;
    if (sess->kv_kivi_enabled) {
        /* Sync drain + residual counters to the new kv_len. The standard
         * pin_prefix flow pre-prefills with KIVI active, so the counters
         * are already aligned (drained = floor(kv_len/R)*R, residual =
         * remainder). Reset preserves this alignment. */
        const size_t drained      = sess->kivi_drained_count;
        sess->kivi_drained_count  = (sess->kv_len / KIVI_K_GROUP_SIZE) * KIVI_K_GROUP_SIZE;
        sess->kivi_residual_count = sess->kv_len - sess->kivi_drained_count;
        /* The counters, not the rows: a drain since the pin has moved later
         * tokens over the prefix's residual rows. Nothing else writes
         * there, so without one they are still in place. */
        if (drained != sess->kivi_drained_count) {
            transformer_kivi_pin_restore(sess);
        }
    }
    sess->logits_valid       = false;
    sess->next_token_pending = 0;
    sess->advance_deferred   = false; /* the returned token is dropped, not appended */
    transformer_mtp_reset(sess);
    /* Gated-DeltaNet layers carry recurrent state with no rewind — a
     * reset clears it to the empty sequence (#281). pin_prefix refuses a
     * prefix for this family, so prefix_length stays 0. On the host mixer
     * it only marks the state fresh: clearing it was a serial 17 ms of
     * memset on the synthetic Ternary-Bonsai-2-27B, ahead of every new
     * conversation's prefill. */
    if (sess->dn_fresh != nullptr) {
        for (size_t li = 0; li < sess->model->n_layers; li++)
            sess->dn_fresh[li] = sess->dn_S[li] != nullptr;
    } else if (sess->dn_conv_state != nullptr || sess->dn_S != nullptr) {
        const struct transformer_arch_state *st = sess->model;
        const size_t key_dim                    = st->config.dn_n_k_heads * st->config.dn_head_k;
        const size_t value_dim                  = st->config.dn_n_v_heads * st->config.dn_head_v;
        const size_t conv_n = (st->config.dn_conv_kernel - 1) * (2 * key_dim + value_dim);
        const size_t s_n    = st->config.dn_n_v_heads * st->config.dn_head_k * st->config.dn_head_v;
        for (size_t li = 0; li < st->n_layers; li++) {
            if (sess->dn_conv_state != nullptr && sess->dn_conv_state[li] != nullptr) {
                (void) transformer_dn_state_zero(
                        st->backend, sess->dn_conv_state[li], conv_n * sizeof(float));
            }
            if (sess->dn_S != nullptr && sess->dn_S[li] != nullptr) {
                (void) transformer_dn_state_zero(st->backend, sess->dn_S[li], s_n * sizeof(float));
            }
        }
    }
}

enum geist_status transformer_session_truncate(struct transformer_arch_session *sess, size_t n) {
    if (sess == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend *be = sess->model->backend;
    if (n > sess->kv_len || n < sess->prefix_length) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "truncate: %zu is outside [%zu, %zu] (the pinned prefix and "
                                "the session's length)",
                                n,
                                sess->prefix_length,
                                sess->kv_len);
        return GEIST_E_INVALID_ARG;
    }
    if (n == sess->kv_len) {
        return GEIST_OK; /* nothing to drop; pending logits stay valid */
    }
    if (n == 0) {
        transformer_session_reset(sess); /* prefix_length is 0 here */
        return GEIST_OK;
    }
    /* Gated-DeltaNet state has no way back to a position (#281): only to
     * empty, or through a snapshot. */
    if (sess->dn_conv_state != nullptr || sess->dn_S != nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "truncate: the model has recurrent (DeltaNet) layers; use "
                                "geist_session_snapshot / geist_session_restore");
        return GEIST_E_UNSUPPORTED;
    }
    /* The MTP draft head keeps its own cache in step with the target. */
    if (sess->mtp_enabled) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "truncate: MTP drafting is enabled");
        return GEIST_E_UNSUPPORTED;
    }
    /* KIVI groups below kivi_drained_count are 2-bit committed and cannot be
     * un-quantized; the speculative rewind clamps there, this refuses. */
    if (sess->kv_kivi_enabled && n < sess->kivi_drained_count) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "truncate: positions below %zu are in the compressed KIVI region",
                                sess->kivi_drained_count);
        return GEIST_E_UNSUPPORTED;
    }
    transformer_recurrent_txn_commit(sess);
    sess->kv_len = n;
    if (sess->kv_kivi_enabled) {
        sess->kivi_residual_count = n - sess->kivi_drained_count;
    }
    sess->logits_valid       = false;
    sess->next_token_pending = 0;
    sess->advance_deferred   = false;
    return GEIST_OK;
}

enum geist_status transformer_session_apply_opts(struct transformer_arch_session *sess,
                                                 const struct geist_session_opts *opts) {
    if (sess == nullptr || opts == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    sess->temperature = opts->temperature;
    sess->top_p       = opts->top_p > 0.0f ? opts->top_p : 1.0f;
    sess->top_k       = opts->top_k;
    if (opts->random_seed != 0) {
        geist_rng_seed(&sess->rng, opts->random_seed);
    }

    /* (Re)allocate the sampler workspace if a non-greedy mode is now in
     * play and the workspace isn't already sized for the vocab. ~4 MB
     * for VOCAB=262144; greedy mode skips this. Every non-greedy path —
     * plain temperature included (#331) — samples out of the workspace. */
    const bool needs_ws = sess->temperature > 0.0f;
    if (needs_ws && sess->sampler_ws.n_vocab != (size_t) sess->model->vocab_size) {
        geist_sampler_workspace_destroy(&sess->sampler_ws);
        const enum geist_status ws =
                geist_sampler_workspace_init(&sess->sampler_ws, (size_t) sess->model->vocab_size);
        if (ws != GEIST_OK) {
            return ws; /* previously swallowed */
        }
    }
    return GEIST_OK;
}
