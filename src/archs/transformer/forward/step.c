/*
 * src/archs/transformer/forward/step.c — decode step, layer loop,
 * embedding lookup, KIVI drain, and the small session-state lifecycle
 * helpers that hang off forward.c.
 *
 * Layer: ARCHITECTURE.
 *
 * Extracted from forward.c during R4 of the C23/AGENT.md cleanup.
 * Contains:
 *
 *   transformer_kivi_drain_full   — group-drain residual KV across layers
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
#include "../forward.h"

#include "quant.h"
#include "gemma4_kernels.h"
#include "kivi.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void transformer_kivi_drain_full(struct transformer_arch_session *sess) {
    struct transformer_arch_state *st = sess->model;
    if (!sess->kv_kivi_enabled)
        return;
    if (sess->kivi_residual_count < KIVI_K_GROUP_SIZE)
        return;
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
        if (es == GEIST_OK) {
            return GEIST_OK;
        }
        return es;
    }

    float            *dst = (float *) v->buffer_map(out_h_buf);
    enum geist_status s   = dequant_one_row(be, &st->embed_table, (size_t) token_id, dst);
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
    return GEIST_OK;
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
        transformer_kivi_drain_full(sess);
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
        sess->kivi_drained_count  = (sess->kv_len / KIVI_K_GROUP_SIZE) * KIVI_K_GROUP_SIZE;
        sess->kivi_residual_count = sess->kv_len - sess->kivi_drained_count;
    }
    sess->logits_valid       = false;
    sess->next_token_pending = 0;
    transformer_mtp_reset(sess);
    /* Gated-DeltaNet layers carry recurrent state with no rewind — a
     * reset clears it to the empty sequence (#281). Prefix pinning is
     * unsupported for this family (prefix_length stays 0). */
    if (sess->dn_conv_state != nullptr || sess->dn_S != nullptr) {
        const struct transformer_arch_state *st = sess->model;
        const size_t key_dim                    = st->config.dn_n_k_heads * st->config.dn_head_k;
        const size_t value_dim                  = st->config.dn_n_v_heads * st->config.dn_head_v;
        const size_t conv_n = (st->config.dn_conv_kernel - 1) * (2 * key_dim + value_dim);
        const size_t s_n    = st->config.dn_n_v_heads * st->config.dn_head_k * st->config.dn_head_v;
        for (size_t li = 0; li < st->n_layers; li++) {
            if (sess->dn_conv_state != nullptr && sess->dn_conv_state[li] != nullptr) {
                float *p = (float *) st->backend->desc->vtbl->buffer_map(sess->dn_conv_state[li]);
                if (p != nullptr) {
                    memset(p, 0, conv_n * sizeof(float));
                    st->backend->desc->vtbl->buffer_unmap(sess->dn_conv_state[li]);
                }
            }
            if (sess->dn_S != nullptr && sess->dn_S[li] != nullptr) {
                float *p = (float *) st->backend->desc->vtbl->buffer_map(sess->dn_S[li]);
                if (p != nullptr) {
                    memset(p, 0, s_n * sizeof(float));
                    st->backend->desc->vtbl->buffer_unmap(sess->dn_S[li]);
                }
            }
        }
    }
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
