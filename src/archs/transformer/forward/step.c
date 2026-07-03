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

void transformer_kivi_drain_full(struct transformer_arch_state *st) {
    if (!st->sess->kv_kivi_enabled)
        return;
    if (st->sess->kivi_residual_count < KIVI_K_GROUP_SIZE)
        return;
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const size_t                     R  = KIVI_K_GROUP_SIZE;

    while (st->sess->kivi_residual_count >= R) {
        for (size_t li = 0; li < st->n_layers; li++) {
            if (st->layers[li].is_kv_shared)
                continue;
            const size_t hd       = st->layers[li].head_dim;
            float       *k_res    = (float *) v->buffer_map(st->sess->k_residual[li]);
            float       *v_res    = (float *) v->buffer_map(st->sess->v_residual[li]);
            uint8_t     *k_q4     = (uint8_t *) v->buffer_map(st->sess->k_kivi_q[li]);
            uint8_t     *v_q4     = (uint8_t *) v->buffer_map(st->sess->v_kivi_q[li]);
            float       *k_scales = (float *) v->buffer_map(st->sess->k_kivi_scales[li]);
            float       *k_zeros  = (float *) v->buffer_map(st->sess->k_kivi_zeros[li]);
            float       *v_scales = (float *) v->buffer_map(st->sess->v_kivi_scales[li]);
            float       *v_zeros  = (float *) v->buffer_map(st->sess->v_kivi_zeros[li]);
            kivi_drain_one_layer(k_res,
                                 v_res,
                                 k_q4,
                                 v_q4,
                                 k_scales,
                                 k_zeros,
                                 v_scales,
                                 v_zeros,
                                 st->sess->kivi_drained_count,
                                 st->sess->kivi_residual_count,
                                 R,
                                 hd,
                                 st->n_kv_heads);
            v->buffer_unmap(st->sess->k_residual[li]);
            v->buffer_unmap(st->sess->v_residual[li]);
            v->buffer_unmap(st->sess->k_kivi_q[li]);
            v->buffer_unmap(st->sess->v_kivi_q[li]);
            v->buffer_unmap(st->sess->k_kivi_scales[li]);
            v->buffer_unmap(st->sess->k_kivi_zeros[li]);
            v->buffer_unmap(st->sess->v_kivi_scales[li]);
            v->buffer_unmap(st->sess->v_kivi_zeros[li]);
        }
        st->sess->kivi_drained_count += R;
        st->sess->kivi_residual_count -= R;
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
[[nodiscard]] enum geist_status transformer_run_all_layers(struct transformer_arch_state *st,
                                                           size_t               q_position,
                                                           size_t               seq,
                                                           struct geist_buffer *initial_h_buf,
                                                           struct geist_buffer *per_layer_input_buf,
                                                           struct geist_buffer *out_h_buf) {

    struct geist_backend            *be                    = st->backend;
    const struct geist_backend_vtbl *v                     = be->desc->vtbl;
    const size_t                     row_bytes_h           = st->d_model * sizeof(float);
    const size_t                     row_bytes_ple         = st->hidden_per_layer * sizeof(float);
    const size_t                     row_bytes_per_tok_ple = st->ple_out * sizeof(float);

    /* Seed scratch_h_a with seq rows of HIDDEN. */
    {
        const uint8_t *src = (const uint8_t *) v->buffer_map(initial_h_buf);
        uint8_t       *dst = (uint8_t *) v->buffer_map(st->sess->scratch_h_a);
        memcpy(dst, src, seq * row_bytes_h);
        v->buffer_unmap(initial_h_buf);
        v->buffer_unmap(st->sess->scratch_h_a);
    }

    struct geist_buffer *h_in  = st->sess->scratch_h_a;
    struct geist_buffer *h_out = st->sess->scratch_h_b;

    for (size_t li = 0; li < st->n_layers; li++) {
        /* P1.5.b: gather this layer's per_layer_input slices into
         * scratch_ple_lookup, but only when the family actually has
         * PLE (per_layer_input_buf != nullptr). Non-PLE families
         * (Llama / Mistral) pass nullptr through to forward_one_layer
         * which then skips the PLE injection block. */
        struct geist_buffer *layer_ple_buf = nullptr;
        if (per_layer_input_buf != nullptr) {
            if (v->buffer_copy != nullptr) {
                /* Device-side gather: keeps batched GPU backends from
                 * flushing their pipeline for a host memcpy each layer. */
                enum geist_status cs = GEIST_OK;
                for (size_t t = 0; cs == GEIST_OK && t < seq; t++) {
                    cs = v->buffer_copy(st->sess->scratch_ple_lookup,
                                        t * row_bytes_ple,
                                        per_layer_input_buf,
                                        t * row_bytes_per_tok_ple +
                                            li * row_bytes_ple,
                                        row_bytes_ple);
                }
                if (cs != GEIST_OK) {
                    return cs;
                }
            } else {
                const uint8_t *src =
                        (const uint8_t *) v->buffer_map(per_layer_input_buf);
                uint8_t *dst =
                        (uint8_t *) v->buffer_map(st->sess->scratch_ple_lookup);
                for (size_t t = 0; t < seq; t++) {
                    memcpy(dst + t * row_bytes_ple,
                           src + t * row_bytes_per_tok_ple + li * row_bytes_ple,
                           row_bytes_ple);
                }
                v->buffer_unmap(per_layer_input_buf);
                v->buffer_unmap(st->sess->scratch_ple_lookup);
            }
            layer_ple_buf = st->sess->scratch_ple_lookup;
        }

        enum geist_status s = transformer_forward_one_layer(st,
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
        const uint8_t *src = (const uint8_t *) v->buffer_map(h_in);
        uint8_t       *dst = (uint8_t *) v->buffer_map(out_h_buf);
        memcpy(dst, src, seq * row_bytes_h);
        v->buffer_unmap(h_in);
        v->buffer_unmap(out_h_buf);
    }
    return GEIST_OK;
}

/* Dequantize one row of the embed_table (Q-format) into a host-pointer
 * region of a HIDDEN-sized scratch buffer. Gemma 3/4 multiplies the
 * embedding by sqrt(d_model) (mirrors lm.c's compute_token_inputs);
 * Llama / BitNet do NOT scale the embedding. The scale is gated on
 * has_ple because PLE is Gemma-family-exclusive. */
[[nodiscard]] static enum geist_status embed_lookup_and_scale(struct transformer_arch_state *st,
                                                              geist_token_t        token_id,
                                                              struct geist_buffer *out_h_buf) {

    if (token_id < 0 || (size_t) token_id >= (size_t) st->vocab_size) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

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
[[nodiscard]] static enum geist_status transformer_run_one_step(struct transformer_arch_state *st,
                                                                geist_token_t  ple_token_id,
                                                                geist_token_t *out_token) {

    enum geist_status s;

    /* 1. PLE precompute for this token using the seeded h. P1.5.b:
     *    family-conditional — non-PLE families skip the precompute and
     *    pass nullptr through to run_all_layers, which then skips the
     *    per-layer gather and the layer body's PLE injection block. */
    struct geist_buffer *ple_buf = nullptr;
    if (st->config.has_ple) {
        s = transformer_compute_per_layer_input(
                st, ple_token_id, st->sess->scratch_h_a, st->sess->scratch_per_layer_input);
        if (s != GEIST_OK) {
            return s;
        }
        ple_buf = st->sess->scratch_per_layer_input;
    }
    (void) ple_token_id; /* unused when !has_ple */

    /* 2. Layer loop (seq=1 for the single-token path). q_position = current
     *    kv_len; advance after. */
    const size_t q_position = st->sess->kv_len;
    s                       = transformer_run_all_layers(
            st, q_position, /* seq = */ 1, st->sess->scratch_h_a, ple_buf, st->sess->scratch_h_b);
    if (s != GEIST_OK) {
        return s;
    }

    geist_token_t best_id;
    s = finalize_logits_one_row(st, 0, &best_id);
    if (s != GEIST_OK) {
        return s;
    }

    /* 3. Advance KV, stash prediction. */
    st->sess->kv_len = q_position + 1;
    if (st->sess->kv_kivi_enabled) {
        st->sess->kivi_residual_count += 1;
        transformer_kivi_drain_full(st);
    }
    st->sess->next_token_pending = best_id;
    st->sess->logits_valid       = true;

    *out_token = best_id;
    return GEIST_OK;
}

enum geist_status transformer_decode_step(struct transformer_arch_state *st,
                                          geist_token_t                  input_token,
                                          geist_token_t                 *out_token) {
    if (st == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    /* Decode is memory-bound; let the backend enter its decode thread regime
     * (cpu_neon caps OMP threads). Restored after the step. */
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const int                        region_tok =
            v->parallel_region_begin ? v->parallel_region_begin(be, GEIST_REGION_DECODE_STEP) : 0;
    /* Embed the input token into scratch_h_a, scale by sqrt(HIDDEN). */
    enum geist_status s = embed_lookup_and_scale(st, input_token, st->sess->scratch_h_a);
    if (s == GEIST_OK) {
        /* PLE uses the token's actual id (text path). */
        s = transformer_run_one_step(st, input_token, out_token);
    }
    if (v->parallel_region_end) {
        v->parallel_region_end(be, region_tok);
    }
    return s;
}

enum geist_status transformer_advance_audio_token(struct transformer_arch_state *st,
                                                  const float                   *h_in_host) {
    if (st == nullptr || h_in_host == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    /* Audio soft-tokens enter the residual stream directly (no embed
     * lookup, no sqrt scale). Copy host bytes into scratch_h_a. */
    {
        const size_t bytes = (size_t) st->d_model * sizeof(float);
        uint8_t     *dst   = (uint8_t *) v->buffer_map(st->sess->scratch_h_a);
        memcpy(dst, h_in_host, bytes);
        v->buffer_unmap(st->sess->scratch_h_a);
    }
    /* PLE token-identity is the pad token (0) per HF masked-scatter. */
    geist_token_t out_unused;
    return transformer_run_one_step(st, 0, &out_unused);
}

void transformer_state_reset(struct transformer_arch_state *st) {
    if (st == nullptr) {
        return;
    }
    /* Truncate to pinned prefix length (0 if no prefix has been pinned).
     * The KV state up to prefix_length stays valid in the cache buffers;
     * future prefill/decode appends start at kv_len. */
    st->sess->kv_len = st->sess->prefix_length;
    if (st->sess->kv_kivi_enabled) {
        /* Sync drain + residual counters to the new kv_len. The standard
         * pin_prefix flow pre-prefills with KIVI active, so the counters
         * are already aligned (drained = floor(kv_len/R)*R, residual =
         * remainder). Reset preserves this alignment. */
        st->sess->kivi_drained_count  = (st->sess->kv_len / KIVI_K_GROUP_SIZE) * KIVI_K_GROUP_SIZE;
        st->sess->kivi_residual_count = st->sess->kv_len - st->sess->kivi_drained_count;
    }
    st->sess->logits_valid       = false;
    st->sess->next_token_pending = 0;
}

void transformer_state_apply_opts(struct transformer_arch_state   *st,
                                  const struct geist_session_opts *opts) {
    if (st == nullptr || opts == nullptr) {
        return;
    }
    st->sess->temperature = opts->temperature;
    st->sess->top_p       = opts->top_p > 0.0f ? opts->top_p : 1.0f;
    st->sess->top_k       = opts->top_k;
    if (opts->random_seed != 0) {
        geist_rng_seed(&st->sess->rng, opts->random_seed);
    }

    /* (Re)allocate the sampler workspace if a non-greedy mode is now in
     * play and the workspace isn't already sized for the vocab. ~3 MB
     * for VOCAB=262144; greedy mode skips this. */
    const bool needs_ws =
            st->sess->temperature > 0.0f &&
            (st->sess->top_k > 1 || (st->sess->top_p > 0.0f && st->sess->top_p < 1.0f));
    if (needs_ws && st->sess->sampler_ws.n_vocab != (size_t) st->vocab_size) {
        geist_sampler_workspace_destroy(&st->sess->sampler_ws);
        (void) geist_sampler_workspace_init(&st->sess->sampler_ws, (size_t) st->vocab_size);
    }
}
