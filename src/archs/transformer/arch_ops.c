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

/* ---- Batched text prefill -------------------------------------------- */

static enum geist_status
prefill_text_batch_inner(struct transformer_arch_state *st, size_t n, const geist_token_t *ids) {
    if (st == nullptr || (n > 0 && ids == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    if (n == 0) {
        return GEIST_OK;
    }
    enum geist_status room = transformer_check_kv_room(st, n);
    if (room != GEIST_OK) {
        return room;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    /* sqrt(d_model) embedding scale is Gemma-3/4-specific; Llama / BitNet
     * don't scale. has_ple gates Gemma family identity. */
    const float embed_scale = st->config.has_ple ? sqrtf((float) st->d_model) : 1.0f;

    for (size_t off = 0; off < n; off += st->m_max) {
        const size_t chunk = (n - off > st->m_max) ? st->m_max : (n - off);

        /* 1. Embed all chunk tokens into scratch_h_a [chunk, HIDDEN]. */
        {
            float *h_dst = (float *) v->buffer_map(st->sess->scratch_h_a);
            for (size_t t = 0; t < chunk; t++) {
                enum geist_status s = dequant_one_row(
                        be, &st->embed_table, (size_t) ids[off + t], h_dst + t * st->d_model);
                if (s != GEIST_OK) {
                    v->buffer_unmap(st->sess->scratch_h_a);
                    return s;
                }
            }
            if (embed_scale != 1.0f) {
                const size_t n_floats = chunk * st->d_model;
                for (size_t i = 0; i < n_floats; i++) {
                    h_dst[i] *= embed_scale;
                }
            }
            v->buffer_unmap(st->sess->scratch_h_a);
        }

        /* 2. Batched PLE precompute. P1.5.b: skipped for non-PLE families. */
        enum geist_status    s       = GEIST_OK;
        struct geist_buffer *ple_buf = nullptr;
        if (st->config.has_ple) {
            s = compute_per_layer_inputs_batch(
                    st, chunk, ids + off, st->sess->scratch_h_a, st->sess->scratch_per_layer_input);
            if (s != GEIST_OK) {
                return s;
            }
            ple_buf = st->sess->scratch_per_layer_input;
        }

        /* 3. Layer loop seq=chunk. */
        const size_t q_pos = st->sess->kv_len;
        s                  = transformer_run_all_layers(
                st, q_pos, chunk, st->sess->scratch_h_a, ple_buf, st->sess->scratch_h_b);
        if (s != GEIST_OK) {
            return s;
        }

        /* 4. Advance kv_len by chunk. */
        st->sess->kv_len += chunk;
        if (st->sess->kv_kivi_enabled) {
            st->sess->kivi_residual_count += chunk;
            transformer_kivi_drain_full(st);
        }

        /* 5. On the final chunk, compute logits for the last token so
         *    ops->decode_step has a pending prediction. */
        if (off + chunk == n) {
            s = finalize_logits_last_row(st, chunk);
            if (s != GEIST_OK) {
                return s;
            }
        }
    }
    return GEIST_OK;
}

enum geist_status transformer_prefill_text_batch(struct transformer_arch_state *st,
                                                 size_t                         n,
                                                 const geist_token_t           *ids) {
    if (st == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    /* Prefill is compute-bound; let the backend enter its prefill thread
     * regime (cpu_neon bumps OMP to all cores). Restored after the pass. */
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const int                        region_tok =
            v->parallel_region_begin ? v->parallel_region_begin(be, GEIST_REGION_PREFILL_BATCH) : 0;
    const enum geist_status s = prefill_text_batch_inner(st, n, ids);
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

enum geist_status transformer_verify_forward(struct transformer_arch_state *st,
                                             size_t                         k,
                                             const geist_token_t           *ids,
                                             geist_token_t                 *out_tokens) {
    if (st == nullptr || k == 0 || ids == nullptr || out_tokens == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (k > st->m_max) {
        /* Spec K should fit in one prefill chunk. Larger requires chunking. */
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status room = transformer_check_kv_room(st, k);
    if (room != GEIST_OK) {
        return room;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const float embed_scale             = st->config.has_ple ? sqrtf((float) st->d_model) : 1.0f;

    /* 1. Embed all k tokens into scratch_h_a [k, HIDDEN]. */
    {
        float *h_dst = (float *) v->buffer_map(st->sess->scratch_h_a);
        for (size_t t = 0; t < k; t++) {
            enum geist_status s =
                    dequant_one_row(be, &st->embed_table, (size_t) ids[t], h_dst + t * st->d_model);
            if (s != GEIST_OK) {
                v->buffer_unmap(st->sess->scratch_h_a);
                return s;
            }
        }
        if (embed_scale != 1.0f) {
            const size_t n_floats = k * st->d_model;
            for (size_t i = 0; i < n_floats; i++) {
                h_dst[i] *= embed_scale;
            }
        }
        v->buffer_unmap(st->sess->scratch_h_a);
    }

    /* 2. PLE precompute. P1.5.b: skipped for non-PLE families. */
    enum geist_status    s       = GEIST_OK;
    struct geist_buffer *ple_buf = nullptr;
    if (st->config.has_ple) {
        s = compute_per_layer_inputs_batch(
                st, k, ids, st->sess->scratch_h_a, st->sess->scratch_per_layer_input);
        if (s != GEIST_OK) {
            return s;
        }
        ple_buf = st->sess->scratch_per_layer_input;
    }

    /* 3. Layer loop. */
    const size_t q_pos = st->sess->kv_len;
    s                  = transformer_run_all_layers(
            st, q_pos, k, st->sess->scratch_h_a, ple_buf, st->sess->scratch_h_b);
    if (s != GEIST_OK) {
        return s;
    }

    /* 4. Advance kv_len by k — caller may truncate later on reject.
     * KIVI: bump residual_count but do NOT drain (drain commits 2-bit
     * permanently; kv_truncate after a reject must roll back across the
     * residual region). The residual buffer is sized R + m_max so up
     * to m_max tokens can sit past the drain threshold safely. Drain
     * happens on subsequent decode_step (accept) or kv_truncate (reject). */
    st->sess->kv_len += k;
    if (st->sess->kv_kivi_enabled) {
        st->sess->kivi_residual_count += k;
    }
    st->sess->logits_valid       = false;
    st->sess->next_token_pending = 0;

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
        s = finalize_logits_one_row(st, 0, &out_tokens[0]);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        s = finalize_logits_batch(st, k, out_tokens);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return GEIST_OK;
}

void transformer_kv_truncate(struct transformer_arch_state *st, size_t new_len) {
    if (st == nullptr)
        return;
    if (new_len > st->sess->kv_len)
        return; /* monotonic shrink only */
    st->sess->kv_len = new_len;
    if (st->sess->kv_kivi_enabled) {
        /* Truncate can't cross a drain boundary backwards (drained groups
         * are 2-bit committed and can't be un-quantized). Speculative
         * K ≤ m_max=64 < R=128, so truncates in practice never reach
         * the drained region. If asked anyway, clamp to drain boundary. */
        const size_t drained = st->sess->kivi_drained_count;
        if (new_len < drained) {
            st->sess->kv_len              = drained;
            st->sess->kivi_residual_count = 0;
        } else {
            st->sess->kivi_residual_count = new_len - drained;
        }
        /* Truncate may settle residual into commit-safe territory
         * (verify_forward burst → accept → truncate at kv_len_old + a). */
        transformer_kivi_drain_full(st);
    }
    st->sess->logits_valid       = false;
    st->sess->next_token_pending = 0;
}

/* ---- Batched audio prefill ------------------------------------------- */

enum geist_status transformer_prefill_audio_batch(struct transformer_arch_state *st,
                                                  size_t                         n,
                                                  const float                   *soft_tokens) {
    if (st == nullptr || (n > 0 && soft_tokens == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    if (n == 0) {
        return GEIST_OK;
    }
    enum geist_status room = transformer_check_kv_room(st, n);
    if (room != GEIST_OK) {
        return room;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    /* Pre-fill a pad-id array for compute_per_layer_inputs_batch.
     * Audio tokens use pad_token_id (0) as PLE row identity. */
    geist_token_t *pad_ids =
            heap_alloc_aligned(st->sess->m_max * sizeof(geist_token_t), alignof(geist_token_t));
    if (pad_ids == nullptr) {
        return GEIST_E_OOM;
    }
    for (size_t i = 0; i < st->sess->m_max; i++) {
        pad_ids[i] = 0;
    }

    enum geist_status rc    = GEIST_OK;
    const size_t      m_max = st->sess->m_max;
    for (size_t off = 0; off < n; off += m_max) {
        const size_t chunk = (n - off > m_max) ? m_max : (n - off);

        /* HF reference (modeling_gemma4.py:2280-2282) builds the PLE
         * input asymmetrically: at multimodal soft-token positions, it
         * uses the RAW pad_token embedding row (`embed_tokens.weight[pad_id]`),
         * UNSCALED — unlike text positions which use scaled embeddings.
         *
         *   pad_embedding = embed_tokens.weight[pad_id]   ← raw row, no embed_scale
         *   llm_inputs_embeds[mm_mask] = pad_embedding
         *   per_layer_inputs = get_per_layer_inputs(pad_ids, llm_inputs_embeds)
         *
         * The LM residual stream at those positions then gets the actual
         * soft tokens (vision/audio) masked-scattered in AFTER the PLE
         * precompute. We replicate this two-stream pattern explicitly:
         *
         *   1. Populate scratch_h_a with pad_embedding rows (UNSCALED),
         *      run the PLE precompute against it.
         *   2. Overwrite scratch_h_a with the actual soft tokens.
         *   3. Run the layer loop with soft tokens in residual + the
         *      PLE buffer we just precomputed.
         */

        /* 1. PLE input = pad_embedding (raw, unscaled). */
        struct geist_buffer *ple_buf = nullptr;
        if (st->config.has_ple) {
            float *h_dst = (float *) v->buffer_map(st->sess->scratch_h_a);
            for (size_t t = 0; t < chunk; t++) {
                enum geist_status s = dequant_one_row(
                        be, &st->embed_table, 0 /* pad_token_id */, h_dst + t * st->d_model);
                if (s != GEIST_OK) {
                    v->buffer_unmap(st->sess->scratch_h_a);
                    rc = s;
                    goto cleanup;
                }
            }
            v->buffer_unmap(st->sess->scratch_h_a);

            rc = compute_per_layer_inputs_batch(
                    st, chunk, pad_ids, st->sess->scratch_h_a, st->sess->scratch_per_layer_input);
            if (rc != GEIST_OK) {
                goto cleanup;
            }
            ple_buf = st->sess->scratch_per_layer_input;
        }

        /* 2. Overwrite scratch_h_a with the actual soft tokens for the
         * layer loop (these live in the LM residual stream). */
        {
            const size_t bytes = chunk * st->d_model * sizeof(float);
            uint8_t     *dst   = (uint8_t *) v->buffer_map(st->sess->scratch_h_a);
            memcpy(dst, (const uint8_t *) (soft_tokens + off * st->d_model), bytes);
            v->buffer_unmap(st->sess->scratch_h_a);
        }

        /* 3. Layer loop. */
        const size_t q_pos = st->sess->kv_len;
        rc                 = transformer_run_all_layers(
                st, q_pos, chunk, st->sess->scratch_h_a, ple_buf, st->sess->scratch_h_b);
        if (rc != GEIST_OK) {
            goto cleanup;
        }

        st->sess->kv_len += chunk;
        if (st->sess->kv_kivi_enabled) {
            st->sess->kivi_residual_count += chunk;
            transformer_kivi_drain_full(st);
        }

        if (off + chunk == n) {
            rc = finalize_logits_last_row(st, chunk);
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
transformer_pin_prefix(struct transformer_arch_state *st, size_t n, const geist_token_t *ids) {
    if (st == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n > 0 && ids == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    /* Truncate to empty so the prefill that follows starts from kv_len=0;
     * this matches lm.c::lm_pin_prefix and gives a clean snapshot point. */
    st->sess->kv_len             = 0;
    st->sess->prefix_length      = 0;
    st->sess->logits_valid       = false;
    st->sess->next_token_pending = 0;
    if (n == 0) {
        return GEIST_OK;
    }
    enum geist_status rc = transformer_prefill_text_batch(st, n, ids);
    if (rc != GEIST_OK) {
        return rc;
    }
    st->sess->prefix_length = st->sess->kv_len;
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

    const struct geist_backend_vtbl *v  = st->backend->desc->vtbl;
    enum geist_status                rc = GEIST_OK;
    char                             key[64];

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
