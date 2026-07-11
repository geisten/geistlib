/*
 * src/archs/transformer/forward/head.c — output-head finalize_logits
 * family (single-row, batched, last-row).
 *
 * Layer: ARCHITECTURE.
 *
 * Extracted from forward.c during R4 of the C23/AGENT.md cleanup.
 * The three finalize_* routines share the same shape: take the
 * residual stream at the post-layer point, optionally apply the embed
 * scale, project through the lm_head linear, and write logits into the
 * session scratch. They differ in batch shape and where the next-token
 * argmax goes.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "profile.h"
#include "../arch_state.h"
#include "../forward.h"

#include "quant.h"
#include "gemma4_kernels.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum head_profile_stage {
    HEAD_PROFILE_COPY = 0,
    HEAD_PROFILE_NORM,
    HEAD_PROFILE_LM_HEAD,
    HEAD_PROFILE_SOFTCAP,
    HEAD_PROFILE_SAMPLE,
    HEAD_PROFILE_COUNT,
};

static uint64_t          g_head_profile_ns[HEAD_PROFILE_COUNT];
static uint64_t          g_head_profile_calls[HEAD_PROFILE_COUNT];
static const char *const g_head_profile_names[HEAD_PROFILE_COUNT] = {
        "copy",
        "norm",
        "lm_head",
        "softcap",
        "sample",
};
static struct transformer_forward_profile g_head_profile = {
        .title       = "transformer head",
        .stage_names = g_head_profile_names,
        .stage_count = HEAD_PROFILE_COUNT,
        .ns          = g_head_profile_ns,
        .calls       = g_head_profile_calls,
};

[[nodiscard]] enum geist_status finalize_logits_one_row(struct transformer_arch_state *st,
                                                        size_t                         row_idx,
                                                        geist_token_t                 *out_token) {

    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    enum geist_status                s;
    const bool                       profile = transformer_profile_enabled(&g_head_profile);
    uint64_t                         t0      = profile ? transformer_profile_now_ns() : 0;

    /* Fresh logits this forward — not yet softcapped (peek_logits applies
     * it lazily unless the temp>0 path below does it in place), and dense
     * unless the spec fast path below marks them sparse. */
    st->sess->logits_softcapped = false;
    st->sess->logits_sparse     = false;

    /* Copy chosen row of scratch_h_b into scratch_h_a (reuse as a clean
     * [1, HIDDEN] buffer for the output head). */
    {
        const size_t      bytes = st->d_model * sizeof(float);
        enum geist_status cs    = GEIST_E_UNSUPPORTED;
        if (v->buffer_copy != nullptr) {
            /* device copy keeps batched GPU backends from flushing */
            cs = v->buffer_copy(
                    st->sess->scratch_h_a, 0, st->sess->scratch_h_b, row_idx * bytes, bytes);
        }
        if (cs != GEIST_OK) {
            const uint8_t *src = (const uint8_t *) v->buffer_map(st->sess->scratch_h_b);
            uint8_t       *dst = (uint8_t *) v->buffer_map(st->sess->scratch_h_a);
            memcpy(dst, src + row_idx * bytes, bytes);
            v->buffer_unmap(st->sess->scratch_h_b);
            v->buffer_unmap(st->sess->scratch_h_a);
        }
    }
    transformer_profile_add(&g_head_profile, HEAD_PROFILE_COPY, t0);

    struct geist_tensor t_h_1d       = view_1d(st->sess->scratch_h_a, st->d_model);
    struct geist_tensor t_w_out_norm = view_1d(st->output_norm.buffer, st->d_model);
    t0                               = profile ? transformer_profile_now_ns() : 0;
    s = v->rmsnorm(be, &t_h_1d, &t_w_out_norm, st->config.rms_eps, &t_h_1d);
    transformer_profile_add(&g_head_profile, HEAD_PROFILE_NORM, t0);
    if (s != GEIST_OK) {
        return s;
    }

    /* Speculative i8-sketch fast path (GEIST_SPEC_HEAD=1). Handles the whole
     * projection + greedy argmax when eligible; otherwise falls through to the
     * exact dense lm_head below. Reads the normalized hidden from scratch_h_a. */
    t0 = profile ? transformer_profile_now_ns() : 0;
    if (transformer_spec_head_try(st, out_token)) {
        transformer_profile_add(&g_head_profile, HEAD_PROFILE_LM_HEAD, t0);
        return GEIST_OK;
    }

    struct geist_tensor t_h_2d      = view_2d(st->sess->scratch_h_a, 1, st->d_model);
    struct geist_tensor t_logits_2d = view_2d(st->sess->scratch_logits, 1, st->vocab_size);
    t0                              = profile ? transformer_profile_now_ns() : 0;
    s                               = linear_w_or_legacy(be,
                                                         v,
                                                         st->sess->scratch_h_a,
                                                         st->sess->scratch_logits,
                                                         &st->embed_table_w,
                                                         /* seq = */ 1,
                                                         &t_h_2d,
                                                         &st->embed_table,
                                                         &t_logits_2d);
    transformer_profile_add(&g_head_profile, HEAD_PROFILE_LM_HEAD, t0);
    if (s != GEIST_OK) {
        return s;
    }

    /* Greedy fast path: device argmax reads back a 4-byte index instead
     * of mapping the 1 MB logits row (softcap skipped — tanh monotonic). */
    if (st->sess->temperature == 0.0f && v->argmax_f32 != nullptr) {
        t0          = profile ? transformer_profile_now_ns() : 0;
        int32_t idx = -1;
        if (v->argmax_f32(be, &t_logits_2d, &idx) == GEIST_OK && idx >= 0 &&
            (size_t) idx < (size_t) st->vocab_size) {
            *out_token = (geist_token_t) idx;
            transformer_profile_add(&g_head_profile, HEAD_PROFILE_SAMPLE, t0);
            return GEIST_OK;
        }
    }

    /* Softcap. P1.5: family-conditional. H1: skip in greedy mode — tanh is
     * monotonic so argmax is identical with or without softcap. Saves
     * ~262 144 × tanhf calls per token (~5% of decode on Gemma 4). */
    const bool sampler_needs_softcap = st->sess->temperature > 0.0f;
    if (st->config.logit_softcap > 0.0f && sampler_needs_softcap) {
        t0            = profile ? transformer_profile_now_ns() : 0;
        float      *p = (float *) v->buffer_map(st->sess->scratch_logits);
        const float c = st->config.logit_softcap;
        for (size_t i = 0; i < (size_t) st->vocab_size; i++) {
            p[i] = tanhf(p[i] / c) * c;
        }
        v->buffer_unmap(st->sess->scratch_logits);
        st->sess->logits_softcapped = true;
        transformer_profile_add(&g_head_profile, HEAD_PROFILE_SOFTCAP, t0);
    }

    /* Sampler dispatch. scratch_logits already holds the softcapped
     * row; on CPU backends buffer_map returns the host pointer directly,
     * so the sampler reads it without a copy. P0.1 (2026-05-15): no more
     * per-call 1 MB heap_alloc_aligned. */
    geist_token_t best_id;
    {
        t0                  = profile ? transformer_profile_now_ns() : 0;
        const float *logits = (const float *) v->buffer_map(st->sess->scratch_logits);
        if (logits == nullptr) {
            return GEIST_E_BACKEND;
        }
        if (st->sess->temperature == 0.0f) {
            best_id = geist_sampler_argmax((size_t) st->vocab_size, logits);
        } else if (st->sess->top_k > 1) {
            best_id = geist_sampler_top_k_ws(&st->sess->sampler_ws,
                                             logits,
                                             st->sess->top_k,
                                             st->sess->temperature,
                                             &st->sess->rng);
        } else if (st->sess->top_p > 0.0f && st->sess->top_p < 1.0f) {
            best_id = geist_sampler_top_p_ws(&st->sess->sampler_ws,
                                             logits,
                                             st->sess->top_p,
                                             st->sess->temperature,
                                             &st->sess->rng);
        } else {
            best_id = geist_sampler_temperature(
                    (size_t) st->vocab_size, logits, st->sess->temperature, &st->sess->rng);
        }
        v->buffer_unmap(st->sess->scratch_logits);
        transformer_profile_add(&g_head_profile, HEAD_PROFILE_SAMPLE, t0);
    }
    *out_token = best_id;
    return GEIST_OK;
}

/* Recompute the DENSE lm_head from the normalized hidden still in scratch_h_a
 * (the spec fast path leaves scratch_logits SPARSE: exact only for its top-K
 * candidates, -inf elsewhere — fine for greedy, wrong for value consumers).
 * Called lazily by peek_logits, so routing/SCORE/perplexity see full logits
 * while decode keeps the sketch win. Not softcapped — peek applies that. */
[[nodiscard]] enum geist_status
transformer_head_dense_recompute(struct transformer_arch_state *st) {
    struct geist_backend            *be     = st->backend;
    const struct geist_backend_vtbl *v      = be->desc->vtbl;
    struct geist_tensor              t_h_2d = view_2d(st->sess->scratch_h_a, 1, st->d_model);
    struct geist_tensor t_logits_2d         = view_2d(st->sess->scratch_logits, 1, st->vocab_size);
    enum geist_status   s                   = linear_w_or_legacy(be,
                                                                 v,
                                                                 st->sess->scratch_h_a,
                                                                 st->sess->scratch_logits,
                                                                 &st->embed_table_w,
                                                                 /* seq = */ 1,
                                                                 &t_h_2d,
                                                                 &st->embed_table,
                                                                 &t_logits_2d);
    if (s == GEIST_OK) {
        st->sess->logits_sparse = false;
    }
    return s;
}

/* Batched variant of finalize_logits_one_row for verify_forward: runs
 * the lm_head linear ONCE over all k rows of scratch_h_b, then does
 * per-row softcap + argmax/sampler. Lights up the M>1 NEON IQ-format
 * prefill kernels for the 262K-wide projection — vs k separate M=1
 * SGEMV calls that re-stream the embed_table weight rows. Writes
 * out_tokens[0..k-1] with the per-position sampled token. */
[[nodiscard]] enum geist_status
finalize_logits_batch(struct transformer_arch_state *st, size_t k, geist_token_t *out_tokens) {

    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    enum geist_status                s;

    /* Source: scratch_h_b [k, HIDDEN]. Apply output_norm row-wise
     * (rmsnorm batches naturally over the leading dim). Reuse
     * scratch_h_a as the normed buffer so we don't trash h_b which
     * the caller may still want. */
    {
        const size_t   bytes = k * st->d_model * sizeof(float);
        const uint8_t *src   = (const uint8_t *) v->buffer_map(st->sess->scratch_h_b);
        uint8_t       *dst   = (uint8_t *) v->buffer_map(st->sess->scratch_h_a);
        memcpy(dst, src, bytes);
        v->buffer_unmap(st->sess->scratch_h_b);
        v->buffer_unmap(st->sess->scratch_h_a);
    }
    struct geist_tensor t_h_2d       = view_2d(st->sess->scratch_h_a, (int64_t) k, st->d_model);
    struct geist_tensor t_w_out_norm = view_1d(st->output_norm.buffer, st->d_model);
    s = v->rmsnorm(be, &t_h_2d, &t_w_out_norm, st->config.rms_eps, &t_h_2d);
    if (s != GEIST_OK) {
        return s;
    }

    /* Single batched linear: [k, HIDDEN] @ embed_table^T → [k, VOCAB]. */
    struct geist_tensor t_logits_2d =
            view_2d(st->sess->scratch_logits, (int64_t) k, st->vocab_size);
    s = linear_w_or_legacy(be,
                           v,
                           st->sess->scratch_h_a,
                           st->sess->scratch_logits,
                           &st->embed_table_w,
                           k,
                           &t_h_2d,
                           &st->embed_table,
                           &t_logits_2d);
    if (s != GEIST_OK) {
        return s;
    }

    /* Per-row softcap + sampler. Softcap is monotonic, so for greedy
     * (temperature=0) argmax is identical with or without it — skip
     * the ~262k tanhf calls per row. Stochastic modes still need it
     * to preserve the correct logit distribution. P0.1 (2026-05-15):
     * the row is softcapped in place inside scratch_logits and the
     * sampler reads the same row directly — no per-row 1 MB
     * heap_alloc_aligned, no per-row memcpy. */
    {
        float *all = (float *) v->buffer_map(st->sess->scratch_logits);
        if (all == nullptr) {
            return GEIST_E_BACKEND;
        }
        const float c                     = st->config.logit_softcap;
        const bool  sampler_needs_softcap = st->sess->temperature > 0.0f;
        const bool  do_softcap            = c > 0.0f && sampler_needs_softcap;
        for (size_t row = 0; row < k; row++) {
            float *p = all + row * (size_t) st->vocab_size;
            if (do_softcap) {
                for (size_t i = 0; i < (size_t) st->vocab_size; i++) {
                    p[i] = tanhf(p[i] / c) * c;
                }
            }

            geist_token_t best_id;
            if (st->sess->temperature == 0.0f) {
                best_id = geist_sampler_argmax((size_t) st->vocab_size, p);
            } else if (st->sess->top_k > 1) {
                best_id = geist_sampler_top_k_ws(&st->sess->sampler_ws,
                                                 p,
                                                 st->sess->top_k,
                                                 st->sess->temperature,
                                                 &st->sess->rng);
            } else if (st->sess->top_p > 0.0f && st->sess->top_p < 1.0f) {
                best_id = geist_sampler_top_p_ws(&st->sess->sampler_ws,
                                                 p,
                                                 st->sess->top_p,
                                                 st->sess->temperature,
                                                 &st->sess->rng);
            } else {
                best_id = geist_sampler_temperature(
                        (size_t) st->vocab_size, p, st->sess->temperature, &st->sess->rng);
            }
            out_tokens[row] = best_id;
        }
        v->buffer_unmap(st->sess->scratch_logits);
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status finalize_logits_last_row(struct transformer_arch_state *st,
                                                         size_t                         seq) {
    geist_token_t     tok = -1;
    enum geist_status s   = finalize_logits_one_row(st, seq - 1, &tok);
    if (s != GEIST_OK) {
        return s;
    }
    st->sess->next_token_pending = tok;
    st->sess->logits_valid       = true;
    return GEIST_OK;
}
