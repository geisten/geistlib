/*
 * src/archs/transformer/forward/layer_ple.c - transformer layer PLE and
 * final output scaling.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include <geist_types.h>
#include "profile.h"

#include <stdint.h>
#include <string.h>

/* Per-layer PLE sub-stage profiler (GEIST_PROFILE_PREFILL=1). Splits the
 * single "ple" bucket of the main profile into its constituent ops so the
 * Gemma 4 prefill gap can be attributed precisely. */
enum ple_profile_stage {
    PLE_GATE = 0,
    PLE_GELU,
    PLE_MUL,
    PLE_PROJ,
    PLE_RMSNORM,
    PLE_ADD,
    PLE_PROFILE_COUNT,
};
static uint64_t          g_ple_ns[PLE_PROFILE_COUNT];
static uint64_t          g_ple_calls[PLE_PROFILE_COUNT];
static const char *const g_ple_names[PLE_PROFILE_COUNT] = {
        "gate_lin",
        "gelu",
        "mul",
        "proj_lin",
        "rmsnorm",
        "add",
};
static struct transformer_forward_profile g_ple_profile = {
        .title       = "transformer ple (per-layer)",
        .stage_names = g_ple_names,
        .stage_count = PLE_PROFILE_COUNT,
        .ns          = g_ple_ns,
        .calls       = g_ple_calls,
};

enum geist_status transformer_layer_run_ple_or_copy(struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_state    *st = ctx->st;
    struct transformer_layer_weights *L  = ctx->L;
    struct geist_backend             *be = ctx->be;
    const struct geist_backend_vtbl  *v  = ctx->v;
    enum geist_status                 s;

    struct geist_tensor t_h_post_ff_2d =
            view_2d(st->sess->scratch_h_post_ff, ctx->SEQ, st->d_model);
    struct geist_tensor t_h_out_2d = view_2d(ctx->h_out_buf, ctx->SEQ, st->d_model);
    if (ctx->apply_ple && ctx->per_layer_input_buf != nullptr) {
        const bool          prof = transformer_profile_enabled(&g_ple_profile);
        uint64_t            t0;
        struct geist_tensor t_gate_ple_2d =
                view_2d(st->sess->scratch_gate_ple, ctx->SEQ, st->hidden_per_layer);
        /* When the full per-token slab [seq, n_layers*hpl] is passed
         * through (batched GPU backends skip the per-layer gather), read
         * this layer's slice as a strided view; a pre-gathered buffer is
         * contiguous. */
        const bool ple_slab =
                ctx->per_layer_input_buf == st->sess->scratch_per_layer_input;
        struct geist_tensor t_ple_in_2d = view_2d_at(
                ctx->per_layer_input_buf,
                ple_slab ? (size_t) ctx->layer_idx *
                                   (size_t) st->hidden_per_layer * sizeof(float)
                         : 0u,
                ctx->SEQ, st->hidden_per_layer);
        if (ple_slab) {
            t_ple_in_2d.stride[0] = (int64_t) st->ple_out;
        }

        /* Fused PLE block (GPU decode fast path): gate GEMV + gelu*ple and
         * proj GEMV + rmsnorm + residual add in two dispatches. Anything
         * the backend doesn't support falls through to the decomposed
         * ops below. */
        if (v->ple_block != nullptr) {
            struct geist_tensor t_w_post_per_1d =
                    view_1d(L->post_per_layer_norm.buffer, st->d_model);
            struct geist_tensor t_proj_scratch_2d =
                    view_2d(st->sess->scratch_proj_ple, ctx->SEQ, st->d_model);
            t0 = prof ? transformer_profile_now_ns() : 0;
            s  = v->ple_block(be,
                              &t_h_post_ff_2d,
                              &L->per_layer_gate,
                              &t_ple_in_2d,
                              &L->per_layer_proj,
                              &t_h_post_ff_2d,
                              &t_w_post_per_1d,
                              ctx->eps,
                              &t_gate_ple_2d,
                              &t_proj_scratch_2d,
                              &t_h_out_2d);
            transformer_profile_add(&g_ple_profile, PLE_GATE, t0);
            if (s == GEIST_OK) {
                return GEIST_OK;
            }
        }

        t0 = prof ? transformer_profile_now_ns() : 0;
        s  = linear_w_or_legacy(be,
                                v,
                                st->sess->scratch_h_post_ff,
                                st->sess->scratch_gate_ple,
                                &L->per_layer_gate_w,
                                ctx->seq,
                                &t_h_post_ff_2d,
                                &L->per_layer_gate,
                                &t_gate_ple_2d);
        transformer_profile_add(&g_ple_profile, PLE_GATE, t0);
        if (s != GEIST_OK) {
            return s;
        }
        t0 = prof ? transformer_profile_now_ns() : 0;
        if (v->gelu_tanh_mul != nullptr &&
            v->gelu_tanh_mul(be, &t_gate_ple_2d, &t_ple_in_2d,
                             &t_gate_ple_2d) == GEIST_OK) {
            transformer_profile_add(&g_ple_profile, PLE_GELU, t0);
        } else {
            s = v->gelu_tanh(be, &t_gate_ple_2d, &t_gate_ple_2d);
            transformer_profile_add(&g_ple_profile, PLE_GELU, t0);
            if (s != GEIST_OK) {
                return s;
            }
            t0 = prof ? transformer_profile_now_ns() : 0;
            s  = v->mul(be, &t_gate_ple_2d, &t_ple_in_2d, &t_gate_ple_2d);
            transformer_profile_add(&g_ple_profile, PLE_MUL, t0);
            if (s != GEIST_OK) {
                return s;
            }
        }

        struct geist_tensor t_proj_ple_2d =
                view_2d(st->sess->scratch_proj_ple, ctx->SEQ, st->d_model);
        struct geist_tensor t_w_post_per = view_1d(L->post_per_layer_norm.buffer, st->d_model);
        t0                               = prof ? transformer_profile_now_ns() : 0;
        s                                = linear_w_or_legacy(be,
                                                              v,
                                                              st->sess->scratch_gate_ple,
                                                              st->sess->scratch_proj_ple,
                                                              &L->per_layer_proj_w,
                                                              ctx->seq,
                                                              &t_gate_ple_2d,
                                                              &L->per_layer_proj,
                                                              &t_proj_ple_2d);
        transformer_profile_add(&g_ple_profile, PLE_PROJ, t0);
        if (s != GEIST_OK) {
            return s;
        }
        t0 = prof ? transformer_profile_now_ns() : 0;
        if (v->rmsnorm_add != nullptr &&
            v->rmsnorm_add(be, &t_h_post_ff_2d, &t_proj_ple_2d,
                           &t_w_post_per, ctx->eps, &t_h_out_2d) ==
                GEIST_OK) {
            transformer_profile_add(&g_ple_profile, PLE_RMSNORM, t0);
        } else {
            s = v->rmsnorm(be, &t_proj_ple_2d, &t_w_post_per, ctx->eps, &t_proj_ple_2d);
            transformer_profile_add(&g_ple_profile, PLE_RMSNORM, t0);
            if (s != GEIST_OK) {
                return s;
            }
            t0 = prof ? transformer_profile_now_ns() : 0;
            s  = v->add(be, &t_h_post_ff_2d, &t_proj_ple_2d, &t_h_out_2d);
            transformer_profile_add(&g_ple_profile, PLE_ADD, t0);
            if (s != GEIST_OK) {
                return s;
            }
        }
    } else {
        const size_t bytes = ctx->seq * st->d_model * sizeof(float);
        uint8_t     *src   = (uint8_t *) v->buffer_map(st->sess->scratch_h_post_ff);
        uint8_t     *dst   = (uint8_t *) v->buffer_map(ctx->h_out_buf);
        memcpy(dst, src, bytes);
        v->buffer_unmap(st->sess->scratch_h_post_ff);
        v->buffer_unmap(ctx->h_out_buf);
    }
    return GEIST_OK;
}

void transformer_layer_scale_output(struct transformer_layer_forward_ctx *ctx) {
    /* Device path first: batched GPU backends keep the per-layer scale
     * on-device instead of flushing their pipeline for a host loop. */
    if (ctx->v->scale_f32 != nullptr) {
        struct geist_tensor t_h =
                view_2d(ctx->h_out_buf, ctx->SEQ, ctx->st->d_model);
        if (ctx->v->scale_f32(ctx->be, &t_h, ctx->L->layer_scalar, &t_h) ==
            GEIST_OK) {
            return;
        }
    }
    float       *hout    = (float *) ctx->v->buffer_map(ctx->h_out_buf);
    const size_t n_total = ctx->seq * ctx->st->d_model;
    for (size_t i = 0; i < n_total; i++) {
        hout[i] *= ctx->L->layer_scalar;
    }
    ctx->v->buffer_unmap(ctx->h_out_buf);
}
