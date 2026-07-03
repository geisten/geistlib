/*
 * src/archs/transformer/forward/layer_ffn.c - transformer layer FFN stage.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include <geist_types.h>
#include "profile.h"

#include <stdlib.h>

enum ffn_profile_stage {
    FFN_PROFILE_NORM = 0,
    FFN_PROFILE_GATE_UP,
    FFN_PROFILE_ACT,
    FFN_PROFILE_MUL,
    FFN_PROFILE_SUB_NORM,
    FFN_PROFILE_DOWN_SCALE,
    FFN_PROFILE_DOWN,
    FFN_PROFILE_POST,
    FFN_PROFILE_COUNT,
};

static uint64_t          g_ffn_profile_ns[FFN_PROFILE_COUNT];
static uint64_t          g_ffn_profile_calls[FFN_PROFILE_COUNT];
static const char *const g_ffn_profile_names[FFN_PROFILE_COUNT] = {
        "norm",
        "gate_up",
        "act",
        "mul",
        "sub_norm",
        "down_scale",
        "down",
        "post",
};
static struct transformer_forward_profile g_ffn_profile = {
        .title       = "transformer ffn",
        .stage_names = g_ffn_profile_names,
        .stage_count = FFN_PROFILE_COUNT,
        .ns          = g_ffn_profile_ns,
        .calls       = g_ffn_profile_calls,
};

static bool ffn_fused_scale_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("GEIST_FFN_FUSED_SCALE");
        enabled         = (env == nullptr || env[0] != '0') ? 1 : 0;
    }
    return enabled != 0;
}

static bool ffn_tile_fusion_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("GEIST_FFN_TILE_FUSION");
        enabled         = (env != nullptr && env[0] == '1') ? 1 : 0;
    }
    return enabled != 0;
}

enum geist_status transformer_layer_run_ffn_block(struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_state    *st = ctx->st;
    struct transformer_layer_weights *L  = ctx->L;
    struct geist_backend             *be = ctx->be;
    const struct geist_backend_vtbl  *v  = ctx->v;
    enum geist_status                 s;

    struct geist_tensor t_h_post_attn_2d =
            view_2d(st->sess->scratch_h_post_attn, ctx->SEQ, st->d_model);
    struct geist_tensor t_pre_ff_2d  = view_2d(st->sess->scratch_pre_ff, ctx->SEQ, st->d_model);
    struct geist_tensor t_w_ffn_norm = view_1d(L->ffn_norm.buffer, st->d_model);
    const bool          profile      = transformer_profile_enabled(&g_ffn_profile);
    uint64_t            t0           = profile ? transformer_profile_now_ns() : 0;
    s = v->rmsnorm(be, &t_h_post_attn_2d, &t_w_ffn_norm, ctx->eps, &t_pre_ff_2d);
    transformer_profile_add(&g_ffn_profile, FFN_PROFILE_NORM, t0);
    if (s != GEIST_OK) {
        return s;
    }

    const bool          has_ffn_sub_norm = ctx->apply_sub_ln && L->ffn_sub_norm.buffer != nullptr;
    struct geist_tensor t_ffn_out_2d = view_2d(st->sess->scratch_ffn_out, ctx->SEQ, st->d_model);
    if (ctx->seq > 1 && ctx->ffn_activation == GEIST_FFN_GEGLU && !has_ffn_sub_norm &&
        !st->runtime_flags.dump_act_sparsity && v->ffn_geglu_q4q6_mN != nullptr &&
        ffn_tile_fusion_enabled()) {
        const float *xp = (const float *) v->buffer_map(st->sess->scratch_pre_ff);
        float       *yp = (float *) v->buffer_map(st->sess->scratch_ffn_out);
        if (xp == nullptr || yp == nullptr) {
            if (xp != nullptr)
                v->buffer_unmap(st->sess->scratch_pre_ff);
            if (yp != nullptr)
                v->buffer_unmap(st->sess->scratch_ffn_out);
            return GEIST_E_BACKEND;
        }
        t0 = profile ? transformer_profile_now_ns() : 0;
        s  = v->ffn_geglu_q4q6_mN(be,
                                  xp,
                                  ctx->seq,
                                  (size_t) st->d_model,
                                  ctx->inter,
                                  &L->gate_proj_w,
                                  &L->up_proj_w,
                                  &L->down_proj_w,
                                  L->down_awq_inv_scale,
                                  yp);
        v->buffer_unmap(st->sess->scratch_pre_ff);
        v->buffer_unmap(st->sess->scratch_ffn_out);
        if (s == GEIST_OK) {
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_GATE_UP, t0);
            goto ffn_post;
        }
        if (s != GEIST_E_UNSUPPORTED) {
            return s;
        }
    }

    struct geist_tensor  t_up_2d = view_2d(st->sess->scratch_up, ctx->SEQ, (int64_t) ctx->inter);
    struct geist_buffer *mid_buf;
    struct geist_tensor  t_mid_2d;
    bool                 mid_already_down_scaled = false;

    if (ctx->ffn_activation == GEIST_FFN_SQUARED_RELU) {
        t0 = profile ? transformer_profile_now_ns() : 0;
        s  = linear_w_or_legacy(be,
                                v,
                                st->sess->scratch_pre_ff,
                                st->sess->scratch_up,
                                &L->up_proj_w,
                                ctx->seq,
                                &t_pre_ff_2d,
                                &L->up_proj,
                                &t_up_2d);
        transformer_profile_add(&g_ffn_profile, FFN_PROFILE_GATE_UP, t0);
        if (s != GEIST_OK) {
            return s;
        }
        t0 = profile ? transformer_profile_now_ns() : 0;
        s  = v->relu_squared(be, &t_up_2d, &t_up_2d);
        transformer_profile_add(&g_ffn_profile, FFN_PROFILE_ACT, t0);
        if (s != GEIST_OK) {
            return s;
        }
        mid_buf  = st->sess->scratch_up;
        t_mid_2d = t_up_2d;
    } else {
        struct geist_tensor t_gate_2d =
                view_2d(st->sess->scratch_gate, ctx->SEQ, (int64_t) ctx->inter);
        t0 = profile ? transformer_profile_now_ns() : 0;
        s  = linear_w_pair_or_legacy(be,
                                     v,
                                     st->sess->scratch_pre_ff,
                                     st->sess->scratch_gate,
                                     st->sess->scratch_up,
                                     &L->gate_proj_w,
                                     &L->up_proj_w,
                                     ctx->seq,
                                     &t_pre_ff_2d,
                                     &L->gate_proj,
                                     &L->up_proj,
                                     &t_gate_2d,
                                     &t_up_2d);
        transformer_profile_add(&g_ffn_profile, FFN_PROFILE_GATE_UP, t0);
        if (s != GEIST_OK) {
            return s;
        }

        if (ctx->ffn_activation == GEIST_FFN_GATED_SQUARED_RELU) {
            t0 = profile ? transformer_profile_now_ns() : 0;
            s  = v->relu_squared(be, &t_gate_2d, &t_gate_2d);
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_ACT, t0);
            if (s != GEIST_OK) {
                return s;
            }
            t0 = profile ? transformer_profile_now_ns() : 0;
            s  = v->mul(be, &t_gate_2d, &t_up_2d, &t_gate_2d);
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_MUL, t0);
        } else if (ctx->ffn_activation == GEIST_FFN_SWIGLU) {
            t0 = profile ? transformer_profile_now_ns() : 0;
            if (v->silu != nullptr) {
                s = v->silu(be, &t_gate_2d, &t_gate_2d);
            } else {
                s = v->gelu_tanh(be, &t_gate_2d, &t_gate_2d);
            }
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_ACT, t0);
            if (s != GEIST_OK) {
                return s;
            }
            t0 = profile ? transformer_profile_now_ns() : 0;
            s  = v->mul(be, &t_gate_2d, &t_up_2d, &t_gate_2d);
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_MUL, t0);
        } else if (v->gelu_tanh_mul_scaled != nullptr && L->down_awq_inv_scale != nullptr &&
                   !has_ffn_sub_norm && ffn_fused_scale_enabled()) {
            t0 = profile ? transformer_profile_now_ns() : 0;
            s  = v->gelu_tanh_mul_scaled(
                    be, &t_gate_2d, &t_up_2d, L->down_awq_inv_scale, &t_gate_2d);
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_ACT, t0);
            mid_already_down_scaled = true;
        } else if (v->gelu_tanh_mul != nullptr) {
            t0 = profile ? transformer_profile_now_ns() : 0;
            s  = v->gelu_tanh_mul(be, &t_gate_2d, &t_up_2d, &t_gate_2d);
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_ACT, t0);
        } else {
            t0 = profile ? transformer_profile_now_ns() : 0;
            s  = v->gelu_tanh(be, &t_gate_2d, &t_gate_2d);
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_ACT, t0);
            if (s != GEIST_OK) {
                return s;
            }
            t0 = profile ? transformer_profile_now_ns() : 0;
            s  = v->mul(be, &t_gate_2d, &t_up_2d, &t_gate_2d);
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_MUL, t0);
        }
        if (s != GEIST_OK) {
            return s;
        }
        mid_buf  = st->sess->scratch_gate;
        t_mid_2d = t_gate_2d;
    }

    if (has_ffn_sub_norm) {
        struct geist_tensor t_ffn_sub_w = view_1d(L->ffn_sub_norm.buffer, (int64_t) ctx->inter);
        t0                              = profile ? transformer_profile_now_ns() : 0;
        s = v->rmsnorm(be, &t_mid_2d, &t_ffn_sub_w, ctx->eps, &t_mid_2d);
        transformer_profile_add(&g_ffn_profile, FFN_PROFILE_SUB_NORM, t0);
        if (s != GEIST_OK) {
            return s;
        }
    }

    if (st->runtime_flags.dump_act_sparsity) {
        if (!mid_already_down_scaled) {
            t0 = profile ? transformer_profile_now_ns() : 0;
            apply_per_channel_inv_scale_inplace(
                    v, mid_buf, ctx->seq, ctx->inter, L->down_awq_inv_scale);
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_DOWN_SCALE, t0);
        }
        transformer_probe_ffn_sparsity(v, true, ctx->layer_idx, mid_buf, ctx->seq * ctx->inter);
        t0 = profile ? transformer_profile_now_ns() : 0;
        s  = linear_w_or_legacy(be,
                                v,
                                mid_buf,
                                st->sess->scratch_ffn_out,
                                &L->down_proj_w,
                                ctx->seq,
                                &t_mid_2d,
                                &L->down_proj,
                                &t_ffn_out_2d);
        transformer_profile_add(&g_ffn_profile, FFN_PROFILE_DOWN, t0);
    } else {
        if (profile) {
            if (!mid_already_down_scaled) {
                t0 = transformer_profile_now_ns();
                apply_per_channel_inv_scale_inplace(
                        v, mid_buf, ctx->seq, ctx->inter, L->down_awq_inv_scale);
                transformer_profile_add(&g_ffn_profile, FFN_PROFILE_DOWN_SCALE, t0);
            }
            t0 = transformer_profile_now_ns();
            s  = linear_w_or_legacy(be,
                                    v,
                                    mid_buf,
                                    st->sess->scratch_ffn_out,
                                    &L->down_proj_w,
                                    ctx->seq,
                                    &t_mid_2d,
                                    &L->down_proj,
                                    &t_ffn_out_2d);
            transformer_profile_add(&g_ffn_profile, FFN_PROFILE_DOWN, t0);
        } else if (mid_already_down_scaled) {
            s = linear_w_or_legacy(be,
                                   v,
                                   mid_buf,
                                   st->sess->scratch_ffn_out,
                                   &L->down_proj_w,
                                   ctx->seq,
                                   &t_mid_2d,
                                   &L->down_proj,
                                   &t_ffn_out_2d);
        } else {
            s = linear_w_scaled_input_or_legacy(be,
                                                v,
                                                mid_buf,
                                                st->sess->scratch_ffn_out,
                                                &L->down_proj_w,
                                                ctx->seq,
                                                ctx->inter,
                                                L->down_awq_inv_scale,
                                                &t_mid_2d,
                                                &L->down_proj,
                                                &t_ffn_out_2d);
        }
    }
    if (s != GEIST_OK) {
        return s;
    }

ffn_post:
    struct geist_tensor t_h_post_ff_2d =
            view_2d(st->sess->scratch_h_post_ff, ctx->SEQ, st->d_model);
    t0 = profile ? transformer_profile_now_ns() : 0;
    if (ctx->apply_gemma_attn_norms) {
        struct geist_tensor t_post_ff_2d =
                view_2d(st->sess->scratch_post_ff, ctx->SEQ, st->d_model);
        struct geist_tensor t_w_post_ffw = view_1d(L->post_ffw_norm.buffer, st->d_model);
        if (v->rmsnorm_add == nullptr ||
            v->rmsnorm_add(be, &t_h_post_attn_2d, &t_ffn_out_2d,
                           &t_w_post_ffw, ctx->eps, &t_h_post_ff_2d) !=
                GEIST_OK) {
            s = v->rmsnorm(be, &t_ffn_out_2d, &t_w_post_ffw, ctx->eps, &t_post_ff_2d);
            if (s != GEIST_OK) {
                return s;
            }
            s = v->add(be, &t_h_post_attn_2d, &t_post_ff_2d, &t_h_post_ff_2d);
            if (s != GEIST_OK) {
                return s;
            }
        }
    } else {
        s = v->add(be, &t_h_post_attn_2d, &t_ffn_out_2d, &t_h_post_ff_2d);
        if (s != GEIST_OK) {
            return s;
        }
    }
    transformer_profile_add(&g_ffn_profile, FFN_PROFILE_POST, t0);
    return GEIST_OK;
}
