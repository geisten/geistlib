/*
 * src/archs/transformer/forward/layer_attn.c - transformer layer
 * attention stage.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include <geist_types.h>
#include "profile.h"

#include <math.h>
#include <string.h>

enum attn_profile_stage {
    ATTN_PROFILE_NORM = 0,
    ATTN_PROFILE_QKV,
    ATTN_PROFILE_Q_PREP,
    ATTN_PROFILE_KV_PREP,
    ATTN_PROFILE_KV_STORE,
    ATTN_PROFILE_CORE,
    ATTN_PROFILE_POST_CORE,
    ATTN_PROFILE_O_PROJ,
    ATTN_PROFILE_POST,
    ATTN_PROFILE_COUNT,
};

static uint64_t          g_attn_profile_ns[ATTN_PROFILE_COUNT];
static uint64_t          g_attn_profile_calls[ATTN_PROFILE_COUNT];
static const char *const g_attn_profile_names[ATTN_PROFILE_COUNT] = {
        "norm",
        "qkv",
        "q_prep",
        "kv_prep",
        "kv_store",
        "core",
        "post_core",
        "o_proj",
        "post",
};
static struct transformer_forward_profile g_attn_profile = {
        .title       = "transformer attention",
        .stage_names = g_attn_profile_names,
        .stage_count = ATTN_PROFILE_COUNT,
        .ns          = g_attn_profile_ns,
        .calls       = g_attn_profile_calls,
};

/* Upper bound on head_dim for the stack scratch in
 * permute_interleaved_rope_inplace. Every transformer family loaded so
 * far uses head_dim ≤ 256 (Gemma 4 = 256); we leave generous head-room
 * but enforce the bound explicitly because the helper would otherwise
 * silently smash the caller's stack on a future arch with larger heads.
 * If a real arch ever needs more, lift to a caller-provided workspace —
 * AGENT.md hot-path rule disallows runtime heap allocations here. */
enum { PERMUTE_ROPE_MAX_HEAD_DIM = 1024 };

static enum geist_status permute_interleaved_rope_inplace(const struct geist_backend_vtbl *v,
                                                          struct geist_buffer             *buf,
                                                          size_t                           seq,
                                                          size_t                           n_heads,
                                                          size_t head_dim) {

    if (head_dim > PERMUTE_ROPE_MAX_HEAD_DIM) {
        return GEIST_E_INVALID_ARG;
    }
    float *x = (float *) v->buffer_map(buf);
    if (x == nullptr) {
        return GEIST_E_BACKEND;
    }
    const size_t half = head_dim / 2;
    for (size_t t = 0; t < seq; t++) {
        for (size_t h = 0; h < n_heads; h++) {
            float *xh = x + (t * n_heads + h) * head_dim;
            float  tmp[PERMUTE_ROPE_MAX_HEAD_DIM];
            for (size_t i = 0; i < half; i++) {
                tmp[i]        = xh[2 * i];
                tmp[i + half] = xh[2 * i + 1];
            }
            memcpy(xh, tmp, head_dim * sizeof(float));
        }
    }
    v->buffer_unmap(buf);
    return GEIST_OK;
}

enum geist_status transformer_layer_run_attention_block(struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_state    *st = ctx->st;
    struct transformer_layer_weights *L  = ctx->L;
    struct geist_backend             *be = ctx->be;
    const struct geist_backend_vtbl  *v  = ctx->v;
    enum geist_status                 s;
    const bool                        profile = transformer_profile_enabled(&g_attn_profile);
    uint64_t                          t0      = profile ? transformer_profile_now_ns() : 0;

    struct geist_tensor t_h_in_2d     = view_2d(ctx->h_in_buf, ctx->SEQ, st->d_model);
    struct geist_tensor t_normed_2d   = view_2d(st->sess->scratch_normed, ctx->SEQ, st->d_model);
    struct geist_tensor t_w_attn_norm = view_1d(L->attn_norm.buffer, st->d_model);

    s = v->rmsnorm(be, &t_h_in_2d, &t_w_attn_norm, ctx->eps, &t_normed_2d);
    transformer_profile_add(&g_attn_profile, ATTN_PROFILE_NORM, t0);
    if (s != GEIST_OK) {
        return s;
    }

    if (ctx->apply_bitnet_input_quant) {
        t0 = profile ? transformer_profile_now_ns() : 0;
        apply_bitnet_input_quant_inplace(v, st->sess->scratch_normed, ctx->seq, st->d_model);
        transformer_profile_add(&g_attn_profile, ATTN_PROFILE_NORM, t0);
    }

    struct geist_tensor t_q_2d = view_2d(st->sess->scratch_q, ctx->SEQ, (int64_t) ctx->q_out);
    t0                         = profile ? transformer_profile_now_ns() : 0;
    if (ctx->compute_kv) {
        struct geist_tensor t_k_2d = view_2d(st->sess->scratch_k, ctx->SEQ, (int64_t) ctx->kv_out);
        struct geist_tensor t_v_2d = view_2d(st->sess->scratch_v, ctx->SEQ, (int64_t) ctx->kv_out);
        s                          = linear_w_triple_or_legacy(be,
                                                               v,
                                                               st->sess->scratch_normed,
                                                               st->sess->scratch_q,
                                                               st->sess->scratch_k,
                                                               st->sess->scratch_v,
                                                               &L->q_proj_w,
                                                               &L->k_proj_w,
                                                               &L->v_proj_w,
                                                               ctx->seq,
                                                               &t_normed_2d,
                                                               &L->q_proj,
                                                               &L->k_proj,
                                                               &L->v_proj,
                                                               &t_q_2d,
                                                               &t_k_2d,
                                                               &t_v_2d);
        transformer_profile_add(&g_attn_profile, ATTN_PROFILE_QKV, t0);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        s = linear_w_or_legacy(be,
                               v,
                               st->sess->scratch_normed,
                               st->sess->scratch_q,
                               &L->q_proj_w,
                               ctx->seq,
                               &t_normed_2d,
                               &L->q_proj,
                               &t_q_2d);
        transformer_profile_add(&g_attn_profile, ATTN_PROFILE_QKV, t0);
        if (s != GEIST_OK) {
            return s;
        }
    }

    t0 = profile ? transformer_profile_now_ns() : 0;
    struct geist_buffer *cos_buf       = L->is_full ? st->rope_cos_full : st->rope_cos_sliding;
    struct geist_buffer *sin_buf       = L->is_full ? st->rope_sin_full : st->rope_sin_sliding;
    const size_t         cos_row_bytes = ctx->hd * sizeof(float);
    struct geist_tensor  t_cos =
            view_2d_at(cos_buf, ctx->q_position * cos_row_bytes, ctx->SEQ, (int64_t) ctx->hd);
    struct geist_tensor t_sin =
            view_2d_at(sin_buf, ctx->q_position * cos_row_bytes, ctx->SEQ, (int64_t) ctx->hd);
    struct geist_tensor t_q_3d =
            view_3d(st->sess->scratch_q, ctx->SEQ, st->n_q_heads, (int64_t) ctx->hd);

    /* Fused q/k/v prep (GPU): per-head norms + RoPE + KV-cache append in
     * two dispatches. Covers the gemma half-split-RoPE path on the plain
     * (f32/f16) cache; anything else falls back to the decomposed ops. */
    bool fused_qkv_prep = false;
    if (ctx->apply_gemma_attn_norms && !ctx->rope_interleaved &&
        v->attn_qkv_prep != nullptr &&
        !ctx->kv_kivi_enabled && !ctx->kv_int8_enabled) {
        struct geist_tensor t_q_norm_w = view_1d(L->q_norm.buffer, (int64_t) ctx->hd);
        if (ctx->compute_kv) {
            struct geist_tensor t_k_3d = view_3d(
                    st->sess->scratch_k, ctx->SEQ, st->n_kv_heads, (int64_t) ctx->hd);
            struct geist_tensor t_v_3d = view_3d(
                    st->sess->scratch_v, ctx->SEQ, st->n_kv_heads, (int64_t) ctx->hd);
            struct geist_tensor t_k_norm_w = view_1d(L->k_norm.buffer, (int64_t) ctx->hd);
            struct geist_tensor t_ones_hd =
                    view_1d(st->sess->scratch_ones_headdim_max, (int64_t) ctx->hd);
            const int64_t cache_rows = (int64_t) (ctx->q_position + ctx->seq);
            struct geist_tensor t_kc =
                    ctx->kv_f16_enabled
                            ? view_3d_f16(ctx->k_cache_buf, cache_rows,
                                          st->n_kv_heads, (int64_t) ctx->hd)
                            : view_3d(ctx->k_cache_buf, cache_rows,
                                      st->n_kv_heads, (int64_t) ctx->hd);
            struct geist_tensor t_vc =
                    ctx->kv_f16_enabled
                            ? view_3d_f16(ctx->v_cache_buf, cache_rows,
                                          st->n_kv_heads, (int64_t) ctx->hd)
                            : view_3d(ctx->v_cache_buf, cache_rows,
                                      st->n_kv_heads, (int64_t) ctx->hd);
            s = v->attn_qkv_prep(be, &t_q_3d, &t_k_3d, &t_v_3d,
                                 &t_q_norm_w, &t_k_norm_w, &t_ones_hd,
                                 &t_cos, &t_sin, ctx->eps, ctx->q_position,
                                 &t_kc, &t_vc);
        } else {
            s = v->attn_qkv_prep(be, &t_q_3d, nullptr, nullptr,
                                 &t_q_norm_w, nullptr, nullptr,
                                 &t_cos, &t_sin, ctx->eps, ctx->q_position,
                                 nullptr, nullptr);
        }
        fused_qkv_prep = (s == GEIST_OK);
    }

    if (!fused_qkv_prep) {
        if (ctx->apply_gemma_attn_norms) {
            struct geist_tensor t_q_perhead =
                    view_2d(st->sess->scratch_q, ctx->SEQ * st->n_q_heads, (int64_t) ctx->hd);
            struct geist_tensor t_q_norm_w = view_1d(L->q_norm.buffer, (int64_t) ctx->hd);
            s = v->rmsnorm(be, &t_q_perhead, &t_q_norm_w, ctx->eps, &t_q_perhead);
            if (s != GEIST_OK) {
                return s;
            }
        }
        if (ctx->rope_interleaved) {
            s = permute_interleaved_rope_inplace(
                    v, st->sess->scratch_q, ctx->seq, st->n_q_heads, ctx->hd);
            if (s != GEIST_OK) {
                return s;
            }
        }
        s = v->rope_apply(be, &t_q_3d, &t_cos, &t_sin);
        if (s != GEIST_OK) {
            return s;
        }
    }

    if (!ctx->apply_gemma_attn_norms) {
        const float scale = 1.0f / sqrtf((float) ctx->hd);
        float      *qp    = (float *) v->buffer_map(st->sess->scratch_q);
        for (size_t i = 0; i < ctx->seq * ctx->q_out; i++) {
            qp[i] *= scale;
        }
        v->buffer_unmap(st->sess->scratch_q);
    }
    transformer_profile_add(&g_attn_profile, ATTN_PROFILE_Q_PREP, t0);

    if (ctx->compute_kv && !fused_qkv_prep) {
        t0 = profile ? transformer_profile_now_ns() : 0;
        struct geist_tensor t_k_3d =
                view_3d(st->sess->scratch_k, ctx->SEQ, st->n_kv_heads, (int64_t) ctx->hd);
        if (ctx->apply_gemma_attn_norms) {
            struct geist_tensor t_k_perhead =
                    view_2d(st->sess->scratch_k, ctx->SEQ * st->n_kv_heads, (int64_t) ctx->hd);
            struct geist_tensor t_v_perhead =
                    view_2d(st->sess->scratch_v, ctx->SEQ * st->n_kv_heads, (int64_t) ctx->hd);
            struct geist_tensor t_k_norm_w = view_1d(L->k_norm.buffer, (int64_t) ctx->hd);
            struct geist_tensor t_ones_hd =
                    view_1d(st->sess->scratch_ones_headdim_max, (int64_t) ctx->hd);
            s = v->rmsnorm(be, &t_k_perhead, &t_k_norm_w, ctx->eps, &t_k_perhead);
            if (s != GEIST_OK) {
                return s;
            }
            s = v->rmsnorm(be, &t_v_perhead, &t_ones_hd, ctx->eps, &t_v_perhead);
            if (s != GEIST_OK) {
                return s;
            }
        }
        if (ctx->rope_interleaved) {
            s = permute_interleaved_rope_inplace(
                    v, st->sess->scratch_k, ctx->seq, st->n_kv_heads, ctx->hd);
            if (s != GEIST_OK) {
                return s;
            }
        }
        s = v->rope_apply(be, &t_k_3d, &t_cos, &t_sin);
        if (s != GEIST_OK) {
            return s;
        }
        transformer_profile_add(&g_attn_profile, ATTN_PROFILE_KV_PREP, t0);

        t0 = profile ? transformer_profile_now_ns() : 0;
        s  = transformer_kv_store_append(ctx);
        transformer_profile_add(&g_attn_profile, ATTN_PROFILE_KV_STORE, t0);
        if (s != GEIST_OK) {
            return s;
        }
    }
    ctx->kv_len_now = ctx->q_position + ctx->seq;

    struct geist_tensor t_attn_3d =
            view_3d(st->sess->scratch_attn, ctx->SEQ, st->n_q_heads, (int64_t) ctx->hd);
    t0 = profile ? transformer_profile_now_ns() : 0;
    s  = transformer_kv_store_attention(ctx, &t_q_3d, &t_attn_3d);
    transformer_profile_add(&g_attn_profile, ATTN_PROFILE_CORE, t0);
    if (s != GEIST_OK) {
        return s;
    }

    t0 = profile ? transformer_profile_now_ns() : 0;
    apply_per_channel_inv_scale_inplace(
            v, st->sess->scratch_attn, ctx->seq, ctx->q_out, L->o_awq_inv_scale);

    struct geist_tensor t_attn_2d = view_2d(st->sess->scratch_attn, ctx->SEQ, (int64_t) ctx->q_out);
    if (ctx->apply_sub_ln && L->attn_sub_norm.buffer != nullptr) {
        struct geist_tensor t_attn_sub_w = view_1d(L->attn_sub_norm.buffer, (int64_t) ctx->q_out);
        s = v->rmsnorm(be, &t_attn_2d, &t_attn_sub_w, ctx->eps, &t_attn_2d);
        if (s != GEIST_OK) {
            return s;
        }
    }
    transformer_profile_add(&g_attn_profile, ATTN_PROFILE_POST_CORE, t0);

    struct geist_tensor t_o_2d = view_2d(st->sess->scratch_o, ctx->SEQ, st->d_model);
    t0                         = profile ? transformer_profile_now_ns() : 0;
    s                          = linear_w_or_legacy(be,
                                                    v,
                                                    st->sess->scratch_attn,
                                                    st->sess->scratch_o,
                                                    &L->o_proj_w,
                                                    ctx->seq,
                                                    &t_attn_2d,
                                                    &L->o_proj,
                                                    &t_o_2d);
    transformer_profile_add(&g_attn_profile, ATTN_PROFILE_O_PROJ, t0);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_tensor t_h_post_attn_2d =
            view_2d(st->sess->scratch_h_post_attn, ctx->SEQ, st->d_model);
    t0 = profile ? transformer_profile_now_ns() : 0;
    if (ctx->apply_gemma_attn_norms) {
        struct geist_tensor t_post_attn_2d =
                view_2d(st->sess->scratch_post_attn, ctx->SEQ, st->d_model);
        struct geist_tensor t_w_post_attn = view_1d(L->post_attn_norm.buffer, st->d_model);
        if (v->rmsnorm_add == nullptr ||
            v->rmsnorm_add(be, &t_h_in_2d, &t_o_2d, &t_w_post_attn,
                           ctx->eps, &t_h_post_attn_2d) != GEIST_OK) {
            s = v->rmsnorm(be, &t_o_2d, &t_w_post_attn, ctx->eps, &t_post_attn_2d);
            if (s != GEIST_OK) {
                return s;
            }
            s = v->add(be, &t_h_in_2d, &t_post_attn_2d, &t_h_post_attn_2d);
            if (s != GEIST_OK) {
                return s;
            }
        }
    } else {
        s = v->add(be, &t_h_in_2d, &t_o_2d, &t_h_post_attn_2d);
        if (s != GEIST_OK) {
            return s;
        }
    }
    transformer_profile_add(&g_attn_profile, ATTN_PROFILE_POST, t0);
    return GEIST_OK;
}
