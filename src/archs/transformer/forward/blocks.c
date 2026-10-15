/*
 * src/archs/transformer/forward/blocks.c - shared layer building blocks.
 *
 * Layer: ARCHITECTURE (transformer forward). Contract: blocks.h.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "blocks.h"
#include "../rotation.h"
#include "gemma4_kernels.h" /* rope_table_width: shared RoPE geometry, lives here today */

enum geist_status blk_rmsnorm(struct transformer_layer_forward_ctx *ctx,
                              size_t                                n,
                              struct geist_buffer                  *x,
                              const struct geist_tensor            *w,
                              struct geist_buffer                  *y) {
    struct geist_tensor t_x = view_2d(x, ctx->SEQ, (int64_t) n);
    struct geist_tensor t_w = view_1d(w->buffer, (int64_t) n);
    struct geist_tensor t_y = view_2d(y, ctx->SEQ, (int64_t) n);
    return ctx->prims->rmsnorm(ctx->be, &t_x, &t_w, ctx->eps, &t_y);
}

enum geist_status blk_add(struct transformer_layer_forward_ctx *ctx,
                          size_t                                n,
                          struct geist_buffer                  *a,
                          struct geist_buffer                  *b,
                          struct geist_buffer                  *y) {
    struct geist_tensor t_a = view_2d(a, ctx->SEQ, (int64_t) n);
    struct geist_tensor t_b = view_2d(b, ctx->SEQ, (int64_t) n);
    struct geist_tensor t_y = view_2d(y, ctx->SEQ, (int64_t) n);
    return ctx->prims->add(ctx->be, &t_a, &t_b, &t_y);
}

enum geist_status blk_linear(struct transformer_layer_forward_ctx *ctx,
                             struct geist_buffer                  *x,
                             const struct geist_tensor            *w,
                             const struct geist_weight            *wq,
                             struct geist_buffer                  *y) {
    struct geist_tensor t_x = view_2d(x, ctx->SEQ, w->shape[1]);
    struct geist_tensor t_y = view_2d(y, ctx->SEQ, w->shape[0]);
    return linear_w_or_legacy(ctx->be, ctx->v, x, y, wq, ctx->seq, &t_x, w, &t_y);
}

void blk_scale_cols(struct transformer_layer_forward_ctx *ctx,
                    size_t                                n,
                    const float                          *scale,
                    struct geist_buffer                  *x) {
    apply_per_channel_inv_scale_inplace(ctx->v, x, ctx->seq, n, scale);
}

/* BitNet embedding models: every projection reads its own normalisation of
 * x (and its activation fake-quant) out of scratch_proj_in. */
[[nodiscard]] static enum geist_status normed_linear(struct transformer_layer_forward_ctx *ctx,
                                                     struct geist_buffer                  *x,
                                                     const struct geist_tensor            *norm_w,
                                                     const struct geist_tensor            *w,
                                                     const struct geist_weight            *wq,
                                                     struct geist_buffer                  *y) {
    struct geist_buffer *in = ctx->sess->scratch_proj_in;
    const size_t         d  = ctx->st->d_model;
    enum geist_status    s  = blk_rmsnorm(ctx, d, x, norm_w, in);
    if (s == GEIST_OK && ctx->apply_bitnet_input_quant)
        bitnet_input_quant(ctx, in, d);
    if (s == GEIST_OK)
        s = blk_linear(ctx, in, w, wq, y);
    return s;
}

enum geist_status blk_qkv(struct transformer_layer_forward_ctx *ctx, struct geist_buffer *x) {
    if (ctx->apply_projection_input_norms) {
        struct transformer_arch_session  *ss = ctx->sess;
        struct transformer_layer_weights *Ln = ctx->L;
        enum geist_status                 s =
                normed_linear(ctx, x, &Ln->q_norm_in, &Ln->q_proj, &Ln->q_proj_w, ss->scratch_q);
        if (s == GEIST_OK)
            s = normed_linear(ctx, x, &Ln->k_norm_in, &Ln->k_proj, &Ln->k_proj_w, ss->scratch_k);
        if (s == GEIST_OK)
            s = normed_linear(ctx, x, &Ln->v_norm_in, &Ln->v_proj, &Ln->v_proj_w, ss->scratch_v);
        return s;
    }
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    struct geist_tensor               t_x  = view_2d(x, ctx->SEQ, ctx->st->d_model);
    struct geist_tensor t_q = view_2d(sess->scratch_q, ctx->SEQ, (int64_t) ctx->q_out);
    struct geist_tensor t_k = view_2d(sess->scratch_k, ctx->SEQ, (int64_t) ctx->kv_out);
    struct geist_tensor t_v = view_2d(sess->scratch_v, ctx->SEQ, (int64_t) ctx->kv_out);
    return linear_w_triple_or_legacy(ctx->be,
                                     ctx->v,
                                     x,
                                     sess->scratch_q,
                                     sess->scratch_k,
                                     sess->scratch_v,
                                     &L->q_proj_w,
                                     &L->k_proj_w,
                                     &L->v_proj_w,
                                     ctx->seq,
                                     &t_x,
                                     &L->q_proj,
                                     &L->k_proj,
                                     &L->v_proj,
                                     &t_q,
                                     &t_k,
                                     &t_v);
}

enum geist_status blk_qk_norm(struct transformer_layer_forward_ctx *ctx,
                              const struct geist_tensor            *q_w,
                              const struct geist_tensor            *k_w) {
    const int64_t       hd   = (int64_t) ctx->hd;
    struct geist_tensor t_q  = view_2d(ctx->sess->scratch_q, ctx->SEQ * ctx->st->n_q_heads, hd);
    struct geist_tensor t_k  = view_2d(ctx->sess->scratch_k, ctx->SEQ * ctx->st->n_kv_heads, hd);
    struct geist_tensor t_qw = view_1d(q_w->buffer, hd);
    struct geist_tensor t_kw = view_1d(k_w->buffer, hd);
    enum geist_status   s    = ctx->prims->rmsnorm(ctx->be, &t_q, &t_qw, ctx->eps, &t_q);
    if (s == GEIST_OK)
        s = ctx->prims->rmsnorm(ctx->be, &t_k, &t_kw, ctx->eps, &t_k);
    return s;
}

/* The cos/sin rows for this layer's RoPE (full or sliding table) at the
 * session's position. */
static void rope_rows(struct transformer_layer_forward_ctx *ctx,
                      struct geist_tensor                  *cos,
                      struct geist_tensor                  *sin) {
    struct transformer_arch_state    *st = ctx->st;
    struct transformer_layer_weights *L  = ctx->L;
    struct geist_buffer *cos_buf         = L->is_full ? st->rope_cos_full : st->rope_cos_sliding;
    struct geist_buffer *sin_buf         = L->is_full ? st->rope_sin_full : st->rope_sin_sliding;
    const size_t         n_rot =
            rope_table_width(ctx->hd, (size_t) L->n_rotated_dims, st->config.rope_partial_block);
    const size_t row_bytes = n_rot * sizeof(float);
    *cos = view_2d_at(cos_buf, ctx->q_position * row_bytes, ctx->SEQ, (int64_t) n_rot);
    *sin = view_2d_at(sin_buf, ctx->q_position * row_bytes, ctx->SEQ, (int64_t) n_rot);
}

enum geist_status blk_q_prep(struct transformer_layer_forward_ctx *ctx,
                             const struct geist_tensor            *q_norm,
                             float                                 q_scale) {
    struct transformer_arch_state   *st   = ctx->st;
    struct transformer_arch_session *sess = ctx->sess;
    struct geist_tensor              t_cos, t_sin;
    rope_rows(ctx, &t_cos, &t_sin);
    struct geist_tensor t_q = view_3d(sess->scratch_q, ctx->SEQ, st->n_q_heads, (int64_t) ctx->hd);

    enum geist_status s = GEIST_OK;
    if (q_norm != nullptr) {
        struct geist_tensor t_qh =
                view_2d(sess->scratch_q, ctx->SEQ * st->n_q_heads, (int64_t) ctx->hd);
        struct geist_tensor t_w = view_1d(q_norm->buffer, (int64_t) ctx->hd);
        s                       = ctx->prims->rmsnorm(ctx->be, &t_qh, &t_w, ctx->eps, &t_qh);
    }
    if (s == GEIST_OK)
        s = ctx->prims->rope_apply(ctx->be, &t_q, &t_cos, &t_sin);
    if (s != GEIST_OK || q_scale == 1.0f) { /* 1: the norms carry the scale (Gemma 4) */
        return s;
    }
    if (st->model_fusions.prim_scale_f32) {
        struct geist_tensor t_q2 = view_2d(sess->scratch_q, ctx->SEQ, (int64_t) ctx->q_out);
        return ctx->prims->scale_f32(ctx->be, &t_q2, q_scale, &t_q2);
    }
    float *qp = (float *) ctx->v->buffer_map(sess->scratch_q);
    if (qp == nullptr) {
        return GEIST_E_BACKEND;
    }
    for (size_t i = 0; i < ctx->seq * ctx->q_out; i++) {
        qp[i] *= q_scale;
    }
    ctx->v->buffer_unmap(sess->scratch_q);
    return GEIST_OK;
}

enum geist_status blk_k_prep(struct transformer_layer_forward_ctx *ctx,
                             const struct geist_tensor            *k_norm,
                             bool                                  v_norm) {
    struct transformer_arch_state   *st   = ctx->st;
    struct transformer_arch_session *sess = ctx->sess;
    const int64_t                    hd   = (int64_t) ctx->hd;
    const int64_t                    rows = ctx->SEQ * st->n_kv_heads;
    struct geist_tensor              t_cos, t_sin;
    rope_rows(ctx, &t_cos, &t_sin);
    struct geist_tensor t_k = view_3d(sess->scratch_k, ctx->SEQ, st->n_kv_heads, hd);

    enum geist_status s = GEIST_OK;
    if (k_norm != nullptr) {
        struct geist_tensor t_kh = view_2d(sess->scratch_k, rows, hd);
        struct geist_tensor t_w  = view_1d(k_norm->buffer, hd);
        s                        = ctx->prims->rmsnorm(ctx->be, &t_kh, &t_w, ctx->eps, &t_kh);
    }
    if (s == GEIST_OK && v_norm) { /* V RMS-normalised without a learned scale */
        struct geist_tensor t_vh   = view_2d(sess->scratch_v, rows, hd);
        struct geist_tensor t_ones = view_1d(sess->scratch_ones_headdim_max, hd);
        s                          = ctx->prims->rmsnorm(ctx->be, &t_vh, &t_ones, ctx->eps, &t_vh);
    }
    if (s == GEIST_OK)
        s = ctx->prims->rope_apply(ctx->be, &t_k, &t_cos, &t_sin);
    return s;
}

enum geist_status blk_rope_qk(struct transformer_layer_forward_ctx *ctx, float q_scale) {
    enum geist_status s = blk_q_prep(ctx, nullptr, q_scale);
    if (s == GEIST_OK)
        s = blk_k_prep(ctx, nullptr, false);
    return s;
}

enum geist_status blk_qkv_prep_fused(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const int64_t                     hd   = (int64_t) ctx->hd;
    struct geist_tensor               t_cos, t_sin;
    rope_rows(ctx, &t_cos, &t_sin);
    struct geist_tensor t_q  = view_3d(sess->scratch_q, ctx->SEQ, st->n_q_heads, hd);
    struct geist_tensor t_qw = view_1d(L->q_norm.buffer, hd);
    if (!ctx->compute_kv) {
        return ctx->fused->attn_qkv_prep(ctx->be,
                                         &t_q,
                                         nullptr,
                                         nullptr,
                                         &t_qw,
                                         nullptr,
                                         nullptr,
                                         &t_cos,
                                         &t_sin,
                                         ctx->eps,
                                         ctx->q_position,
                                         nullptr,
                                         nullptr);
    }
    const int64_t       kvh  = (int64_t) st->n_kv_heads;
    const int64_t       rows = (int64_t) (ctx->q_position + ctx->seq);
    struct geist_tensor t_k  = view_3d(sess->scratch_k, ctx->SEQ, kvh, hd);
    struct geist_tensor t_v  = view_3d(sess->scratch_v, ctx->SEQ, kvh, hd);
    struct geist_tensor t_kw = view_1d(L->k_norm.buffer, hd);
    struct geist_tensor t_on = view_1d(sess->scratch_ones_headdim_max, hd);
    struct geist_tensor t_kc = ctx->kv_f16_enabled ? view_3d_f16(ctx->k_cache_buf, rows, kvh, hd)
                                                   : view_3d(ctx->k_cache_buf, rows, kvh, hd);
    struct geist_tensor t_vc = ctx->kv_f16_enabled ? view_3d_f16(ctx->v_cache_buf, rows, kvh, hd)
                                                   : view_3d(ctx->v_cache_buf, rows, kvh, hd);
    return ctx->fused->attn_qkv_prep(ctx->be,
                                     &t_q,
                                     &t_k,
                                     &t_v,
                                     &t_qw,
                                     &t_kw,
                                     &t_on,
                                     &t_cos,
                                     &t_sin,
                                     ctx->eps,
                                     ctx->q_position,
                                     &t_kc,
                                     &t_vc);
}

enum geist_status blk_attend(struct transformer_layer_forward_ctx *ctx, bool append) {
    enum geist_status s = append ? transformer_kv_store_append(ctx) : GEIST_OK;
    if (s != GEIST_OK) {
        return s;
    }
    ctx->kv_len_now          = ctx->q_position + ctx->seq;
    const int64_t       n_q  = ctx->st->n_q_heads;
    struct geist_tensor t_q  = view_3d(ctx->sess->scratch_q, ctx->SEQ, n_q, (int64_t) ctx->hd);
    struct geist_tensor t_at = view_3d(ctx->sess->scratch_attn, ctx->SEQ, n_q, (int64_t) ctx->hd);
    return transformer_kv_store_attention(ctx, &t_q, &t_at);
}

enum geist_status blk_gate_up(struct transformer_layer_forward_ctx *ctx, struct geist_buffer *x) {
    if (ctx->apply_projection_input_norms) {
        struct transformer_arch_session  *ss = ctx->sess;
        struct transformer_layer_weights *Ln = ctx->L;
        enum geist_status                 s  = normed_linear(
                ctx, x, &Ln->gate_norm_in, &Ln->gate_proj, &Ln->gate_proj_w, ss->scratch_gate);
        if (s == GEIST_OK)
            s = normed_linear(
                    ctx, x, &Ln->up_norm_in, &Ln->up_proj, &Ln->up_proj_w, ss->scratch_up);
        return s;
    }
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    struct geist_tensor               t_x  = view_2d(x, ctx->SEQ, ctx->st->d_model);
    struct geist_tensor t_g  = view_2d(sess->scratch_gate, ctx->SEQ, (int64_t) ctx->inter);
    struct geist_tensor t_up = view_2d(sess->scratch_up, ctx->SEQ, (int64_t) ctx->inter);
    return linear_w_pair_or_legacy(ctx->be,
                                   ctx->v,
                                   x,
                                   sess->scratch_gate,
                                   sess->scratch_up,
                                   &L->gate_proj_w,
                                   &L->up_proj_w,
                                   ctx->seq,
                                   &t_x,
                                   &L->gate_proj,
                                   &L->up_proj,
                                   &t_g,
                                   &t_up);
}

enum geist_status
blk_relu_sq(struct transformer_layer_forward_ctx *ctx, size_t n, struct geist_buffer *x) {
    struct geist_tensor t_x = view_2d(x, ctx->SEQ, (int64_t) n);
    return ctx->prims->relu_squared(ctx->be, &t_x, &t_x);
}

enum geist_status blk_mul(struct transformer_layer_forward_ctx *ctx,
                          size_t                                n,
                          struct geist_buffer                  *gate,
                          struct geist_buffer                  *up) {
    struct geist_tensor t_g = view_2d(gate, ctx->SEQ, (int64_t) n);
    struct geist_tensor t_u = view_2d(up, ctx->SEQ, (int64_t) n);
    return ctx->prims->mul(ctx->be, &t_g, &t_u, &t_g);
}

enum geist_status blk_silu_mul(struct transformer_layer_forward_ctx *ctx,
                               size_t                                n,
                               struct geist_buffer                  *gate,
                               struct geist_buffer                  *up) {
    struct geist_tensor t_g = view_2d(gate, ctx->SEQ, (int64_t) n);
    struct geist_tensor t_u = view_2d(up, ctx->SEQ, (int64_t) n);
    if (ctx->P->fuse_silu_mul) {
        return ctx->fused->silu_mul(ctx->be, &t_g, &t_u, &t_g);
    }
    /* gelu_tanh is the bound alternative for a backend without silu. */
    enum geist_status s = ctx->st->model_fusions.prim_silu
                                  ? ctx->prims->silu(ctx->be, &t_g, &t_g)
                                  : ctx->prims->gelu_tanh(ctx->be, &t_g, &t_g);
    if (s != GEIST_OK) {
        return s;
    }
    return ctx->prims->mul(ctx->be, &t_g, &t_u, &t_g);
}

enum geist_status blk_o_proj(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_layer_weights *L    = ctx->L;
    struct geist_buffer              *attn = ctx->sess->scratch_attn;
    const size_t                      q    = ctx->q_out;

    blk_scale_cols(ctx, q, L->o_awq_inv_scale, attn);
    enum geist_status s = GEIST_OK;
    if (L->attn_sub_norm.buffer != nullptr && st->runtime_flags.bitnet_sub_ln_enabled)
        s = blk_rmsnorm(ctx, q, attn, &L->attn_sub_norm, attn);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, q, attn);
    if (s == GEIST_OK)
        s = blk_linear(ctx, attn, &L->o_proj, &L->o_proj_w, ctx->sess->scratch_o);
    return s;
}

enum geist_status blk_down(struct transformer_layer_forward_ctx *ctx,
                           struct geist_buffer                  *mid,
                           const float                          *down_scale) {
    struct transformer_arch_state    *st = ctx->st;
    struct transformer_layer_weights *L  = ctx->L;
    const size_t                      n  = ctx->inter;

    enum geist_status s = GEIST_OK;
    if (L->ffn_sub_norm.buffer != nullptr && st->runtime_flags.bitnet_sub_ln_enabled)
        s = blk_rmsnorm(ctx, n, mid, &L->ffn_sub_norm, mid);
    /* The sparsity probe counts zeros of the scaled activation before the
     * rotation mixes each block (GEIST_DUMP_ACT_SPARSITY). */
    if (s == GEIST_OK && st->runtime_flags.dump_act_sparsity) {
        blk_scale_cols(ctx, n, down_scale, mid);
        down_scale = nullptr;
        transformer_probe_ffn_sparsity(ctx->v, true, ctx->layer_idx, mid, ctx->seq * n);
    }
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, n, mid);
    if (s == GEIST_OK)
        blk_scale_cols(ctx, n, down_scale, mid);
    if (s == GEIST_OK)
        s = blk_linear(ctx, mid, &L->down_proj, &L->down_proj_w, ctx->sess->scratch_ffn_out);
    return s;
}

enum geist_status blk_norm_add(struct transformer_layer_forward_ctx *ctx,
                               bool                                  fused,
                               const struct geist_tensor            *w,
                               struct geist_buffer                  *x,
                               struct geist_buffer                  *res,
                               struct geist_buffer                  *tmp,
                               struct geist_buffer                  *out) {
    const int64_t       d     = ctx->st->d_model;
    struct geist_tensor t_x   = view_2d(x, ctx->SEQ, d);
    struct geist_tensor t_res = view_2d(res, ctx->SEQ, d);
    struct geist_tensor t_out = view_2d(out, ctx->SEQ, d);
    struct geist_tensor t_w   = view_1d(w->buffer, d);
    /* Bound at plan build (#352): a failing fused op is an error, never a
     * host fallback over output it may already have written. */
    if (fused) {
        return ctx->fused->rmsnorm_add(ctx->be, &t_res, &t_x, &t_w, ctx->eps, &t_out);
    }
    enum geist_status s = blk_rmsnorm(ctx, (size_t) d, x, w, tmp);
    if (s == GEIST_OK)
        s = blk_add(ctx, (size_t) d, res, tmp, out);
    return s;
}

enum geist_status blk_ffn_swiglu(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const size_t                      d    = st->d_model;
    struct geist_buffer              *x    = sess->scratch_pre_ff;

    /* Each step runs only while s is GEIST_OK. */
    enum geist_status s = blk_rmsnorm(ctx, d, sess->scratch_h_post_attn, &L->ffn_norm, x);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, d, x);
    if (s == GEIST_OK)
        s = blk_gate_up(ctx, x);
    if (s == GEIST_OK)
        s = blk_silu_mul(ctx, ctx->inter, sess->scratch_gate, sess->scratch_up);
    if (s == GEIST_OK)
        s = blk_down(ctx, sess->scratch_gate, L->down_awq_inv_scale);
    if (s == GEIST_OK)
        s = blk_add(ctx, d, sess->scratch_h_post_attn, sess->scratch_ffn_out, ctx->h_out_buf);
    return s;
}

enum geist_status blk_ffn_geglu(struct transformer_layer_forward_ctx *ctx,
                                struct geist_buffer                  *out) {
    struct transformer_arch_state            *st     = ctx->st;
    struct transformer_arch_session          *sess   = ctx->sess;
    struct transformer_layer_weights         *L      = ctx->L;
    const struct transformer_layer_exec_plan *P      = ctx->P;
    const struct geist_backend_fused         *fused  = ctx->fused;
    struct geist_backend                     *be     = ctx->be;
    const size_t                              d      = st->d_model;
    const size_t                              n      = ctx->inter;
    struct geist_buffer                      *x      = sess->scratch_pre_ff;
    struct geist_buffer                      *mid    = sess->scratch_gate;
    struct geist_buffer                      *h      = sess->scratch_h_post_attn;
    struct geist_tensor                       t_h    = view_2d(h, ctx->SEQ, (int64_t) d);
    struct geist_tensor                       t_x    = view_2d(x, ctx->SEQ, (int64_t) d);
    struct geist_tensor                       t_gate = view_2d(mid, ctx->SEQ, (int64_t) n);
    struct geist_tensor t_up       = view_2d(sess->scratch_up, ctx->SEQ, (int64_t) n);
    struct geist_tensor t_wn       = view_1d(L->ffn_norm.buffer, (int64_t) d);
    const float        *down_scale = L->down_awq_inv_scale;
    enum geist_status   s;

    /* The fusions are bound at plan build (exec_plan.c); bound means the
     * call must succeed. Front: rmsnorm + gate/up + gelu*up in one dispatch. */
    const bool front = ctx->seq == 1 && P->fuse_ffn_norm_gate_up_m1;
    if (front) {
        s = fused->ffn_norm_gate_up(be, &t_h, &t_wn, ctx->eps, &L->gate_proj, &L->up_proj, &t_gate);
    } else {
        s = blk_rmsnorm(ctx, d, h, &L->ffn_norm, x);
        if (s == GEIST_OK)
            s = transformer_rotate_rows(st, ctx->seq, d, x);
    }
    if (s != GEIST_OK) {
        return s;
    }

    if (ctx->seq > 1 && P->fuse_ffn_geglu_tile_mN) { /* the whole FFN, prefill */
        const float *xp = (const float *) ctx->v->buffer_map(x);
        float       *yp = (float *) ctx->v->buffer_map(sess->scratch_ffn_out);
        if (xp == nullptr || yp == nullptr) {
            if (xp != nullptr)
                ctx->v->buffer_unmap(x);
            if (yp != nullptr)
                ctx->v->buffer_unmap(sess->scratch_ffn_out);
            return GEIST_E_BACKEND;
        }
        s = fused->ffn_geglu_q4q6_mN(be,
                                     ctx->seq,
                                     d,
                                     n,
                                     xp,
                                     &L->gate_proj_w,
                                     &L->up_proj_w,
                                     &L->down_proj_w,
                                     down_scale,
                                     yp);
        ctx->v->buffer_unmap(x);
        ctx->v->buffer_unmap(sess->scratch_ffn_out);
    } else {
        if (!front && ctx->seq == 1 && P->fuse_ffn_gate_up_m1) {
            s = fused->ffn_gate_up(be, &t_x, &L->gate_proj, &L->up_proj, &t_gate);
        } else if (!front) {
            s = blk_gate_up(ctx, x);
            if (s == GEIST_OK && P->fuse_gelu_mul_scaled) {
                s          = fused->gelu_tanh_mul_scaled(be, &t_gate, &t_up, down_scale, &t_gate);
                down_scale = nullptr; /* already applied */
            } else if (s == GEIST_OK && P->fuse_gelu_mul) {
                s = fused->gelu_tanh_mul(be, &t_gate, &t_up, &t_gate);
            } else if (s == GEIST_OK) {
                s = ctx->prims->gelu_tanh(be, &t_gate, &t_gate);
                if (s == GEIST_OK)
                    s = blk_mul(ctx, n, mid, sess->scratch_up);
            }
        }
        if (s == GEIST_OK)
            s = blk_down(ctx, mid, down_scale);
    }
    if (s == GEIST_OK)
        s = blk_norm_add(ctx,
                         P->fuse_rmsnorm_add,
                         &L->post_ffw_norm,
                         sess->scratch_ffn_out,
                         h,
                         sess->scratch_post_ff,
                         out);
    return s;
}
