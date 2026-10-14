/*
 * src/archs/transformer/forward/blocks.c - shared layer building blocks.
 *
 * Layer: ARCHITECTURE (transformer forward). Contract: blocks.h.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "blocks.h"
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

enum geist_status blk_qkv(struct transformer_layer_forward_ctx *ctx, struct geist_buffer *x) {
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

enum geist_status blk_rope_qk(struct transformer_layer_forward_ctx *ctx, float q_scale) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;

    struct geist_buffer *cos_buf = L->is_full ? st->rope_cos_full : st->rope_cos_sliding;
    struct geist_buffer *sin_buf = L->is_full ? st->rope_sin_full : st->rope_sin_sliding;
    const size_t         n_rot =
            rope_table_width(ctx->hd, (size_t) L->n_rotated_dims, st->config.rope_partial_block);
    const size_t        row_bytes = n_rot * sizeof(float);
    struct geist_tensor t_cos =
            view_2d_at(cos_buf, ctx->q_position * row_bytes, ctx->SEQ, (int64_t) n_rot);
    struct geist_tensor t_sin =
            view_2d_at(sin_buf, ctx->q_position * row_bytes, ctx->SEQ, (int64_t) n_rot);
    struct geist_tensor t_q = view_3d(sess->scratch_q, ctx->SEQ, st->n_q_heads, (int64_t) ctx->hd);
    struct geist_tensor t_k = view_3d(sess->scratch_k, ctx->SEQ, st->n_kv_heads, (int64_t) ctx->hd);

    enum geist_status s = ctx->prims->rope_apply(ctx->be, &t_q, &t_cos, &t_sin);
    if (s != GEIST_OK) {
        return s;
    }
    if (st->model_fusions.prim_scale_f32) {
        struct geist_tensor t_q2 = view_2d(sess->scratch_q, ctx->SEQ, (int64_t) ctx->q_out);
        s                        = ctx->prims->scale_f32(ctx->be, &t_q2, q_scale, &t_q2);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        float *qp = (float *) ctx->v->buffer_map(sess->scratch_q);
        if (qp == nullptr) {
            return GEIST_E_BACKEND;
        }
        for (size_t i = 0; i < ctx->seq * ctx->q_out; i++) {
            qp[i] *= q_scale;
        }
        ctx->v->buffer_unmap(sess->scratch_q);
    }
    return ctx->prims->rope_apply(ctx->be, &t_k, &t_cos, &t_sin);
}

enum geist_status blk_attend(struct transformer_layer_forward_ctx *ctx) {
    enum geist_status s = transformer_kv_store_append(ctx);
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
