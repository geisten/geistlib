/*
 * src/archs/transformer/forward/family_bitnet.c - BitNet b1.58 layer forward.
 *
 * Layer: ARCHITECTURE.
 *
 * Experiment #37, part B: one family's layer written as a straight
 * composition of shared building blocks instead of the flag-driven generic
 * stages (layer_attn.c, layer_ffn.c, layer_ple.c). It reads no family flag:
 * exec_plan.c binds it only for a model that is exactly this layer
 * (bitnet_forward_eligible), so every feature the generic stages branch on
 * is either fixed here or absent.
 *
 *   h_post_attn = h_in + o_proj(SubLN(attention(rope(qkv(actq(norm(h_in)))))))
 *   h_out       = h_post_attn + down(SubLN(act(gate, up)(norm(h_post_attn))))
 *
 * The order of operations is the generic path's for this feature set, so
 * the output is bit-identical (GEIST_FAMILY_FORWARD=0 runs the generic path
 * for comparison). The KV store is still reached through the layer ctx;
 * narrowing its signature is the next step of the experiment.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "gemma4_kernels.h" /* rope_table_width: shared RoPE geometry, lives here today */
#include <geist_types.h>

#include <math.h>

[[nodiscard]] static enum geist_status bitnet_attention(struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_state         *st    = ctx->st;
    struct transformer_arch_session       *sess  = ctx->sess;
    struct transformer_layer_weights      *L     = ctx->L;
    struct geist_backend                  *be    = ctx->be;
    const struct geist_backend_vtbl       *v     = ctx->v;
    const struct geist_backend_primitives *prims = ctx->prims;
    enum geist_status                      s;

    struct geist_tensor t_h_in   = view_2d(ctx->h_in_buf, ctx->SEQ, st->d_model);
    struct geist_tensor t_normed = view_2d(sess->scratch_normed, ctx->SEQ, st->d_model);
    struct geist_tensor t_w_norm = view_1d(L->attn_norm.buffer, st->d_model);
    s                            = prims->rmsnorm(be, &t_h_in, &t_w_norm, ctx->eps, &t_normed);
    if (s != GEIST_OK) {
        return s;
    }
    bitnet_input_quant(ctx, sess->scratch_normed, st->d_model);

    struct geist_tensor t_q = view_2d(sess->scratch_q, ctx->SEQ, (int64_t) ctx->q_out);
    struct geist_tensor t_k = view_2d(sess->scratch_k, ctx->SEQ, (int64_t) ctx->kv_out);
    struct geist_tensor t_v = view_2d(sess->scratch_v, ctx->SEQ, (int64_t) ctx->kv_out);
    s                       = linear_w_triple_or_legacy(be,
                                                        v,
                                                        sess->scratch_normed,
                                                        sess->scratch_q,
                                                        sess->scratch_k,
                                                        sess->scratch_v,
                                                        &L->q_proj_w,
                                                        &L->k_proj_w,
                                                        &L->v_proj_w,
                                                        ctx->seq,
                                                        &t_normed,
                                                        &L->q_proj,
                                                        &L->k_proj,
                                                        &L->v_proj,
                                                        &t_q,
                                                        &t_k,
                                                        &t_v);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_buffer *cos_buf = L->is_full ? st->rope_cos_full : st->rope_cos_sliding;
    struct geist_buffer *sin_buf = L->is_full ? st->rope_sin_full : st->rope_sin_sliding;
    const size_t         n_rot =
            rope_table_width(ctx->hd, (size_t) L->n_rotated_dims, st->config.rope_partial_block);
    const size_t        row_bytes = n_rot * sizeof(float);
    struct geist_tensor t_cos =
            view_2d_at(cos_buf, ctx->q_position * row_bytes, ctx->SEQ, (int64_t) n_rot);
    struct geist_tensor t_sin =
            view_2d_at(sin_buf, ctx->q_position * row_bytes, ctx->SEQ, (int64_t) n_rot);
    struct geist_tensor t_q_3d =
            view_3d(sess->scratch_q, ctx->SEQ, st->n_q_heads, (int64_t) ctx->hd);
    struct geist_tensor t_k_3d =
            view_3d(sess->scratch_k, ctx->SEQ, st->n_kv_heads, (int64_t) ctx->hd);

    s = prims->rope_apply(be, &t_q_3d, &t_cos, &t_sin);
    if (s != GEIST_OK) {
        return s;
    }
    const float q_scale = 1.0f / sqrtf((float) ctx->hd);
    if (st->model_fusions.prim_scale_f32) {
        s = prims->scale_f32(be, &t_q, q_scale, &t_q);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        float *qp = (float *) v->buffer_map(sess->scratch_q);
        if (qp == nullptr) {
            return GEIST_E_BACKEND;
        }
        for (size_t i = 0; i < ctx->seq * ctx->q_out; i++) {
            qp[i] *= q_scale;
        }
        v->buffer_unmap(sess->scratch_q);
    }
    s = prims->rope_apply(be, &t_k_3d, &t_cos, &t_sin);
    if (s != GEIST_OK) {
        return s;
    }

    s = transformer_kv_store_append(ctx);
    if (s != GEIST_OK) {
        return s;
    }
    ctx->kv_len_now = ctx->q_position + ctx->seq;

    struct geist_tensor t_attn_3d =
            view_3d(sess->scratch_attn, ctx->SEQ, st->n_q_heads, (int64_t) ctx->hd);
    s = transformer_kv_store_attention(ctx, &t_q_3d, &t_attn_3d);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_tensor t_attn  = view_2d(sess->scratch_attn, ctx->SEQ, (int64_t) ctx->q_out);
    struct geist_tensor t_w_sub = view_1d(L->attn_sub_norm.buffer, (int64_t) ctx->q_out);
    s                           = prims->rmsnorm(be, &t_attn, &t_w_sub, ctx->eps, &t_attn);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_tensor t_o = view_2d(sess->scratch_o, ctx->SEQ, st->d_model);
    s                       = linear_w_or_legacy(be,
                                                 v,
                                                 sess->scratch_attn,
                                                 sess->scratch_o,
                                                 &L->o_proj_w,
                                                 ctx->seq,
                                                 &t_attn,
                                                 &L->o_proj,
                                                 &t_o);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_tensor t_h_post_attn = view_2d(sess->scratch_h_post_attn, ctx->SEQ, st->d_model);
    return prims->add(be, &t_h_in, &t_o, &t_h_post_attn);
}

[[nodiscard]] static enum geist_status bitnet_ffn(struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_state         *st    = ctx->st;
    struct transformer_arch_session       *sess  = ctx->sess;
    struct transformer_layer_weights      *L     = ctx->L;
    struct geist_backend                  *be    = ctx->be;
    const struct geist_backend_vtbl       *v     = ctx->v;
    const struct geist_backend_primitives *prims = ctx->prims;
    enum geist_status                      s;

    struct geist_tensor t_h_post_attn = view_2d(sess->scratch_h_post_attn, ctx->SEQ, st->d_model);
    struct geist_tensor t_pre_ff      = view_2d(sess->scratch_pre_ff, ctx->SEQ, st->d_model);
    struct geist_tensor t_w_norm      = view_1d(L->ffn_norm.buffer, st->d_model);
    s = prims->rmsnorm(be, &t_h_post_attn, &t_w_norm, ctx->eps, &t_pre_ff);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_tensor t_gate = view_2d(sess->scratch_gate, ctx->SEQ, (int64_t) ctx->inter);
    struct geist_tensor t_up   = view_2d(sess->scratch_up, ctx->SEQ, (int64_t) ctx->inter);
    s                          = linear_w_pair_or_legacy(be,
                                                         v,
                                                         sess->scratch_pre_ff,
                                                         sess->scratch_gate,
                                                         sess->scratch_up,
                                                         &L->gate_proj_w,
                                                         &L->up_proj_w,
                                                         ctx->seq,
                                                         &t_pre_ff,
                                                         &L->gate_proj,
                                                         &L->up_proj,
                                                         &t_gate,
                                                         &t_up);
    if (s != GEIST_OK) {
        return s;
    }

    /* bitnet-b1.58 (Microsoft 2B-4T): relu(gate)^2 * up; community
     * "bitnet": SwiGLU. Eligibility admits exactly these two. */
    if (ctx->ffn_activation == GEIST_FFN_GATED_SQUARED_RELU) {
        s = prims->relu_squared(be, &t_gate, &t_gate);
        if (s == GEIST_OK) {
            s = prims->mul(be, &t_gate, &t_up, &t_gate);
        }
    } else if (ctx->P->fuse_silu_mul) {
        s = ctx->fused->silu_mul(be, &t_gate, &t_up, &t_gate);
    } else {
        s = st->model_fusions.prim_silu ? prims->silu(be, &t_gate, &t_gate)
                                        : prims->gelu_tanh(be, &t_gate, &t_gate);
        if (s == GEIST_OK) {
            s = prims->mul(be, &t_gate, &t_up, &t_gate);
        }
    }
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_tensor t_w_sub = view_1d(L->ffn_sub_norm.buffer, (int64_t) ctx->inter);
    s                           = prims->rmsnorm(be, &t_gate, &t_w_sub, ctx->eps, &t_gate);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_tensor t_ffn_out = view_2d(sess->scratch_ffn_out, ctx->SEQ, st->d_model);
    s                             = linear_w_or_legacy(be,
                                                       v,
                                                       sess->scratch_gate,
                                                       sess->scratch_ffn_out,
                                                       &L->down_proj_w,
                                                       ctx->seq,
                                                       &t_gate,
                                                       &L->down_proj,
                                                       &t_ffn_out);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_tensor t_h_out = view_2d(ctx->h_out_buf, ctx->SEQ, st->d_model);
    return prims->add(be, &t_h_post_attn, &t_ffn_out, &t_h_out);
}

enum geist_status transformer_bitnet_forward_layer(struct transformer_layer_forward_ctx *ctx) {
    enum geist_status s = bitnet_attention(ctx);
    if (s != GEIST_OK) {
        return s;
    }
    return bitnet_ffn(ctx);
}
