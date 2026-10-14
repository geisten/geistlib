/*
 * src/archs/transformer/families/bitnet.c - the BitNet b1.58 family unit.
 *
 * Layer: ARCHITECTURE.
 *
 * Everything that makes a transformer a BitNet one lives here (#37):
 *
 *   populate / populate_layers  GGUF metadata and per-layer geometry
 *   wire_layer                  the SubLN gammas (attn_sub_norm, ffn_sub_norm)
 *   attention_block, ffn_block  the layer forward, composed of shared blocks
 *
 *   h_post_attn = h_in + o_proj(SubLN(attention(rope(qkv(actq(norm(h_in)))))))
 *   h_out       = h_post_attn + down(SubLN(act(gate, up)(norm(h_post_attn))))
 *
 * The generic loader and layer stages know nothing about BitNet; removing
 * the family is this file, its two registry entries in arch_family.c and its
 * line in mk/common.mk. The blocks read no family flag: what this family
 * cannot run (a GEGLU activation key) is refused at load.
 *
 * Shared with the BitNet embedding models (gemma3/qwen3 archs, not this
 * family): the activation fake-quant (bitnet_input_quant) and the SubLN
 * slots, which their per-projection norms reuse.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../arch_state.h"
#include "../forward/internal.h"
#include "../rotation.h"
#include "../weight_load/internal.h"

#include "gemma4_kernels.h" /* rope_table_width: shared RoPE geometry, lives here today */
#include "gguf_reader.h"
#include <geist_types.h>

#include <math.h>
#include <stdio.h>

/* ---- BitNet b1.58 ------------------------------------------------------ */
/*
 * BitNet b1.58 is Llama-style transformer with two architectural
 * additions over the Llama family populator:
 *
 *   1. SubLN: an extra RMSNorm before each BitLinear (between
 *      attn-output and o_proj, between FFN activation and down_proj),
 *      gated by has_sub_ln.
 *
 *   2. FFN activation: BitNet b1.58 2B-4T (Microsoft flagship) uses *gated*
 *      squared-ReLU — relu(gate)^2 * up, with gate/up/down all present (HF
 *      hidden_act="relu2"). Community 1bitLLM/bitnet_b1_58-* use SwiGLU. The
 *      activation is not in the GGUF, so ffn_activation_from_meta picks the
 *      default by general.architecture ("bitnet-b1.58" -> gated squared-ReLU,
 *      "bitnet" -> SwiGLU) and a *.feed_forward_activation key overrides it.
 *
 * Tensor weights are typically TQ2_0. Tokenizer is Llama3-style BPE
 * for 2B-4T, Llama2-style SentencePiece for 3B — both through the
 * GGUF-embedded tokenizer path.
 */
static enum geist_ffn_activation_kind ffn_activation_from_meta(struct gguf_ctx *gguf) {
    /* Read general.architecture + an optional *.feed_forward_activation override
     * and let geist_ffn_activation_select (pure, unit-tested) decide. The default
     * is arch-keyed because the activation is NOT in the GGUF and the BitNet
     * families differ: "bitnet-b1.58" (Microsoft 2B-4T) needs gated squared-ReLU
     * (SwiGLU scores chance on MMLU) while community "bitnet" uses SwiGLU.
     * The official 2B-4T converter / mainline
     * llama.cpp don't emit the activation key. */
    size_t      al = 0, len = 0;
    const char *arch = gguf_get_meta_string(gguf, "general.architecture", &al);
    const char *s    = gguf_get_meta_string(gguf, "bitnet-b1.58.feed_forward_activation", &len);
    if (s == nullptr) {
        s = gguf_get_meta_string(gguf, "bitnet.feed_forward_activation", &len);
    }
    if (s == nullptr) {
        s = gguf_get_meta_string(gguf, "general.feed_forward_activation", &len);
    }
    return geist_ffn_activation_select(al, s != nullptr ? len : 0, arch, s);
}

static void populate_bitnet_b158(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Neutral config + the BitNet additions: SubLN and the arch-keyed
     * FFN activation. Llama-style underneath, but NEOX RoPE (the
     * bitnet.cpp converter does NOT pre-permute Q/K) — the neutral
     * rope_interleaved=false is correct, unlike the llama family. */
    st->config.has_sub_ln     = true;
    st->config.ffn_activation = ffn_activation_from_meta(gguf);

    /* Microsoft's bitnet.cpp converter writes metadata keys with the
     * full architecture string as the prefix ("bitnet-b1.58.*"), unlike
     * the llama family which uses just "llama.*". Try both — the
     * official GGUF uses "bitnet-b1.58.*", community converters may
     * use "bitnet.*". */
    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.block_count", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.block_count", &u))
        st->n_layers = u;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.embedding_length", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.attention.head_count", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.attention.head_count_kv", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_u32(gguf, "bitnet-b1.58.vocab_size", &u) ||
        gguf_get_meta_u32(gguf, "bitnet.vocab_size", &u)) {
        st->vocab_size = u;
    } else {
        /* Community BitNet GGUFs (e.g. gianni-cor's TQ2_0 / Q4_0 builds)
         * omit the vocab_size metadata key. Fall back to the
         * tokenizer.ggml.tokens array length, which is always present
         * because the tokenizer rides inside the GGUF. */
        uint32_t       elem_vt = 0;
        uint64_t       count   = 0;
        const uint8_t *payload = nullptr;
        if (gguf_get_meta_array_info(gguf, "tokenizer.ggml.tokens", &elem_vt, &count, &payload)) {
            st->vocab_size = (size_t) count;
        }
    }
    if (gguf_get_meta_f32(gguf, "bitnet-b1.58.attention.layer_norm_rms_epsilon", &f) ||
        gguf_get_meta_f32(gguf, "bitnet.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    /* No PLE — keep the PLE-derived dims at zero so any accidental
     * access traps on size==0. */
    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static bool populate_layers_bitnet_b158(struct transformer_arch_state *st) {
    /* BitNet is Llama-uniform: all layers full attention, no KV sharing,
     * head_dim = d_model / n_q_heads, single FFN intermediate from
     * bitnet.feed_forward_length. RoPE rotates the full head_dim. */
    const size_t head_dim     = (st->n_q_heads > 0) ? (st->d_model / st->n_q_heads) : 0;
    uint32_t     intermediate = 0;
    if (st->gguf != nullptr) {
        if (!gguf_get_meta_u32((struct gguf_ctx *) st->gguf,
                               "bitnet-b1.58.feed_forward_length",
                               &intermediate)) {
            gguf_get_meta_u32(
                    (struct gguf_ctx *) st->gguf, "bitnet.feed_forward_length", &intermediate);
        }
    }
    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        L->layer_idx                        = (int) i;
        L->is_full                          = true;
        L->is_kv_shared                     = false;
        L->head_dim                         = head_dim;
        L->q_out                            = st->n_q_heads * head_dim;
        L->kv_out                           = st->n_kv_heads * head_dim;
        L->intermediate                     = intermediate;
        L->sliding_window                   = 0;
        /* Default: Llama2 convention (10000). BitNet b1.58 3B uses the
         * Llama2 rope_theta; the 2B-4T flagship is Llama3-derived and
         * sets the metadata key, which the loop below picks up. */
        L->rope_theta     = 10000.0f;
        L->n_rotated_dims = (int) head_dim;
    }
    if (st->gguf != nullptr) {
        float freq_base;
        if (gguf_get_meta_f32(
                    (struct gguf_ctx *) st->gguf, "bitnet-b1.58.rope.freq_base", &freq_base) ||
            gguf_get_meta_f32((struct gguf_ctx *) st->gguf, "bitnet.rope.freq_base", &freq_base)) {
            for (size_t i = 0; i < st->n_layers; i++) {
                st->layers[i].rope_theta = freq_base;
            }
        }
    }
    return true;
}

/* ---- Tensors ---------------------------------------------------------- */

[[nodiscard]] static enum geist_status bitnet_wire_layer(struct transformer_arch_state    *st,
                                                         struct gguf_ctx                  *gguf,
                                                         struct transformer_layer_weights *L) {
    /* The blocks below implement relu^2 (gated or not) and SwiGLU. A GEGLU
     * key would need the GEGLU fusions; refuse rather than run it unfused. */
    if (st->config.ffn_activation == GEIST_FFN_GEGLU) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "bitnet: feed_forward_activation 'geglu' is not supported");
        return GEIST_E_UNSUPPORTED;
    }
    /* SubLN: an RMSNorm on the attention output before o_proj (q_out wide)
     * and on the FFN activation before down_proj (intermediate wide). */
    char path[64];
    snprintf(path, sizeof path, "blk.%d.attn_sub_norm.weight", L->layer_idx);
    enum geist_status s = load_layer_norm(st, gguf, L, path, L->q_out, &L->attn_sub_norm);
    if (s != GEIST_OK) {
        return s;
    }
    snprintf(path, sizeof path, "blk.%d.ffn_sub_norm.weight", L->layer_idx);
    return load_layer_norm(st, gguf, L, path, L->intermediate, &L->ffn_sub_norm);
}

/* ---- Forward ---------------------------------------------------------- */

[[nodiscard]] static enum geist_status
bitnet_attention_block(struct transformer_layer_forward_ctx *ctx) {

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
    s = transformer_rotate_rows(st, ctx->seq, st->d_model, sess->scratch_normed);
    if (s != GEIST_OK) {
        return s;
    }

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
    apply_per_channel_inv_scale_inplace(
            v, sess->scratch_attn, ctx->seq, ctx->q_out, L->o_awq_inv_scale);

    struct geist_tensor t_attn = view_2d(sess->scratch_attn, ctx->SEQ, (int64_t) ctx->q_out);
    if (st->runtime_flags.bitnet_sub_ln_enabled) { /* GEIST_BITNET_NO_SUBLN ablates it */
        struct geist_tensor t_w_sub = view_1d(L->attn_sub_norm.buffer, (int64_t) ctx->q_out);
        s                           = prims->rmsnorm(be, &t_attn, &t_w_sub, ctx->eps, &t_attn);
        if (s != GEIST_OK) {
            return s;
        }
    }
    s = transformer_rotate_rows(st, ctx->seq, ctx->q_out, sess->scratch_attn);
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

[[nodiscard]] static enum geist_status bitnet_ffn_block(struct transformer_layer_forward_ctx *ctx) {

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
    s = transformer_rotate_rows(st, ctx->seq, st->d_model, sess->scratch_pre_ff);
    if (s != GEIST_OK) {
        return s;
    }

    /* mid = act(gate, up): relu(gate)^2 * up (bitnet-b1.58), silu(gate) * up
     * (community "bitnet"), or relu(up)^2 without a gate (squared_relu key). */
    struct geist_tensor  t_up = view_2d(sess->scratch_up, ctx->SEQ, (int64_t) ctx->inter);
    struct geist_buffer *mid_buf;
    struct geist_tensor  t_mid;
    if (ctx->ffn_activation == GEIST_FFN_SQUARED_RELU) {
        s = linear_w_or_legacy(be,
                               v,
                               sess->scratch_pre_ff,
                               sess->scratch_up,
                               &L->up_proj_w,
                               ctx->seq,
                               &t_pre_ff,
                               &L->up_proj,
                               &t_up);
        if (s == GEIST_OK) {
            s = prims->relu_squared(be, &t_up, &t_up);
        }
        mid_buf = sess->scratch_up;
        t_mid   = t_up;
    } else {
        struct geist_tensor t_gate = view_2d(sess->scratch_gate, ctx->SEQ, (int64_t) ctx->inter);
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
        if (s == GEIST_OK && ctx->ffn_activation == GEIST_FFN_GATED_SQUARED_RELU) {
            s = prims->relu_squared(be, &t_gate, &t_gate);
            if (s == GEIST_OK) {
                s = prims->mul(be, &t_gate, &t_up, &t_gate);
            }
        } else if (s == GEIST_OK && ctx->P->fuse_silu_mul) {
            s = ctx->fused->silu_mul(be, &t_gate, &t_up, &t_gate);
        } else if (s == GEIST_OK) {
            s = st->model_fusions.prim_silu ? prims->silu(be, &t_gate, &t_gate)
                                            : prims->gelu_tanh(be, &t_gate, &t_gate);
            if (s == GEIST_OK) {
                s = prims->mul(be, &t_gate, &t_up, &t_gate);
            }
        }
        mid_buf = sess->scratch_gate;
        t_mid   = t_gate;
    }
    if (s != GEIST_OK) {
        return s;
    }

    if (st->runtime_flags.bitnet_sub_ln_enabled) {
        struct geist_tensor t_w_sub = view_1d(L->ffn_sub_norm.buffer, (int64_t) ctx->inter);
        s                           = prims->rmsnorm(be, &t_mid, &t_w_sub, ctx->eps, &t_mid);
        if (s != GEIST_OK) {
            return s;
        }
    }
    /* The sparsity probe counts zeros of the scaled activation before the
     * rotation mixes each block (GEIST_DUMP_ACT_SPARSITY). */
    const float *down_scale = L->down_awq_inv_scale;
    if (st->runtime_flags.dump_act_sparsity) {
        apply_per_channel_inv_scale_inplace(v, mid_buf, ctx->seq, ctx->inter, down_scale);
        down_scale = nullptr;
        transformer_probe_ffn_sparsity(v, true, ctx->layer_idx, mid_buf, ctx->seq * ctx->inter);
    }
    s = transformer_rotate_rows(st, ctx->seq, ctx->inter, mid_buf);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_tensor t_ffn_out = view_2d(sess->scratch_ffn_out, ctx->SEQ, st->d_model);
    s                             = linear_w_scaled_input_or_legacy(be,
                                                                    v,
                                                                    mid_buf,
                                                                    sess->scratch_ffn_out,
                                                                    &L->down_proj_w,
                                                                    ctx->seq,
                                                                    ctx->inter,
                                                                    down_scale,
                                                                    &t_mid,
                                                                    &L->down_proj,
                                                                    &t_ffn_out);
    if (s != GEIST_OK) {
        return s;
    }

    /* No PLE stage in this family: the residual goes straight to h_out. */
    struct geist_tensor t_h_out = view_2d(ctx->h_out_buf, ctx->SEQ, st->d_model);
    return prims->add(be, &t_h_post_attn, &t_ffn_out, &t_h_out);
}

/* ---- Registry entries ------------------------------------------------- */

const struct transformer_family transformer_family_bitnet_b158 = {
        .name            = "bitnet-b1.58",
        .populate        = populate_bitnet_b158,
        .populate_layers = populate_layers_bitnet_b158,
        .wire_layer      = bitnet_wire_layer,
        .attention_block = bitnet_attention_block,
        .ffn_block       = bitnet_ffn_block,
};

/* Community converters (e.g. gianni-cor/bitnet_b1_58-3B-TQ2_0) sometimes
 * emit "general.architecture" = "bitnet" without the b1.58 suffix. Same
 * unit — the metadata-key fallback chain in populate_bitnet_b158 already
 * tries "bitnet.*" after "bitnet-b1.58.*". */
const struct transformer_family transformer_family_bitnet = {
        .name            = "bitnet",
        .populate        = populate_bitnet_b158,
        .populate_layers = populate_layers_bitnet_b158,
        .wire_layer      = bitnet_wire_layer,
        .attention_block = bitnet_attention_block,
        .ffn_block       = bitnet_ffn_block,
};
