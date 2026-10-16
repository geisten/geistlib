/*
 * src/archs/transformer/families/qwen35.c - the Qwen3.5 / 3.6 / 3.8 hybrid unit.
 *
 * Layer: ARCHITECTURE.
 *
 * Dense hybrids (incl. Ternary-Bonsai-2-27B): most layers mix tokens with a
 * gated DeltaNet (layer.c dispatches those to the DeltaNet block), every
 * few layers is full attention with per-head QK-norm, partial RoPE and an
 * output gate (q_proj emits query and gate; attn *= sigmoid(gate)); SwiGLU
 * FFN; optional prism Hadamard rotation; trailing MTP block(s).
 *
 *   h_post_attn = h_in + o_proj(gate(attention(rope(qk_norm(qkv(norm(h_in)))))))
 *   h_out       = blk_ffn_swiglu(h_post_attn)
 *
 * The output gate and the rotation are model features the blocks apply.
 *
 * Removing the family is this file, its registry lines in arch_family.c and
 * arch_family.h, and its line in mk/common.mk (plus the DeltaNet block and
 * the MTP drafter, which only this family uses).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../rotation.h"

#include "gguf_reader.h"

#include <math.h>

/* ---- qwen35 (Qwen3.5/3.6/3.8 hybrid) ----------------------------------- */
/*
 * Hybrid token-mixer stack: with full_attention_interval N (default 4),
 * layer i is softmax attention iff (i+1) % N == 0, else gated DeltaNet
 * (recurrent conv + delta-rule state — see forward/layer_deltanet.c).
 * The attention layers are standard GQA with QK-norm, partial NEOX RoPE
 * (rope.dimension_count of head_dim, theta 1e7) and a sigmoid output
 * gate fed by a joint q+gate projection (has_attn_output_gate).
 *
 * block_count INCLUDES nextn_predict_layers trailing MTP block(s). They
 * are not part of the autoregressive stack: n_layers excludes them and
 * n_mtp_layers records them for separate loading/execution.
 *
 * GGUF norm weights arrive with the zero-centered (+1) already baked in
 * by the converter, so the standard rmsnorm path is correct as-is.
 */
static void populate_qwen35(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    st->config.has_qk_norms         = true;
    st->config.has_attn_output_gate = true;
    st->config.ffn_activation       = GEIST_FFN_SWIGLU;

    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "qwen35.block_count", &u))
        st->n_layers = u;
    uint32_t nextn = 0;
    gguf_get_meta_u32(gguf, "qwen35.nextn_predict_layers", &nextn);
    if (nextn < st->n_layers) {
        st->n_mtp_layers = nextn;
        st->n_layers -= nextn;
    } else if (nextn != 0) {
        /* Make malformed geometry fail in state_create instead of treating
         * MTP blocks as autoregressive layers. */
        st->n_layers     = 0;
        st->n_mtp_layers = nextn;
    }
    if (gguf_get_meta_u32(gguf, "qwen35.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "qwen35.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen35.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_f32(gguf, "qwen35.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    { /* vocab from the tokenizer table (no qwen35.vocab_size key) */
        uint32_t       elem_vt = 0;
        uint64_t       count   = 0;
        const uint8_t *payload = nullptr;
        if (gguf_get_meta_array_info(gguf, "tokenizer.ggml.tokens", &elem_vt, &count, &payload))
            st->vocab_size = (size_t) count;
    }
    /* DeltaNet geometry. GGUF key mapping (#281): state_size =
     * head_k (= head_v), group_count = n_k_heads, time_step_rank =
     * n_v_heads, inner_size = n_v_heads * head_v. */
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.group_count", &u))
        st->config.dn_n_k_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.time_step_rank", &u))
        st->config.dn_n_v_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.state_size", &u))
        st->config.dn_head_k = u;
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.inner_size", &u) && st->config.dn_n_v_heads > 0)
        st->config.dn_head_v = u / st->config.dn_n_v_heads;
    if (gguf_get_meta_u32(gguf, "qwen35.ssm.conv_kernel", &u))
        st->config.dn_conv_kernel = u;
    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static bool populate_layers_qwen35(struct transformer_arch_state *st) {
    struct gguf_ctx *g = (struct gguf_ctx *) st->gguf;

    uint32_t head_dim = 0, intermediate = 0, rot = 0, interval = 4;
    float    freq_base = 10000000.0f;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "qwen35.attention.key_length", &head_dim);
        gguf_get_meta_u32(g, "qwen35.feed_forward_length", &intermediate);
        gguf_get_meta_u32(g, "qwen35.rope.dimension_count", &rot);
        gguf_get_meta_u32(g, "qwen35.full_attention_interval", &interval);
        gguf_get_meta_f32(g, "qwen35.rope.freq_base", &freq_base);
    }
    if (head_dim == 0 || intermediate == 0 || interval == 0 || rot == 0 || rot > head_dim ||
        st->config.dn_n_k_heads == 0 || st->config.dn_n_v_heads == 0 || st->config.dn_head_k == 0 ||
        st->config.dn_head_v == 0 || st->config.dn_conv_kernel < 2) {
        return false;
    }
    /* Optional explicit bool array overrides the interval formula. */
    uint32_t       rec_vt = 0;
    uint64_t       rec_n  = 0;
    const uint8_t *rec    = nullptr;
    if (g != nullptr &&
        gguf_get_meta_array_info(g, "qwen35.attention.recurrent_layers", &rec_vt, &rec_n, &rec)) {
        if (rec_vt != GGUF_META_VT_BOOL || rec_n < st->n_layers)
            rec = nullptr;
    }

    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        st->config.rope_partial_block       = true; /* ggml layout, see arch_config.h */
        const bool is_attn = rec != nullptr ? rec[i] == 0 : ((i + 1) % (size_t) interval == 0);
        L->layer_idx       = (int) i;
        L->mixer           = is_attn ? GEIST_MIXER_ATTN : GEIST_MIXER_DELTANET;
        L->is_full         = true;
        L->is_kv_shared    = false;
        L->head_dim        = head_dim;
        L->q_out           = st->n_q_heads * head_dim;
        L->kv_out          = st->n_kv_heads * head_dim;
        L->intermediate    = intermediate;
        L->sliding_window  = 0;
        L->rope_theta      = freq_base;
        L->n_rotated_dims  = (int) rot;
    }
    for (size_t i = 0; i < st->n_mtp_layers; i++) {
        struct transformer_layer_weights *L = &st->mtp_layers[i].block;
        L->layer_idx                        = (int) (st->n_layers + i);
        L->mixer                            = GEIST_MIXER_ATTN;
        L->is_full                          = true;
        L->is_kv_shared                     = false;
        L->head_dim                         = head_dim;
        L->q_out                            = st->n_q_heads * head_dim;
        L->kv_out                           = st->n_kv_heads * head_dim;
        L->intermediate                     = intermediate;
        L->sliding_window                   = 0;
        L->rope_theta                       = freq_base;
        L->n_rotated_dims                   = (int) rot;
    }
    return true;
}

/* ---- Forward ---------------------------------------------------------- */

/* Each step runs only while s is GEIST_OK, so the first failure is what
 * the stage returns and nothing after it touches a buffer. Also the MTP
 * block's attention (forward/layer.c, with an empty plan). */
[[nodiscard]] static enum geist_status
qwen35_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const size_t                      d    = st->d_model;
    struct geist_buffer              *x    = sess->scratch_normed;

    enum geist_status s = blk_rmsnorm(ctx, d, ctx->h_in_buf, &L->attn_norm, x);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, d, x);
    if (s == GEIST_OK)
        s = blk_qkv(ctx, x); /* joint q+gate, split */
    if (s == GEIST_OK)
        s = blk_qk_norm(ctx, &L->q_norm, &L->k_norm);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, 1.0f / sqrtf((float) ctx->hd));
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    if (s == GEIST_OK)
        s = blk_o_proj(ctx); /* sigmoid gate, rotation */
    if (s == GEIST_OK)
        s = blk_add(ctx, d, ctx->h_in_buf, sess->scratch_o, sess->scratch_h_post_attn);
    return s;
}

/* ---- Registry entry --------------------------------------------------- */

const struct transformer_family transformer_family_qwen35 = {
        .name     = "qwen35",
        .rotation = true,
        /* 27B weights (Ternary-Bonsai-2-27B, Qwen3.5-27B) dequantized once
         * per chunk are the prefill's cost: twice the chunk from 4 GiB on
         * (256 is no better than 128). */
        .prefill_delta   = {.from_bytes = (size_t) 4 << 30, .m_max_rows = 64},
        .populate        = populate_qwen35,
        .populate_layers = populate_layers_qwen35,
        .attention_block = qwen35_attention_block,
        .ffn_block       = blk_ffn_swiglu,
};
