/*
 * src/archs/transformer/families/bitnet.c - the BitNet b1.58 family unit.
 *
 * Layer: ARCHITECTURE.
 *
 * Everything that makes a transformer a BitNet one (#37): GGUF metadata,
 * layer geometry, the SubLN gammas and the layer forward, composed of the
 * shared blocks in forward/blocks.h:
 *
 *   h_post_attn = h_in + o_proj(SubLN(attention(rope(qkv(actq(norm(h_in)))))))
 *   h_out       = h_post_attn + down(SubLN(act(gate, up)(norm(h_post_attn))))
 *
 * BitNet is Llama-style otherwise: every layer full attention, no KV
 * sharing, NEOX RoPE over the whole head (the bitnet.cpp converter does not
 * pre-permute Q/K). Weights are typically I2_S or TQ2_0.
 *
 * Removing the family is this file, its registry lines in arch_family.c and
 * arch_family.h, and its line in mk/common.mk.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../rotation.h"
#include "../weight_load/internal.h"

#include "gguf_reader.h"

#include <math.h>
#include <stdio.h>

/* ---- Metadata --------------------------------------------------------- */

/* Microsoft's converter prefixes keys with the full architecture string
 * ("bitnet-b1.58.*"), community converters with "bitnet.*". */
static bool meta_u32(struct gguf_ctx *g, const char *key, uint32_t *out) {
    char k[96];
    snprintf(k, sizeof k, "bitnet-b1.58.%s", key);
    if (gguf_get_meta_u32(g, k, out)) {
        return true;
    }
    snprintf(k, sizeof k, "bitnet.%s", key);
    return gguf_get_meta_u32(g, k, out);
}

static bool meta_f32(struct gguf_ctx *g, const char *key, float *out) {
    char k[96];
    snprintf(k, sizeof k, "bitnet-b1.58.%s", key);
    if (gguf_get_meta_f32(g, k, out)) {
        return true;
    }
    snprintf(k, sizeof k, "bitnet.%s", key);
    return gguf_get_meta_f32(g, k, out);
}

/* The activation is not in the official GGUF: "bitnet-b1.58" (Microsoft
 * 2B-4T, HF hidden_act="relu2") defaults to gated squared-ReLU — under
 * SwiGLU it scores chance on MMLU — and community "bitnet" to SwiGLU. A
 * *.feed_forward_activation key overrides it (geist_ffn_activation_select). */
static enum geist_ffn_activation_kind activation_from_meta(struct gguf_ctx *g) {
    size_t      al = 0, len = 0;
    const char *arch = gguf_get_meta_string(g, "general.architecture", &al);
    const char *act  = gguf_get_meta_string(g, "bitnet-b1.58.feed_forward_activation", &len);
    if (act == nullptr) {
        act = gguf_get_meta_string(g, "bitnet.feed_forward_activation", &len);
    }
    if (act == nullptr) {
        act = gguf_get_meta_string(g, "general.feed_forward_activation", &len);
    }
    return geist_ffn_activation_select(al, act != nullptr ? len : 0, arch, act);
}

static void bitnet_populate(struct gguf_ctx *g, struct transformer_arch_state *st) {
    st->config.has_sub_ln     = true;
    st->config.ffn_activation = activation_from_meta(g);

    uint32_t u;
    float    f;
    if (meta_u32(g, "block_count", &u)) {
        st->n_layers = u;
    }
    if (meta_u32(g, "embedding_length", &u)) {
        st->d_model = u;
    }
    if (meta_u32(g, "attention.head_count", &u)) {
        st->n_q_heads = u;
    }
    if (meta_u32(g, "attention.head_count_kv", &u)) {
        st->n_kv_heads = u;
    }
    if (meta_u32(g, "vocab_size", &u)) {
        st->vocab_size = u;
    } else {
        /* Community GGUFs omit vocab_size; the tokenizer rides inside. */
        uint32_t       elem_vt = 0;
        uint64_t       count   = 0;
        const uint8_t *payload = nullptr;
        if (gguf_get_meta_array_info(g, "tokenizer.ggml.tokens", &elem_vt, &count, &payload)) {
            st->vocab_size = (size_t) count;
        }
    }
    if (meta_f32(g, "attention.layer_norm_rms_epsilon", &f)) {
        st->config.rms_eps = f;
    }
    /* No PLE: zero dims make any accidental access trap on size 0. */
    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static bool bitnet_populate_layers(struct transformer_arch_state *st) {
    struct gguf_ctx *g            = (struct gguf_ctx *) st->gguf;
    const size_t     head_dim     = st->n_q_heads > 0 ? st->d_model / st->n_q_heads : 0;
    uint32_t         intermediate = 0;
    /* Llama2's rope_theta unless the GGUF sets it (2B-4T does). */
    float rope_theta = 10000.0f;
    if (g != nullptr) {
        (void) meta_u32(g, "feed_forward_length", &intermediate);
        (void) meta_f32(g, "rope.freq_base", &rope_theta);
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
        L->rope_theta                       = rope_theta;
        L->n_rotated_dims                   = (int) head_dim;
    }
    return true;
}

/* ---- Tensors ---------------------------------------------------------- */

[[nodiscard]] static enum geist_status bitnet_wire_layer(struct transformer_arch_state    *st,
                                                         struct gguf_ctx                  *gguf,
                                                         struct transformer_layer_weights *L) {
    /* The forward runs relu^2 (gated or not) and SwiGLU; refuse the rest
     * (today GEGLU) here rather than run something else at decode. */
    const enum geist_ffn_activation_kind act = st->config.ffn_activation;
    if (act != GEIST_FFN_SWIGLU && act != GEIST_FFN_SQUARED_RELU &&
        act != GEIST_FFN_GATED_SQUARED_RELU) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "bitnet: unsupported feed_forward_activation (e.g. 'geglu')");
        return GEIST_E_UNSUPPORTED;
    }
    /* SubLN gammas: before o_proj (q_out wide) and down_proj (intermediate). */
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

/* Each step runs only while s is GEIST_OK, so the first failure is what
 * the block returns and nothing after it touches a buffer. */

[[nodiscard]] static enum geist_status
bitnet_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const size_t                      d    = st->d_model;
    struct geist_buffer              *x    = sess->scratch_normed;

    enum geist_status s = blk_rmsnorm(ctx, d, ctx->h_in_buf, &L->attn_norm, x);
    if (s == GEIST_OK)
        bitnet_input_quant(ctx, x, d);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, d, x);
    if (s == GEIST_OK)
        s = blk_qkv(ctx, x);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, 1.0f / sqrtf((float) ctx->hd));
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    if (s == GEIST_OK)
        s = blk_o_proj(ctx); /* SubLN, GEIST_BITNET_NO_SUBLN, AWQ, rotation */
    if (s == GEIST_OK)
        s = blk_add(ctx, d, ctx->h_in_buf, sess->scratch_o, sess->scratch_h_post_attn);
    return s;
}

[[nodiscard]] static enum geist_status bitnet_ffn_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state       *st   = ctx->st;
    struct transformer_arch_session     *sess = ctx->sess;
    struct transformer_layer_weights    *L    = ctx->L;
    const size_t                         d    = st->d_model;
    const size_t                         n    = ctx->inter;
    struct geist_buffer                 *x    = sess->scratch_pre_ff;
    struct geist_buffer                 *up   = sess->scratch_up;
    const enum geist_ffn_activation_kind act  = ctx->ffn_activation;

    enum geist_status s = blk_rmsnorm(ctx, d, sess->scratch_h_post_attn, &L->ffn_norm, x);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, d, x);

    /* mid = relu(gate)^2 * up (bitnet-b1.58), silu(gate) * up (community
     * "bitnet") or relu(up)^2 without a gate (squared_relu key). */
    struct geist_buffer *mid = act == GEIST_FFN_SQUARED_RELU ? up : sess->scratch_gate;
    if (s == GEIST_OK && act == GEIST_FFN_SQUARED_RELU)
        s = blk_linear(ctx, x, &L->up_proj, &L->up_proj_w, up);
    if (s == GEIST_OK && act != GEIST_FFN_SQUARED_RELU)
        s = blk_gate_up(ctx, x);
    if (s == GEIST_OK && (act == GEIST_FFN_SQUARED_RELU || act == GEIST_FFN_GATED_SQUARED_RELU))
        s = blk_relu_sq(ctx, n, mid);
    if (s == GEIST_OK && act == GEIST_FFN_GATED_SQUARED_RELU)
        s = blk_mul(ctx, n, mid, up);
    if (s == GEIST_OK && act == GEIST_FFN_SWIGLU)
        s = blk_silu_mul(ctx, n, mid, up);

    if (s == GEIST_OK)
        s = blk_down(ctx, mid, L->down_awq_inv_scale); /* SubLN, probe, rotation */
    /* No PLE stage in this family: the residual goes straight to h_out. */
    if (s == GEIST_OK)
        s = blk_add(ctx, d, sess->scratch_h_post_attn, sess->scratch_ffn_out, ctx->h_out_buf);
    return s;
}

/* ---- Registry entries ------------------------------------------------- */

const struct transformer_family transformer_family_bitnet_b158 = {
        .name            = "bitnet-b1.58",
        .populate        = bitnet_populate,
        .populate_layers = bitnet_populate_layers,
        .wire_layer      = bitnet_wire_layer,
        .attention_block = bitnet_attention_block,
        .ffn_block       = bitnet_ffn_block,
};

/* Community converters (e.g. gianni-cor/bitnet_b1_58-3B-TQ2_0) write
 * "general.architecture" = "bitnet". Same unit; meta_* tries both prefixes. */
const struct transformer_family transformer_family_bitnet = {
        .name            = "bitnet",
        .populate        = bitnet_populate,
        .populate_layers = bitnet_populate_layers,
        .wire_layer      = bitnet_wire_layer,
        .attention_block = bitnet_attention_block,
        .ffn_block       = bitnet_ffn_block,
};
