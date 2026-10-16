/*
 * src/archs/transformer/families/qwen3moe.c - the Qwen3-MoE family unit
 * (#48).
 *
 * Layer: ARCHITECTURE.
 *
 * Qwen3's attention (GQA, per-head QK-norm before NEOX RoPE, head_dim from
 * metadata) with a mixture-of-experts FFN: a router over
 * qwen3moe.expert_count SwiGLU experts of width
 * qwen3moe.expert_feed_forward_length, the expert_used_count best per
 * token, their weights renormalised to sum 1 (Qwen3-MoE's norm_topk_prob;
 * llama.cpp builds the same graph for this architecture).
 *
 *   h_post_attn = h_in + o_proj(attention(rope(qk_norm(qkv(norm(h_in))))))
 *   h_out       = blk_moe_ffn(h_post_attn)
 *
 * The experts run on CPU backends: blk_moe_ffn routes and gathers rows on
 * the host. A GPU backend is refused at load until it has a placement that
 * keeps the gathers off the host.
 *
 * Removing the family is this file, its registry lines in arch_family.c and
 * arch_family.h, and its line in mk/common.mk.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../rotation.h"
#include "../weight_load.h"

#include "gguf_reader.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void populate_qwen3moe(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    st->config.has_qk_norms   = true;
    st->config.ffn_activation = GEIST_FFN_SWIGLU;
    st->config.moe_norm_topk  = true;

    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "qwen3moe.block_count", &u))
        st->n_layers = u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.expert_count", &u))
        st->config.moe_n_experts = u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.expert_used_count", &u))
        st->config.moe_n_used = u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.vocab_size", &u)) {
        st->vocab_size = u;
    } else {
        uint32_t       elem_vt = 0;
        uint64_t       count   = 0;
        const uint8_t *payload = nullptr;
        if (gguf_get_meta_array_info(gguf, "tokenizer.ggml.tokens", &elem_vt, &count, &payload)) {
            st->vocab_size = (size_t) count;
        }
    }
    if (gguf_get_meta_f32(gguf, "qwen3moe.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    bool b;
    if (gguf_get_meta_bool(gguf, "qwen3moe.expert_weights_norm", &b))
        st->config.moe_norm_topk = b; /* absent: Qwen3-MoE's norm_topk_prob */

    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static bool populate_layers_qwen3moe(struct transformer_arch_state *st) {
    struct gguf_ctx *g        = (struct gguf_ctx *) st->gguf;
    uint32_t         head_dim = 0, expert_ff = 0;
    float            freq_base = 1000000.0f;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "qwen3moe.attention.key_length", &head_dim);
        gguf_get_meta_u32(g, "qwen3moe.expert_feed_forward_length", &expert_ff);
        gguf_get_meta_f32(g, "qwen3moe.rope.freq_base", &freq_base);
    }
    if (head_dim == 0 && st->n_q_heads > 0) {
        head_dim = (uint32_t) (st->d_model / st->n_q_heads);
    }
    /* A router needs at least one expert to pick and no more than exist. */
    if (head_dim == 0 || expert_ff == 0 || st->config.moe_n_experts == 0 ||
        st->config.moe_n_used == 0 || st->config.moe_n_used > st->config.moe_n_experts ||
        st->config.moe_n_experts > GEIST_MOE_MAX_EXPERTS) {
        return false;
    }
    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        L->layer_idx                        = (int) i;
        L->is_full                          = true;
        L->is_kv_shared                     = false;
        L->head_dim                         = head_dim;
        L->q_out                            = st->n_q_heads * head_dim;
        L->kv_out                           = st->n_kv_heads * head_dim;
        L->intermediate                     = expert_ff; /* one expert's width */
        L->sliding_window                   = 0;
        L->rope_theta                       = freq_base;
        L->n_rotated_dims                   = (int) head_dim;
    }
    return true;
}

/* The router and the three expert tensors of one layer. */
[[nodiscard]] static enum geist_status wire_layer_qwen3moe(struct transformer_arch_state    *st,
                                                           struct gguf_ctx                  *gguf,
                                                           struct transformer_layer_weights *L) {
    struct geist_backend *be = st->backend;
    if (strncmp(be->desc->name, "cpu_", 4) != 0) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "qwen3moe: the experts run on CPU backends only, not %s (#48)",
                                be->desc->name);
        return GEIST_E_UNSUPPORTED;
    }
    const size_t n_exp = st->config.moe_n_experts;
    const size_t d     = st->d_model;
    const size_t ff    = L->intermediate;
    char         name[64];

    /* A router this unit does not implement is refused, never approximated:
     * llama.cpp's gating enum (1 = softmax) and a shared expert. */
    uint32_t gating = 1;
    if (gguf_get_meta_u32(gguf, "qwen3moe.expert_gating_func", &gating) && gating != 1) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "qwen3moe: expert_gating_func %u is not softmax (1)",
                                gating);
        return GEIST_E_UNSUPPORTED;
    }
    snprintf(name, sizeof name, "blk.%d.ffn_gate_inp_shexp.weight", L->layer_idx);
    if (gguf_get_tensor(gguf, name) != nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "qwen3moe: shared expert '%s' is not supported", name);
        return GEIST_E_UNSUPPORTED;
    }

    /* The router's top-k is a discrete choice: it keeps its own precision. */
    snprintf(name, sizeof name, "blk.%d.ffn_gate_inp.weight", L->layer_idx);
    enum geist_status s =
            load_layer_matrix(st, gguf, L, name, n_exp, d, true, &L->moe_router, &L->moe_router_w);
    if (s == GEIST_OK) {
        snprintf(name, sizeof name, "blk.%d.ffn_gate_exps.weight", L->layer_idx);
        s = load_layer_experts(st, gguf, L, name, n_exp, ff, d, &L->moe_gate, &L->moe_gate_w);
    }
    if (s == GEIST_OK) {
        snprintf(name, sizeof name, "blk.%d.ffn_up_exps.weight", L->layer_idx);
        s = load_layer_experts(st, gguf, L, name, n_exp, ff, d, &L->moe_up, &L->moe_up_w);
    }
    if (s == GEIST_OK) {
        snprintf(name, sizeof name, "blk.%d.ffn_down_exps.weight", L->layer_idx);
        s = load_layer_experts(st, gguf, L, name, n_exp, d, ff, &L->moe_down, &L->moe_down_w);
    }
    return s;
}

/* Qwen3's attention, unchanged. Each step runs only while s is GEIST_OK. */
[[nodiscard]] static enum geist_status
qwen3moe_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const size_t                      d    = st->d_model;
    struct geist_buffer              *x    = sess->scratch_normed;

    enum geist_status s = blk_rmsnorm(ctx, d, ctx->h_in_buf, &L->attn_norm, x);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, d, x);
    if (s == GEIST_OK)
        s = blk_qkv(ctx, x);
    if (s == GEIST_OK)
        s = blk_qk_norm(ctx, &L->q_norm, &L->k_norm);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, 1.0f / sqrtf((float) ctx->hd));
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    if (s == GEIST_OK)
        s = blk_o_proj(ctx);
    if (s == GEIST_OK)
        s = blk_add(ctx, d, ctx->h_in_buf, sess->scratch_o, sess->scratch_h_post_attn);
    return s;
}

const struct transformer_family transformer_family_qwen3moe = {
        .name            = "qwen3moe",
        .populate        = populate_qwen3moe,
        .populate_layers = populate_layers_qwen3moe,
        .wire_layer      = wire_layer_qwen3moe,
        .attention_block = qwen3moe_attention_block,
        .ffn_block       = blk_moe_ffn,
};
