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

    family_populate_common(gguf, "qwen3moe", nullptr, st);
    st->vocab_size = family_vocab_size(gguf, "qwen3moe", nullptr);
    uint32_t u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.expert_count", &u))
        st->config.moe_n_experts = u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.expert_used_count", &u))
        st->config.moe_n_used = u;
    bool b;
    if (gguf_get_meta_bool(gguf, "qwen3moe.expert_weights_norm", &b))
        st->config.moe_norm_topk = b; /* absent: Qwen3-MoE's norm_topk_prob */
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
    family_uniform_layers(st, head_dim, expert_ff /* one expert's width */, freq_base);
    return true;
}

/* The router and the three expert tensors of one layer. A shared expert
 * is qwen35moe's (#169), not this architecture's: refused. */
[[nodiscard]] static enum geist_status wire_layer_qwen3moe(struct transformer_arch_state    *st,
                                                           struct gguf_ctx                  *gguf,
                                                           struct transformer_layer_weights *L) {
    char name[64];
    snprintf(name, sizeof name, "blk.%d.ffn_gate_inp_shexp.weight", L->layer_idx);
    if (gguf_get_tensor(gguf, name) != nullptr) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "qwen3moe: shared expert '%s' is not supported",
                                name);
        return GEIST_E_UNSUPPORTED;
    }
    return moe_wire_layer(st, gguf, L, "qwen3moe");
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
        .session_alloc   = moe_session_alloc,
        .session_free    = moe_session_free,
        .attention_block = qwen3moe_attention_block,
        .ffn_block       = blk_moe_ffn,
        /* An MoE amplifies KV quantization error about 16x over a dense
         * model of the same quantization; F16 matches FP32 against the exact
         * reference (benchmark/results/QWEN3MOE.md). */
        .kv_auto = GEIST_KV_F16,
        /* INT4 misses the quality bar here (#53): Qwen3-30B-A3B Q4_K_M,
         * teacher-forced against FP32 KV on cpu_neon, mean JS 0.020 nats
         * and 89 % top-1 (INT8: 0.012, 91 %). */
        .kv_int4_instead = GEIST_KV_F16,
};
