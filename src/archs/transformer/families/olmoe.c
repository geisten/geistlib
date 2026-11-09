/*
 * src/archs/transformer/families/olmoe.c - the OLMoE family unit (#241).
 *
 * Layer: ARCHITECTURE.
 *
 * Plain multi-head attention (no GQA in OLMoE-1B-7B, NEOX RoPE, head_dim
 * d_model / n_heads) with a mixture-of-experts FFN. Two things differ from
 * Qwen3-MoE, both as in llama.cpp's graph for this architecture:
 *   - the QK norms run over a token's whole q and k rows, before the split
 *     into heads, so their weights have d_model elements;
 *   - the selected experts keep their softmax probabilities over all
 *     experts: the weights are not renormalised and do not sum to 1.
 * The expert width is olmoe.feed_forward_length; the file has no key_length
 * and no expert_weights_norm.
 *
 *   h_post_attn = h_in + o_proj(attend(rope(qk_norm(qkv(rot(attn_norm(h_in)))))))
 *   h_out       = blk_moe_ffn(h_post_attn)
 *
 * Removing the family is this file and its line in family_list.h
 * (make check-static builds it alone, #345).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../weight_load.h"

#include "gguf_reader.h"

#include <math.h>
#include <stdio.h>

static void populate_olmoe(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    st->config.has_qk_norms      = true;
    st->config.qk_norm_whole_row = true;
    st->config.ffn_activation    = GEIST_FFN_SWIGLU;
    st->config.moe_norm_topk     = false;

    family_populate_common(gguf, "olmoe", nullptr, st);
    st->vocab_size = family_vocab_size(gguf, "olmoe", nullptr);
    uint32_t u;
    if (gguf_get_meta_u32(gguf, "olmoe.expert_count", &u))
        st->config.moe_n_experts = u;
    if (gguf_get_meta_u32(gguf, "olmoe.expert_used_count", &u))
        st->config.moe_n_used = u;
}

static bool populate_layers_olmoe(struct transformer_arch_state *st) {
    struct gguf_ctx *g         = (struct gguf_ctx *) st->gguf;
    const size_t     head_dim  = st->n_q_heads > 0 ? st->d_model / st->n_q_heads : 0;
    uint32_t         expert_ff = 0;
    float            freq_base = 10000.0f;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "olmoe.feed_forward_length", &expert_ff);
        gguf_get_meta_f32(g, "olmoe.rope.freq_base", &freq_base);
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

[[nodiscard]] static enum geist_status wire_layer_olmoe(struct transformer_arch_state    *st,
                                                        struct gguf_ctx                  *gguf,
                                                        struct transformer_layer_weights *L) {
    return moe_wire_layer(st, gguf, L, "olmoe");
}

/* blk_attention with the QK norms over whole rows. Each step runs only
 * while s is GEIST_OK. After the norms, the plan's fused prep (#514) does
 * both RoPEs, the q scale and the KV append in one call where the backend
 * has it, as blk_attention does for Llama (its per-head norms stay off:
 * these ran over the whole row). */
[[nodiscard]] static enum geist_status
olmoe_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_layer_weights *L       = ctx->L;
    const float                       q_scale = 1.0f / sqrtf((float) ctx->hd);
    const bool                        fused =
            ctx->P->fuse_attn_rope_append && !ctx->kv_kivi_enabled && !ctx->kv_int8_enabled;

    enum geist_status s = blk_attn_in(ctx, nullptr, nullptr);
    if (s == GEIST_OK)
        s = blk_qk_norm_whole_row(ctx, &L->q_norm, &L->k_norm);
    if (s == GEIST_OK)
        s = fused ? blk_rope_append_fused(ctx, nullptr, nullptr, q_scale)
                  : blk_rope_qk(ctx, q_scale);
    if (s == GEIST_OK)
        s = blk_attend(ctx, !fused);
    return s == GEIST_OK ? blk_attn_out(ctx, nullptr) : s;
}

TRANSFORMER_FAMILY(olmoe,
                   populate_olmoe,
                   populate_layers_olmoe,
                   olmoe_attention_block,
                   blk_moe_ffn,
                   .wire_layer    = wire_layer_olmoe,
                   .session_alloc = moe_session_alloc,
                   .session_free  = moe_session_free,
                   /* As qwen3moe: routing amplifies KV quantization error. */
                   .kv_auto         = GEIST_KV_F16,
                   .kv_int4_instead = GEIST_KV_F16);
