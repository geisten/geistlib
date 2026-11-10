/*
 * src/archs/transformer/families/phimoe.c - the Phi-MoE family unit
 * (GGUF architecture "phimoe": Phi-mini-MoE, Phi-3.5-MoE; #523).
 *
 * Layer: ARCHITECTURE.
 *
 * Microsoft's PhiMoEForCausalLM: GQA attention with biases on q, k, v and
 * o, NEOX RoPE over the whole head, scale 1 / sqrt(head_dim), and a
 * mixture-of-experts SwiGLU FFN of expert_count experts, two per token.
 * Three things no other family has:
 *
 *   - the attention, FFN and output norms are torch.nn.LayerNorm with a
 *     bias (config.norm_layernorm, prims->layernorm), not RMSNorms;
 *   - the lm_head has a bias (config.lm_head_bias);
 *   - the router is SparseMixer's inference rule
 *     (GEIST_MOE_GATING_SPARSEMIXER, moe_route): the two largest logits,
 *     each weighted by its softmax among the logits within 2 * 0.01 of it,
 *     relative; the two weights are independent and usually both 1.
 *
 *   h_post_attn = h_in + o_proj(attend(rope(qkv(layernorm(h_in)) + b))) + b_o
 *   h_out       = h_post_attn + sum_{e in top2} w_e * expert_e(layernorm(h_post_attn))
 *   logits      = lm_head(layernorm(h_n)) + b_out
 *
 * This is the reference implementation's graph (modeling_slimmoe.py of
 * microsoft/Phi-mini-MoE-instruct: nn.LayerNorm, sparsemixer() in eval
 * mode), not llama.cpp's for the architecture. llama.cpp (models/phi3.cpp
 * at d81235049, unchanged on master 2026-11-10) runs the three norms as
 * RMSNorm plus bias and weights the two experts by the softmax over all,
 * renormalised to sum 1. On Phi-mini-MoE Q4_K_M (wikitext-2, 8 x 512, M1
 * Max, Metal) llama.cpp scores perplexity 8.983 and this unit run with
 * llama.cpp's two choices 8.993 (+0.12 %); as written here it scores 6.792.
 * The router is nearly all of that (8.934 with only the router changed,
 * 6.811 with only the norms). tests/test_phimoe_unit.c holds both graphs
 * in double.
 *
 * Not loaded: a file with LongRoPE factors (blk.0.rope_factors_long,
 * Phi-3.5-MoE's 128k context). attention.sliding_window (2047) is not
 * applied, as in llama.cpp and in the reference's default (SDPA) attention;
 * the reference's eager attention masks keys more than 2047 back.
 *
 * Removing the family is this file and its line in family_list.h
 * (make check-static builds it alone, #345).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../weight_load.h"
#include "../weight_load/internal.h"

#include "gguf_reader.h"

#include <math.h>
#include <stdio.h>

static const char ARCH[] = "phimoe";

static void populate_phimoe(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    struct geist_arch_config *c = &st->config;
    c->ffn_activation           = GEIST_FFN_SWIGLU;
    c->norm_layernorm           = true;
    c->lm_head_bias             = gguf_get_tensor(gguf, "output.bias") != nullptr;
    family_populate_common(gguf, ARCH, nullptr, st);
    st->vocab_size = family_vocab_size(gguf, ARCH, nullptr);
    family_populate_moe(gguf, ARCH, st);
    c->moe_norm_topk = false; /* SparseMixer's weights are final */
    c->moe_gating    = GEIST_MOE_GATING_SPARSEMIXER;
}

static bool populate_layers_phimoe(struct transformer_arch_state *st) {
    struct gguf_ctx *g         = (struct gguf_ctx *) st->gguf;
    const uint32_t   head_dim  = family_head_dim(st, ARCH);
    uint32_t         expert_ff = 0;
    float            freq_base = 10000.0f;
    if (g == nullptr || gguf_get_tensor(g, "blk.0.rope_factors_long.weight") != nullptr) {
        return false; /* LongRoPE (Phi-3.5-MoE) */
    }
    (void) family_meta_u32(g, ARCH, nullptr, "feed_forward_length", &expert_ff);
    (void) family_meta_f32(g, ARCH, nullptr, "rope.freq_base", &freq_base);
    /* SparseMixer is defined for two experts per token. */
    if (head_dim == 0 || expert_ff == 0 || st->config.moe_n_used != 2 ||
        !family_moe_ok(&st->config)) {
        return false;
    }
    family_uniform_layers(st, head_dim, expert_ff /* one expert's width */, freq_base);
    return true;
}

[[nodiscard]] static enum geist_status wire_layer_phimoe(struct transformer_arch_state    *st,
                                                         struct gguf_ctx                  *gguf,
                                                         struct transformer_layer_weights *L) {
    const size_t d = st->d_model;
    char         name[64];
    const struct {
        const char          *suffix;
        size_t               n;
        struct geist_tensor *out;
    } vecs[] = {
            {"attn_norm.bias", d, &L->attn_norm_b},
            {"ffn_norm.bias", d, &L->ffn_norm_b},
            {"attn_q.bias", L->q_out, &L->q_bias},
            {"attn_k.bias", L->kv_out, &L->k_bias},
            {"attn_v.bias", L->kv_out, &L->v_bias},
            {"attn_output.bias", d, &L->o_bias},
    };
    enum geist_status s = GEIST_OK;
    for (size_t i = 0; i < sizeof vecs / sizeof vecs[0] && s == GEIST_OK; i++) {
        snprintf(name, sizeof name, "blk.%d.%s", L->layer_idx, vecs[i].suffix);
        s = load_layer_norm(st, gguf, L, name, vecs[i].n, vecs[i].out);
    }
    return s == GEIST_OK ? moe_wire_layer(st, gguf, L, ARCH) : s;
}

/* Each step runs only while s is GEIST_OK. */
[[nodiscard]] static enum geist_status
phimoe_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_layer_weights *L    = ctx->L;
    struct transformer_arch_session  *sess = ctx->sess;

    enum geist_status s = blk_attn_in(ctx, nullptr, nullptr); /* the LayerNorm: attn_norm_b */
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->q_out, &L->q_bias, sess->scratch_q);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->kv_out, &L->k_bias, sess->scratch_k);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->kv_out, &L->v_bias, sess->scratch_v);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, 1.0f / sqrtf((float) ctx->hd));
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    if (s == GEIST_OK)
        s = blk_o_proj(ctx);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->st->d_model, &L->o_bias, sess->scratch_o);
    return s == GEIST_OK ? blk_attn_residual(ctx, nullptr) : s;
}

TRANSFORMER_FAMILY(phimoe,
                   populate_phimoe,
                   populate_layers_phimoe,
                   phimoe_attention_block,
                   blk_moe_ffn, /* the LayerNorm: ffn_norm_b */
                   .wire_layer    = wire_layer_phimoe,
                   .session_alloc = moe_session_alloc,
                   .session_free  = moe_session_free,
                   /* As the other MoE families: routing amplifies KV quantization. */
                   .kv_auto         = GEIST_KV_F16,
                   .kv_int4_instead = GEIST_KV_F16);
