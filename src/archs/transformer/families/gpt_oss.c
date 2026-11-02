/*
 * src/archs/transformer/families/gpt_oss.c - the GPT-OSS family unit (#243).
 *
 * Layer: ARCHITECTURE.
 *
 * OpenAI's gpt-oss-20b / -120b, as llama.cpp's graph for the architecture
 * "gpt-oss" runs them:
 *   - attention: GQA, head_dim from attention.key_length, q/k/v/o with
 *     biases, NEOX RoPE with YaRN, scale 1/sqrt(head_dim), and a learned
 *     per-head sink logit in every softmax (attn_sinks);
 *   - layers alternate a sliding window (even layers, attention.sliding_window
 *     keys including the token itself) and full attention;
 *   - a mixture-of-experts FFN after the RMSNorm the file calls
 *     post_attention_norm: the router (with bias) picks the
 *     expert_used_count largest logits and weights them by the softmax over
 *     those alone; each expert (biases after gate, up and down) runs
 *     llama.cpp's swiglu_oai.
 *
 *   h_post_attn = h_in + o_proj(attend_sinks(rope(qkv(norm(h_in)) + b))) + b_o
 *   h_out       = blk_moe_ffn(h_post_attn)
 *
 * Removing the family is this file, its registry lines in arch_family.c and
 * arch_family.h, and its line in mk/common.mk.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../weight_load.h"
#include "../weight_load/internal.h"

#include "gguf_dequant.h"
#include "gguf_reader.h"

#include <math.h>
#include <stdio.h>

static const char ARCH[] = "gpt-oss";

static void populate_gpt_oss(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    st->config.ffn_activation = GEIST_FFN_SWIGLU;
    family_populate_common(gguf, ARCH, nullptr, st);
    st->vocab_size = family_vocab_size(gguf, ARCH, nullptr);
    family_populate_moe(gguf, ARCH, st);
    st->config.moe_norm_topk = false; /* the weights are a softmax already */
    st->config.moe_gating    = GEIST_MOE_GATING_SOFTMAX_WEIGHT;
    st->config.moe_act       = GEIST_MOE_ACT_SWIGLU_OAI;
}

static bool populate_layers_gpt_oss(struct transformer_arch_state *st) {
    struct gguf_ctx *g         = (struct gguf_ctx *) st->gguf;
    const uint32_t   head_dim  = family_head_dim(st, ARCH);
    uint32_t         expert_ff = 0, window = 0;
    float            freq_base = 10000.0f;
    if (g == nullptr || !family_populate_yarn(g, ARCH, st)) {
        return false;
    }
    (void) family_meta_u32(g, ARCH, nullptr, "expert_feed_forward_length", &expert_ff);
    (void) family_meta_u32(g, ARCH, nullptr, "attention.sliding_window", &window);
    (void) family_meta_f32(g, ARCH, nullptr, "rope.freq_base", &freq_base);
    if (head_dim == 0 || expert_ff == 0 || window == 0 || !family_moe_ok(&st->config)) {
        return false;
    }
    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        family_uniform_layer(st, i, head_dim, expert_ff, freq_base, (int) head_dim, L);
        /* llama.cpp's swa pattern 2: even layers slide, odd ones see all. */
        L->is_full        = i % 2 == 1;
        L->sliding_window = L->is_full ? 0 : window;
    }
    return true;
}

/* A [n] F32 host copy of tensor `suffix` of layer L, or nullptr (error set). */
static float *host_vector(struct transformer_arch_state          *st,
                          struct gguf_ctx                        *gguf,
                          const struct transformer_layer_weights *L,
                          const char                             *suffix,
                          size_t                                  n) {
    char name[64];
    snprintf(name, sizeof name, "blk.%d.%s", L->layer_idx, suffix);
    const struct gguf_tensor_t *t = gguf_get_tensor(gguf, name);
    float *out = t != nullptr && gguf_tensor_elem_count(t) == n ? gguf_dequant_to_fp32(t) : nullptr;
    if (out == nullptr) {
        geist_backend_set_error(
                st->backend, GEIST_E_FORMAT, "%s: '%s' is missing or not [%zu]", ARCH, name, n);
    }
    return out;
}

[[nodiscard]] static enum geist_status wire_layer_gpt_oss(struct transformer_arch_state    *st,
                                                          struct gguf_ctx                  *gguf,
                                                          struct transformer_layer_weights *L) {
    char         name[64];
    const size_t d     = st->d_model;
    const size_t n_exp = st->config.moe_n_experts;
    const struct {
        const char          *suffix;
        size_t               n;
        struct geist_tensor *out;
    } vecs[] = {
            /* The norm before the FFN; the generic loader's ffn_norm slot. */
            {"post_attention_norm.weight", d, &L->ffn_norm},
            {"attn_q.bias", L->q_out, &L->q_bias},
            {"attn_k.bias", L->kv_out, &L->k_bias},
            {"attn_v.bias", L->kv_out, &L->v_bias},
            {"attn_output.bias", d, &L->o_bias},
            {"attn_sinks.weight", st->n_q_heads, &L->attn_sinks_t},
    };
    enum geist_status s = GEIST_OK;
    for (size_t i = 0; i < sizeof vecs / sizeof vecs[0] && s == GEIST_OK; i++) {
        snprintf(name, sizeof name, "blk.%d.%s", L->layer_idx, vecs[i].suffix);
        s = load_layer_norm(st, gguf, L, name, vecs[i].n, vecs[i].out);
    }
    if (s == GEIST_OK)
        s = moe_wire_layer(st, gguf, L, ARCH);
    if (s != GEIST_OK)
        return s;
    L->attn_sinks   = host_vector(st, gguf, L, "attn_sinks.weight", st->n_q_heads);
    L->moe_router_b = host_vector(st, gguf, L, "ffn_gate_inp.bias", n_exp);
    L->moe_gate_b   = host_vector(st, gguf, L, "ffn_gate_exps.bias", n_exp * L->intermediate);
    L->moe_up_b     = host_vector(st, gguf, L, "ffn_up_exps.bias", n_exp * L->intermediate);
    L->moe_down_b   = host_vector(st, gguf, L, "ffn_down_exps.bias", n_exp * d);
    if (L->attn_sinks == nullptr || L->moe_router_b == nullptr || L->moe_gate_b == nullptr ||
        L->moe_up_b == nullptr || L->moe_down_b == nullptr) {
        return GEIST_E_FORMAT;
    }
    return GEIST_OK;
}

/* Each step runs only while s is GEIST_OK. */
[[nodiscard]] static enum geist_status
gpt_oss_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_layer_weights *L    = ctx->L;
    struct transformer_arch_session  *sess = ctx->sess;

    enum geist_status s = blk_attn_in(ctx, nullptr, nullptr);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->q_out, &L->q_bias, sess->scratch_q);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->kv_out, &L->k_bias, sess->scratch_k);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->kv_out, &L->v_bias, sess->scratch_v);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, 1.0f / sqrtf((float) ctx->hd));
    if (s == GEIST_OK)
        s = blk_attend(ctx, true); /* L->attn_sinks: the sink path */
    if (s == GEIST_OK)
        s = blk_o_proj(ctx);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->st->d_model, &L->o_bias, sess->scratch_o);
    return s == GEIST_OK ? blk_attn_residual(ctx, nullptr) : s;
}

const struct transformer_family transformer_family_gpt_oss = {
        .name            = "gpt-oss",
        .populate        = populate_gpt_oss,
        .populate_layers = populate_layers_gpt_oss,
        .wire_layer      = wire_layer_gpt_oss,
        .session_alloc   = moe_session_alloc,
        .session_free    = moe_session_free,
        .attention_block = gpt_oss_attention_block,
        .ffn_block       = blk_moe_ffn,
        /* The sinks run on an F32 or F16 cache only (kv_store.c); F16 as the
         * other MoE families. */
        .kv_auto         = GEIST_KV_F16,
        .kv_int4_instead = GEIST_KV_F16,
};
