/*
 * src/archs/transformer/families/granitemoe.c - the Granite-MoE family unit
 * (#241).
 *
 * Layer: ARCHITECTURE.
 *
 * Granite 3.0's A400M and A800M: the stack, RoPE and FFN of Mixtral (GQA,
 * interleaved RoPE, renormalised softmax top-k), plus four multipliers from
 * the file, as in llama.cpp's graph for this architecture:
 *
 *   h_0         = embedding_scale * token_embd(id)
 *   h_post_attn = h_in + residual_scale * o_proj(attend(attention_scale, ...))
 *   h_out       = h_post_attn + residual_scale * sum(experts)
 *   logits      = lm_head(output_norm(h_n)) / logit_scale
 *
 * Only the attention scale reaches the forward path. The other three are
 * load-time constants, so no backend and no fused path needs to know them:
 *   - every consumer of h is an RMSNorm, and with z = h / r that norm gives
 *     the same result when its epsilon is eps / r^2. So z_0 = h_0 / r with
 *     the residual scale gone from both sums is exactly the model above:
 *     residual_scale moves into the embedding multiplier and the epsilon;
 *   - lm_head is linear, so the logit divisor moves into output_norm's
 *     weights.
 *
 * The shared expert of GraniteMoeShared is refused: no published model has
 * one. Removing the family is this file and its line in family_list.h
 * (make check-static builds it alone, #345).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../weight_load.h"

#include "gguf_reader.h"

#include <math.h>

static void populate_granitemoe(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    st->config.ffn_activation   = GEIST_FFN_SWIGLU;
    st->config.rope_interleaved = true; /* the converter is Llama's */

    family_populate_common(gguf, "granitemoe", nullptr, st);
    st->vocab_size = family_vocab_size(gguf, "granitemoe", nullptr);
    family_populate_moe(gguf, "granitemoe", st);
}

/* 0 or absent means "no scale", as in llama.cpp. */
[[nodiscard]] static bool scale_or_default(struct gguf_ctx *g, const char *key, float *scale) {
    float f = 0.0f;
    if (!family_meta_f32(g, "granitemoe", nullptr, key, &f) || f == 0.0f)
        return true;
    *scale = f;
    return isfinite(f) && f > 0.0f;
}

static bool populate_layers_granitemoe(struct transformer_arch_state *st) {
    struct gguf_ctx *g        = (struct gguf_ctx *) st->gguf;
    const size_t     head_dim = st->n_q_heads > 0 ? st->d_model / st->n_q_heads : 0;
    if (g == nullptr || head_dim == 0 || !family_moe_ok(&st->config)) {
        return false;
    }
    uint32_t expert_ff = 0, shared_ff = 0;
    float    freq_base = 10000.0f;
    (void) family_meta_u32(g, "granitemoe", nullptr, "feed_forward_length", &expert_ff);
    (void) family_meta_u32(
            g, "granitemoe", nullptr, "expert_shared_feed_forward_length", &shared_ff);
    (void) family_meta_f32(g, "granitemoe", nullptr, "rope.freq_base", &freq_base);

    float attention = 1.0f / sqrtf((float) head_dim);
    float embedding = 1.0f, residual = 1.0f, logit = 0.0f;
    if (expert_ff == 0 || shared_ff != 0 || !scale_or_default(g, "attention.scale", &attention) ||
        !scale_or_default(g, "embedding_scale", &embedding) ||
        !scale_or_default(g, "residual_scale", &residual) ||
        !scale_or_default(g, "logit_scale", &logit) || logit == 0.0f) {
        return false;
    }
    st->config.attn_q_scale = attention;
    st->config.embed_scale  = embedding / residual;
    st->config.rms_eps /= residual * residual;
    st->config.output_norm_scale = 1.0f / logit;
    family_uniform_layers(st, head_dim, expert_ff /* one expert's width */, freq_base);
    return true;
}

[[nodiscard]] static enum geist_status wire_layer_granitemoe(struct transformer_arch_state    *st,
                                                             struct gguf_ctx                  *gguf,
                                                             struct transformer_layer_weights *L) {
    return moe_wire_layer(st, gguf, L, "granitemoe");
}

[[nodiscard]] static enum geist_status
granitemoe_attention_block(struct transformer_layer_forward_ctx *ctx) {
    return blk_attention(ctx, nullptr, nullptr, ctx->st->config.attn_q_scale, nullptr, true);
}

TRANSFORMER_FAMILY(granitemoe,
                   populate_granitemoe,
                   populate_layers_granitemoe,
                   granitemoe_attention_block,
                   blk_moe_ffn,
                   .wire_layer    = wire_layer_granitemoe,
                   .session_alloc = moe_session_alloc,
                   .session_free  = moe_session_free,
                   /* As qwen3moe: routing amplifies KV quantization error. */
                   .kv_auto         = GEIST_KV_F16,
                   .kv_int4_instead = GEIST_KV_F16);
