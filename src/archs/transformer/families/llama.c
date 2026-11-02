/*
 * src/archs/transformer/families/llama.c - the Llama family unit.
 *
 * Layer: ARCHITECTURE.
 *
 * Llama 2/3 and models converted as "llama" (e.g. SmolLM2): uniform
 * pre-norm stack, GQA, SwiGLU, full attention, no KV sharing, no PLE.
 * Interleaved (NORM) RoPE: the converter pre-permutes Q/K, which the
 * loader undoes from config.rope_interleaved.
 *
 *   h_post_attn = h_in + o_proj(attention(rope(qkv(norm(h_in)))))
 *   h_out       = blk_ffn_swiglu(h_post_attn)
 *
 * Mixtral (#241) is the same stack under the same architecture name with a
 * mixture-of-experts FFN: transformer_family_select takes the variant below
 * when the file has llama.expert_count. llama.feed_forward_length is one
 * expert's width, and the selected weights are renormalised, as in
 * llama.cpp's graph for this architecture.
 *
 *   h_out       = blk_moe_ffn(h_post_attn)
 *
 * Removing the family is this file, its registry lines in arch_family.c and
 * arch_family.h, and its line in mk/common.mk.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../weight_load.h"

#include "gguf_reader.h"

#include <math.h>
#include <stdio.h>

/* ---- Llama ------------------------------------------------------------- */

static void populate_llama(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Neutral config from state_create covers Llama; the two family
     * specifics: SwiGLU (already the neutral default, stated for
     * clarity) and interleaved RoPE (weights are pre-permuted). */
    st->config.ffn_activation   = GEIST_FFN_SWIGLU;
    st->config.rope_interleaved = true; /* LLAMA_ARCH = NORM RoPE */

    family_populate_common(gguf, "llama", nullptr, st);
    st->vocab_size = family_vocab_size(gguf, "llama", nullptr);
}

static bool populate_layers_llama(struct transformer_arch_state *st) {
    /* Llama is uniform: all layers full-attention, no KV sharing,
     * head_dim derived from d_model / n_q_heads. RoPE rotates the
     * full head_dim (no partial rotation like Gemma's 25% on full
     * layers). FFN intermediate is uniform: llama.feed_forward_length. */
    struct gguf_ctx *g            = (struct gguf_ctx *) st->gguf;
    const size_t     head_dim     = (st->n_q_heads > 0) ? (st->d_model / st->n_q_heads) : 0;
    uint32_t         intermediate = 0;
    float            rope_theta   = 100000.0f; /* llama.rope.freq_base overrides */
    if (g != nullptr) {
        gguf_get_meta_u32(g, "llama.feed_forward_length", &intermediate);
        gguf_get_meta_f32(g, "llama.rope.freq_base", &rope_theta);
    }
    family_uniform_layers(st, head_dim, intermediate, rope_theta);
    return true;
}

/* ---- Forward ---------------------------------------------------------- */

[[nodiscard]] static enum geist_status
llama_attention_block(struct transformer_layer_forward_ctx *ctx) {
    return blk_attention(ctx, nullptr, nullptr, 1.0f / sqrtf((float) ctx->hd), nullptr, true);
}

/* ---- Mixtral ---------------------------------------------------------- */

static void populate_llama_moe(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    populate_llama(gguf, st);
    family_populate_moe(gguf, "llama", st);
}

static bool populate_layers_llama_moe(struct transformer_arch_state *st) {
    return family_moe_ok(&st->config) && st->n_q_heads > 0 && populate_layers_llama(st) &&
           st->layers[0].intermediate > 0;
}

[[nodiscard]] static enum geist_status wire_layer_llama_moe(struct transformer_arch_state    *st,
                                                            struct gguf_ctx                  *gguf,
                                                            struct transformer_layer_weights *L) {
    /* One tensor per expert is a layout llama.cpp no longer reads either;
     * the load path and the GPU's aliased expert buffers want one tensor. */
    char name[64];
    snprintf(name, sizeof name, "blk.%d.ffn_gate.0.weight", L->layer_idx);
    if (gguf_get_tensor(gguf, name) != nullptr) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "llama: '%s' stores one tensor per expert; rewrite the file with "
                                "tools/gguf_merge_experts.py",
                                name);
        return GEIST_E_UNSUPPORTED;
    }
    return moe_wire_layer(st, gguf, L, "llama");
}

/* ---- Registry entries -------------------------------------------------- */

const struct transformer_family transformer_family_llama = {
        .name                 = "llama",
        .populate             = populate_llama,
        .populate_layers      = populate_layers_llama,
        .fused_residual_norms = true,
        .attention_block      = llama_attention_block,
        .ffn_block            = blk_ffn_swiglu,
};

const struct transformer_family transformer_family_llama_moe = {
        .name            = "llama",
        .populate        = populate_llama_moe,
        .populate_layers = populate_layers_llama_moe,
        .wire_layer      = wire_layer_llama_moe,
        .session_alloc   = moe_session_alloc,
        .session_free    = moe_session_free,
        .attention_block = llama_attention_block,
        .ffn_block       = blk_moe_ffn,
        /* As qwen3moe: routing amplifies KV quantization error. */
        .kv_auto         = GEIST_KV_F16,
        .kv_int4_instead = GEIST_KV_F16,
};
