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
 * Removing the family is this file, its registry lines in arch_family.c and
 * arch_family.h, and its line in mk/common.mk.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../rotation.h"

#include "gguf_reader.h"

#include <math.h>

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

/* Each step runs only while s is GEIST_OK, so the first failure is what
 * the stage returns and nothing after it touches a buffer. */
[[nodiscard]] static enum geist_status
llama_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state   *st   = ctx->st;
    struct transformer_arch_session *sess = ctx->sess;
    const size_t                     d    = st->d_model;
    struct geist_buffer             *x    = sess->scratch_normed;

    enum geist_status s = blk_attn_norm(ctx, x);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, d, x);
    if (s == GEIST_OK)
        s = blk_qkv(ctx, x);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, 1.0f / sqrtf((float) ctx->hd));
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    if (s == GEIST_OK)
        s = blk_o_proj(ctx);
    if (s == GEIST_OK)
        s = blk_attn_residual(ctx, nullptr);
    return s;
}

/* ---- Registry entry --------------------------------------------------- */

const struct transformer_family transformer_family_llama = {
        .name                 = "llama",
        .populate             = populate_llama,
        .populate_layers      = populate_layers_llama,
        .fused_residual_norms = true,
        .attention_block      = llama_attention_block,
        .ffn_block            = blk_ffn_swiglu,
};
