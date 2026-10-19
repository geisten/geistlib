/*
 * src/archs/transformer/families/qwen3.c - the Qwen3 family unit.
 *
 * Layer: ARCHITECTURE.
 *
 * Llama-style stack (GQA, SwiGLU, full attention, NEOX RoPE) plus a
 * per-head QK-norm before RoPE, and a head_dim decoupled from d_model.
 *
 *   h_post_attn = h_in + o_proj(attention(rope(qk_norm(qkv(norm(h_in))))))
 *   h_out       = blk_ffn_swiglu(h_post_attn)
 *
 * The BitNet embedding models that ship as "qwen3" (an RMSNorm on every
 * projection's input, populate_bitnet_embedding_meta) run the same
 * composition: the blocks apply the per-projection norms and the norm on
 * o_proj's and down's input when the model has them.
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

/* ---- Qwen3 ------------------------------------------------------------- */
/*
 * Llama-style uniform stack (GQA, SwiGLU, full attention, no KV sharing,
 * no PLE, no softcap) with two deviations:
 *
 *   1. Per-head QK-norm: RMSNorm over [head_dim] on Q and K before RoPE
 *      (has_qk_norms — the Gemma V-norm / scale semantics do NOT apply,
 *      Qwen3 keeps the standard 1/sqrt(head_dim) scale).
 *   2. head_dim is decoupled from d_model: qwen3.attention.key_length
 *      (128 on 0.6B where d_model/n_heads would give 64), so
 *      q_out = n_heads * head_dim can EXCEED d_model. o_proj is
 *      [d_model, q_out] and scratch sizes from q_out — already the
 *      Gemma-4 full-attn shape, no special casing needed.
 *
 * RoPE is NEOX-style half-split like Gemma (no weight pre-permute), so
 * rope_interleaved stays false — unlike llama.
 */
static void populate_qwen3(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Neutral config + the two qwen3 specifics: per-head QK-norm and
     * SwiGLU. NEOX RoPE (no pre-permute) is the neutral default. */
    st->config.has_qk_norms   = true;
    st->config.ffn_activation = GEIST_FFN_SWIGLU;

    family_populate_common(gguf, "qwen3", nullptr, st);
    st->vocab_size = family_vocab_size(gguf, "qwen3", nullptr); /* Qwen GGUFs: no key */

    populate_bitnet_embedding_meta(gguf, "qwen3", st);
}

static bool populate_layers_qwen3(struct transformer_arch_state *st) {
    struct gguf_ctx *g = (struct gguf_ctx *) st->gguf;

    /* A pooling this build cannot perform is a load failure, not something
     * to fall back from: every fallback here would return a vector that
     * looks fine and means something else. */
    if (st->config.pooling_unsupported) {
        return false;
    }

    /* head_dim from metadata — NOT d_model / n_heads (0.6B: key_length
     * 128 vs quotient 64). Missing key on a geometry where the quotient
     * would be wrong is exactly the #258 failure mode, so fall back to
     * the quotient only as a last resort and fail on zero. */
    uint32_t head_dim = 0, intermediate = 0;
    float    freq_base = 1000000.0f;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "qwen3.attention.key_length", &head_dim);
        gguf_get_meta_u32(g, "qwen3.feed_forward_length", &intermediate);
        gguf_get_meta_f32(g, "qwen3.rope.freq_base", &freq_base);
    }
    if (head_dim == 0 && st->n_q_heads > 0) {
        head_dim = (uint32_t) (st->d_model / st->n_q_heads);
    }
    if (head_dim == 0 || intermediate == 0) {
        return false;
    }
    family_uniform_layers(st, head_dim, intermediate, freq_base);
    return true;
}

/* ---- Tensors ---------------------------------------------------------- */

/* The embedding variant's per-projection input norms, where it has them. */
[[nodiscard]] static enum geist_status qwen3_wire_layer(struct transformer_arch_state    *st,
                                                        struct gguf_ctx                  *gguf,
                                                        struct transformer_layer_weights *L) {
    if (!st->config.has_projection_input_norms) {
        return GEIST_OK;
    }
    return load_projection_input_norms(st, gguf, L);
}

/* ---- Forward ---------------------------------------------------------- */

/* Each step runs only while s is GEIST_OK, so the first failure is what
 * the stage returns and nothing after it touches a buffer. */
[[nodiscard]] static enum geist_status
qwen3_attention_block(struct transformer_layer_forward_ctx *ctx) {
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

/* ---- Registry entry --------------------------------------------------- */

const struct transformer_family transformer_family_qwen3 = {
        .name            = "qwen3",
        .populate        = populate_qwen3,
        .populate_layers = populate_layers_qwen3,
        .wire_layer      = qwen3_wire_layer,
        .attention_block = qwen3_attention_block,
        .ffn_block       = blk_ffn_swiglu,
};
