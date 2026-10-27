/*
 * src/archs/transformer/families/gemma3.c - the Gemma 3 family unit.
 *
 * Layer: ARCHITECTURE.
 *
 * Stock Gemma 3 (e.g. google/gemma-3-270m-it) and the BitNet embedding 270M
 * built on it. Sandwich norms (pre and post around attention and FFN),
 * per-head QK-norm, queries scaled by query_pre_attn_scalar^-1/2 (V is not
 * normalised, unlike Gemma 4), GEGLU, sqrt(d_model) embedding scale, and
 * five local sliding-window layers for every global one.
 *
 *   h_post_attn = h_in + post_attn_norm(o_proj(attention(rope(qk_norm(qkv(norm(h_in)))))))
 *   h_out       = blk_ffn_geglu(h_post_attn)
 *
 * The embedding variant's per-projection input norms are applied by the
 * blocks (blk_qkv, blk_gate_up, blk_o_proj, blk_down) when the model has them.
 *
 * Removing the family is this file, its registry lines in arch_family.c and
 * arch_family.h, and its line in mk/common.mk.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../weight_load/internal.h"

#include "gguf_reader.h"

#include <math.h>

/* ---- gemma3 (BitNet-embedding-270M's backbone) -------------------------- */
/*
 * Gemma 3 text, as the BitNet embedding 270M ships it. Structurally it sits
 * between llama and gemma4: it has Gemma's extra residual norms (post-attn,
 * post-FFW) and QK-norms and the sqrt(d_model) embedding scale, but NO
 * per-layer embeddings and no KV sharing. A stock (generative) Gemma 3 also
 * alternates local sliding-window and global layers (populate_layers_gemma3).
 *
 * Two geometry notes, both of which the default derivations get wrong:
 *   - head_dim is 256 while d_model/n_heads is 640/4 = 160, so key_length
 *     must come from metadata (the same trap qwen3 documents).
 *   - the queries are scaled by query_pre_attn_scalar^-1/2 (1/16 on the
 *     270M) and V is not normalized: unlike Gemma 4, whose Q/K norms carry
 *     the scale (config.gemma3_q_scale, #674).
 *
 * Scope: the BitNet embedding 270M (test_published_embedding_e2e) and the
 * stock google/gemma-3-270m-it (perplexity against llama.cpp, #674).
 */
static void populate_gemma3(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    st->config.has_gemma_attn_norms = true;
    st->config.has_qk_norms         = true;
    st->config.has_embed_scale      = true;
    st->config.ffn_activation       = GEIST_FFN_GEGLU;

    family_populate_common(gguf, "gemma3", nullptr, st);
    st->vocab_size = family_vocab_size(gguf, "gemma3", nullptr);

    populate_bitnet_embedding_meta(gguf, "gemma3", st);
}

static bool populate_layers_gemma3(struct transformer_arch_state *st) {
    struct gguf_ctx *g = (struct gguf_ctx *) st->gguf;

    if (st->config.pooling_unsupported) {
        return false;
    }

    uint32_t intermediate = 0;
    float    freq_base    = 10000.0f;
    /* Stock Gemma 3 (#674): with attention.sliding_window set, five of every
     * six layers are local — that window and the local RoPE base
     * rope.freq_base_swa (10000) — and every sixth is global with
     * rope.freq_base (1e6 on the 270M). llama.cpp's gemma3 reads the same
     * keys with the same defaults. Running every layer global with the
     * global base gave the 270M a perplexity of 3489 against llama.cpp's
     * 45. Embedding models keep the all-global stack the BitNet embedding
     * 270M was verified with (test_published_embedding_e2e). */
    uint32_t sliding = 0, swa_period = 6;
    float    freq_base_swa = 10000.0f;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "gemma3.feed_forward_length", &intermediate);
        gguf_get_meta_f32(g, "gemma3.rope.freq_base", &freq_base);
        if (st->config.pooling == GEIST_POOLING_NONE) {
            gguf_get_meta_u32(g, "gemma3.attention.sliding_window", &sliding);
            gguf_get_meta_u32(g, "gemma3.attention.sliding_window_pattern", &swa_period);
            gguf_get_meta_f32(g, "gemma3.rope.freq_base_swa", &freq_base_swa);
        }
    }
    /* The quotient only as a last resort: for the 270M it is 160 and the
     * real head_dim is 256, so a GGUF without the key is the #258 failure
     * mode. */
    const uint32_t head_dim = family_head_dim(st, "gemma3");
    if (head_dim == 0 || intermediate == 0) {
        return false;
    }

    /* query_pre_attn_scalar: head_dim, except the 27B's d_model / n_heads —
     * the rule llama.cpp's gemma3 hard-codes (the GGUF has no key). */
    const float qpas          = st->n_layers == 62 && st->n_q_heads > 0
                                        ? (float) st->d_model / (float) st->n_q_heads
                                        : (float) head_dim;
    st->config.gemma3_q_scale = 1.0f / sqrtf(qpas);
    for (size_t i = 0; i < st->n_layers; i++) {
        /* llama.cpp's set_swa_pattern: local unless il % period is the
         * period's last slot; period 0 means every layer local */
        const bool local = sliding > 0 && (swa_period == 0 || i % swa_period < swa_period - 1);
        struct transformer_layer_weights *L = &st->layers[i];
        family_uniform_layer(st,
                             i,
                             head_dim,
                             intermediate,
                             local ? freq_base_swa : freq_base,
                             (int) head_dim,
                             L);
        L->is_full        = !local;
        L->sliding_window = local ? sliding : 0;
    }
    return true;
}

/* ---- Tensors ---------------------------------------------------------- */

/* The sandwich norms, and the embedding variant's per-projection input
 * norms where it has them. */
[[nodiscard]] static enum geist_status gemma3_wire_layer(struct transformer_arch_state    *st,
                                                         struct gguf_ctx                  *gguf,
                                                         struct transformer_layer_weights *L) {
    enum geist_status s = load_sandwich_norms(st, gguf, L);
    if (s == GEIST_OK && st->config.has_projection_input_norms) {
        s = load_projection_input_norms(st, gguf, L);
    }
    return s;
}

/* ---- Forward ---------------------------------------------------------- */

[[nodiscard]] static enum geist_status
gemma3_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_layer_weights *L = ctx->L;
    return blk_attention(
            ctx, &L->q_norm, &L->k_norm, ctx->st->config.gemma3_q_scale, &L->post_attn_norm, false);
}

/* No PLE stage in this family: the FFN writes h_out directly. */
[[nodiscard]] static enum geist_status gemma3_ffn_block(struct transformer_layer_forward_ctx *ctx) {
    return blk_ffn_geglu(ctx, ctx->h_out_buf);
}

/* ---- Registry entry --------------------------------------------------- */

const struct transformer_family transformer_family_gemma3 = {
        .name                 = "gemma3",
        .populate             = populate_gemma3,
        .populate_layers      = populate_layers_gemma3,
        .wire_layer           = gemma3_wire_layer,
        .fused_residual_norms = true,
        .attention_block      = gemma3_attention_block,
        .ffn_block            = gemma3_ffn_block,
};
