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
#include "../rotation.h"

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

    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "gemma3.block_count", &u))
        st->n_layers = u;
    if (gguf_get_meta_u32(gguf, "gemma3.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "gemma3.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "gemma3.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_u32(gguf, "gemma3.vocab_size", &u)) {
        st->vocab_size = u;
    } else {
        uint32_t       elem_vt = 0;
        uint64_t       count   = 0;
        const uint8_t *payload = nullptr;
        if (gguf_get_meta_array_info(gguf, "tokenizer.ggml.tokens", &elem_vt, &count, &payload)) {
            st->vocab_size = (size_t) count;
        }
    }
    if (gguf_get_meta_f32(gguf, "gemma3.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;

    populate_bitnet_embedding_meta(gguf, "gemma3", st);

    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static bool populate_layers_gemma3(struct transformer_arch_state *st) {
    struct gguf_ctx *g = (struct gguf_ctx *) st->gguf;

    if (st->config.pooling_unsupported) {
        return false;
    }

    uint32_t head_dim = 0, intermediate = 0;
    float    freq_base = 10000.0f;
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
        gguf_get_meta_u32(g, "gemma3.attention.key_length", &head_dim);
        gguf_get_meta_u32(g, "gemma3.feed_forward_length", &intermediate);
        gguf_get_meta_f32(g, "gemma3.rope.freq_base", &freq_base);
        if (st->config.pooling == GEIST_POOLING_NONE) {
            gguf_get_meta_u32(g, "gemma3.attention.sliding_window", &sliding);
            gguf_get_meta_u32(g, "gemma3.attention.sliding_window_pattern", &swa_period);
            gguf_get_meta_f32(g, "gemma3.rope.freq_base_swa", &freq_base_swa);
        }
    }
    /* Only as a last resort: for the 270M the quotient is 160 and the real
     * head_dim is 256, so a GGUF without the key is the #258 failure mode. */
    if (head_dim == 0 && st->n_q_heads > 0) {
        head_dim = (uint32_t) (st->d_model / st->n_q_heads);
    }
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
        struct transformer_layer_weights *L = &st->layers[i];
        L->layer_idx                        = (int) i;
        /* llama.cpp's set_swa_pattern: local unless il % period is the
         * period's last slot; period 0 means every layer local */
        const bool local  = sliding > 0 && (swa_period == 0 || i % swa_period < swa_period - 1);
        L->mixer          = GEIST_MIXER_ATTN;
        L->is_full        = !local;
        L->is_kv_shared   = false;
        L->head_dim       = head_dim;
        L->q_out          = st->n_q_heads * head_dim;
        L->kv_out         = st->n_kv_heads * head_dim;
        L->intermediate   = intermediate;
        L->sliding_window = local ? sliding : 0;
        L->rope_theta     = local ? freq_base_swa : freq_base;
        L->n_rotated_dims = (int) head_dim;
    }
    return true;
}

/* ---- Forward ---------------------------------------------------------- */

/* Each step runs only while s is GEIST_OK, so the first failure is what
 * the stage returns and nothing after it touches a buffer. */
[[nodiscard]] static enum geist_status
gemma3_attention_block(struct transformer_layer_forward_ctx *ctx) {
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
        s = blk_rope_qk(ctx, st->config.gemma3_q_scale);
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    if (s == GEIST_OK)
        s = blk_o_proj(ctx);
    if (s == GEIST_OK)
        s = blk_norm_add(ctx,
                         ctx->P->fuse_attn_rmsnorm_add,
                         &L->post_attn_norm,
                         sess->scratch_o,
                         ctx->h_in_buf,
                         sess->scratch_post_attn,
                         sess->scratch_h_post_attn);
    return s;
}

/* No PLE stage in this family: the FFN writes h_out directly. */
[[nodiscard]] static enum geist_status gemma3_ffn_block(struct transformer_layer_forward_ctx *ctx) {
    return blk_ffn_geglu(ctx, ctx->h_out_buf);
}

/* ---- Registry entry --------------------------------------------------- */

const struct transformer_family transformer_family_gemma3 = {
        .name            = "gemma3",
        .populate        = populate_gemma3,
        .populate_layers = populate_layers_gemma3,
        .attention_block = gemma3_attention_block,
        .ffn_block       = gemma3_ffn_block,
};
