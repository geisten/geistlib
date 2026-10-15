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

    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "llama.block_count", &u))
        st->n_layers = u;
    if (gguf_get_meta_u32(gguf, "llama.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "llama.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "llama.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_u32(gguf, "llama.vocab_size", &u))
        st->vocab_size = u;
    if (gguf_get_meta_f32(gguf, "llama.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    /* Llama doesn't have PLE — zero out the PLE-derived dims so any
     * accidental access traps on size==0. */
    st->hidden_per_layer = 0;
    st->ple_out          = 0;
}

static bool populate_layers_llama(struct transformer_arch_state *st) {
    /* Llama is uniform: all layers full-attention, no KV sharing,
     * head_dim derived from d_model / n_q_heads. RoPE rotates the
     * full head_dim (no partial rotation like Gemma's 25% on full
     * layers). FFN intermediate is uniform: llama.feed_forward_length. */
    const size_t head_dim     = (st->n_q_heads > 0) ? (st->d_model / st->n_q_heads) : 0;
    uint32_t     intermediate = 0;
    if (st->gguf != nullptr) {
        gguf_get_meta_u32((struct gguf_ctx *) st->gguf, "llama.feed_forward_length", &intermediate);
    }
    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        L->layer_idx                        = (int) i;
        L->is_full                          = true;
        L->is_kv_shared                     = false;
        L->head_dim                         = head_dim;
        L->q_out                            = st->n_q_heads * head_dim;
        L->kv_out                           = st->n_kv_heads * head_dim;
        L->intermediate                     = intermediate;
        L->sliding_window                   = 0;
        L->rope_theta                       = 100000.0f; /* default; freq_base overrides below */
        L->n_rotated_dims                   = (int) head_dim;
    }
    /* RoPE theta override from llama.rope.freq_base. */
    if (st->gguf != nullptr) {
        float freq_base;
        if (gguf_get_meta_f32((struct gguf_ctx *) st->gguf, "llama.rope.freq_base", &freq_base)) {
            for (size_t i = 0; i < st->n_layers; i++) {
                st->layers[i].rope_theta = freq_base;
            }
        }
    }
    return true;
}

/* ---- Forward ---------------------------------------------------------- */

/* Each step runs only while s is GEIST_OK, so the first failure is what
 * the stage returns and nothing after it touches a buffer. */
[[nodiscard]] static enum geist_status
llama_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const size_t                      d    = st->d_model;
    const size_t                      q    = ctx->q_out;
    struct geist_buffer              *x    = sess->scratch_normed;
    struct geist_buffer              *attn = sess->scratch_attn;

    enum geist_status s = blk_rmsnorm(ctx, d, ctx->h_in_buf, &L->attn_norm, x);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, d, x);
    if (s == GEIST_OK)
        s = blk_qkv(ctx, x);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, 1.0f / sqrtf((float) ctx->hd));
    if (s == GEIST_OK)
        s = blk_attend(ctx);
    if (s == GEIST_OK)
        blk_scale_cols(ctx, q, L->o_awq_inv_scale, attn);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, q, attn);
    if (s == GEIST_OK)
        s = blk_linear(ctx, attn, &L->o_proj, &L->o_proj_w, sess->scratch_o);
    if (s == GEIST_OK)
        s = blk_add(ctx, d, ctx->h_in_buf, sess->scratch_o, sess->scratch_h_post_attn);
    return s;
}

/* ---- Registry entry --------------------------------------------------- */

const struct transformer_family transformer_family_llama = {
        .name            = "llama",
        .populate        = populate_llama,
        .populate_layers = populate_layers_llama,
        .attention_block = llama_attention_block,
        .ffn_block       = blk_ffn_swiglu,
};
