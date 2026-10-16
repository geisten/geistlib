/*
 * src/archs/transformer/families/gemma4.c - the Gemma 4 family unit.
 *
 * Layer: ARCHITECTURE.
 *
 * Gemma 4 (E2B, E4B): sandwich norms, per-head Q/K norms that carry the
 * attention scale, a scale-free V norm, sliding-window and full layers with
 * partial RoPE on the full ones, KV sharing (the last layers read an earlier
 * layer's cache and project only q), GEGLU, and per-layer embeddings (PLE)
 * with a per-layer output scale.
 *
 *   h_post_attn = h_in + post_attn_norm(o_proj(attention(prep(qkv(norm(h_in))))))
 *   h_post_ff   = blk_ffn_geglu(h_post_attn)
 *   h_out       = layer_scalar * PLE(h_post_ff)
 *
 * prep is the plan-bound fused op on GPU backends (norms + RoPE + KV append
 * in one dispatch), else blk_q_prep / blk_k_prep.
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
#include <stdio.h>

/* ---- Gemma 4 ----------------------------------------------------------- */

/* Fallback attention pattern for GGUFs without the
 * gemma4.attention.sliding_window_pattern key: the E2B layout — 4
 * sliding-window layers + 1 full-attention layer, repeated 7 times =
 * 35 layers. E4B (5+1 × 7 = 42 layers) and anything else MUST carry
 * the metadata key (#258). */
static const bool GEMMA4_LAYER_IS_FULL[] = {
        false, false, false, false, true,  false, false, false, false, true,  false, false,
        false, false, true,  false, false, false, false, true,  false, false, false, false,
        true,  false, false, false, false, true,  false, false, false, false, true,
};

static void populate_gemma4(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    /* Gemma-4 owns its defaults — state_create installs a NEUTRAL config
     * and each family declares what it HAS. Structural values are the
     * E2B geometry, kept as fallbacks for pre-key GGUFs; current unsloth
     * GGUFs override every one of them from metadata below. PLE scales
     * are architectural constants (not in GGUF metadata). */
    st->n_layers         = 35;
    st->d_model          = 1536;
    st->vocab_size       = 262144; /* no gemma4.vocab_size key exists */
    st->n_q_heads        = 8;
    st->n_kv_heads       = 1;
    st->hidden_per_layer = 256;

    st->config.logit_softcap        = 30.0f;
    st->config.has_embed_scale      = true;
    st->config.has_ple              = true;
    st->config.ple_input_scale      = 0.7071067811865476f;
    st->config.ple_model_proj_scale = 0.02551551815399144f;
    st->config.ple_table_scale      = 16.0f;
    /* E2B KV-share sources; populate_layers_gemma4 re-derives them
     * from the attention pattern (#258). */
    st->config.kv_sliding_src       = 13;
    st->config.kv_full_src          = 14;
    st->config.has_gemma_attn_norms = true;
    st->config.has_qk_norms         = true;
    st->config.ffn_activation       = GEIST_FFN_GEGLU;

    uint32_t u;
    float    f;
    if (gguf_get_meta_u32(gguf, "gemma4.block_count", &u))
        st->n_layers = u;
    if (gguf_get_meta_u32(gguf, "gemma4.embedding_length", &u))
        st->d_model = u;
    if (gguf_get_meta_u32(gguf, "gemma4.attention.head_count", &u))
        st->n_q_heads = u;
    if (gguf_get_meta_u32(gguf, "gemma4.attention.head_count_kv", &u))
        st->n_kv_heads = u;
    if (gguf_get_meta_u32(gguf, "gemma4.embedding_length_per_layer_input", &u))
        st->hidden_per_layer = u;
    st->ple_out = st->n_layers * st->hidden_per_layer;
    if (gguf_get_meta_f32(gguf, "gemma4.attention.layer_norm_rms_epsilon", &f))
        st->config.rms_eps = f;
    if (gguf_get_meta_f32(gguf, "gemma4.final_logit_softcapping", &f))
        st->config.logit_softcap = f;
}

static bool populate_layers_gemma4(struct transformer_arch_state *st) {
    struct gguf_ctx *g = (struct gguf_ctx *) st->gguf;

    /* Attention pattern: bool array, true = sliding-window layer. E2B
     * is 4+1 × 7 (35 layers), E4B is 5+1 × 7 (42 layers) — variant
     * geometry, so it must come from metadata. GGUFs without the key
     * fall back to the static E2B table; any other layer count without
     * the key is not derivable → fail (caller names the geometry). */
    uint32_t       pat_vt = 0;
    uint64_t       pat_n  = 0;
    const uint8_t *pat    = nullptr;
    if (g == nullptr ||
        !gguf_get_meta_array_info(
                g, "gemma4.attention.sliding_window_pattern", &pat_vt, &pat_n, &pat) ||
        pat_vt != GGUF_META_VT_BOOL || pat_n != st->n_layers) {
        pat = nullptr;
        if (st->n_layers != sizeof GEMMA4_LAYER_IS_FULL / sizeof GEMMA4_LAYER_IS_FULL[0]) {
            return false;
        }
    }

    /* KV sharing: the LAST shared_kv_layers layers alias earlier
     * caches. E2B: 35-20 → owners 0..14. E4B: 42-18 → owners 0..23. */
    uint32_t shared_kv = 20;
    if (g != nullptr)
        gguf_get_meta_u32(g, "gemma4.attention.shared_kv_layers", &shared_kv);
    if (shared_kv > st->n_layers) {
        return false;
    }
    const size_t kv_share_threshold = st->n_layers - shared_kv;

    /* Per-type head dims + sliding window + RoPE bases, E2B defaults. */
    uint32_t full_hd = 512, swa_hd = 256, sliding_window = 512;
    float    theta_full = 1000000.0f, theta_swa = 10000.0f;
    /* FFN inner dim: scalar key (uniform, E4B) or per-layer u32/i32
     * array (E2B: 6144 owners / 12288 shared). */
    uint32_t       ffn_uniform = 0;
    const uint8_t *ffn_arr     = nullptr;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "gemma4.attention.key_length", &full_hd);
        gguf_get_meta_u32(g, "gemma4.attention.key_length_swa", &swa_hd);
        gguf_get_meta_u32(g, "gemma4.attention.sliding_window", &sliding_window);
        gguf_get_meta_f32(g, "gemma4.rope.freq_base", &theta_full);
        gguf_get_meta_f32(g, "gemma4.rope.freq_base_swa", &theta_swa);
        uint32_t ffn_vt = 0;
        uint64_t ffn_n  = 0;
        if (!gguf_get_meta_u32(g, "gemma4.feed_forward_length", &ffn_uniform) &&
            gguf_get_meta_array_info(g, "gemma4.feed_forward_length", &ffn_vt, &ffn_n, &ffn_arr)) {
            if ((ffn_vt != GGUF_META_VT_U32 && ffn_vt != GGUF_META_VT_I32) ||
                ffn_n != st->n_layers) {
                ffn_arr = nullptr;
            }
        }
    }

    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        L->layer_idx                        = (int) i;
        L->is_full      = pat != nullptr ? pat[i] == 0 : GEMMA4_LAYER_IS_FULL[i];
        L->is_kv_shared = i >= kv_share_threshold;
        L->head_dim     = L->is_full ? full_hd : swa_hd;
        L->q_out        = st->n_q_heads * L->head_dim;
        L->kv_out       = st->n_kv_heads * L->head_dim;
        if (ffn_arr != nullptr) {
            uint32_t v;
            memcpy(&v, ffn_arr + i * sizeof v, sizeof v);
            L->intermediate = v;
        } else if (ffn_uniform > 0) {
            L->intermediate = ffn_uniform;
        } else {
            L->intermediate = L->is_kv_shared ? 12288 : 6144; /* pre-key E2B GGUFs */
        }
        L->sliding_window = L->is_full ? 0 : sliding_window;
        L->rope_theta     = L->is_full ? theta_full : theta_swa;
        /* Full-attn layers rotate 25% of head_dim (partial_rotary_factor
         * 0.25 — family constant; the GGUF's rope.dimension_count key
         * carries head_dim, not the rotated width, so it can't be used).
         * Sliding layers rotate the full head_dim. */
        L->n_rotated_dims = L->is_full ? (int) (full_hd / 4) : (int) L->head_dim;
    }

    /* KV-share sources: each shared layer reads the cache of the LAST
     * owner layer of its own attention type. E2B: full←14, sliding←13.
     * E4B: full←23, sliding←22. */
    int last_full = -1, last_sliding = -1;
    for (size_t i = 0; i < kv_share_threshold; i++) {
        if (st->layers[i].is_full)
            last_full = (int) i;
        else
            last_sliding = (int) i;
    }
    if (shared_kv > 0 && (last_full < 0 || last_sliding < 0)) {
        return false; /* shared layers with no owner of one type */
    }
    st->config.kv_full_src    = last_full;
    st->config.kv_sliding_src = last_sliding;
    return true;
}

/* ---- Forward ---------------------------------------------------------- */

/* Each step runs only while s is GEIST_OK, so the first failure is what
 * the stage returns and nothing after it touches a buffer. */
[[nodiscard]] static enum geist_status
gemma4_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const size_t                      d    = st->d_model;
    struct geist_buffer              *x    = sess->scratch_normed;
    /* The fused prep covers the plain (f32/f16) cache only. */
    const bool fused = ctx->P->fuse_attn_qkv_prep && !ctx->kv_kivi_enabled && !ctx->kv_int8_enabled;

    enum geist_status s = blk_rmsnorm(ctx, d, ctx->h_in_buf, &L->attn_norm, x);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, ctx->seq, d, x);
    if (s == GEIST_OK && ctx->compute_kv)
        s = blk_qkv(ctx, x);
    if (s == GEIST_OK && !ctx->compute_kv) /* KV-shared layer: q only */
        s = blk_linear(ctx, x, &L->q_proj, &L->q_proj_w, sess->scratch_q);
    if (s == GEIST_OK && fused)
        s = blk_qkv_prep_fused(ctx);
    if (s == GEIST_OK && !fused)
        s = blk_q_prep(ctx, &L->q_norm, 1.0f); /* the Q/K norms carry the scale */
    if (s == GEIST_OK && !fused && ctx->compute_kv)
        s = blk_k_prep(ctx, &L->k_norm, true);
    if (s == GEIST_OK)
        s = blk_attend(ctx, ctx->compute_kv && !fused);
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

/* FFN, then the per-layer embedding gate and the layer scale. Without a
 * per-layer input the FFN writes h_out directly and PLE is skipped. */
[[nodiscard]] static enum geist_status gemma4_ffn_block(struct transformer_layer_forward_ctx *ctx) {
    struct geist_buffer *out = ctx->run_ple ? ctx->sess->scratch_h_post_ff : ctx->h_out_buf;
    enum geist_status    s   = blk_ffn_geglu(ctx, out);
    if (s == GEIST_OK)
        s = transformer_layer_run_ple_or_copy(ctx);
    if (s == GEIST_OK)
        s = transformer_layer_scale_output(ctx);
    return s;
}

/* ---- Registry entry --------------------------------------------------- */

const struct transformer_family transformer_family_gemma4 = {
        .name            = "gemma4",
        .towers          = true, /* the only towers that exist: 1536-dim soft tokens */
        .populate        = populate_gemma4,
        .populate_layers = populate_layers_gemma4,
        .attention_block = gemma4_attention_block,
        .ffn_block       = gemma4_ffn_block,
};
