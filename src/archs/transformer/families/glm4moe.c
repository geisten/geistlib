/*
 * src/archs/transformer/families/glm4moe.c - the GLM-4.5 family unit
 * (GGUF architecture "glm4moe": GLM-4.5, GLM-4.5-Air, GLM-4.6, #242).
 *
 * Layer: ARCHITECTURE.
 *
 * As llama.cpp's graph for the architecture (models/glm4-moe.cpp) runs it:
 *
 *   q, k, v     = attn_{q,k,v}(attn_norm(h_in)) + their biases
 *   q, k        = per-head RMSNorm when the file has attn_{q,k}_norm (GLM-4.5;
 *                 the Air has none), then NEOX RoPE on the first
 *                 rope.dimension_count dims of each head (half of head_dim),
 *                 ggml's partial layout (config.rope_partial_block)
 *   h_post_attn = h_in + o_proj(attend(q, k, v)),  scale 1 / sqrt(head_dim)
 *   x           = post_attention_norm(h_post_attn)  (the pre-FFN norm)
 *   h_out       = h_post_attn + dense SwiGLU(x)            (the first
 *                 leading_dense_block_count layers)
 *               = h_post_attn + experts(x) + shared(x)     (the others)
 *
 * The router is DeepSeek-V3's without groups: sigmoid scores, exp_probs_b
 * shifting the selection only, the selected weights renormalised
 * (expert_weights_norm) and scaled (expert_weights_scale), moe_route. The
 * shared experts are one SwiGLU of expert_shared_count experts' width,
 * weight 1. llama.cpp's defaults where the file is silent: sigmoid, no
 * renormalisation, no scale; it reads no expert groups for this
 * architecture, so neither does this.
 *
 * block_count includes the NextN (MTP) layers, nextn_predict_layers of
 * them at the end. They are not loaded: the stack is the layers before.
 * The M-RoPE variant (GLM-4.5V, rope.dimension_sections) is refused.
 *
 * Removing the family is this file and its line in family_list.h
 * (make check-static builds it alone, #345).
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

static const char ARCH[] = "glm4moe";

/* Layers before this one are dense. */
static size_t n_dense_lead(const struct transformer_arch_state *st) {
    uint32_t n = 0;
    if (st->gguf != nullptr)
        (void) family_meta_u32(
                (struct gguf_ctx *) st->gguf, ARCH, nullptr, "leading_dense_block_count", &n);
    return n;
}

static void populate_glm4moe(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    struct geist_arch_config *c = &st->config;
    c->ffn_activation           = GEIST_FFN_SWIGLU;
    c->rope_partial_block       = true; /* ggml's partial NEOX layout */
    c->has_qk_norms             = gguf_get_tensor(gguf, "blk.0.attn_q_norm.weight") != nullptr;
    family_populate_common(gguf, ARCH, nullptr, st);
    st->vocab_size = family_vocab_size(gguf, ARCH, nullptr);
    family_populate_moe(gguf, ARCH, st);

    /* The NextN layers sit at the end of block_count; the stack stops
     * before them. More than block_count leaves 0 layers, refused. */
    uint32_t u = 0;
    if (family_meta_u32(gguf, ARCH, nullptr, "nextn_predict_layers", &u))
        st->n_layers = u < st->n_layers ? st->n_layers - u : 0;

    bool norm = false;
    char key[96];
    snprintf(key, sizeof key, "%s.expert_weights_norm", ARCH);
    c->moe_norm_topk = gguf_get_meta_bool(gguf, key, &norm) && norm;
    c->moe_gating    = GEIST_MOE_GATING_SIGMOID;
    if (family_meta_u32(gguf, ARCH, nullptr, "expert_gating_func", &u) && u != 0)
        c->moe_gating = (enum geist_moe_gating) u;
    (void) family_meta_f32(gguf, ARCH, nullptr, "expert_weights_scale", &c->moe_w_scale);
    c->moe_n_shared = 0; /* the shared experts run as one dense FFN */
}

static bool populate_layers_glm4moe(struct transformer_arch_state *st) {
    struct gguf_ctx *g  = (struct gguf_ctx *) st->gguf;
    const uint32_t   hd = family_head_dim(st, ARCH);
    uint32_t         ff = 0, eff = 0, rot = 0, n_shared = 0;
    float            base = 10000.0f;
    char             key[96];
    if (g == nullptr)
        return false;
    uint32_t       vt         = 0;
    uint64_t       n_sections = 0;
    const uint8_t *payload    = nullptr;
    snprintf(key, sizeof key, "%s.rope.dimension_sections", ARCH);
    if (gguf_get_meta_array_info(g, key, &vt, &n_sections, &payload))
        return false; /* M-RoPE (GLM-4.5V) */
    (void) family_meta_u32(g, ARCH, nullptr, "feed_forward_length", &ff);
    (void) family_meta_u32(g, ARCH, nullptr, "expert_feed_forward_length", &eff);
    (void) family_meta_f32(g, ARCH, nullptr, "rope.freq_base", &base);
    if (!family_meta_u32(g, ARCH, nullptr, "rope.dimension_count", &rot))
        rot = hd;
    (void) family_meta_u32(g, ARCH, nullptr, "expert_shared_count", &n_shared);
    const size_t lead = n_dense_lead(st);
    /* The shared experts run in the session's FFN scratch, which the
     * widest layer sizes. */
    const size_t widest = lead > 0 && ff > eff ? ff : eff;
    if (hd == 0 || rot == 0 || rot > hd || rot % 2 != 0 || ff == 0 || eff == 0 ||
        st->n_layers == 0 || lead > st->n_layers || (size_t) n_shared * eff > widest ||
        !family_moe_ok(&st->config)) {
        return false;
    }
    for (size_t i = 0; i < st->n_layers; i++)
        family_uniform_layer(st, i, hd, i < lead ? ff : eff, base, (int) rot, &st->layers[i]);
    return true;
}

/* Tensor `suffix` of layer L as a [n] F32 host vector, or nullptr with the
 * error set and *s GEIST_E_FORMAT. */
static float *host_vector(struct transformer_arch_state          *st,
                          struct gguf_ctx                        *gguf,
                          const struct transformer_layer_weights *L,
                          const char                             *suffix,
                          size_t                                  n,
                          enum geist_status                      *s) {
    char name[64];
    snprintf(name, sizeof name, "blk.%d.%s", L->layer_idx, suffix);
    const struct gguf_tensor_t *t = gguf_get_tensor(gguf, name);
    float *out = t != nullptr && gguf_tensor_elem_count(t) == n ? gguf_dequant_to_fp32(t) : nullptr;
    if (out == nullptr) {
        geist_backend_set_error(
                st->backend, GEIST_E_FORMAT, "%s: '%s' is missing or not [%zu]", ARCH, name, n);
        *s = GEIST_E_FORMAT;
    }
    return out;
}

[[nodiscard]] static enum geist_status wire_layer_glm4moe(struct transformer_arch_state    *st,
                                                          struct gguf_ctx                  *gguf,
                                                          struct transformer_layer_weights *L) {
    const size_t d = st->d_model;
    char         name[64];
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
    };
    enum geist_status s = GEIST_OK;
    for (size_t i = 0; i < sizeof vecs / sizeof vecs[0] && s == GEIST_OK; i++) {
        snprintf(name, sizeof name, "blk.%d.%s", L->layer_idx, vecs[i].suffix);
        s = load_layer_norm(st, gguf, L, name, vecs[i].n, vecs[i].out);
    }
    if (s != GEIST_OK)
        return s;
    /* gate, up, down: the dense FFN's, or the shared experts'. */
    const bool  dense = (size_t) L->layer_idx < n_dense_lead(st);
    const char *sfx   = dense ? "" : "_shexp";
    size_t      ff    = L->intermediate;
    if (!dense) {
        uint32_t n_shared = 0;
        (void) family_meta_u32(gguf, ARCH, nullptr, "expert_shared_count", &n_shared);
        ff = (size_t) n_shared * L->intermediate;
        s  = moe_wire_layer(st, gguf, L, ARCH);
        if (s == GEIST_OK)
            L->moe_sel_b =
                    host_vector(st, gguf, L, "exp_probs_b.bias", st->config.moe_n_experts, &s);
    }
    if (s != GEIST_OK || ff == 0)
        return s;
    const struct {
        const char          *what;
        size_t               n_out, n_in;
        struct geist_tensor *t;
        struct geist_weight *w;
    } ffn[] = {
            {"ffn_gate", ff, d, &L->gate_proj, &L->gate_proj_w},
            {"ffn_up", ff, d, &L->up_proj, &L->up_proj_w},
            {"ffn_down", d, ff, &L->down_proj, &L->down_proj_w},
    };
    for (size_t i = 0; i < sizeof ffn / sizeof ffn[0] && s == GEIST_OK; i++) {
        snprintf(name, sizeof name, "blk.%d.%s%s.weight", L->layer_idx, ffn[i].what, sfx);
        s = load_layer_matrix(
                st, gguf, L, name, ffn[i].n_out, ffn[i].n_in, false, ffn[i].t, ffn[i].w);
    }
    return s;
}

/* Each step runs only while s is GEIST_OK. */
[[nodiscard]] static enum geist_status
glm4moe_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_layer_weights *L    = ctx->L;
    struct transformer_arch_session  *sess = ctx->sess;

    enum geist_status s = blk_attn_in(ctx, nullptr, nullptr);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->q_out, &L->q_bias, sess->scratch_q);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->kv_out, &L->k_bias, sess->scratch_k);
    if (s == GEIST_OK)
        s = blk_add_bias(ctx, ctx->kv_out, &L->v_bias, sess->scratch_v);
    if (s == GEIST_OK && ctx->st->config.has_qk_norms)
        s = blk_qk_norm(ctx, &L->q_norm, &L->k_norm);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, 1.0f / sqrtf((float) ctx->hd));
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    return s == GEIST_OK ? blk_attn_out(ctx, nullptr) : s;
}

[[nodiscard]] static enum geist_status
glm4moe_ffn_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    if (L->moe_gate_w == nullptr)
        return blk_ffn_swiglu(ctx);
    enum geist_status s = blk_moe_ffn(ctx); /* h_out = h_post_attn + experts */
    if (s != GEIST_OK || L->gate_proj.buffer == nullptr)
        return s;
    /* The shared experts on blk_moe_ffn's normed input, weight 1. */
    ctx->inter = (size_t) L->gate_proj.shape[0];
    s          = blk_gate_up(ctx, sess->scratch_pre_ff);
    if (s == GEIST_OK)
        s = blk_silu_mul(ctx, ctx->inter, sess->scratch_gate, sess->scratch_up);
    if (s == GEIST_OK)
        s = blk_down(ctx, sess->scratch_gate, nullptr);
    if (s == GEIST_OK)
        s = blk_add(ctx, ctx->st->d_model, ctx->h_out_buf, sess->scratch_ffn_out, ctx->h_out_buf);
    ctx->inter = L->intermediate;
    return s;
}

TRANSFORMER_FAMILY(glm4moe,
                   populate_glm4moe,
                   populate_layers_glm4moe,
                   glm4moe_attention_block,
                   glm4moe_ffn_block,
                   .wire_layer    = wire_layer_glm4moe,
                   .session_alloc = moe_session_alloc,
                   .session_free  = moe_session_free,
                   /* As the other MoE families: routing amplifies KV quantization. */
                   .kv_auto         = GEIST_KV_F16,
                   .kv_int4_instead = GEIST_KV_F16);
