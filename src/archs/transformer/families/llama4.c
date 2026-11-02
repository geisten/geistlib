/*
 * src/archs/transformer/families/llama4.c - the Llama 4 family unit (GGUF
 * architecture "llama4", #244).
 *
 * Layer: ARCHITECTURE.
 *
 * Llama 4 Scout / Maverick as llama.cpp's graph for the architecture runs
 * them (llama4.cpp):
 *   - attention: GQA, interleaved (NORM) RoPE with the file's
 *     rope_freqs.weight frequency factors, scale 1/sqrt(head_dim); with
 *     attention.sliding_window absent (Scout and Maverick) every fourth
 *     layer (index 3, 7, ...) has no RoPE and sees the whole context, its q
 *     scaled by llama.cpp's temperature log(floor((pos + 1) / 8192) + 1) *
 *     0.1 + 1, the other layers attend within chunks of 8192 keys
 *     (attention.sliding_window_pattern, default 4); a file whose
 *     sliding_window is 0 ropes and attends fully everywhere. On RoPE
 *     layers of a model with fewer than 128 experts, q and k are
 *     RMS-normalised without a weight after RoPE;
 *   - FFN: every interleave_moe_layer_step-th layer (1-based) is a mixture
 *     of experts, the rest dense SwiGLU. The router picks the
 *     expert_used_count largest logits; each picked expert runs on its
 *     input row times sigmoid(logit) and adds its output unweighted
 *     (llama.cpp's weight_before_ffn); one shared expert of the experts'
 *     width adds with weight 1.
 *
 *   h_post_attn = h_in + o_proj(attend(qk_norm?(rope?(qkv(attn_norm(h_in))))))
 *   h_out       = dense: blk_ffn_swiglu;  else: blk_moe_ffn + shared(ffn_norm(h_post_attn))
 *
 * attention.chunk_size and attention.temperature_length, read when present,
 * override llama.cpp's fixed 8192 (no converter writes them; the unit test
 * shrinks the chunk to reach it).
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
#include "heap.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static const char ARCH[] = "llama4";

/* llama.cpp's constants for the chunked layers and the NoPE temperature. */
static constexpr uint32_t CHUNK = 8192, NO_ROPE_STEP = 4;

struct llama4_meta {
    uint32_t moe_step;    /* MoE layer when (i + 1) % moe_step == 0; 0: none */
    uint32_t chunk;       /* 0: no chunked layers, RoPE everywhere */
    uint32_t pattern;     /* chunked unless i % pattern == pattern - 1 */
    uint32_t temp_length; /* the temperature's floor scale */
    bool     kq_norm;
};

static struct llama4_meta meta(const struct transformer_arch_state *st) {
    struct gguf_ctx   *g      = (struct gguf_ctx *) st->gguf;
    struct llama4_meta m      = {.chunk = CHUNK, .pattern = NO_ROPE_STEP, .temp_length = CHUNK};
    uint32_t           window = 1;
    if (g == nullptr)
        return m;
    (void) family_meta_u32(g, ARCH, nullptr, "interleave_moe_layer_step", &m.moe_step);
    if (family_meta_u32(g, ARCH, nullptr, "attention.sliding_window", &window) && window == 0)
        m.chunk = 0;
    (void) family_meta_u32(g, ARCH, nullptr, "attention.sliding_window_pattern", &m.pattern);
    (void) family_meta_u32(g, ARCH, nullptr, "attention.chunk_size", &m.chunk);
    (void) family_meta_u32(g, ARCH, nullptr, "attention.temperature_length", &m.temp_length);
    m.kq_norm = st->config.moe_n_experts != 128;
    return m;
}

/* Layer li has RoPE (and attends in chunks): llama.cpp's use_rope and its
 * swa pattern coincide for this architecture. */
static bool rope_layer(const struct llama4_meta *m, size_t li) {
    return m->chunk == 0 || m->pattern == 0 || li % m->pattern != m->pattern - 1;
}

static bool moe_layer(const struct llama4_meta *m, size_t li) {
    return m->moe_step > 0 && (li + 1) % m->moe_step == 0;
}

static void populate_llama4(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    struct geist_arch_config *c = &st->config;
    c->ffn_activation           = GEIST_FFN_SWIGLU;
    c->rope_interleaved         = true; /* LLAMA_ROPE_TYPE_NORM */
    family_populate_common(gguf, ARCH, nullptr, st);
    st->vocab_size = family_vocab_size(gguf, ARCH, nullptr);
    family_populate_moe(gguf, ARCH, st);
    c->moe_norm_topk     = false;
    c->moe_gating        = GEIST_MOE_GATING_SIGMOID;
    c->moe_weight_before = true;
    c->moe_n_shared      = 0; /* the shared expert runs as a dense FFN */
    (void) family_meta_f32(gguf, ARCH, nullptr, "expert_weights_scale", &c->moe_w_scale);
    /* The rope frequency factors, one per pair of a head. */
    const struct gguf_tensor_t *t = gguf_get_tensor(gguf, "rope_freqs.weight");
    if (t != nullptr && (st->rope_freq_factors = gguf_dequant_to_fp32(t)) != nullptr)
        st->n_rope_freq_factors = gguf_tensor_elem_count(t);
}

static bool populate_layers_llama4(struct transformer_arch_state *st) {
    struct gguf_ctx         *g  = (struct gguf_ctx *) st->gguf;
    const uint32_t           hd = family_head_dim(st, ARCH);
    const struct llama4_meta m  = meta(st);
    uint32_t                 ff = 0, eff = 0;
    float                    base = 500000.0f;
    if (g == nullptr)
        return false;
    (void) family_meta_u32(g, ARCH, nullptr, "feed_forward_length", &ff);
    (void) family_meta_u32(g, ARCH, nullptr, "expert_feed_forward_length", &eff);
    (void) family_meta_f32(g, ARCH, nullptr, "rope.freq_base", &base);
    if (hd == 0 || eff == 0 || m.moe_step == 0 || (m.moe_step > 1 && ff == 0) ||
        !family_moe_ok(&st->config) || (m.chunk > 0 && m.temp_length == 0) ||
        (st->rope_freq_factors != nullptr && st->n_rope_freq_factors * 2 != hd)) {
        return false;
    }
    for (size_t i = 0; i < st->n_layers; i++) {
        struct transformer_layer_weights *L = &st->layers[i];
        family_uniform_layer(st, i, hd, moe_layer(&m, i) ? eff : ff, base, (int) hd, L);
        L->is_full    = !rope_layer(&m, i) || m.chunk == 0;
        L->attn_chunk = L->is_full ? 0 : m.chunk;
    }
    return true;
}

[[nodiscard]] static enum geist_status wire_layer_llama4(struct transformer_arch_state    *st,
                                                         struct gguf_ctx                  *gguf,
                                                         struct transformer_layer_weights *L) {
    const struct llama4_meta m = meta(st);
    const size_t             d = st->d_model;
    char                     name[64];
#define T(suffix) (snprintf(name, sizeof name, "blk.%d." suffix, L->layer_idx), name)
    enum geist_status s = GEIST_OK;
    if (!moe_layer(&m, (size_t) L->layer_idx)) {
        s = load_layer_matrix(st,
                              gguf,
                              L,
                              T("ffn_gate.weight"),
                              L->intermediate,
                              d,
                              false,
                              &L->gate_proj,
                              &L->gate_proj_w);
        if (s == GEIST_OK)
            s = load_layer_matrix(st,
                                  gguf,
                                  L,
                                  T("ffn_up.weight"),
                                  L->intermediate,
                                  d,
                                  false,
                                  &L->up_proj,
                                  &L->up_proj_w);
        if (s == GEIST_OK)
            s = load_layer_matrix(st,
                                  gguf,
                                  L,
                                  T("ffn_down.weight"),
                                  d,
                                  L->intermediate,
                                  false,
                                  &L->down_proj,
                                  &L->down_proj_w);
        return s;
    }
    /* The router, the experts and the shared expert (the experts' width). */
    s = moe_wire_layer(st, gguf, L, ARCH);
    if (s == GEIST_OK)
        s = load_layer_matrix(st,
                              gguf,
                              L,
                              T("ffn_gate_shexp.weight"),
                              L->intermediate,
                              d,
                              false,
                              &L->gate_proj,
                              &L->gate_proj_w);
    if (s == GEIST_OK)
        s = load_layer_matrix(st,
                              gguf,
                              L,
                              T("ffn_up_shexp.weight"),
                              L->intermediate,
                              d,
                              false,
                              &L->up_proj,
                              &L->up_proj_w);
    if (s == GEIST_OK)
        s = load_layer_matrix(st,
                              gguf,
                              L,
                              T("ffn_down_shexp.weight"),
                              d,
                              L->intermediate,
                              false,
                              &L->down_proj,
                              &L->down_proj_w);
#undef T
    return s;
}

/* q *= the NoPE layers' temperature times the attention scale, per row:
 * log(floor((pos + 1) / temp_length) + 1) * 0.1 + 1 (llama.cpp's
 * f_attn_temp_offset 1, scale 0.1). On the host. */
[[nodiscard]] static enum geist_status nope_q_scale(struct transformer_layer_forward_ctx *ctx,
                                                    const struct llama4_meta             *m) {
    float *q = (float *) ctx->v->buffer_map(ctx->sess->scratch_q);
    if (q == nullptr)
        return GEIST_E_BACKEND;
    const float base = 1.0f / sqrtf((float) ctx->hd);
    for (size_t t = 0; t < ctx->seq; t++) {
        const size_t pos = ctx->q_position + t;
        const float  temp =
                m->chunk == 0
                        ? 1.0f
                        : logf(floorf((float) (pos + 1) / (float) m->temp_length) + 1.0f) * 0.1f +
                                  1.0f;
        for (size_t i = 0; i < ctx->q_out; i++)
            q[t * ctx->q_out + i] *= base * temp;
    }
    ctx->v->buffer_unmap(ctx->sess->scratch_q);
    return GEIST_OK;
}

/* RMS norm without a weight over each head of q and k (Llama4TextL2Norm). */
[[nodiscard]] static enum geist_status qk_unit_norm(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session *sess = ctx->sess;
    const int64_t                    hd   = (int64_t) ctx->hd;
    struct geist_tensor t_q    = view_2d(sess->scratch_q, ctx->SEQ * ctx->st->n_q_heads, hd);
    struct geist_tensor t_k    = view_2d(sess->scratch_k, ctx->SEQ * ctx->st->n_kv_heads, hd);
    struct geist_tensor t_ones = view_1d(sess->scratch_ones_headdim_max, hd);
    enum geist_status   s      = ctx->prims->rmsnorm(ctx->be, &t_q, &t_ones, ctx->eps, &t_q);
    if (s == GEIST_OK)
        s = ctx->prims->rmsnorm(ctx->be, &t_k, &t_ones, ctx->eps, &t_k);
    return s;
}

/* Each step runs only while s is GEIST_OK. RoPE, then the unit norm, then
 * the scale: the scale multiplies q after the norm, as llama.cpp folds it
 * into the q.k product. */
[[nodiscard]] static enum geist_status
llama4_attention_block(struct transformer_layer_forward_ctx *ctx) {
    const struct llama4_meta m    = meta(ctx->st);
    const bool               rope = rope_layer(&m, (size_t) ctx->layer_idx);
    enum geist_status        s    = blk_attn_in(ctx, nullptr, nullptr);
    if (s == GEIST_OK && rope) {
        s = blk_rope_qk(ctx, 1.0f);
        if (s == GEIST_OK && m.kq_norm)
            s = qk_unit_norm(ctx);
        if (s == GEIST_OK)
            s = nope_q_scale(ctx, &(struct llama4_meta) {.chunk = 0});
    } else if (s == GEIST_OK) {
        s = nope_q_scale(ctx, &m);
    }
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    return s == GEIST_OK ? blk_attn_out(ctx, nullptr) : s;
}

[[nodiscard]] static enum geist_status llama4_ffn_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    if (L->moe_gate_w == nullptr) {
        return blk_ffn_swiglu(ctx);
    }
    enum geist_status s = blk_moe_ffn(ctx); /* h_out = h_post_attn + experts */
    /* The shared expert on blk_moe_ffn's normed input, weight 1. */
    if (s == GEIST_OK)
        s = blk_gate_up(ctx, sess->scratch_pre_ff);
    if (s == GEIST_OK)
        s = blk_silu_mul(ctx, ctx->inter, sess->scratch_gate, sess->scratch_up);
    if (s == GEIST_OK)
        s = blk_down(ctx, sess->scratch_gate, nullptr);
    if (s == GEIST_OK)
        s = blk_add(ctx, ctx->st->d_model, ctx->h_out_buf, sess->scratch_ffn_out, ctx->h_out_buf);
    return s;
}

const struct transformer_family transformer_family_llama4 = {
        .name            = "llama4",
        .populate        = populate_llama4,
        .populate_layers = populate_layers_llama4,
        .wire_layer      = wire_layer_llama4,
        .session_alloc   = moe_session_alloc,
        .session_free    = moe_session_free,
        .attention_block = llama4_attention_block,
        .ffn_block       = llama4_ffn_block,
        /* As the other MoE families: routing amplifies KV quantization. */
        .kv_auto         = GEIST_KV_F16,
        .kv_int4_instead = GEIST_KV_F16,
};
