/*
 * src/archs/transformer/families/deepseek2.c - the DeepSeek-V2 / V3 family
 * unit (GGUF architecture "deepseek2", #242).
 *
 * Layer: ARCHITECTURE.
 *
 * As llama.cpp's graph for the architecture (deepseek2.cpp) runs a file
 * with the unsplit attn_kv_b (every DeepSeek-V2-Lite GGUF converted before
 * 2025): multi-head latent attention, decompressed to plain multi-head
 * attention per token:
 *   q       = q_proj(x)                      (V3: q_b(rmsnorm(q_a(x))))
 *   kv_a    = kv_a_mqa(x) = [latent | k_pe]
 *   [k_nope | v] per head = kv_b(rmsnorm(latent))
 *   k       = [k_pe | k_nope], k_pe shared by every head
 * RoPE (interleaved pairs, NORM) on the 64 pe dims only, YaRN with
 * DeepSeek's log multiplier folded into the q scale (mscale^2 / sqrt(192)),
 * the rope tables themselves unscaled. The loader moves each q head's and
 * kv_a's pe rows first, deinterleaved (config.rope_il_n), so the cache holds
 * [pe | nope] heads of 192 that the shared partial-RoPE blocks rotate; V
 * heads (128) are zero-padded to 192 and the first 128 dims of each output
 * head go to o_proj.
 *
 * FFN: the first leading_dense_block_count layers are dense SwiGLU; the
 * others a router over expert_count experts (softmax, or sigmoid with a
 * selection bias and expert groups on V3) plus the shared experts, one
 * SwiGLU of expert_shared_count experts' width, weight 1, added after.
 *
 *   h_post_attn = h_in + o_proj(mla(attn_norm(h_in)))
 *   h_out       = dense: blk_ffn_swiglu;  else: blk_moe_ffn + shared(ffn_norm(h_post_attn))
 *
 * Refused: files with the split attn_k_b / attn_v_b (absorbed MLA), which
 * this decompressed path does not read.
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
#include "heap.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static const char ARCH[] = "deepseek2";

/* Layers before this one are dense; set by populate from the file. */
static size_t n_dense_lead(const struct transformer_arch_state *st) {
    uint32_t n = 0;
    if (st->gguf != nullptr)
        (void) family_meta_u32(
                (struct gguf_ctx *) st->gguf, ARCH, nullptr, "leading_dense_block_count", &n);
    return n;
}

static void populate_deepseek2(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    struct geist_arch_config *c = &st->config;
    c->ffn_activation           = GEIST_FFN_SWIGLU;
    family_populate_common(gguf, ARCH, nullptr, st);
    st->vocab_size = family_vocab_size(gguf, ARCH, nullptr);
    family_populate_moe(gguf, ARCH, st);
    /* llama.cpp's defaults for this architecture: weights not renormalised
     * unless the file says so, softmax unless it names another router. */
    bool     norm = false;
    uint32_t u    = 0;
    char     key[96];
    snprintf(key, sizeof key, "%s.expert_weights_norm", ARCH);
    c->moe_norm_topk = gguf_get_meta_bool(gguf, key, &norm) && norm;
    c->moe_gating    = GEIST_MOE_GATING_SOFTMAX;
    if (family_meta_u32(gguf, ARCH, nullptr, "expert_gating_func", &u) && u != 0)
        c->moe_gating = (enum geist_moe_gating) u;
    (void) family_meta_f32(gguf, ARCH, nullptr, "expert_weights_scale", &c->moe_w_scale);
    if (family_meta_u32(gguf, ARCH, nullptr, "expert_group_count", &u))
        c->moe_n_group = u;
    if (family_meta_u32(gguf, ARCH, nullptr, "expert_group_used_count", &u))
        c->moe_topk_group = u;
    c->moe_n_shared = 0; /* the shared experts run as one dense FFN */

    if (family_meta_u32(gguf, ARCH, nullptr, "attention.kv_lora_rank", &u))
        c->mla_kv_rank = u;
    if (family_meta_u32(gguf, ARCH, nullptr, "attention.q_lora_rank", &u))
        c->mla_q_rank = u;
    if (family_meta_u32(gguf, ARCH, nullptr, "rope.dimension_count", &u))
        c->mla_rope = u;
    if (family_meta_u32(gguf, ARCH, nullptr, "attention.value_length", &u))
        c->mla_v_dim = u;
    c->rope_interleaved   = true; /* LLAMA_ROPE_TYPE_NORM */
    c->rope_partial_block = true; /* the pe dims form their own block */
    c->rope_il_n          = c->mla_rope;
}

/* YaRN as llama.cpp runs deepseek2 (#242): the tables unscaled, the
 * magnitude in the q scale. False for metadata it cannot follow. */
static bool populate_yarn_deepseek2(struct gguf_ctx *g, struct transformer_arch_state *st) {
    struct geist_arch_config *c  = &st->config;
    const float               hd = (float) family_head_dim(st, ARCH);
    char                      k[96];
    size_t                    len = 0;
    snprintf(k, sizeof k, "%s.rope.scaling.type", ARCH);
    const char *type = gguf_get_meta_string(g, k, &len);
    c->attn_q_scale  = 1.0f / sqrtf(hd);
    if (type == nullptr || len != 4 || memcmp(type, "yarn", 4) != 0) {
        return type == nullptr; /* another scaling type is not read */
    }
    float    factor = 0.0f, beta_fast = 32.0f, beta_slow = 1.0f, log_mul = 0.0f;
    uint32_t orig = 0;
    (void) family_meta_f32(g, ARCH, nullptr, "rope.scaling.factor", &factor);
    (void) family_meta_u32(g, ARCH, nullptr, "rope.scaling.original_context_length", &orig);
    (void) family_meta_f32(g, ARCH, nullptr, "rope.scaling.yarn_beta_fast", &beta_fast);
    (void) family_meta_f32(g, ARCH, nullptr, "rope.scaling.yarn_beta_slow", &beta_slow);
    (void) family_meta_f32(g, ARCH, nullptr, "rope.scaling.yarn_log_multiplier", &log_mul);
    if (!(factor > 1.0f) || orig == 0 || !(beta_fast > beta_slow) || !(beta_slow > 0.0f)) {
        return false;
    }
    c->rope_yarn_factor    = factor;
    c->rope_yarn_orig_ctx  = orig;
    c->rope_yarn_beta_fast = beta_fast;
    c->rope_yarn_beta_slow = beta_slow;
    /* llama.cpp: the converter's 0.1 is cancelled (log_mul / 0.1), the
     * context's attn_factor then cancels rope_yarn's 1 + 0.1 ln factor, so
     * the tables carry magnitude 1, and the graph scales q.k by
     * mscale^2 / sqrt(head_dim), mscale = 1 + 0.1 * (log_mul / 0.1) ln factor. */
    c->rope_yarn_mscale = 1.0f;
    const float mscale  = 1.0f + log_mul * logf(factor);
    c->attn_q_scale     = mscale * mscale / sqrtf(hd);
    return true;
}

static bool populate_layers_deepseek2(struct transformer_arch_state *st) {
    struct gguf_ctx                *g  = (struct gguf_ctx *) st->gguf;
    const struct geist_arch_config *c  = &st->config;
    const uint32_t                  hd = family_head_dim(st, ARCH);
    uint32_t                        ff = 0, eff = 0;
    float                           base = 10000.0f;
    if (g == nullptr || !populate_yarn_deepseek2(g, st))
        return false;
    (void) family_meta_u32(g, ARCH, nullptr, "feed_forward_length", &ff);
    (void) family_meta_u32(g, ARCH, nullptr, "expert_feed_forward_length", &eff);
    (void) family_meta_f32(g, ARCH, nullptr, "rope.freq_base", &base);
    const size_t lead = n_dense_lead(st);
    if (hd == 0 || c->mla_kv_rank == 0 || c->mla_rope == 0 || c->mla_rope % 2 != 0 ||
        c->mla_rope >= hd || c->mla_v_dim == 0 || c->mla_v_dim > hd || ff == 0 || eff == 0 ||
        st->n_q_heads == 0 || st->n_kv_heads != st->n_q_heads || lead > st->n_layers ||
        !family_moe_ok(c)) {
        return false;
    }
    for (size_t i = 0; i < st->n_layers; i++) {
        family_uniform_layer(
                st, i, hd, i < lead ? ff : eff, base, (int) c->mla_rope, &st->layers[i]);
    }
    return true;
}

/* A [n] F32 host copy of tensor `suffix` of layer L, or nullptr when the
 * file has none (`required`: an error then). */
static float *host_vector(struct transformer_arch_state          *st,
                          struct gguf_ctx                        *gguf,
                          const struct transformer_layer_weights *L,
                          const char                             *suffix,
                          size_t                                  n,
                          bool                                    required,
                          enum geist_status                      *s) {
    char name[64];
    snprintf(name, sizeof name, "blk.%d.%s", L->layer_idx, suffix);
    const struct gguf_tensor_t *t = gguf_get_tensor(gguf, name);
    if (t == nullptr && !required)
        return nullptr;
    float *out = t != nullptr && gguf_tensor_elem_count(t) == n ? gguf_dequant_to_fp32(t) : nullptr;
    if (out == nullptr) {
        geist_backend_set_error(
                st->backend, GEIST_E_FORMAT, "%s: '%s' is missing or not [%zu]", ARCH, name, n);
        *s = GEIST_E_FORMAT;
    }
    return out;
}

[[nodiscard]] static enum geist_status wire_layer_deepseek2(struct transformer_arch_state    *st,
                                                            struct gguf_ctx                  *gguf,
                                                            struct transformer_layer_weights *L) {
    const struct geist_arch_config *c     = &st->config;
    const size_t                    d     = st->d_model;
    const size_t                    hd    = L->head_dim;
    const size_t                    heads = st->n_q_heads;
    const size_t                    nope  = hd - c->mla_rope;
    char                            name[64];
    snprintf(name, sizeof name, "blk.%d.attn_k_b.weight", L->layer_idx);
    if (gguf_get_tensor(gguf, name) != nullptr) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "%s: '%s' is the split (absorbed) MLA layout; this build reads "
                                "attn_kv_b",
                                ARCH,
                                name);
        return GEIST_E_UNSUPPORTED;
    }
#define T(suffix) (snprintf(name, sizeof name, "blk.%d." suffix, L->layer_idx), name)
    enum geist_status s = GEIST_OK;
    if (c->mla_q_rank > 0) {
        s = load_layer_matrix(st,
                              gguf,
                              L,
                              T("attn_q_a.weight"),
                              c->mla_q_rank,
                              d,
                              false,
                              &L->mla_q_a,
                              &L->mla_q_a_w);
        if (s == GEIST_OK)
            s = load_layer_norm(
                    st, gguf, L, T("attn_q_a_norm.weight"), c->mla_q_rank, &L->mla_q_a_norm);
        if (s == GEIST_OK)
            s = load_layer_matrix_rope_il(st,
                                          gguf,
                                          L,
                                          T("attn_q_b.weight"),
                                          L->q_out,
                                          c->mla_q_rank,
                                          hd,
                                          &L->q_proj,
                                          &L->q_proj_w);
    } else {
        s = load_layer_matrix_rope_il(
                st, gguf, L, T("attn_q.weight"), L->q_out, d, hd, &L->q_proj, &L->q_proj_w);
    }
    const size_t kv_a = c->mla_kv_rank + c->mla_rope;
    if (s == GEIST_OK)
        s = load_layer_matrix_rope_il(st,
                                      gguf,
                                      L,
                                      T("attn_kv_a_mqa.weight"),
                                      kv_a,
                                      d,
                                      kv_a,
                                      &L->mla_kv_a,
                                      &L->mla_kv_a_w);
    if (s == GEIST_OK)
        L->mla_kv_a_norm =
                host_vector(st, gguf, L, "attn_kv_a_norm.weight", c->mla_kv_rank, true, &s);
    if (s == GEIST_OK)
        s = load_layer_matrix(st,
                              gguf,
                              L,
                              T("attn_kv_b.weight"),
                              heads * (nope + c->mla_v_dim),
                              c->mla_kv_rank,
                              false,
                              &L->mla_kv_b,
                              &L->mla_kv_b_w);
    if (s == GEIST_OK)
        s = load_layer_matrix(st,
                              gguf,
                              L,
                              T("attn_output.weight"),
                              d,
                              heads * c->mla_v_dim,
                              false,
                              &L->o_proj,
                              &L->o_proj_w);
    if (s != GEIST_OK)
        return s;
    if ((size_t) L->layer_idx < n_dense_lead(st)) {
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
    /* The shared experts, one FFN of expert_shared_count experts' width. */
    uint32_t n_shared = 0;
    (void) family_meta_u32(gguf, ARCH, nullptr, "expert_shared_count", &n_shared);
    const size_t sff = (size_t) n_shared * L->intermediate;
    s                = moe_wire_layer(st, gguf, L, ARCH);
    if (s == GEIST_OK && sff > 0) {
        s = load_layer_matrix(st,
                              gguf,
                              L,
                              T("ffn_gate_shexp.weight"),
                              sff,
                              d,
                              false,
                              &L->gate_proj,
                              &L->gate_proj_w);
        if (s == GEIST_OK)
            s = load_layer_matrix(st,
                                  gguf,
                                  L,
                                  T("ffn_up_shexp.weight"),
                                  sff,
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
                                  sff,
                                  false,
                                  &L->down_proj,
                                  &L->down_proj_w);
    }
#undef T
    if (s == GEIST_OK)
        L->moe_sel_b = host_vector(st, gguf, L, "exp_probs_b.bias", c->moe_n_experts, false, &s);
    return s;
}

/* ---- Session ---------------------------------------------------------- */

[[nodiscard]] static enum geist_status
deepseek2_session_alloc(struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st = sess->model;
    const struct geist_arch_config      *c  = &st->config;
    const size_t                         m  = sess->m_max;
    const size_t      nope = st->layers[0].head_dim - c->mla_rope, heads = st->n_q_heads;
    enum geist_status s = moe_session_alloc(sess);
    const struct {
        size_t                n;
        struct geist_buffer **out;
    } bufs[] = {
            {c->mla_kv_rank + c->mla_rope, &sess->mla_kv_a},
            {c->mla_kv_rank, &sess->mla_lat},
            {heads * (nope + c->mla_v_dim), &sess->mla_kv},
            {c->mla_q_rank, &sess->mla_q_a},
    };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0] && s == GEIST_OK; i++)
        if (bufs[i].n > 0)
            s = transformer_scratch_alloc(st->backend, m * bufs[i].n * sizeof(float), bufs[i].out);
    return s;
}

static void deepseek2_session_free(struct transformer_arch_session *sess) {
    moe_session_free(sess);
    struct geist_backend *be = sess->model->backend;
    transformer_buffer_release(be, &sess->mla_kv_a);
    transformer_buffer_release(be, &sess->mla_lat);
    transformer_buffer_release(be, &sess->mla_kv);
    transformer_buffer_release(be, &sess->mla_q_a);
}

/* ---- Forward ---------------------------------------------------------- */

/* kv_a's latent normalised into mla_lat, then K and V of every head from
 * kv_b's output: k = [k_pe | k_nope], v = [v | 0]. On the host. */
[[nodiscard]] static enum geist_status mla_kv(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const struct geist_arch_config   *c    = &st->config;
    const struct geist_backend_vtbl  *v    = ctx->v;
    const size_t rank = c->mla_kv_rank, pe = c->mla_rope, hd = ctx->hd, nope = hd - pe;
    const size_t vd = c->mla_v_dim, heads = st->n_q_heads, kv_a = rank + pe;

    /* The latent: rows [pe, kv_a) of kv_a after the load's reorder. */
    float       *a  = (float *) v->buffer_map(sess->mla_kv_a);
    float       *lt = (float *) v->buffer_map(sess->mla_lat);
    const float *nw = L->mla_kv_a_norm;
    if (a == nullptr || lt == nullptr)
        return GEIST_E_BACKEND;
    for (size_t t = 0; t < ctx->seq; t++) {
        const float *src = a + t * kv_a + pe;
        double       ss  = 0.0;
        for (size_t i = 0; i < rank; i++)
            ss += (double) src[i] * src[i];
        const float r = 1.0f / sqrtf((float) (ss / (double) rank) + ctx->eps);
        for (size_t i = 0; i < rank; i++)
            lt[t * rank + i] = src[i] * r * nw[i];
    }
    v->buffer_unmap(sess->mla_lat);
    v->buffer_unmap(sess->mla_kv_a);
    enum geist_status s =
            blk_linear(ctx, sess->mla_lat, &L->mla_kv_b, &L->mla_kv_b_w, sess->mla_kv);
    if (s != GEIST_OK)
        return s;
    a                = (float *) v->buffer_map(sess->mla_kv_a);
    const float *kv  = (const float *) v->buffer_map(sess->mla_kv);
    float       *k   = (float *) v->buffer_map(sess->scratch_k);
    float       *val = (float *) v->buffer_map(sess->scratch_v);
    if (a == nullptr || kv == nullptr || k == nullptr || val == nullptr)
        return GEIST_E_BACKEND;
    for (size_t t = 0; t < ctx->seq; t++) {
        for (size_t h = 0; h < heads; h++) {
            const float *kvh = kv + (t * heads + h) * (nope + vd);
            float       *kh  = k + (t * heads + h) * hd;
            float       *vh  = val + (t * heads + h) * hd;
            memcpy(kh, a + t * kv_a, pe * sizeof(float));
            memcpy(kh + pe, kvh, nope * sizeof(float));
            memcpy(vh, kvh + nope, vd * sizeof(float));
            memset(vh + vd, 0, (hd - vd) * sizeof(float));
        }
    }
    v->buffer_unmap(sess->scratch_v);
    v->buffer_unmap(sess->scratch_k);
    v->buffer_unmap(sess->mla_kv);
    v->buffer_unmap(sess->mla_kv_a);
    return GEIST_OK;
}

/* The first mla_v_dim of each output head, packed in place: scratch_attn
 * [seq, heads * hd] -> [seq, heads * mla_v_dim]. */
[[nodiscard]] static enum geist_status mla_pack_out(struct transformer_layer_forward_ctx *ctx) {
    const size_t hd = ctx->hd, vd = ctx->st->config.mla_v_dim, heads = ctx->st->n_q_heads;
    float       *o = (float *) ctx->v->buffer_map(ctx->sess->scratch_attn);
    if (o == nullptr)
        return GEIST_E_BACKEND;
    for (size_t r = 0; r < ctx->seq * heads; r++)
        memmove(o + r * vd, o + r * hd, vd * sizeof(float));
    ctx->v->buffer_unmap(ctx->sess->scratch_attn);
    return GEIST_OK;
}

/* Each step runs only while s is GEIST_OK. */
[[nodiscard]] static enum geist_status
deepseek2_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const struct geist_arch_config   *c    = &ctx->st->config;
    struct geist_buffer              *x    = sess->scratch_normed;

    enum geist_status s = blk_attn_norm(ctx, x);
    if (s == GEIST_OK && c->mla_q_rank > 0) {
        s = blk_linear(ctx, x, &L->mla_q_a, &L->mla_q_a_w, sess->mla_q_a);
        if (s == GEIST_OK)
            s = blk_rmsnorm(ctx, c->mla_q_rank, sess->mla_q_a, &L->mla_q_a_norm, sess->mla_q_a);
        if (s == GEIST_OK)
            s = blk_linear(ctx, sess->mla_q_a, &L->q_proj, &L->q_proj_w, sess->scratch_q);
    } else if (s == GEIST_OK) {
        s = blk_linear(ctx, x, &L->q_proj, &L->q_proj_w, sess->scratch_q);
    }
    if (s == GEIST_OK)
        s = blk_linear(ctx, x, &L->mla_kv_a, &L->mla_kv_a_w, sess->mla_kv_a);
    if (s == GEIST_OK)
        s = mla_kv(ctx);
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, c->attn_q_scale);
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    if (s == GEIST_OK)
        s = mla_pack_out(ctx);
    if (s == GEIST_OK)
        s = blk_linear(ctx, sess->scratch_attn, &L->o_proj, &L->o_proj_w, sess->scratch_o);
    return s == GEIST_OK ? blk_attn_residual(ctx, nullptr) : s;
}

[[nodiscard]] static enum geist_status
deepseek2_ffn_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    if (L->moe_gate_w == nullptr) {
        return blk_ffn_swiglu(ctx);
    }
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

TRANSFORMER_FAMILY(deepseek2,
                   populate_deepseek2,
                   populate_layers_deepseek2,
                   deepseek2_attention_block,
                   deepseek2_ffn_block,
                   .wire_layer    = wire_layer_deepseek2,
                   .session_alloc = deepseek2_session_alloc,
                   .session_free  = deepseek2_session_free,
                   /* As the other MoE families: routing amplifies KV quantization. */
                   .kv_auto         = GEIST_KV_F16,
                   .kv_int4_instead = GEIST_KV_F16);
