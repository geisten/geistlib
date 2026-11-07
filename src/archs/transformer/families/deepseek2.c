/*
 * src/archs/transformer/families/deepseek2.c - the DeepSeek-V2 / V3 family
 * unit (GGUF architecture "deepseek2", #242, #375).
 *
 * Layer: ARCHITECTURE.
 *
 * Multi-head latent attention as llama.cpp's graph for the architecture
 * (deepseek2.cpp) runs it, in the form the file was converted for:
 *   q       = q_proj(x)                      (V3: q_b(rmsnorm(q_a(x))))
 *   kv_a    = kv_a_mqa(x) = [latent | k_pe],  lat = rmsnorm(latent)
 * RoPE (interleaved pairs, NORM) on the 64 pe dims only, YaRN with
 * DeepSeek's log multiplier folded into the q scale (mscale^2 /
 * sqrt(nope + pe)), the rope tables themselves unscaled. The loader moves
 * each q head's and kv_a's pe rows first, deinterleaved (config.rope_il_n),
 * so the shared partial-RoPE blocks rotate the leading pe dims of a head.
 *
 * Decompressed (the unsplit attn_kv_b of DeepSeek-V2-Lite files converted
 * before 2025): [k_nope | v] per head = kv_b(lat), the cache holds per head
 * k = [k_pe | k_nope] (192) and v zero-padded to it, and the first
 * mla_v_dim of each output head go to o_proj. Assembled on the host.
 *
 * Absorbed (config.mla_absorbed: attention.key_length_mla and the split
 * attn_k_b / attn_v_b, every DeepSeek-V3 / R1, Kimi K2 and GigaChat3 file,
 * #375): llama.cpp's is_mla path. One KV head of [k_pe | lat] (576) for K
 * and for V, each q head [q_pe | k_b[h] q_nope] attends it (MQA), and
 * v_b[h] takes the latent dims of its output head to mla_v_dim, so o_proj
 * reads the same [heads * mla_v_dim] as above. The cache holds 576 floats
 * per token and layer twice (K, and V = K before RoPE, of which only the
 * latent dims are read) instead of 2 * heads * 192. On a CPU backend k_b,
 * v_b and the assembly run on the host, per head on the backend's kernels.
 *
 * On a GPU (#376: a backend with prims->scale_f32 that takes row views
 * and fused->linear_t_ids, Metal) both forms run on the device, no host
 * round trip per layer: the latent's RMSNorm on its row view, K and V
 * assembled by strided copies (scale_f32 times 1), and absorbed, k_b / v_b
 * of every head as one linear_t_ids whose "experts" are the heads. A GPU
 * without them refuses the absorbed layout and assembles the decompressed
 * one on the host.
 *
 * FFN: the first leading_dense_block_count layers are dense SwiGLU; the
 * others a router over expert_count experts (softmax, or sigmoid with a
 * selection bias and expert groups on V3) plus the shared experts, one
 * SwiGLU of expert_shared_count experts' width, weight 1, added after.
 *
 *   h_post_attn = h_in + o_proj(mla(attn_norm(h_in)))
 *   h_out       = dense: blk_ffn_swiglu;  else: blk_moe_ffn + shared(ffn_norm(h_post_attn))
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
#include "checked.h"
#include "heap.h"
#include "par.h"

#include <math.h>
#include <stdatomic.h>
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
    c->mla_k_dim = family_head_dim(st, ARCH);
    /* llama.cpp's is_mla: both MLA head sizes in the file. Then key_length
     * and value_length are the absorbed cache's (rope + rank, rank). One of
     * them alone leaves mla_k_dim 0, which populate_layers refuses. */
    uint32_t   kmla = 0, vmla = 0;
    const bool has_k = family_meta_u32(gguf, ARCH, nullptr, "attention.key_length_mla", &kmla);
    const bool has_v = family_meta_u32(gguf, ARCH, nullptr, "attention.value_length_mla", &vmla);
    if (has_k || has_v) {
        c->mla_absorbed = has_k && has_v;
        c->mla_k_dim    = c->mla_absorbed ? kmla : 0;
        c->mla_v_dim    = c->mla_absorbed ? vmla : 0;
    }
    c->rope_interleaved   = true; /* LLAMA_ROPE_TYPE_NORM */
    c->rope_partial_block = true; /* the pe dims form their own block */
    c->rope_il_n          = c->mla_rope;
}

/* YaRN as llama.cpp runs deepseek2 (#242): the tables unscaled, the
 * magnitude in the q scale. False for metadata it cannot follow. */
static bool populate_yarn_deepseek2(struct gguf_ctx *g, struct transformer_arch_state *st) {
    struct geist_arch_config *c  = &st->config;
    const float               hd = (float) c->mla_k_dim; /* q.k before any absorption */
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
    const size_t kd   = c->mla_k_dim;
    if (hd == 0 || kd == 0 || c->mla_kv_rank == 0 || c->mla_rope == 0 || c->mla_rope % 2 != 0 ||
        c->mla_rope >= kd || c->mla_v_dim == 0 || ff == 0 || eff == 0 || st->n_q_heads == 0 ||
        lead > st->n_layers || !family_moe_ok(c)) {
        return false;
    }
    /* Absorbed: one KV head of [k_pe | latent]. Decompressed: a K head per
     * q head, V padded to it. */
    const bool absorbed_ok = hd == c->mla_kv_rank + c->mla_rope && st->n_kv_heads == 1;
    const bool plain_ok    = kd == hd && c->mla_v_dim <= hd && st->n_kv_heads == st->n_q_heads;
    if (c->mla_absorbed ? !absorbed_ok : !plain_ok) {
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

/* The 3-D tensor `name` must be [ne0, ne1, ne2] (GGUF order): k_b's
 * per-head [nope, rank] is transposed against kv_b's rows (llama.cpp's
 * converter), which the element count alone would not tell. */
[[nodiscard]] static enum geist_status check_dims3(
        struct transformer_arch_state *st, const char *name, size_t ne0, size_t ne1, size_t ne2) {
    const struct gguf_tensor_t *t = gguf_get_tensor((struct gguf_ctx *) st->gguf, name);
    if (t != nullptr && t->n_dims == 3 && t->dims[0] == ne0 && t->dims[1] == ne1 &&
        t->dims[2] == ne2)
        return GEIST_OK;
    geist_backend_set_error(st->backend,
                            GEIST_E_FORMAT,
                            "%s: '%s' is missing or not [%zu, %zu, %zu]",
                            ARCH,
                            name,
                            ne0,
                            ne1,
                            ne2);
    return GEIST_E_FORMAT;
}

/* MLA on the device (#376): a GPU backend with strided copies
 * (prims->scale_f32 on row views) and id-indexed matvecs. */
static bool mla_device(const struct transformer_arch_state *st) {
    const struct geist_backend_descriptor *d = st->backend->desc;
    return strncmp(d->name, "cpu_", 4) != 0 && d->prims->scale_f32 != nullptr &&
           d->fused != nullptr && d->fused->linear_t_ids != nullptr;
}

/* Absorbed MLA's per-head k_b [rank, nope] and v_b [mla_v_dim, rank], each
 * with a kernel of the CPU backend's that the host calls per head. */
[[nodiscard]] static enum geist_status wire_absorbed(struct transformer_arch_state    *st,
                                                     struct gguf_ctx                  *gguf,
                                                     struct transformer_layer_weights *L,
                                                     size_t                            nope) {
    const struct geist_arch_config *c     = &st->config;
    const size_t                    heads = st->n_q_heads, rank = c->mla_kv_rank;
    char                            kb[64], vb[64];
    snprintf(kb, sizeof kb, "blk.%d.attn_k_b.weight", L->layer_idx);
    snprintf(vb, sizeof vb, "blk.%d.attn_v_b.weight", L->layer_idx);
    L->mla_n_heads      = heads; /* what the teardown walks */
    enum geist_status s = check_dims3(st, kb, nope, rank, heads);
    if (s == GEIST_OK)
        s = check_dims3(st, vb, rank, c->mla_v_dim, heads);
    if (s == GEIST_OK)
        s = load_layer_slices(st, gguf, L, kb, heads, rank, nope, &L->mla_k_b, &L->mla_k_b_w);
    if (s == GEIST_OK)
        s = load_layer_slices(
                st, gguf, L, vb, heads, c->mla_v_dim, rank, &L->mla_v_b, &L->mla_v_b_w);
    for (size_t h = 0; s == GEIST_OK && h < heads; h++) {
        const struct geist_weight *w[] = {&L->mla_k_b_w[h], &L->mla_v_b_w[h]};
        for (size_t i = 0; i < 2 && s == GEIST_OK; i++) {
            if (w[i]->linear_m1 == nullptr || w[i]->linear_mN == nullptr) {
                geist_backend_set_error(st->backend,
                                        GEIST_E_UNSUPPORTED,
                                        "%s: %s has no row kernel for '%s'",
                                        ARCH,
                                        st->backend->desc->name,
                                        i == 0 ? kb : vb);
                s = GEIST_E_UNSUPPORTED;
            }
        }
    }
    return s;
}

[[nodiscard]] static enum geist_status wire_layer_deepseek2(struct transformer_arch_state    *st,
                                                            struct gguf_ctx                  *gguf,
                                                            struct transformer_layer_weights *L) {
    const struct geist_arch_config *c     = &st->config;
    const size_t                    d     = st->d_model;
    const size_t                    kd    = c->mla_k_dim;
    const size_t                    heads = st->n_q_heads;
    const size_t                    nope  = kd - c->mla_rope;
    char                            name[64];
    snprintf(name, sizeof name, "blk.%d.attn_k_b.weight", L->layer_idx);
    if (!c->mla_absorbed && gguf_get_tensor(gguf, name) != nullptr) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "%s: '%s' is the split (absorbed) MLA layout, but the file has no "
                                "attention.key_length_mla / value_length_mla",
                                ARCH,
                                name);
        return GEIST_E_UNSUPPORTED;
    }
    /* A GPU runs k_b / v_b per head on the device or not at all. */
    const bool gpu = strncmp(st->backend->desc->name, "cpu_", 4) != 0;
    if (c->mla_absorbed && gpu && !mla_device(st)) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "%s: the absorbed MLA layout (attn_k_b / attn_v_b) needs "
                                "linear_t_ids and scale_f32 on a GPU, which %s lacks (#376)",
                                ARCH,
                                st->backend->desc->name);
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
                                          heads * kd,
                                          c->mla_q_rank,
                                          kd,
                                          &L->q_proj,
                                          &L->q_proj_w);
    } else {
        s = load_layer_matrix_rope_il(
                st, gguf, L, T("attn_q.weight"), heads * kd, d, kd, &L->q_proj, &L->q_proj_w);
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
    if (s == GEIST_OK && mla_device(st))
        s = load_layer_norm(
                st, gguf, L, T("attn_kv_a_norm.weight"), c->mla_kv_rank, &L->mla_kv_a_norm_t);
    if (s == GEIST_OK && c->mla_absorbed)
        s = wire_absorbed(st, gguf, L, nope);
    else if (s == GEIST_OK)
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
    const size_t nope = c->mla_k_dim - c->mla_rope, heads = st->n_q_heads, rank = c->mla_kv_rank;
    const bool   ab     = c->mla_absorbed;
    enum geist_status s = moe_session_alloc(sess);
    /* Absorbed: mla_kv holds q before the absorption, mla_hx / mla_hy a
     * head's rows in and out of k_b and v_b; the latent goes straight to
     * scratch_k. */
    const size_t hx = nope > rank ? nope : rank, hy = rank > c->mla_v_dim ? rank : c->mla_v_dim;
    const struct {
        size_t                n;
        struct geist_buffer **out;
    } bufs[] = {
            {rank + c->mla_rope, &sess->mla_kv_a},
            {ab ? 0 : rank, &sess->mla_lat},
            {heads * (ab ? c->mla_k_dim : nope + c->mla_v_dim), &sess->mla_kv},
            {c->mla_q_rank, &sess->mla_q_a},
            {ab ? heads * hx : 0, &sess->mla_hx},
            {ab ? heads * hy : 0, &sess->mla_hy},
    };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0] && s == GEIST_OK; i++) {
        size_t bytes = 0;
        if (bufs[i].n == 0)
            continue;
        if (ckd_mul(&bytes, m, bufs[i].n) || ckd_mul(&bytes, bytes, sizeof(float)))
            return GEIST_E_OOM;
        s = transformer_scratch_alloc(st->backend, bytes, bufs[i].out);
    }
    /* #376: on a GPU, slot t * heads + h runs head h's k_b / v_b. */
    size_t n_ids = 0;
    if (s == GEIST_OK && ab && mla_device(st)) {
        const struct geist_backend_vtbl *v = st->backend->desc->vtbl;
        if (ckd_mul(&n_ids, m, heads) || n_ids > UINT32_MAX)
            return GEIST_E_OOM;
        s             = v->buffer_create(st->backend,
                                         n_ids * sizeof(uint32_t),
                                         GEIST_BUFFER_IO,
                                         GEIST_MEMORY_AUTO,
                                         &sess->mla_ids);
        uint32_t *ids = s == GEIST_OK ? v->buffer_map(sess->mla_ids) : nullptr;
        if (s == GEIST_OK && ids == nullptr)
            s = GEIST_E_BACKEND;
        for (size_t i = 0; ids != nullptr && i < n_ids; i++)
            ids[i] = (uint32_t) (i % heads);
        if (ids != nullptr)
            v->buffer_unmap(sess->mla_ids);
    }
    return s;
}

static void deepseek2_session_free(struct transformer_arch_session *sess) {
    moe_session_free(sess);
    struct geist_backend *be = sess->model->backend;
    transformer_buffer_release(be, &sess->mla_kv_a);
    transformer_buffer_release(be, &sess->mla_lat);
    transformer_buffer_release(be, &sess->mla_kv);
    transformer_buffer_release(be, &sess->mla_q_a);
    transformer_buffer_release(be, &sess->mla_hx);
    transformer_buffer_release(be, &sess->mla_hy);
    if (sess->mla_ids != nullptr) {
        be->desc->vtbl->buffer_destroy(be, sess->mla_ids);
        sess->mla_ids = nullptr;
    }
}

/* ---- Forward ---------------------------------------------------------- */

/* rmsnorm(src) * w into dst over the latent's n dims, in double as the
 * decompressed path always did. */
static void latent_norm(size_t n, float eps, const float *src, const float *w, float *dst) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++)
        ss += (double) src[i] * src[i];
    const float r = 1.0f / sqrtf((float) (ss / (double) n) + eps);
    for (size_t i = 0; i < n; i++)
        dst[i] = src[i] * r * w[i];
}

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
    float *a  = (float *) v->buffer_map(sess->mla_kv_a);
    float *lt = (float *) v->buffer_map(sess->mla_lat);
    if (a == nullptr || lt == nullptr)
        return GEIST_E_BACKEND;
    for (size_t t = 0; t < ctx->seq; t++)
        latent_norm(rank, ctx->eps, a + t * kv_a + pe, L->mla_kv_a_norm, lt + t * rank);
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

/* ---- Absorbed MLA (#375) ----------------------------------------------- *
 * Per head h, over the m rows of the step: y = W[h] x, x gathered from
 * `src` (row stride src_row floats, the head's dims at src_off + h *
 * src_head), into mla_hx; y in mla_hy. Heads run in parallel, each on one
 * thread with the backend's row kernel (inside a parallel region the
 * kernel's own parallel-for runs inline), so a head's bytes do not depend
 * on the thread count. */
struct mla_heads_job {
    const struct geist_weight *w; /* [heads] */
    struct geist_backend      *be;
    size_t                     m, heads, n_in, n_out, src_row, src_head, src_off;
    const float               *src;
    float                     *hx, *hy;
    atomic_bool                failed;
};

static void mla_heads_items(void *p, size_t h0, size_t h1) {
    struct mla_heads_job *j = p;
    for (size_t h = h0; h < h1; h++) {
        float                     *x = j->hx + h * j->m * j->n_in;
        float                     *y = j->hy + h * j->m * j->n_out;
        const struct geist_weight *w = &j->w[h];
        for (size_t t = 0; t < j->m; t++)
            memcpy(x + t * j->n_in,
                   j->src + t * j->src_row + j->src_off + h * j->src_head,
                   j->n_in * sizeof(float));
        if (j->m == 1 && w->linear_m1 != nullptr)
            w->linear_m1(x, w, j->be, y);
        else if (j->m > 1 && w->linear_mN != nullptr)
            w->linear_mN(j->m, x, w, j->be, y);
        else
            atomic_store_explicit(&j->failed, true, memory_order_relaxed);
    }
}

[[nodiscard]] static enum geist_status mla_heads(struct mla_heads_job *j) {
    atomic_init(&j->failed, false);
    geist_par_for(j->heads, mla_heads_items, j);
    return atomic_load_explicit(&j->failed, memory_order_relaxed) ? GEIST_E_UNSUPPORTED : GEIST_OK;
}

/* K = V = [k_pe | rmsnorm(latent)] (one KV head; RoPE follows on K) and
 * every q head [q_pe | k_b[h] q_nope], from kv_a and q (mla_kv). */
[[nodiscard]] static enum geist_status mla_absorb_qkv(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_state    *st   = ctx->st;
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const struct geist_arch_config   *c    = &st->config;
    const struct geist_backend_vtbl  *v    = ctx->v;
    const size_t rank = c->mla_kv_rank, pe = c->mla_rope, hd = ctx->hd, kd = c->mla_k_dim;
    const size_t heads = st->n_q_heads, m = ctx->seq;
    struct geist_buffer *const bufs[] = {sess->mla_kv_a,
                                         sess->mla_kv,
                                         sess->scratch_q,
                                         sess->scratch_k,
                                         sess->scratch_v,
                                         sess->mla_hx,
                                         sess->mla_hy};
    float                     *p[sizeof bufs / sizeof bufs[0]];
    bool                       ok = true;
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        ok = (p[i] = (float *) v->buffer_map(bufs[i])) != nullptr && ok;
    enum geist_status s = ok ? GEIST_OK : GEIST_E_BACKEND;
    if (s == GEIST_OK) {
        const float *a = p[0], *qr = p[1];
        float       *q = p[2], *k = p[3], *val = p[4];
        for (size_t t = 0; t < m; t++) {
            float *kt = k + t * hd;
            memcpy(kt, a + t * hd, pe * sizeof(float));
            latent_norm(rank, ctx->eps, a + t * hd + pe, L->mla_kv_a_norm, kt + pe);
            memcpy(val + t * hd, kt, hd * sizeof(float));
        }
        struct mla_heads_job j = {.w        = L->mla_k_b_w,
                                  .be       = ctx->be,
                                  .m        = m,
                                  .heads    = heads,
                                  .n_in     = kd - pe,
                                  .n_out    = rank,
                                  .src_row  = heads * kd,
                                  .src_head = kd,
                                  .src_off  = pe,
                                  .src      = qr,
                                  .hx       = p[5],
                                  .hy       = p[6]};
        s                      = mla_heads(&j);
        for (size_t t = 0; s == GEIST_OK && t < m; t++) {
            for (size_t h = 0; h < heads; h++) {
                float *qh = q + (t * heads + h) * hd;
                memcpy(qh, qr + (t * heads + h) * kd, pe * sizeof(float));
                memcpy(qh + pe, p[6] + (h * m + t) * rank, rank * sizeof(float));
            }
        }
        if (s != GEIST_OK)
            geist_backend_set_error(ctx->be, s, "%s: k_b has no row kernel", ARCH);
    }
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if (p[i] != nullptr)
            v->buffer_unmap(bufs[i]);
    return s;
}

/* Each output head's latent dims [pe, hd) through v_b[h], packed into
 * scratch_attn as [seq, heads * mla_v_dim] for o_proj. */
[[nodiscard]] static enum geist_status mla_absorb_out(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session *sess = ctx->sess;
    const struct geist_arch_config  *c    = &ctx->st->config;
    const struct geist_backend_vtbl *v    = ctx->v;
    const size_t      heads = ctx->st->n_q_heads, m = ctx->seq, hd = ctx->hd, vd = c->mla_v_dim;
    float            *o  = (float *) v->buffer_map(sess->scratch_attn);
    float            *hx = (float *) v->buffer_map(sess->mla_hx);
    float            *hy = (float *) v->buffer_map(sess->mla_hy);
    enum geist_status s =
            o != nullptr && hx != nullptr && hy != nullptr ? GEIST_OK : GEIST_E_BACKEND;
    if (s == GEIST_OK) {
        struct mla_heads_job j = {.w        = ctx->L->mla_v_b_w,
                                  .be       = ctx->be,
                                  .m        = m,
                                  .heads    = heads,
                                  .n_in     = c->mla_kv_rank,
                                  .n_out    = vd,
                                  .src_row  = heads * hd,
                                  .src_head = hd,
                                  .src_off  = c->mla_rope,
                                  .src      = o,
                                  .hx       = hx,
                                  .hy       = hy};
        s                      = mla_heads(&j);
        /* Every head has read its rows: o is free to take the packed ones. */
        for (size_t t = 0; s == GEIST_OK && t < m; t++)
            for (size_t h = 0; h < heads; h++)
                memcpy(o + (t * heads + h) * vd, hy + (h * m + t) * vd, vd * sizeof(float));
        if (s != GEIST_OK)
            geist_backend_set_error(ctx->be, s, "%s: v_b has no row kernel", ARCH);
    }
    struct geist_buffer *const bufs[] = {sess->scratch_attn, sess->mla_hx, sess->mla_hy};
    const float *const         ptrs[] = {o, hx, hy};
    for (size_t i = 0; i < 3; i++)
        if (ptrs[i] != nullptr)
            v->buffer_unmap(bufs[i]);
    return s;
}

/* ---- MLA on the device (#376) ------------------------------------------ */

/* [rows, cols] F32 view of b from float `off`, its rows `stride` apart. */
static struct geist_tensor
rows_view(struct geist_buffer *b, size_t off, size_t rows, size_t cols, size_t stride) {
    return (struct geist_tensor) {.buffer = b,
                                  .offset = off * sizeof(float),
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) stride, 1}};
}

/* dst = src * scale over two row views (1: a strided copy, 0: zeros). */
[[nodiscard]] static enum geist_status dev_copy(struct transformer_layer_forward_ctx *ctx,
                                                struct geist_tensor                   src,
                                                struct geist_tensor                   dst,
                                                float                                 scale) {
    return ctx->prims->scale_f32(ctx->be, &src, scale, &dst);
}

/* The latent's RMSNorm on the device: rows [pe, pe + rank) of each of
 * the m rows of `src` (row stride src_row) into `dst` (row stride
 * dst_row, from float dst_off); src and dst may be the same view. */
[[nodiscard]] static enum geist_status dev_latent_norm(struct transformer_layer_forward_ctx *ctx,
                                                       struct geist_buffer                  *src,
                                                       size_t               src_row,
                                                       struct geist_buffer *dst,
                                                       size_t               dst_off,
                                                       size_t               dst_row) {
    const struct geist_arch_config *c    = &ctx->st->config;
    const size_t                    rank = c->mla_kv_rank;
    struct geist_tensor             x    = rows_view(src, c->mla_rope, ctx->seq, rank, src_row);
    struct geist_tensor             y    = rows_view(dst, dst_off, ctx->seq, rank, dst_row);
    struct geist_tensor             w    = view_1d_at(ctx->L->mla_kv_a_norm_t.buffer,
                                                      (size_t) ctx->L->mla_kv_a_norm_t.offset,
                                                      (int64_t) rank);
    return ctx->prims->rmsnorm(ctx->be, &x, &w, ctx->eps, &y);
}

/* Decompressed: kv_a's latent normalised into mla_lat, kv_b, then K =
 * [k_pe | k_nope] and V = [v | 0] per head by strided copies. */
[[nodiscard]] static enum geist_status mla_kv_dev(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const struct geist_arch_config   *c    = &ctx->st->config;
    const size_t      rank = c->mla_kv_rank, pe = c->mla_rope, hd = ctx->hd, nope = hd - pe;
    const size_t      vd = c->mla_v_dim, heads = ctx->st->n_q_heads, kv_a = rank + pe, m = ctx->seq;
    const size_t      kv = nope + vd, mh = m * heads;
    enum geist_status s = dev_latent_norm(ctx, sess->mla_kv_a, kv_a, sess->mla_lat, 0, rank);
    if (s == GEIST_OK)
        s = blk_linear(ctx, sess->mla_lat, &L->mla_kv_b, &L->mla_kv_b_w, sess->mla_kv);
    if (s == GEIST_OK)
        s = dev_copy(ctx,
                     rows_view(sess->mla_kv, 0, mh, nope, kv),
                     rows_view(sess->scratch_k, pe, mh, nope, hd),
                     1.0f);
    /* k_pe, shared by the heads: one copy per head. */
    for (size_t h = 0; h < heads && s == GEIST_OK; h++)
        s = dev_copy(ctx,
                     rows_view(sess->mla_kv_a, 0, m, pe, kv_a),
                     rows_view(sess->scratch_k, h * hd, m, pe, heads * hd),
                     1.0f);
    if (s == GEIST_OK)
        s = dev_copy(ctx,
                     rows_view(sess->mla_kv, nope, mh, vd, kv),
                     rows_view(sess->scratch_v, 0, mh, vd, hd),
                     1.0f);
    /* V's padding: zeros, from as many of the head's kv_b outputs. */
    if (s == GEIST_OK && hd > vd)
        s = dev_copy(ctx,
                     rows_view(sess->mla_kv, 0, mh, hd - vd, kv),
                     rows_view(sess->scratch_v, vd, mh, hd - vd, hd),
                     0.0f);
    return s;
}

/* Decompressed: the first mla_v_dim of each output head packed into
 * mla_kv ([seq, heads * mla_v_dim], o_proj's input). */
[[nodiscard]] static enum geist_status mla_pack_out_dev(struct transformer_layer_forward_ctx *ctx) {
    const size_t mh = ctx->seq * ctx->st->n_q_heads, vd = ctx->st->config.mla_v_dim;
    return dev_copy(ctx,
                    rows_view(ctx->sess->scratch_attn, 0, mh, vd, ctx->hd),
                    rows_view(ctx->sess->mla_kv, 0, mh, vd, vd),
                    1.0f);
}

/* k_b or v_b of every head over the m * heads rows x into y: the heads
 * are linear_t_ids' experts, slot t * heads + h taking head h (mla_ids). */
[[nodiscard]] static enum geist_status mla_heads_dev(struct transformer_layer_forward_ctx *ctx,
                                                     const struct geist_tensor            *w,
                                                     const struct geist_weight            *w0,
                                                     struct geist_tensor                   x,
                                                     struct geist_tensor                   y) {
    const size_t        heads = ctx->st->n_q_heads;
    struct geist_tensor all   = w[0];
    all.shape[0] *= (int64_t) heads;
    const enum geist_status s = ctx->fused->linear_t_ids(
            ctx->be, heads, ctx->seq, heads, true, w0, &all, ctx->sess->mla_ids, &x, &y);
    if (s != GEIST_OK)
        geist_backend_set_error(ctx->be,
                                s,
                                "%s: layer %d: %s per head on the device: %s",
                                ARCH,
                                ctx->layer_idx,
                                w == ctx->L->mla_k_b ? "k_b" : "v_b",
                                geist_backend_errmsg(ctx->be));
    return s;
}

/* Absorbed: kv_a is in scratch_k already. K = [k_pe | rmsnorm(latent)],
 * V = K before RoPE, q = [q_pe | k_b[h] q_nope] per head. */
[[nodiscard]] static enum geist_status
mla_absorb_qkv_dev(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session *sess = ctx->sess;
    const struct geist_arch_config  *c    = &ctx->st->config;
    const size_t      rank = c->mla_kv_rank, pe = c->mla_rope, hd = ctx->hd, kd = c->mla_k_dim;
    const size_t      m = ctx->seq, mh = m * ctx->st->n_q_heads;
    enum geist_status s = dev_latent_norm(ctx, sess->scratch_k, hd, sess->scratch_k, pe, hd);
    if (s == GEIST_OK)
        s = dev_copy(ctx,
                     rows_view(sess->scratch_k, 0, m, hd, hd),
                     rows_view(sess->scratch_v, 0, m, hd, hd),
                     1.0f);
    if (s == GEIST_OK)
        s = dev_copy(ctx,
                     rows_view(sess->mla_kv, 0, mh, pe, kd),
                     rows_view(sess->scratch_q, 0, mh, pe, hd),
                     1.0f);
    if (s == GEIST_OK)
        s = mla_heads_dev(ctx,
                          ctx->L->mla_k_b,
                          &ctx->L->mla_k_b_w[0],
                          rows_view(sess->mla_kv, pe, mh, kd - pe, kd),
                          rows_view(sess->scratch_q, pe, mh, rank, hd));
    return s;
}

/* Absorbed: each output head's latent dims through v_b[h] into mla_hy
 * ([seq, heads * mla_v_dim], o_proj's input). */
[[nodiscard]] static enum geist_status
mla_absorb_out_dev(struct transformer_layer_forward_ctx *ctx) {
    const struct geist_arch_config *c  = &ctx->st->config;
    const size_t                    mh = ctx->seq * ctx->st->n_q_heads, vd = c->mla_v_dim;
    return mla_heads_dev(
            ctx,
            ctx->L->mla_v_b,
            &ctx->L->mla_v_b_w[0],
            rows_view(ctx->sess->scratch_attn, c->mla_rope, mh, c->mla_kv_rank, ctx->hd),
            rows_view(ctx->sess->mla_hy, 0, mh, vd, vd));
}

/* Each step runs only while s is GEIST_OK. */
[[nodiscard]] static enum geist_status
deepseek2_attention_block(struct transformer_layer_forward_ctx *ctx) {
    struct transformer_arch_session  *sess = ctx->sess;
    struct transformer_layer_weights *L    = ctx->L;
    const struct geist_arch_config   *c    = &ctx->st->config;
    struct geist_buffer              *x    = sess->scratch_normed;
    const bool                        dev  = mla_device(ctx->st);
    const bool                        ab   = c->mla_absorbed;
    /* Absorbed: q before the absorption goes to mla_kv; on the device kv_a
     * is K's row as it stands (one KV head of [k_pe | latent]). */
    struct geist_buffer *q   = ab ? sess->mla_kv : sess->scratch_q;
    struct geist_buffer *kva = ab && dev ? sess->scratch_k : sess->mla_kv_a;
    /* o_proj's input: [seq, heads * mla_v_dim]. */
    struct geist_buffer *att = !dev ? sess->scratch_attn : ab ? sess->mla_hy : sess->mla_kv;

    enum geist_status s = blk_attn_norm(ctx, x);
    if (s == GEIST_OK && c->mla_q_rank > 0) {
        s = blk_linear(ctx, x, &L->mla_q_a, &L->mla_q_a_w, sess->mla_q_a);
        if (s == GEIST_OK)
            s = blk_rmsnorm(ctx, c->mla_q_rank, sess->mla_q_a, &L->mla_q_a_norm, sess->mla_q_a);
        if (s == GEIST_OK)
            s = blk_linear(ctx, sess->mla_q_a, &L->q_proj, &L->q_proj_w, q);
    } else if (s == GEIST_OK) {
        s = blk_linear(ctx, x, &L->q_proj, &L->q_proj_w, q);
    }
    if (s == GEIST_OK)
        s = blk_linear(ctx, x, &L->mla_kv_a, &L->mla_kv_a_w, kva);
    if (s == GEIST_OK)
        s = ab ? (dev ? mla_absorb_qkv_dev(ctx) : mla_absorb_qkv(ctx))
               : (dev ? mla_kv_dev(ctx) : mla_kv(ctx));
    if (s == GEIST_OK)
        s = blk_rope_qk(ctx, c->attn_q_scale);
    if (s == GEIST_OK)
        s = blk_attend(ctx, true);
    if (s == GEIST_OK)
        s = ab ? (dev ? mla_absorb_out_dev(ctx) : mla_absorb_out(ctx))
               : (dev ? mla_pack_out_dev(ctx) : mla_pack_out(ctx));
    if (s == GEIST_OK)
        s = blk_linear(ctx, att, &L->o_proj, &L->o_proj_w, sess->scratch_o);
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
