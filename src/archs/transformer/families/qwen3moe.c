/*
 * src/archs/transformer/families/qwen3moe.c - the Qwen3-MoE family unit
 * (#48).
 *
 * Layer: ARCHITECTURE.
 *
 * Qwen3's attention (GQA, per-head QK-norm before NEOX RoPE, head_dim from
 * metadata) with a mixture-of-experts FFN: a router over
 * qwen3moe.expert_count SwiGLU experts of width
 * qwen3moe.expert_feed_forward_length, the expert_used_count best per
 * token, their weights renormalised to sum 1 (Qwen3-MoE's norm_topk_prob;
 * llama.cpp builds the same graph for this architecture).
 *
 *   h_post_attn = h_in + o_proj(attention(rope(qk_norm(qkv(norm(h_in))))))
 *   h_out       = blk_moe_ffn(h_post_attn)
 *
 * The experts run on CPU backends: blk_moe_ffn routes and gathers rows on
 * the host. A GPU backend is refused at load until it has a placement that
 * keeps the gathers off the host.
 *
 * Removing the family is this file, its registry lines in arch_family.c and
 * arch_family.h, and its line in mk/common.mk.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../arch_family.h"
#include "../forward/blocks.h"
#include "../moe_gpu.h"
#include "../rotation.h"
#include "../weight_load.h"

#include "gguf_reader.h"
#include "heap.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void populate_qwen3moe(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    st->config.has_qk_norms   = true;
    st->config.ffn_activation = GEIST_FFN_SWIGLU;
    st->config.moe_norm_topk  = true;

    family_populate_common(gguf, "qwen3moe", nullptr, st);
    st->vocab_size = family_vocab_size(gguf, "qwen3moe", nullptr);
    uint32_t u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.expert_count", &u))
        st->config.moe_n_experts = u;
    if (gguf_get_meta_u32(gguf, "qwen3moe.expert_used_count", &u))
        st->config.moe_n_used = u;
    bool b;
    if (gguf_get_meta_bool(gguf, "qwen3moe.expert_weights_norm", &b))
        st->config.moe_norm_topk = b; /* absent: Qwen3-MoE's norm_topk_prob */
}

static bool populate_layers_qwen3moe(struct transformer_arch_state *st) {
    struct gguf_ctx *g        = (struct gguf_ctx *) st->gguf;
    uint32_t         head_dim = 0, expert_ff = 0;
    float            freq_base = 1000000.0f;
    if (g != nullptr) {
        gguf_get_meta_u32(g, "qwen3moe.attention.key_length", &head_dim);
        gguf_get_meta_u32(g, "qwen3moe.expert_feed_forward_length", &expert_ff);
        gguf_get_meta_f32(g, "qwen3moe.rope.freq_base", &freq_base);
    }
    if (head_dim == 0 && st->n_q_heads > 0) {
        head_dim = (uint32_t) (st->d_model / st->n_q_heads);
    }
    /* A router needs at least one expert to pick and no more than exist. */
    if (head_dim == 0 || expert_ff == 0 || st->config.moe_n_experts == 0 ||
        st->config.moe_n_used == 0 || st->config.moe_n_used > st->config.moe_n_experts ||
        st->config.moe_n_experts > GEIST_MOE_MAX_EXPERTS) {
        return false;
    }
    family_uniform_layers(st, head_dim, expert_ff /* one expert's width */, freq_base);
    return true;
}

/* The router and the three expert tensors of one layer. */
[[nodiscard]] static enum geist_status wire_layer_qwen3moe(struct transformer_arch_state    *st,
                                                           struct gguf_ctx                  *gguf,
                                                           struct transformer_layer_weights *L) {
    struct geist_backend *be = st->backend;
    if (strncmp(be->desc->name, "cpu_", 4) != 0) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "qwen3moe: the experts run on CPU backends only, not %s (#48)",
                                be->desc->name);
        return GEIST_E_UNSUPPORTED;
    }
    const size_t n_exp = st->config.moe_n_experts;
    const size_t d     = st->d_model;
    const size_t ff    = L->intermediate;
    char         name[64];

    /* A router this unit does not implement is refused, never approximated:
     * llama.cpp's gating enum (1 = softmax) and a shared expert. */
    uint32_t gating = 1;
    if (gguf_get_meta_u32(gguf, "qwen3moe.expert_gating_func", &gating) && gating != 1) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "qwen3moe: expert_gating_func %u is not softmax (1)",
                                gating);
        return GEIST_E_UNSUPPORTED;
    }
    snprintf(name, sizeof name, "blk.%d.ffn_gate_inp_shexp.weight", L->layer_idx);
    if (gguf_get_tensor(gguf, name) != nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "qwen3moe: shared expert '%s' is not supported", name);
        return GEIST_E_UNSUPPORTED;
    }

    /* The router's top-k is a discrete choice: it keeps its own precision. */
    snprintf(name, sizeof name, "blk.%d.ffn_gate_inp.weight", L->layer_idx);
    enum geist_status s =
            load_layer_matrix(st, gguf, L, name, n_exp, d, true, &L->moe_router, &L->moe_router_w);
    if (s == GEIST_OK) {
        snprintf(name, sizeof name, "blk.%d.ffn_gate_exps.weight", L->layer_idx);
        s = load_layer_experts(st, gguf, L, name, n_exp, ff, d, &L->moe_gate, &L->moe_gate_w);
    }
    if (s == GEIST_OK) {
        snprintf(name, sizeof name, "blk.%d.ffn_up_exps.weight", L->layer_idx);
        s = load_layer_experts(st, gguf, L, name, n_exp, ff, d, &L->moe_up, &L->moe_up_w);
    }
    if (s == GEIST_OK) {
        snprintf(name, sizeof name, "blk.%d.ffn_down_exps.weight", L->layer_idx);
        s = load_layer_experts(st, gguf, L, name, n_exp, d, ff, &L->moe_down, &L->moe_down_w);
    }
    return s;
}

/* ---- Session state ---------------------------------------------------- */

/* Router logits, the gathered rows of one expert and the weighted sum, plus
 * the host-side routing (blk_moe_ffn). */
[[nodiscard]] static enum geist_status
qwen3moe_session_alloc(struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st    = sess->model;
    struct geist_backend                *be    = st->backend;
    const size_t                         n_exp = st->config.moe_n_experts;
    const size_t                         n_sel = sess->m_max * st->config.moe_n_used;
    enum geist_status                    s =
            transformer_scratch_alloc(be, sess->m_max * n_exp * sizeof(float), &sess->moe_logits);
    if (s == GEIST_OK)
        s = transformer_scratch_alloc(
                be, sess->m_max * st->d_model * sizeof(float), &sess->moe_acc);
    if (s != GEIST_OK)
        return s;
    size_t inter = 0;
    for (size_t li = 0; li < st->n_layers; li++)
        inter = st->layers[li].intermediate > inter ? st->layers[li].intermediate : inter;
    sess->moe_ids       = heap_alloc_array_aligned(uint32_t, n_sel);
    sess->moe_wts       = heap_alloc_array_aligned(float, n_sel);
    sess->moe_rows      = heap_alloc_array_aligned(uint32_t, n_sel);
    sess->moe_row_start = heap_alloc_array_aligned(size_t, n_exp + 1);
    sess->moe_xg        = heap_alloc_array_aligned(float, n_sel * st->d_model);
    sess->moe_y         = heap_alloc_array_aligned(float, n_sel * st->d_model);
    sess->moe_g         = heap_alloc_array_aligned(float, n_sel *inter);
    sess->moe_u         = heap_alloc_array_aligned(float, n_sel *inter);
    if (sess->moe_ids == nullptr || sess->moe_wts == nullptr || sess->moe_rows == nullptr ||
        sess->moe_row_start == nullptr || sess->moe_xg == nullptr || sess->moe_y == nullptr ||
        sess->moe_g == nullptr || sess->moe_u == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "transformer: MoE routing state alloc failed");
        return GEIST_E_OOM;
    }
    moe_gpu_session_alloc(sess);
    return GEIST_OK;
}

static void qwen3moe_session_free(struct transformer_arch_session *sess) {
    moe_gpu_session_free(sess);
    void *host[] = {sess->moe_xg,
                    sess->moe_y,
                    sess->moe_g,
                    sess->moe_u,
                    sess->moe_ids,
                    sess->moe_wts,
                    sess->moe_rows,
                    sess->moe_row_start};
    for (size_t i = 0; i < sizeof host / sizeof host[0]; i++)
        safe_free(&host[i]);
    sess->moe_xg = sess->moe_y = sess->moe_g = sess->moe_u = nullptr;
    sess->moe_wts                                          = nullptr;
    sess->moe_ids = sess->moe_rows = nullptr;
    sess->moe_row_start            = nullptr;
    transformer_buffer_release(sess->model->backend, &sess->moe_logits);
    transformer_buffer_release(sess->model->backend, &sess->moe_acc);
}

/* Qwen3's attention, unchanged. Each step runs only while s is GEIST_OK. */
[[nodiscard]] static enum geist_status
qwen3moe_attention_block(struct transformer_layer_forward_ctx *ctx) {
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

const struct transformer_family transformer_family_qwen3moe = {
        .name            = "qwen3moe",
        .populate        = populate_qwen3moe,
        .populate_layers = populate_layers_qwen3moe,
        .wire_layer      = wire_layer_qwen3moe,
        .session_alloc   = qwen3moe_session_alloc,
        .session_free    = qwen3moe_session_free,
        .attention_block = qwen3moe_attention_block,
        .ffn_block       = blk_moe_ffn,
        /* An MoE amplifies KV quantization error about 16x over a dense
         * model of the same quantization; F16 matches FP32 against the exact
         * reference (benchmark/results/QWEN3MOE.md). */
        .kv_auto = GEIST_KV_F16,
};
