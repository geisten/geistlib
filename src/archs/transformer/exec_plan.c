/*
 * src/archs/transformer/exec_plan.c - per-layer transformer execution plan.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "exec_plan.h"
#include "arch_state.h"

#include "error.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdlib.h>
#include <string.h>

/* Env gates for the FFN fusions, read once at plan build. GEGLU epilogue
 * fusion with the AWQ scale folded in is on unless GEIST_FFN_FUSED_SCALE=0;
 * the whole-FFN tile kernel is opt-in via GEIST_FFN_TILE_FUSION=1. */
static bool ffn_fused_scale_enabled(void) {
    const char *env = getenv("GEIST_FFN_FUSED_SCALE");
    return env == nullptr || env[0] != '0';
}

static bool ffn_tile_fusion_enabled(void) {
    const char *env = getenv("GEIST_FFN_TILE_FUSION");
    return env != nullptr && env[0] == '1';
}

static bool probe(struct geist_backend *be, struct geist_fusion_query q) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    /* The probed slot must actually exist — a probe answering for a
     * nullptr kernel would bind a crash. */
    bool have = false;
    switch (q.op) {
    case GEIST_FUSED_GELU_TANH_MUL:
        have = fused->gelu_tanh_mul != nullptr;
        break;
    case GEIST_FUSED_SILU_MUL:
        have = fused->silu_mul != nullptr;
        break;
    case GEIST_FUSED_GELU_TANH_MUL_SCALED:
        have = fused->gelu_tanh_mul_scaled != nullptr;
        break;
    case GEIST_FUSED_FFN_GEGLU_Q4Q6_MN:
        have = fused->ffn_geglu_q4q6_mN != nullptr;
        break;
    case GEIST_FUSED_FFN_GATE_UP:
        have = fused->ffn_gate_up != nullptr;
        break;
    case GEIST_FUSED_FFN_NORM_GATE_UP:
        have = fused->ffn_norm_gate_up != nullptr;
        break;
    case GEIST_FUSED_RMSNORM_ADD:
        have = fused->rmsnorm_add != nullptr;
        break;
    case GEIST_FUSED_ATTN_QKV_PREP:
        have = fused->attn_qkv_prep != nullptr;
        break;
    case GEIST_FUSED_PLE_BLOCK:
        have = fused->ple_block != nullptr;
        break;
    case GEIST_FUSED_EMBEDDING_LOOKUP_SCALED:
        have = fused->embedding_lookup_scaled != nullptr;
        break;
    case GEIST_FUSED_ARGMAX_F32:
        have = fused->argmax_f32 != nullptr;
        break;
    }
    return have && fused->supported != nullptr && fused->supported(be, &q);
}

enum geist_status transformer_exec_plan_build(struct transformer_arch_state *st) {
    if (st == nullptr || st->layers == nullptr || st->n_layers == 0) {
        return GEIST_E_INVALID_ARG;
    }
    transformer_exec_plan_destroy(st);

    st->layer_plans = heap_alloc_aligned(st->n_layers * sizeof(*st->layer_plans),
                                         alignof(struct transformer_layer_exec_plan));
    if (st->layer_plans == nullptr) {
        return GEIST_E_OOM;
    }
    memset(st->layer_plans, 0, st->n_layers * sizeof(*st->layer_plans));

    for (size_t i = 0; i < st->n_layers; i++) {
        const struct transformer_layer_weights *L = &st->layers[i];
        struct transformer_layer_exec_plan     *P = &st->layer_plans[i];
        P->kv_src     = L->is_kv_shared
                                ? (L->is_full ? st->config.kv_full_src : st->config.kv_sliding_src)
                                : (int) i;
        P->compute_kv = !L->is_kv_shared;
        P->apply_gemma_attn_norms = st->config.has_gemma_attn_norms;
        P->apply_qk_norms         = st->config.has_qk_norms;
        P->apply_sub_ln           = st->config.has_sub_ln;
        P->apply_bitlinear_subln  = st->config.has_bitlinear_subln;
        P->apply_ple              = st->config.has_ple;
        P->rope_interleaved       = st->config.rope_interleaved;
        P->ffn_activation         = st->config.ffn_activation;

        /* ---- FFN-front fusion binding (see exec_plan.h). Prefill is
         * probed at the largest m any session may use, so the answer
         * covers every session on this model. */
        struct geist_backend *be = st->backend;
        /* Prefill probes answer for any m ≤ the backend's per-call row
         * limit (caps.max_m; 0 = uncapped — probe at the model m_max). */
        const size_t m_cap        = (be->desc != nullptr && be->desc->caps.max_m > 0)
                                            ? be->desc->caps.max_m
                                            : st->m_max;
        const bool   geglu        = st->config.ffn_activation == GEIST_FFN_GEGLU;
        const bool   has_sub_norm = (P->apply_sub_ln && L->ffn_sub_norm.buffer != nullptr &&
                                     st->runtime_flags.bitnet_sub_ln_enabled) ||
                                    P->apply_bitlinear_subln;
        /* Every FFN-front fusion folds the pre-FFN rmsnorm into the gate/up
         * matmul and feeds both projections the SAME row. has_bitlinear_subln
         * gives gate and up their own gamma, so there is no shared row to
         * fold — the fusions are not slower here, they are wrong. */
        const bool                      bl_subln = P->apply_bitlinear_subln;
        const struct geist_fusion_query base     = {
                .d_model = st->d_model,
                .inter   = L->intermediate,
                .gate_w  = &L->gate_proj_w,
                .up_w    = &L->up_proj_w,
                .down_w  = &L->down_proj_w,
        };
        struct geist_fusion_query q;

        q    = base;
        q.op = GEIST_FUSED_FFN_NORM_GATE_UP;
        q.m  = 1;
        P->fuse_ffn_norm_gate_up_m1 =
                geglu && !bl_subln && L->down_awq_inv_scale == nullptr && probe(be, q);

        q    = base;
        q.op = GEIST_FUSED_FFN_GATE_UP;
        q.m  = 1;
        P->fuse_ffn_gate_up_m1 =
                geglu && !bl_subln && L->down_awq_inv_scale == nullptr && probe(be, q);

        q                         = base;
        q.op                      = GEIST_FUSED_FFN_GEGLU_Q4Q6_MN;
        q.m                       = m_cap;
        P->fuse_ffn_geglu_tile_mN = geglu && !has_sub_norm &&
                                    !st->runtime_flags.dump_act_sparsity &&
                                    ffn_tile_fusion_enabled() && probe(be, q);

        q                       = base;
        q.op                    = GEIST_FUSED_GELU_TANH_MUL_SCALED;
        q.m                     = m_cap;
        P->fuse_gelu_mul_scaled = L->down_awq_inv_scale != nullptr && !has_sub_norm &&
                                  ffn_fused_scale_enabled() && probe(be, q);

        q                = base;
        q.op             = GEIST_FUSED_GELU_TANH_MUL;
        q.m              = m_cap;
        P->fuse_gelu_mul = probe(be, q);

        q                = base;
        q.op             = GEIST_FUSED_SILU_MUL;
        q.m              = m_cap;
        P->fuse_silu_mul = probe(be, q);

        q                   = base;
        q.op                = GEIST_FUSED_RMSNORM_ADD;
        q.m                 = m_cap;
        P->fuse_rmsnorm_add = probe(be, q);

        /* The post-attention norm+residual is the same op at a different
         * site: d_model rows, none of the FFN weights. Probed separately
         * rather than reusing fuse_rmsnorm_add above (#352). */
        q = (struct geist_fusion_query) {
                .op = GEIST_FUSED_RMSNORM_ADD, .m = m_cap, .d_model = st->d_model};
        P->fuse_attn_rmsnorm_add = probe(be, q);

        /* The PLE gate epilogue runs on hidden_per_layer-wide rows. */
        q = (struct geist_fusion_query) {
                .op = GEIST_FUSED_GELU_TANH_MUL, .m = m_cap, .d_model = st->hidden_per_layer};
        P->fuse_ple_gelu_mul = P->apply_ple && probe(be, q);

        /* Session KV-mode conditions (kivi/int8 off, f16-vs-f32 cache
         * views) stay inline at the call site — sess->kv_*_enabled is
         * the per-session overlay, frozen at session_alloc. */
        q                     = base;
        q.op                  = GEIST_FUSED_ATTN_QKV_PREP;
        q.m                   = m_cap;
        q.head_dim            = L->head_dim;
        q.n_q_heads           = st->n_q_heads;
        q.n_kv_heads          = st->n_kv_heads;
        P->fuse_attn_qkv_prep = P->apply_gemma_attn_norms && !P->rope_interleaved && probe(be, q);

        struct geist_fusion_query pq = {
                .op      = GEIST_FUSED_PLE_BLOCK,
                .d_model = st->d_model,
                .inter   = st->hidden_per_layer,
                .gate_w  = &L->per_layer_gate_w,
                .up_w    = &L->per_layer_proj_w, /* proj rides the up slot */
        };
        pq.m                 = 1;
        P->fuse_ple_block_m1 = P->apply_ple && probe(be, pq);
        pq.m                 = m_cap;
        P->fuse_ple_block_mN = P->apply_ple && probe(be, pq);
    }

    /* ---- Model-level fusion decisions (lookup tables + greedy head). */
    {
        struct geist_backend     *be          = st->backend;
        struct geist_fusion_query q           = {.op = GEIST_FUSED_EMBEDDING_LOOKUP_SCALED, .m = 1};
        q.d_model                             = st->d_model;
        q.table_dtype                         = st->embed_table.dtype;
        st->model_fusions.embed_lookup_scaled = probe(be, q);
        q.d_model                             = st->ple_out;
        q.table_dtype                         = st->ple_table.dtype;
        st->model_fusions.ple_lookup_scaled   = st->config.has_ple && probe(be, q);
        q                                     = (struct geist_fusion_query) {
                .op = GEIST_FUSED_ARGMAX_F32, .m = 1, .d_model = st->vocab_size};
        st->model_fusions.argmax = probe(be, q);

        /* Optional primitives: a plain null test, bound once (#352). There
         * is no `supported` probe for prims — the pointer IS the
         * capability. */
        const struct geist_backend_primitives *prims = be->desc->prims;
        st->model_fusions.prim_scale_f32             = prims->scale_f32 != nullptr;
        st->model_fusions.prim_silu                  = prims->silu != nullptr;
        st->model_fusions.backend_buffer_copy        = be->desc->vtbl->buffer_copy != nullptr;

        /* relu_squared has no bound alternative: it cannot be composed from
         * the other prims (there is no relu, and no max), and the two call
         * sites in layer_ffn.c invoke it UNCONDITIONALLY. Metal declares it
         * nullptr. A model whose FFN activation selects it, on a backend
         * that lacks it, was therefore a null function-pointer call in the
         * per-layer path — refuse it here instead, where the message can
         * name the cause. */
        if (geist_ffn_needs_relu_squared(st->config.ffn_activation) &&
            prims->relu_squared == nullptr) {
            geist_error_set_create_time(
                    GEIST_E_UNSUPPORTED,
                    "transformer_exec_plan_build",
                    "backend '%s' does not implement the relu_squared primitive, which this "
                    "model's FFN activation requires (gguf 'activation' = squared_relu / relu2 / "
                    "gated_squared_relu, or architecture bitnet-b1.58)",
                    geist_backend_name(be));
            return GEIST_E_UNSUPPORTED;
        }
    }
    return GEIST_OK;
}

void transformer_exec_plan_destroy(struct transformer_arch_state *st) {
    if (st == nullptr || st->layer_plans == nullptr) {
        return;
    }
    void *p = st->layer_plans;
    safe_free(&p);
    st->layer_plans = nullptr;
}
