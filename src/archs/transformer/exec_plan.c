/*
 * src/archs/transformer/exec_plan.c - per-layer transformer execution plan.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "exec_plan.h"
#include "arch_state.h"

#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include "quant.h"

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
        P->apply_sub_ln           = st->config.has_sub_ln;
        P->apply_ple              = st->config.has_ple;
        P->rope_interleaved       = st->config.rope_interleaved;
        P->ffn_activation         = st->config.ffn_activation;

        /* ---- FFN-front fusion binding (see exec_plan.h). Prefill is
         * probed at the largest m any session may use, so the answer
         * covers every session on this model. */
        struct geist_backend *be           = st->backend;
        const size_t          m_cap        = (be->desc != nullptr && be->desc->caps.batched_submit)
                                                     ? 512u
                                                     : (size_t) GEIST_QUANT_M_CAP;
        const bool            geglu        = st->config.ffn_activation == GEIST_FFN_GEGLU;
        const bool            has_sub_norm = P->apply_sub_ln && L->ffn_sub_norm.buffer != nullptr &&
                                             st->runtime_flags.bitnet_sub_ln_enabled;
        const struct geist_fusion_query base = {
                .d_model = st->d_model,
                .inter   = L->intermediate,
                .gate_w  = &L->gate_proj_w,
                .up_w    = &L->up_proj_w,
                .down_w  = &L->down_proj_w,
        };
        struct geist_fusion_query q;

        q                           = base;
        q.op                        = GEIST_FUSED_FFN_NORM_GATE_UP;
        q.m                         = 1;
        P->fuse_ffn_norm_gate_up_m1 = geglu && L->down_awq_inv_scale == nullptr && probe(be, q);

        q                      = base;
        q.op                   = GEIST_FUSED_FFN_GATE_UP;
        q.m                    = 1;
        P->fuse_ffn_gate_up_m1 = geglu && L->down_awq_inv_scale == nullptr && probe(be, q);

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
