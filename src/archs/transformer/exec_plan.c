/*
 * src/archs/transformer/exec_plan.c - per-layer transformer execution plan.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "exec_plan.h"
#include "arch_state.h"

#include "heap.h"

#include <geist.h>
#include <string.h>

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
