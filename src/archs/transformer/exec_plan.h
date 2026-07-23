/*
 * src/archs/transformer/exec_plan.h - per-layer transformer execution plan.
 *
 * Layer: ARCHITECTURE.
 *
 * Built once after weight loading. The first iteration stores resolved
 * booleans/enums that the hot path can consume without re-deriving model
 * family decisions. Later iterations can replace these fields with bound
 * function pointers for fused CPU kernels.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_EXEC_PLAN_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_EXEC_PLAN_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/exec_plan.h is internal to the architecture layer."
#endif

#include "arch_config.h"

#include <geist.h>

#include <stdbool.h>
#include <stddef.h>

struct transformer_arch_state;
struct transformer_arch_session;
struct transformer_layer_forward_ctx;

/* No KV-append / attention-kind state here: those are derived from
 * st->sess->kv_{kivi,int8,f16}_enabled at hot-path entry. A cached copy
 * would be baked from default_sess at construction time and become stale
 * when transformer_session_attach swaps in a session with a different KV
 * mode (see review #7 / V6). */
struct transformer_layer_exec_plan {
    int                            kv_src;
    bool                           compute_kv;
    bool                           apply_gemma_attn_norms;
    bool                           apply_sub_ln;
    bool                           apply_ple;
    bool                           rope_interleaved;
    enum geist_ffn_activation_kind ffn_activation;
};

enum geist_status transformer_exec_plan_build(struct transformer_arch_state *st);
void              transformer_exec_plan_destroy(struct transformer_arch_state *st);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_EXEC_PLAN_H */
