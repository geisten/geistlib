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

typedef enum geist_status (*transformer_layer_stage_fn)(struct transformer_layer_forward_ctx *ctx);

/* No KV-append / attention-kind enum here: those are derived from
 * st->sess->kv_{kivi,int8}_enabled at hot-path entry. The plan would
 * otherwise be baked from default_sess at construction time and become
 * stale when transformer_session_attach swaps in a session with a
 * different KV mode (see review #7 / V6). When a later iteration wants
 * to cache the dispatch for fused kernels, it must be rebuilt on every
 * session attach. */
struct transformer_layer_exec_plan {
    int                            kv_src;
    bool                           compute_kv;
    bool                           apply_gemma_attn_norms;
    bool                           apply_sub_ln;
    bool                           apply_ple;
    bool                           rope_interleaved;
    enum geist_ffn_activation_kind ffn_activation;
    transformer_layer_stage_fn     run_attention_block;
    transformer_layer_stage_fn     run_ffn_block;
    transformer_layer_stage_fn     run_ple_or_copy;
};

enum transformer_kv_append_kind {
    TRANSFORMER_KV_APPEND_FP32 = 0,
    TRANSFORMER_KV_APPEND_INT8,
    TRANSFORMER_KV_APPEND_KIVI,
};

enum transformer_attention_kind {
    TRANSFORMER_ATTENTION_FP32 = 0,
    TRANSFORMER_ATTENTION_INT8,
    TRANSFORMER_ATTENTION_KIVI,
};

/* Session-dependent dispatch choices. Unlike layer_plans, this must be
 * rebuilt for every session because KV representation is a session option.
 */
struct transformer_session_exec_plan {
    bool                            kv_int8_enabled;
    bool                            kv_kivi_enabled;
    bool                            kv_f16_enabled;
    enum transformer_kv_append_kind kv_append_kind;
    enum transformer_attention_kind attention_kind;
};

enum geist_status transformer_exec_plan_build(struct transformer_arch_state *st);
void              transformer_exec_plan_destroy(struct transformer_arch_state *st);
void              transformer_session_exec_plan_build(struct transformer_arch_session *sess);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_EXEC_PLAN_H */
