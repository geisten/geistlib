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
 * sess->kv_{kivi,int8,f16}_enabled at hot-path entry. A cached copy
 * would be baked from default_sess at construction time and become stale
 * when transformer_session_attach swaps in a session with a different KV
 * mode (see review #7 / V6). */
struct transformer_layer_exec_plan {
    int                            kv_src;
    bool                           compute_kv;
    bool                           apply_gemma_attn_norms;
    bool                           apply_qk_norms;
    bool                           apply_sub_ln;
    bool                           apply_ple;
    bool                           rope_interleaved;
    enum geist_ffn_activation_kind ffn_activation;

    /* ---- Probe-and-bind: FFN-front fusion decisions, made once here
     * from the backend's fused->supported probe plus the layer-constant
     * conditions (activation kind, AWQ scales, SubLN, env gates). The
     * hot path branches on these bits and calls the fused op
     * UNCONDITIONALLY — probe true means the kernel must succeed
     * (geist_backend.h contract), so there is no per-call
     * GEIST_E_UNSUPPORTED negotiation on bound stages. _m1 = decode
     * (seq==1); _mN = prefill (any seq ≤ the session m cap). */
    bool fuse_ffn_norm_gate_up_m1; /* fused rmsnorm + gate/up front */
    bool fuse_ffn_gate_up_m1;      /* fused gate/up front (norm separate) */
    bool fuse_ffn_geglu_tile_mN;   /* whole-FFN q4/q6 tile kernel (prefill) */
    bool fuse_gelu_mul_scaled;     /* GEGLU epilogue with AWQ scale, any m */
    bool fuse_gelu_mul;            /* GEGLU epilogue, any m */
    bool fuse_silu_mul;            /* SwiGLU epilogue, any m */
    bool fuse_rmsnorm_add;         /* post-norm residual, any m */
    bool fuse_attn_qkv_prep;       /* per-head norms + RoPE + KV append; the
                                    * session KV-mode conditions stay inline
                                    * (sess->kv_*_enabled is the session
                                    * overlay) */
    bool fuse_ple_block_m1;        /* fused PLE block, decode */
    bool fuse_ple_block_mN;        /* fused PLE block, prefill */

    /* Two more sites bound in batch 2 (#352). Each needs its OWN probe:
     * geist_fusion_query carries shapes and the layer's weight pointers, so
     * a bit probed for the FFN geometry does not answer for the attention
     * post-norm or the PLE gate, even where a given backend happens to
     * accept any geometry today. */
    bool fuse_attn_rmsnorm_add; /* post-attn norm+residual, d_model rows */
    bool fuse_ple_gelu_mul;     /* PLE gate epilogue, hidden_per_layer rows */
};

/* Model-level fusion decisions (not per-layer): lookup tables and the
 * greedy head. Filled by transformer_exec_plan_build alongside the
 * per-layer plans. */
struct transformer_model_fusion_plan {
    bool embed_lookup_scaled; /* embed_table on-device lookup+scale */
    bool ple_lookup_scaled;   /* ple_table on-device lookup+scale */
    bool argmax;              /* device argmax over [1, vocab] logits */

    /* ---- Probe-and-bind for the OPTIONAL primitives (#352). Same
     * contract as the fused ops above: bound true means the hot path calls
     * the prim UNCONDITIONALLY and propagates its status; bound false
     * selects the host path, decided once here rather than re-tested at
     * every call. Backend capability, so model-level rather than per-layer.
     *
     * These replaced call sites of the form
     *
     *     if (prims->op == nullptr || prims->op(...) != GEIST_OK) { host }
     *
     * which conflate "absent" with "failed". The second arm re-did work the
     * op may already have done — layer.c's PLE combine carries a comment
     * about exactly that hazard for `add`, while the `scale_f32` sites had
     * the same shape and no such guard. Probe true ⇒ must succeed
     * (geist_backend.h), so a failure is now returned, not papered over. */
    bool prim_scale_f32; /* Metal + Vulkan; the CPU backends take the host loop */
    bool prim_silu;      /* every in-tree backend; gelu_tanh is the bound alternative */
};

enum geist_status transformer_exec_plan_build(struct transformer_arch_state *st);
void              transformer_exec_plan_destroy(struct transformer_arch_state *st);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_EXEC_PLAN_H */
