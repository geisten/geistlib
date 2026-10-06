/*
 * src/backends/cpu_neon/attention_int4.c — attention over the packed INT4
 * KV cache with FEAT_DotProd: cpu_neon's fused->attention_kv_int4.
 *
 * Layer: BACKEND (cpu_neon).
 *
 * The loop is in attention_kv.h, shared with the INT8 cache; each cache row
 * is unpacked from head_dim / 2 bytes (int4_kv.h) before the int8 dot. The
 * architecture's portable loop is the decomposed twin. Without
 * FEAT_DotProd this file is empty and the slot stays null.
 */

#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
constexpr bool NKV_INT4 = true;
#include "attention_kv.h" /* the loop and the checks, shared with attention_int8.c */

bool cpu_neon_attention_kv_int4_supported(const struct geist_fusion_query *q) {
    return q != nullptr && q->head_dim >= 2 && q->head_dim % 2 == 0 &&
           q->head_dim <= NKV_HEAD_DIM_MAX && q->n_kv_heads >= 1 && q->n_q_heads >= q->n_kv_heads &&
           q->n_q_heads % q->n_kv_heads == 0;
}

enum geist_status cpu_neon_attention_kv_int4(struct geist_backend                 *be,
                                             const struct geist_attention_kv_args *args) {
    if (be == nullptr || args == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    return nkv_attention(be,
                         args->q,
                         args->k,
                         args->k_scale,
                         args->v,
                         args->v_scale,
                         args->out,
                         args->q_offset,
                         args->sliding_window);
}

#endif /* __ARM_NEON && __ARM_FEATURE_DOTPROD */
