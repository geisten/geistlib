/*
 * src/backends/common/host_ops.h — the view checks and small ops every CPU
 * backend runs the same way.
 *
 * Layer: BACKEND (shared by cpu_scalar, cpu_neon, cpu_x86). Each backend
 * keeps its own kernels; what lives here is the part in front of them:
 * resolving the tensors of an op to host pointers, checking that their
 * shapes agree, and reporting a refusal under be->desc->name. The ops whose
 * whole body is shared (kv_append_f16, hadamard_rotate) are vtable entry
 * points the backends install directly.
 */
#pragma once

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "host_ops.h is internal to the backend layer."
#endif

#include "host_buffer.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_types.h>
#include <geist_weight.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---- F32 elementwise ops ------------------------------------------------ */

/* The host views of an F32 elementwise op. */
struct geist_host_ew {
    const float *x, *z, *w;
    float       *y;
    size_t       n;    /* elements of x, y and z */
    size_t       feat; /* last dimension of x, with GEIST_EW_FEAT or GEIST_EW_W */
};

/* What geist_host_ew_views requires beyond x and y. */
enum geist_host_ew_need {
    GEIST_EW_XY   = 0,       /* y = op(x) */
    GEIST_EW_Z    = 1u << 0, /* a second operand z, as many elements as x */
    GEIST_EW_FEAT = 1u << 1, /* rows of feat = x's last dimension */
    GEIST_EW_W    = 1u << 2, /* a weight w of feat elements (implies FEAT) */
};

/* Resolve x and y, and z and w where `need` asks for them (nullptr
 * otherwise), as F32 DENSE host views (host_buffer.h). x, y and z must hold
 * the same number of elements; feat must divide it.
 *
 * Returns GEIST_E_INVALID_ARG, with the backend's error set to
 * "<backend> <op>: bad inputs", for a missing or malformed view or a shape
 * that does not agree; *v is then unspecified. */
[[nodiscard]] enum geist_status geist_host_ew_views(struct geist_backend      *be,
                                                    const char                *op,
                                                    unsigned                   need,
                                                    const struct geist_tensor *x,
                                                    const struct geist_tensor *z,
                                                    const struct geist_tensor *w,
                                                    struct geist_tensor       *y,
                                                    struct geist_host_ew      *v);

/* ---- Attention ------------------------------------------------------------ */

/* The host views of prims->attention: q and out F32 [n_q, n_q_heads,
 * head_dim], k and v [n_kv, n_kv_heads, head_dim], F32 or (f16) IEEE half. */
struct geist_host_attn {
    const float *q;
    const void  *k, *v; /* const float * or, with f16, const uint16_t * */
    float       *out;
    size_t       n_q, n_q_heads, head_dim, n_kv, n_kv_heads;
    bool         f16;
};

/* Resolve the views of prims->attention: k's dtype picks F32 or F16 for k
 * and v, and the shapes must agree, with the query heads a multiple of the
 * KV heads (the kernels map query head h to KV head h / (n_q_heads /
 * n_kv_heads)). Returns GEIST_E_INVALID_ARG, the backend's error set,
 * otherwise. */
[[nodiscard]] enum geist_status geist_host_attn_views(struct geist_backend      *be,
                                                      const struct geist_tensor *q,
                                                      const struct geist_tensor *k,
                                                      const struct geist_tensor *v,
                                                      struct geist_tensor       *out,
                                                      struct geist_host_attn    *a);

/* The host views of fused->attention_kv_int8 / _int4 (geist_attention_kv_args):
 * q and out F32 [n_q, n_q_heads, head_dim], k I8 [n_kv, n_kv_heads,
 * head_dim], v I8 alike or, int4, U8 with head_dim / 2 bytes per row, the
 * scales F32 [n_kv, n_kv_heads]. */
struct geist_host_attn_kvq {
    const float  *q, *k_scale, *v_scale;
    const int8_t *k;
    const void   *v;
    float        *out;
    size_t        n_q, n_q_heads, head_dim, n_kv, n_kv_heads;
};

/* Resolve and check the views of args (q_offset included: every query's
 * position lies in the cache). `supported` is the backend's probe answer
 * for `fused_op`, asked with the views' head_dim and head counts. Returns
 * GEIST_E_INVALID_ARG for a view or shape that does not hold and
 * GEIST_E_UNSUPPORTED for a geometry the probe refuses, the backend's error
 * set to "<backend> <op>: ..." either way. */
[[nodiscard]] enum geist_status
geist_host_attn_kvq_views(struct geist_backend                 *be,
                          const char                           *op,
                          const struct geist_attention_kv_args *args,
                          bool                                  int4_v,
                          enum geist_fused_op                   fused_op,
                          bool (*supported)(const struct geist_fusion_query *q),
                          struct geist_host_attn_kvq *a);

/* ---- Shared entry points -------------------------------------------------- */

/* fused->kv_append_f16: the k_src / v_src rows (F32) to IEEE half at row
 * q_position of the F16 caches, whose views end with those rows. */
[[nodiscard]] enum geist_status geist_host_kv_append_f16(struct geist_backend      *be,
                                                         const struct geist_tensor *k_src,
                                                         const struct geist_tensor *v_src,
                                                         size_t                     q_position,
                                                         struct geist_tensor       *k_cache,
                                                         struct geist_tensor       *v_cache);

/* fused->hadamard_rotate on host views: geist_hadamard_apply (hadamard.h). */
[[nodiscard]] enum geist_status geist_host_hadamard_rotate(struct geist_backend             *be,
                                                           const struct geist_hadamard_args *args);

/* ---- Weights -------------------------------------------------------------- */

/* Hand a backend repack of `bytes` bytes (heap.h, freed with the model) to
 * w: aux_fp32 / aux_n, GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK
 * plus `flags`, and, when layout is non-zero, backend_layout and
 * backend_alignment. bytes must fit aux_n (the callers bound it). */
static inline void geist_weight_attach_aux(struct geist_weight *w,
                                           const void          *blob,
                                           size_t               bytes,
                                           uint16_t             layout,
                                           uint16_t             alignment,
                                           uint16_t             flags) {
    w->aux_fp32 = (const float *) blob;
    w->aux_n    = (int32_t) bytes;
    w->flags = (uint16_t) (w->flags | GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK | flags);
    if (layout != 0) {
        w->backend_layout    = layout;
        w->backend_alignment = alignment;
    }
}
