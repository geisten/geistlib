/*
 * geist_weight.h — Pre-resolved linear-weight descriptor.
 *
 * @stability EXPERIMENTAL — added 2026-05-15 in refactor v2 (P1.1).
 *
 * Replaces the per-op vtable dispatch for the linear() weight path. A
 * struct geist_weight is constructed at model load time: the backend's
 * resolve_weight() function inspects (dtype, n_in, n_out) and writes
 * direct function pointers for the M=1 (decode) and M>1 (prefill)
 * kernels. Hot-path callers then invoke
 *
 *     w->linear_mN(x, w, m, be, y);
 *
 * without any vtable indirection or per-call dtype switch.
 *
 * Rationale: at load time we know everything needed to pick a kernel —
 * weight dtype, shape, the active backend. Re-deciding on every call
 * (current cpu_neon_linear switch over Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/IQ2_S/
 * IQ3_S × M=1/M>1) is wasted work in the hot path. This struct moves
 * that decision out.
 *
 * Memory ownership:
 *   - `raw` aliases the gguf_reader's mmap (P0.3) or a backend-owned
 *     arena (P1.1.f). geist_weight does not own it.
 *   - `aux_fp32` is owned by the model (heap_alloc_aligned, freed in
 *     model destroy). Used for backend-prefolded auxiliary data
 *     (e.g. AWQ inverse scales, dequantized scale tables).
 *   - The struct itself lives inside the model's layer-weight arrays;
 *     no separate allocation.
 *
 * Hot-path contract:
 *   - linear_m1 / linear_mN must be allocation-free.
 *   - linear_pair_mN, when installed, must be allocation-free and may
 *     optimize two same-input projections together.
 *   - Caller-provided x and y must already be host-resident; both
 *     are raw float pointers (no buffer_map indirection for the
 *     activation tensors that flow through linear in the new path).
 *   - Caller passes m = number of input rows for the multi-row path;
 *     m == 1 uses linear_m1.
 *
 * Backend contract for resolve_weight:
 *   - On success, both linear_m1 and linear_mN are set to non-null
 *     concrete functions. The backend may set both to the same
 *     function if it has a single shape-agnostic kernel.
 *   - On unsupported (dtype, n_in, n_out), return GEIST_E_UNSUPPORTED
 *     and leave the function pointers unset.
 *   - resolve_weight runs once per weight tensor at model load; it
 *     may itself allocate aux_fp32 via heap.h.
 */
#ifndef GEIST_WEIGHT_H
#define GEIST_WEIGHT_H

#include <geist.h>
#include <geist_types.h>  /* enum geist_dtype */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct geist_weight;

/* Linear y = x @ W^T for M=1 (decode-style). Sizes match the weight's
 * shape: x has n_in floats, y has n_out floats. `be` is the backend
 * whose resolver installed this kernel; kernels reach backend-private
 * scratch storage (workspace, caches, hw probe) through be->state. The
 * engine guarantees `be` is non-null and `be->state` is valid for the
 * lifetime of the call. */
typedef void (*geist_kernel_linear_m1_fn)(const float               *x,
                                          const struct geist_weight *w,
                                          struct geist_backend      *be,
                                          float                     *y);

/* Linear Y = X @ W^T for M>1 (prefill-style). X is [m, n_in],
 * Y is [m, n_out], both row-major dense FP32. See linear_m1_fn for
 * the `be` contract. */
typedef void (*geist_kernel_linear_mN_fn)(const float               *x,
                                          const struct geist_weight *w,
                                          size_t                     m,
                                          struct geist_backend      *be,
                                          float                     *y);

/* Optional fused/pair path for two M=1 projections sharing the same X.
 * Backends use this to avoid duplicate scheduling overhead in decode
 * for cases such as FFN gate/up. */
typedef void (*geist_kernel_linear_pair_m1_fn)(const float               *x,
                                               const struct geist_weight *w0,
                                               const struct geist_weight *w1,
                                               struct geist_backend      *be,
                                               float                     *y0,
                                               float                     *y1);

/* Optional fused/pair path for two M>1 projections sharing the same X.
 * Backends use this for cases such as FFN gate/up where activation
 * quantization can be shared across two weights. */
typedef void (*geist_kernel_linear_pair_mN_fn)(const float               *x,
                                               const struct geist_weight *w0,
                                               const struct geist_weight *w1,
                                               size_t                     m,
                                               struct geist_backend      *be,
                                               float                     *y0,
                                               float                     *y1);

enum geist_weight_flags {
    /* aux_fp32 holds AWQ inverse-scales of length n_out. */
    GEIST_W_HAS_AWQ_INV  = 1U << 0,
    /* The weight has been transpiled into a backend-private layout
     * (P1.1.f end-state). raw points to backend arena, not mmap. */
    GEIST_W_BACKEND_OWNS = 1U << 1,
    /* aux_fp32 points to heap-owned backend auxiliary bytes that the
     * model state must free on destroy. aux_n stores byte count when
     * the auxiliary payload is not an FP32 array. */
    GEIST_W_AUX_HEAP_OWNED = 1U << 2,
    /* aux_fp32 points at backend-private packed bytes, not FP32 data.
     * backend_layout describes the exact representation. */
    GEIST_W_AUX_BACKEND_REPACK = 1U << 3,
};

enum geist_weight_backend_layout {
    GEIST_W_LAYOUT_SOURCE = 0,
    GEIST_W_LAYOUT_TQ2_0_TL1,
    GEIST_W_LAYOUT_TQ2_0_Q8A,
    GEIST_W_LAYOUT_DEQUANT_TILE,
    GEIST_W_LAYOUT_Q4_K_PREDECODE,
    GEIST_W_LAYOUT_Q4_K_PREDECODE_NTILE4,
    GEIST_W_LAYOUT_Q6_K_PREDECODE_NTILE4,
    GEIST_W_LAYOUT_Q6_K_PREDECODE_NTILE4_STREAM,
    GEIST_W_LAYOUT_Q6_K_X8_GEMV,
    GEIST_W_LAYOUT_Q4_0_X8_GEMV,
    GEIST_W_LAYOUT_Q6_K_PD8_GEMV,
};

struct geist_weight {
    const void *raw;         /* mmap or backend-arena bytes */
    int32_t     n_in;
    int32_t     n_out;
    uint16_t    dtype;       /* enum geist_dtype */
    uint16_t    flags;       /* enum geist_weight_flags bitmask */
    uint16_t    backend_layout; /* enum geist_weight_backend_layout */
    uint16_t    backend_alignment;

    geist_kernel_linear_m1_fn linear_m1;
    geist_kernel_linear_mN_fn linear_mN;
    geist_kernel_linear_pair_m1_fn linear_pair_m1;
    geist_kernel_linear_pair_mN_fn linear_pair_mN;

    /* Optional pre-folded auxiliary FP32 data; semantics depend on
     * flags. nullptr if none. Length encoded in aux_n. */
    const float *aux_fp32;
    int32_t      aux_n;
};

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GEIST_WEIGHT_H */
