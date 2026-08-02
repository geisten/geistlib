/*
 * src/backends/cpu_neon/internal.h — shared internal types for cpu_neon TUs.
 *
 * Layer: BACKEND. Internal to cpu_neon — other backends MUST NOT include
 * this. struct geist_buffer is intentionally backend-private.
 *
 * Buffer layout currently mirrors cpu_scalar (both wrap a host pointer)
 * but lives in its own translation unit so backend boundaries stay clean.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_INTERNAL_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_INTERNAL_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_neon/internal.h is internal to the backend layer."
#endif

#include <pthread.h>
#include <stdatomic.h>
#include <geist.h>
#include <geist_types.h>
#include "hw_probe.h"
#include "kernel_catalog.h"

struct geist_buffer {
    void                  *host;
    size_t                 bytes;
    enum geist_buffer_role role;
    unsigned int           memory_flags;
};

/* Kernel scratch for temporary int8 / fp32 / int32 buffers. One
 * workspace PER CALLING THREAD, owned by the backend: concurrent
 * sessions (one thread each, see geist_arch.h) quantize activations
 * and stage dequant tiles here, so a single shared instance was a
 * cross-session data race (caught by test_multi_session_parallel_int).
 * Per-thread instances hang off cpu_neon_state.ws_head and are all
 * freed at backend destroy — keeping the lifetime fix of review #4 /
 * V12 (no scratch surviving the backend) while restoring isolation.
 *
 * Within one workspace the users still follow master-prepares-then-
 * OMP-fan-out: the owning thread fills the scratch, its OMP workers
 * read it.
 *
 * Lifetime:
 *   - a thread's first cpu_neon_ws() call on a backend mints its
 *     workspace (lock-free list push, TLS-cached lookup)
 *   - buffers grow on demand from inside the kernels (heap.h)
 *   - freed in one place: cpu_neon_ws_destroy_all() at backend
 *     destroy. No OMP barrier needed.
 */
struct cpu_neon_workspace {
    /* tq2_0 q8a M=1 path: per-call int8 activation + per-block sums. */
    int8_t  *m1_xq;
    size_t   m1_xq_cap;
    int32_t *m1_bsum;
    size_t   m1_bsum_cap;
    /* tq2_0 q8a M>1 path: m × n_in int8 activations, m scales,
     * m × blocks_per_row per-block sums. */
    int8_t  *mN_xq;
    size_t   mN_xq_cap;
    float   *mN_sc;
    size_t   mN_sc_cap;
    int32_t *mN_bsum;
    size_t   mN_bsum_cap;
    /* q4_K/q5_K W*A8 M>1 path: shared activation quant workspace. */
    int8_t  *qk_mN_xq;
    size_t   qk_mN_xq_cap;
    float   *qk_mN_sc;
    size_t   qk_mN_sc_cap;
    int32_t *qk_mN_sum32;
    size_t   qk_mN_sum32_cap;
    /* Q4_K/Q6_K dequant→cblas_sgemm prefill path (m ≥ threshold):
     * 32-row tile of dequant'd weights in fp32. Cap is in floats
     * (DEQ_TILE_ROWS × n_in_max). 64-byte aligned for AMX SGEMM. */
    float *dequant_w_fp32;
    size_t dequant_w_fp32_cap;
    /* Experimental fused FFN tile path: gate/up/mid float tiles plus
     * quantized mid activations for Q6_K down accumulation. */
    float  *ffn_gate;
    size_t  ffn_gate_cap;
    float  *ffn_up;
    size_t  ffn_up_cap;
    float  *ffn_mid;
    size_t  ffn_mid_cap;
    int8_t *ffn_mid_q8;
    size_t  ffn_mid_q8_cap;
    float  *ffn_mid_sc;
    size_t  ffn_mid_sc_cap;
    /* F32 elementwise workspace (currently vForce tanh input). */
    float *elt_f32;
    size_t elt_f32_cap;
};

void cpu_neon_workspace_destroy(struct cpu_neon_workspace *ws);

/* One node per (backend, thread) pair; ws_head is a lock-free push-only
 * list. Nodes of exited threads stay until backend destroy — bounded by
 * the number of threads that ever touched the backend. */
struct cpu_neon_ws_node {
    pthread_t                 tid;
    struct cpu_neon_workspace ws;
    struct cpu_neon_ws_node *_Atomic next;
};

struct cpu_neon_state {
    struct geist_hw_probe         hw;
    struct cpu_neon_kernel_policy policy;
    uint64_t                      ws_generation; /* unique per state instance;
                                                  * guards the TLS cache against
                                                  * a recycled state address */
    struct cpu_neon_ws_node *_Atomic ws_head;
};

/* The calling thread's workspace on this backend (minted on first use).
 * Returns nullptr only on OOM of the ~200-byte node. */
struct cpu_neon_workspace *cpu_neon_ws(struct cpu_neon_state *st);
void                       cpu_neon_ws_destroy_all(struct cpu_neon_state *st);
uint64_t                   cpu_neon_ws_next_generation(void);

/* TQ2_0 (ternary BitNet b1.58) compute kernels — implemented in
 * kernels/tq2_0.c, referenced by the resolver table in weight_resolve.c.
 * q8a_* are the int8-SDOT paths (dotprod hosts); the bare m1 is the fp32
 * fallback for non-dotprod hosts. */
void cpu_neon_w_tq2_0_q8a_m1(const float               *x,
                             const struct geist_weight *w,
                             struct geist_backend      *be,
                             float                     *y);
void cpu_neon_w_tq2_0_q8a_mN(
        const float *x, const struct geist_weight *w, size_t m, struct geist_backend *be, float *y);
void cpu_neon_w_tq2_0_m1(const float               *x,
                         const struct geist_weight *w,
                         struct geist_backend      *be,
                         float                     *y);

/* I2_S (BitNet b1.58 official): ternary W1.58 × A8, int8-SDOT. Same compute
 * as tq2_0/q8a but the in-byte 2-bit field order is reversed and a single
 * per-tensor scale (at raw + n_in*n_out/4) is applied per row. Dotprod only. */
void cpu_neon_w_i2_s_q8a_m1(const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y);
void cpu_neon_w_i2_s_q8a_mN(
        const float *x, const struct geist_weight *w, size_t m, struct geist_backend *be, float *y);

/* Fused F16 × A32 GEMV (M=1) — in-register vcvt_f32_f16, no f32 materialization.
 * Used for the BitNet-2B-4T tied f16 lm_head (the decode bottleneck). */
void cpu_neon_w_f16_m1(const float               *x,
                       const struct geist_weight *w,
                       struct geist_backend      *be,
                       float                     *y);

/* P1.1.b → P2.e: load-time weight resolver. Inspects w->dtype
 * and writes direct M=1 / M>1 kernel function pointers. Returns
 * GEIST_E_UNSUPPORTED for dtypes the backend doesn't implement —
 * after P2.e there's no legacy linear() fallback, so an unsupported
 * dtype fails fast at first dispatch. */
struct geist_weight;
[[nodiscard]] enum geist_status cpu_neon_resolve_weight(struct geist_backend *be,
                                                        struct geist_weight  *w);

/* Capability answer for GEIST_OP_LINEAR with a weight of `w_dtype`,
 * derived from the same kernel table and the same runtime ISA gating
 * cpu_neon_resolve_weight dispatches on. supports_op must not carry its
 * own dtype list — the two drifted apart once already (Q5_K / IQ2_S /
 * IQ3_S reported EMULATED and TQ2_0 / I2_S reported NONE while all five
 * had native kernels). */
enum geist_support cpu_neon_linear_support(const struct geist_backend *be,
                                           enum geist_dtype            w_dtype);

/* Element-wise + rmsnorm — wraps gemma4_kernels.c FP32 ops. */
[[nodiscard]] enum geist_status cpu_neon_add(struct geist_backend      *be,
                                             const struct geist_tensor *a,
                                             const struct geist_tensor *b,
                                             struct geist_tensor       *y);
[[nodiscard]] enum geist_status cpu_neon_mul(struct geist_backend      *be,
                                             const struct geist_tensor *a,
                                             const struct geist_tensor *b,
                                             struct geist_tensor       *y);
[[nodiscard]] enum geist_status
cpu_neon_gelu_tanh(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y);
[[nodiscard]] enum geist_status cpu_neon_gelu_tanh_mul(struct geist_backend      *be,
                                                       const struct geist_tensor *x,
                                                       const struct geist_tensor *z,
                                                       struct geist_tensor       *y);
[[nodiscard]] enum geist_status cpu_neon_gelu_tanh_mul_scaled(struct geist_backend      *be,
                                                              const struct geist_tensor *x,
                                                              const struct geist_tensor *z,
                                                              const float               *scale,
                                                              struct geist_tensor       *y);
[[nodiscard]] enum geist_status cpu_neon_relu_squared(struct geist_backend      *be,
                                                      const struct geist_tensor *x,
                                                      struct geist_tensor       *y);
[[nodiscard]] enum geist_status
cpu_neon_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y);
[[nodiscard]] enum geist_status cpu_neon_rmsnorm(struct geist_backend      *be,
                                                 const struct geist_tensor *x,
                                                 const struct geist_tensor *w,
                                                 float                      eps,
                                                 struct geist_tensor       *y);

/* Transformer ops — wraps gemma4_kernels.c reference kernels. */
[[nodiscard]] enum geist_status cpu_neon_rope_apply(struct geist_backend      *be,
                                                    struct geist_tensor       *x,
                                                    const struct geist_tensor *cos,
                                                    const struct geist_tensor *sin);
[[nodiscard]] enum geist_status cpu_neon_embedding_lookup(struct geist_backend      *be,
                                                          const struct geist_tensor *embed_table,
                                                          geist_token_t              token_id,
                                                          struct geist_tensor       *out);
[[nodiscard]] enum geist_status cpu_neon_attention(struct geist_backend      *be,
                                                   const struct geist_tensor *q,
                                                   const struct geist_tensor *k,
                                                   const struct geist_tensor *v,
                                                   size_t                     q_offset,
                                                   size_t                     sliding_window,
                                                   struct geist_tensor       *out);
[[nodiscard]] enum geist_status cpu_neon_ffn_geglu_q4q6_mN(struct geist_backend      *be,
                                                           const float               *x,
                                                           size_t                     m,
                                                           size_t                     d_model,
                                                           size_t                     inter,
                                                           const struct geist_weight *gate,
                                                           const struct geist_weight *up,
                                                           const struct geist_weight *down,
                                                           const float               *down_scale,
                                                           float                     *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_INTERNAL_H */
