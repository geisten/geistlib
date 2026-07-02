/*
 * src/backends/cpu_scalar/internal.h — shared internal types for the
 * cpu_scalar backend translation units.
 *
 * Layer: BACKEND. Internal to cpu_scalar — other backends MUST NOT include
 * this. struct geist_buffer is intentionally backend-private; engine code
 * sees it only as an opaque forward declaration via <geist.h>.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_SCALAR_INTERNAL_H
#define GEIST_INTERNAL_BACKEND_CPU_SCALAR_INTERNAL_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_scalar/internal.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_types.h>

/* CPU buffer = host pointer + size + role. */
struct geist_buffer {
    void                  *host;
    size_t                 bytes;
    enum geist_buffer_role role;
    unsigned int           memory_flags;
};

/* P1.1.b → P2.e: load-time weight resolver. Inspects w->dtype and writes
 * direct M=1 / M>1 kernel function pointers. After the linear-slot drop
 * (P2-final), this is the only path the forward layer dispatches through. */
struct geist_weight;
[[nodiscard]] enum geist_status cpu_scalar_resolve_weight(struct geist_backend *be,
                                                          struct geist_weight  *w);

/* Element-wise + rmsnorm ops — F32 DENSE on all inputs/outputs. */
[[nodiscard]] enum geist_status cpu_scalar_add(struct geist_backend      *be,
                                               const struct geist_tensor *a,
                                               const struct geist_tensor *b,
                                               struct geist_tensor       *y);
[[nodiscard]] enum geist_status cpu_scalar_mul(struct geist_backend      *be,
                                               const struct geist_tensor *a,
                                               const struct geist_tensor *b,
                                               struct geist_tensor       *y);
[[nodiscard]] enum geist_status cpu_scalar_gelu_tanh(struct geist_backend      *be,
                                                     const struct geist_tensor *x,
                                                     struct geist_tensor       *y);
[[nodiscard]] enum geist_status cpu_scalar_gelu_tanh_mul(struct geist_backend      *be,
                                                         const struct geist_tensor *x,
                                                         const struct geist_tensor *z,
                                                         struct geist_tensor       *y);
[[nodiscard]] enum geist_status cpu_scalar_gelu_tanh_mul_scaled(struct geist_backend      *be,
                                                                const struct geist_tensor *x,
                                                                const struct geist_tensor *z,
                                                                const float               *scale,
                                                                struct geist_tensor       *y);
[[nodiscard]] enum geist_status cpu_scalar_relu_squared(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        struct geist_tensor       *y);
[[nodiscard]] enum geist_status
cpu_scalar_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y);
[[nodiscard]] enum geist_status cpu_scalar_rmsnorm(struct geist_backend      *be,
                                                   const struct geist_tensor *x,
                                                   const struct geist_tensor *w,
                                                   float                      eps,
                                                   struct geist_tensor       *y);

/* Transformer-specific ops — wrap gemma4_kernels.c reference kernels. */
[[nodiscard]] enum geist_status cpu_scalar_rope_apply(struct geist_backend      *be,
                                                      struct geist_tensor       *x,
                                                      const struct geist_tensor *cos,
                                                      const struct geist_tensor *sin);
[[nodiscard]] enum geist_status cpu_scalar_embedding_lookup(struct geist_backend      *be,
                                                            const struct geist_tensor *embed_table,
                                                            geist_token_t              token_id,
                                                            struct geist_tensor       *out);
[[nodiscard]] enum geist_status cpu_scalar_attention(struct geist_backend      *be,
                                                     const struct geist_tensor *q,
                                                     const struct geist_tensor *k,
                                                     const struct geist_tensor *v,
                                                     size_t                     q_offset,
                                                     size_t                     sliding_window,
                                                     struct geist_tensor       *out);

#endif /* GEIST_INTERNAL_BACKEND_CPU_SCALAR_INTERNAL_H */
