/*
 * src/backends/cpu_x86/elementwise.h — cpu_x86 gelu_tanh, SiLU, RMSNorm and
 * add overrides.
 *
 * Layer: BACKEND (cpu_x86, internal). See elementwise.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_ELEMENTWISE_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_ELEMENTWISE_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/elementwise.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_backend.h>

[[nodiscard]] enum geist_status
cpu_x86_gelu_tanh(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y);

[[nodiscard]] enum geist_status cpu_x86_gelu_tanh_mul(struct geist_backend      *be,
                                                      const struct geist_tensor *x,
                                                      const struct geist_tensor *z,
                                                      struct geist_tensor       *y);

[[nodiscard]] enum geist_status cpu_x86_gelu_tanh_mul_scaled(struct geist_backend      *be,
                                                             const struct geist_tensor *x,
                                                             const struct geist_tensor *z,
                                                             const float               *scale,
                                                             struct geist_tensor       *y);

[[nodiscard]] enum geist_status
cpu_x86_silu(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y);

/* silu(x) * z, bit-identical to cpu_x86_silu then cpu_scalar_mul. */
[[nodiscard]] enum geist_status cpu_x86_silu_mul(struct geist_backend      *be,
                                                 const struct geist_tensor *x,
                                                 const struct geist_tensor *z,
                                                 struct geist_tensor       *y);

/* Rows spread over the team; the sum of squares in double, as cpu_scalar. */
[[nodiscard]] enum geist_status cpu_x86_rmsnorm(struct geist_backend      *be,
                                                const struct geist_tensor *x,
                                                const struct geist_tensor *w,
                                                float                      eps,
                                                struct geist_tensor       *y);

/* a + b, bit-identical to cpu_scalar_add. */
[[nodiscard]] enum geist_status cpu_x86_add(struct geist_backend      *be,
                                            const struct geist_tensor *a,
                                            const struct geist_tensor *b,
                                            struct geist_tensor       *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_ELEMENTWISE_H */
