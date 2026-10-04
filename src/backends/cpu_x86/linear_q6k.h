/*
 * src/backends/cpu_x86/linear_q6k.h — cpu_x86 Q6_K and Q3_K linear (decode and
 * prefill).
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Q6_K weights — typically ffn_down and a tied output matrix — are
 * bandwidth-bound at decode, which reads the native Q6_K bytes (w->raw,
 * ~0.82 B/wt). Prefill reads a W8A8 predecode (1.5 B/wt) built at resolve
 * time; linear_q6k.c describes its two layouts. The blob is owned by the
 * weight (heap.h) and freed at model destroy.
 *
 * Q3_K runs the same kernels on its GGUF bytes, decode and prefill.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q6K_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q6K_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_q6k.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

/* Build the W8A8 prefill blob, install the M=1 and M>1 kernel pointers.
 * Returns GEIST_OK on success, GEIST_E_OOM on allocation failure,
 * GEIST_E_INVALID_ARG on malformed shape. */
[[nodiscard]] enum geist_status cpu_x86_linear_q6k_resolve(struct geist_weight *w);

/* M=1 kernel installed by the resolver. q6k_gemv_m1 on the native Q6_K
 * bytes. */
void cpu_x86_linear_q6k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y);

/* M>1 (prefill) kernel installed by the resolver. Tiled W8A8 GEMM. */
void cpu_x86_linear_q6k_mN(
        size_t m, const float *x, const struct geist_weight *w, struct geist_backend *be, float *y);

/* Bind the Q3_K kernels (q3k_gemv_m1 / q3k_gemm on w->raw; nothing is
 * allocated). False, and w untouched, when the shape is not a whole number of
 * 256-element blocks. */
bool cpu_x86_linear_q3k_bind(struct geist_weight *w);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q6K_H */
