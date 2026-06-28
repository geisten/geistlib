/*
 * src/backends/cpu_x86/linear_q6k.h — cpu_x86 Q6_K M=1 (decode) path.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Q6_K weights — typically the tied Gemma 4 lm_head (256K rows × 1536) —
 * are bandwidth-bound at decode. The cpu_scalar path dequants one row at
 * a time inside the inner loop, which is ~5× off OpenBLAS-tuned fp32
 * sgemv. We trade ~1.5 GB of memory (Q6_K → fp32) at model load for that
 * 5–8× decode-time win. The blob is owned by the weight (heap.h) and
 * freed at model destroy.
 *
 * Phase 2 will replace this with a BF16 SoA + native VDPBF16PS-SGEMV,
 * halving the memory footprint at no perf cost.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q6K_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q6K_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_q6k.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

/* Predecode Q6_K → fp32 row-major, install the M=1 sgemv kernel pointer.
 * Returns GEIST_OK on success, GEIST_E_OOM on allocation failure,
 * GEIST_E_INVALID_ARG on malformed shape. */
[[nodiscard]] enum geist_status cpu_x86_linear_q6k_resolve(struct geist_weight *w);

/* M=1 kernel installed by the resolver. cblas_sgemv via geist_sgemv. */
void cpu_x86_linear_q6k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y);

/* M>1 (prefill) kernel installed by the resolver. Tiled W8A8 GEMM. */
void cpu_x86_linear_q6k_mN(
        const float *x, const struct geist_weight *w, size_t m, struct geist_backend *be, float *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q6K_H */
