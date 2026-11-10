/*
 * src/backends/cpu_neon/kernels/kq_epilogue.h — the float step of the Q4_K
 * and Q6_K W*A8 kernels, one spelling for every path (#527).
 *
 * Layer: BACKEND (cpu_neon, internal).
 *
 * Per super-block a kernel's integer sums are exact; the float step that
 * scales them and adds them into the row's accumulator decides the row's
 * bits. Written as `acc += sx * (d * s - dmin * m)`, the build's
 * -ffast-math lets the compiler associate and contract it differently in a
 * vectorised 4-row tile than in a scalar 1-row tail, so a token's result
 * depended on whether it was one of the m % 4 rows of the call, that is on
 * how many rows an MoE expert got from the prefill chunk. These helpers
 * spell the order with fused operations, which the compiler does not
 * reassociate, so every path computes:
 *
 *   Q4_K  t = fma(d, s, -(dmin * m))      acc = fma(sx, t, acc)
 *   Q6_K  acc = fma(d * sx, s, acc)
 *
 * with d, dmin the weight's super-block scales, sx the activation's and s,
 * m the exact integer sums. The vector forms do the same per lane.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_KQ_EPILOGUE_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_KQ_EPILOGUE_H

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>

/* Q4_K: the super-block's d * s - dmin * m. */
static inline float q4k_epi_t(float d, float dmin, int32_t s, int32_t m) {
    return fmaf(d, (float) s, -(dmin * (float) m));
}

/* Q4_K: acc plus the super-block's sx * (d * s - dmin * m). */
static inline float q4k_epi(float acc, float sx, float d, float dmin, int32_t s, int32_t m) {
    return fmaf(sx, q4k_epi_t(d, dmin, s, m), acc);
}

/* q4k_epi_t per lane. */
static inline float32x4_t q4k_epi_t4(float32x4_t d, float32x4_t dmin, int32x4_t s, int32x4_t m) {
    return vfmaq_f32(vnegq_f32(vmulq_f32(dmin, vcvtq_f32_s32(m))), d, vcvtq_f32_s32(s));
}

/* Q6_K: acc plus the super-block's d * sx * s. */
static inline float q6k_epi(float acc, float d, float sx, int32_t s) {
    return fmaf(d * sx, (float) s, acc);
}

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_KQ_EPILOGUE_H */
