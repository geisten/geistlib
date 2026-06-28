/*
 * src/backends/cpu_x86/linear_f32q.h — F32 dense → W8A8 quantized linear.
 *
 * Layer: BACKEND (cpu_x86, internal). See linear_f32q.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_F32Q_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_F32Q_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_f32q.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

#include <stddef.h>
#include <stdint.h>

/* Quantize one fp32 weight row to W8A8 (per-16-block asymmetric int8). */
void f32_to_w8a8_row(
        size_t n_in, const float *row, uint8_t *u_w, float *w_scales, float *w_offsets);

/* Build the W8A8 blob from an F32 weight and install the m1/mN kernels.
 * Returns GEIST_E_INVALID_ARG if n_in % 16 != 0 (caller keeps cblas). */
[[nodiscard]] enum geist_status cpu_x86_linear_f32q_resolve(struct geist_weight *w);

void cpu_x86_linear_f32q_m1(const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y);
void cpu_x86_linear_f32q_mN(
        const float *x, const struct geist_weight *w, size_t m, struct geist_backend *be, float *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_F32Q_H */
