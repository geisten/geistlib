/*
 * audio_linear.c — quantized matmul kernels for the audio tower + the
 * load-time binding that picks them (#236).
 *
 * Two independent implementations per kernel: NEON (moved verbatim from
 * encoder_forward.c) and portable scalar. Selection is RUNTIME state from
 * geist_hw_probe_fill — a binary built with SIMD flags still binds scalar
 * on a host whose probe lacks the feature, so the audio path can never
 * SIGILL on a lesser core. GEIST_AUDIO_KERNEL=scalar forces the portable
 * kernels (parity testing / triage).
 *
 * x86 note: scalar carries the W8A8 semantics until the cpu_x86 VNNI W8A8
 * GEMV is bound here (#237) — one more catalog entry, no new plumbing.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER /* hw_probe lives in the engine base */

#include "audio_linear.h"

#include "heap.h"
#include "hw_probe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------- scalar -------------------------------- */

static void w8a8_scalar(const int8_t *w_q8,
                        const float  *w_scales,
                        const float  *x,
                        float         scale_x_inv,
                        float         scale_x,
                        size_t        m,
                        size_t        in_dim,
                        size_t        out_dim,
                        float        *y) {
    int8_t *x_q8 = heap_alloc_array_aligned(int8_t, m *in_dim);
    for (size_t i = 0; i < m * in_dim; i++) {
        /* roundf = ties away from zero, matching the NEON path's vcvtaq;
         * saturate like the NEON path's vqmovn chain. */
        float q = roundf(x[i] * scale_x_inv);
        x_q8[i] = (int8_t) (q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q));
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < out_dim; n++) {
        const int8_t *wrow   = w_q8 + n * in_dim;
        const float   wscale = w_scales[n];
        for (size_t i = 0; i < m; i++) {
            const int8_t *xrow = x_q8 + i * in_dim;
            int32_t       isum = 0;
            for (size_t k = 0; k < in_dim; k++)
                isum += (int32_t) wrow[k] * (int32_t) xrow[k];
            y[i * out_dim + n] = wscale * scale_x * (float) isum;
        }
    }
    safe_free((void **) &x_q8);
}

static void w8a32_scalar(const int8_t *w_q8,
                         const float  *w_scales,
                         const float  *x,
                         size_t        m,
                         size_t        in_dim,
                         size_t        out_dim,
                         float        *y) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < out_dim; n++) {
        const float wscale = w_scales[n];
        for (size_t i = 0; i < m; i++) {
            float dot = 0.0f;
            for (size_t k = 0; k < in_dim; k++)
                dot += (float) w_q8[n * in_dim + k] * x[i * in_dim + k];
            y[i * out_dim + n] = wscale * dot;
        }
    }
}

/* -------------------------------- NEON --------------------------------- */

#if defined(__ARM_NEON)
#include <arm_neon.h>

#if defined(__ARM_FEATURE_DOTPROD)
static void w8a8_neon(const int8_t *w_q8,
                      const float  *w_scales,
                      const float  *x,
                      float         scale_x_inv,
                      float         scale_x,
                      size_t        m,
                      size_t        in_dim,
                      size_t        out_dim,
                      float        *y) {
    int8_t           *x_q8  = heap_alloc_array_aligned(int8_t, m *in_dim);
    const float32x4_t inv_v = vdupq_n_f32(scale_x_inv);
    for (size_t i = 0; i < m; i++) {
        const float *xrow = x + i * in_dim;
        int8_t      *qrow = x_q8 + i * in_dim;
        size_t       k    = 0;
        for (; k + 16 <= in_dim; k += 16) {
            int32x4_t q0  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xrow + k + 0), inv_v));
            int32x4_t q1  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xrow + k + 4), inv_v));
            int32x4_t q2  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xrow + k + 8), inv_v));
            int32x4_t q3  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xrow + k + 12), inv_v));
            int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            vst1q_s8(qrow + k, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
        }
        for (; k < in_dim; k++)
            qrow[k] = (int8_t) lrintf(xrow[k] * scale_x_inv);
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < out_dim; n++) {
        const int8_t *wrow   = w_q8 + n * in_dim;
        const float   wscale = w_scales[n];
        for (size_t i = 0; i < m; i++) {
            const int8_t *xrow = x_q8 + i * in_dim;
            int32x4_t     acc  = vdupq_n_s32(0);
            size_t        k    = 0;
            for (; k + 16 <= in_dim; k += 16) {
                acc = vdotq_s32(acc, vld1q_s8(wrow + k), vld1q_s8(xrow + k));
            }
            int32_t isum = vaddvq_s32(acc);
            for (; k < in_dim; k++)
                isum += (int32_t) wrow[k] * (int32_t) xrow[k];
            y[i * out_dim + n] = wscale * scale_x * (float) isum;
        }
    }
    safe_free((void **) &x_q8);
}
#endif /* __ARM_FEATURE_DOTPROD */

static void w8a32_neon(const int8_t *w_q8,
                       const float  *w_scales,
                       const float  *x,
                       size_t        m,
                       size_t        in_dim,
                       size_t        out_dim,
                       float        *y) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < out_dim; n++) {
        const int8_t *wrow   = w_q8 + n * in_dim;
        const float   wscale = w_scales[n];
        for (size_t i = 0; i < m; i++) {
            const float *xrow = x + i * in_dim;
            float32x4_t  acc0 = vdupq_n_f32(0.0f);
            float32x4_t  acc1 = vdupq_n_f32(0.0f);
            float32x4_t  acc2 = vdupq_n_f32(0.0f);
            float32x4_t  acc3 = vdupq_n_f32(0.0f);
            size_t       k    = 0;
            for (; k + 16 <= in_dim; k += 16) {
                int8x16_t qv = vld1q_s8(wrow + k);
                /* int8x16 -> int16x8 x2 -> int32x4 x4 -> float32x4 x4 */
                int16x8_t   qa = vmovl_s8(vget_low_s8(qv));
                int16x8_t   qb = vmovl_s8(vget_high_s8(qv));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qa)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qa)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qb)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qb)));
                acc0           = vfmaq_f32(acc0, f0, vld1q_f32(xrow + k + 0));
                acc1           = vfmaq_f32(acc1, f1, vld1q_f32(xrow + k + 4));
                acc2           = vfmaq_f32(acc2, f2, vld1q_f32(xrow + k + 8));
                acc3           = vfmaq_f32(acc3, f3, vld1q_f32(xrow + k + 12));
            }
            float32x4_t sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float       dot = vaddvq_f32(sum);
            for (; k < in_dim; k++)
                dot += (float) wrow[k] * xrow[k];
            y[i * out_dim + n] = wscale * dot;
        }
    }
}
#endif /* __ARM_NEON */

/* ------------------------------- binding ------------------------------- */

static struct audio_linear_ops g_ops;
static bool                    g_bound = false;

const struct audio_linear_ops *audio_linear_rebind(void) {
    g_bound = false;
    return audio_linear_bind();
}

const struct audio_linear_ops *audio_linear_bind(void) {
    if (g_bound) {
        return &g_ops;
    }
    g_ops = (struct audio_linear_ops) {
            .w8a8  = w8a8_scalar,
            .w8a32 = w8a32_scalar,
            .name  = "scalar",
    };

    const char *force       = getenv("GEIST_AUDIO_KERNEL");
    const bool  want_scalar = force != nullptr && strcmp(force, "scalar") == 0;
    if (!want_scalar) {
#if defined(__ARM_NEON)
        struct geist_hw_probe hp;
        geist_hw_probe_fill(&hp);
        if (hp.has_neon) {
            g_ops.w8a32 = w8a32_neon;
            g_ops.name  = "neon";
#if defined(__ARM_FEATURE_DOTPROD)
            if (hp.has_dotprod) {
                g_ops.w8a8 = w8a8_neon;
            }
#endif
        }
#endif
    }
    g_bound = true;
    return &g_ops;
}
