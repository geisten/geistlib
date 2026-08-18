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
 * Catalog entries: NEON (dotprod-gated w8a8), AVX-512 VNNI w8a8 (#237,
 * function-level target attribute over the x86-64-v3 baseline), portable
 * scalar for everything else.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER /* hw_probe lives in the engine base */

#include "audio_linear.h"

#include "heap.h"
#include "hw_probe.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
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

/* ----------------------------- AVX-512 VNNI ----------------------------- */

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

/* Function-level target attribute: the arch layer is built at the
 * x86-64-v3 baseline; only this kernel carries AVX-512, and the binding
 * only installs it when the runtime probe confirms VNNI — no SIGILL on
 * lesser hosts. */
#define AUDIO_VNNI_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))

AUDIO_VNNI_TARGET
static inline int32_t hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s         = _mm_hadd_epi32(s, s);
    s         = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/* VPDPBUSD multiplies u8 × s8. Quantize activations UNSIGNED (q + 128),
 * dot against the signed weights, and correct with the weight-row sum:
 *   Σ (q+128)·w = Σ q·w + 128·Σw   ⇒   isum = dot − (Σw << 7).
 * Σw costs one dpbusd against ones per 32 weights, computed while the row
 * is hot in cache — no second pass over the weight matrix. */
AUDIO_VNNI_TARGET
static void w8a8_avx512vnni(const int8_t *w_q8,
                            const float  *w_scales,
                            const float  *x,
                            float         scale_x_inv,
                            float         scale_x,
                            size_t        m,
                            size_t        in_dim,
                            size_t        out_dim,
                            float        *y) {
    uint8_t *x_u8 = heap_alloc_array_aligned(uint8_t, m *in_dim);
    for (size_t i = 0; i < m * in_dim; i++) {
        /* Same rounding as the scalar/NEON kernels (ties away from zero),
         * then shift into u8 range. */
        float q = roundf(x[i] * scale_x_inv);
        q       = q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q);
        x_u8[i] = (uint8_t) ((int32_t) q + 128);
    }

    const __m256i ones = _mm256_set1_epi8(1);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < out_dim; n++) {
        const int8_t *wrow   = w_q8 + n * in_dim;
        const float   wscale = w_scales[n];

        __m256i acc_w = _mm256_setzero_si256();
        size_t  k     = 0;
        for (; k + 32 <= in_dim; k += 32) {
            acc_w = _mm256_dpbusd_epi32(
                    acc_w, ones, _mm256_loadu_si256((const __m256i *) (wrow + k)));
        }
        int32_t row_sum = hsum_epi32(acc_w);
        for (; k < in_dim; k++)
            row_sum += (int32_t) wrow[k];

        for (size_t i = 0; i < m; i++) {
            const uint8_t *xrow = x_u8 + i * in_dim;
            __m256i        acc  = _mm256_setzero_si256();
            size_t         kk   = 0;
            for (; kk + 32 <= in_dim; kk += 32) {
                acc = _mm256_dpbusd_epi32(acc,
                                          _mm256_loadu_si256((const __m256i *) (xrow + kk)),
                                          _mm256_loadu_si256((const __m256i *) (wrow + kk)));
            }
            int32_t isum = hsum_epi32(acc);
            for (; kk < in_dim; kk++)
                isum += ((int32_t) xrow[kk]) * (int32_t) wrow[kk];
            isum -= row_sum * 128; /* undo the +128 shift (row_sum may be
                                    * negative — a shift would be UB) */
            y[i * out_dim + n] = wscale * scale_x * (float) isum;
        }
    }
    safe_free((void **) &x_u8);
}
#endif /* __x86_64__ */

/* ------------------------------- binding ------------------------------- */

/* Immutable kernel tables — resolve() returns a pointer into these, so a
 * concurrent first-call can never observe a half-written struct (#251
 * deleted the mutable g_ops/g_bound pair and the rebind test hook; the
 * parity test calls resolve() directly instead). */
static const struct audio_linear_ops OPS_SCALAR = {
        .w8a8  = w8a8_scalar,
        .w8a32 = w8a32_scalar,
        .name  = "scalar",
};
#if defined(__ARM_NEON)
static const struct audio_linear_ops OPS_NEON = {
#if defined(__ARM_FEATURE_DOTPROD)
        .w8a8 = w8a8_neon,
#else
        .w8a8 = w8a8_scalar,
#endif
        .w8a32 = w8a32_neon,
        .name  = "neon",
};
#endif
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
static const struct audio_linear_ops OPS_VNNI = {
        .w8a8  = w8a8_avx512vnni,
        .w8a32 = w8a32_scalar, /* fp32 loop auto-vectorizes at x86-64-v3 */
        .name  = "avx512vnni",
};
#endif

const struct audio_linear_ops *audio_linear_resolve(bool force_scalar) {
    if (force_scalar) {
        return &OPS_SCALAR;
    }
#if defined(__ARM_NEON)
    struct geist_hw_probe hp;
    geist_hw_probe_fill(&hp);
    if (hp.has_neon) {
#if defined(__ARM_FEATURE_DOTPROD)
        if (!hp.has_dotprod) {
            /* Compiled with dotprod but running on a core without it:
             * OPS_NEON's w8a8 would SIGILL — scalar keeps it safe. */
            return &OPS_SCALAR;
        }
#endif
        return &OPS_NEON;
    }
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    struct geist_hw_probe hp;
    geist_hw_probe_fill(&hp);
    /* Down-only clamp with the same semantics as the cpu_x86 dispatcher
     * (kernel_w4a8.c parse_force_env): tiers below VNNI clamp it away,
     * avx512_vnni / avx512_bf16 keep it, unknown values mean no
     * override. */
    const char *fisa    = getenv("GEIST_FORCE_ISA");
    bool        vnni_ok = true;
    if (fisa != nullptr &&
        (strcmp(fisa, "scalar") == 0 || strcmp(fisa, "avx2") == 0 || strcmp(fisa, "avx512") == 0)) {
        vnni_ok = false;
    }
    if (hp.has_avx512_vnni && vnni_ok) {
        return &OPS_VNNI;
    }
#endif
    return &OPS_SCALAR;
}

const struct audio_linear_ops *audio_linear_bind(void) {
    /* Cache the probe result; the tables are immutable, so a plain
     * atomic pointer with relaxed ordering is fully safe — worst case
     * two first-callers both resolve to the same table. */
    static _Atomic(const struct audio_linear_ops *) cached = nullptr;
    const struct audio_linear_ops *p = atomic_load_explicit(&cached, memory_order_relaxed);
    if (p == nullptr) {
        const char *force = getenv("GEIST_AUDIO_KERNEL");
        p                 = audio_linear_resolve(force != nullptr && strcmp(force, "scalar") == 0);
        atomic_store_explicit(&cached, p, memory_order_relaxed);
    }
    return p;
}
