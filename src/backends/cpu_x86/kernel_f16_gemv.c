/*
 * src/backends/cpu_x86/kernel_f16_gemv.c — F16-weight decode GEMV (F16C+FMA).
 *
 * Layer: BACKEND (cpu_x86). Baseline -march=x86-64-v3 (F16C + FMA in v3),
 * so no per-TU ISA flags and no SIGILL risk on any v3 host.
 *
 * See kernel_f16_gemv.h. geist_par_for over output rows; each row does an
 * 8-wide F16C convert + FMA, two independent accumulators to hide latency.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_f16_gemv.h"
#include "linear_util.h"

#include "quant.h" /* fp16_to_fp32 (scalar tail) */

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "par.h"

/* One GEMV's rows for geist_par_for; `w` is the f16 or int8 weight. */
struct gemv_rows {
    size_t       n_in;
    const float *x;
    const void  *w;
    const float *scales; /* q8w only */
    float       *y;
};

static void f16_rows(void *ctx, size_t r0, size_t r1) {
    const struct gemv_rows c     = *(const struct gemv_rows *) ctx;
    const size_t           n_in  = c.n_in;
    const float           *x     = c.x;
    const uint16_t        *w_f16 = c.w;
    for (size_t r = r0; r < r1; r++) {
        const uint16_t *wr   = w_f16 + r * n_in;
        __m256          acc0 = _mm256_setzero_ps();
        __m256          acc1 = _mm256_setzero_ps();
        size_t          k    = 0;
        for (; k + 16 <= n_in; k += 16) {
            const __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (wr + k)));
            const __m256 w1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (wr + k + 8)));
            acc0            = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k), acc0);
            acc1            = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8), acc1);
        }
        for (; k + 8 <= n_in; k += 8) {
            const __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (wr + k)));
            acc0            = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k), acc0);
        }
        float s = hsum_ps_hadd(_mm256_add_ps(acc0, acc1));
        for (; k < n_in; k++) {
            s += fp16_to_fp32(wr[k]) * x[k];
        }
        c.y[r] = s;
    }
}

void f16_gemv_m1(
        size_t n_out, size_t n_in, const float *x, const uint16_t w_f16[], float y[static n_out]) {
    /* The spec head calls this via a one-row weight view, once per finalist;
     * geist_par_for runs n_out == 1 on the caller with no region, so the
     * SAME compiled row loop serves both the dense head and the spec phase-3
     * (bit-identical logits by construction, #102). */
    struct gemv_rows c = {n_in, x, w_f16, nullptr, y};
    geist_par_for(n_out, f16_rows, &c);
}

/* f16_to_q8w's rows for geist_par_for. */
struct q8w_quant {
    size_t          n_in;
    const uint16_t *w_f16;
    int8_t         *wq;
    float          *scales;
};

static void q8w_quant_rows(void *ctx, size_t r0, size_t r1) {
    const struct q8w_quant c      = *(const struct q8w_quant *) ctx;
    const size_t           n_in   = c.n_in;
    float                 *scales = c.scales;
    for (size_t r = r0; r < r1; r++) {
        const uint16_t *wr   = c.w_f16 + r * n_in;
        float           amax = 0.0f;
        for (size_t k = 0; k < n_in; k++) {
            const float a = fabsf(fp16_to_fp32(wr[k]));
            if (a > amax) {
                amax = a;
            }
        }
        const float scale = amax / 127.0f;
        const float id    = amax > 0.0f ? 127.0f / amax : 0.0f;
        scales[r]         = scale;
        int8_t *qr        = c.wq + r * n_in;
        for (size_t k = 0; k < n_in; k++) {
            int32_t q = (int32_t) lrintf(fp16_to_fp32(wr[k]) * id);
            if (q > 127) {
                q = 127;
            }
            if (q < -127) {
                q = -127;
            }
            qr[k] = (int8_t) q;
        }
    }
}

void f16_to_q8w(size_t n_out, size_t n_in, const uint16_t w_f16[], int8_t wq[], float scales[]) {
    struct q8w_quant c = {n_in, w_f16, wq, scales};
    geist_par_for(n_out, q8w_quant_rows, &c);
}

static void q8w_rows(void *ctx, size_t r0, size_t r1) {
    const struct gemv_rows c    = *(const struct gemv_rows *) ctx;
    const size_t           n_in = c.n_in;
    const float           *x    = c.x;
    const int8_t          *wq   = c.w;
    for (size_t r = r0; r < r1; r++) {
        const int8_t *qr   = wq + r * n_in;
        __m256        acc0 = _mm256_setzero_ps();
        __m256        acc1 = _mm256_setzero_ps();
        size_t        k    = 0;
        for (; k + 16 <= n_in; k += 16) {
            const __m256 w0 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (qr + k))));
            const __m256 w1 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (qr + k + 8))));
            acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k), acc0);
            acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8), acc1);
        }
        for (; k + 8 <= n_in; k += 8) {
            const __m256 w0 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (qr + k))));
            acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k), acc0);
        }
        float s = hsum_ps_hadd(_mm256_add_ps(acc0, acc1));
        for (; k < n_in; k++) {
            s += (float) qr[k] * x[k];
        }
        c.y[r] = s * c.scales[r];
    }
}

void q8w_gemv_m1(size_t       n_out,
                 size_t       n_in,
                 const float *x,
                 const int8_t wq[],
                 const float  scales[],
                 float        y[static n_out]) {
    struct gemv_rows c = {n_in, x, wq, scales, y};
    geist_par_for(n_out, q8w_rows, &c);
}
