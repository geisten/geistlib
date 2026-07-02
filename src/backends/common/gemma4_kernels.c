#include "gemma4_kernels.h"
#include <geist_types.h>
#include "heap.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* Dense fp32 matmul/matvec goes through the geist_gemm facade (Accelerate /
 * OpenBLAS / native fallback, selected at build time). */
#include "geist_gemm.h"

void bf16_array_to_fp32(const uint16_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = bf16_to_fp32(src[i]);
}

float *bf16_alloc_fp32(const uint16_t *src, size_t n) {
    float *dst = heap_alloc_array_aligned(float, n);
    if (!dst)
        return nullptr;
    bf16_array_to_fp32(src, dst, n);
    return dst;
}

void gelu_tanh_fp32(const float *x, size_t n, float *y) {
    /* gelu_tanh: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) */
    const float kAlpha = 0.7978845608028654f; /* sqrt(2/π) */
    const float kBeta  = 0.044715f;
    for (size_t i = 0; i < n; i++) {
        float xi    = x[i];
        float inner = kAlpha * (xi + kBeta * xi * xi * xi);
        y[i]        = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

/* Opt-in Apple-only fast-tanh path: when env GEIST_FAST_TANH=1 is set at
 * process start, gelu_tanh_mul_fp32 uses Accelerate's vForce vvtanhf, which is
 * ~10x faster than scalar tanhf on M1 (vectorized via NEON/AMX) but
 * differs from libm tanhf by 1-2 ULP per call. Across 16 ViT layers
 * the ULP drift compounds to ~2e-2 absolute on magnitudes ~50, well
 * within fp32 noise but outside the strict parity gate.
 *
 * Cost on Mac vision tower: 1234 ms -> 275 ms (-960 ms / -27% on
 * remaining 3.6 s tower, ~13% total tower speedup). */
#if defined(__APPLE__)
static int fast_tanh_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("GEIST_FAST_TANH");
        v             = (s != nullptr && s[0] == '1') ? 1 : 0;
    }
    return v;
}
#endif

void gelu_tanh_mul_fp32(const float *x, const float *z, size_t n, float *y) {
    const float kAlpha = 0.7978845608028654f;
    const float kBeta  = 0.044715f;
#if defined(__APPLE__)
    if (fast_tanh_enabled()) {
        extern void vvtanhf(float *y, const float *x, const int *n_int);
        for (size_t i = 0; i < n; i++) {
            const float xi = x[i];
            y[i]           = kAlpha * (xi + kBeta * xi * xi * xi);
        }
        int n_int = (int) n;
        vvtanhf(y, y, &n_int);
        for (size_t i = 0; i < n; i++) {
            y[i] = 0.5f * x[i] * (1.0f + y[i]) * z[i];
        }
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) {
        const float xi    = x[i];
        const float inner = kAlpha * (xi + kBeta * xi * xi * xi);
        y[i]              = (0.5f * xi * (1.0f + tanhf(inner))) * z[i];
    }
}

void relu_squared_fp32(const float *x, size_t n, float *y) {
    /* BitNet b1.58 2B-4T FFN activation: y = max(x, 0)^2. The compiler
     * auto-vectorizes this loop into NEON vmaxq+vmulq on aarch64; no
     * intrinsics needed here. Fused threshold + square avoids a second
     * pass through y. */
    for (size_t i = 0; i < n; i++) {
        float xi = x[i] > 0.0f ? x[i] : 0.0f;
        y[i]     = xi * xi;
    }
}

void silu_fp32_ooo(const float *x, size_t n, float *y) {
    /* SiLU (Swish): y = x * sigmoid(x) = x / (1 + exp(-x)).
     * Llama 2/3 + BitNet b1.58 3B SwiGLU activation. */
    for (size_t i = 0; i < n; i++) {
        const float xi = x[i];
        y[i]           = xi / (1.0f + expf(-xi));
    }
}

void add_fp32(const float *a, const float *b, size_t n, float *y) {
    for (size_t i = 0; i < n; i++)
        y[i] = a[i] + b[i];
}

void mul_fp32(const float *a, const float *b, size_t n, float *y) {
    for (size_t i = 0; i < n; i++)
        y[i] = a[i] * b[i];
}

void rope_compute(size_t seq_len,
                  size_t head_dim,
                  size_t n_rotated_dims,
                  float  theta,
                  float *cos_out,
                  float *sin_out) {
    /* inv_freq[i] = theta^(-2i/head_dim) for i in [0, n_rotated_dims/2),
     * 0 for i in [n_rotated_dims/2, head_dim/2). Zeros yield cos=1, sin=0
     * which is identity rotation — supports Gemma 4 partial RoPE. */
    size_t half     = head_dim / 2;
    size_t n_active = n_rotated_dims / 2;
    if (n_active > half)
        n_active = half;
    float *inv_freq = heap_alloc_array_aligned(float, half);
    for (size_t i = 0; i < n_active; i++) {
        inv_freq[i] = powf(theta, -((float) (2 * i) / (float) head_dim));
    }
    for (size_t i = n_active; i < half; i++) {
        inv_freq[i] = 0.0f;
    }
    for (size_t s = 0; s < seq_len; s++) {
        float pos = (float) s;
        for (size_t i = 0; i < half; i++) {
            float angle                      = pos * inv_freq[i];
            float c                          = cosf(angle);
            float si                         = sinf(angle);
            cos_out[s * head_dim + i]        = c;
            cos_out[s * head_dim + half + i] = c;
            sin_out[s * head_dim + i]        = si;
            sin_out[s * head_dim + half + i] = si;
        }
    }
    safe_free((void **) &inv_freq);
}

void rope_apply(float       *x,
                const float *cos,
                const float *sin,
                size_t       seq_len,
                size_t       n_heads,
                size_t       head_dim) {
    size_t half = head_dim / 2;
    for (size_t s = 0; s < seq_len; s++) {
        const float *cos_s = cos + s * head_dim;
        const float *sin_s = sin + s * head_dim;
        for (size_t h = 0; h < n_heads; h++) {
            float *xh = x + (s * n_heads + h) * head_dim;
            /* rotate_half: out = x*cos + [-x_high, x_low]*sin
             * For i in [0, half): out[i]      = x[i]      * cos[i]      - x[i+half] * sin[i]
             * For i in [half, dim): out[i+h] = x[i+half] * cos[i+half] + x[i]      * sin[i+half]
             * Note cos[i] == cos[i+half] and sin[i] == sin[i+half] (duplicated). */
            for (size_t i = 0; i < half; i++) {
                float a      = xh[i];
                float b      = xh[i + half];
                xh[i]        = a * cos_s[i] - b * sin_s[i];
                xh[i + half] = b * cos_s[i + half] + a * sin_s[i + half];
            }
        }
    }
}

void rope_compute_at(size_t pos_offset,
                     size_t n_positions,
                     size_t head_dim,
                     size_t n_rotated_dims,
                     float  theta,
                     float *cos_out,
                     float *sin_out) {
    size_t half     = head_dim / 2;
    size_t n_active = n_rotated_dims / 2;
    if (n_active > half)
        n_active = half;
    float *inv_freq = heap_alloc_array_aligned(float, half);
    for (size_t i = 0; i < n_active; i++) {
        inv_freq[i] = powf(theta, -((float) (2 * i) / (float) head_dim));
    }
    for (size_t i = n_active; i < half; i++)
        inv_freq[i] = 0.0f;
    for (size_t s = 0; s < n_positions; s++) {
        float pos = (float) (pos_offset + s);
        for (size_t i = 0; i < half; i++) {
            float angle                      = pos * inv_freq[i];
            float c                          = cosf(angle);
            float si                         = sinf(angle);
            cos_out[s * head_dim + i]        = c;
            cos_out[s * head_dim + half + i] = c;
            sin_out[s * head_dim + i]        = si;
            sin_out[s * head_dim + half + i] = si;
        }
    }
    safe_free((void **) &inv_freq);
}

void attention_mqa_causal_kv(const float *q,
                             const float *k,
                             const float *v,
                             size_t       n_q,
                             size_t       n_kv,
                             size_t       q_offset,
                             size_t       n_q_heads,
                             size_t       n_kv_heads,
                             size_t       head_dim,
                             size_t       sliding_window,
                             float       *out) {
    const float scale = 1.0f; /* Gemma 4: scaling encoded via Q/K-norms. */
    (void) head_dim;
    const size_t kv_group_size = n_q_heads / n_kv_heads;

    /* Per-call score scratch. This common fallback has no backend workspace
     * to borrow (unlike the cpu_neon SDPA path), so it allocates and frees
     * per call rather than holding a process-lifetime _Thread_local buffer. */
    float *scores = heap_alloc_array_aligned(float, n_kv);
    if (scores == nullptr) {
        fprintf(stderr, "attention_mqa_causal_kv: OOM (%zu scores)\n", n_kv);
        return;
    }
    for (size_t t = 0; t < n_q; t++) {
        size_t q_pos = q_offset + t;
        size_t s_lo  = 0;
        if (sliding_window > 0 && q_pos + 1 > sliding_window) {
            s_lo = q_pos + 1 - sliding_window;
        }
        size_t s_hi = q_pos; /* inclusive */
        if (s_hi >= n_kv)
            s_hi = n_kv - 1;

        for (size_t h = 0; h < n_q_heads; h++) {
            size_t       kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            float max_score = -INFINITY;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *kv_kh = k + (s * n_kv_heads + kv_h) * head_dim;
                float        dot   = 0.0f;
                for (size_t i = 0; i < head_dim; i++)
                    dot += qv[i] * kv_kh[i];
                float sc  = dot * scale;
                scores[s] = sc;
                if (sc > max_score)
                    max_score = sc;
            }
            double sum_exp = 0.0;
            for (size_t s = s_lo; s <= s_hi; s++) {
                float e   = expf(scores[s] - max_score);
                scores[s] = e;
                sum_exp += e;
            }
            float inv_sum = (float) (1.0 / sum_exp);

            float *outv = out + (t * n_q_heads + h) * head_dim;
            for (size_t i = 0; i < head_dim; i++)
                outv[i] = 0.0f;
            for (size_t s = s_lo; s <= s_hi; s++) {
                float        w  = scores[s] * inv_sum;
                const float *vv = v + (s * n_kv_heads + kv_h) * head_dim;
                for (size_t i = 0; i < head_dim; i++)
                    outv[i] += w * vv[i];
            }
        }
    }
    safe_free((void **) &scores);
}

void attention_mqa_causal(const float *q,
                          const float *k,
                          const float *v,
                          size_t       seq_len,
                          size_t       n_q_heads,
                          size_t       n_kv_heads,
                          size_t       head_dim,
                          size_t       sliding_window,
                          float       *out) {
    /* Gemma 4 sets self_attn.scaling = 1.0 (not the conventional
     * 1/sqrt(head_dim)). The Q/K RMSNorms normalize magnitudes such
     * that explicit attention scaling becomes redundant. */
    const float scale = 1.0f;
    (void) head_dim;
    const size_t kv_group_size = n_q_heads / n_kv_heads;

    /* Working buffer for one row of attention scores (length seq_len). */
    float *scores = heap_alloc_array_aligned(float, seq_len);

    for (size_t t = 0; t < seq_len; t++) {
        for (size_t h = 0; h < n_q_heads; h++) {
            size_t       kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            /* Determine valid key range: [s_lo, t] inclusive */
            size_t s_lo = 0;
            if (sliding_window > 0 && t + 1 > sliding_window) {
                s_lo = t + 1 - sliding_window;
            }

            /* Scores: q · k_s for each s in [s_lo, t]. */
            float max_score = -INFINITY;
            for (size_t s = s_lo; s <= t; s++) {
                const float *kv_kh = k + (s * n_kv_heads + kv_h) * head_dim;
                float        dot   = 0.0f;
                for (size_t i = 0; i < head_dim; i++)
                    dot += qv[i] * kv_kh[i];
                float sc  = dot * scale;
                scores[s] = sc;
                if (sc > max_score)
                    max_score = sc;
            }

            /* Softmax */
            double sum_exp = 0.0;
            for (size_t s = s_lo; s <= t; s++) {
                float e   = expf(scores[s] - max_score);
                scores[s] = e;
                sum_exp += e;
            }
            float inv_sum = (float) (1.0 / sum_exp);

            /* Output: weighted sum over V. */
            float *outv = out + (t * n_q_heads + h) * head_dim;
            for (size_t i = 0; i < head_dim; i++)
                outv[i] = 0.0f;
            for (size_t s = s_lo; s <= t; s++) {
                float        w  = scores[s] * inv_sum;
                const float *vv = v + (s * n_kv_heads + kv_h) * head_dim;
                for (size_t i = 0; i < head_dim; i++)
                    outv[i] += w * vv[i];
            }
        }
    }
    safe_free((void **) &scores);
}

void linear_fp32(const float *x,
                 const float *weight,
                 const float *bias,
                 size_t       m,
                 size_t       n_in,
                 size_t       n_out,
                 float       *y) {
    if (m == 1) {
        /* gemv path: y = W * x, W is (n_out, n_in) row-major. Lower per-call
         * overhead than sgemm for the M=1 decode case. */
        geist_sgemv(
                GEIST_OP_N, (int) n_out, (int) n_in, 1.0f, weight, (int) n_in, x, 1, 0.0f, y, 1);
    } else {
        /* y = x @ weight^T : C[m,n_out] = A[m,n_in] @ B[n_out,n_in]^T. */
        geist_sgemm(GEIST_OP_N,
                    GEIST_OP_T,
                    (int) m,
                    (int) n_out,
                    (int) n_in,
                    1.0f,
                    x,
                    (int) n_in,
                    weight,
                    (int) n_in,
                    0.0f,
                    y,
                    (int) n_out);
    }
    if (bias) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n_out; j++)
                y[i * n_out + j] += bias[j];
        }
    }
}

void rmsnorm_fp32(
        const float *x, const float *weight, size_t n_rows, size_t hidden, float eps, float *y) {
    for (size_t r = 0; r < n_rows; r++) {
        const float *xr = x + r * hidden;
        float       *yr = y + r * hidden;

        /* Mean-of-squares: NEON path on ARM (16-wide unroll, 4 lanes of
         * fp64 accumulators converted from fp32 squares per iteration
         * to maintain fp64-equivalent precision for the strict-parity
         * codepath HF compares against). Scalar fp64 fallback elsewhere. */
        double sum_sq = 0.0;
#if defined(__ARM_NEON)
        size_t      i  = 0;
        float64x2_t a0 = vdupq_n_f64(0.0);
        float64x2_t a1 = vdupq_n_f64(0.0);
        float64x2_t a2 = vdupq_n_f64(0.0);
        float64x2_t a3 = vdupq_n_f64(0.0);
        for (; i + 8 <= hidden; i += 8) {
            float32x4_t v0  = vld1q_f32(xr + i + 0);
            float32x4_t v1  = vld1q_f32(xr + i + 4);
            float32x4_t sq0 = vmulq_f32(v0, v0);
            float32x4_t sq1 = vmulq_f32(v1, v1);
            a0              = vaddq_f64(a0, vcvt_f64_f32(vget_low_f32(sq0)));
            a1              = vaddq_f64(a1, vcvt_high_f64_f32(sq0));
            a2              = vaddq_f64(a2, vcvt_f64_f32(vget_low_f32(sq1)));
            a3              = vaddq_f64(a3, vcvt_high_f64_f32(sq1));
        }
        float64x2_t total = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
        sum_sq            = vgetq_lane_f64(total, 0) + vgetq_lane_f64(total, 1);
        for (; i < hidden; i++) {
            float v = xr[i];
            sum_sq += (double) v * (double) v;
        }
#else
        for (size_t i = 0; i < hidden; i++) {
            float v = xr[i];
            sum_sq += (double) v * (double) v;
        }
#endif
        float mean_sq = (float) (sum_sq / (double) hidden);
        float rsqrt   = 1.0f / sqrtf(mean_sq + eps);

        /* Scale (+ optional weight) — NEON 16-wide unroll. */
#if defined(__ARM_NEON)
        const float32x4_t vr = vdupq_n_f32(rsqrt);
        size_t            j  = 0;
        if (weight) {
            for (; j + 16 <= hidden; j += 16) {
                float32x4_t x0 = vld1q_f32(xr + j + 0);
                float32x4_t x1 = vld1q_f32(xr + j + 4);
                float32x4_t x2 = vld1q_f32(xr + j + 8);
                float32x4_t x3 = vld1q_f32(xr + j + 12);
                float32x4_t w0 = vld1q_f32(weight + j + 0);
                float32x4_t w1 = vld1q_f32(weight + j + 4);
                float32x4_t w2 = vld1q_f32(weight + j + 8);
                float32x4_t w3 = vld1q_f32(weight + j + 12);
                vst1q_f32(yr + j + 0, vmulq_f32(vmulq_f32(x0, vr), w0));
                vst1q_f32(yr + j + 4, vmulq_f32(vmulq_f32(x1, vr), w1));
                vst1q_f32(yr + j + 8, vmulq_f32(vmulq_f32(x2, vr), w2));
                vst1q_f32(yr + j + 12, vmulq_f32(vmulq_f32(x3, vr), w3));
            }
            for (; j < hidden; j++)
                yr[j] = xr[j] * rsqrt * weight[j];
        } else {
            for (; j + 16 <= hidden; j += 16) {
                float32x4_t x0 = vld1q_f32(xr + j + 0);
                float32x4_t x1 = vld1q_f32(xr + j + 4);
                float32x4_t x2 = vld1q_f32(xr + j + 8);
                float32x4_t x3 = vld1q_f32(xr + j + 12);
                vst1q_f32(yr + j + 0, vmulq_f32(x0, vr));
                vst1q_f32(yr + j + 4, vmulq_f32(x1, vr));
                vst1q_f32(yr + j + 8, vmulq_f32(x2, vr));
                vst1q_f32(yr + j + 12, vmulq_f32(x3, vr));
            }
            for (; j < hidden; j++)
                yr[j] = xr[j] * rsqrt;
        }
#else
        if (weight) {
            for (size_t i = 0; i < hidden; i++)
                yr[i] = xr[i] * rsqrt * weight[i];
        } else {
            for (size_t i = 0; i < hidden; i++)
                yr[i] = xr[i] * rsqrt;
        }
#endif
    }
}
