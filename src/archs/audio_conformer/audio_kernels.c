#include "audio_kernels.h"

#include <math.h>
#include <stdlib.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>

/* 4-wide fp32 exp approximation for softmax: range-reduce x via x = n*ln(2)+r,
 * compute exp(r) with a 5-term polynomial (Taylor over [-ln2/2, ln2/2]) then
 * scale by 2^n via direct exponent injection. Softmax input is x - max ≤ 0;
 * we clamp from below to avoid producing denormals/NaN for huge negatives.
 * Max relative error vs libm expf for x ∈ [-87, 0] is ~3e-7 — well under
 * the ULP noise that softmax tolerates. */
static inline float32x4_t vexpq_f32_softmax(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));

    const float32x4_t kInvLog2 = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t kLog2    = vdupq_n_f32(0.6931471805599453f);

    float32x4_t fx  = vmulq_f32(x, kInvLog2);
    float32x4_t n_f = vrndnq_f32(fx);
    int32x4_t   n_i = vcvtq_s32_f32(n_f);
    /* r = x - n*ln(2)  ∈  [-ln2/2, ln2/2]. */
    float32x4_t r = vfmsq_f32(x, n_f, kLog2);

    /* exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120  via Horner. */
    float32x4_t poly = vdupq_n_f32(1.0f / 120.0f);
    poly             = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), r, poly);
    poly             = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), r, poly);
    poly             = vfmaq_f32(vdupq_n_f32(0.5f), r, poly);
    poly             = vfmaq_f32(vdupq_n_f32(1.0f), r, poly);
    poly             = vfmaq_f32(vdupq_n_f32(1.0f), r, poly);

    /* 2^n via exponent field injection. */
    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23);
    return vmulq_f32(poly, vreinterpretq_f32_s32(exp_bits));
}
#endif

/* Conv2d (chw, weight oc,ic,kh,kw, ph/pw zero-pad). Standard full-output
 * variant: computes all h_out × w_out positions. */
void conv2d_fp32(const float *in,
                 const float *w,
                 float       *out,
                 int          c_in,
                 int          c_out,
                 int          h_in,
                 int          w_in,
                 int          kh,
                 int          kw,
                 int          sh,
                 int          sw,
                 int          ph,
                 int          pw) {
    conv2d_fp32_from(in, w, out, c_in, c_out, h_in, w_in, kh, kw, sh, sw, ph, pw, 0);
}

/* Incremental variant: only writes outputs at oh >= start_h. Caller must
 * have written outputs [0, start_h) already (or initialized them). With
 * pad=1 + kernel=3 + stride=2, output at oh reads in at h-axis positions
 * [oh*2-1, oh*2, oh*2+1] which are deterministic in `in` regardless of
 * h_in (boundary clipping by h_in is unchanged). This is what makes the
 * incremental subsample bit-equivalent to the full re-run for positions
 * already computed - extending h_in only adds new output positions; old
 * outputs are stable. See docs/audio-chunk-streaming/plan.md Phase 3. */
void conv2d_fp32_from(const float *in,
                      const float *w,
                      float       *out,
                      int          c_in,
                      int          c_out,
                      int          h_in,
                      int          w_in,
                      int          kh,
                      int          kw,
                      int          sh,
                      int          sw,
                      int          ph,
                      int          pw,
                      int          start_h) {
    const int    h_out     = (h_in + 2 * ph - kh) / sh + 1;
    const int    w_out     = (w_in + 2 * pw - kw) / sw + 1;
    const size_t in_plane  = (size_t) h_in * w_in;
    const size_t out_plane = (size_t) h_out * w_out;
    const size_t k_plane   = (size_t) kh * kw;

    if (start_h < 0)
        start_h = 0;
    if (start_h >= h_out)
        return;

    for (int oc = 0; oc < c_out; oc++) {
        for (int oh = start_h; oh < h_out; oh++) {
            for (int ow = 0; ow < w_out; ow++) {
                float acc = 0.0f;
                for (int ic = 0; ic < c_in; ic++) {
                    const float *in_c  = in + (size_t) ic * in_plane;
                    const float *w_oic = w + ((size_t) oc * c_in + ic) * k_plane;
                    for (int khi = 0; khi < kh; khi++) {
                        const int ih = oh * sh + khi - ph;
                        if (ih < 0 || ih >= h_in)
                            continue;
                        for (int kwi = 0; kwi < kw; kwi++) {
                            const int iw = ow * sw + kwi - pw;
                            if (iw < 0 || iw >= w_in)
                                continue;
                            acc += w_oic[khi * kw + kwi] * in_c[(size_t) ih * w_in + iw];
                        }
                    }
                }
                out[(size_t) oc * out_plane + (size_t) oh * w_out + ow] = acc;
            }
        }
    }
}

void layernorm_fp32_ws(
        const float *x, const float *gamma, size_t batch, size_t d, float eps, float *y) {
    const float inv_d = 1.0f / (float) d;
    for (size_t b = 0; b < batch; b++) {
        const float *xb = x + b * d;
        float       *yb = y + b * d;

        float mean = 0.0f;
        for (size_t i = 0; i < d; i++)
            mean += xb[i];
        mean *= inv_d;

        float var = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float dx = xb[i] - mean;
            var += dx * dx;
        }
        var *= inv_d;

        const float scale = 1.0f / sqrtf(var + eps);
        for (size_t i = 0; i < d; i++) {
            yb[i] = (xb[i] - mean) * scale * gamma[i];
        }
    }
}

void silu_fp32(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        x[i]    = v / (1.0f + expf(-v));
    }
}

void clamp_fp32(float *x, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        x[i]    = v < lo ? lo : (v > hi ? hi : v);
    }
}

void glu_fp32(const float *x, size_t batch, size_t d, float *y) {
    /* x is (batch, 2*d), y is (batch, d).
     * y[b, i] = x[b, i] * sigmoid(x[b, d + i]). */
    for (size_t b = 0; b < batch; b++) {
        const float *xa = x + b * 2 * d;
        const float *xb = xa + d;
        float       *yb = y + b * d;
        for (size_t i = 0; i < d; i++) {
            yb[i] = xa[i] * (1.0f / (1.0f + expf(-xb[i])));
        }
    }
}

void depthwise_conv1d_causal_fp32(
        const float *in, const float *w, float *out, int channels, int t_in, int kernel) {
    const int left_pad = kernel - 1;
    for (int c = 0; c < channels; c++) {
        const float *in_c  = in + (size_t) c * t_in;
        const float *w_c   = w + (size_t) c * kernel;
        float       *out_c = out + (size_t) c * t_in;
        for (int t = 0; t < t_in; t++) {
            float acc = 0.0f;
            for (int k = 0; k < kernel; k++) {
                int src = t + k - left_pad;
                if (src >= 0 && src < t_in)
                    acc += w_c[k] * in_c[src];
            }
            out_c[t] = acc;
        }
    }
}

void softmax_fp32(float *x, size_t n_rows, size_t d) {
    /* GEIST_FAST_TANH=1 enables vForce vvexpf on macOS (~3x faster than the
     * NEON poly path, ~1-2 ULP drift). Reuses the existing Vision env flag.
     * Pi 5 and other ARM targets use the NEON poly exp by default. */
    static int fast_expf = -1;
    if (fast_expf < 0) {
        const char *s = getenv("GEIST_FAST_TANH");
        fast_expf     = (s != nullptr && s[0] == '1') ? 1 : 0;
    }
#if defined(__APPLE__)
    extern void vvexpf(float *y, const float *x, const int *n_int);
#endif

    for (size_t r = 0; r < n_rows; r++) {
        float *row = x + r * d;

        /* 1) max scan. */
        float m = row[0];
#if defined(__ARM_NEON)
        {
            size_t      i  = 0;
            float32x4_t vm = vdupq_n_f32(row[0]);
            for (; i + 4 <= d; i += 4)
                vm = vmaxq_f32(vm, vld1q_f32(row + i));
            m = vmaxvq_f32(vm);
            for (; i < d; i++)
                if (row[i] > m)
                    m = row[i];
        }
#else
        for (size_t i = 1; i < d; i++)
            if (row[i] > m)
                m = row[i];
#endif

        /* 2) sub-max, exp, sum. */
        float sum = 0.0f;
#if defined(__APPLE__)
        if (fast_expf) {
            const int n_int = (int) d;
            for (size_t i = 0; i < d; i++)
                row[i] -= m;
            vvexpf(row, row, &n_int);
            for (size_t i = 0; i < d; i++)
                sum += row[i];
        } else
#endif
        {
#if defined(__ARM_NEON)
            const float32x4_t vneg_m = vdupq_n_f32(-m);
            float32x4_t       vsum   = vdupq_n_f32(0.0f);
            size_t            j      = 0;
            for (; j + 4 <= d; j += 4) {
                float32x4_t v = vaddq_f32(vld1q_f32(row + j), vneg_m);
                v             = vexpq_f32_softmax(v);
                vst1q_f32(row + j, v);
                vsum = vaddq_f32(vsum, v);
            }
            sum = vaddvq_f32(vsum);
            for (; j < d; j++) {
                row[j] = expf(row[j] - m);
                sum += row[j];
            }
#else
            for (size_t i = 0; i < d; i++) {
                row[i] = expf(row[i] - m);
                sum += row[i];
            }
#endif
        }

        /* 3) normalize. */
        const float inv = 1.0f / sum;
#if defined(__ARM_NEON)
        {
            const float32x4_t vinv = vdupq_n_f32(inv);
            size_t            k    = 0;
            for (; k + 4 <= d; k += 4) {
                vst1q_f32(row + k, vmulq_f32(vld1q_f32(row + k), vinv));
            }
            for (; k < d; k++)
                row[k] *= inv;
        }
#else
        for (size_t i = 0; i < d; i++)
            row[i] *= inv;
#endif
    }
}
