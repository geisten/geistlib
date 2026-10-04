/* tests/deltanet_ref.h — scalar Gated-DeltaNet mixer oracle shared by the
 * backend DeltaNet parity tests (Vulkan, Metal). It mirrors the host path in
 * layer_deltanet.c one token at a time:
 *
 *   causal conv over the (conv_kernel - 1)-row window + SiLU, per-head q/k
 *   L2 norm (q also scaled by 1/sqrt(head_k)), gated delta-rule update of the
 *   recurrent state, then RMS norm * norm_w * silu(z) into z.
 *
 * Value heads share key heads tiled (hk = h % n_k_heads), the 27B layout.
 *
 * Extents (products over runtime dims, so documented here rather than as
 * [static]):
 *   qkv          seq * (2 * n_k_heads * head_k + n_v_heads * head_v)  in/out
 *   z            seq * n_v_heads * head_v                             in/out
 *   beta, alpha  seq * n_v_heads
 *   conv_w       conv_dim * conv_kernel  (conv_dim = the qkv row width)
 *   conv_state   (conv_kernel - 1) * conv_dim                         in/out
 *   delta_state  n_v_heads * head_k * head_v                          in/out
 *
 * On return qkv holds the post-conv, normed rows, z the mixer output, and
 * both states have advanced by seq tokens. conv_kernel must be >= 2. */
#ifndef GEIST_TESTS_DELTANET_REF_H
#define GEIST_TESTS_DELTANET_REF_H

#include "test_helpers.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static inline void deltanet_mix_ref(size_t       seq,
                                    size_t       n_k_heads,
                                    size_t       n_v_heads,
                                    size_t       head_k,
                                    size_t       head_v,
                                    size_t       conv_kernel,
                                    float        eps,
                                    const float *beta,
                                    const float *alpha,
                                    const float *conv_w,
                                    const float  ssm_a[static n_v_heads],
                                    const float  dt_bias[static n_v_heads],
                                    const float  norm_w[static head_v],
                                    float       *qkv,
                                    float       *z,
                                    float       *conv_state,
                                    float       *delta_state) {
    const size_t keyd = n_k_heads * head_k, vd = n_v_heads * head_v, cd = 2 * keyd + vd;
    const size_t K   = conv_kernel;
    float       *y   = xmalloc(cd * sizeof(float));
    float       *out = xmalloc(head_v * sizeof(float));
    for (size_t t = 0; t < seq; t++) {
        for (size_t c = 0; c < cd; c++) {
            float acc = 0.0f;
            for (size_t r = 0; r < K; r++) {
                const float x = r + 1 < K ? conv_state[r * cd + c] : qkv[t * cd + c];
                acc += conv_w[c * K + r] * x;
            }
            y[c] = geist_test_silu(acc);
        }
        memmove(conv_state, conv_state + cd, (K - 2) * cd * sizeof(float));
        memcpy(conv_state + (K - 2) * cd, qkv + t * cd, cd * sizeof(float));
        memcpy(qkv + t * cd, y, cd * sizeof(float));

        for (size_t h = 0; h < n_k_heads; h++) {
            double qss = 0.0, kss = 0.0;
            for (size_t i = 0; i < head_k; i++) {
                qss += (double) y[h * head_k + i] * y[h * head_k + i];
                kss += (double) y[keyd + h * head_k + i] * y[keyd + h * head_k + i];
            }
            const float qi = (float) (1.0 / sqrt(qss + eps)) / sqrtf((float) head_k);
            const float ki = (float) (1.0 / sqrt(kss + eps));
            for (size_t i = 0; i < head_k; i++) {
                qkv[t * cd + h * head_k + i] *= qi;
                qkv[t * cd + keyd + h * head_k + i] *= ki;
            }
        }
        for (size_t h = 0; h < n_v_heads; h++) {
            const size_t hk = h % n_k_heads;
            const float  b  = 1.0f / (1.0f + expf(-beta[t * n_v_heads + h]));
            const float  decay =
                    expf(ssm_a[h] * log1pf(expf(alpha[t * n_v_heads + h] + dt_bias[h])));
            for (size_t j = 0; j < head_v; j++) {
                out[j] = 0.0f;
            }
            for (size_t j = 0; j < head_v; j++) {
                float mem = 0.0f;
                for (size_t i = 0; i < head_k; i++) {
                    float *s = delta_state + (h * head_k + i) * head_v + j;
                    *s *= decay;
                    mem += *s * qkv[t * cd + keyd + hk * head_k + i];
                }
                const float d = (qkv[t * cd + 2 * keyd + h * head_v + j] - mem) * b;
                for (size_t i = 0; i < head_k; i++) {
                    float *s = delta_state + (h * head_k + i) * head_v + j;
                    *s += qkv[t * cd + keyd + hk * head_k + i] * d;
                    out[j] += *s * qkv[t * cd + hk * head_k + i];
                }
            }
            double ss = 0.0;
            for (size_t j = 0; j < head_v; j++) {
                ss += (double) out[j] * out[j];
            }
            const float inv = (float) (1.0 / sqrt(ss / (double) head_v + eps));
            for (size_t j = 0; j < head_v; j++) {
                z[t * vd + h * head_v + j] =
                        out[j] * inv * norm_w[j] * geist_test_silu(z[t * vd + h * head_v + j]);
            }
        }
    }
    free(y);
    free(out);
}

#endif /* GEIST_TESTS_DELTANET_REF_H */
