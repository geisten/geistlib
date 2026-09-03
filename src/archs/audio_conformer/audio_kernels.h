/*
 * Audio-encoder-specific kernels: 2D convolution, weight-only LayerNorm,
 * ReLU. All FP32, NCHW (PyTorch native) layout.
 */
#ifndef AUDIO_KERNELS_H
#define AUDIO_KERNELS_H

#include <stddef.h>

/* Conv2D, NCHW, no bias. Stride and padding configurable.
 *   in:  (c_in, h_in, w_in)
 *   w:   (c_out, c_in, kh, kw)
 *   out: (c_out, h_out, w_out)
 *     h_out = (h_in + 2*ph - kh) / sh + 1
 *     w_out = (w_in + 2*pw - kw) / sw + 1
 * Indices outside [0, h_in) / [0, w_in) are treated as zeros (zero-padding). */
void conv2d_fp32(int c_in,
                 int c_out,
                 int h_in,
                 int w_in,
                 int kh,
                 int kw,
                 int sh,
                 int sw,
                 int ph,
                 int pw,
                 const float in[static c_in * h_in * w_in],
                 const float w[static c_out * c_in * kh * kw],
                 float      *out);

/* Incremental Conv2D: only writes outputs at oh >= start_h. With
 * deterministic kernel taps relative to oh, outputs at oh < start_h
 * computed under an earlier (smaller) h_in remain bit-equivalent to the
 * current values, so the caller can keep the prior cached writes and
 * skip them here. Used by the Phase-3 incremental subsample. */
void conv2d_fp32_from(int c_in,
                      int c_out,
                      int h_in,
                      int w_in,
                      int kh,
                      int kw,
                      int sh,
                      int sw,
                      int ph,
                      int pw,
                      int start_h,
                      const float in[static c_in * h_in * w_in],
                      const float w[static c_out * c_in * kh * kw],
                      float      *out);

/* LayerNorm with weight-only (no bias), normalising over the last axis.
 *   x:     (batch, d)   — `batch` rows of length `d`
 *   gamma: (d,)
 *   y[b, i] = ((x[b, i] - mean(x[b])) / sqrt(var(x[b]) + eps)) * gamma[i]
 * In-place safe (y == x). */
void layernorm_fp32_ws(size_t      batch,
                       size_t      d,
                       float       eps,
                       const float x[static batch * d],
                       const float gamma[static d],
                       float       y[static batch * d]);

/* In-place SiLU (a.k.a. swish): y = x * sigmoid(x). */
void silu_fp32(size_t n, float x[static n]);

/* In-place clamp: x[i] = clamp(x[i], lo, hi). */
void clamp_fp32(size_t n, float x[static n], float lo, float hi);

/* In-place GLU on rows: split last axis (length 2*d) in half, output dim d.
 *   y[r, i] = x[r, i] * sigmoid(x[r, d + i])    for i in [0, d)
 * x and y may alias (in-place); only the first d * batch elements of y are written. */
void glu_fp32(size_t batch, size_t d, const float x[static batch * 2 * d], float *y);

/* Causal depthwise 1D conv: kernel of length k, channels c, no bias.
 * Pads (k-1) zeros on the LEFT (causal). Stride 1, no dilation.
 *   in:  (c, t)        — column-major over time per channel
 *   w:   (c, k)
 *   out: (c, t)
 * Each output[c, i] = sum_{j=0..k-1} w[c, j] * in[c, i + j - (k-1)]
 *  with in[c, t] = 0 for t < 0. */
void depthwise_conv1d_causal_fp32(int         channels,
                                  int         t_in,
                                  int         kernel,
                                  const float in[static channels * t_in],
                                  const float w[static channels * kernel],
                                  float       out[static channels * t_in]);

/* In-place softmax on each row of length d (n_rows × d total). */
void softmax_fp32(size_t n_rows, size_t d, float *x);

#endif
