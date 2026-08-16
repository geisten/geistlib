/*
 * audio_linear — quantized matmul kernels for the audio tower, bound once
 * at encoder creation from the runtime hardware probe (the same load-time
 * binding pattern the backend kernel catalogs use), instead of compile-time
 * #if in the forward pass.
 *
 * Layer: ARCHITECTURE (audio_conformer). Intrinsics live ONLY in
 * audio_linear.c; callers dispatch through the bound function pointers.
 */
#ifndef AUDIO_LINEAR_H
#define AUDIO_LINEAR_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "audio_linear.h is internal to the architecture layer."
#endif

#include <stddef.h>
#include <stdint.h>

/* W8A8: y[m,n] = scale_x * w_scales[n] * sum_k q8(x)[m,k] * w_q8[n,k].
 * Activations quantized once with the STATIC scale derived from the layer's
 * clip range; caller passes scale_x_inv = 1 / scale_x. */
typedef void (*audio_linear_w8a8_fn)(const int8_t *w_q8,
                                     const float  *w_scales,
                                     const float  *x,
                                     float         scale_x_inv,
                                     float         scale_x,
                                     size_t        m,
                                     size_t        in_dim,
                                     size_t        out_dim,
                                     float        *y);

/* W8A32: y[m,n] = sum_k x[m,k] * (w_q8[n,k] * w_scales[n]). */
typedef void (*audio_linear_w8a32_fn)(const int8_t *w_q8,
                                      const float  *w_scales,
                                      const float  *x,
                                      size_t        m,
                                      size_t        in_dim,
                                      size_t        out_dim,
                                      float        *y);

struct audio_linear_ops {
    audio_linear_w8a8_fn  w8a8;
    audio_linear_w8a32_fn w8a32;
    const char           *name; /* "neon" / "scalar" — logged at bind, pinned by tests */
};

/* Bind from the hw probe, once (cached). GEIST_AUDIO_KERNEL=scalar forces
 * the portable kernels regardless of the probe — the parity test's lever.
 * audio_encoder_create calls this at load; every later call is a pointer
 * read. */
const struct audio_linear_ops *audio_linear_bind(void);

/* Test hook: drop the cached binding and re-evaluate probe + env. Lets the
 * parity test compare forced-scalar against probed-best in one process. */
const struct audio_linear_ops *audio_linear_rebind(void);

#endif /* AUDIO_LINEAR_H */
