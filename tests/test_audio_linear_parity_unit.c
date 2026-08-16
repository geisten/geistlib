/*
 * test_audio_linear_parity_unit — the audio tower's quantized matmuls agree
 * across kernel bindings (#236).
 *
 * Hermetic: synthetic weights/activations at real tower dimensions, no
 * fixtures. Compares the probed-best binding against the forced-scalar one
 * (GEIST_AUDIO_KERNEL=scalar via the audio_linear_rebind test hook):
 *
 *   - w8a8:  integer dot -> results must agree to fp32 rounding of the
 *            final two float multiplies (the int sum itself is exact).
 *   - w8a32: fp32 accumulation order differs between kernels -> relative
 *            tolerance sized for in_dim-term sums.
 *
 * Also checks both against a float64 reference so two identically-broken
 * kernels can't vouch for each other.
 */
#define _POSIX_C_SOURCE 200809L /* setenv */

#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/audio_conformer/audio_linear.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Real tower shapes: FFN 1024->4096 (W8A8), attention 1024->1024 (W8A32). */
#define M 5
#define IN_DIM 1024
#define OUT_DIM 512

/* Deterministic LCG so the fixture never varies between runs/platforms. */
static uint32_t rng_state = 0x2bad5eedu;
static float    frand(float lo, float hi) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((float) (rng_state >> 8) / (float) (1u << 24));
}

static double max_rel_err(const float *a, const float *b, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double denom = fmax(fabs((double) a[i]), 1e-3);
        double rel   = fabs((double) a[i] - (double) b[i]) / denom;
        if (rel > worst)
            worst = rel;
    }
    return worst;
}

int main(void) {
    static int8_t w_q8[OUT_DIM * IN_DIM];
    static float  w_scales[OUT_DIM];
    static float  x[M * IN_DIM];
    static float  y_best[M * OUT_DIM], y_scalar[M * OUT_DIM];

    for (size_t i = 0; i < sizeof(w_q8); i++)
        w_q8[i] = (int8_t) lrintf(frand(-127.0f, 127.0f));
    for (size_t i = 0; i < OUT_DIM; i++)
        w_scales[i] = frand(0.001f, 0.02f);
    /* Activations inside a typical clip range; scale_x derived the same way
     * clip_linear_apply does. */
    const float clip = 8.0f;
    for (size_t i = 0; i < M * IN_DIM; i++)
        x[i] = frand(-clip, clip);
    const float scale_x = clip / 127.0f;

    int fails = 0;

    /* --- bindings: probed-best vs forced-scalar -------------------------- */
    unsetenv("GEIST_AUDIO_KERNEL");
    const struct audio_linear_ops *best = audio_linear_rebind();
    printf("probed binding: %s\n", best->name);
    audio_linear_w8a8_fn  best_w8a8  = best->w8a8;
    audio_linear_w8a32_fn best_w8a32 = best->w8a32;

    setenv("GEIST_AUDIO_KERNEL", "scalar", 1);
    const struct audio_linear_ops *scalar = audio_linear_rebind();
    if (strcmp(scalar->name, "scalar") != 0) {
        fprintf(stderr, "FAIL: GEIST_AUDIO_KERNEL=scalar bound '%s'\n", scalar->name);
        fails++;
    }
    audio_linear_w8a8_fn  scalar_w8a8  = scalar->w8a8;
    audio_linear_w8a32_fn scalar_w8a32 = scalar->w8a32;
    unsetenv("GEIST_AUDIO_KERNEL");
    audio_linear_rebind(); /* leave the process in the default binding */

    /* --- w8a8 parity ------------------------------------------------------ */
    best_w8a8(w_q8, w_scales, x, 1.0f / scale_x, scale_x, M, IN_DIM, OUT_DIM, y_best);
    scalar_w8a8(w_q8, w_scales, x, 1.0f / scale_x, scale_x, M, IN_DIM, OUT_DIM, y_scalar);
    double err = max_rel_err(y_best, y_scalar, M * OUT_DIM);
    printf("w8a8  best-vs-scalar max rel err: %.3e\n", err);
    if (err > 1e-5) {
        fprintf(stderr, "FAIL: w8a8 binding disagreement (%.3e)\n", err);
        fails++;
    }

    /* w8a8 vs float64 reference of the quantized computation. */
    {
        double worst = 0.0;
        for (size_t i = 0; i < M; i++) {
            for (size_t n = 0; n < OUT_DIM; n++) {
                long long isum = 0;
                for (size_t k = 0; k < IN_DIM; k++) {
                    float q = roundf(x[i * IN_DIM + k] / scale_x);
                    q       = q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q);
                    isum += (long long) w_q8[n * IN_DIM + k] * (long long) q;
                }
                double ref   = (double) w_scales[n] * (double) scale_x * (double) isum;
                double got   = (double) y_scalar[i * OUT_DIM + n];
                double denom = fmax(fabs(ref), 1e-3);
                double rel   = fabs(ref - got) / denom;
                if (rel > worst)
                    worst = rel;
            }
        }
        printf("w8a8  scalar-vs-f64ref max rel err: %.3e\n", worst);
        if (worst > 1e-5) {
            fprintf(stderr, "FAIL: w8a8 scalar drifts from reference (%.3e)\n", worst);
            fails++;
        }
    }

    /* --- w8a32 parity ----------------------------------------------------- */
    best_w8a32(w_q8, w_scales, x, M, IN_DIM, OUT_DIM, y_best);
    scalar_w8a32(w_q8, w_scales, x, M, IN_DIM, OUT_DIM, y_scalar);
    err = max_rel_err(y_best, y_scalar, M * OUT_DIM);
    printf("w8a32 best-vs-scalar max rel err: %.3e\n", err);
    if (err > 2e-3) { /* fp32 sums over 1024 large terms with cancellation;
                       * measured ~3e-4 across bindings, 2e-3 leaves headroom
                       * while a broken kernel is off by orders of magnitude */
        fprintf(stderr, "FAIL: w8a32 binding disagreement (%.3e)\n", err);
        fails++;
    }

    /* w8a32 vs float64 reference. */
    {
        double worst = 0.0;
        for (size_t i = 0; i < M; i++) {
            for (size_t n = 0; n < OUT_DIM; n++) {
                double acc = 0.0;
                for (size_t k = 0; k < IN_DIM; k++)
                    acc += (double) w_q8[n * IN_DIM + k] * (double) x[i * IN_DIM + k];
                double ref   = (double) w_scales[n] * acc;
                double got   = (double) y_scalar[i * OUT_DIM + n];
                double denom = fmax(fabs(ref), 1e-3);
                double rel   = fabs(ref - got) / denom;
                if (rel > worst)
                    worst = rel;
            }
        }
        printf("w8a32 scalar-vs-f64ref max rel err: %.3e\n", worst);
        if (worst > 2e-3) {
            fprintf(stderr, "FAIL: w8a32 scalar drifts from reference (%.3e)\n", worst);
            fails++;
        }
    }

    if (fails == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL (%d)\n", fails);
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
