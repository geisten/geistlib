/*
 * test_q8_0_prefill_unit — cpu_neon's W8A8 Q8_0 prefill kernel (#7).
 *
 * linear_q8_0_w8a8_prefill_pre against a double reference of the same
 * quantized operands: y[t][n] = sum_b d_w[n][b] * d_x[t][b] * sum_i q*x.
 * Shapes cover full 4-row tiles and a ragged last tile (n_out 4, 7, 64,
 * 130) and full / partial groups of 8 tokens (m 8, 13, 64), with the SDOT
 * tile kernel on and off (GEIST_Q8_0_SDOT_PREFILL is read once, so the
 * second run is a child process).
 */
#define _POSIX_C_SOURCE 200809L

#include "quant.h"
#include "quant_blocks.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__ARM_NEON) || !defined(__ARM_FEATURE_DOTPROD)
int main(void) {
    printf("SKIP: needs NEON dotprod\n");
    return GEIST_TEST_SKIP;
}
#else

static uint32_t prng(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

static uint16_t f2h(float f) {
    _Float16 h = (_Float16) f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

static int check(size_t m, size_t n_out, size_t n_in) {
    const size_t         nb = n_in / 32;
    uint32_t             s  = (uint32_t) (m * 131 + n_out * 7 + n_in);
    struct block_q8_0_t *w  = malloc(n_out * nb * sizeof *w);
    int8_t              *xq = malloc(m * n_in);
    float               *xs = malloc(m * nb * sizeof *xs);
    float               *y  = malloc(m * n_out * sizeof *y);
    if (!w || !xq || !xs || !y) {
        return 1;
    }
    for (size_t i = 0; i < n_out * nb; i++) {
        w[i].d = f2h(0.001f + 0.01f * (float) (prng(&s) % 1000) / 1000.0f);
        for (int k = 0; k < 32; k++) {
            w[i].qs[k] = (int8_t) ((int) (prng(&s) % 255) - 127);
        }
    }
    for (size_t i = 0; i < m * n_in; i++) {
        xq[i] = (int8_t) ((int) (prng(&s) % 255) - 127);
    }
    for (size_t i = 0; i < m * nb; i++) {
        xs[i] = 0.002f + 0.01f * (float) (prng(&s) % 1000) / 1000.0f;
    }
    linear_q8_0_w8a8_prefill_pre(m, n_in, n_out, xq, xs, w, y);
    double worst = 0.0, scale = 0.0;
    for (size_t t = 0; t < m; t++) {
        for (size_t n = 0; n < n_out; n++) {
            double ref = 0.0;
            for (size_t b = 0; b < nb; b++) {
                int32_t dot = 0;
                for (int k = 0; k < 32; k++) {
                    dot += (int32_t) w[n * nb + b].qs[k] * xq[t * n_in + b * 32 + (size_t) k];
                }
                _Float16 h;
                memcpy(&h, &w[n * nb + b].d, 2);
                ref += (double) (float) h * (double) xs[t * nb + b] * (double) dot;
            }
            scale          = fmax(scale, fabs(ref));
            const double d = fabs((double) y[t * n_out + n] - ref);
            worst          = d > worst || isnan(d) ? d : worst;
        }
    }
    const bool ok = worst <= 1e-5 * (scale + 1e-6);
    printf("  m=%3zu n_out=%3zu n_in=%3zu: max|d| %.3g of scale %.3g %s\n",
           m,
           n_out,
           n_in,
           worst,
           scale,
           ok ? "ok" : "FAIL");
    free(w);
    free(xq);
    free(xs);
    free(y);
    return ok ? 0 : 1;
}

int main(void) {
    static const size_t shapes[][3] = {
            {8, 4, 256},
            {13, 7, 256},
            {64, 64, 512},
            {64, 130, 1024},
            {8, 130, 256},
            {5, 64, 256},
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
        fails += check(shapes[i][0], shapes[i][1], shapes[i][2]);
    }
    printf("%s: %s\n",
           fails ? "FAIL" : "PASS",
           getenv("GEIST_Q8_0_SDOT_PREFILL") ? "scalar rows" : "SDOT tiles");
    return fails ? GEIST_TEST_FAIL : 0;
}
#endif
