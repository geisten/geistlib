/*
 * test_q5_0_w5a8_unit — cpu_neon's W5A8 Q5_0 kernels (#528).
 *
 * linear_q5_0_decode_w5a8_pre and linear_q5_0_w5a8_prefill_pre against a
 * double reference of the same quantized operands,
 *   y[t][n] = sum_b d_w[n][b] * d_x[t][b] * sum_i (code_i - 16) * x_i,
 * with the 5-bit codes read bit by bit from the block (element j: low
 * nibble of qs[j] plus bit j of qh; element j + 16: high nibble plus bit
 * j + 16), so a wrong fifth bit, nibble half or byte of qh in the NEON
 * unpack fails. Shapes cover full 4-row tiles and a ragged last tile
 * (n_out 4, 7, 64, 130), full / partial groups of 8 tokens and fewer than
 * 8 (m 1, 5, 8, 13, 64). Every row of the prefill must also be the same
 * bits as a one-row prefill of that token: the row tiles and the per-row
 * tail give a token one result, whatever came with it (#527).
 */
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

/* Element e's code minus 16, from the block's bytes. */
static int32_t code(const struct block_q5_0_t *b, size_t e) {
    const size_t   j   = e % 16;
    const uint32_t nib = e < 16 ? (b->qs[j] & 0x0Fu) : (uint32_t) (b->qs[j] >> 4);
    const uint32_t bit = (uint32_t) (b->qh[e / 8] >> (e % 8)) & 1u;
    return (int32_t) (nib | (bit << 4)) - 16;
}

static double
ref_dot(size_t nb, const struct block_q5_0_t *row, const int8_t *xq, const float *xs) {
    double ref = 0.0;
    for (size_t b = 0; b < nb; b++) {
        int32_t dot = 0;
        for (size_t k = 0; k < 32; k++) {
            dot += code(&row[b], k) * xq[b * 32 + k];
        }
        _Float16 h;
        memcpy(&h, &row[b].d, 2);
        ref += (double) (float) h * (double) xs[b] * (double) dot;
    }
    return ref;
}

static int check(size_t m, size_t n_out, size_t n_in) {
    const size_t         nb = n_in / 32;
    uint32_t             s  = (uint32_t) (m * 131 + n_out * 7 + n_in);
    struct block_q5_0_t *w  = malloc(n_out * nb * sizeof *w);
    int8_t              *xq = malloc(m * n_in);
    float               *xs = malloc(m * nb * sizeof *xs);
    float               *y  = malloc(m * n_out * sizeof *y);
    float               *y1 = malloc(n_out * sizeof *y1);
    if (!w || !xq || !xs || !y || !y1) {
        return 1;
    }
    for (size_t i = 0; i < n_out * nb; i++) {
        w[i].d = f2h(0.001f + 0.01f * (float) (prng(&s) % 1000) / 1000.0f);
        for (int k = 0; k < 4; k++) {
            w[i].qh[k] = (uint8_t) prng(&s);
        }
        for (int k = 0; k < 16; k++) {
            w[i].qs[k] = (uint8_t) prng(&s);
        }
    }
    for (size_t i = 0; i < m * n_in; i++) {
        xq[i] = (int8_t) ((int) (prng(&s) % 255) - 127);
    }
    for (size_t i = 0; i < m * nb; i++) {
        xs[i] = 0.002f + 0.01f * (float) (prng(&s) % 1000) / 1000.0f;
    }
    linear_q5_0_w5a8_prefill_pre(m, n_in, n_out, xq, xs, w, y);
    double worst = 0.0, scale = 0.0;
    bool   same = true;
    for (size_t t = 0; t < m; t++) {
        for (size_t n = 0; n < n_out; n++) {
            const double ref = ref_dot(nb, w + n * nb, xq + t * n_in, xs + t * nb);
            scale            = fmax(scale, fabs(ref));
            const double d   = fabs((double) y[t * n_out + n] - ref);
            worst            = d > worst || isnan(d) ? d : worst;
        }
        linear_q5_0_w5a8_prefill_pre(1, n_in, n_out, xq + t * n_in, xs + t * nb, w, y1);
        same = same && memcmp(y1, y + t * n_out, n_out * sizeof *y1) == 0;
    }
    /* Decode, on token 0. */
    linear_q5_0_decode_w5a8_pre(n_in, n_out, xs, xq, w, y1);
    for (size_t n = 0; n < n_out; n++) {
        const double d = fabs((double) y1[n] - ref_dot(nb, w + n * nb, xq, xs));
        worst          = d > worst || isnan(d) ? d : worst;
    }
    const bool ok = worst <= 1e-5 * (scale + 1e-6) && same;
    printf("  m=%3zu n_out=%3zu n_in=%4zu: max|d| %.3g of scale %.3g, rows %s %s\n",
           m,
           n_out,
           n_in,
           worst,
           scale,
           same ? "as one-row calls" : "DIFFER from one-row calls",
           ok ? "ok" : "FAIL");
    free(w);
    free(xq);
    free(xs);
    free(y);
    free(y1);
    return ok ? 0 : 1;
}

int main(void) {
    static const size_t shapes[][3] = {
            {1, 7, 256},
            {5, 64, 256},
            {8, 4, 256},
            {13, 7, 256},
            {64, 64, 512},
            {64, 130, 1024},
            {8, 130, 256},
            {13, 64, 1408},
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
        fails += check(shapes[i][0], shapes[i][1], shapes[i][2]);
    }
    printf("%s: cpu_neon Q5_0 W5A8 decode and prefill\n", fails ? "FAIL" : "PASS");
    return fails ? GEIST_TEST_FAIL : 0;
}
#endif
