/*
 * test_vision_kernels_unit — the Gemma 4 vision tower's own kernels
 * (src/archs/vision_siglip/vision_kernels.c) against double-precision
 * references written straight from their header contracts.
 *
 * The tower's parity tests need the extracted vision_tower.safetensors and
 * HF dumps, which CI does not have, so without this test the kernels ran in
 * no CI suite at all (coverage ratchet, #141). Shapes are small and odd on
 * purpose: several heads, a head_dim whose quarter is > 1, a non-square
 * patch grid, and a padding position (-1, which the RoPE treats as 0).
 */
#include "vision_kernels.h"

#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_rng = 0x9e3779b9u;

static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float) (g_rng >> 8) / (float) (1u << 24) * 2.0f - 1.0f;
}

static void fill(size_t n, float x[static n]) {
    for (size_t i = 0; i < n; i++) {
        x[i] = frand();
    }
}

/* Largest |got - want| / max(1, |want|) over n values. */
static double max_rel_err(size_t n, const float got[static n], const double want[static n]) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double) got[i] - want[i]) / fmax(1.0, fabs(want[i]));
        if (!(d <= worst)) { /* also catches NaN */
            worst = d;
        }
    }
    return worst;
}

static int test_avgpool(void) {
    enum { GH = 6, GW = 9, HID = 5, PH = GH / 3, PW = GW / 3 };
    float  in[GH * GW * HID];
    float  out[PH * PW * HID];
    double want[PH * PW * HID];
    fill(GH * GW * HID, in);
    for (size_t i = 0; i < PH * PW * HID; i++) {
        out[i] = NAN; /* the kernel must overwrite, not accumulate */
    }
    for (size_t py = 0; py < PH; py++) {
        for (size_t px = 0; px < PW; px++) {
            for (size_t k = 0; k < HID; k++) {
                double s = 0.0;
                for (size_t dy = 0; dy < 3; dy++) {
                    for (size_t dx = 0; dx < 3; dx++) {
                        s += in[((py * 3 + dy) * GW + px * 3 + dx) * HID + k];
                    }
                }
                want[(py * PW + px) * HID + k] = s / 9.0;
            }
        }
    }
    avgpool2d_k3_fp32(in, out, GH, GW, HID);
    const double err = max_rel_err(PH * PW * HID, out, want);
    printf("  avgpool2d_k3      max rel err %.2e\n", err);
    return geist_expect(err < 1e-6, "avgpool2d_k3_fp32 matches the 3x3 mean");
}

static int test_rope_2d(void) {
    enum { NT = 5, NH = 3, HD = 16, HALF = HD / 2, PAIR = HALF / 2 };
    const float   theta             = 100.0f;
    const int32_t positions[NT * 2] = {0, 0, 3, 1, 1, 4, -1, 2, 2, -1};
    float         x[NT * NH * HD];
    double        want[NT * NH * HD];
    fill(NT * NH * HD, x);

    for (size_t t = 0; t < NT; t++) {
        for (size_t axis = 0; axis < 2; axis++) {
            const int32_t raw = positions[t * 2 + axis];
            const double  pos = raw < 0 ? 0.0 : (double) raw;
            for (size_t h = 0; h < NH; h++) {
                const float *xh = x + (t * NH + h) * HD + axis * HALF;
                double      *wh = want + (t * NH + h) * HD + axis * HALF;
                for (size_t k = 0; k < PAIR; k++) {
                    const double inv_freq = pow((double) theta, -(double) (2 * k) / (double) HALF);
                    const double a = xh[k], b = xh[k + PAIR];
                    const double c = cos(pos * inv_freq), s = sin(pos * inv_freq);
                    wh[k]        = a * c - b * s;
                    wh[k + PAIR] = b * c + a * s;
                }
            }
        }
    }
    rope_2d_split_fp32(NT, NH, HD, theta, x, positions);
    const double err = max_rel_err(NT * NH * HD, x, want);
    printf("  rope_2d_split     max rel err %.2e\n", err);
    return geist_expect(err < 1e-5, "rope_2d_split_fp32 rotates each half by its own axis");
}

static int test_attention(void) {
    enum { NT = 7, NH = 3, HD = 8, N = NT * NH * HD };
    float  q[N], k[N], v[N], out[N];
    double want[N];
    fill(N, q);
    fill(N, k);
    fill(N, v);
    for (size_t i = 0; i < N; i++) {
        out[i] = NAN;
    }

    /* softmax(q k^T) v per head, unscaled: Gemma 4's vision attention has
     * scaling 1.0 (q and k are RMS-normed before RoPE). */
    for (size_t h = 0; h < NH; h++) {
        for (size_t i = 0; i < NT; i++) {
            double sc[NT];
            double mx = -INFINITY;
            for (size_t j = 0; j < NT; j++) {
                double d = 0.0;
                for (size_t e = 0; e < HD; e++) {
                    d += (double) q[(i * NH + h) * HD + e] * k[(j * NH + h) * HD + e];
                }
                sc[j] = d;
                mx    = fmax(mx, d);
            }
            double sum = 0.0;
            for (size_t j = 0; j < NT; j++) {
                sc[j] = exp(sc[j] - mx);
                sum += sc[j];
            }
            for (size_t e = 0; e < HD; e++) {
                double o = 0.0;
                for (size_t j = 0; j < NT; j++) {
                    o += sc[j] / sum * v[(j * NH + h) * HD + e];
                }
                want[(i * NH + h) * HD + e] = o;
            }
        }
    }
    vision_attention_bidir_fp32(NT, NH, HD, q, k, v, out);
    const double err = max_rel_err(N, out, want);
    printf("  attention_bidir   max rel err %.2e\n", err);
    return geist_expect(err < 1e-5, "vision_attention_bidir_fp32 matches softmax(q k^T) v");
}

int main(void) {
    int fails = 0;
    fails += test_avgpool();
    fails += test_rope_2d();
    fails += test_attention();
    if (fails != 0) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: vision avg-pool, 2D RoPE and bidirectional attention match the double "
           "references\n");
    return GEIST_TEST_PASS;
}
