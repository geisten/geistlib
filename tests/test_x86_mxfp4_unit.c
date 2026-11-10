/*
 * test_x86_mxfp4_unit — cpu_x86's MXFP4 linear (GPT-OSS's experts) against
 * cpu_scalar's kernel at the shapes GPT-OSS runs (#374), and, where the host
 * runs the AVX-512 VNNI tiles, that a token's output does not depend on how
 * many rows the call has.
 *
 * Bound, derived as in test_x86_q4_0_unit: the kernel quantizes each
 * 32-element activation block to int8 with d_x = amax / 127 and round to
 * nearest, so each activation moves by at most d_x / 2, and the code x int8
 * products are summed exactly. Per output
 *
 *   |y_x86 - y_scalar| <= sum_b sum_i |w_i| d_x,b / 2   (activation rounding)
 *                       + 4e-5 sum_i |w_i x_i|          (two fp32 sums)
 *
 * Shapes: one block, an odd block count (the tiles' last half pair), 2880
 * (GPT-OSS's width and expert size) and 4096; n_out 1, 5 and 37 (every
 * remainder of the 4-row groups) and 2880; m = 1 (decode) up to a 128-row
 * prefill chunk, with every remainder of the 4-token tiles. Exponents cover
 * the E8M0 denormals (e = 0, 1) as well as 2^-8 .. 2^2. Every output must be
 * written (y poisoned, compared by bits: this file builds with
 * -ffast-math).
 *
 * Row-count independence of the prefill kernel (VNNI tier only; the AVX2
 * kernels sum the m % 4 tail tokens in another order): every token of every
 * m-row call must equal, bit for bit, the same token in a 2-row call. (The
 * m = 1 entry is the AVX2 GEMV on every tier, so it is held to the bound.)
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(GEIST_BACKEND_CPU_X86) || !defined(GEIST_BACKEND_CPU_SCALAR) || \
        !(defined(__x86_64__) || defined(_M_X64))
int main(void) {
    printf("SKIP: needs cpu_x86 and cpu_scalar in this build\n");
    return GEIST_TEST_SKIP;
}
#else

#define GEIST_INTERNAL_BACKEND_LAYER
#include "src/backends/cpu_x86/kernel_w4a8.h" /* w4a8_dispatcher_tier */

#include "heap.h"
#include "quant.h"

struct shape {
    size_t n_in, n_out;
};
static const struct shape SHAPES[] = {
        {32, 37}, {96, 5}, {96, 1}, {2880, 37}, {2880, 2880}, {4096, 5}};
static const size_t MS[]  = {1, 2, 3, 4, 5, 7, 16, 17, 64, 128};
constexpr size_t    M_MAX = 128;

static uint32_t g_rng = 0x3C6EF372u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
static float frand(void) {
    return ((float) (next_u32() & 0xffff) / 65536.0f) * 2.0f - 1.0f;
}

/* [n_out, n_in] random blocks; exponents 120..130, every 7th block one of
 * the denormal scales e = 0 / 1 or a small normal one. */
static uint8_t *make_tensor(size_t n_out, size_t n_in, size_t *out_bytes) {
    const size_t nb   = n_out * n_in / MXFP4_BLOCK_ELEMS;
    uint8_t     *blob = xmalloc(nb * MXFP4_BLOCK_BYTES);
    for (size_t b = 0; b < nb; b++) {
        uint8_t *blk = blob + b * MXFP4_BLOCK_BYTES;
        blk[0]       = (uint8_t) (120u + next_u32() % 11u);
        if (b % 7 == 3) {
            blk[0] = (uint8_t) (next_u32() % 3u); /* 0, 1: denormal; 2: 2^-126 */
        }
        for (size_t i = 1; i < MXFP4_BLOCK_BYTES; i++) {
            blk[i] = (uint8_t) next_u32();
        }
    }
    *out_bytes = nb * MXFP4_BLOCK_BYTES;
    return blob;
}

static void poison(size_t n, float *y) {
    const uint32_t bits = 0xFF800001u; /* a NaN no kernel produces */
    for (size_t i = 0; i < n; i++) {
        memcpy(&y[i], &bits, sizeof bits);
    }
}

static bool poisoned(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    return bits == 0xFF800001u;
}

/* Outputs of y (x86) off y_ref (cpu_scalar) by more than the bound. */
static size_t count_over_bound(size_t         n_out,
                               size_t         n_in,
                               size_t         m,
                               const uint8_t *W,
                               const float   *x,
                               const float   *y,
                               const float   *y_ref,
                               double        *worst) {
    float       *row       = xmalloc(n_in * sizeof(float));
    const size_t row_bytes = n_in / MXFP4_BLOCK_ELEMS * MXFP4_BLOCK_BYTES;
    size_t       over      = 0;
    for (size_t j = 0; j < n_out; j++) {
        dequant_mxfp4_row(n_in, W + j * row_bytes, row);
        for (size_t t = 0; t < m; t++) {
            const float *xt    = x + t * n_in;
            double       bound = 0.0, mag = 0.0;
            for (size_t b = 0; b < n_in / 32; b++) {
                float amax = 0.0f;
                for (size_t k = 0; k < 32; k++) {
                    amax = fmaxf(amax, fabsf(xt[b * 32 + k]));
                }
                for (size_t k = 0; k < 32; k++) {
                    const double wv = (double) row[b * 32 + k];
                    bound += fabs(wv) * (double) amax / 127.0 / 2.0;
                    mag += fabs(wv * (double) xt[b * 32 + k]);
                }
            }
            const float  v   = y[t * n_out + j];
            const double err = fabs((double) v - (double) y_ref[t * n_out + j]);
            const double lim = bound + 4e-5 * mag + 1e-30;
            if (poisoned(v) || !(err <= lim)) {
                over++;
            } else if (err / lim > *worst) {
                *worst = err / lim;
            }
        }
    }
    free(row);
    return over;
}

/* Whether cpu_x86 binds the AVX-512 VNNI tiles on this host (the gate of
 * linear_util.h's vnni_tiles_usable, GEIST_FORCE_ISA included). */
static bool vnni_tier(void) {
    return w4a8_dispatcher_tier() >= W4A8_ISA_AVX512_VNNI && __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
}

static int check_shape(struct geist_backend *x86, struct geist_backend *ref, struct shape s) {
    int          fails = 0;
    size_t       bytes = 0;
    uint8_t     *W     = make_tensor(s.n_out, s.n_in, &bytes);
    float       *x     = xmalloc(M_MAX * s.n_in * sizeof(float));
    float       *y     = xmalloc(M_MAX * s.n_out * sizeof(float));
    float       *y_ref = xmalloc(M_MAX * s.n_out * sizeof(float));
    float       *x2    = xmalloc(2 * s.n_in * sizeof(float));
    float       *y2    = xmalloc(2 * s.n_out * sizeof(float));
    char         what[192];
    const bool   tiles = vnni_tier();
    const size_t n_ms  = s.n_out >= 2880 ? 6 : sizeof MS / sizeof MS[0]; /* scalar ref is slow */
    for (size_t i = 0; i < M_MAX * s.n_in; i++) {
        /* rows of different magnitude, and one row of zeros (S = 0) */
        const size_t t = i / s.n_in;
        x[i] = t == 6 ? 0.0f : frand() * (float) (1 + t % 4) * (t % 5 == 2 ? 100.0f : 1.0f);
    }
    struct geist_weight wx = {.raw        = W,
                              .raw_nbytes = bytes,
                              .n_in       = (int32_t) s.n_in,
                              .n_out      = (int32_t) s.n_out,
                              .dtype      = GEIST_DTYPE_MXFP4};
    struct geist_weight wr = wx;
    if (x86->desc->vtbl->resolve_weight(x86, &wx) != GEIST_OK ||
        ref->desc->vtbl->resolve_weight(ref, &wr) != GEIST_OK || wx.linear_m1 == nullptr ||
        wx.linear_mN == nullptr || wr.linear_m1 == nullptr || wr.linear_mN == nullptr) {
        fails += geist_expect(false, "MXFP4 resolves with both kernels on cpu_x86 and cpu_scalar");
        goto out;
    }
    fails += geist_expect(wx.linear_m1 != wr.linear_m1 && wx.linear_mN != wr.linear_mN,
                          "cpu_x86 binds MXFP4 kernels of its own");
    for (size_t mi = 0; mi < n_ms; mi++) {
        const size_t m     = MS[mi];
        double       worst = 0.0;
        poison(m * s.n_out, y);
        if (m == 1) {
            wx.linear_m1(x, &wx, x86, y);
            wr.linear_m1(x, &wr, ref, y_ref);
        } else {
            wx.linear_mN(m, x, &wx, x86, y);
            wr.linear_mN(m, x, &wr, ref, y_ref);
        }
        const size_t over = count_over_bound(s.n_out, s.n_in, m, W, x, y, y_ref, &worst);
        snprintf(what,
                 sizeof what,
                 "n_in=%zu n_out=%zu m=%zu: every output within the bound of cpu_scalar's "
                 "(worst %.2f of it)",
                 s.n_in,
                 s.n_out,
                 m,
                 worst);
        fails += geist_expect(over == 0, what);
        if (!tiles || m == 1) {
            continue;
        }
        bool same = true;
        for (size_t t = 0; t < m && same; t++) {
            memcpy(x2, x + t * s.n_in, s.n_in * sizeof(float));
            memcpy(x2 + s.n_in, x, s.n_in * sizeof(float));
            wx.linear_mN(2, x2, &wx, x86, y2);
            same = memcmp(y2, y + t * s.n_out, s.n_out * sizeof(float)) == 0;
        }
        snprintf(what,
                 sizeof what,
                 "n_in=%zu n_out=%zu m=%zu: each token's row is its bits in a 2-row call",
                 s.n_in,
                 s.n_out,
                 m);
        fails += geist_expect(same, what);
    }
out:
    free(W);
    free(x);
    free(y);
    free(y_ref);
    free(x2);
    free(y2);
    return fails;
}

int main(void) {
    struct geist_backend *x86 = nullptr, *ref = nullptr;
    if (geist_backend_create("cpu_x86", nullptr, nullptr, &x86) != GEIST_OK ||
        geist_backend_create("cpu_scalar", nullptr, nullptr, &ref) != GEIST_OK) {
        printf("SKIP: a CPU backend could not be created\n");
        geist_backend_destroy(x86);
        geist_backend_destroy(ref);
        return GEIST_TEST_SKIP;
    }
    printf("cpu_x86 MXFP4 tier: %s\n", vnni_tier() ? "AVX-512 VNNI tiles" : "AVX2");
    int fails = 0;
    for (size_t k = 0; k < sizeof SHAPES / sizeof SHAPES[0]; k++) {
        fails += check_shape(x86, ref, SHAPES[k]);
    }
    geist_backend_destroy(x86);
    geist_backend_destroy(ref);
    if (fails == 0) {
        printf("PASS test_x86_mxfp4_unit\n");
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

#endif
