/*
 * test_x86_q5k_unit — cpu_x86's Q5_K linear on the GGUF bytes against the
 * cpu_scalar oracle, with an error bound derived, not tuned.
 *
 * The kernel (linear_q4k_raw.c, shared with Q4_K)
 * quantizes each 256-element activation block to int8 with
 * d_x = amax / 127 and round-to-nearest, so every activation element moves
 * by at most d_x / 2, and the 5-bit x int8 products are summed exactly.
 * Hence, per output,
 *
 *   |y_x86 - y_ref| <= sum_b sum_i |w_i| * d_x,b / 2   (activation rounding)
 *                    + 2e-5 * sum_i |w_i x_i|          (fp32 accumulation)
 *
 * with w_i dequantized by the format's own row decoder. A wrong nibble or
 * high bit, a wrong scale or min unpack, a wrong row, a truncating round or
 * a dropped tail blows past it. Also checked: every
 * output written (y poisoned, compared by bit pattern — this file builds
 * with -ffast-math), and that cpu_x86 did not leave cpu_scalar's kernel
 * bound.
 *
 * Shapes cover one block, an odd block count, model widths, n_out that is
 * not a multiple of anything, and m = 1 (decode) up to a 64-row prefill
 * chunk; m = 2, 3, 5 and 7 leave the 4-token tiles every remainder.
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

#if !defined(GEIST_BACKEND_CPU_X86) || !defined(GEIST_BACKEND_CPU_SCALAR)
int main(void) {
    printf("SKIP: needs cpu_x86 and cpu_scalar in this build\n");
    return GEIST_TEST_SKIP;
}
#else

#include "heap.h"
#include "quant.h"

static const size_t N_INS[]  = {256, 768, 2560};
static const size_t N_OUTS[] = {7, 64};
static const size_t MS[]     = {1, 2, 3, 4, 5, 7, 16, 64};
constexpr size_t    M_MAX    = 64;
constexpr size_t    QK       = 256;

/* fp16 values exactly representable (d, dmin). */
static const uint16_t D_BITS[] = {0x1C00, 0x1800, 0x2000, 0x1E00}; /* 2^-8, 2^-9, 2^-7, 1.5*2^-8 */
constexpr size_t      N_D      = sizeof D_BITS / sizeof *D_BITS;

static uint32_t g_rng = 0x5B5B2026u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

struct shape_result {
    size_t unwritten;
    size_t over_bound;
    double worst_ratio; /* max |dy| / bound */
};

static void check_outputs(size_t               n_in,
                          size_t               n_out,
                          size_t               m,
                          const uint8_t       *raw,
                          const float         *x,
                          const float         *y_ref,
                          const float         *y,
                          struct shape_result *r) {
    constexpr float POISON = -7.5e30f;
    uint32_t        poison_bits;
    memcpy(&poison_bits, &POISON, sizeof poison_bits);
    const size_t nb = n_in / QK;
    for (size_t i = 0; i < m; i++) {
        const float *xi = x + i * n_in;
        for (size_t j = 0; j < n_out; j++) {
            uint32_t bits;
            memcpy(&bits, &y[i * n_out + j], sizeof bits);
            if (bits == poison_bits) {
                r->unwritten++;
                continue;
            }
            float wrow[2560];
            dequant_q5_K_row(n_in, raw + j * nb * Q5_K_BLOCK_BYTES, wrow);
            double bound = 0.0, mag = 0.0;
            for (size_t b = 0; b < nb; b++) {
                float amax = 0.0f;
                for (size_t k = 0; k < QK; k++) {
                    amax = fmaxf(amax, fabsf(xi[b * QK + k]));
                }
                const double half_dx = (double) amax / 127.0 / 2.0;
                for (size_t k = 0; k < QK; k++) {
                    const double wv = (double) wrow[b * QK + k];
                    bound += fabs(wv) * half_dx;
                    mag += fabs(wv * (double) xi[b * QK + k]);
                }
            }
            const double lim = bound + 2e-5 * mag + 1e-7;
            const double d   = fabs((double) y[i * n_out + j] - (double) y_ref[i * n_out + j]);
            if (!(d <= lim)) {
                r->over_bound++;
            }
            const double ratio = bound > 0.0 ? d / bound : 0.0;
            r->worst_ratio     = ratio > r->worst_ratio ? ratio : r->worst_ratio;
        }
    }
}

static int check_shape(size_t                n_in,
                       size_t                n_out,
                       struct geist_backend *be_ref,
                       struct geist_backend *be_x86,
                       const float          *x,
                       float                *y_ref,
                       float                *y) {
    const size_t nb     = n_in / QK;
    const size_t nbytes = n_out * nb * Q5_K_BLOCK_BYTES;
    uint8_t     *raw    = heap_alloc_array_aligned(uint8_t, nbytes);
    if (raw == nullptr) {
        fprintf(stderr, "ERROR: weight allocation failed\n");
        return 1;
    }
    for (size_t b = 0; b < n_out * nb; b++) {
        uint8_t *blk = raw + b * Q5_K_BLOCK_BYTES;
        for (size_t k = 4; k < Q5_K_BLOCK_BYTES; k++) {
            blk[k] =
                    (uint8_t) next_u32(); /* any 6-bit scales and mins, any nibbles and high bits */
        }
        const uint16_t d = D_BITS[next_u32() % N_D], dmin = D_BITS[next_u32() % N_D];
        memcpy(blk, &d, sizeof d);
        memcpy(blk + 2, &dmin, sizeof dmin);
    }
    struct geist_weight w_ref = {.raw        = raw,
                                 .raw_nbytes = nbytes,
                                 .n_in       = (int32_t) n_in,
                                 .n_out      = (int32_t) n_out,
                                 .dtype      = GEIST_DTYPE_Q5_K};
    struct geist_weight w_x86 = w_ref;
    int                 fails = 0;
    if (be_ref->desc->vtbl->resolve_weight(be_ref, &w_ref) != GEIST_OK ||
        be_x86->desc->vtbl->resolve_weight(be_x86, &w_x86) != GEIST_OK) {
        fprintf(stderr, "FAIL: n_in=%zu n_out=%zu: resolve_weight refused\n", n_in, n_out);
        fails = 1;
        goto out;
    }
    if (w_x86.linear_m1 == w_ref.linear_m1 || w_x86.linear_mN == w_ref.linear_mN) {
        fprintf(stderr, "FAIL: cpu_x86 left cpu_scalar's Q5_K kernel bound\n");
        fails = 1;
        goto out;
    }
    for (size_t mi = 0; mi < sizeof MS / sizeof *MS; mi++) {
        const size_t m = MS[mi];
        if (m == 1) {
            w_ref.linear_m1(x, &w_ref, be_ref, y_ref);
        } else {
            w_ref.linear_mN(m, x, &w_ref, be_ref, y_ref);
        }
        for (size_t i = 0; i < m * n_out; i++) {
            y[i] = -7.5e30f;
        }
        if (m == 1) {
            w_x86.linear_m1(x, &w_x86, be_x86, y);
        } else {
            w_x86.linear_mN(m, x, &w_x86, be_x86, y);
        }
        struct shape_result r = {0};
        check_outputs(n_in, n_out, m, raw, x, y_ref, y, &r);
        if (r.unwritten != 0 || r.over_bound != 0) {
            fprintf(stderr,
                    "FAIL: n_in=%zu n_out=%zu m=%zu: %zu unwritten, %zu past the bound\n",
                    n_in,
                    n_out,
                    m,
                    r.unwritten,
                    r.over_bound);
            fails++;
        } else {
            printf("  n_in=%-4zu n_out=%-2zu m=%-2zu worst |dy|/bound %.3f\n",
                   n_in,
                   n_out,
                   m,
                   r.worst_ratio);
        }
    }
out:;
    void *p = raw;
    safe_free(&p);
    return fails;
}

int main(void) {
    struct geist_backend *be_x86 = nullptr;
    struct geist_backend *be_ref = nullptr;
    if (geist_backend_create("cpu_x86", nullptr, nullptr, &be_x86) != GEIST_OK ||
        be_x86 == nullptr) {
        printf("SKIP: cpu_x86 backend did not register on this host\n");
        return GEIST_TEST_SKIP;
    }
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be_ref) != GEIST_OK ||
        be_ref == nullptr) {
        geist_backend_destroy(be_x86);
        printf("SKIP: cpu_scalar backend did not register\n");
        return GEIST_TEST_SKIP;
    }
    const size_t n_in_max  = N_INS[sizeof N_INS / sizeof *N_INS - 1];
    const size_t n_out_max = N_OUTS[sizeof N_OUTS / sizeof *N_OUTS - 1];
    float       *x         = heap_alloc_array_aligned(float, M_MAX *n_in_max);
    float       *y_ref     = heap_alloc_array_aligned(float, M_MAX *n_out_max);
    float       *y         = heap_alloc_array_aligned(float, M_MAX *n_out_max);
    if (x == nullptr || y_ref == nullptr || y == nullptr) {
        fprintf(stderr, "ERROR: activation allocation failed\n");
        return GEIST_TEST_ERROR;
    }

    int fails = 0;
    for (size_t a = 0; a < sizeof N_INS / sizeof *N_INS; a++) {
        const size_t n_in = N_INS[a];
        /* Row-major [M_MAX, n_in], varied magnitude per row, a zero block in
         * row 0 (amax == 0 must give an all-zero block, not NaN). */
        for (size_t i = 0; i < M_MAX; i++) {
            const float row_scale = 0.25f + (float) (i % 7);
            for (size_t k = 0; k < n_in; k++) {
                x[i * n_in + k] =
                        row_scale * ((float) (next_u32() >> 8) * (2.0f / 16777216.0f) - 1.0f);
            }
        }
        for (size_t k = 0; k < QK; k++) {
            x[k] = 0.0f;
        }
        for (size_t b = 0; b < sizeof N_OUTS / sizeof *N_OUTS; b++) {
            fails += check_shape(n_in, N_OUTS[b], be_ref, be_x86, x, y_ref, y);
        }
    }

    safe_free((void **) &x);
    safe_free((void **) &y_ref);
    safe_free((void **) &y);
    geist_backend_destroy(be_ref);
    geist_backend_destroy(be_x86);
    if (fails != 0) {
        fprintf(stderr, "FAIL: %d check(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: cpu_x86 Q5_K within the activation-rounding bound on every shape\n");
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_X86 && GEIST_BACKEND_CPU_SCALAR */
