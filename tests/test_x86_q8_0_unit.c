/*
 * test_x86_q8_0_unit — cpu_x86's native Q8_0 x Q8_0 linear against the
 * cpu_scalar oracle, with an error bound derived, not tuned.
 *
 * The kernel quantizes each 32-element activation block to int8 with
 * d_x = amax / 127 and round-to-nearest, so every activation element moves
 * by at most d_x / 2, and the int8 products are summed exactly. Hence, per
 * output,
 *
 *   |y_x86 - y_ref| <= sum_b sum_i |w_i| * d_x,b / 2   (activation rounding)
 *                    + 2e-5 * sum_i |w_i x_i|          (fp32 accumulation)
 *
 * which holds for any data. A wrong block, a wrong row, a truncating round
 * or a dropped tail blows past it. Also checked: every output written (y
 * poisoned, compared by bit pattern — this file builds with -ffast-math), and
 * that cpu_x86 did not leave cpu_scalar's kernel bound.
 *
 * Shapes cover one block, an odd block count (the dot's tail), the SmolLM2
 * widths, n_out that is not a multiple of anything, and m = 1 (decode) up to
 * a full 64-row prefill chunk; m = 2, 3 and 7 leave the 4-token tiles a
 * remainder of 2 and 3 tokens.
 *
 * M>1 has two kernels: the AVX2 one and, where the ISA gate allows it, the
 * AVX-512 VNNI register tiles. The gate is decided once per process, so the
 * whole matrix runs twice: in a child with GEIST_FORCE_ISA=avx2, then with
 * the default dispatch. On a host without VNNI both runs take the AVX2
 * kernel; the CI SDE leg (-spr) covers the tiles there.
 */
#define _POSIX_C_SOURCE 200809L /* fork, setenv, waitpid */

#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#if !defined(GEIST_BACKEND_CPU_X86) || !defined(GEIST_BACKEND_CPU_SCALAR)
int main(void) {
    printf("SKIP: needs cpu_x86 and cpu_scalar in this build\n");
    return GEIST_TEST_SKIP;
}
#else

#define GEIST_INTERNAL_BACKEND_LAYER
#include "src/backends/cpu_x86/kernel_w4a8.h" /* w4a8_dispatcher_tier: which M>1 kernel */

#include "heap.h"
#include "quant.h"

static const size_t N_INS[]  = {32, 96, 960, 2560};
static const size_t N_OUTS[] = {7, 64};
static const size_t MS[]     = {1, 2, 3, 7, 16, 64};
constexpr size_t    M_MAX    = 64;

/* Block scales exactly representable in fp16 (so the test knows d_w). */
static const uint16_t D_BITS[] = {0x1C00, 0x1800, 0x2000, 0x1E00}; /* 2^-8, 2^-9, 2^-7, 1.5*2^-8 */
static const float    D_VALS[] = {0.00390625f, 0.001953125f, 0.0078125f, 0.005859375f};

static uint32_t g_rng = 0xA5A5F00Du;
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
    const size_t nb = n_in / Q8_0_BLOCK_ELEMS;
    for (size_t i = 0; i < m; i++) {
        const float *xi = x + i * n_in;
        for (size_t j = 0; j < n_out; j++) {
            uint32_t bits;
            memcpy(&bits, &y[i * n_out + j], sizeof bits);
            if (bits == poison_bits) {
                r->unwritten++;
                continue;
            }
            double bound = 0.0, mag = 0.0;
            for (size_t b = 0; b < nb; b++) {
                const uint8_t *blk = raw + (j * nb + b) * Q8_0_BLOCK_BYTES;
                uint16_t       d_bits;
                memcpy(&d_bits, blk, sizeof d_bits);
                float dw = 0.0f;
                for (size_t k = 0; k < sizeof D_BITS / sizeof *D_BITS; k++) {
                    if (D_BITS[k] == d_bits) {
                        dw = D_VALS[k];
                    }
                }
                float amax = 0.0f;
                for (size_t k = 0; k < Q8_0_BLOCK_ELEMS; k++) {
                    amax = fmaxf(amax, fabsf(xi[b * Q8_0_BLOCK_ELEMS + k]));
                }
                const double half_dx = (double) amax / 127.0 / 2.0;
                for (size_t k = 0; k < Q8_0_BLOCK_ELEMS; k++) {
                    const double wv = (double) dw * (double) (int8_t) blk[2 + k];
                    bound += fabs(wv) * half_dx;
                    mag += fabs(wv * (double) xi[b * Q8_0_BLOCK_ELEMS + k]);
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
    const size_t nb     = n_in / Q8_0_BLOCK_ELEMS;
    const size_t nbytes = n_out * nb * Q8_0_BLOCK_BYTES;
    uint8_t     *raw    = heap_alloc_array_aligned(uint8_t, nbytes);
    if (raw == nullptr) {
        fprintf(stderr, "ERROR: weight allocation failed\n");
        return 1;
    }
    for (size_t b = 0; b < n_out * nb; b++) {
        uint8_t       *blk = raw + b * Q8_0_BLOCK_BYTES;
        const uint16_t d   = D_BITS[next_u32() % (sizeof D_BITS / sizeof *D_BITS)];
        memcpy(blk, &d, sizeof d);
        for (size_t k = 0; k < Q8_0_BLOCK_ELEMS; k++) {
            blk[2 + k] = (uint8_t) next_u32(); /* full int8 range, -128 included */
        }
    }
    struct geist_weight w_ref = {.raw        = raw,
                                 .raw_nbytes = nbytes,
                                 .n_in       = (int32_t) n_in,
                                 .n_out      = (int32_t) n_out,
                                 .dtype      = GEIST_DTYPE_Q8_0};
    struct geist_weight w_x86 = w_ref;
    int                 fails = 0;
    if (be_ref->desc->vtbl->resolve_weight(be_ref, &w_ref) != GEIST_OK ||
        be_x86->desc->vtbl->resolve_weight(be_x86, &w_x86) != GEIST_OK) {
        fprintf(stderr, "FAIL: n_in=%zu n_out=%zu: resolve_weight refused\n", n_in, n_out);
        fails = 1;
        goto out;
    }
    if (w_x86.linear_m1 == w_ref.linear_m1 || w_x86.linear_mN == w_ref.linear_mN) {
        fprintf(stderr, "FAIL: cpu_x86 left cpu_scalar's Q8_0 kernel bound\n");
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
out:
    if ((w_x86.flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w_x86.aux_fp32 != nullptr) {
        void *aux = (void *) (uintptr_t) w_x86.aux_fp32;
        safe_free(&aux);
    }
    void *p = raw;
    safe_free(&p);
    return fails;
}

/* The whole matrix under the ISA gate this process was started with. */
static int run_all(void) {
    printf("ISA tier %s (GEIST_FORCE_ISA=%s)\n",
           w4a8_isa_name(w4a8_dispatcher_tier()),
           getenv("GEIST_FORCE_ISA") != nullptr ? getenv("GEIST_FORCE_ISA") : "<unset>");
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
        for (size_t k = 0; k < Q8_0_BLOCK_ELEMS; k++) {
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
    return GEIST_TEST_PASS;
}

int main(void) {
    /* Nothing has probed the ISA yet: the child's clamp takes effect. */
    fflush(stdout);
    const pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return GEIST_TEST_ERROR;
    }
    if (pid == 0) {
        setenv("GEIST_FORCE_ISA", "avx2", 1);
        const int rc = run_all();
        fflush(stdout);
        _exit(rc);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) {
        perror("waitpid");
        return GEIST_TEST_ERROR;
    }
    const int forced = WIFEXITED(status) ? WEXITSTATUS(status) : GEIST_TEST_ERROR;
    const int native = run_all();
    if (forced == GEIST_TEST_SKIP && native == GEIST_TEST_SKIP) {
        return GEIST_TEST_SKIP;
    }
    if (forced == GEIST_TEST_ERROR || native == GEIST_TEST_ERROR) {
        return GEIST_TEST_ERROR;
    }
    if (forced != GEIST_TEST_PASS || native != GEIST_TEST_PASS) {
        fprintf(stderr, "FAIL: forced-avx2 run %d, default run %d\n", forced, native);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: cpu_x86 Q8_0 within the activation-rounding bound on every shape, "
           "both M>1 kernels\n");
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_X86 && GEIST_BACKEND_CPU_SCALAR */
