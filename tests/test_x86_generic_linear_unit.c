/*
 * test_x86_generic_linear_unit — cpu_x86 binds its own kernels for every
 * dtype without a native x86 one, and they match the cpu_scalar oracle.
 *
 * Those dtypes (Q8_0, Q4_0, Q4_1, Q3_K, Q5_K, the IQ formats, TQ2_0, PQ2_0,
 * BF16, F16 prefill) used to stay bound to cpu_scalar's own kernels — the
 * single-threaded, heap-allocating reference — on the default x86 backend.
 * Two checks per dtype:
 *
 *   1. cpu_x86's resolver did not leave cpu_scalar's function pointers in
 *      the weight (compared against what cpu_scalar's resolver installs);
 *   2. the outputs match cpu_scalar for m = 1 (decode) and m > 1 (prefill),
 *      every element written (poisoned y, compared by bit pattern — this
 *      file builds with -ffast-math, see test_x86_kernel_no_alloc_unit.c).
 *
 * Both paths dequantize the same bits with the same row decoder; only the
 * dot differs (fp32 x 32 lanes here, double in the oracle), so the bound is
 * float rounding, not quantization noise.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if !defined(GEIST_BACKEND_CPU_X86) || !defined(GEIST_BACKEND_CPU_SCALAR)
int main(void) {
    printf("SKIP: needs cpu_x86 and cpu_scalar in this build\n");
    return GEIST_TEST_SKIP;
}
#else

#include "heap.h"
#include "quant.h"

constexpr size_t N_IN  = 512; /* whole blocks for every format (256 / 128 / 32) */
constexpr size_t N_OUT = 40;
constexpr size_t M_MAX = 16;
/* fp32 vs double accumulation over 512 products: measured below 1e-6 of the
 * output rms. 1e-4 catches any real mistake (a wrong row, a dropped block)
 * by orders of magnitude. */
constexpr double TOL_REL = 1e-4;

constexpr uint16_t F16_0_004 = 0x1C19; /* fp16(0.004) */
constexpr uint16_t F16_0_002 = 0x1819; /* fp16(0.002) */

static const size_t MS[] = {1, 2, 5, 16};

/* Block layout facts the test needs: size, and where the fp16 scales sit
 * (random bytes there could decode to NaN/Inf and make parity meaningless). */
struct fmt {
    const char *name;
    uint16_t    dtype;
    size_t      block_elems;
    size_t      block_bytes;
    size_t      n_scales;
    size_t      scale_off[2];
    bool        m1_native; /* cpu_x86 has its own M=1 kernel: skip the m = 1 parity */
};

static const struct fmt FMTS[] = {
        {"Q4_0", GEIST_DTYPE_Q4_0, Q4_0_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, 1, {0, 0}, false},
        {"Q4_1", GEIST_DTYPE_Q4_1, Q4_1_BLOCK_ELEMS, Q4_1_BLOCK_BYTES, 2, {0, 2}, false},
        {"Q8_0", GEIST_DTYPE_Q8_0, Q8_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, 1, {0, 0}, false},
        {"Q3_K", GEIST_DTYPE_Q3_K, Q3_K_BLOCK_ELEMS, Q3_K_BLOCK_BYTES, 1, {108, 0}, false},
        {"Q5_K", GEIST_DTYPE_Q5_K, Q5_K_BLOCK_ELEMS, Q5_K_BLOCK_BYTES, 2, {0, 2}, false},
        {"IQ2_S", GEIST_DTYPE_IQ2_S, IQ2_S_BLOCK_ELEMS, IQ2_S_BLOCK_BYTES, 1, {0, 0}, false},
        {"IQ3_S", GEIST_DTYPE_IQ3_S, IQ3_S_BLOCK_ELEMS, IQ3_S_BLOCK_BYTES, 1, {0, 0}, false},
        {"IQ4_NL", GEIST_DTYPE_IQ4_NL, IQ4_NL_BLOCK_ELEMS, IQ4_NL_BLOCK_BYTES, 1, {0, 0}, false},
        {"IQ4_XS", GEIST_DTYPE_IQ4_XS, IQ4_XS_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, 1, {0, 0}, false},
        {"TQ2_0", GEIST_DTYPE_TQ2_0, TQ2_0_BLOCK_ELEMS, TQ2_0_BLOCK_BYTES, 1, {64, 0}, false},
        {"PQ2_0", GEIST_DTYPE_PQ2_0, PQ2_0_BLOCK_ELEMS, PQ2_0_BLOCK_BYTES, 1, {0, 0}, false},
        /* Dense halves: values, not blocks; built from floats below. */
        {"BF16", GEIST_DTYPE_BF16, 1, 2, 0, {0, 0}, false},
        {"F16", GEIST_DTYPE_F16, 1, 2, 0, {0, 0}, true},
};

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
static float next_unit(void) { /* uniform in [-1, 1) */
    return (float) (next_u32() >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

/* fp32 -> fp16 bits for the small, normal values used here (|v| < 1). */
static uint16_t to_f16(float v) {
    uint32_t b;
    memcpy(&b, &v, sizeof b);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t) ((b >> 23) & 0xFFu) - 127 + 15;
    if (exp <= 0) {
        return (uint16_t) sign; /* flush tiny values to signed zero */
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | ((b >> 13) & 0x3FFu));
}

static uint8_t *make_weight(const struct fmt *f, size_t *nbytes_out) {
    const size_t n_blocks = N_IN / f->block_elems * N_OUT;
    const size_t nbytes   = n_blocks * f->block_bytes;
    uint8_t     *raw      = heap_alloc_array_aligned(uint8_t, nbytes);
    if (raw == nullptr) {
        return nullptr;
    }
    if (f->dtype == GEIST_DTYPE_F16 || f->dtype == GEIST_DTYPE_BF16) {
        for (size_t i = 0; i < N_IN * N_OUT; i++) {
            const float v = 0.05f * next_unit();
            uint16_t    h;
            if (f->dtype == GEIST_DTYPE_F16) {
                h = to_f16(v);
            } else {
                uint32_t b;
                memcpy(&b, &v, sizeof b);
                h = (uint16_t) (b >> 16);
            }
            memcpy(raw + 2 * i, &h, sizeof h);
        }
    } else {
        for (size_t i = 0; i < nbytes; i++) {
            raw[i] = (uint8_t) next_u32();
        }
        for (size_t b = 0; b < n_blocks; b++) {
            for (size_t s = 0; s < f->n_scales; s++) {
                const uint16_t h = s == 0 ? F16_0_004 : F16_0_002;
                memcpy(raw + b * f->block_bytes + f->scale_off[s], &h, sizeof h);
            }
        }
    }
    *nbytes_out = nbytes;
    return raw;
}

static void free_aux(struct geist_weight *w) {
    if ((w->flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w->aux_fp32 != nullptr) {
        void *aux = (void *) (uintptr_t) w->aux_fp32;
        safe_free(&aux);
    }
}

static int check_fmt(const struct fmt     *f,
                     struct geist_backend *be_ref,
                     struct geist_backend *be_x86,
                     const float          *x,
                     float                *y_ref,
                     float                *y) {
    size_t   nbytes = 0;
    uint8_t *raw    = make_weight(f, &nbytes);
    if (raw == nullptr) {
        fprintf(stderr, "ERROR: weight allocation failed\n");
        return 1;
    }
    struct geist_weight w_ref = {.raw        = raw,
                                 .raw_nbytes = nbytes,
                                 .n_in       = (int32_t) N_IN,
                                 .n_out      = (int32_t) N_OUT,
                                 .dtype      = f->dtype};
    struct geist_weight w_x86 = w_ref;
    int                 fails = 0;
    if (be_ref->desc->vtbl->resolve_weight(be_ref, &w_ref) != GEIST_OK ||
        be_x86->desc->vtbl->resolve_weight(be_x86, &w_x86) != GEIST_OK) {
        fprintf(stderr, "FAIL: %s: resolve_weight refused a valid weight\n", f->name);
        fails = 1;
        goto out;
    }
    if (w_x86.linear_mN == w_ref.linear_mN ||
        (!f->m1_native && w_x86.linear_m1 == w_ref.linear_m1)) {
        fprintf(stderr, "FAIL: %s: cpu_x86 left cpu_scalar's reference kernel bound\n", f->name);
        fails = 1;
        goto out;
    }

    constexpr float POISON = -7.5e30f;
    uint32_t        poison_bits;
    memcpy(&poison_bits, &POISON, sizeof poison_bits);
    for (size_t mi = 0; mi < sizeof MS / sizeof *MS; mi++) {
        const size_t m = MS[mi];
        if (m == 1 && f->m1_native) {
            continue; /* a lossy native kernel (F16 -> Q8 lm_head); covered elsewhere */
        }
        if (m == 1) {
            w_ref.linear_m1(x, &w_ref, be_ref, y_ref);
        } else {
            w_ref.linear_mN(m, x, &w_ref, be_ref, y_ref);
        }
        for (size_t i = 0; i < m * N_OUT; i++) {
            y[i] = POISON;
        }
        if (m == 1) {
            w_x86.linear_m1(x, &w_x86, be_x86, y);
        } else {
            w_x86.linear_mN(m, x, &w_x86, be_x86, y);
        }
        size_t unwritten = 0;
        double max_abs = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < m * N_OUT; i++) {
            uint32_t bits;
            memcpy(&bits, &y[i], sizeof bits);
            if (bits == poison_bits) {
                unwritten++;
                continue;
            }
            const double d = fabs((double) y[i] - (double) y_ref[i]);
            max_abs        = d > max_abs ? d : max_abs;
            sum_sq += (double) y_ref[i] * (double) y_ref[i];
        }
        const double rms = sqrt(sum_sq / (double) (m * N_OUT));
        const double rel = rms > 0.0 ? max_abs / rms : max_abs;
        if (unwritten != 0 || !(rel <= TOL_REL)) {
            fprintf(stderr,
                    "FAIL: %s m=%zu: %zu unwritten, max|dy|/rms %.3g (tol %.0e)\n",
                    f->name,
                    m,
                    unwritten,
                    rel,
                    TOL_REL);
            fails++;
        } else {
            printf("  %-6s m=%-2zu max|dy|/rms %.2e\n", f->name, m, rel);
        }
    }
out:
    free_aux(&w_x86);
    free_aux(&w_ref);
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
    float *x     = heap_alloc_array_aligned(float, M_MAX *N_IN);
    float *y_ref = heap_alloc_array_aligned(float, M_MAX *N_OUT);
    float *y     = heap_alloc_array_aligned(float, M_MAX *N_OUT);
    if (x == nullptr || y_ref == nullptr || y == nullptr) {
        fprintf(stderr, "ERROR: activation allocation failed\n");
        return GEIST_TEST_ERROR;
    }
    for (size_t i = 0; i < M_MAX * N_IN; i++) {
        x[i] = next_unit();
    }

    int fails = 0;
    for (size_t fi = 0; fi < sizeof FMTS / sizeof *FMTS; fi++) {
        fails += check_fmt(&FMTS[fi], be_ref, be_x86, x, y_ref, y);
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
    printf("PASS: %zu dtypes on cpu_x86's own kernels, matching cpu_scalar\n",
           sizeof FMTS / sizeof *FMTS);
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_X86 && GEIST_BACKEND_CPU_SCALAR */
