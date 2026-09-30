/*
 * test_x86_large_k_unit — cpu_x86 Q4_K / Q6_K linears at K > 16384.
 *
 * The decode GEMVs quantized the activation row into a fixed 64-entry stack
 * array of 256-element super-blocks (K = 16384) and returned early for any
 * longer row, leaving y unwritten; the callers have no fallback. ffn_down's K
 * is the FFN width, so every model wider than 16384 (Qwen3-14B 17408,
 * Gemma-3-27B 21504, Llama-3-70B 28672) decoded from stale scratch on x86.
 *
 * Every cpu_x86 Q4_K / Q6_K path is driven on both sides of the old limit —
 * m = 1 decode, m % 4 != 0 (per-row), m < 16 (AVX2 tiles), m % 16 == 0 (the
 * AVX-512 panel where the host has it) — against the cpu_scalar reference.
 * y is poisoned before each call and compared by bit pattern, so an element
 * the kernel never wrote fails even when stale bytes happen to look sane
 * (bits, not `!=`: this file builds with -ffast-math, see
 * test_x86_kernel_no_alloc_unit.c).
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

constexpr size_t   N_OUT     = 32; /* multiple of 16: every tile path applies */
constexpr size_t   M_MAX     = 32;
constexpr uint16_t F16_0_004 = 0x1C19; /* fp16(0.004): super-block d */
constexpr uint16_t F16_0_002 = 0x1819; /* fp16(0.002): Q4_K dmin */
/* The activation is int8-quantized (one scale per row segment for Q4_K, per
 * super-block for Q6_K), so x86 and the fp32 reference differ by
 * quantization noise: max |dy| / rms(y_ref) measures 0.8-1.5% on every path
 * at every K here. 4% leaves headroom and still fails anything that computed
 * the wrong thing, not merely rounded it. */
constexpr double TOL_REL = 0.04;

static const size_t KS[] = {16384, 16640, 17408, 28672};
static const size_t MS[] = {1, 3, 4, 8, 16, 17, 32};

struct fmt {
    const char *name;
    uint16_t    dtype;
    size_t      block_bytes;
};

static const struct fmt FMTS[] = {
        {"Q4_K", (uint16_t) GEIST_DTYPE_Q4_K, Q4_K_BLOCK_BYTES},
        {"Q6_K", (uint16_t) GEIST_DTYPE_Q6_K, Q6_K_BLOCK_BYTES},
};

static uint32_t g_rng = 0x2545F491u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static void put_f16(uint8_t *p, uint16_t h) {
    p[0] = (uint8_t) (h & 0xFFu);
    p[1] = (uint8_t) (h >> 8);
}

/* Random quants and sub-scales, fixed small valid super-block scales. */
static uint8_t *make_blocks(const struct fmt *f, size_t k) {
    const size_t n_blocks = k / 256 * N_OUT;
    uint8_t     *raw      = heap_alloc_array_aligned(uint8_t, n_blocks * f->block_bytes);
    if (raw == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < n_blocks * f->block_bytes; i++) {
        raw[i] = (uint8_t) next_u32();
    }
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = raw + b * f->block_bytes;
        if (f->dtype == GEIST_DTYPE_Q4_K) {
            put_f16(blk + 0, F16_0_004); /* d */
            put_f16(blk + 2, F16_0_002); /* dmin */
        } else {
            put_f16(blk + 208, F16_0_004); /* d, at the Q6_K block tail */
        }
    }
    return raw;
}

static void free_aux(struct geist_weight *w) {
    if ((w->flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w->aux_fp32 != nullptr) {
        void *aux = (void *) (uintptr_t) w->aux_fp32;
        safe_free(&aux);
    }
}

static void run_linear(const struct geist_weight *w,
                       struct geist_backend      *be,
                       size_t                     m,
                       const float               *x,
                       float                     *y) {
    if (m == 1) {
        w->linear_m1(x, w, be, y);
    } else {
        w->linear_mN(m, x, w, be, y);
    }
}

/* Returns the number of failing (m) shapes for this (format, K). */
static int check_one(const struct fmt     *f,
                     size_t                k,
                     struct geist_backend *be_ref,
                     struct geist_backend *be_x86,
                     const float          *x,
                     float                *y_ref,
                     float                *y) {
    uint8_t *raw = make_blocks(f, k);
    if (raw == nullptr) {
        fprintf(stderr, "ERROR: weight allocation failed\n");
        return 1;
    }
    const size_t        nbytes = k / 256 * N_OUT * f->block_bytes;
    struct geist_weight w_ref  = {.raw        = raw,
                                  .raw_nbytes = nbytes,
                                  .n_in       = (int32_t) k,
                                  .n_out      = (int32_t) N_OUT,
                                  .dtype      = f->dtype};
    struct geist_weight w_x86  = w_ref;
    int                 fails  = 0;
    if (be_ref->desc->vtbl->resolve_weight(be_ref, &w_ref) != GEIST_OK ||
        be_x86->desc->vtbl->resolve_weight(be_x86, &w_x86) != GEIST_OK) {
        fprintf(stderr, "FAIL: %s K=%zu: resolve_weight refused the shape\n", f->name, k);
        fails = 1;
        goto out;
    }

    constexpr float POISON = -7.5e30f;
    uint32_t        poison_bits;
    memcpy(&poison_bits, &POISON, sizeof poison_bits);
    for (size_t mi = 0; mi < sizeof MS / sizeof *MS; mi++) {
        const size_t m = MS[mi];
        run_linear(&w_ref, be_ref, m, x, y_ref);
        for (size_t i = 0; i < m * N_OUT; i++) {
            y[i] = POISON;
        }
        run_linear(&w_x86, be_x86, m, x, y);

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
        const bool   ok  = unwritten == 0 && rel <= TOL_REL;
        if (!ok) {
            fprintf(stderr,
                    "FAIL: %s K=%zu m=%zu: %zu of %zu outputs unwritten, max|dy|/rms %.4f "
                    "(tol %.2f)\n",
                    f->name,
                    k,
                    m,
                    unwritten,
                    m * N_OUT,
                    rel,
                    TOL_REL);
            fails++;
        } else {
            printf("  %s K=%-5zu m=%-2zu max|dy|/rms %.4f\n", f->name, k, m, rel);
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

    const size_t k_max = KS[sizeof KS / sizeof *KS - 1];
    float       *x     = heap_alloc_array_aligned(float, M_MAX *k_max);
    float       *y_ref = heap_alloc_array_aligned(float, M_MAX *N_OUT);
    float       *y     = heap_alloc_array_aligned(float, M_MAX *N_OUT);
    if (x == nullptr || y_ref == nullptr || y == nullptr) {
        fprintf(stderr, "ERROR: activation allocation failed\n");
        return GEIST_TEST_ERROR;
    }

    int fails = 0;
    for (size_t ki = 0; ki < sizeof KS / sizeof *KS; ki++) {
        const size_t k = KS[ki];
        /* Row-major [M_MAX, k]; uniform in [-1, 1). */
        for (size_t i = 0; i < M_MAX * k; i++) {
            x[i] = (float) (next_u32() >> 8) * (2.0f / 16777216.0f) - 1.0f;
        }
        for (size_t fi = 0; fi < sizeof FMTS / sizeof *FMTS; fi++) {
            fails += check_one(&FMTS[fi], k, be_ref, be_x86, x, y_ref, y);
        }
    }

    safe_free((void **) &x);
    safe_free((void **) &y_ref);
    safe_free((void **) &y);
    geist_backend_destroy(be_ref);
    geist_backend_destroy(be_x86);

    if (fails != 0) {
        fprintf(stderr, "FAIL: %d shape(s) wrong or unwritten\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: Q4_K and Q6_K correct and fully written at K up to %zu on every path\n", k_max);
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_X86 && GEIST_BACKEND_CPU_SCALAR */
