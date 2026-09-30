/*
 * test_x86_q6k_layout_unit — cpu_x86 Q6_K on each of its prefill layouts.
 *
 * A Q6_K weight keeps one W8A8 predecode for prefill: the W8x16 interleave
 * (n_out % 16 == 0 on a VNNI host), the W8x8 interleave (n_out % 8 == 0),
 * or row-major (otherwise, and on hosts without VNNI); the interleaves are
 * built one row group at a time. Decode reads the native Q6_K bytes. For
 * n_out = 32, 40 and 36 (one per layout) and m from decode to past a
 * prefill tile, every output must be written (y poisoned, compared by bit
 * pattern — this file builds with -ffast-math) and match the cpu_scalar
 * reference within the int8-activation tolerance. A group repacked into the
 * wrong rows, or a layout read as another, fails by far more than that.
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

constexpr size_t   M_MAX     = 33;
constexpr uint16_t F16_0_004 = 0x1C19; /* fp16(0.004): super-block d */
/* int8 activations vs the fp32 reference: ~1 % measured on every Q6_K path
 * (test_x86_large_k_unit); 4 % still fails a wrong row. */
constexpr double TOL_REL = 0.04;

static const size_t KS[]    = {256, 2048};
static const size_t NOUTS[] = {32, 36, 40}; /* W8x16, row-major, W8x8 */
static const size_t MS[]    = {1, 2, 5, 16, 17, 33};

static uint32_t g_rng = 0x3C6EF372u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
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

/* Returns the number of failing m for this (K, N). */
static int check_shape(size_t                k,
                       size_t                n_out,
                       struct geist_backend *be_ref,
                       struct geist_backend *be_x86,
                       const float          *x,
                       float                *y_ref,
                       float                *y) {
    const size_t n_blocks = k / 256 * n_out;
    uint8_t     *raw      = heap_alloc_array_aligned(uint8_t, n_blocks *Q6_K_BLOCK_BYTES);
    if (raw == nullptr) {
        fprintf(stderr, "ERROR: weight allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < n_blocks * Q6_K_BLOCK_BYTES; i++) {
        raw[i] = (uint8_t) next_u32();
    }
    for (size_t i = 0; i < n_blocks; i++) {
        uint8_t *d = raw + i * Q6_K_BLOCK_BYTES + 208; /* d sits at the block tail */
        d[0]       = (uint8_t) (F16_0_004 & 0xFFu);
        d[1]       = (uint8_t) (F16_0_004 >> 8);
    }
    struct geist_weight w_ref = {.raw        = raw,
                                 .raw_nbytes = n_blocks * Q6_K_BLOCK_BYTES,
                                 .n_in       = (int32_t) k,
                                 .n_out      = (int32_t) n_out,
                                 .dtype      = GEIST_DTYPE_Q6_K};
    struct geist_weight w_x86 = w_ref;
    int                 fails = 0;
    if (be_ref->desc->vtbl->resolve_weight(be_ref, &w_ref) != GEIST_OK ||
        be_x86->desc->vtbl->resolve_weight(be_x86, &w_x86) != GEIST_OK) {
        fprintf(stderr, "FAIL: K=%zu N=%zu: resolve_weight refused the shape\n", k, n_out);
        fails = 1;
        goto out;
    }

    constexpr float POISON = -7.5e30f;
    uint32_t        poison_bits;
    memcpy(&poison_bits, &POISON, sizeof poison_bits);
    double worst = 0.0;
    for (size_t mi = 0; mi < sizeof MS / sizeof *MS; mi++) {
        const size_t m = MS[mi];
        run_linear(&w_ref, be_ref, m, x, y_ref);
        for (size_t i = 0; i < m * n_out; i++) {
            y[i] = POISON;
        }
        run_linear(&w_x86, be_x86, m, x, y);
        size_t unwritten = 0;
        double max_abs = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < m * n_out; i++) {
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
        const double rms = sqrt(sum_sq / (double) (m * n_out));
        const double rel = rms > 0.0 ? max_abs / rms : max_abs;
        worst            = rel > worst ? rel : worst;
        if (unwritten != 0 || !(rel <= TOL_REL)) {
            fprintf(stderr,
                    "FAIL: K=%zu N=%zu m=%zu: %zu of %zu outputs unwritten, max|dy|/rms %.4f\n",
                    k,
                    n_out,
                    m,
                    unwritten,
                    m * n_out,
                    rel);
            fails++;
        }
    }
    printf("  K=%-4zu N=%zu worst max|dy|/rms %.4f\n", k, n_out, worst);
out:
    if ((w_x86.flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w_x86.aux_fp32 != nullptr) {
        void *aux = (void *) (uintptr_t) w_x86.aux_fp32;
        safe_free(&aux);
    }
    if ((w_ref.flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w_ref.aux_fp32 != nullptr) {
        void *aux = (void *) (uintptr_t) w_ref.aux_fp32;
        safe_free(&aux);
    }
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
    size_t k_max = 0, n_max = 0;
    for (size_t i = 0; i < sizeof KS / sizeof *KS; i++) {
        k_max = KS[i] > k_max ? KS[i] : k_max;
    }
    for (size_t i = 0; i < sizeof NOUTS / sizeof *NOUTS; i++) {
        n_max = NOUTS[i] > n_max ? NOUTS[i] : n_max;
    }
    float *x     = heap_alloc_array_aligned(float, M_MAX *k_max);
    float *y_ref = heap_alloc_array_aligned(float, M_MAX *n_max);
    float *y     = heap_alloc_array_aligned(float, M_MAX *n_max);
    int    fails = 0;
    if (x == nullptr || y_ref == nullptr || y == nullptr) {
        fprintf(stderr, "ERROR: buffer allocation failed\n");
        fails = 1;
        goto done;
    }
    for (size_t i = 0; i < M_MAX * k_max; i++) {
        x[i] = (float) (next_u32() >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }
    for (size_t ki = 0; ki < sizeof KS / sizeof *KS; ki++) {
        for (size_t ni = 0; ni < sizeof NOUTS / sizeof *NOUTS; ni++) {
            fails += check_shape(KS[ki], NOUTS[ni], be_ref, be_x86, x, y_ref, y);
        }
    }
done:
    safe_free((void **) &x);
    safe_free((void **) &y_ref);
    safe_free((void **) &y);
    geist_backend_destroy(be_ref);
    geist_backend_destroy(be_x86);
    if (fails != 0) {
        fprintf(stderr, "FAIL: %d shape(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: cpu_x86 Q6_K matches cpu_scalar on the W8x16, W8x8 and row-major layouts\n");
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_X86 && GEIST_BACKEND_CPU_SCALAR */
