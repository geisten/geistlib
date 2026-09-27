/*
 * test_x86_q4k_prefill_tail_unit — cpu_x86 Q4_K prefill for every chunk
 * length, the tail rows included.
 *
 * A prefill chunk holds (prompt length mod 64) tokens, so its m is rarely a
 * multiple of 16. The Q4_K M>1 path runs the AVX-512 16x16 panel on the
 * first m16 = m rounded down to 16 rows, the AVX2 GEMV on rows [m16, m4)
 * (m4 = m rounded down to 4), and the M=1 GEMV on the last m % 4 rows. For
 * every m from 1 to 64 (and N with and without a 16-wide panel, and one
 * that is not a multiple of 8, where the weight is kept as W4A8 and every
 * row takes the M=1 path):
 *
 *   - every output is written (y poisoned, compared by bit pattern — this
 *     file builds with -ffast-math) and within the int8-activation tolerance
 *     of the cpu_scalar reference;
 *   - rows [0, m16) are bit-identical to a call with exactly m16 rows, and
 *     rows [m4, m) to the M=1 path, row by row: a split that reads the wrong
 *     Q8_Kx4 group or writes the wrong output row cannot pass.
 *
 * On a host without AVX-512 the first property still holds for every row
 * and the second for the tail; the panel rows then come from the AVX2 GEMV
 * on both sides.
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

constexpr size_t   M_MAX     = 64;     /* the prefill chunk */
constexpr uint16_t F16_0_004 = 0x1C19; /* fp16(0.004): super-block d */
constexpr uint16_t F16_0_002 = 0x1819; /* fp16(0.002): dmin */
/* int8 activations vs the fp32 reference: max |dy| / rms(y_ref) is ~1 %
 * (test_x86_large_k_unit measures 0.8-1.5 %); 4 % still fails a wrong row. */
constexpr double TOL_REL = 0.04;

static const size_t KS[] = {256, 2048};
/* 32: the panel; 36: n % 8 != 0, the W4A8 layout; 40: no 16-wide panel. */
static const size_t NOUTS[] = {32, 36, 40};

static uint32_t g_rng = 0x6C8E9CF5u;
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

static bool same_bits(size_t n, const float *a, const float *b) {
    return memcmp(a, b, n * sizeof *a) == 0;
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

struct bufs {
    const float *x;     /* [M_MAX, k] */
    float       *y_ref; /* [M_MAX, n_out] */
    float       *y;     /* [M_MAX, n_out] */
    float       *y_aux; /* [M_MAX, n_out] */
};

/* Returns the number of failing m for this (K, N). */
static int check_shape(size_t                k,
                       size_t                n_out,
                       struct geist_backend *be_ref,
                       struct geist_backend *be_x86,
                       const struct bufs    *b) {
    const size_t n_blocks = k / 256 * n_out;
    uint8_t     *raw      = heap_alloc_array_aligned(uint8_t, n_blocks *Q4_K_BLOCK_BYTES);
    if (raw == nullptr) {
        fprintf(stderr, "ERROR: weight allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < n_blocks * Q4_K_BLOCK_BYTES; i++) {
        raw[i] = (uint8_t) next_u32();
    }
    for (size_t i = 0; i < n_blocks; i++) {
        put_f16(raw + i * Q4_K_BLOCK_BYTES + 0, F16_0_004);
        put_f16(raw + i * Q4_K_BLOCK_BYTES + 2, F16_0_002);
    }
    struct geist_weight w_ref = {.raw        = raw,
                                 .raw_nbytes = n_blocks * Q4_K_BLOCK_BYTES,
                                 .n_in       = (int32_t) k,
                                 .n_out      = (int32_t) n_out,
                                 .dtype      = GEIST_DTYPE_Q4_K};
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
    for (size_t m = 1; m <= M_MAX; m++) {
        const size_t m16 = m / 16 * 16;
        const size_t m4  = m / 4 * 4;
        run_linear(&w_ref, be_ref, m, b->x, b->y_ref);
        for (size_t i = 0; i < m * n_out; i++) {
            b->y[i] = POISON;
        }
        run_linear(&w_x86, be_x86, m, b->x, b->y);

        size_t unwritten = 0;
        double max_abs = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < m * n_out; i++) {
            uint32_t bits;
            memcpy(&bits, &b->y[i], sizeof bits);
            if (bits == poison_bits) {
                unwritten++;
                continue;
            }
            const double d = fabs((double) b->y[i] - (double) b->y_ref[i]);
            max_abs        = d > max_abs ? d : max_abs;
            sum_sq += (double) b->y_ref[i] * (double) b->y_ref[i];
        }
        const double rms = sqrt(sum_sq / (double) (m * n_out));
        const double rel = rms > 0.0 ? max_abs / rms : max_abs;
        worst            = rel > worst ? rel : worst;
        bool ok          = unwritten == 0 && rel <= TOL_REL;
        if (!ok) {
            fprintf(stderr,
                    "FAIL: K=%zu N=%zu m=%zu: %zu of %zu outputs unwritten, max|dy|/rms %.4f\n",
                    k,
                    n_out,
                    m,
                    unwritten,
                    m * n_out,
                    rel);
        }

        /* Panel rows: the same bits as a call with exactly m16 rows. */
        if (ok && m16 > 0 && m16 < m) {
            run_linear(&w_x86, be_x86, m16, b->x, b->y_aux);
            if (!same_bits(m16 * n_out, b->y, b->y_aux)) {
                fprintf(stderr,
                        "FAIL: K=%zu N=%zu m=%zu: rows [0,%zu) differ from m=%zu\n",
                        k,
                        n_out,
                        m,
                        m16,
                        m16);
                ok = false;
            }
        }
        /* Tail rows: the same bits as the M=1 path on that row. */
        for (size_t r = m4; ok && m > 1 && r < m; r++) {
            w_x86.linear_m1(b->x + r * k, &w_x86, be_x86, b->y_aux);
            if (!same_bits(n_out, b->y + r * n_out, b->y_aux)) {
                fprintf(stderr,
                        "FAIL: K=%zu N=%zu m=%zu: tail row %zu differs from M=1\n",
                        k,
                        n_out,
                        m,
                        r);
                ok = false;
            }
        }
        fails += ok ? 0 : 1;
    }
    printf("  K=%-4zu N=%zu m=1..%zu worst max|dy|/rms %.4f\n", k, n_out, M_MAX, worst);
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
    float *y_aux = heap_alloc_array_aligned(float, M_MAX *n_max);
    int    fails = 0;
    if (x == nullptr || y_ref == nullptr || y == nullptr || y_aux == nullptr) {
        fprintf(stderr, "ERROR: buffer allocation failed\n");
        fails = 1;
        goto done;
    }
    for (size_t i = 0; i < M_MAX * k_max; i++) {
        x[i] = (float) (next_u32() >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }
    const struct bufs b = {.x = x, .y_ref = y_ref, .y = y, .y_aux = y_aux};
    for (size_t ki = 0; ki < sizeof KS / sizeof *KS; ki++) {
        for (size_t ni = 0; ni < sizeof NOUTS / sizeof *NOUTS; ni++) {
            fails += check_shape(KS[ki], NOUTS[ni], be_ref, be_x86, &b);
        }
    }
done:
    safe_free((void **) &x);
    safe_free((void **) &y_ref);
    safe_free((void **) &y);
    safe_free((void **) &y_aux);
    geist_backend_destroy(be_ref);
    geist_backend_destroy(be_x86);
    if (fails != 0) {
        fprintf(stderr, "FAIL: %d shape(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: cpu_x86 Q4_K prefill writes and matches every row for m = 1..64\n");
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_X86 && GEIST_BACKEND_CPU_SCALAR */
