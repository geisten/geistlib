/*
 * src/backends/cpu_neon/calibrate.c — measurement sondes for the
 * calibration driver.
 *
 * Layer: BACKEND (cpu_neon).
 *
 * FIDELITY CONTRACT (learned the hard way): sondes time the
 * RESOLVER-INSTALLED mN path on a throwaway backend instance with the
 * policy variant forced — never hand-picked quant.h kernels. The first
 * draft timed raw kernels and confidently mis-calibrated
 * qk_sgemm_threshold to "never" while the real Q4_K path (gemma pp256
 * A/B) showed the SGEMM side 25 % ahead: real paths are composed by
 * the resolver (tile variants, layout installs), so only resolved
 * pointers measure the truth a model would run.
 *
 * Panels are synthetic (pseudo-random payload, d = fp16 1.0, bytes 2-3
 * zeroed so dequant stays finite); timing per variant is warmup + the
 * MINIMUM of the timed runs — the noise-resistant estimator. The
 * driver adds median-of-three + a disagreement note on top.
 *
 * Deliberately uncalibrated: x8 switches (RAM tradeoff = user policy),
 * q4k/q6k_sgemm_prefill as bools (their Pi seed is a QUALITY gate, not
 * a perf judgment — only the m-threshold is measured), tq2_0_native_mn
 * (native entry is resolver-internal; needs a refactor first).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"
#include "kernel_catalog.h"

#include "heap.h"
#include "quant.h"

#include <geist_backend.h>
#include <geist_weight.h>

#include <stdint.h>
#include <string.h>
#include <time.h>

static constexpr size_t CAL_N_OUT = 2048; /* FFN-class panel width */
static constexpr size_t CAL_N_IN  = 4096;
static constexpr size_t CAL_M     = 64; /* chunk-class m for the bool A/Bs */

static uint64_t cal_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static void cal_fill_blocks(uint8_t *dst, size_t rows, size_t block_bytes, size_t blocks_per_row) {
    uint32_t lcg = 0x12345u;
    for (size_t r = 0; r < rows; r++) {
        for (size_t b = 0; b < blocks_per_row; b++) {
            uint8_t *blk = dst + (r * blocks_per_row + b) * block_bytes;
            for (size_t i = 0; i < block_bytes; i++) {
                lcg    = lcg * 1664525u + 1013904223u;
                blk[i] = (uint8_t) (lcg >> 24);
            }
            blk[0] = 0x00; /* d = fp16 1.0 */
            blk[1] = 0x3C;
            blk[2] = 0x00; /* dmin / scales_h / payload — keep finite */
            blk[3] = 0x00;
        }
    }
}

/* Policy mutator selecting one variant of the A/B. */
typedef void (*cal_policy_mut)(struct cpu_neon_kernel_policy *p);

/* Time the resolver-installed mN path for (panel, policy variant): a
 * throwaway backend gets the mutated policy (marked calibrated so no
 * overlay reruns), resolves the synthetic weight, and the installed
 * w.linear_mN is timed. Aux-layout installs are disabled by the shared
 * mutations (they are m1/RAM levers, not mN paths) so variants stay
 * comparable; a heap aux is still freed defensively. */
static enum geist_status cal_time_resolved(const uint8_t   *panel,
                                           size_t           panel_bytes,
                                           enum geist_dtype dtype,
                                           cal_policy_mut   mut,
                                           const float     *x,
                                           float           *y,
                                           size_t           m,
                                           uint64_t         budget_ns,
                                           uint64_t        *out_ns) {
    struct geist_backend *tb = nullptr;
    if (geist_backend_create("cpu_neon", nullptr, nullptr, &tb) != GEIST_OK) {
        return GEIST_E_BACKEND;
    }
    struct cpu_neon_state *bst = (struct cpu_neon_state *) tb->state;
    mut(&bst->policy);
    bst->policy.q4k_predecode = false;
    bst->policy.q6k_x8_gemv   = false;
    bst->policy.q4_0_x8_gemv  = false;
    bst->policy_calibrated    = true;

    struct geist_weight w = {
            .raw        = panel,
            .raw_nbytes = panel_bytes,
            .n_in       = (int64_t) CAL_N_IN,
            .n_out      = (int64_t) CAL_N_OUT,
            .dtype      = dtype,
    };
    enum geist_status s = tb->desc->vtbl->resolve_weight(tb, &w);
    if (s != GEIST_OK || w.linear_mN == nullptr) {
        geist_backend_destroy(tb);
        return s != GEIST_OK ? s : GEIST_E_UNSUPPORTED;
    }
    w.linear_mN(m, x, &w, tb, y); /* warmup: page-in + pool spin-up */
    uint64_t       best  = UINT64_MAX;
    const uint64_t start = cal_now_ns();
    for (int i = 0; i < 5; i++) {
        const uint64_t t0 = cal_now_ns();
        w.linear_mN(m, x, &w, tb, y);
        const uint64_t dt = cal_now_ns() - t0;
        if (dt < best) {
            best = dt;
        }
        if (cal_now_ns() - start > budget_ns) {
            break;
        }
    }
    if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w.aux_fp32 != nullptr) {
        void *aux = (void *) (uintptr_t) w.aux_fp32;
        safe_free(&aux);
    }
    geist_backend_destroy(tb);
    *out_ns = best;
    return GEIST_OK;
}

struct cal_ab {
    enum geist_dtype dtype;
    size_t           block_bytes;
    size_t           block_elems;
    cal_policy_mut   variant_a; /* reported as value 1 when faster */
    cal_policy_mut   variant_b; /* reported as value 0 */
};

static enum geist_status
cal_measure_ab(const struct cal_ab *ab, uint64_t budget_ns, size_t m, int64_t *out_value) {
    const size_t blocks_per_row = CAL_N_IN / ab->block_elems;
    const size_t panel_bytes    = CAL_N_OUT * blocks_per_row * ab->block_bytes;
    uint8_t     *wbuf           = heap_alloc_aligned(panel_bytes, 64);
    float       *x              = heap_alloc_aligned(m * CAL_N_IN * sizeof(float), 64);
    float       *y              = heap_alloc_aligned(m * CAL_N_OUT * sizeof(float), 64);
    if (wbuf == nullptr || x == nullptr || y == nullptr) {
        safe_free((void **) &wbuf);
        safe_free((void **) &x);
        safe_free((void **) &y);
        return GEIST_E_OOM;
    }
    cal_fill_blocks(wbuf, CAL_N_OUT, ab->block_bytes, blocks_per_row);
    for (size_t i = 0; i < m * CAL_N_IN; i++) {
        x[i] = (float) ((int) (i % 19u) - 9) * 0.25f;
    }
    uint64_t          t_a  = 0;
    uint64_t          t_b  = 0;
    const uint64_t    half = budget_ns / 2u;
    enum geist_status s =
            cal_time_resolved(wbuf, panel_bytes, ab->dtype, ab->variant_a, x, y, m, half, &t_a);
    if (s == GEIST_OK) {
        s = cal_time_resolved(wbuf, panel_bytes, ab->dtype, ab->variant_b, x, y, m, half, &t_b);
    }
    safe_free((void **) &wbuf);
    safe_free((void **) &x);
    safe_free((void **) &y);
    if (s != GEIST_OK) {
        return s;
    }
    *out_value = t_a <= t_b ? 1 : 0;
    return GEIST_OK;
}

/* ---- variant mutators -------------------------------------------------- */
static void mut_q5k_on(struct cpu_neon_kernel_policy *p) {
    p->q5k_native_mn = true;
}
static void mut_q5k_off(struct cpu_neon_kernel_policy *p) {
    p->q5k_native_mn = false;
}
static void mut_q8_0_on(struct cpu_neon_kernel_policy *p) {
    p->q8_0_native_mn = true;
}
static void mut_q8_0_off(struct cpu_neon_kernel_policy *p) {
    p->q8_0_native_mn = false;
}
static void mut_q4_01_on(struct cpu_neon_kernel_policy *p) {
    p->q4_01_native_mn = true;
}
static void mut_q4_01_off(struct cpu_neon_kernel_policy *p) {
    p->q4_01_native_mn = false;
}
static void mut_iq4xs_on(struct cpu_neon_kernel_policy *p) {
    p->iq4xs_native_mn = true;
}
static void mut_iq4xs_off(struct cpu_neon_kernel_policy *p) {
    p->iq4xs_native_mn = false;
}
/* threshold sweep: force the sgemm side wholesale on/off */
static void mut_q4k_sgemm_always(struct cpu_neon_kernel_policy *p) {
    p->q4k_sgemm_prefill  = true;
    p->qk_sgemm_threshold = 1;
}
static void mut_q4k_sgemm_never(struct cpu_neon_kernel_policy *p) {
    p->q4k_sgemm_prefill = false;
}

static enum geist_status
cal_measure_q5k_native_mn(struct geist_backend *be, uint64_t budget_ns, int64_t *out_value) {
    (void) be;
    const struct cal_ab ab = {
            GEIST_DTYPE_Q5_K, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS, mut_q5k_on, mut_q5k_off};
    return cal_measure_ab(&ab, budget_ns, CAL_M, out_value);
}

static enum geist_status
cal_measure_q8_0_native_mn(struct geist_backend *be, uint64_t budget_ns, int64_t *out_value) {
    (void) be;
    const struct cal_ab ab = {GEIST_DTYPE_Q8_0, 34u, 32u, mut_q8_0_on, mut_q8_0_off};
    return cal_measure_ab(&ab, budget_ns, CAL_M, out_value);
}

static enum geist_status
cal_measure_q4_01_native_mn(struct geist_backend *be, uint64_t budget_ns, int64_t *out_value) {
    (void) be;
    const struct cal_ab ab = {GEIST_DTYPE_Q4_0, 18u, 32u, mut_q4_01_on, mut_q4_01_off};
    return cal_measure_ab(&ab, budget_ns, CAL_M, out_value);
}

static enum geist_status
cal_measure_iq4xs_native_mn(struct geist_backend *be, uint64_t budget_ns, int64_t *out_value) {
    (void) be;
    const struct cal_ab ab = {GEIST_DTYPE_IQ4_XS,
                              IQ4_XS_BLOCK_BYTES,
                              IQ4_XS_BLOCK_ELEMS,
                              mut_iq4xs_on,
                              mut_iq4xs_off};
    return cal_measure_ab(&ab, budget_ns, CAL_M, out_value);
}

static enum geist_status
cal_measure_qk_sgemm_threshold(struct geist_backend *be, uint64_t budget_ns, int64_t *out_value) {
    (void) be;
    static const size_t M_SWEEP[] = {8, 16, 32, 64, 128};
    enum { N_SWEEP = 5 };
    const struct cal_ab ab = {GEIST_DTYPE_Q4_K,
                              Q4_K_BLOCK_BYTES,
                              Q4_K_BLOCK_ELEMS,
                              mut_q4k_sgemm_always,
                              mut_q4k_sgemm_never};
    /* Smallest m where the sgemm side wins; above the sweep = "never"
     * on this machine (one past the largest point). */
    int64_t threshold = (int64_t) (M_SWEEP[N_SWEEP - 1] * 2u);
    for (size_t i = 0; i < N_SWEEP; i++) {
        int64_t                 sgemm_wins = 0;
        const enum geist_status s =
                cal_measure_ab(&ab, budget_ns / N_SWEEP, M_SWEEP[i], &sgemm_wins);
        if (s != GEIST_OK) {
            return s;
        }
        if (sgemm_wins == 1) {
            threshold = (int64_t) M_SWEEP[i];
            break;
        }
    }
    *out_value = threshold;
    return GEIST_OK;
}

static const struct geist_tunable cpu_neon_tunables_v1[] = {
        {.name = "q5k_native_mn", .kind = GEIST_TUNABLE_BOOL, .measure = cal_measure_q5k_native_mn},
        {.name    = "qk_sgemm_threshold",
         .kind    = GEIST_TUNABLE_SIZE,
         .measure = cal_measure_qk_sgemm_threshold},
        {.name    = "q8_0_native_mn",
         .kind    = GEIST_TUNABLE_BOOL,
         .measure = cal_measure_q8_0_native_mn},
        {.name    = "q4_01_native_mn",
         .kind    = GEIST_TUNABLE_BOOL,
         .measure = cal_measure_q4_01_native_mn},
        {.name    = "iq4xs_native_mn",
         .kind    = GEIST_TUNABLE_BOOL,
         .measure = cal_measure_iq4xs_native_mn},
};

const struct geist_tunable *cpu_neon_tunables(size_t *out_count) {
    *out_count = sizeof(cpu_neon_tunables_v1) / sizeof(cpu_neon_tunables_v1[0]);
    return cpu_neon_tunables_v1;
}
