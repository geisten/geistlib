/*
 * src/backends/cpu_neon/calibrate.c — measurement sondes for the
 * calibration driver (PR 1: two exemplars).
 *
 * Layer: BACKEND (cpu_neon).
 *
 * Each sonde answers ONE policy question for THIS machine by timing the
 * real alternatives on a synthetic weight panel:
 *
 *   q5k_native_mn      BOOL  native W5A8 mN kernel vs dequant+SGEMM
 *   qk_sgemm_threshold SIZE  smallest m where dequant+SGEMM beats the
 *                            native Q4_K mN kernel (the crossover the
 *                            Mac seed pins at 32 by measurement)
 *
 * Panels use fixed byte patterns with sane fp16 scales (d=1.0, dmin=0)
 * so dequant stays finite; values are irrelevant to timing. Sondes are
 * driver-called (median-of-three lives there); locally each variant
 * gets one warmup call and the minimum of the timed iterations —
 * minimum, not mean, is the noise-resistant estimator for "what can
 * the hardware do".
 *
 * The remaining *_native_mn / prefill-path sondes follow this pattern
 * in PR 2 (#358 program; interview record in the PR).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"
#include "kernel_catalog.h"

#include "geist_gemm.h"
#include "heap.h"
#include "quant.h"

#include <geist_backend.h>

#include <stdint.h>
#include <string.h>
#include <time.h>

static constexpr size_t CAL_N_OUT = 512;  /* panel rows */
static constexpr size_t CAL_N_IN  = 4096; /* panel cols */

static uint64_t cal_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

/* Fill a K-quant super-block panel: pseudo-random nibble bytes, d=1.0
 * (fp16 0x3C00) and dmin=0 at the block head so dequant is finite. */
static void cal_fill_kquant(uint8_t *dst, size_t rows, size_t block_bytes, size_t blocks_per_row) {
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
            blk[2] = 0x00; /* dmin = 0 */
            blk[3] = 0x00;
        }
    }
}

static void cal_fill_x(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] = (float) ((int) (i % 19u) - 9) * 0.25f;
    }
}

/* Minimum wall time of `fn(ctx)` over up to `iters` runs bounded by
 * budget_ns (always at least one timed run after one warmup). */
typedef void (*cal_fn)(void *ctx);

static uint64_t cal_time_min(cal_fn fn, void *ctx, int iters, uint64_t budget_ns) {
    fn(ctx); /* warmup: page-in + pool spin-up */
    uint64_t       best  = UINT64_MAX;
    const uint64_t start = cal_now_ns();
    for (int i = 0; i < iters; i++) {
        const uint64_t t0 = cal_now_ns();
        fn(ctx);
        const uint64_t dt = cal_now_ns() - t0;
        if (dt < best) {
            best = dt;
        }
        if (cal_now_ns() - start > budget_ns) {
            break;
        }
    }
    return best;
}

struct cal_panel {
    const uint8_t *w;
    const float   *x;
    float         *y;
    float         *wf; /* dequant scratch for the sgemm variant */
    size_t         m;
    size_t         block_bytes;
    void (*dequant_row)(size_t, const void *, float *);
    void (*native)(const float *, const void *, size_t, size_t, size_t, float *);
};

static void cal_run_native(void *ctx) {
    struct cal_panel *p = ctx;
    p->native(p->x, p->w, p->m, CAL_N_IN, CAL_N_OUT, p->y);
}

/* The dequant+SGEMM variant mirrors the trampoline's per-call work:
 * dequant every row, then one dense sgemm. */
static void cal_run_sgemm(void *ctx) {
    struct cal_panel *p              = ctx;
    const size_t      blocks_per_row = CAL_N_IN / 256u;
    const size_t      bytes_per_row  = blocks_per_row * p->block_bytes;
    for (size_t r = 0; r < CAL_N_OUT; r++) {
        p->dequant_row(CAL_N_IN, p->w + r * bytes_per_row, p->wf + r * CAL_N_IN);
    }
    geist_sgemm(GEIST_OP_N,
                GEIST_OP_T,
                (int) p->m,
                (int) CAL_N_OUT,
                (int) CAL_N_IN,
                1.0f,
                p->x,
                (int) CAL_N_IN,
                p->wf,
                (int) CAL_N_IN,
                0.0f,
                p->y,
                (int) CAL_N_OUT);
}

static enum geist_status cal_panel_alloc(struct cal_panel *p, size_t block_bytes, size_t m_max) {
    const size_t blocks_per_row = CAL_N_IN / 256u;
    p->block_bytes              = block_bytes;
    uint8_t *w                  = heap_alloc_aligned(CAL_N_OUT * blocks_per_row * block_bytes, 64);
    float   *x                  = heap_alloc_aligned(m_max * CAL_N_IN * sizeof(float), 64);
    float   *y                  = heap_alloc_aligned(m_max * CAL_N_OUT * sizeof(float), 64);
    float   *wf                 = heap_alloc_aligned(CAL_N_OUT * CAL_N_IN * sizeof(float), 64);
    if (w == nullptr || x == nullptr || y == nullptr || wf == nullptr) {
        safe_free((void **) &w);
        safe_free((void **) &x);
        safe_free((void **) &y);
        safe_free((void **) &wf);
        return GEIST_E_OOM;
    }
    cal_fill_kquant(w, CAL_N_OUT, block_bytes, blocks_per_row);
    cal_fill_x(x, m_max * CAL_N_IN);
    p->w  = w;
    p->x  = x;
    p->y  = y;
    p->wf = wf;
    return GEIST_OK;
}

static void cal_panel_free(struct cal_panel *p) {
    safe_free((void **) &p->w);
    safe_free((void **) &p->x);
    safe_free((void **) &p->y);
    safe_free((void **) &p->wf);
}

static enum geist_status
cal_measure_q5k_native_mn(struct geist_backend *be, uint64_t budget_ns, int64_t *out_value) {
    (void) be;
    struct cal_panel  p = {0};
    enum geist_status s = cal_panel_alloc(&p, Q5_K_BLOCK_BYTES, 64);
    if (s != GEIST_OK) {
        return s;
    }
    p.m                   = 64;
    p.dequant_row         = dequant_q5_K_row;
    p.native              = linear_q5k_w5a8_prefill;
    const uint64_t half   = budget_ns / 2u;
    const uint64_t native = cal_time_min(cal_run_native, &p, 5, half);
    const uint64_t sgemm  = cal_time_min(cal_run_sgemm, &p, 5, half);
    cal_panel_free(&p);
    *out_value = native <= sgemm ? 1 : 0;
    return GEIST_OK;
}

static enum geist_status
cal_measure_qk_sgemm_threshold(struct geist_backend *be, uint64_t budget_ns, int64_t *out_value) {
    (void) be;
    static const size_t M_SWEEP[] = {8, 16, 32, 64, 128};
    enum { N_SWEEP = 5 };
    struct cal_panel  p = {0};
    enum geist_status s = cal_panel_alloc(&p, Q4_K_BLOCK_BYTES, M_SWEEP[N_SWEEP - 1]);
    if (s != GEIST_OK) {
        return s;
    }
    p.dequant_row            = dequant_q4_K_row;
    p.native                 = linear_q4k_w4a8_prefill;
    const uint64_t per_point = budget_ns / (2u * N_SWEEP);
    /* Smallest m where dequant+SGEMM wins; above the sweep = "never"
     * (report one past the largest point — policy treats it as
     * effectively disabling the sgemm path on this machine). */
    int64_t threshold = (int64_t) (M_SWEEP[N_SWEEP - 1] * 2u);
    for (size_t i = 0; i < N_SWEEP; i++) {
        p.m                   = M_SWEEP[i];
        const uint64_t native = cal_time_min(cal_run_native, &p, 4, per_point);
        const uint64_t sgemm  = cal_time_min(cal_run_sgemm, &p, 4, per_point);
        if (sgemm < native) {
            threshold = (int64_t) M_SWEEP[i];
            break;
        }
    }
    cal_panel_free(&p);
    *out_value = threshold;
    return GEIST_OK;
}

static const struct geist_tunable cpu_neon_tunables_v1[] = {
        {.name = "q5k_native_mn", .kind = GEIST_TUNABLE_BOOL, .measure = cal_measure_q5k_native_mn},
        {.name    = "qk_sgemm_threshold",
         .kind    = GEIST_TUNABLE_SIZE,
         .measure = cal_measure_qk_sgemm_threshold},
};

const struct geist_tunable *cpu_neon_tunables(size_t *out_count) {
    *out_count = sizeof(cpu_neon_tunables_v1) / sizeof(cpu_neon_tunables_v1[0]);
    return cpu_neon_tunables_v1;
}
