/*
 * tuning.c — per-device thresholds for the metal backend.
 *
 * Layer: BACKEND (metal). The kernel choices below are crossovers, not
 * constants: where a wider threadgroup starts to pay depends on the GPU's
 * core count and memory latency, and every seed here was measured on one
 * machine (M1 Max, Bonsai-27B shapes). Rather than growing a per-device
 * branch, they go through the same three-stage resolution cpu_neon uses:
 * built-in seed, then an applied calibration value (the driver keys its
 * blob on the micro-architecture, so an M5 gets its own), then the env
 * override. The sondes below are what `geist_backend_calibrate` runs.
 */
#include "metal_internal.h"

#include "quant.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Seeds: M1 Max, 27B shapes. See the tunable comments for the measurement.
 *
 * wide_rows_min_cols has no sonde: at rows == 1 the rmsnorm and F32 GEMV
 * kernels are dispatch-latency-bound, and repeated A/Bs on an idle M1 Max
 * put the 256- vs 1024-thread difference inside the run-to-run noise (the
 * crossover wandered between 2048 and 8192 across runs). A knob that
 * cannot be measured reliably is a seed plus an env override, not a
 * calibration value. */
static constexpr uint32_t SEED_PQ2_N8_MIN_N_OUT   = 6144u;
static constexpr uint32_t SEED_WIDE_ROWS_MIN_COLS = 1024u;

/* Sonde geometry: decode shape (rows == 1) on an FFN-class input width.
 * SONDE_COPIES weights are cycled so the reads come from DRAM the way a
 * real decode step's do — a single panel would sit in the last-level
 * cache and measure the wrong regime. A variant has to win by
 * SONDE_MARGIN to count, so run-to-run noise does not decide. */
static constexpr size_t SONDE_N_IN       = 5120;
static constexpr size_t SONDE_REPS       = 16;
static constexpr size_t SONDE_WARM       = 4;
static constexpr size_t SONDE_MAX_COPIES = 48;
/* Enough distinct weights to outrun any last-level cache (M1 Max: 48 MB). */
static constexpr size_t SONDE_WORKING_SET = 128u << 20;
static constexpr double SONDE_MARGIN      = 0.03;
static const size_t     SONDE_OUT[]       = {1024, 2048, 4096, 6144, 8192, 12288};

static uint64_t tuning_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static uint32_t tuning_env_u32(const char *name, uint32_t fallback) {
    const char *v = getenv(name);
    if (v == nullptr || *v == '\0') {
        return fallback;
    }
    char               *end = nullptr;
    const unsigned long n   = strtoul(v, &end, 10);
    return end != v && n > 0u && n <= UINT32_MAX ? (uint32_t) n : fallback;
}

static uint32_t
tuning_resolve(struct geist_backend *be, const char *tunable, const char *env, uint32_t seed) {
    int64_t v = 0;
    if (geist_calibration_lookup(be, tunable, &v) && v > 0 && v <= UINT32_MAX) {
        seed = (uint32_t) v;
    }
    return tuning_env_u32(env, seed);
}

/* At create time the calibration store is still empty -- a blob can only
 * be applied to an already-created backend -- so this seeds the values
 * and metal_tuning_resolve folds the blob in later, the way cpu_neon
 * overlays its policy at the first weight resolve. */
void metal_tuning_init(struct geist_backend *be, struct metal_state *st) {
    (void) be;
    st->tuning.pq2_n8_min_n_out   = SEED_PQ2_N8_MIN_N_OUT;
    st->tuning.wide_rows_min_cols = SEED_WIDE_ROWS_MIN_COLS;
    st->tuning.resolved           = false;
}

void metal_tuning_resolve(struct geist_backend *be, struct metal_state *st) {
    if (st == nullptr || st->tuning.resolved) {
        return;
    }
    st->tuning.resolved         = true;
    st->tuning.pq2_n8_min_n_out = tuning_resolve(
            be, "pq2_n8_min_n_out", "GEIST_METAL_PQ2_N8_MIN_N_OUT", SEED_PQ2_N8_MIN_N_OUT);
    st->tuning.wide_rows_min_cols = tuning_resolve(
            be, "wide_rows_min_cols", "GEIST_METAL_WIDE_ROWS_MIN_COLS", SEED_WIDE_ROWS_MIN_COLS);
}

/* ---- Sondes ------------------------------------------------------------
 *
 * Both measure a crossover by running the same shape twice with the
 * threshold pinned to each side, so they exercise the production dispatch
 * rather than a copy of it. They restore the live threshold before
 * returning; a failure anywhere leaves the seed in place and reports OK
 * with the seed value, because a backend that cannot measure must not
 * block calibration of the others.
 */

static bool sonde_fill_pq2(uint8_t *raw, size_t bytes) {
    uint32_t lcg = 0x12345u;
    for (size_t b = 0; b + PQ2_0_BLOCK_BYTES <= bytes; b += PQ2_0_BLOCK_BYTES) {
        raw[b]     = 0x00; /* d = fp16 1.0 */
        raw[b + 1] = 0x3C;
        for (size_t i = 2; i < PQ2_0_BLOCK_BYTES; i++) {
            lcg        = lcg * 1664525u + 1013904223u;
            raw[b + i] = (uint8_t) ((lcg >> 24) & 0x55u); /* codes 0/1 only */
        }
    }
    return true;
}

/* Best-of-three wall time per decode GEMV of [n_out, SONDE_N_IN], over
 * SONDE_COPIES distinct weights so the reads miss cache. */
static bool sonde_time_pq2_gemv(struct geist_backend *be, size_t n_out, double *out_ns) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    const size_t bytes  = n_out * (SONDE_N_IN / PQ2_0_BLOCK_ELEMS) * PQ2_0_BLOCK_BYTES;
    const size_t copies = bytes == 0 ? 0
                          : SONDE_WORKING_SET / bytes < SONDE_MAX_COPIES
                                  ? (SONDE_WORKING_SET / bytes < 2 ? 2 : SONDE_WORKING_SET / bytes)
                                  : SONDE_MAX_COPIES;
    struct geist_buffer *bw[SONDE_MAX_COPIES] = {nullptr};
    struct geist_weight  w[SONDE_MAX_COPIES]  = {0};
    if (copies == 0) {
        return false;
    }
    struct geist_buffer *bx = nullptr, *by = nullptr;
    bool ok = v->buffer_create(be, SONDE_N_IN * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &bx) ==
                      GEIST_OK &&
              v->buffer_create(be, n_out * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &by) ==
                      GEIST_OK;
    for (size_t c = 0; ok && c < copies; c++) {
        ok           = v->buffer_create(be, bytes, GEIST_BUFFER_WEIGHT, 0, &bw[c]) == GEIST_OK;
        uint8_t *raw = ok ? v->buffer_map(bw[c]) : nullptr;
        ok           = raw != nullptr;
        if (ok) {
            sonde_fill_pq2(raw, bytes);
            w[c] = (struct geist_weight) {.raw        = raw,
                                          .raw_nbytes = bytes,
                                          .n_in       = (int32_t) SONDE_N_IN,
                                          .n_out      = (int32_t) n_out,
                                          .dtype      = (uint16_t) GEIST_DTYPE_PQ2_0};
            ok   = v->resolve_weight(be, &w[c]) == GEIST_OK && w[c].linear_m1 != nullptr;
        }
    }
    double best = 0.0;
    if (ok) {
        float *x = v->buffer_map(bx);
        float *y = v->buffer_map(by);
        ok       = x != nullptr && y != nullptr;
        if (ok) {
            for (size_t i = 0; i < SONDE_N_IN; i++) {
                x[i] = (float) (i % 17u) * 0.01f - 0.08f;
            }
            for (size_t i = 0; i < SONDE_WARM; i++) {
                w[i % copies].linear_m1(x, &w[i % copies], be, y);
            }
            for (int rep = 0; rep < 3; rep++) {
                const uint64_t t0 = tuning_now_ns();
                for (size_t i = 0; i < SONDE_REPS; i++) {
                    w[i % copies].linear_m1(x, &w[i % copies], be, y);
                }
                const double dt = (double) (tuning_now_ns() - t0) / (double) SONDE_REPS;
                if (best == 0.0 || dt < best) {
                    best = dt;
                }
            }
        }
    }
    for (size_t c = 0; c < SONDE_MAX_COPIES; c++) {
        v->buffer_destroy(be, bw[c]);
    }
    v->buffer_destroy(be, bx);
    v->buffer_destroy(be, by);
    *out_ns = best;
    return ok && best > 0.0;
}

/* The n_out from which 8 rows per simdgroup beat 4: the wider kernel
 * halves the activation traffic but halves the grid too, so it needs
 * enough output rows to still fill the GPU. */
static enum geist_status
metal_measure_pq2_n8_min_n_out(struct geist_backend *be, uint64_t budget_ns, int64_t *out_value) {
    struct metal_state *st = be != nullptr ? be->state : nullptr;
    if (st == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    metal_tuning_resolve(be, st);
    const uint32_t live     = st->tuning.pq2_n8_min_n_out;
    const uint64_t deadline = tuning_now_ns() + budget_ns;
    uint32_t       cross    = SEED_PQ2_N8_MIN_N_OUT;
    for (size_t i = 0; i < sizeof SONDE_OUT / sizeof SONDE_OUT[0]; i++) {
        if (tuning_now_ns() >= deadline) {
            break;
        }
        double n4 = 0.0, n8 = 0.0;
        st->tuning.pq2_n8_min_n_out = UINT32_MAX; /* pin the 4-row kernel */
        const bool ok4              = sonde_time_pq2_gemv(be, SONDE_OUT[i], &n4);
        st->tuning.pq2_n8_min_n_out = 0u; /* pin the 8-row kernel */
        const bool ok8              = sonde_time_pq2_gemv(be, SONDE_OUT[i], &n8);
        if (!ok4 || !ok8) {
            break;
        }
        if (n8 < n4 * (1.0 - SONDE_MARGIN)) {
            cross = (uint32_t) SONDE_OUT[i];
            break;
        }
        cross = (uint32_t) SONDE_OUT[i] * 2u; /* still losing: look higher */
    }
    st->tuning.pq2_n8_min_n_out = live;
    *out_value                  = (int64_t) cross;
    return GEIST_OK;
}

static const struct geist_tunable metal_tunables_v1[] = {
        {.name    = "pq2_n8_min_n_out",
         .kind    = GEIST_TUNABLE_SIZE,
         .measure = metal_measure_pq2_n8_min_n_out},
};

const struct geist_tunable *metal_tunables(size_t *out_count) {
    *out_count = sizeof metal_tunables_v1 / sizeof metal_tunables_v1[0];
    return metal_tunables_v1;
}
