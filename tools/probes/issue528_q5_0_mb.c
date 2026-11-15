/* issue528_q5_0_mb: cpu_neon's Q5_0 linear at DeepSeek-V2-Lite's expert
 * shape (ffn_down 1408 -> 2048), per row count: the M=1 kernel and the M>1
 * kernel at m = 1..128, each as one call on the whole team ("team") and as
 * one call per thread with every thread busy ("busy", how forward/moe.c runs
 * all but the widest expert). Best of 7 blocks. Built against a revision's
 * libgeist.a by issue528_q5_0_mb.sh. Probe branch only. */
#include "test_helpers.h"
#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>
#include "heap.h"
#include "par.h"
#include "quant_fixtures.h"
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

struct job {
    const struct geist_weight *w;
    struct geist_backend      *be;
    size_t                     m, reps, n_in, n_out;
    const float               *x;
    float                     *y;
};

static void call(const struct job *j, float *y) {
    if (j->m == 0)
        j->w->linear_m1(j->x, j->w, j->be, y);
    else
        j->w->linear_mN(j->m, j->x, j->w, j->be, y);
}

static void busy_body(void *ctx, size_t t0, size_t t1) {
    const struct job *j = ctx;
    for (size_t t = t0; t < t1; t++)
        for (size_t r = 0; r < j->reps; r++)
            call(j, j->y + t * 128 * j->n_out);
}

int main(void) {
    struct geist_backend *be;
    if (geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK)
        return 1;
    const size_t n_in = 1408, n_out = 2048, T = geist_par_max_threads();
    size_t       d = 0, bytes = 0;
    for (size_t i = 0; i < N_DTYPES; i++)
        if (DTYPES[i].dt == GEIST_DTYPE_Q5_0)
            d = i;
    uint8_t            *raw = make_weight(d, n_in, n_out, &bytes);
    struct geist_weight w   = {.raw        = raw,
                               .raw_nbytes = bytes,
                               .n_in       = (int32_t) n_in,
                               .n_out      = (int32_t) n_out,
                               .dtype      = (uint16_t) GEIST_DTYPE_Q5_0};
    if (be->desc->vtbl->resolve_weight(be, &w) != GEIST_OK)
        return 1;
    float *x = xmalloc(128 * n_in * sizeof(float));
    float *y = xmalloc(T * 128 * n_out * sizeof(float));
    for (size_t i = 0; i < 128 * n_in; i++)
        x[i] = next_f();
    const size_t ms[] = {0, 1, 2, 4, 8, 12, 16, 32, 64, 128};
    printf("threads %zu\n", T);
    for (size_t k = 0; k < sizeof ms / sizeof ms[0]; k++) {
        struct job j = {.w = &w, .be = be, .m = ms[k], .n_in = n_in, .n_out = n_out, .x = x, .y = y};
        j.reps       = ms[k] >= 32 ? 4 : 12;
        double team = 1e9, busy = 1e9;
        for (int r = 0; r < 7; r++) {
            double t0 = now();
            for (size_t c = 0; c < j.reps; c++)
                call(&j, y);
            double dt = (now() - t0) / (double) j.reps;
            team      = dt < team ? dt : team;
            t0        = now();
            geist_par_for(T, busy_body, &j);
            dt   = (now() - t0) / (double) j.reps;
            busy = dt < busy ? dt : busy;
        }
        if (ms[k] == 0)
            printf("m1        team %8.1f us   busy %8.1f us\n", team * 1e6, busy * 1e6);
        else
            printf("mN m=%-3zu  team %8.1f us   busy %8.1f us\n", ms[k], team * 1e6, busy * 1e6);
    }
    return 0;
}
