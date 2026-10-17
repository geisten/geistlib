/*
 * bench_stream — STREAM Triad (a[i] = b[i] + scale * c[i]) on Pi 5 to
 * establish the actual sustained DRAM bandwidth ceiling. Compare against
 * what bench_ptqtp achieves to decide if the kernel is bandwidth-bound.
 *
 * Reports GB/s for COPY, SCALE, ADD, TRIAD. TRIAD is the canonical
 * comparison number (3 fp32 reads/writes per index = 12 bytes/elem).
 *
 * Usage:
 *   bench_stream                       — default 64 MB array, 10 reps
 *   bench_stream <MB> <reps>           — custom
 *
 * Threading: respects OMP_NUM_THREADS. Match production (=4) to compare
 * against the multi-threaded PTQTP kernel.
 */
#include "test_helpers.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

int main(int argc, char **argv) {
    const size_t MB   = (argc >= 2) ? (size_t) atoi(argv[1]) : 64;
    const int    reps = (argc >= 3) ? atoi(argv[2]) : 10;
    const size_t N    = (MB * 1024UL * 1024UL) / sizeof(double);

    double *a = (double *) aligned_alloc(64, N * sizeof(double));
    double *b = (double *) aligned_alloc(64, N * sizeof(double));
    double *c = (double *) aligned_alloc(64, N * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "alloc fail\n");
        return 1;
    }

/* First-touch parallel init so pages get faulted into the right NUMA
 * node (Pi 5 is single-socket so this is mostly cosmetic). */
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }

    const double scale     = 3.0;
    double       best_copy = 1e30, best_scale = 1e30, best_add = 1e30, best_triad = 1e30;

    for (int r = 0; r < reps; r++) {
        double t0, t1;

        t0 = geist_test_now_s();
#pragma omp parallel for
        for (size_t i = 0; i < N; i++)
            c[i] = a[i];
        t1 = geist_test_now_s();
        if (t1 - t0 < best_copy)
            best_copy = t1 - t0;

        t0 = geist_test_now_s();
#pragma omp parallel for
        for (size_t i = 0; i < N; i++)
            b[i] = scale * c[i];
        t1 = geist_test_now_s();
        if (t1 - t0 < best_scale)
            best_scale = t1 - t0;

        t0 = geist_test_now_s();
#pragma omp parallel for
        for (size_t i = 0; i < N; i++)
            c[i] = a[i] + b[i];
        t1 = geist_test_now_s();
        if (t1 - t0 < best_add)
            best_add = t1 - t0;

        t0 = geist_test_now_s();
#pragma omp parallel for
        for (size_t i = 0; i < N; i++)
            a[i] = b[i] + scale * c[i];
        t1 = geist_test_now_s();
        if (t1 - t0 < best_triad)
            best_triad = t1 - t0;
    }

    /* Bytes moved per kernel: COPY 2N, SCALE 2N, ADD 3N, TRIAD 3N (each *8). */
    const double mb = (double) (MB);
    printf("STREAM (N=%zu doubles = %zu MB, %d reps, OMP_NUM_THREADS=%s)\n",
           N,
           MB,
           reps,
           getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "unset");
    printf("  COPY :  %7.2f GB/s\n", (2.0 * mb / 1024.0) / best_copy);
    printf("  SCALE:  %7.2f GB/s\n", (2.0 * mb / 1024.0) / best_scale);
    printf("  ADD  :  %7.2f GB/s\n", (3.0 * mb / 1024.0) / best_add);
    printf("  TRIAD:  %7.2f GB/s\n", (3.0 * mb / 1024.0) / best_triad);

    /* Anti-DCE. */
    if (a[0] == 0.0 && a[N - 1] == 0.0)
        fprintf(stderr, "(zero output)\n");

    free(a);
    free(b);
    free(c);
    return 0;
}
