/*
 * bench_sampler — per-token cost of top-k / top-p over large vocabularies.
 *
 * Issue #331: the _ws sampler variants must be allocation-free and must not
 * pay O(n log n) per token. Reports ns/call plus the heap allocation delta
 * measured across the timed loop (must be 0).
 *
 * Usage: bench_sampler [iters]
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "src/engine/sampler.h"

#include "heap.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

/* Logit field with a realistic shape: a handful of dominant tokens over a
 * long low-probability tail. Deterministic — same array for every variant. */
static void fill_logits(size_t n, float *logits, uint64_t seed) {
    struct geist_rng rng;
    geist_rng_seed(&rng, seed);
    for (size_t i = 0; i < n; i++) {
        logits[i] = -6.0f + 4.0f * geist_rng_next_unit(&rng);
    }
    for (int i = 0; i < 24; i++) {
        size_t idx  = (size_t) (geist_rng_next_unit(&rng) * (float) (n - 1));
        logits[idx] = 4.0f + 6.0f * geist_rng_next_unit(&rng);
    }
}

int main(int argc, char **argv) {
    const int    iters    = argc > 1 ? atoi(argv[1]) : 2000;
    const size_t vocabs[] = {32000, 151936, 262144};
    const int    topks[]  = {1, 40, 200, 4096, 40000};

    printf("%-10s %-14s %12s %12s %8s\n", "vocab", "case", "ns/call", "M tok/s", "allocs");
    for (size_t v = 0; v < sizeof vocabs / sizeof *vocabs; v++) {
        const size_t n      = vocabs[v];
        float       *logits = (float *) malloc(n * sizeof(float));
        if (logits == nullptr) {
            fprintf(stderr, "OOM\n");
            return 1;
        }
        fill_logits(n, logits, 0xC0FFEEu + v);

        struct geist_sampler_workspace ws = {0};
        if (geist_sampler_workspace_init(&ws, n) != GEIST_OK) {
            fprintf(stderr, "ws init failed\n");
            return 1;
        }
        struct geist_rng rng;
        geist_rng_seed(&rng, 4242);

        for (size_t t = 0; t < sizeof topks / sizeof *topks; t++) {
            const int k = topks[t];
            if ((size_t) k >= n)
                continue;
            volatile geist_token_t sink = 0;
            const uint64_t         a0   = heap_alloc_count();
            const double           t0   = now_s();
            for (int i = 0; i < iters; i++) {
                sink = geist_sampler_top_k_ws(&ws, logits, k, 0.8f, &rng);
            }
            const double   dt     = now_s() - t0;
            const uint64_t allocs = heap_alloc_count() - a0;
            (void) sink;
            char label[32];
            snprintf(label, sizeof label, "top_k=%d", k);
            printf("%-10zu %-14s %12.1f %12.3f %8llu\n",
                   n,
                   label,
                   dt / iters * 1e9,
                   (double) iters / dt * 1e-6,
                   (unsigned long long) allocs);
        }

        const float tps[] = {0.5f, 0.9f, 0.99f};
        for (size_t t = 0; t < sizeof tps / sizeof *tps; t++) {
            volatile geist_token_t sink = 0;
            const uint64_t         a0   = heap_alloc_count();
            const double           t0   = now_s();
            for (int i = 0; i < iters; i++) {
                sink = geist_sampler_top_p_ws(&ws, logits, tps[t], 0.8f, &rng);
            }
            const double   dt     = now_s() - t0;
            const uint64_t allocs = heap_alloc_count() - a0;
            (void) sink;
            char label[32];
            snprintf(label, sizeof label, "top_p=%.2f", (double) tps[t]);
            printf("%-10zu %-14s %12.1f %12.3f %8llu\n",
                   n,
                   label,
                   dt / iters * 1e9,
                   (double) iters / dt * 1e-6,
                   (unsigned long long) allocs);
        }

        geist_sampler_workspace_destroy(&ws);
        free(logits);
    }
    return 0;
}
