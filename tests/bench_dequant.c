/*
 * bench_dequant — isolate cost of dequant_q3_K_row + dequant_q4_K_row.
 * Measures GB/s for the scalar (current) implementation against synthetic
 * tensor data sized like Gemma 4 E2B FFN weights.
 */
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    /* Two representative tensor sizes:
     *   FFN gate/up: 1536 × 6144 = 9.4M elements
     *   Attn q/k/v:  1536 × 1536 = 2.4M elements
     */
    size_t      sizes[] = {1536 * 6144, 1536 * 1536};
    const char *names[] = {"FFN(gate/up/down)", "Attn(q/k/v/o)"};
    int         iters   = argc > 1 ? atoi(argv[1]) : 10;

    for (int s = 0; s < 2; s++) {
        size_t n_elems = sizes[s];

        /* Q4_K: 144 bytes per 256 elements */
        size_t   q4_blocks = n_elems / Q4_K_BLOCK_ELEMS;
        size_t   q4_bytes  = q4_blocks * Q4_K_BLOCK_BYTES;
        uint8_t *q4_data   = (uint8_t *) malloc(q4_bytes);
        for (size_t i = 0; i < q4_bytes; i++)
            q4_data[i] = (uint8_t) (i * 31 + 7);

        /* Q3_K: 110 bytes per 256 elements */
        size_t   q3_blocks = n_elems / Q3_K_BLOCK_ELEMS;
        size_t   q3_bytes  = q3_blocks * Q3_K_BLOCK_BYTES;
        uint8_t *q3_data   = (uint8_t *) malloc(q3_bytes);
        for (size_t i = 0; i < q3_bytes; i++)
            q3_data[i] = (uint8_t) (i * 17 + 3);

        float *out = (float *) malloc(n_elems * sizeof(float));

        /* Warm */
        dequant_q4_K_row(n_elems, q4_data, out);
        dequant_q3_K_row(n_elems, q3_data, out);

        double t0 = now_s();
        for (int i = 0; i < iters; i++)
            dequant_q4_K_row(n_elems, q4_data, out);
        double t1    = now_s();
        double q4_ms = (t1 - t0) / iters * 1000.0;
        double q4_gb = (q4_bytes / 1e9) / ((t1 - t0) / iters);

        t0 = now_s();
        for (int i = 0; i < iters; i++)
            dequant_q3_K_row(n_elems, q3_data, out);
        t1           = now_s();
        double q3_ms = (t1 - t0) / iters * 1000.0;
        double q3_gb = (q3_bytes / 1e9) / ((t1 - t0) / iters);

        printf("%-22s  Q4_K dequant %5.2f ms (%.2f GB/s read)   Q3_K dequant %5.2f ms (%.2f GB/s "
               "read)\n",
               names[s],
               q4_ms,
               q4_gb,
               q3_ms,
               q3_gb);

        free(q4_data);
        free(q3_data);
        free(out);
    }
    return 0;
}
