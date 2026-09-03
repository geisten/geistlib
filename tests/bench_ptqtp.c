/*
 * bench_ptqtp — focused microbench for the PTQTP 2-plane decoder.
 *
 * Loads one tensor from a real PTQTP file and loops the GEMV kernel for
 * ≥5 seconds wall time (so perf has plenty of samples). Reports
 * ms/call + GB/s effective. Wrap in `perf stat` / `perf record` for the
 * hardware-counter side.
 *
 * Usage:
 *   bench_ptqtp <ptqtp.bin>                 — bench blk.0.ffn_gate.weight (largest layer matmul)
 *   bench_ptqtp <ptqtp.bin> <tensor_name>   — bench just one
 *   bench_ptqtp <ptqtp.bin> ALL              — sweep canonical shapes
 *
 * Threading: respects OMP_NUM_THREADS. Run with =1 for single-thread,
 * =4 for production OMP=4 measurement.
 */
#include "gguf_ptqtp.h"
#include "quant.h"
#include "ptqtp_kernel.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double) tv.tv_sec * 1000.0 + (double) tv.tv_usec / 1000.0;
}

/* Kernel variants for A/B comparison. */
typedef enum { KERNEL_FP32_ALPHA, KERNEL_FP16_ALPHA } kernel_variant_t;

static void bench_one(const struct ptqtp_ctx *ctx,
                      const char             *name,
                      double                  target_sec,
                      kernel_variant_t        variant) {
    const struct ptqtp_tensor_t *t = ptqtp_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "  %-32s NOT FOUND\n", name);
        return;
    }
    const uint32_t gs         = ptqtp_group_size(ctx);
    const size_t   n_in       = t->n_in;
    const size_t   n_out      = t->n_out;
    const size_t   trit_bytes = (size_t) n_out * n_in / 2;
    const size_t   alpha_elem_size =
            (variant == KERNEL_FP16_ALPHA) ? sizeof(uint16_t) : sizeof(float);
    const size_t alpha_bytes    = (size_t) n_out * t->n_groups * 2 * alpha_elem_size;
    const size_t bytes_per_call = trit_bytes + alpha_bytes;
    const double mb_per_call    = (double) bytes_per_call / (1024.0 * 1024.0);

    int8_t *x_q8 = (int8_t *) aligned_alloc(64, n_in * sizeof(int8_t));
    float  *y    = (float *) aligned_alloc(64, n_out * sizeof(float));
    if (!x_q8 || !y) {
        fprintf(stderr, "  alloc fail\n");
        return;
    }
    float *x = (float *) aligned_alloc(64, n_in * sizeof(float));
    for (size_t i = 0; i < n_in; i++)
        x[i] = ((float) i * 0.0137f) - 7.3f;
    float scale_x = quantize_x_int8_sym(n_in, x, x_q8);
    free(x);

#define CALL_KERNEL()                                                                           \
    do {                                                                                        \
        if (variant == KERNEL_FP16_ALPHA)                                                       \
            ptqtp_gemv_2plane_fp16alpha(n_in, n_out, gs, x_q8, scale_x, t->trits, t->alpha, y); \
        else                                                                                    \
            ptqtp_gemv_2plane_fp32alpha(                                                        \
                    n_in, n_out, gs, x_q8, scale_x, t->trits, t->alpha_fp32, y);                \
    } while (0)

    const double tw = now_ms();
    CALL_KERNEL();
    const double single_ms = now_ms() - tw;
    int          n_iter    = (int) ((target_sec * 1000.0) / (single_ms + 0.001)) + 5;
    if (n_iter < 3)
        n_iter = 3;
    if (n_iter > 100000)
        n_iter = 100000;

    const double t0 = now_ms();
    for (int it = 0; it < n_iter; it++)
        CALL_KERNEL();
    const double dt_ms_total = now_ms() - t0;
    const double dt_ms       = dt_ms_total / n_iter;
    const double gbps        = (double) bytes_per_call / (dt_ms * 1e6);

    printf("  [%s] %-32s n_out=%6zu n_in=%5zu  %7.1f MB  %7.2f ms/call  %5.2f GB/s  (%d it, "
           "%.1fs)\n",
           variant == KERNEL_FP16_ALPHA ? "fp16α" : "fp32α",
           name,
           n_out,
           n_in,
           mb_per_call,
           dt_ms,
           gbps,
           n_iter,
           dt_ms_total / 1000.0);
    fflush(stdout);

    if (y[0] == 0.0f && y[n_out - 1] == 0.0f)
        fprintf(stderr, "(zero output)\n");

    free(x_q8);
    free(y);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <ptqtp.bin> [<tensor_name>|ALL]\n", argv[0]);
        return 2;
    }
    const char       *err = nullptr;
    struct ptqtp_ctx *ctx = ptqtp_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "ptqtp_open: %s\n", err ? err : "?");
        return 1;
    }

    /* Default target wall for the hot loop — 5 sec gives perf >5000 samples
     * at -F 999. Sweep mode uses 1.5 sec each to stay quick. */
    const char *target_name = (argc == 3) ? argv[2] : "blk.0.ffn_gate.weight";

    printf("bench_ptqtp — PTQTP 2-plane GEMV throughput on %s\n", argv[1]);
    printf("(OMP_NUM_THREADS=%s)\n",
           getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "unset");
    printf("  tensor                                                  size   per-call   "
           "bandwidth\n");

    if (target_name && strcmp(target_name, "ALL") == 0) {
        static const char *shapes[] = {
                "blk.0.attn_q.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_down.weight",
                "token_embd.weight",
                nullptr,
        };
        for (int i = 0; shapes[i]; i++) {
            bench_one(ctx, shapes[i], 1.5, KERNEL_FP32_ALPHA);
            bench_one(ctx, shapes[i], 1.5, KERNEL_FP16_ALPHA);
        }
    } else {
        bench_one(ctx, target_name, 5.0, KERNEL_FP32_ALPHA);
        bench_one(ctx, target_name, 5.0, KERNEL_FP16_ALPHA);
    }

    ptqtp_close(ctx);
    return 0;
}
