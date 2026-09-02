/*
 * test_ptqtp — load gemma4-e2b.ptqtp.bin, run linear_ptqtp_decode_2plane on
 * one tensor, compare against an FP32 reference reconstruction.
 *
 * Usage: test_ptqtp <ptqtp.bin> [tensor_name]
 *   defaults to "blk.0.attn_q.weight"
 *
 * Pass criterion: cos sim ≥ 0.99999 between fast NEON kernel and FP32 ref.
 * (The kernels do the same math, this measures only int8 quant + accumulation
 * order rounding noise.)
 */
#include "gguf_ptqtp.h"
#include "quant.h"
#include "ptqtp_kernel.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same lookup tables as kernel, in scalar form. */
static const int8_t T1_LUT[16] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
static const int8_t T2_LUT[16] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0};

/* Reconstruct FP32 weights row r from PTQTP storage. Then do plain dot with x. */
static void ptqtp_reference_row(const struct ptqtp_tensor_t *t,
                                size_t                       group_size,
                                size_t                       row,
                                const float                 *x,
                                float                       *y_out) {
    const size_t    n_in      = t->n_in;
    const uint8_t  *row_trits = t->trits + row * (n_in / 2);
    const uint16_t *row_alpha = t->alpha + row * t->n_groups * 2;

    float acc = 0.0f;
    for (size_t g = 0; g < t->n_groups; g++) {
        const uint8_t *g_trits = row_trits + g * (group_size / 2);
        const float    a1      = fp16_to_fp32(row_alpha[g * 2 + 0]);
        const float    a2      = fp16_to_fp32(row_alpha[g * 2 + 1]);
        for (size_t k = 0; k < group_size / 2; k++) {
            const uint8_t b   = g_trits[k];
            const int8_t  T1a = T1_LUT[b & 0x0F];
            const int8_t  T2a = T2_LUT[b & 0x0F];
            const int8_t  T1b = T1_LUT[b >> 4];
            const int8_t  T2b = T2_LUT[b >> 4];
            const float   xa  = x[g * group_size + 2 * k];
            const float   xb  = x[g * group_size + 2 * k + 1];
            acc += (a1 * (float) T1a + a2 * (float) T2a) * xa;
            acc += (a1 * (float) T1b + a2 * (float) T2b) * xb;
        }
    }
    *y_out = acc;
}

static double cos_sim(size_t n, const float *a, const float *b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return na > 0 && nb > 0 ? dot / (sqrt(na) * sqrt(nb)) : 1.0;
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 2, "<ptqtp.bin> [tensor_name]");
    const char *path  = argv[1];
    const char *tname = argc >= 3 ? argv[2] : "blk.0.attn_q.weight";

    const char       *err = nullptr;
    struct ptqtp_ctx *ctx = ptqtp_open(path, &err);
    if (!ctx) {
        fprintf(stderr, "ptqtp_open: %s\n", err ? err : "?");
        return 1;
    }
    fprintf(stderr, "n_tensors=%zu group_size=%u\n", ptqtp_n_tensors(ctx), ptqtp_group_size(ctx));

    const struct ptqtp_tensor_t *t = ptqtp_get_tensor(ctx, tname);
    if (!t) {
        fprintf(stderr, "tensor %s not found\n", tname);
        return 1;
    }
    fprintf(stderr,
            "tensor %s: %u out × %u in, %u groups, cos@quant=%.4f\n",
            t->name,
            t->n_out,
            t->n_in,
            t->n_groups,
            t->cos_sim);

    /* Synthetic input. Deterministic for reproducibility. */
    float *x = (float *) malloc(t->n_in * sizeof(float));
    for (uint32_t i = 0; i < t->n_in; i++)
        x[i] = sinf((float) i * 0.0137f) * 0.5f - 0.07f;

    /* Quantize x once. */
    int8_t *x_q8    = (int8_t *) malloc(t->n_in);
    float   scale_x = quantize_x_int8_sym(t->n_in, x, x_q8);
    fprintf(stderr, "scale_x=%.6f\n", scale_x);

    /* Fast kernel: full output. */
    float *y_fast = (float *) malloc(t->n_out * sizeof(float));
    ptqtp_gemv_2plane_fp32alpha(t->n_in,
                                t->n_out,
                                ptqtp_group_size(ctx),
                                x_q8,
                                scale_x,
                                t->trits,
                                t->alpha_fp32,
                                y_fast);

    /* Reference: same per-row math but in pure FP32 (no x quantization).
     * calloc instead of malloc: gcc-14 can't see through ptqtp_reference_row
     * to know each y_ref[r] is initialized, so it warns on the y_ref[0] read
     * below. Zero-init silences the false positive without changing behavior. */
    float *y_ref = (float *) calloc(t->n_out, sizeof(float));
    for (uint32_t r = 0; r < t->n_out; r++) {
        ptqtp_reference_row(t, ptqtp_group_size(ctx), r, x, &y_ref[r]);
    }

    /* Stats. */
    double cs     = cos_sim(t->n_out, y_fast, y_ref);
    float  ymin_f = y_fast[0], ymax_f = y_fast[0];
    float  ymin_r = y_ref[0], ymax_r = y_ref[0];
    for (uint32_t i = 1; i < t->n_out; i++) {
        if (y_fast[i] < ymin_f)
            ymin_f = y_fast[i];
        if (y_fast[i] > ymax_f)
            ymax_f = y_fast[i];
        if (y_ref[i] < ymin_r)
            ymin_r = y_ref[i];
        if (y_ref[i] > ymax_r)
            ymax_r = y_ref[i];
    }
    fprintf(stderr,
            "fast: y[0..3]=%.4f %.4f %.4f %.4f  range=[%.3f, %.3f]\n",
            y_fast[0],
            y_fast[1],
            y_fast[2],
            y_fast[3],
            ymin_f,
            ymax_f);
    fprintf(stderr,
            "ref:  y[0..3]=%.4f %.4f %.4f %.4f  range=[%.3f, %.3f]\n",
            y_ref[0],
            y_ref[1],
            y_ref[2],
            y_ref[3],
            ymin_r,
            ymax_r);
    fprintf(stderr, "cos_sim(fast, ref) = %.7f\n", cs);

    int rc = (cs >= 0.99999) ? 0 : 1;
    fprintf(stderr,
            "%s: cos %.6f %s 0.99999\n",
            rc == 0 ? "PASS" : "FAIL",
            cs,
            rc == 0 ? "≥" : "<");

    /* GEMM cross-check: same input replicated across M rows must produce
     * M identical output rows that each match the GEMV result. */
    const size_t Ms[] = {2, 4, 16};
    for (size_t mi = 0; mi < sizeof(Ms) / sizeof(Ms[0]); mi++) {
        const size_t M      = Ms[mi];
        int8_t      *x_q8_m = (int8_t *) malloc(M * t->n_in);
        float       *scales = (float *) malloc(M * sizeof(float));
        float       *y_gemm = (float *) malloc(M * t->n_out * sizeof(float));
        for (size_t i = 0; i < M; i++) {
            memcpy(x_q8_m + i * t->n_in, x_q8, t->n_in);
            scales[i] = scale_x;
        }
        ptqtp_gemm_2plane_fp32alpha(M,
                                    t->n_in,
                                    t->n_out,
                                    ptqtp_group_size(ctx),
                                    x_q8_m,
                                    scales,
                                    t->trits,
                                    t->alpha_fp32,
                                    y_gemm);
        size_t mismatches = 0;
        for (size_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < t->n_out; j++) {
                if (y_gemm[i * t->n_out + j] != y_fast[j])
                    mismatches++;
            }
        }
        fprintf(stderr, "GEMM M=%zu: %zu mismatches vs GEMV (expect 0)\n", M, mismatches);
        if (mismatches != 0)
            rc = 1;
        free(x_q8_m);
        free(scales);
        free(y_gemm);
    }

    free(x);
    free(x_q8);
    free(y_fast);
    free(y_ref);
    ptqtp_close(ctx);
    return rc;
}
