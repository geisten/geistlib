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

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same lookup tables as kernel, in scalar form. */
static const int8_t T1_LUT[16] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
static const int8_t T2_LUT[16] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0};
/* GEMV and GEMM have different loop nests, so -ffast-math may legally choose
 * different FMA/reassociation sequences. Gate their numerical error tightly
 * instead of requiring an optimizer-dependent bit pattern. The absolute floor
 * protects values near zero; the relative term scales the bound elsewhere. */
static constexpr float ptqtp_abs_tolerance = 4.0f * FLT_EPSILON;
static constexpr float ptqtp_rel_tolerance = 1.0e-6f;

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

/* Independent arithmetic oracle: scalar LUT decode, exact integer group dots,
 * and double accumulation over the already-quantized activation. This isolates
 * kernel arithmetic from the separate fp32-input quantization quality gate. */
static float ptqtp_quantized_reference_row(const struct ptqtp_tensor_t *t,
                                           size_t                       n_in,
                                           size_t                       group_size,
                                           size_t                       row,
                                           float                        scale_x,
                                           const int8_t                 x_q8[static n_in]) {
    const uint8_t *row_trits = t->trits + row * (n_in / 2);
    const float   *row_alpha = t->alpha_fp32 + row * t->n_groups * 2;
    double         acc       = 0.0;
    for (size_t g = 0; g < t->n_groups; g++) {
        const uint8_t *g_trits = row_trits + g * (group_size / 2);
        int32_t        acc1 = 0, acc2 = 0;
        for (size_t k = 0; k < group_size / 2; k++) {
            const uint8_t b  = g_trits[k];
            const int8_t  xa = x_q8[g * group_size + 2 * k];
            const int8_t  xb = x_q8[g * group_size + 2 * k + 1];
            acc1 += (int32_t) T1_LUT[b & 0x0F] * xa + (int32_t) T1_LUT[b >> 4] * xb;
            acc2 += (int32_t) T2_LUT[b & 0x0F] * xa + (int32_t) T2_LUT[b >> 4] * xb;
        }
        acc += (double) row_alpha[g * 2] * acc1 + (double) row_alpha[g * 2 + 1] * acc2;
    }
    return (float) ((double) scale_x * acc);
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

static uint32_t float_ulp_distance(float a, float b) {
    uint32_t a_bits, b_bits;
    memcpy(&a_bits, &a, sizeof(a_bits));
    memcpy(&b_bits, &b, sizeof(b_bits));
    const uint32_t a_ordered = (a_bits & 0x80000000u) != 0 ? ~a_bits : a_bits | 0x80000000u;
    const uint32_t b_ordered = (b_bits & 0x80000000u) != 0 ? ~b_bits : b_bits | 0x80000000u;
    return a_ordered >= b_ordered ? a_ordered - b_ordered : b_ordered - a_ordered;
}

struct error_stats {
    size_t   mismatches;
    size_t   out_of_tolerance;
    float    max_abs;
    float    max_rel;
    uint32_t max_ulps;
};

[[nodiscard]] static struct error_stats
compare_outputs(size_t n, const float expect[static n], const float actual[static n]) {
    struct error_stats stats = {0};
    for (size_t i = 0; i < n; i++) {
        const float    abs_d = fabsf(actual[i] - expect[i]);
        const float    rel_d = abs_d / fmaxf(fabsf(expect[i]), 1.0e-30f);
        const uint32_t ulps  = float_ulp_distance(actual[i], expect[i]);
        stats.mismatches += actual[i] != expect[i];
        stats.out_of_tolerance +=
                abs_d > ptqtp_abs_tolerance + ptqtp_rel_tolerance * fabsf(expect[i]);
        stats.max_abs  = fmaxf(stats.max_abs, abs_d);
        stats.max_rel  = fmaxf(stats.max_rel, rel_d);
        stats.max_ulps = stats.max_ulps >= ulps ? stats.max_ulps : ulps;
    }
    return stats;
}

[[nodiscard]] static struct error_stats
compare_replicated_outputs(size_t m, size_t n, const float expect[static n], const float *actual) {
    struct error_stats stats = {0};
    for (size_t row = 0; row < m; row++) {
        const struct error_stats row_stats = compare_outputs(n, expect, &actual[row * n]);
        stats.mismatches += row_stats.mismatches;
        stats.out_of_tolerance += row_stats.out_of_tolerance;
        stats.max_abs  = fmaxf(stats.max_abs, row_stats.max_abs);
        stats.max_rel  = fmaxf(stats.max_rel, row_stats.max_rel);
        stats.max_ulps = stats.max_ulps >= row_stats.max_ulps ? stats.max_ulps : row_stats.max_ulps;
    }
    return stats;
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
    float *y_ref       = (float *) calloc(t->n_out, sizeof(float));
    float *y_quant_ref = xmalloc((size_t) t->n_out * sizeof(*y_quant_ref));
    for (uint32_t r = 0; r < t->n_out; r++) {
        ptqtp_reference_row(t, ptqtp_group_size(ctx), r, x, &y_ref[r]);
        y_quant_ref[r] =
                ptqtp_quantized_reference_row(t, t->n_in, ptqtp_group_size(ctx), r, scale_x, x_q8);
    }

    const struct error_stats gemv_ref = compare_outputs(t->n_out, y_quant_ref, y_fast);
    fprintf(stderr,
            "arithmetic tolerance: |delta| <= %.9g + %.9g * |reference|\n",
            ptqtp_abs_tolerance,
            ptqtp_rel_tolerance);
    fprintf(stderr,
            "GEMV vs quantized ref: %zu out of tolerance "
            "(max_abs=%.9g max_rel=%.9g max_ulps=%u)\n",
            gemv_ref.out_of_tolerance,
            gemv_ref.max_abs,
            gemv_ref.max_rel,
            gemv_ref.max_ulps);

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
    if (gemv_ref.out_of_tolerance != 0)
        rc = 1;

    /* GEMM cross-check: same input replicated across M rows must produce
     * M numerically equivalent rows that match GEMV and the independent
     * quantized-input reference within the arithmetic tolerance above. */
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
        const struct error_stats gemm_gemv =
                compare_replicated_outputs(M, t->n_out, y_fast, y_gemm);
        const struct error_stats gemm_ref =
                compare_replicated_outputs(M, t->n_out, y_quant_ref, y_gemm);
        fprintf(stderr,
                "GEMM M=%zu: %zu bit mismatches, %zu out of tolerance vs GEMV "
                "(max_abs=%.9g max_rel=%.9g max_ulps=%u)\n",
                M,
                gemm_gemv.mismatches,
                gemm_gemv.out_of_tolerance,
                gemm_gemv.max_abs,
                gemm_gemv.max_rel,
                gemm_gemv.max_ulps);
        fprintf(stderr,
                "GEMM M=%zu vs quantized ref: %zu out of tolerance "
                "(max_abs=%.9g max_rel=%.9g max_ulps=%u)\n",
                M,
                gemm_ref.out_of_tolerance,
                gemm_ref.max_abs,
                gemm_ref.max_rel,
                gemm_ref.max_ulps);
        if (gemm_gemv.out_of_tolerance != 0 || gemm_ref.out_of_tolerance != 0)
            rc = 1;
        free(x_q8_m);
        free(scales);
        free(y_gemm);
    }

    free(x);
    free(x_q8);
    free(y_fast);
    free(y_ref);
    free(y_quant_ref);
    ptqtp_close(ctx);
    return rc;
}
