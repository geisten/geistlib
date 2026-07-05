/* Validate linear_q4k_decode_fp32 vs sgemm-on-FP32-dequant. */
#include "gguf_reader.h"
#include "quant.h"
#include "gguf_dequant.h"
#include "gemma4_kernels.h"
#include "test_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e6 + (double) ts.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 3, "<model.gguf> <tensor_name>");
    const char                 *err = nullptr;
    struct gguf_ctx            *ctx = gguf_open(argv[1], &err);
    const struct gguf_tensor_t *t   = gguf_get_tensor(ctx, argv[2]);
    if (!t || t->dtype != GGUF_TYPE_Q4_K) {
        fprintf(stderr, "tensor not Q4_K\n");
        return 1;
    }
    size_t n_in = t->dims[0], n_out = t->dims[1];
    fprintf(stderr, "Tensor %s: Q4_K (n_in=%zu, n_out=%zu)\n", argv[2], n_in, n_out);

    /* Random input */
    float *x = (float *) malloc(n_in * sizeof(float));
    for (size_t i = 0; i < n_in; i++)
        x[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;

    /* Method A: dequant whole tensor then sgemv */
    float *w_fp32 = gguf_dequant_to_fp32(t);
    float *y_a    = (float *) malloc(n_out * sizeof(float));
    int    n_iter = 5;
    double t0     = now_us();
    for (int it = 0; it < n_iter; it++) {
        linear_fp32(x, w_fp32, nullptr, 1, n_in, n_out, y_a);
    }
    double t_a = (now_us() - t0) / n_iter;
    fprintf(stderr, "Method A (sgemv on FP32): %.2f ms/call\n", t_a / 1000.0);

    /* Method B: on-the-fly Q4_K decode (FP32 NEON) */
    float *y_b = (float *) malloc(n_out * sizeof(float));
    t0         = now_us();
    for (int it = 0; it < n_iter; it++) {
        linear_q4k_decode_fp32(x, t->data, n_in, n_out, y_b);
    }
    double t_b = (now_us() - t0) / n_iter;
    fprintf(stderr, "Method B (Q4_K decode FP32): %.2f ms/call\n", t_b / 1000.0);
    fprintf(stderr, "Speedup B/A: %.2fx\n", t_a / t_b);

    /* Method C: W4A8 — int8 activations × Q4_K weights via vdotq_s32 */
    float *y_c = (float *) malloc(n_out * sizeof(float));
    t0         = now_us();
    for (int it = 0; it < n_iter; it++) {
        linear_q4k_decode_w4a8(x, t->data, n_in, n_out, y_c);
    }
    double t_c = (now_us() - t0) / n_iter;
    fprintf(stderr, "Method C (Q4_K W4A8 i8dot):  %.2f ms/call\n", t_c / 1000.0);
    fprintf(stderr, "Speedup C/A: %.2fx, C/B: %.2fx\n", t_a / t_c, t_b / t_c);

    /* Method D: predecoded Q4_K W4A8. This is the backend-owned layout
     * used by cpu_neon resolve_weight on Apple. */
    const size_t packed_bytes = q4k_predecode_size_bytes(n_in, n_out);
    void        *packed       = malloc(packed_bytes);
    if (packed == nullptr || q4k_predecode_pack(t->data, n_in, n_out, packed) != 0) {
        fprintf(stderr, "Q4_K predecode pack failed\n");
        return 1;
    }
    const size_t packed_nt_bytes = q4k_predecode_ntile4_size_bytes(n_in, n_out);
    void        *packed_nt       = malloc(packed_nt_bytes);
    if (packed_nt == nullptr || q4k_predecode_ntile4_pack(t->data, n_in, n_out, packed_nt) != 0) {
        fprintf(stderr, "Q4_K ntile4 predecode pack failed\n");
        return 1;
    }
    float *y_d = (float *) malloc(n_out * sizeof(float));
    t0         = now_us();
    for (int it = 0; it < n_iter; it++) {
        linear_q4k_decode_w4a8_predecoded(x, packed, n_in, n_out, y_d);
    }
    double t_d = (now_us() - t0) / n_iter;
    fprintf(stderr, "Method D (Q4_K predecoded W4A8): %.2f ms/call\n", t_d / 1000.0);
    fprintf(stderr, "Speedup D/C: %.2fx\n", t_c / t_d);

    const size_t m_test  = 8;
    float       *xm      = (float *) malloc(m_test * n_in * sizeof(float));
    int8_t      *xm_q8   = (int8_t *) malloc(m_test * n_in);
    int32_t     *sum32   = (int32_t *) malloc(m_test * (n_in / 32) * sizeof(int32_t));
    float       *scale_x = (float *) malloc(m_test * sizeof(float));
    float *scale_blocks  = (float *) malloc(m_test * (n_in / Q4_K_BLOCK_ELEMS) * sizeof(float));
    float *y_pd          = (float *) malloc(m_test * n_out * sizeof(float));
    float *y_mt          = (float *) malloc(m_test * n_out * sizeof(float));
    float *y_nt          = (float *) malloc(m_test * n_out * sizeof(float));
    float *y_np          = (float *) malloc(m_test * n_out * sizeof(float));
    float *y_pair0       = (float *) malloc(m_test * n_out * sizeof(float));
    float *y_pair1       = (float *) malloc(m_test * n_out * sizeof(float));
    float *y_up_single   = (float *) malloc(m_test * n_out * sizeof(float));
    float *y_bq          = (float *) malloc(m_test * n_out * sizeof(float));
    float *y_ref_m       = (float *) malloc(m_test * n_out * sizeof(float));
    if (!xm || !xm_q8 || !sum32 || !scale_x || !scale_blocks || !y_pd || !y_mt || !y_nt || !y_np ||
        !y_pair0 || !y_pair1 || !y_up_single || !y_bq || !y_ref_m) {
        fprintf(stderr, "alloc failed for mtile check\n");
        return 1;
    }
    for (size_t i = 0; i < m_test * n_in; i++) {
        xm[i] = ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < m_test; i++) {
        scale_x[i] =
                quantize_x_for_q4k(xm + i * n_in, n_in, xm_q8 + i * n_in, sum32 + i * (n_in / 32));
    }
    linear_q4k_w4a8_prefill_predecoded(xm_q8, scale_x, sum32, m_test, packed, n_in, n_out, y_pd);
    linear_q4k_w4a8_prefill_predecoded_mtile4(
            xm_q8, scale_x, sum32, m_test, packed, n_in, n_out, y_mt);
    float *y_mt8 = (float *) malloc(m_test * n_out * sizeof(float));
    if (!y_mt8) {
        fprintf(stderr, "alloc failed for y_mt8\n");
        return 1;
    }
    linear_q4k_w4a8_prefill_predecoded_mtile8(
            xm_q8, scale_x, sum32, m_test, packed, n_in, n_out, y_mt8);
    double max_mtile8 = 0.0;
    for (size_t i = 0; i < m_test * n_out; i++) {
        const double d = fabs((double) y_mt[i] - (double) y_mt8[i]);
        if (d > max_mtile8)
            max_mtile8 = d;
    }
    fprintf(stderr, "Numeric mtile4 vs mtile8 m=%zu: max|d|=%.4e\n", m_test, max_mtile8);
    if (max_mtile8 > 1e-5) {
        fprintf(stderr, "FAIL: mtile8 Q4_K output diverges from mtile4 path\n");
        return 1;
    }
    free(y_mt8);
    linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4(
            xm_q8, scale_x, sum32, m_test, packed, n_in, n_out, y_nt);
    linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed(
            xm_q8, scale_x, sum32, m_test, packed_nt, n_in, n_out, y_np);
    /* mtile8_ntile4: same packed format, wider M-tile */
    float *y_np8 = (float *) malloc(m_test * n_out * sizeof(float));
    if (!y_np8) {
        fprintf(stderr, "alloc failed for y_np8\n");
        return 1;
    }
    linear_q4k_w4a8_prefill_predecoded_mtile8_ntile4_packed(
            xm_q8, scale_x, sum32, m_test, packed_nt, n_in, n_out, y_np8);
    double max_np8 = 0.0;
    for (size_t i = 0; i < m_test * n_out; i++) {
        const double d = fabs((double) y_np[i] - (double) y_np8[i]);
        if (d > max_np8)
            max_np8 = d;
    }
    fprintf(stderr, "Numeric ntile4 vs mtile8_ntile4 m=%zu: max|d|=%.4e\n", m_test, max_np8);
    if (max_np8 > 1e-5) {
        fprintf(stderr, "FAIL: mtile8_ntile4 Q4_K output diverges from ntile4 path\n");
        return 1;
    }
    free(y_np8);
    double max_mtile = 0.0;
    double max_ntile = 0.0;
    double max_npack = 0.0;
    for (size_t i = 0; i < m_test * n_out; i++) {
        const double d = fabs((double) y_pd[i] - (double) y_mt[i]);
        if (d > max_mtile)
            max_mtile = d;
        const double dn = fabs((double) y_mt[i] - (double) y_nt[i]);
        if (dn > max_ntile)
            max_ntile = dn;
        const double dp = fabs((double) y_nt[i] - (double) y_np[i]);
        if (dp > max_npack)
            max_npack = dp;
    }
    fprintf(stderr, "Numeric predecoded vs mtile4 m=%zu: max|d|=%.4e\n", m_test, max_mtile);
    fprintf(stderr, "Numeric mtile4 vs ntile4 m=%zu: max|d|=%.4e\n", m_test, max_ntile);
    fprintf(stderr, "Numeric ntile4 vs ntile4-packed m=%zu: max|d|=%.4e\n", m_test, max_npack);
    if (max_mtile > 1e-5) {
        fprintf(stderr, "FAIL: mtile4 Q4_K output diverges from predecoded path\n");
        return 1;
    }
    if (max_ntile > 1e-5) {
        fprintf(stderr, "FAIL: ntile4 Q4_K output diverges from mtile4 path\n");
        return 1;
    }
    if (max_npack > 1e-5) {
        fprintf(stderr, "FAIL: packed ntile4 Q4_K output diverges from ntile4 path\n");
        return 1;
    }

    const char *gate_suffix = "ffn_gate.weight";
    const char *gate_pos    = strstr(argv[2], gate_suffix);
    if (gate_pos != nullptr) {
        char         up_name[256];
        const size_t prefix_len = (size_t) (gate_pos - argv[2]);
        if (prefix_len + strlen("ffn_up.weight") < sizeof(up_name)) {
            memcpy(up_name, argv[2], prefix_len);
            strcpy(up_name + prefix_len, "ffn_up.weight");
            const struct gguf_tensor_t *t_up = gguf_get_tensor(ctx, up_name);
            if (t_up != nullptr && t_up->dtype == GGUF_TYPE_Q4_K && t_up->dims[0] == n_in &&
                t_up->dims[1] == n_out) {
                const size_t packed_up_bytes = q4k_predecode_ntile4_size_bytes(n_in, n_out);
                void        *packed_up       = malloc(packed_up_bytes);
                if (packed_up == nullptr ||
                    q4k_predecode_ntile4_pack(t_up->data, n_in, n_out, packed_up) != 0) {
                    fprintf(stderr, "Q4_K pair ntile4 predecode pack failed\n");
                    return 1;
                }
                linear_q4k_w4a8_prefill_pair_predecoded_mtile4_ntile4_packed(xm_q8,
                                                                             scale_x,
                                                                             sum32,
                                                                             m_test,
                                                                             packed_nt,
                                                                             packed_up,
                                                                             n_in,
                                                                             n_out,
                                                                             y_pair0,
                                                                             y_pair1);
                linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed(
                        xm_q8, scale_x, sum32, m_test, packed_up, n_in, n_out, y_up_single);
                double max_pair0 = 0.0;
                double max_pair1 = 0.0;
                for (size_t i = 0; i < m_test * n_out; i++) {
                    const double d0 = fabs((double) y_np[i] - (double) y_pair0[i]);
                    const double d1 = fabs((double) y_up_single[i] - (double) y_pair1[i]);
                    if (d0 > max_pair0)
                        max_pair0 = d0;
                    if (d1 > max_pair1)
                        max_pair1 = d1;
                }
                fprintf(stderr,
                        "Numeric packed pair gate/up m=%zu: max|d0|=%.4e max|d1|=%.4e\n",
                        m_test,
                        max_pair0,
                        max_pair1);
                if (max_pair0 > 1e-5 || max_pair1 > 1e-5) {
                    fprintf(stderr, "FAIL: packed pair Q4_K output diverges from single paths\n");
                    return 1;
                }
                free(packed_up);
            }
        }
    }

    /* SGEMM-path parity: dequant Q4_K → fp32 + cblas_sgemm vs W4A8 NEON.
     * Tolerance is looser (cosine ≥ 0.999) because the SGEMM path keeps
     * activations fp32 while the NEON path quantizes them to int8 — the
     * outputs differ by the activation-quant error, which is bounded by
     * the per-row scale of x_q8. */
    {
        extern void  cblas_sgemm(int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 float,
                                 const float *,
                                 int,
                                 const float *,
                                 int,
                                 float,
                                 float *,
                                 int);
        extern void  dequant_q4_K_row(const void *, float *, size_t);
        const int    Cb_RowMajor = 101, Cb_NoTrans = 111, Cb_Trans = 112;
        const size_t DEQ_TILE = 32;
        float       *y_sg     = (float *) malloc(m_test * n_out * sizeof(float));
        float       *tile     = (float *) malloc(DEQ_TILE * n_in * sizeof(float));
        if (y_sg != nullptr && tile != nullptr) {
            const size_t blk_bytes = (n_in / Q4_K_BLOCK_ELEMS) * 144;
            for (size_t r0 = 0; r0 < n_out; r0 += DEQ_TILE) {
                const size_t tr = (n_out - r0 < DEQ_TILE) ? (n_out - r0) : DEQ_TILE;
                dequant_q4_K_row((const uint8_t *) t->data + r0 * blk_bytes, tile, tr * n_in);
                cblas_sgemm(Cb_RowMajor,
                            Cb_NoTrans,
                            Cb_Trans,
                            (int) m_test,
                            (int) tr,
                            (int) n_in,
                            1.0f,
                            xm,
                            (int) n_in,
                            tile,
                            (int) n_in,
                            0.0f,
                            y_sg + r0,
                            (int) n_out);
            }
            double dot = 0, na = 0, nb = 0;
            for (size_t i = 0; i < m_test * n_out; i++) {
                dot += (double) y_np[i] * y_sg[i];
                na += (double) y_np[i] * y_np[i];
                nb += (double) y_sg[i] * y_sg[i];
            }
            const double cos = dot / sqrt(na * nb);
            fprintf(stderr, "Numeric ntile4-packed vs sgemm m=%zu: cos=%.6f\n", m_test, cos);
            if (cos < 0.999) {
                fprintf(stderr, "FAIL: sgemm Q4_K output diverges (cos %.6f < 0.999)\n", cos);
                return 1;
            }
        }
        free(y_sg);
        free(tile);
    }

    for (size_t i = 0; i < m_test; i++) {
        quantize_x_for_q4k_blocks(xm + i * n_in,
                                  n_in,
                                  xm_q8 + i * n_in,
                                  sum32 + i * (n_in / 32),
                                  scale_blocks + i * (n_in / Q4_K_BLOCK_ELEMS));
        linear_q4k_decode_fp32(xm + i * n_in, t->data, n_in, n_out, y_ref_m + i * n_out);
    }
    linear_q4k_w4a8_prefill_predecoded_mtile4_bscale(
            xm_q8, scale_blocks, sum32, m_test, packed, n_in, n_out, y_bq);
    double max_bq     = 0.0;
    double sum_ref2   = 0.0;
    double sum_bq2    = 0.0;
    double sum_ref_bq = 0.0;
    for (size_t i = 0; i < m_test * n_out; i++) {
        const double d = (double) y_ref_m[i] - (double) y_bq[i];
        if (fabs(d) > max_bq)
            max_bq = fabs(d);
        sum_ref2 += (double) y_ref_m[i] * (double) y_ref_m[i];
        sum_bq2 += (double) y_bq[i] * (double) y_bq[i];
        sum_ref_bq += (double) y_ref_m[i] * (double) y_bq[i];
    }
    const double cos_bq = sum_ref_bq / sqrt(sum_ref2 * sum_bq2);
    fprintf(stderr,
            "Numeric FP32 vs block-Q8 mtile4 m=%zu: max|d|=%.4e cos=%.10f\n",
            m_test,
            max_bq,
            cos_bq);
    if (max_bq > 3.0e-2 || cos_bq < 0.99998) {
        fprintf(stderr, "FAIL: block-Q8 mtile4 Q4_K output diverges from FP32 reference\n");
        return 1;
    }

    /* Compare numerical agreement */
    double max_dab = 0, max_dac = 0, max_dbc = 0, max_dcd = 0;
    double sum_a = 0, sum_b = 0, sum_c = 0, sum_d = 0;
    double sum_aa = 0, sum_bb = 0, sum_cc = 0, sum_dd = 0;
    double sum_ab = 0, sum_ac = 0, sum_bc = 0, sum_cd = 0;
    for (size_t i = 0; i < n_out; i++) {
        float dab = y_a[i] - y_b[i];
        float dac = y_a[i] - y_c[i];
        float dbc = y_b[i] - y_c[i];
        float dcd = y_c[i] - y_d[i];
        if (fabsf(dab) > max_dab)
            max_dab = fabsf(dab);
        if (fabsf(dac) > max_dac)
            max_dac = fabsf(dac);
        if (fabsf(dbc) > max_dbc)
            max_dbc = fabsf(dbc);
        if (fabsf(dcd) > max_dcd)
            max_dcd = fabsf(dcd);
        sum_a += y_a[i];
        sum_b += y_b[i];
        sum_c += y_c[i];
        sum_d += y_d[i];
        sum_aa += (double) y_a[i] * y_a[i];
        sum_bb += (double) y_b[i] * y_b[i];
        sum_cc += (double) y_c[i] * y_c[i];
        sum_dd += (double) y_d[i] * y_d[i];
        sum_ab += (double) y_a[i] * y_b[i];
        sum_ac += (double) y_a[i] * y_c[i];
        sum_bc += (double) y_b[i] * y_c[i];
        sum_cd += (double) y_c[i] * y_d[i];
    }
    double cos_ab = sum_ab / sqrt(sum_aa * sum_bb);
    double cos_ac = sum_ac / sqrt(sum_aa * sum_cc);
    double cos_bc = sum_bc / sqrt(sum_bb * sum_cc);
    double cos_cd = sum_cd / sqrt(sum_cc * sum_dd);
    fprintf(stderr, "Numeric A vs B: max|d|=%.4e  cos=%.10f\n", max_dab, cos_ab);
    fprintf(stderr, "Numeric A vs C: max|d|=%.4e  cos=%.10f\n", max_dac, cos_ac);
    fprintf(stderr, "Numeric B vs C: max|d|=%.4e  cos=%.10f\n", max_dbc, cos_bc);
    fprintf(stderr, "Numeric C vs D: max|d|=%.4e  cos=%.10f\n", max_dcd, cos_cd);
    fprintf(stderr, "Sums: a=%.3f b=%.3f c=%.3f d=%.3f\n", sum_a, sum_b, sum_c, sum_d);
    if (max_dcd > 1e-5f || cos_cd < 0.999999f) {
        fprintf(stderr, "FAIL: predecoded Q4_K output diverges from W4A8 path\n");
        return 1;
    }

    free(x);
    free(w_fp32);
    free(y_a);
    free(y_b);
    free(y_c);
    free(y_d);
    free(xm);
    free(xm_q8);
    free(sum32);
    free(scale_x);
    free(scale_blocks);
    free(y_pd);
    free(y_mt);
    free(y_nt);
    free(y_np);
    free(y_pair0);
    free(y_pair1);
    free(y_up_single);
    free(y_bq);
    free(y_ref_m);
    free(packed);
    free(packed_nt);
    gguf_close(ctx);
    return 0;
}
