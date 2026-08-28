/*
 * bench_q4k_kernel — focused microbench for the Q4_K W4A8 hot loop.
 *
 * Loads one or more Q4_K weight tensors from a real GGUF, pre-quantizes
 * an input vector ONCE, then loops `linear_q4k_decode_w4a8_pre` enough
 * times to get stable timing (≥ 200ms total per shape).
 *
 * Reports:
 *   ms/call          — average per-call wall time
 *   GB/s effective   — Q4_K weight bytes read per call / time
 *   cyc/byte (est)   — at a fixed clock; only meaningful when wrapped in `perf stat`
 *
 * Usage:
 *   bench_q4k_kernel <gguf>                 — bench all five canonical shapes
 *   bench_q4k_kernel <gguf> <tensor_name>   — bench just one
 *
 * On Pi 5, wrap in:
 *   perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,branch-misses \
 *     -- ./bench_q4k_kernel gemma4-e2b-Q4_K_M.gguf blk.0.attn_q.weight
 *
 * Threading: single-threaded by default (set OMP_NUM_THREADS=1) so the
 * microbench measures inner-kernel efficiency, not OMP scheduling.
 */
#include "gguf_reader.h"
#include "quant.h"
#include "gguf_dequant.h"
#include "gemma4_kernels.h"
#include "test_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void bench_one(const struct gguf_tensor_t *t, const char *name) {
    const size_t n_in  = t->dims[0];
    const size_t n_out = t->dims[1];
    size_t       bytes_per_call;
    const char  *kind;
    switch (t->dtype) {
    case GGUF_TYPE_Q4_K:
        if (n_in % Q4_K_BLOCK_ELEMS != 0)
            goto skip_align;
        bytes_per_call = (n_out * n_in / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES;
        kind           = "Q4_K W4A8";
        break;
    case GGUF_TYPE_Q6_K:
        if (n_in % Q6_K_BLOCK_ELEMS != 0)
            goto skip_align;
        bytes_per_call = (n_out * n_in / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES;
        kind           = "Q6_K fp32 ";
        break;
    case GGUF_TYPE_Q4_0:
        if (n_in % Q4_0_BLOCK_ELEMS != 0)
            goto skip_align;
        bytes_per_call = (n_out * n_in / Q4_0_BLOCK_ELEMS) * Q4_0_BLOCK_BYTES;
        kind           = "Q4_0 W4A8";
        break;
    default:
        fprintf(stderr, "  %-32s SKIP (dtype %d)\n", name, t->dtype);
        return;
    }

    /* Allocate workspace. Q4_K path also needs pre-quantized x; Q6_K is FP32-in. */
    float   *x     = (float *) aligned_alloc(64, n_in * sizeof(float));
    int8_t  *x_q8  = (int8_t *) aligned_alloc(64, n_in * sizeof(int8_t));
    int32_t *sum32 = (int32_t *) aligned_alloc(64, (n_in / 32) * sizeof(int32_t));
    float   *y     = (float *) aligned_alloc(64, n_out * sizeof(float));
    if (!x || !x_q8 || !sum32 || !y) {
        fprintf(stderr, "  %-32s SKIP (alloc fail)\n", name);
        free(x);
        free(x_q8);
        free(sum32);
        free(y);
        return;
    }
    for (size_t i = 0; i < n_in; i++)
        x[i] = ((float) i * 0.0137f) - 7.3f;
    float scale_x = 0.0f;
    if (t->dtype == GGUF_TYPE_Q4_K)
        scale_x = quantize_x_for_q4k(x, n_in, x_q8, sum32);

#define CALL_KERNEL()                                                                  \
    do {                                                                               \
        if (t->dtype == GGUF_TYPE_Q4_K)                                                \
            linear_q4k_decode_w4a8_pre(x_q8, scale_x, sum32, t->data, n_in, n_out, y); \
        else if (t->dtype == GGUF_TYPE_Q4_0)                                           \
            linear_q4_0_decode_w4a8(x, t->data, n_in, n_out, y);                       \
        else                                                                           \
            linear_q6k_decode_fp32(x, t->data, n_in, n_out, y);                        \
    } while (0)

    /* Calibrate iter count so the run takes ≥ 200ms. */
    const double t_warm = now_ms();
    CALL_KERNEL();
    const double single_ms = now_ms() - t_warm;
    int          n_iter    = (int) (200.0 / (single_ms + 0.001)) + 5;
    if (n_iter > 5000)
        n_iter = 5000;
    if (n_iter < 3)
        n_iter = 3;

    /* Q4_0: also check + bench the x8 interleaved layout. */
    void *x8p = NULL;
    if (t->dtype == GGUF_TYPE_Q4_0) {
        const size_t x8b = (q4_0_x8_gemv_size_bytes(n_in, n_out) + 63) & ~(size_t) 63;
        x8p              = x8b > 0 ? aligned_alloc(64, x8b) : NULL;
        if (x8p != NULL && q4_0_x8_gemv_pack(t->data, n_in, n_out, x8p) == 0) {
            float *y_ref = (float *) aligned_alloc(64, n_out * sizeof(float));
            linear_q4_0_decode_w4a8(x, t->data, n_in, n_out, y_ref);
            linear_q4_0_decode_w4a8_x8(x, x8p, n_in, n_out, y);
            double md = 0, sc = 1e-6;
            for (size_t i = 0; i < n_out; i++) {
                const double d = fabs((double) y[i] - (double) y_ref[i]);
                if (d > md)
                    md = d;
                if (fabs((double) y_ref[i]) > sc)
                    sc = fabs((double) y_ref[i]);
            }
            printf("  %-32s [x8 parity] max rel diff %.2e\n", name, md / sc);
            free(y_ref);
            const double tx0 = now_ms();
            for (int it = 0; it < n_iter; it++)
                linear_q4_0_decode_w4a8_x8(x, x8p, n_in, n_out, y);
            const double xdt = (now_ms() - tx0) / n_iter;
            /* mN parity: int8 GEMM on x8 vs the row-major prefill kernel */
            {
                enum { MP = 8 };
                float *xm  = (float *) aligned_alloc(64, MP * n_in * sizeof(float));
                float *ym  = (float *) aligned_alloc(64, MP * n_out * sizeof(float));
                float *ymr = (float *) aligned_alloc(64, MP * n_out * sizeof(float));
                if (xm && ym && ymr) {
                    for (size_t i = 0; i < MP * n_in; i++)
                        xm[i] = ((float) (i % 977)) * 0.021f - 9.7f;
                    linear_q4_0_w4a8_prefill(xm, MP, t->data, n_in, n_out, ymr);
                    linear_q4_0_w4a8_prefill_x8(xm, MP, x8p, n_in, n_out, ym);
                    double md = 0, sc = 1e-6;
                    for (size_t i = 0; i < MP * n_out; i++) {
                        const double d = fabs((double) ym[i] - (double) ymr[i]);
                        if (d > md)
                            md = d;
                        if (fabs((double) ymr[i]) > sc)
                            sc = fabs((double) ymr[i]);
                    }
                    printf("  %-32s [x8 mN parity m=8] max rel diff %.2e\n", name, md / sc);
                }
                free(xm);
                free(ym);
                free(ymr);
            }
            printf("  %-32s [Q4_0 x8  ] n_out=%6zu n_in=%5zu %19.2f ms  %5.2f GB/s\n",
                   name,
                   n_out,
                   n_in,
                   xdt,
                   (double) bytes_per_call / (xdt * 1e6));
        }
    }

    /* Hot loop. */
    const double t0 = now_ms();
    for (int it = 0; it < n_iter; it++)
        CALL_KERNEL();
    const double dt_ms       = (now_ms() - t0) / n_iter;
    const double gbps        = (double) bytes_per_call / (dt_ms * 1e6);
    const double mb_per_call = (double) bytes_per_call / (1024.0 * 1024.0);

    printf("  %-32s [%s] n_out=%6zu n_in=%5zu %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
           name,
           kind,
           n_out,
           n_in,
           mb_per_call,
           dt_ms,
           gbps,
           n_iter);
    fflush(stdout);

    /* Anti-DCE sink. */
    if (y[0] == 0.0f && y[n_out - 1] == 0.0f)
        fprintf(stderr, "(zero output)\n");

    /* For Q4_K only: bench m=8 prefill mtile4 vs mtile8 against the
     * pre-decoded weight format, single-thread. Comparison is on the
     * same dispatch path with the same activation quant; only the
     * inner M-tile width differs. */
    if (t->dtype == GGUF_TYPE_Q4_K) {
        const size_t M_BENCH =
                (getenv("GEIST_BENCH_M") != NULL) ? (size_t) atoi(getenv("GEIST_BENCH_M")) : 8;
        const size_t n_chunks    = n_in / 32;
        const size_t pd_bytes    = q4k_predecode_size_bytes(n_in, n_out);
        const size_t pd_nt_bytes = q4k_predecode_ntile4_size_bytes(n_in, n_out);
        void        *packed      = malloc(pd_bytes);
        void        *packed_nt   = malloc(pd_nt_bytes);
        int8_t      *xm_q8       = (int8_t *) malloc(M_BENCH * n_in * sizeof(int8_t));
        int32_t     *xm_sum      = (int32_t *) malloc(M_BENCH * n_chunks * sizeof(int32_t));
        float       *xm_sx       = (float *) malloc(M_BENCH * sizeof(float));
        float       *ym4         = (float *) malloc(M_BENCH * n_out * sizeof(float));
        float       *ym8         = (float *) malloc(M_BENCH * n_out * sizeof(float));
        float       *ymn84       = (float *) malloc(M_BENCH * n_out * sizeof(float));
        int          pack_rc     = 1;
        int          pack_nt_rc  = 1;
        if (packed && packed_nt && xm_q8 && xm_sum && xm_sx && ym4 && ym8 && ymn84) {
            pack_rc    = q4k_predecode_pack(t->data, n_in, n_out, packed);
            pack_nt_rc = q4k_predecode_ntile4_pack(t->data, n_in, n_out, packed_nt);
        } else {
            fprintf(stderr,
                    "  %-32s [Q4_K m=%zu] alloc failed (packed=%p xm_q8=%p xm_sum=%p xm_sx=%p "
                    "ym4=%p ym8=%p)\n",
                    name,
                    M_BENCH,
                    (void *) packed,
                    (void *) xm_q8,
                    (void *) xm_sum,
                    (void *) xm_sx,
                    (void *) ym4,
                    (void *) ym8);
        }
        if (pack_rc != 0) {
            fprintf(stderr, "  %-32s [Q4_K m=%zu] predecode_pack rc=%d\n", name, M_BENCH, pack_rc);
        }
        if (pack_rc == 0) {
            for (size_t r = 0; r < M_BENCH; r++) {
                for (size_t i = 0; i < n_in; i++) {
                    x[i] = ((float) ((r * 13 + i) % 4096) * 0.0019f) - 3.9f;
                }
                xm_sx[r] = quantize_x_for_q4k(x, n_in, xm_q8 + r * n_in, xm_sum + r * n_chunks);
            }
            const double tw4 = now_ms();
            linear_q4k_w4a8_prefill_predecoded_mtile4(
                    xm_q8, xm_sx, xm_sum, M_BENCH, packed, n_in, n_out, ym4);
            const double s4  = now_ms() - tw4;
            int          it4 = (int) (200.0 / (s4 + 0.001)) + 3;
            if (it4 > 2000)
                it4 = 2000;
            if (it4 < 3)
                it4 = 3;
            const double t04 = now_ms();
            for (int it = 0; it < it4; it++)
                linear_q4k_w4a8_prefill_predecoded_mtile4(
                        xm_q8, xm_sx, xm_sum, M_BENCH, packed, n_in, n_out, ym4);
            const double dt4 = (now_ms() - t04) / it4;

            const double tw8 = now_ms();
            linear_q4k_w4a8_prefill_predecoded_mtile8(
                    xm_q8, xm_sx, xm_sum, M_BENCH, packed, n_in, n_out, ym8);
            const double s8  = now_ms() - tw8;
            int          it8 = (int) (200.0 / (s8 + 0.001)) + 3;
            if (it8 > 2000)
                it8 = 2000;
            if (it8 < 3)
                it8 = 3;
            const double t08 = now_ms();
            for (int it = 0; it < it8; it++)
                linear_q4k_w4a8_prefill_predecoded_mtile8(
                        xm_q8, xm_sx, xm_sum, M_BENCH, packed, n_in, n_out, ym8);
            const double dt8 = (now_ms() - t08) / it8;
            (void) (dt4 / dt8); /* legacy speedup (mt4 → mt8) — see new columns below */

            /* SGEMM path: dequant Q4_K → fp32 + cblas_sgemm per tile. */
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
            double       dt_sg    = 0.0;
            float       *tile     = (float *) malloc(DEQ_TILE * n_in * sizeof(float));
            float       *x_fp32   = (float *) malloc(M_BENCH * n_in * sizeof(float));
            if (tile && x_fp32) {
                for (size_t i = 0; i < M_BENCH * n_in; i++) {
                    x_fp32[i] = ((float) ((i * 17) % 4096) * 0.0019f) - 3.9f;
                }
                const size_t blk_bytes = (n_in / Q4_K_BLOCK_ELEMS) * 144; /* Q4_K_BLOCK_BYTES */
                /* warm */
                for (size_t r0 = 0; r0 < n_out; r0 += DEQ_TILE) {
                    const size_t tr = (n_out - r0 < DEQ_TILE) ? (n_out - r0) : DEQ_TILE;
                    dequant_q4_K_row((const uint8_t *) t->data + r0 * blk_bytes, tile, tr * n_in);
                    cblas_sgemm(Cb_RowMajor,
                                Cb_NoTrans,
                                Cb_Trans,
                                (int) M_BENCH,
                                (int) tr,
                                (int) n_in,
                                1.0f,
                                x_fp32,
                                (int) n_in,
                                tile,
                                (int) n_in,
                                0.0f,
                                ym8 + r0,
                                (int) n_out);
                }
                const double tw_sg = now_ms();
                for (size_t r0 = 0; r0 < n_out; r0 += DEQ_TILE) {
                    const size_t tr = (n_out - r0 < DEQ_TILE) ? (n_out - r0) : DEQ_TILE;
                    dequant_q4_K_row((const uint8_t *) t->data + r0 * blk_bytes, tile, tr * n_in);
                    cblas_sgemm(Cb_RowMajor,
                                Cb_NoTrans,
                                Cb_Trans,
                                (int) M_BENCH,
                                (int) tr,
                                (int) n_in,
                                1.0f,
                                x_fp32,
                                (int) n_in,
                                tile,
                                (int) n_in,
                                0.0f,
                                ym8 + r0,
                                (int) n_out);
                }
                const double s_sg  = now_ms() - tw_sg;
                int          it_sg = (int) (200.0 / (s_sg + 0.001)) + 3;
                if (it_sg > 1000)
                    it_sg = 1000;
                if (it_sg < 3)
                    it_sg = 3;
                const double t0_sg = now_ms();
                for (int it = 0; it < it_sg; it++) {
                    for (size_t r0 = 0; r0 < n_out; r0 += DEQ_TILE) {
                        const size_t tr = (n_out - r0 < DEQ_TILE) ? (n_out - r0) : DEQ_TILE;
                        dequant_q4_K_row(
                                (const uint8_t *) t->data + r0 * blk_bytes, tile, tr * n_in);
                        cblas_sgemm(Cb_RowMajor,
                                    Cb_NoTrans,
                                    Cb_Trans,
                                    (int) M_BENCH,
                                    (int) tr,
                                    (int) n_in,
                                    1.0f,
                                    x_fp32,
                                    (int) n_in,
                                    tile,
                                    (int) n_in,
                                    0.0f,
                                    ym8 + r0,
                                    (int) n_out);
                    }
                }
                dt_sg = (now_ms() - t0_sg) / it_sg;
            }
            free(tile);
            free(x_fp32);
            printf("  %-32s [Q4_K m=%zu sgemm path] dt_sgemm=%6.2f ms (vs mt8=%5.2f → "
                   "Δ=%+5.1f%%)\n",
                   name,
                   M_BENCH,
                   dt_sg,
                   dt8,
                   dt_sg > 0 ? (dt8 / dt_sg - 1.0) * 100.0 : 0.0);
            fflush(stdout);

            /* mtile4_ntile4_packed on the ntile4 packed format. */
            double dt44n = 0.0;
            if (pack_nt_rc == 0) {
                const double tw44n = now_ms();
                linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed(
                        xm_q8, xm_sx, xm_sum, M_BENCH, packed_nt, n_in, n_out, ymn84);
                const double s44n  = now_ms() - tw44n;
                int          it44n = (int) (200.0 / (s44n + 0.001)) + 3;
                if (it44n > 2000)
                    it44n = 2000;
                if (it44n < 3)
                    it44n = 3;
                const double t044n = now_ms();
                for (int it = 0; it < it44n; it++)
                    linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed(
                            xm_q8, xm_sx, xm_sum, M_BENCH, packed_nt, n_in, n_out, ymn84);
                dt44n = (now_ms() - t044n) / it44n;
            }
            /* mtile8_ntile4_packed on the ntile4 packed format. */
            double dt84 = 0.0;
            if (pack_nt_rc == 0) {
                const double tw84 = now_ms();
                linear_q4k_w4a8_prefill_predecoded_mtile8_ntile4_packed(
                        xm_q8, xm_sx, xm_sum, M_BENCH, packed_nt, n_in, n_out, ymn84);
                const double s84  = now_ms() - tw84;
                int          it84 = (int) (200.0 / (s84 + 0.001)) + 3;
                if (it84 > 2000)
                    it84 = 2000;
                if (it84 < 3)
                    it84 = 3;
                const double t084 = now_ms();
                for (int it = 0; it < it84; it++)
                    linear_q4k_w4a8_prefill_predecoded_mtile8_ntile4_packed(
                            xm_q8, xm_sx, xm_sum, M_BENCH, packed_nt, n_in, n_out, ymn84);
                dt84 = (now_ms() - t084) / it84;
            }
            const double speedup84       = (dt84 > 0) ? dt44n / dt84 : 0.0;
            const double speedup8_vs_44n = (dt84 > 0) ? dt44n / dt8 : 0.0;
            printf("  %-32s [Q4_K m=%zu] mt4=%5.2f mt4_nt4p=%5.2f mt8=%5.2f mt8_nt4p=%5.2f ms | "
                   "mt8 vs mt4_nt4p Δ=%+5.1f%% | mt8_nt4p vs mt4_nt4p Δ=%+5.1f%%\n",
                   name,
                   M_BENCH,
                   dt4,
                   dt44n,
                   dt8,
                   dt84,
                   (speedup8_vs_44n - 1.0) * 100.0,
                   (speedup84 - 1.0) * 100.0);
            fflush(stdout);
        }
        free(packed);
        free(packed_nt);
        free(xm_q8);
        free(xm_sum);
        free(xm_sx);
        free(ym4);
        free(ym8);
        free(ymn84);
    }

    /* For Q6_K only: also bench (a) the W6A8 NEON variant and (b) the
     * dequant→FP32 sgemv path used today for lm_head. Verify W6A8 matches
     * the FP32 reference numerically (cosine similarity ≥ 0.999). */
    if (t->dtype == GGUF_TYPE_Q6_K) {
        /* Reference output from FP32 path stored in `y` from the calibration
         * runs above. Make a copy then re-compute via W6A8 to compare. */
        float *y_ref = (float *) aligned_alloc(64, n_out * sizeof(float));
        memcpy(y_ref, y, n_out * sizeof(float));

        /* W6A8: needs symmetric int8 quantization of x (no sum32 needed). */
        const float scale_x6 = quantize_x_int8_sym(x, n_in, x_q8);
        linear_q6k_decode_w6a8_pre(x_q8, scale_x6, t->data, n_in, n_out, y);

        /* Cosine similarity vs reference. */
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < n_out; i++) {
            dot += (double) y_ref[i] * y[i];
            na += (double) y_ref[i] * y_ref[i];
            nb += (double) y[i] * y[i];
        }
        const double cos_sim = dot / sqrt(na * nb);

        /* Bench W6A8 hot loop. */
        const double tw = now_ms();
        linear_q6k_decode_w6a8_pre(x_q8, scale_x6, t->data, n_in, n_out, y);
        const double single_ms3 = now_ms() - tw;
        int          n_iter3    = (int) (200.0 / (single_ms3 + 0.001)) + 5;
        if (n_iter3 > 5000)
            n_iter3 = 5000;
        if (n_iter3 < 3)
            n_iter3 = 3;
        const double t03 = now_ms();
        for (int it = 0; it < n_iter3; it++)
            linear_q6k_decode_w6a8_pre(x_q8, scale_x6, t->data, n_in, n_out, y);
        const double dt_ms3 = (now_ms() - t03) / n_iter3;
        const double gbps3  = (double) bytes_per_call / (dt_ms3 * 1e6);
        printf("  %-32s [Q6_K W6A8] (same shape)         %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)  "
               "cos=%.6f\n",
               name,
               (double) bytes_per_call / (1024.0 * 1024.0),
               dt_ms3,
               gbps3,
               n_iter3,
               cos_sim);
        fflush(stdout);

        free(y_ref);

        /* FP32 sgemv comparison only worth it for the lm_head size. */
        if (n_out * n_in >= 1024 * 1024) {
            float *w_fp32 = gguf_dequant_to_fp32(t);
            if (w_fp32) {
                const double t_warm2 = now_ms();
                linear_fp32(x, w_fp32, nullptr, 1, n_in, n_out, y);
                const double single_ms2 = now_ms() - t_warm2;
                int          n_iter2    = (int) (200.0 / (single_ms2 + 0.001)) + 5;
                if (n_iter2 > 5000)
                    n_iter2 = 5000;
                if (n_iter2 < 3)
                    n_iter2 = 3;
                const double t02 = now_ms();
                for (int it = 0; it < n_iter2; it++)
                    linear_fp32(x, w_fp32, nullptr, 1, n_in, n_out, y);
                const double dt_ms2 = (now_ms() - t02) / n_iter2;
                const size_t bytes2 = n_out * n_in * sizeof(float);
                const double gbps2  = (double) bytes2 / (dt_ms2 * 1e6);
                const double mb2    = (double) bytes2 / (1024.0 * 1024.0);
                printf("  %-32s [FP32 sgemv] (same shape)        %7.1f MB  %7.2f ms  %5.2f GB/s  "
                       "(%d it)\n",
                       name,
                       mb2,
                       dt_ms2,
                       gbps2,
                       n_iter2);
                fflush(stdout);
                free(w_fp32);
            }
        }
    }

    free(x);
    free(x_q8);
    free(sum32);
    free(y);
    return;

skip_align:
    fprintf(stderr, "  %-32s SKIP (n_in %zu not block-aligned)\n", name, n_in);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <gguf> [<tensor_name>]\n", argv[0]);
        return 2;
    }
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "gguf_open: %s\n", err);
        return 1;
    }

    printf("bench_q4k_kernel — single-thread W4A8 throughput on %s\n", argv[1]);
    printf("  shape                            (n_out × n_in)         size   wall            "
           "bandwidth\n");

    if (argc == 3) {
        const struct gguf_tensor_t *t = gguf_get_tensor(ctx, argv[2]);
        if (!t) {
            fprintf(stderr, "tensor not found: %s\n", argv[2]);
            return 1;
        }
        bench_one(t, argv[2]);
    } else {
        /* Canonical Gemma 4 E2B per-decode-call shapes (layer 0 representative)
         * + lm_head (token_embd is tied; Q6_K in Q4_K_M model). */
        static const char *shapes[] = {
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_output.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_up.weight",
                "blk.0.ffn_down.weight",
                "token_embd.weight", /* aliased as lm_head — DOMINANT decode cost */
                nullptr,
        };
        for (int i = 0; shapes[i]; i++) {
            const struct gguf_tensor_t *t = gguf_get_tensor(ctx, shapes[i]);
            if (!t) {
                fprintf(stderr, "  %-32s NOT FOUND\n", shapes[i]);
                continue;
            }
            bench_one(t, shapes[i]);
        }
    }

    gguf_close(ctx);
    return 0;
}
