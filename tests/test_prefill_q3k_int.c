/* Sweep all Q3_K tensors and check linear_q3k_w3a8_prefill matches FP32 ref. */
#include "quant.h"
#include "gguf_dequant.h"
#include "gguf_reader.h"
#include "test_helpers.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum CBLAS_ORDER { CblasRowMajor = 101 } CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 } CBLAS_TRANSPOSE;
extern void cblas_sgemm(CBLAS_ORDER,
                        CBLAS_TRANSPOSE,
                        CBLAS_TRANSPOSE,
                        int          M,
                        int          N,
                        int          K,
                        float        alpha,
                        const float *A,
                        int          lda,
                        const float *B,
                        int          ldb,
                        float        beta,
                        float       *C,
                        int          ldc);

int main(int argc, char **argv) {
    const char      *model = geist_test_gguf_arg(argc, argv, 1);
    const char      *err   = nullptr;
    struct gguf_ctx *ctx   = gguf_open(model, &err);
    if (!ctx) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    int n_tested = 0, n_bad = 0;
    for (size_t ti = 0; ti < gguf_tensor_count(ctx); ti++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(ctx, ti);
        if (t->dtype != GGUF_TYPE_Q3_K || t->n_dims != 2)
            continue;
        if (t->dims[0] % Q3_K_BLOCK_ELEMS != 0)
            continue;

        const size_t nin = (size_t) t->dims[0], nout = (size_t) t->dims[1];
        const size_t M = 22;
        srand(42 + (int) ti);
        float *xx = (float *) malloc(M * nin * sizeof(float));
        for (size_t i = 0; i < M * nin; i++)
            xx[i] = ((float) rand() / (float) RAND_MAX - 0.5f) * 2.0f;

        float *W_fp32 = gguf_dequant_to_fp32(t);
        float *yref   = (float *) calloc(M * nout, sizeof(float));
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    (int) M,
                    (int) nout,
                    (int) nin,
                    1.0f,
                    xx,
                    (int) nin,
                    W_fp32,
                    (int) nin,
                    0.0f,
                    yref,
                    (int) nout);

        float *yfast = (float *) calloc(M * nout, sizeof(float));
        linear_q3k_w3a8_prefill(M, nin, nout, xx, t->data, yfast);

        double sse = 0, refsq = 0, fastsq = 0, dotrf = 0;
        for (size_t i = 0; i < M * nout; i++) {
            double e = (double) yref[i] - (double) yfast[i];
            sse += e * e;
            refsq += (double) yref[i] * yref[i];
            fastsq += (double) yfast[i] * yfast[i];
            dotrf += (double) yref[i] * yfast[i];
        }
        double rmse    = sqrt(sse / ((double) M * (double) nout));
        double rms_ref = sqrt(refsq / ((double) M * (double) nout));
        double cos_sim = dotrf / (sqrt(refsq) * sqrt(fastsq));
        double rel     = 100.0 * rmse / rms_ref;
        printf("%-40s  (%5zu, %5zu) rel=%.3f%%  cos=%.6f%s\n",
               t->name,
               nin,
               nout,
               rel,
               cos_sim,
               (cos_sim < 0.9999) ? "  <-- BAD" : "");
        if (cos_sim < 0.9999)
            n_bad++;
        n_tested++;
        free(xx);
        free(W_fp32);
        free(yref);
        free(yfast);
    }
    printf("Tested %d Q3_K tensors, %d bad (cos < 0.9999)\n", n_tested, n_bad);
    if (n_tested == 0) {
        /* Not a pass. A model without a single Q3_K tensor (every Q4_K_M
         * build, including the CI reference) means this gate ran green
         * while checking nothing — the failure mode a green checkmark is
         * least likely to reveal. */
        GEIST_SKIP("no Q3_K tensor in this model — nothing to check");
    }
    gguf_close(ctx);
    return n_bad == 0 ? 0 : 1;
}
