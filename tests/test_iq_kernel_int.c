/*
 * test_iq_kernel_int — correctness of linear_iq2s_decode_w2a8 and
 * linear_iq3s_decode_w3a8 against the dequant + cblas_sgemv reference.
 *
 * Loads one IQ2_S and one IQ3_S tensor from a real GGUF (gemma4-e2b-IQ2_M),
 * runs the new W2A8/W3A8 NEON kernels on a random input vector, and
 * compares against the FP32 reference computed by dequant_iq{2,3}_s_row +
 * sgemv. Pass criterion: cosine similarity ≥ 0.999 (both kernels are
 * int8-quantized in x so we accept the slight rounding vs FP32 reference).
 *
 * Phase 0.3 (IQ2_M throughput unblock) — verifies the kernels are
 * functionally correct before benchmarking.
 *
 * SKIPs cleanly if the IQ2_M model is not available.
 */
#include "quant.h"
#include "gguf_reader.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward-declare cblas_sgemv (same pattern as the backend code).
 * Used only as a portable reference; the NEON path under test never
 * touches cblas. */
typedef enum CBLAS_ORDER_ { CblasRowMajor_ = 101 } CBLAS_ORDER_;
typedef enum CBLAS_TRANSPOSE_ { CblasNoTrans_ = 111, CblasTrans_ = 112 } CBLAS_TRANSPOSE_;
extern void cblas_sgemv(int /*order*/,
                        int /*TransA*/,
                        int          M,
                        int          N,
                        float        alpha,
                        const float *A,
                        int          lda,
                        const float *x,
                        int          incx,
                        float        beta,
                        float       *y,
                        int          incy);
extern void cblas_sgemm(int /*order*/,
                        int /*TransA*/,
                        int /*TransB*/,
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

static float cosine_similarity(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double) a[i] * (double) b[i];
        na += (double) a[i] * (double) a[i];
        nb += (double) b[i] * (double) b[i];
    }
    if (na == 0.0 || nb == 0.0)
        return 0.0f;
    return (float) (dot / (sqrt(na) * sqrt(nb)));
}

static int verify_kernel(const struct gguf_tensor_t *t,
                         const char                 *label,
                         void (*kernel)(const float *, const void *, size_t, size_t, float *),
                         void (*dequant_row)(const void *, float *, size_t),
                         size_t block_bytes,
                         size_t block_elems) {
    /* Tensor layout: [n_out rows, n_in cols] row-major. */
    const size_t n_in  = t->dims[0];
    const size_t n_out = t->dims[1];
    printf("=== %s: tensor %s, %zu x %zu ===\n", label, t->name, n_out, n_in);

    if (n_in % block_elems != 0) {
        fprintf(stderr, "  skip: n_in=%zu not a multiple of block_elems=%zu\n", n_in, block_elems);
        return 0;
    }

    float *x = malloc(n_in * sizeof(float));
    for (size_t i = 0; i < n_in; i++) {
        x[i] = ((float) rand() / (float) RAND_MAX - 0.5f) * 2.0f;
    }

    /* Reference: dequant the whole tensor then sgemv against x. */
    float       *w_fp32           = malloc(n_out * n_in * sizeof(float));
    const size_t n_blocks_per_row = n_in / block_elems;
    for (size_t r = 0; r < n_out; r++) {
        dequant_row((const uint8_t *) t->data + r * n_blocks_per_row * block_bytes,
                    w_fp32 + r * n_in,
                    n_in);
    }
    float *y_ref = malloc(n_out * sizeof(float));
    cblas_sgemv(CblasRowMajor_,
                CblasNoTrans_,
                (int) n_out,
                (int) n_in,
                1.0f,
                w_fp32,
                (int) n_in,
                x,
                1,
                0.0f,
                y_ref,
                1);
    free(w_fp32);

    /* Kernel under test. */
    float *y_fast = malloc(n_out * sizeof(float));
    kernel(x, t->data, n_in, n_out, y_fast);

    const float cos          = cosine_similarity(y_ref, y_fast, n_out);
    float       max_abs_diff = 0.0f;
    for (size_t i = 0; i < n_out; i++) {
        const float d = fabsf(y_ref[i] - y_fast[i]);
        if (d > max_abs_diff)
            max_abs_diff = d;
    }
    printf("  cos=%.6f  max|Δ|=%.4f  y_ref[0..3]=%.3f %.3f %.3f  y_fast[0..3]=%.3f %.3f %.3f\n",
           cos,
           max_abs_diff,
           y_ref[0],
           y_ref[1],
           y_ref[2],
           y_fast[0],
           y_fast[1],
           y_fast[2]);

    free(x);
    free(y_ref);
    free(y_fast);

    if (cos < 0.999f) {
        fprintf(stderr, "  FAIL: cos %.6f < 0.999\n", cos);
        return 1;
    }
    return 0;
}

int main(void) {
    /* Prefer the IQ2_M model so we get both IQ2_S and IQ3_S tensors. */
    const char *override_path = getenv("GEIST_GGUF_PATH");
    const char *iq2m_path     = "gguf_artifacts/gemma4-e2b-IQ2_M.gguf";
    const char *model_path    = override_path != nullptr ? override_path : iq2m_path;

    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(model_path, &err);
    if (ctx == nullptr) {
        printf("SKIP: cannot open %s: %s\n", model_path, err != nullptr ? err : "(no detail)");
        return GEIST_TEST_SKIP;
    }

    /* Find one IQ2_S and one IQ3_S tensor of typical FFN/attn size. */
    const struct gguf_tensor_t *iq2s = nullptr;
    const struct gguf_tensor_t *iq3s = nullptr;
    const size_t                nt   = gguf_tensor_count(ctx);
    for (size_t i = 0; i < nt; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(ctx, i);
        if (t == nullptr || t->n_dims != 2)
            continue;
        if (iq2s == nullptr && t->dtype == GGUF_TYPE_IQ2_S && t->dims[0] % IQ2_S_BLOCK_ELEMS == 0) {
            iq2s = t;
        }
        if (iq3s == nullptr && t->dtype == GGUF_TYPE_IQ3_S && t->dims[0] % IQ3_S_BLOCK_ELEMS == 0) {
            iq3s = t;
        }
    }

    if (iq2s == nullptr && iq3s == nullptr) {
        printf("SKIP: no IQ2_S or IQ3_S tensor found in %s\n", model_path);
        gguf_close(ctx);
        return GEIST_TEST_SKIP;
    }

    srand(42);
    int fails = 0;
    if (iq2s != nullptr) {
        fails += verify_kernel(iq2s,
                               "IQ2_S W2A8",
                               linear_iq2s_decode_w2a8,
                               dequant_iq2_s_row,
                               IQ2_S_BLOCK_BYTES,
                               IQ2_S_BLOCK_ELEMS);
    } else {
        printf("(no IQ2_S tensor in this model)\n");
    }
    if (iq3s != nullptr) {
        fails += verify_kernel(iq3s,
                               "IQ3_S W3A8",
                               linear_iq3s_decode_w3a8,
                               dequant_iq3_s_row,
                               IQ3_S_BLOCK_BYTES,
                               IQ3_S_BLOCK_ELEMS);
    } else {
        printf("(no IQ3_S tensor in this model)\n");
    }

    /* ---- M>1 prefill kernels ---- */
    const size_t m_values[] = {2, 4, 8};
    for (size_t k = 0; k < sizeof m_values / sizeof m_values[0]; k++) {
        const size_t m = m_values[k];
        if (iq2s != nullptr) {
            const size_t n_in  = iq2s->dims[0];
            const size_t n_out = iq2s->dims[1];
            const size_t nbpr  = n_in / IQ2_S_BLOCK_ELEMS;
            printf("=== IQ2_S W2A8 prefill m=%zu n_out=%zu n_in=%zu ===\n", m, n_out, n_in);
            float *x      = malloc(m * n_in * sizeof(float));
            float *y_fast = malloc(m * n_out * sizeof(float));
            float *y_ref  = malloc(m * n_out * sizeof(float));
            for (size_t i = 0; i < m * n_in; i++) {
                x[i] = ((float) rand() / (float) RAND_MAX - 0.5f) * 2.0f;
            }
            float *w_fp32 = malloc(n_out * n_in * sizeof(float));
            for (size_t r = 0; r < n_out; r++) {
                dequant_iq2_s_row((const uint8_t *) iq2s->data + r * nbpr * IQ2_S_BLOCK_BYTES,
                                  w_fp32 + r * n_in,
                                  n_in);
            }
            cblas_sgemm(CblasRowMajor_,
                        CblasNoTrans_,
                        CblasTrans_,
                        (int) m,
                        (int) n_out,
                        (int) n_in,
                        1.0f,
                        x,
                        (int) n_in,
                        w_fp32,
                        (int) n_in,
                        0.0f,
                        y_ref,
                        (int) n_out);
            free(w_fp32);
            linear_iq2s_w2a8_prefill(x, iq2s->data, m, n_in, n_out, y_fast);
            float min_cos = 1.0f;
            for (size_t i = 0; i < m; i++) {
                const float c = cosine_similarity(y_ref + i * n_out, y_fast + i * n_out, n_out);
                if (c < min_cos)
                    min_cos = c;
                if (c < 0.999f) {
                    fprintf(stderr, "  row %zu cos=%.6f < 0.999\n", i, c);
                    fails++;
                }
            }
            printf("  min row cosine=%.6f over m=%zu rows\n", min_cos, m);
            free(x);
            free(y_fast);
            free(y_ref);
        }
        if (iq3s != nullptr) {
            const size_t n_in  = iq3s->dims[0];
            const size_t n_out = iq3s->dims[1];
            const size_t nbpr  = n_in / IQ3_S_BLOCK_ELEMS;
            printf("=== IQ3_S W3A8 prefill m=%zu n_out=%zu n_in=%zu ===\n", m, n_out, n_in);
            float *x      = malloc(m * n_in * sizeof(float));
            float *y_fast = malloc(m * n_out * sizeof(float));
            float *y_ref  = malloc(m * n_out * sizeof(float));
            for (size_t i = 0; i < m * n_in; i++) {
                x[i] = ((float) rand() / (float) RAND_MAX - 0.5f) * 2.0f;
            }
            float *w_fp32 = malloc(n_out * n_in * sizeof(float));
            for (size_t r = 0; r < n_out; r++) {
                dequant_iq3_s_row((const uint8_t *) iq3s->data + r * nbpr * IQ3_S_BLOCK_BYTES,
                                  w_fp32 + r * n_in,
                                  n_in);
            }
            cblas_sgemm(CblasRowMajor_,
                        CblasNoTrans_,
                        CblasTrans_,
                        (int) m,
                        (int) n_out,
                        (int) n_in,
                        1.0f,
                        x,
                        (int) n_in,
                        w_fp32,
                        (int) n_in,
                        0.0f,
                        y_ref,
                        (int) n_out);
            free(w_fp32);
            linear_iq3s_w3a8_prefill(x, iq3s->data, m, n_in, n_out, y_fast);
            float min_cos = 1.0f;
            for (size_t i = 0; i < m; i++) {
                const float c = cosine_similarity(y_ref + i * n_out, y_fast + i * n_out, n_out);
                if (c < min_cos)
                    min_cos = c;
                if (c < 0.999f) {
                    fprintf(stderr, "  row %zu cos=%.6f < 0.999\n", i, c);
                    fails++;
                }
            }
            printf("  min row cosine=%.6f over m=%zu rows\n", min_cos, m);
            free(x);
            free(y_fast);
            free(y_ref);
        }
    }

    gguf_close(ctx);
    if (fails == 0) {
        printf("PASS: IQ2_S / IQ3_S kernels match reference within tolerance\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
