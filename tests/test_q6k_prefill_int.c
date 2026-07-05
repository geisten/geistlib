/*
 * test_q6k_prefill_int — correctness of linear_q6k_w6a8_prefill against
 * the FP32 dequant + sgemv reference, for M ∈ {2, 4, 8}.
 *
 * Loads a real Q6_K weight tensor (typically lm_head) from a GGUF model,
 * generates M random input vectors, and compares the new M>1 NEON kernel
 * to the FP32 reference path (dequant_q6_K_row + cblas_sgemv per row).
 *
 * Pass criterion: cosine similarity ≥ 0.999 for every output row of every
 * M (int8-quantized x is the only material noise source vs FP32).
 *
 * Speculative-decoding verify pass uses this kernel at M=K (typically
 * K=4-8) on lm_head, replacing the slow FP32 dequant + sgemm fallback.
 *
 * SKIPs cleanly if no GGUF model is reachable.
 */
#include "quant.h"
#include "gguf_reader.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum CBLAS_ORDER_ { CblasRowMajor_ = 101 } CBLAS_ORDER_;
typedef enum CBLAS_TRANSPOSE_ { CblasNoTrans_ = 111, CblasTrans_ = 112 } CBLAS_TRANSPOSE_;
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

static float cosine_sim(const float *a, const float *b, size_t n) {
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

static int verify_q6k_prefill(const struct gguf_tensor_t *t, size_t m) {
    const size_t n_in             = t->dims[0];
    const size_t n_out            = t->dims[1];
    const size_t n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;
    if (n_in % Q6_K_BLOCK_ELEMS != 0) {
        printf("  skip: n_in=%zu not multiple of %zu\n", n_in, Q6_K_BLOCK_ELEMS);
        return 0;
    }
    printf("=== Q6_K m=%zu n_out=%zu n_in=%zu (%s) ===\n", m, n_out, n_in, t->name);

    float *x      = malloc(m * n_in * sizeof(float));
    float *y_fast = malloc(m * n_out * sizeof(float));
    float *y_ref  = malloc(m * n_out * sizeof(float));
    for (size_t i = 0; i < m * n_in; i++) {
        x[i] = ((float) rand() / (float) RAND_MAX - 0.5f) * 2.0f;
    }

    /* Reference: dequant whole weight then sgemm M × n_in × n_out. */
    float *w_fp32 = malloc(n_out * n_in * sizeof(float));
    for (size_t r = 0; r < n_out; r++) {
        dequant_q6_K_row((const uint8_t *) t->data + r * n_blocks_per_row * Q6_K_BLOCK_BYTES,
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

    /* Kernel under test. */
    linear_q6k_w6a8_prefill(x, t->data, m, n_in, n_out, y_fast);

    /* Per-row cosine sim. Each row is one of the M activations. */
    int   fails   = 0;
    float min_cos = 1.0f;
    for (size_t i = 0; i < m; i++) {
        const float cos = cosine_sim(y_ref + i * n_out, y_fast + i * n_out, n_out);
        if (cos < min_cos)
            min_cos = cos;
        if (cos < 0.999f) {
            fprintf(stderr, "  row %zu cos=%.6f < 0.999\n", i, cos);
            fails++;
        }
    }
    printf("  min row cosine=%.6f over m=%zu rows\n", min_cos, m);

    if (m == 2 && n_out % 8 == 0) {
        const size_t x8_bytes = q6k_x8_gemv_size_bytes(n_in, n_out);
        void        *x8       = x8_bytes > 0 ? malloc(x8_bytes) : nullptr;
        if (x8 == nullptr || q6k_x8_gemv_pack(t->data, n_in, n_out, x8) != 0) {
            fprintf(stderr, "  Q6_K x8 gemv pack failed\n");
            fails++;
        } else {
            memset(y_fast, 0, n_out * sizeof(float));
            linear_q6k_decode_w6a8_x8(x, x8, n_in, n_out, y_fast);
            const float cos = cosine_sim(y_ref, y_fast, n_out);
            if (cos < 0.999f) {
                fprintf(stderr, "  x8 gemv cos=%.6f < 0.999\n", cos);
                fails++;
            }
            printf("  x8 gemv cosine=%.6f\n", cos);
        }
        free(x8);
    }

    const size_t packed_bytes = q6k_predecode_ntile4_size_bytes(n_in, n_out);
    void        *packed       = packed_bytes > 0 ? malloc(packed_bytes) : nullptr;
    if (packed == nullptr || q6k_predecode_ntile4_pack(t->data, n_in, n_out, packed) != 0) {
        fprintf(stderr, "  Q6_K ntile4 pack failed\n");
        fails++;
    } else {
        int8_t *x_q8    = malloc(m * n_in);
        float  *scale_x = malloc(m * sizeof(float));
        if (x_q8 == nullptr || scale_x == nullptr) {
            fprintf(stderr, "  Q6_K ntile4 scratch alloc failed\n");
            fails++;
        } else {
            for (size_t i = 0; i < m; i++) {
                scale_x[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);
            }
            memset(y_fast, 0, m * n_out * sizeof(float));
            linear_q6k_w6a8_prefill_predecoded_ntile4(
                    x_q8, scale_x, m, packed, n_in, n_out, y_fast);
            float min_cos_packed = 1.0f;
            for (size_t i = 0; i < m; i++) {
                const float cos = cosine_sim(y_ref + i * n_out, y_fast + i * n_out, n_out);
                if (cos < min_cos_packed)
                    min_cos_packed = cos;
                if (cos < 0.999f) {
                    fprintf(stderr, "  ntile4 row %zu cos=%.6f < 0.999\n", i, cos);
                    fails++;
                }
            }
            printf("  ntile4 min row cosine=%.6f over m=%zu rows\n", min_cos_packed, m);
        }
        free(x_q8);
        free(scale_x);
    }
    free(packed);

    const size_t stream_bytes = q6k_predecode_ntile4_stream_size_bytes(n_in, n_out);
    void        *stream       = stream_bytes > 0 ? malloc(stream_bytes) : nullptr;
    if (stream == nullptr || q6k_predecode_ntile4_stream_pack(t->data, n_in, n_out, stream) != 0) {
        fprintf(stderr, "  Q6_K ntile4 stream pack failed\n");
        fails++;
    } else {
        int8_t *x_q8    = malloc(m * n_in);
        float  *scale_x = malloc(m * sizeof(float));
        if (x_q8 == nullptr || scale_x == nullptr) {
            fprintf(stderr, "  Q6_K ntile4 stream scratch alloc failed\n");
            fails++;
        } else {
            for (size_t i = 0; i < m; i++) {
                scale_x[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);
            }
            memset(y_fast, 0, m * n_out * sizeof(float));
            linear_q6k_w6a8_prefill_predecoded_ntile4_stream(
                    x_q8, scale_x, m, stream, n_in, n_out, y_fast);
            float min_cos_packed = 1.0f;
            for (size_t i = 0; i < m; i++) {
                const float cos = cosine_sim(y_ref + i * n_out, y_fast + i * n_out, n_out);
                if (cos < min_cos_packed)
                    min_cos_packed = cos;
                if (cos < 0.999f) {
                    fprintf(stderr, "  ntile4 stream row %zu cos=%.6f < 0.999\n", i, cos);
                    fails++;
                }
            }
            printf("  ntile4 stream min row cosine=%.6f over m=%zu rows\n", min_cos_packed, m);
        }
        free(x_q8);
        free(scale_x);
    }
    free(stream);

    free(x);
    free(y_fast);
    free(y_ref);
    return fails;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(model_path, &err);
    if (ctx == nullptr) {
        printf("SKIP: gguf_open(%s): %s\n", model_path, err ? err : "(no detail)");
        return GEIST_TEST_SKIP;
    }

    /* Pick the FIRST Q6_K tensor with at least 1 super-block per row.
     * lm_head is large (262144 rows on Gemma 4) — restrict to smaller
     * tensors to keep the test fast; fall back to anything Q6_K. */
    const struct gguf_tensor_t *q6k_small = nullptr;
    const struct gguf_tensor_t *q6k_any   = nullptr;
    const size_t                nt        = gguf_tensor_count(ctx);
    for (size_t i = 0; i < nt; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(ctx, i);
        if (t == nullptr || t->dtype != GGUF_TYPE_Q6_K || t->n_dims != 2)
            continue;
        if (t->dims[0] % Q6_K_BLOCK_ELEMS != 0)
            continue;
        if (q6k_any == nullptr)
            q6k_any = t;
        if (q6k_small == nullptr && t->dims[0] <= 2048 && t->dims[1] <= 4096) {
            q6k_small = t;
        }
    }
    const struct gguf_tensor_t *t = q6k_small ? q6k_small : q6k_any;
    if (t == nullptr) {
        printf("SKIP: no 2-D Q6_K tensor in %s\n", model_path);
        gguf_close(ctx);
        return GEIST_TEST_SKIP;
    }

    srand(42);
    int          fails      = 0;
    const size_t m_values[] = {2, 4, 8};
    for (size_t k = 0; k < sizeof m_values / sizeof m_values[0]; k++) {
        fails += verify_q6k_prefill(t, m_values[k]);
    }
    gguf_close(ctx);

    if (fails == 0) {
        printf("PASS: Q6_K M>1 W6A8 prefill matches FP32 reference within tolerance\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
