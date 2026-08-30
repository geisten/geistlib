/*
 * test_step4_qkv_norm — Step 4 of Sub-Task D.
 *
 * Embedding -> input_layernorm -> {Q,K,V} proj -> {Q,K,V} norm
 * (norms are per-head RMSNorm on last dim head_dim=256).
 *
 * Outputs three FP32 binaries:
 *   <out_prefix>.q.bin   shape (seq, 8, 256)
 *   <out_prefix>.k.bin   shape (seq, 1, 256)
 *   <out_prefix>.v.bin   shape (seq, 1, 256)
 *
 * Compared in validate_step4.py against q_norm/k_norm/v_norm hooks.
 */
#include "safetensors_reader.h"
#include "gemma4_kernels.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 1536
#define Q_HEADS 8
#define KV_HEADS 1
#define HEAD_DIM 256
#define Q_OUT (Q_HEADS * HEAD_DIM)   /* 2048 */
#define KV_OUT (KV_HEADS * HEAD_DIM) /* 256 */
#define RMS_EPS 1e-6f

static int32_t *read_input_ids(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int32_t *ids = (int32_t *) malloc((size_t) sz);
    xfread(ids, 1, (size_t) sz, f);
    fclose(f);
    *n_out = (size_t) sz / 4;
    return ids;
}

static void write_bin(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    xfwrite(data, 1, bytes, f);
    fclose(f);
}

static float *load_bf16(struct st_ctx *ctx, const char *name, size_t expected) {
    const struct st_tensor_t *t = st_get(ctx, name);
    if (!t || t->dtype != ST_DTYPE_BF16) {
        fprintf(stderr, "missing/bad: %s\n", name);
        return nullptr;
    }
    size_t elems = 1;
    for (size_t i = 0; i < t->rank; i++)
        elems *= t->shape[i];
    if (elems != expected) {
        fprintf(stderr, "elem mismatch %s: got %zu expected %zu\n", name, elems, expected);
        return nullptr;
    }
    return bf16_alloc_fp32(elems, (const uint16_t *) t->data);
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 4, "<model.safetensors> <ids.bin> <out_prefix>");

    const char    *err = nullptr;
    struct st_ctx *ctx = st_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    const struct st_tensor_t *embed = st_get(ctx, "model.language_model.embed_tokens.weight");
    float *in_ln_w = load_bf16(ctx, "model.language_model.layers.0.input_layernorm.weight", HIDDEN);
    float *q_w     = load_bf16(
            ctx, "model.language_model.layers.0.self_attn.q_proj.weight", (size_t) Q_OUT * HIDDEN);
    float *k_w = load_bf16(
            ctx, "model.language_model.layers.0.self_attn.k_proj.weight", (size_t) KV_OUT * HIDDEN);
    float *v_w = load_bf16(
            ctx, "model.language_model.layers.0.self_attn.v_proj.weight", (size_t) KV_OUT * HIDDEN);
    float *q_norm_w =
            load_bf16(ctx, "model.language_model.layers.0.self_attn.q_norm.weight", HEAD_DIM);
    float *k_norm_w =
            load_bf16(ctx, "model.language_model.layers.0.self_attn.k_norm.weight", HEAD_DIM);
    /* v_norm has no weight in Gemma 4 (with_scale=False) — pass nullptr to rmsnorm. */
    if (!embed || !in_ln_w || !q_w || !k_w || !v_w || !q_norm_w || !k_norm_w) {
        fprintf(stderr, "weight load failed\n");
        st_close(ctx);
        return 1;
    }

    size_t   n_ids = 0;
    int32_t *ids   = read_input_ids(argv[2], &n_ids);

    /* Step 1+2: embedding + input_layernorm */
    const float     embed_scale = sqrtf((float) HIDDEN);
    float          *h           = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    const uint16_t *table       = (const uint16_t *) embed->data;
    for (size_t t = 0; t < n_ids; t++) {
        const uint16_t *row = table + (size_t) ids[t] * HIDDEN;
        for (size_t i = 0; i < HIDDEN; i++)
            h[t * HIDDEN + i] = bf16_to_fp32(row[i]) * embed_scale;
    }
    float *normed = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, h, in_ln_w, RMS_EPS, normed);

    /* Step 3: Q/K/V proj */
    float *q = (float *) malloc(n_ids * Q_OUT * sizeof(float));
    float *k = (float *) malloc(n_ids * KV_OUT * sizeof(float));
    float *v = (float *) malloc(n_ids * KV_OUT * sizeof(float));
    linear_fp32(n_ids, HIDDEN, Q_OUT, normed, q_w, nullptr, q);
    linear_fp32(n_ids, HIDDEN, KV_OUT, normed, k_w, nullptr, k);
    linear_fp32(n_ids, HIDDEN, KV_OUT, normed, v_w, nullptr, v);

    /* Step 4: per-head RMSNorm on last dim. Q is (n_ids, 8 heads, 256);
     * pass as (n_ids*8 rows, 256 hidden). Same idea for K/V (n_ids*1). */
    float *q_n = (float *) malloc(n_ids * Q_OUT * sizeof(float));
    float *k_n = (float *) malloc(n_ids * KV_OUT * sizeof(float));
    float *v_n = (float *) malloc(n_ids * KV_OUT * sizeof(float));
    rmsnorm_fp32(n_ids * Q_HEADS, HEAD_DIM, q, q_norm_w, RMS_EPS, q_n);
    rmsnorm_fp32(n_ids * KV_HEADS, HEAD_DIM, k, k_norm_w, RMS_EPS, k_n);
    rmsnorm_fp32(n_ids * KV_HEADS, HEAD_DIM, v, /*weight=*/nullptr, RMS_EPS, v_n);

    /* Write outputs */
    char path[1024];
    snprintf(path, sizeof(path), "%s.q.bin", argv[3]);
    write_bin(path, q_n, n_ids * Q_OUT * sizeof(float));
    snprintf(path, sizeof(path), "%s.k.bin", argv[3]);
    write_bin(path, k_n, n_ids * KV_OUT * sizeof(float));
    snprintf(path, sizeof(path), "%s.v.bin", argv[3]);
    write_bin(path, v_n, n_ids * KV_OUT * sizeof(float));
    fprintf(stderr, "wrote %s.{q,k,v}.bin (n_ids=%zu)\n", argv[3], n_ids);

    free(q_n);
    free(k_n);
    free(v_n);
    free(q);
    free(k);
    free(v);
    free(normed);
    free(h);
    free(in_ln_w);
    free(q_w);
    free(k_w);
    free(v_w);
    free(q_norm_w);
    free(k_norm_w);
    free(ids);
    st_close(ctx);
    return 0;
}
