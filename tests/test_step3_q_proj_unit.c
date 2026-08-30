/*
 * test_step3_q_proj — Step 3 of Sub-Task D.
 *
 * Embedding -> input_layernorm -> q_proj
 * Compares against dumps[T1]['layer_00_self_attn_q_proj']
 *   shape: (1, seq, 2048)  (8 heads × 256 head_dim, MQA layer)
 */
#include "safetensors_reader.h"
#include "gemma4_kernels.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 1536
#define Q_OUT 2048 /* sliding-attn: 8 heads × 256 head_dim */
#define RMS_EPS 1e-6f

static int32_t *read_input_ids(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return nullptr;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int32_t *ids = (int32_t *) malloc((size_t) sz);
    xfread(ids, 1, (size_t) sz, f);
    fclose(f);
    *n_out = (size_t) sz / 4;
    return ids;
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 4, "<model.safetensors> <ids.bin> <out.bin>");

    const char    *err = nullptr;
    struct st_ctx *ctx = st_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    const struct st_tensor_t *embed = st_get(ctx, "model.language_model.embed_tokens.weight");
    const struct st_tensor_t *in_ln =
            st_get(ctx, "model.language_model.layers.0.input_layernorm.weight");
    const struct st_tensor_t *q_w =
            st_get(ctx, "model.language_model.layers.0.self_attn.q_proj.weight");
    if (!embed || !in_ln || !q_w) {
        fprintf(stderr, "missing weight\n");
        st_close(ctx);
        return 1;
    }
    if (q_w->rank != 2 || q_w->shape[0] != Q_OUT || q_w->shape[1] != HIDDEN) {
        fprintf(stderr, "q_proj shape unexpected: [%zu,%zu]\n", q_w->shape[0], q_w->shape[1]);
        st_close(ctx);
        return 1;
    }

    float *in_ln_w = bf16_alloc_fp32(HIDDEN, (const uint16_t *) in_ln->data);
    float *q_w_f   = bf16_alloc_fp32((size_t) Q_OUT * HIDDEN, (const uint16_t *) q_w->data);

    size_t   n_ids = 0;
    int32_t *ids   = read_input_ids(argv[2], &n_ids);
    fprintf(stderr, "n_ids=%zu\n", n_ids);

    /* Step 1: embedding */
    const float     embed_scale = sqrtf((float) HIDDEN);
    float          *h           = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    const uint16_t *table       = (const uint16_t *) embed->data;
    for (size_t t = 0; t < n_ids; t++) {
        const uint16_t *row = table + (size_t) ids[t] * HIDDEN;
        for (size_t i = 0; i < HIDDEN; i++)
            h[t * HIDDEN + i] = bf16_to_fp32(row[i]) * embed_scale;
    }

    /* Step 2: input_layernorm */
    float *normed = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, h, in_ln_w, RMS_EPS, normed);

    /* Step 3: q_proj */
    float *q = (float *) malloc(n_ids * Q_OUT * sizeof(float));
    linear_fp32(n_ids, HIDDEN, Q_OUT, normed, q_w_f, /*bias=*/nullptr, q);

    FILE *fo = fopen(argv[3], "wb");
    xfwrite(q, sizeof(float), n_ids * Q_OUT, fo);
    fclose(fo);
    fprintf(stderr, "wrote %s (%zu × %d fp32)\n", argv[3], n_ids, Q_OUT);

    free(q);
    free(normed);
    free(h);
    free(q_w_f);
    free(in_ln_w);
    free(ids);
    st_close(ctx);
    return 0;
}
