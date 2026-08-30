/*
 * test_step2_input_layernorm — Step 2 of Sub-Task D.
 *
 * Embedding (validated by Step 1) -> input_layernorm.
 * Compares against dumps[T1]['layer_00_input_layernorm'].
 *
 * Build:
 *   cc -std=c23 -Wall -Wextra -O2 \
 *      safetensors_reader.c gemma4_kernels.c \
 *      test_step2_input_layernorm.c -o test_step2_input_layernorm
 */
#include "safetensors_reader.h"
#include "gemma4_kernels.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 1536
#define RMS_EPS 1e-6f

static int32_t *read_input_ids(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return nullptr;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % 4 != 0) {
        fclose(f);
        return nullptr;
    }
    int32_t *ids = (int32_t *) malloc((size_t) sz);
    if (fread(ids, 1, (size_t) sz, f) != (size_t) sz) {
        free(ids);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    *n_out = (size_t) sz / 4;
    return ids;
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 4, "<model.safetensors> <input_ids.bin> <out.bin>");

    const char    *err = nullptr;
    struct st_ctx *ctx = st_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "st_open: %s\n", err);
        return 1;
    }

    /* Resolve weights */
    const struct st_tensor_t *embed = st_get(ctx, "model.language_model.embed_tokens.weight");
    const struct st_tensor_t *norm =
            st_get(ctx, "model.language_model.layers.0.input_layernorm.weight");
    if (!embed || !norm) {
        fprintf(stderr, "missing weight: embed=%p norm=%p\n", (void *) embed, (void *) norm);
        st_close(ctx);
        return 1;
    }
    if (norm->dtype != ST_DTYPE_BF16 || norm->rank != 1 || norm->shape[0] != HIDDEN) {
        fprintf(stderr, "bad norm shape\n");
        st_close(ctx);
        return 1;
    }

    /* Convert norm weight BF16 -> FP32 once */
    float *norm_w = bf16_alloc_fp32(HIDDEN, (const uint16_t *) norm->data);
    if (!norm_w) {
        st_close(ctx);
        return 1;
    }

    size_t   n_ids = 0;
    int32_t *ids   = read_input_ids(argv[2], &n_ids);
    if (!ids) {
        free(norm_w);
        st_close(ctx);
        return 1;
    }
    fprintf(stderr, "n_ids=%zu\n", n_ids);

    /* Compute embedding (Step 1, in-place into hidden_states) */
    const float embed_scale   = sqrtf((float) HIDDEN);
    float      *hidden_states = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    if (!hidden_states) {
        free(norm_w);
        free(ids);
        st_close(ctx);
        return 1;
    }

    const uint16_t *table = (const uint16_t *) embed->data;
    for (size_t t = 0; t < n_ids; t++) {
        const uint16_t *row = table + (size_t) ids[t] * HIDDEN;
        float          *dst = hidden_states + t * HIDDEN;
        for (size_t i = 0; i < HIDDEN; i++)
            dst[i] = bf16_to_fp32(row[i]) * embed_scale;
    }

    /* Apply input_layernorm in-place */
    float *out = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    if (!out) {
        free(hidden_states);
        free(norm_w);
        free(ids);
        st_close(ctx);
        return 1;
    }
    rmsnorm_fp32(n_ids, HIDDEN, hidden_states, norm_w, RMS_EPS, out);

    /* Write output */
    FILE *fo = fopen(argv[3], "wb");
    if (!fo) {
        perror("fopen out");
        return 1;
    }
    xfwrite(out, sizeof(float), n_ids * HIDDEN, fo);
    fclose(fo);
    fprintf(stderr, "wrote %s (%zu × %d fp32)\n", argv[3], n_ids, HIDDEN);

    free(out);
    free(hidden_states);
    free(norm_w);
    free(ids);
    st_close(ctx);
    return 0;
}
