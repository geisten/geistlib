/*
 * test_step1_embedding_unit — Gemma 4 parity step 1 (embedding).
 *
 * Validates: token embedding lookup + Gemma4TextScaledWordEmbedding scaling.
 *
 * Loads Gemma 4 safetensors via our reader, picks out
 * `model.language_model.embed_tokens.weight` (262144, 1536) BF16.
 * For each token id in T1.npz["input_ids"]:
 *   - Reads row from embedding table (BF16 → FP32 conversion)
 *   - Multiplies by sqrt(hidden_size) = sqrt(1536)
 *   - Writes the resulting [seq_len, 1536] FP32 tensor to step1_token_embed.bin
 *
 * Companion validate_step1.py loads dumps/T1.npz["token_embed"] and the
 * binary, prints diff stats.
 *
 * Run:
 *   test_step1_embedding_unit ../gemma-4-E2B-it/model.safetensors \
 *      dumps/T1.npz_input_ids.bin step1_token_embed.bin
 * (input_ids file is produced by validate_step1.py prepass.)
 */
#include "safetensors_reader.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Convert one BF16 value (uint16 storage) to FP32. */
static inline float bf16_to_fp32(uint16_t bf) {
    uint32_t bits = (uint32_t) bf << 16; /* upper half of fp32 mantissa */
    float    f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Read a contiguous run of int32 token IDs from a flat binary file. */
static int32_t *read_input_ids(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen input_ids");
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % 4 != 0) {
        fclose(f);
        fprintf(stderr, "bad input_ids size %ld\n", sz);
        return nullptr;
    }
    int32_t *ids = (int32_t *) malloc((size_t) sz);
    if (!ids) {
        fclose(f);
        return nullptr;
    }
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

    const char    *errmsg = nullptr;
    struct st_ctx *ctx    = st_open(argv[1], &errmsg);
    if (!ctx) {
        fprintf(stderr, "st_open failed: %s\n", errmsg ? errmsg : "?");
        return 1;
    }

    const struct st_tensor_t *embed = st_get(ctx, "model.language_model.embed_tokens.weight");
    if (!embed) {
        fprintf(stderr, "embed_tokens.weight not found\n");
        st_close(ctx);
        return 1;
    }
    if (embed->dtype != ST_DTYPE_BF16 || embed->rank != 2 || embed->shape[0] != 262144 ||
        embed->shape[1] != 1536) {
        fprintf(stderr,
                "unexpected embedding shape: dtype=%s rank=%zu [%zu,%zu]\n",
                st_dtype_name(embed->dtype),
                embed->rank,
                embed->shape[0],
                embed->shape[1]);
        st_close(ctx);
        return 1;
    }
    const size_t hidden      = embed->shape[1];
    const float  embed_scale = sqrtf((float) hidden);
    fprintf(stderr, "embed_scale = sqrt(%zu) = %.6f\n", hidden, embed_scale);

    size_t   n_ids = 0;
    int32_t *ids   = read_input_ids(argv[2], &n_ids);
    if (!ids) {
        st_close(ctx);
        return 1;
    }
    fprintf(stderr, "n_ids = %zu\n", n_ids);

    /* Output: [n_ids, hidden] FP32, contiguous row-major. */
    float *out = (float *) malloc(n_ids * hidden * sizeof(float));
    if (!out) {
        free(ids);
        st_close(ctx);
        return 1;
    }

    const uint16_t *table = (const uint16_t *) embed->data;
    for (size_t t = 0; t < n_ids; t++) {
        int32_t id = ids[t];
        if (id < 0 || (size_t) id >= embed->shape[0]) {
            fprintf(stderr, "id out of range: %d\n", id);
            free(ids);
            free(out);
            st_close(ctx);
            return 1;
        }
        const uint16_t *row = table + (size_t) id * hidden;
        float          *dst = out + t * hidden;
        for (size_t i = 0; i < hidden; i++) {
            dst[i] = bf16_to_fp32(row[i]) * embed_scale;
        }
    }

    FILE *fo = fopen(argv[3], "wb");
    if (!fo) {
        perror("fopen out");
        free(ids);
        free(out);
        st_close(ctx);
        return 1;
    }
    if (fwrite(out, sizeof(float), n_ids * hidden, fo) != n_ids * hidden) {
        perror("fwrite");
        fclose(fo);
        free(ids);
        free(out);
        st_close(ctx);
        return 1;
    }
    fclose(fo);

    fprintf(stderr,
            "wrote %s (%zu × %zu fp32 = %zu bytes)\n",
            argv[3],
            n_ids,
            hidden,
            n_ids * hidden * 4);
    free(ids);
    free(out);
    st_close(ctx);
    return 0;
}
