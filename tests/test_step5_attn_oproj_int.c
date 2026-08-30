/*
 * test_step5_attn_oproj — Steps 5+6+7 of Sub-Task D combined.
 *
 * Embedding -> input_layernorm -> Q/K/V proj -> Q/K/V norm
 *           -> RoPE on Q/K -> sliding-window-512 causal attention (MQA)
 *           -> O proj
 * Compares against dumps[T*]['layer_00_self_attn_o_proj'] (1, seq, 1536).
 *
 * Bigger step than the previous ones because there are no intermediate
 * hooks between v_norm and o_proj — RoPE + attention + o_proj must be
 * validated together. If this fails, drop in a RoPE hook in
 * dump_activations.py to localize.
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
#define Q_OUT (Q_HEADS * HEAD_DIM)
#define KV_OUT (KV_HEADS * HEAD_DIM)
#define SLIDING_WINDOW 512
#define ROPE_THETA 10000.0f
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
    GEIST_REQUIRE_ARGS(argc, 4, "<model.safetensors> <ids.bin> <out.bin>");

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
    float *o_w = load_bf16(
            ctx, "model.language_model.layers.0.self_attn.o_proj.weight", (size_t) HIDDEN * Q_OUT);
    if (!embed || !in_ln_w || !q_w || !k_w || !v_w || !q_norm_w || !k_norm_w || !o_w) {
        fprintf(stderr, "weight load failed\n");
        st_close(ctx);
        return 1;
    }

    size_t   n_ids = 0;
    int32_t *ids   = read_input_ids(argv[2], &n_ids);

    /* embedding + input_layernorm */
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

    /* Q/K/V proj */
    float *q = (float *) malloc(n_ids * Q_OUT * sizeof(float));
    float *k = (float *) malloc(n_ids * KV_OUT * sizeof(float));
    float *v = (float *) malloc(n_ids * KV_OUT * sizeof(float));
    linear_fp32(n_ids, HIDDEN, Q_OUT, normed, q_w, nullptr, q);
    linear_fp32(n_ids, HIDDEN, KV_OUT, normed, k_w, nullptr, k);
    linear_fp32(n_ids, HIDDEN, KV_OUT, normed, v_w, nullptr, v);

    /* Per-head Q/K/V norm (in-place is fine since rmsnorm supports y == x).
     * v has no norm weight (with_scale=False -> just the rms division). */
    rmsnorm_fp32(n_ids * Q_HEADS, HEAD_DIM, q, q_norm_w, RMS_EPS, q);
    rmsnorm_fp32(n_ids * KV_HEADS, HEAD_DIM, k, k_norm_w, RMS_EPS, k);
    rmsnorm_fp32(n_ids * KV_HEADS, HEAD_DIM, v, /*weight=*/nullptr, RMS_EPS, v);

    /* RoPE on Q and K (V is not rotated). */
    float *cos_buf = (float *) malloc(n_ids * HEAD_DIM * sizeof(float));
    float *sin_buf = (float *) malloc(n_ids * HEAD_DIM * sizeof(float));
    rope_compute(n_ids, HEAD_DIM, HEAD_DIM, ROPE_THETA, cos_buf, sin_buf);
    rope_apply(n_ids, Q_HEADS, HEAD_DIM, q, cos_buf, sin_buf);
    rope_apply(n_ids, KV_HEADS, HEAD_DIM, k, cos_buf, sin_buf);

    /* Attention with MQA broadcast and sliding-window-512 causal mask. */
    float *attn_out = (float *) malloc(n_ids * Q_OUT * sizeof(float));
    attention_mqa_causal(n_ids, Q_HEADS, KV_HEADS, HEAD_DIM, SLIDING_WINDOW, q, k, v, attn_out);

    /* O projection */
    float *out = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    linear_fp32(n_ids, Q_OUT, HIDDEN, attn_out, o_w, nullptr, out);

    FILE *fo = fopen(argv[3], "wb");
    xfwrite(out, sizeof(float), n_ids * HIDDEN, fo);
    fclose(fo);
    fprintf(stderr, "wrote %s (%zu × %d fp32)\n", argv[3], n_ids, HIDDEN);

    free(out);
    free(attn_out);
    free(cos_buf);
    free(sin_buf);
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
    free(o_w);
    free(ids);
    st_close(ctx);
    return 0;
}
