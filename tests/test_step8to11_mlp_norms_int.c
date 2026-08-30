/*
 * test_step8to11_mlp_norms — Steps 8/9/10/11 of Sub-Task D combined.
 *
 * Continues from o_proj output (step 7) through:
 *   8. post_attention_layernorm
 *   ↳ first residual addition (input → post_attn_norm output)
 *   9. pre_feedforward_layernorm
 *  10. MLP: gate_proj, up_proj, gelu-tanh, multiply, down_proj
 *  11. post_feedforward_layernorm
 *
 * Outputs five FP32 binaries, one per validation hook:
 *   <prefix>.post_attn_norm.bin
 *   <prefix>.pre_ff_norm.bin
 *   <prefix>.gate_proj.bin
 *   <prefix>.up_proj.bin
 *   <prefix>.down_proj.bin
 *   <prefix>.post_ff_norm.bin
 */
#include "safetensors_reader.h"
#include "gemma4_kernels.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 1536
#define INTER 6144 /* intermediate_size for layer 0 (not double-wide) */
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
static void write_bin(const char *path, const void *d, size_t b) {
    FILE *f = fopen(path, "wb");
    xfwrite(d, 1, b, f);
    fclose(f);
}
static float *load_bf16(struct st_ctx *ctx, const char *name, size_t expected) {
    const struct st_tensor_t *t = st_get(ctx, name);
    if (!t || t->dtype != ST_DTYPE_BF16) {
        fprintf(stderr, "miss/bad: %s\n", name);
        return nullptr;
    }
    size_t elems = 1;
    for (size_t i = 0; i < t->rank; i++)
        elems *= t->shape[i];
    if (elems != expected) {
        fprintf(stderr, "elem mismatch %s: %zu vs %zu\n", name, elems, expected);
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

    /* Load all weights up-front. */
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
    float *post_attn_w =
            load_bf16(ctx, "model.language_model.layers.0.post_attention_layernorm.weight", HIDDEN);
    float *pre_ff_w = load_bf16(
            ctx, "model.language_model.layers.0.pre_feedforward_layernorm.weight", HIDDEN);
    float *gate_w = load_bf16(
            ctx, "model.language_model.layers.0.mlp.gate_proj.weight", (size_t) INTER * HIDDEN);
    float *up_w = load_bf16(
            ctx, "model.language_model.layers.0.mlp.up_proj.weight", (size_t) INTER * HIDDEN);
    float *down_w = load_bf16(
            ctx, "model.language_model.layers.0.mlp.down_proj.weight", (size_t) HIDDEN * INTER);
    float *post_ff_w = load_bf16(
            ctx, "model.language_model.layers.0.post_feedforward_layernorm.weight", HIDDEN);
    if (!embed || !in_ln_w || !q_w || !k_w || !v_w || !q_norm_w || !k_norm_w || !o_w ||
        !post_attn_w || !pre_ff_w || !gate_w || !up_w || !down_w || !post_ff_w) {
        fprintf(stderr, "weight load failed\n");
        return 1;
    }

    size_t   n_ids = 0;
    int32_t *ids   = read_input_ids(argv[2], &n_ids);
    fprintf(stderr, "n_ids=%zu\n", n_ids);

    /* === Steps 1–7 (validated) === */
    const float embed_scale = sqrtf((float) HIDDEN);
    float      *h_pre = (float *) malloc(n_ids * HIDDEN * sizeof(float)); /* "residual_1" buffer */
    const uint16_t *table = (const uint16_t *) embed->data;
    for (size_t t = 0; t < n_ids; t++) {
        const uint16_t *row = table + (size_t) ids[t] * HIDDEN;
        for (size_t i = 0; i < HIDDEN; i++)
            h_pre[t * HIDDEN + i] = bf16_to_fp32(row[i]) * embed_scale;
    }
    float *normed = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, h_pre, in_ln_w, RMS_EPS, normed);

    float *q = (float *) malloc(n_ids * Q_OUT * sizeof(float));
    float *k = (float *) malloc(n_ids * KV_OUT * sizeof(float));
    float *v = (float *) malloc(n_ids * KV_OUT * sizeof(float));
    linear_fp32(n_ids, HIDDEN, Q_OUT, normed, q_w, nullptr, q);
    linear_fp32(n_ids, HIDDEN, KV_OUT, normed, k_w, nullptr, k);
    linear_fp32(n_ids, HIDDEN, KV_OUT, normed, v_w, nullptr, v);
    rmsnorm_fp32(n_ids * Q_HEADS, HEAD_DIM, q, q_norm_w, RMS_EPS, q);
    rmsnorm_fp32(n_ids * KV_HEADS, HEAD_DIM, k, k_norm_w, RMS_EPS, k);
    rmsnorm_fp32(n_ids * KV_HEADS, HEAD_DIM, v, nullptr, RMS_EPS, v);

    float *cos_b = (float *) malloc(n_ids * HEAD_DIM * sizeof(float));
    float *sin_b = (float *) malloc(n_ids * HEAD_DIM * sizeof(float));
    rope_compute(n_ids, HEAD_DIM, HEAD_DIM, ROPE_THETA, cos_b, sin_b);
    rope_apply(n_ids, Q_HEADS, HEAD_DIM, q, cos_b, sin_b);
    rope_apply(n_ids, KV_HEADS, HEAD_DIM, k, cos_b, sin_b);

    float *attn_out = (float *) malloc(n_ids * Q_OUT * sizeof(float));
    attention_mqa_causal(n_ids, Q_HEADS, KV_HEADS, HEAD_DIM, SLIDING_WINDOW, q, k, v, attn_out);

    float *o = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    linear_fp32(n_ids, Q_OUT, HIDDEN, attn_out, o_w, nullptr, o);
    /* o is now the o_proj output, validated in Step 5+6+7. */

    /* === Step 8: post_attention_layernorm === */
    float *post_attn = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, o, post_attn_w, RMS_EPS, post_attn);

    char path[1024];
    snprintf(path, sizeof(path), "%s.post_attn_norm.bin", argv[3]);
    write_bin(path, post_attn, n_ids * HIDDEN * sizeof(float));

    /* === Residual 1: h_post_attn = h_pre + post_attn === */
    float *h_post_attn = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    add_fp32(n_ids * HIDDEN, h_pre, post_attn, h_post_attn);
    /* h_post_attn becomes "residual_2" later */

    /* === Step 9: pre_feedforward_layernorm === */
    float *pre_ff_normed = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, h_post_attn, pre_ff_w, RMS_EPS, pre_ff_normed);
    snprintf(path, sizeof(path), "%s.pre_ff_norm.bin", argv[3]);
    write_bin(path, pre_ff_normed, n_ids * HIDDEN * sizeof(float));

    /* === Step 10: MLP === */
    float *gate_out = (float *) malloc(n_ids * INTER * sizeof(float));
    float *up_out   = (float *) malloc(n_ids * INTER * sizeof(float));
    linear_fp32(n_ids, HIDDEN, INTER, pre_ff_normed, gate_w, nullptr, gate_out);
    linear_fp32(n_ids, HIDDEN, INTER, pre_ff_normed, up_w, nullptr, up_out);
    snprintf(path, sizeof(path), "%s.gate_proj.bin", argv[3]);
    write_bin(path, gate_out, n_ids * INTER * sizeof(float));
    snprintf(path, sizeof(path), "%s.up_proj.bin", argv[3]);
    write_bin(path, up_out, n_ids * INTER * sizeof(float));

    /* GELU-tanh on gate, then mul with up, then down_proj */
    float *gate_act = (float *) malloc(n_ids * INTER * sizeof(float));
    gelu_tanh_fp32(n_ids * INTER, gate_out, gate_act);
    float *gate_up = (float *) malloc(n_ids * INTER * sizeof(float));
    mul_fp32(n_ids * INTER, gate_act, up_out, gate_up);
    float *down_out = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    linear_fp32(n_ids, INTER, HIDDEN, gate_up, down_w, nullptr, down_out);
    snprintf(path, sizeof(path), "%s.down_proj.bin", argv[3]);
    write_bin(path, down_out, n_ids * HIDDEN * sizeof(float));

    /* === Step 11: post_feedforward_layernorm === */
    float *post_ff = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, down_out, post_ff_w, RMS_EPS, post_ff);
    snprintf(path, sizeof(path), "%s.post_ff_norm.bin", argv[3]);
    write_bin(path, post_ff, n_ids * HIDDEN * sizeof(float));

    fprintf(stderr,
            "wrote %s.{post_attn_norm,pre_ff_norm,gate_proj,up_proj,down_proj,post_ff_norm}.bin\n",
            argv[3]);

    /* free everything */
    free(post_ff);
    free(down_out);
    free(gate_up);
    free(gate_act);
    free(up_out);
    free(gate_out);
    free(pre_ff_normed);
    free(h_post_attn);
    free(post_attn);
    free(o);
    free(attn_out);
    free(cos_b);
    free(sin_b);
    free(q);
    free(k);
    free(v);
    free(normed);
    free(h_pre);
    free(in_ln_w);
    free(q_w);
    free(k_w);
    free(v_w);
    free(q_norm_w);
    free(k_norm_w);
    free(o_w);
    free(post_attn_w);
    free(pre_ff_w);
    free(gate_w);
    free(up_w);
    free(down_w);
    free(post_ff_w);
    free(ids);
    st_close(ctx);
    return 0;
}
