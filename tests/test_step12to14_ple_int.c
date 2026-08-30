/*
 * test_step12to14_ple — Steps 12, 13, 14 of Sub-Task D combined.
 *
 * Implements:
 *  12. Model-level PLE pre-compute (runs once at model entry):
 *        per_layer_inputs = (per_layer_proj_norm(per_layer_model_projection(embeds) / sqrt(1536))
 *                            + embed_tokens_per_layer(ids) * sqrt(256))
 *                           * (1 / sqrt(2))
 *      Shape: (seq, num_layers=35, hidden_per_layer=256)
 *  13. Per-Layer PLE merge for layer 0:
 *        residual_3 = h_after_post_ff_norm_residual
 *        h = per_layer_input_gate(h)        # 1536 -> 256
 *        h = gelu_tanh(h)
 *        h = h * per_layer_inputs[:, 0, :]
 *        h = per_layer_projection(h)        # 256 -> 1536
 *        h = post_per_layer_input_norm(h)
 *        h = residual_3 + h
 *  14. h *= layer_scalar
 *
 * Reproduces Steps 1-11 internally, then validates against:
 *   - ple_lookup                                   (1, seq, 8960)  [pre-norm raw lookup ×
 * sqrt(256)]
 *   - layer_00_per_layer_input_gate                (1, seq, 256)
 *   - layer_00_per_layer_projection                (1, seq, 1536)
 *   - layer_00_post_per_layer_input_norm           (1, seq, 1536)
 *   - layer_00_output                              (1, seq, 1536)
 */
#include "safetensors_reader.h"
#include "gemma4_kernels.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 1536
#define INTER 6144
#define Q_HEADS 8
#define KV_HEADS 1
#define HEAD_DIM 256
#define Q_OUT (Q_HEADS * HEAD_DIM)
#define KV_OUT (KV_HEADS * HEAD_DIM)
#define SLIDING_WINDOW 512
#define ROPE_THETA 10000.0f
#define RMS_EPS 1e-6f

#define NUM_LAYERS 35
#define HIDDEN_PER_LAYER 256
#define PLE_OUT (NUM_LAYERS * HIDDEN_PER_LAYER) /* 8960 */

#define PLE_INPUT_SCALE 0.7071067811865476f       /* 2 ** -0.5 */
#define PLE_MODEL_PROJ_SCALE 0.02551551815399144f /* 1 / sqrt(1536) */
#define PLE_TABLE_SCALE 16.0f                     /* sqrt(256) */

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
static void write_bin(const char *p, const void *d, size_t b) {
    FILE *f = fopen(p, "wb");
    xfwrite(d, 1, b, f);
    fclose(f);
}
static float *load_bf16(struct st_ctx *c, const char *n, size_t expected) {
    const struct st_tensor_t *t = st_get(c, n);
    if (!t || t->dtype != ST_DTYPE_BF16) {
        fprintf(stderr, "miss/bad: %s\n", n);
        return nullptr;
    }
    size_t elems = 1;
    for (size_t i = 0; i < t->rank; i++)
        elems *= t->shape[i];
    if (elems != expected) {
        fprintf(stderr, "elem mismatch %s: %zu vs %zu\n", n, elems, expected);
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

    /* === Load all weights === */
    const struct st_tensor_t *embed = st_get(ctx, "model.language_model.embed_tokens.weight");
    const struct st_tensor_t *ple_table =
            st_get(ctx, "model.language_model.embed_tokens_per_layer.weight");
    if (!embed || !ple_table) {
        fprintf(stderr, "embed/ple_table missing\n");
        return 1;
    }
    if (ple_table->shape[0] != 262144 || ple_table->shape[1] != PLE_OUT) {
        fprintf(stderr, "bad ple_table shape\n");
        return 1;
    }

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

    /* PLE-related weights */
    float *model_proj_w      = load_bf16(ctx,
                                         "model.language_model.per_layer_model_projection.weight",
                                         (size_t) PLE_OUT * HIDDEN);
    float *model_proj_norm_w = load_bf16(
            ctx, "model.language_model.per_layer_projection_norm.weight", HIDDEN_PER_LAYER);
    float *layer_input_gate_w =
            load_bf16(ctx,
                      "model.language_model.layers.0.per_layer_input_gate.weight",
                      (size_t) HIDDEN_PER_LAYER * HIDDEN);
    float *layer_proj_w = load_bf16(ctx,
                                    "model.language_model.layers.0.per_layer_projection.weight",
                                    (size_t) HIDDEN * HIDDEN_PER_LAYER);
    float *post_per_layer_norm_w = load_bf16(
            ctx, "model.language_model.layers.0.post_per_layer_input_norm.weight", HIDDEN);
    float *layer_scalar_w = load_bf16(ctx, "model.language_model.layers.0.layer_scalar", 1);

    if (!in_ln_w || !model_proj_w || !model_proj_norm_w || !layer_input_gate_w || !layer_proj_w ||
        !post_per_layer_norm_w || !layer_scalar_w) {
        fprintf(stderr, "weight load failed\n");
        return 1;
    }
    fprintf(stderr, "layer_scalar = %.10f\n", layer_scalar_w[0]);

    size_t   n_ids = 0;
    int32_t *ids   = read_input_ids(argv[2], &n_ids);
    fprintf(stderr, "n_ids=%zu\n", n_ids);

    /* === Step 1: token embedding ===
     * Note: input_embeds are needed by per_layer_model_projection (Step 12)
     * AND as residual_1 (kept in h_pre). */
    const float     embed_scale = sqrtf((float) HIDDEN);
    float          *h_pre       = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    const uint16_t *etable      = (const uint16_t *) embed->data;
    for (size_t t = 0; t < n_ids; t++) {
        const uint16_t *row = etable + (size_t) ids[t] * HIDDEN;
        for (size_t i = 0; i < HIDDEN; i++)
            h_pre[t * HIDDEN + i] = bf16_to_fp32(row[i]) * embed_scale;
    }

    /* === Step 12: PLE pre-compute === */
    /* (a) PLE table lookup × sqrt(256) — this matches the `ple_lookup` hook */
    float          *ple_table_lookup = (float *) malloc(n_ids * PLE_OUT * sizeof(float));
    const uint16_t *ptable           = (const uint16_t *) ple_table->data;
    for (size_t t = 0; t < n_ids; t++) {
        const uint16_t *row = ptable + (size_t) ids[t] * PLE_OUT;
        for (size_t i = 0; i < PLE_OUT; i++)
            ple_table_lookup[t * PLE_OUT + i] = bf16_to_fp32(row[i]) * PLE_TABLE_SCALE;
    }
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s.ple_lookup.bin", argv[3]);
        write_bin(path, ple_table_lookup, n_ids * PLE_OUT * sizeof(float));
    }

    /* (b) Model-level projection of input_embeds → (seq, 8960) × scale */
    float *ple_proj = (float *) malloc(n_ids * PLE_OUT * sizeof(float));
    linear_fp32(n_ids, HIDDEN, PLE_OUT, h_pre, model_proj_w, nullptr, ple_proj);
    for (size_t i = 0; i < n_ids * PLE_OUT; i++)
        ple_proj[i] *= PLE_MODEL_PROJ_SCALE;

    /* (c) Reshape implicit; apply RMSNorm per (35, 256) slice */
    rmsnorm_fp32(
            n_ids * NUM_LAYERS, HIDDEN_PER_LAYER, ple_proj, model_proj_norm_w, RMS_EPS, ple_proj);

    /* (d) Mix: per_layer_inputs = (ple_proj + ple_table_lookup) * (1/sqrt(2)) */
    float *per_layer_inputs = (float *) malloc(n_ids * PLE_OUT * sizeof(float));
    for (size_t i = 0; i < n_ids * PLE_OUT; i++) {
        per_layer_inputs[i] = (ple_proj[i] + ple_table_lookup[i]) * PLE_INPUT_SCALE;
    }
    /* per_layer_inputs has shape (seq, 35, 256). For layer i, the slice is per_layer_inputs[:, i,
     * :]. */

    /* === Steps 2-11: Layer 0 forward up to post_ff_norm + 2nd residual === */
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

    /* post_attn_norm + residual_1 */
    float *post_attn = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, o, post_attn_w, RMS_EPS, post_attn);
    float *h_post_attn = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    add_fp32(n_ids * HIDDEN, h_pre, post_attn, h_post_attn);

    /* pre_ff_norm */
    float *pre_ff_normed = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, h_post_attn, pre_ff_w, RMS_EPS, pre_ff_normed);

    /* MLP */
    float *gate_out = (float *) malloc(n_ids * INTER * sizeof(float));
    float *up_out   = (float *) malloc(n_ids * INTER * sizeof(float));
    linear_fp32(n_ids, HIDDEN, INTER, pre_ff_normed, gate_w, nullptr, gate_out);
    linear_fp32(n_ids, HIDDEN, INTER, pre_ff_normed, up_w, nullptr, up_out);
    gelu_tanh_fp32(n_ids * INTER, gate_out, gate_out);
    mul_fp32(n_ids * INTER, gate_out, up_out, gate_out); /* gate_out = gelu(gate) * up */
    float *down_out = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    linear_fp32(n_ids, INTER, HIDDEN, gate_out, down_w, nullptr, down_out);

    /* post_ff_norm + residual_2 */
    float *post_ff = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, down_out, post_ff_w, RMS_EPS, post_ff);
    float *h_post_ff = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    add_fp32(n_ids * HIDDEN, h_post_attn, post_ff, h_post_ff);
    /* h_post_ff is the input to PLE merge. residual_3 = h_post_ff. */

    /* === Step 13: Per-Layer PLE merge === */
    /* per_layer_input_gate(h): 1536 -> 256 */
    float *gate_ple = (float *) malloc(n_ids * HIDDEN_PER_LAYER * sizeof(float));
    linear_fp32(n_ids, HIDDEN, HIDDEN_PER_LAYER, h_post_ff, layer_input_gate_w, nullptr, gate_ple);
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s.per_layer_input_gate.bin", argv[3]);
        write_bin(path, gate_ple, n_ids * HIDDEN_PER_LAYER * sizeof(float));
    }

    /* GELU + element-wise mul with per_layer_inputs[:, 0, :] */
    gelu_tanh_fp32(n_ids * HIDDEN_PER_LAYER, gate_ple, gate_ple);
    /* per_layer_inputs has shape (seq, 35, 256). Slice [:, 0, :] is at offset
     * t * 35 * 256 + 0 * 256, i.e. the FIRST 256 floats per token. */
    for (size_t t = 0; t < n_ids; t++) {
        float       *g         = gate_ple + t * HIDDEN_PER_LAYER;
        const float *ple_slice = per_layer_inputs + t * PLE_OUT + 0 * HIDDEN_PER_LAYER;
        for (size_t i = 0; i < HIDDEN_PER_LAYER; i++)
            g[i] *= ple_slice[i];
    }

    /* per_layer_projection: 256 -> 1536 */
    float *proj_ple = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    linear_fp32(n_ids, HIDDEN_PER_LAYER, HIDDEN, gate_ple, layer_proj_w, nullptr, proj_ple);
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s.per_layer_projection.bin", argv[3]);
        write_bin(path, proj_ple, n_ids * HIDDEN * sizeof(float));
    }

    /* post_per_layer_input_norm */
    float *normed_ple = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, proj_ple, post_per_layer_norm_w, RMS_EPS, normed_ple);
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s.post_per_layer_input_norm.bin", argv[3]);
        write_bin(path, normed_ple, n_ids * HIDDEN * sizeof(float));
    }

    /* residual_3 + normed_ple */
    float *layer_out = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    add_fp32(n_ids * HIDDEN, h_post_ff, normed_ple, layer_out);

    /* === Step 14: layer_scalar multiply === */
    float ls = layer_scalar_w[0];
    for (size_t i = 0; i < n_ids * HIDDEN; i++)
        layer_out[i] *= ls;

    {
        char path[1024];
        snprintf(path, sizeof(path), "%s.layer_output.bin", argv[3]);
        write_bin(path, layer_out, n_ids * HIDDEN * sizeof(float));
    }
    fprintf(stderr,
            "wrote "
            "%s.{ple_lookup,per_layer_input_gate,per_layer_projection,post_per_layer_input_norm,"
            "layer_output}.bin\n",
            argv[3]);

    free(layer_out);
    free(normed_ple);
    free(proj_ple);
    free(gate_ple);
    free(h_post_ff);
    free(post_ff);
    free(down_out);
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
    free(per_layer_inputs);
    free(ple_proj);
    free(ple_table_lookup);
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
    free(model_proj_w);
    free(model_proj_norm_w);
    free(layer_input_gate_w);
    free(layer_proj_w);
    free(post_per_layer_norm_w);
    free(layer_scalar_w);
    free(ids);
    st_close(ctx);
    return 0;
}
