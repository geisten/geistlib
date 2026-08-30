/*
 * test_layers0to4 — Sub-Task E: chain layers 0-4 of Gemma 4 E2B.
 *
 * Layers 0-3 are sliding_attention (head_dim=256, theta=10000, full RoPE).
 * Layer 4 is full_attention (head_dim=512, theta=1e6, partial RoPE 25%).
 *
 * Validates against `layer_NN_output` hooks for N in 0..4.
 *
 * The PLE pre-compute (Step 12 from Sub-Task D) runs ONCE at model entry,
 * producing per_layer_inputs[seq, 35, 256]. Each layer i uses slice [:, i, :].
 */
#include "safetensors_reader.h"
#include "gemma4_kernels.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 1536
#define NUM_LAYERS 35
#define HIDDEN_PER_LAYER 256
#define PLE_OUT (NUM_LAYERS * HIDDEN_PER_LAYER) /* 8960 */
#define N_Q_HEADS 8
#define N_KV_HEADS 1
#define RMS_EPS 1e-6f

#define PLE_INPUT_SCALE 0.7071067811865476f
#define PLE_MODEL_PROJ_SCALE 0.02551551815399144f
#define PLE_TABLE_SCALE 16.0f /* sqrt(256) */

/* Per-layer weight bundle. */
typedef struct {
    int   layer_idx;
    bool  is_full;        /* false = sliding */
    int   head_dim;       /* 256 sliding, 512 full */
    int   q_out;          /* n_q_heads * head_dim */
    int   kv_out;         /* n_kv_heads * head_dim */
    int   intermediate;   /* 6144 (or 12288 for kv-shared) */
    int   sliding_window; /* 512 sliding, 0 full */
    float rope_theta;
    int   n_rotated_dims; /* head_dim sliding, 128 full (partial 25%) */

    const float *input_ln_w;
    const float *q_proj_w;
    const float *k_proj_w;
    const float *v_proj_w;
    const float *o_proj_w;
    const float *q_norm_w;
    const float *k_norm_w;
    const float *post_attn_ln_w;
    const float *pre_ff_ln_w;
    const float *gate_w;
    const float *up_w;
    const float *down_w;
    const float *post_ff_ln_w;
    const float *per_layer_gate_w;
    const float *per_layer_proj_w;
    const float *post_per_layer_norm_w;
    float        layer_scalar;
} LayerW;

static int32_t *read_input_ids(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int32_t *ids = (int32_t *) xmalloc((size_t) sz);
    xfread(ids, 1, (size_t) sz, f);
    fclose(f);
    *n = (size_t) sz / 4;
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

/* Build a name like "model.language_model.layers.<idx>.<suffix>". */
static void layer_path(char *buf, size_t bufsz, int idx, const char *suffix) {
    snprintf(buf, bufsz, "model.language_model.layers.%d.%s", idx, suffix);
}

/* Load all weights for one layer. Caller frees via free_layer_w. */
static bool load_layer_w(struct st_ctx *ctx, LayerW *L, const char *layer_type) {
    L->is_full        = strcmp(layer_type, "full_attention") == 0;
    L->head_dim       = L->is_full ? 512 : 256;
    L->q_out          = N_Q_HEADS * L->head_dim;
    L->kv_out         = N_KV_HEADS * L->head_dim;
    L->intermediate   = 6144;
    L->sliding_window = L->is_full ? 0 : 512;
    L->rope_theta     = L->is_full ? 1000000.0f : 10000.0f;
    L->n_rotated_dims = L->is_full ? 128 : L->head_dim;

    char p[256];
    layer_path(p, sizeof(p), L->layer_idx, "input_layernorm.weight");
    L->input_ln_w = load_bf16(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "self_attn.q_proj.weight");
    L->q_proj_w = load_bf16(ctx, p, (size_t) L->q_out * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "self_attn.k_proj.weight");
    L->k_proj_w = load_bf16(ctx, p, (size_t) L->kv_out * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "self_attn.v_proj.weight");
    L->v_proj_w = load_bf16(ctx, p, (size_t) L->kv_out * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "self_attn.o_proj.weight");
    L->o_proj_w = load_bf16(ctx, p, (size_t) HIDDEN * L->q_out);
    layer_path(p, sizeof(p), L->layer_idx, "self_attn.q_norm.weight");
    L->q_norm_w = load_bf16(ctx, p, L->head_dim);
    layer_path(p, sizeof(p), L->layer_idx, "self_attn.k_norm.weight");
    L->k_norm_w = load_bf16(ctx, p, L->head_dim);
    layer_path(p, sizeof(p), L->layer_idx, "post_attention_layernorm.weight");
    L->post_attn_ln_w = load_bf16(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "pre_feedforward_layernorm.weight");
    L->pre_ff_ln_w = load_bf16(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "mlp.gate_proj.weight");
    L->gate_w = load_bf16(ctx, p, (size_t) L->intermediate * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "mlp.up_proj.weight");
    L->up_w = load_bf16(ctx, p, (size_t) L->intermediate * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "mlp.down_proj.weight");
    L->down_w = load_bf16(ctx, p, (size_t) HIDDEN * L->intermediate);
    layer_path(p, sizeof(p), L->layer_idx, "post_feedforward_layernorm.weight");
    L->post_ff_ln_w = load_bf16(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "per_layer_input_gate.weight");
    L->per_layer_gate_w = load_bf16(ctx, p, (size_t) HIDDEN_PER_LAYER * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "per_layer_projection.weight");
    L->per_layer_proj_w = load_bf16(ctx, p, (size_t) HIDDEN * HIDDEN_PER_LAYER);
    layer_path(p, sizeof(p), L->layer_idx, "post_per_layer_input_norm.weight");
    L->post_per_layer_norm_w = load_bf16(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "layer_scalar");
    float *ls = load_bf16(ctx, p, 1);
    if (!ls)
        return false;
    L->layer_scalar = ls[0];
    free(ls);
    return L->input_ln_w && L->q_proj_w && L->k_proj_w && L->v_proj_w && L->o_proj_w &&
           L->q_norm_w && L->k_norm_w && L->post_attn_ln_w && L->pre_ff_ln_w && L->gate_w &&
           L->up_w && L->down_w && L->post_ff_ln_w && L->per_layer_gate_w && L->per_layer_proj_w &&
           L->post_per_layer_norm_w;
}

static void free_layer_w(LayerW *L) {
    free((void *) L->input_ln_w);
    free((void *) L->q_proj_w);
    free((void *) L->k_proj_w);
    free((void *) L->v_proj_w);
    free((void *) L->o_proj_w);
    free((void *) L->q_norm_w);
    free((void *) L->k_norm_w);
    free((void *) L->post_attn_ln_w);
    free((void *) L->pre_ff_ln_w);
    free((void *) L->gate_w);
    free((void *) L->up_w);
    free((void *) L->down_w);
    free((void *) L->post_ff_ln_w);
    free((void *) L->per_layer_gate_w);
    free((void *) L->per_layer_proj_w);
    free((void *) L->post_per_layer_norm_w);
}

/* Forward one layer in-place: h (seq, HIDDEN) gets transformed to h_out.
 * per_layer_input is the layer's PLE slice (seq, HIDDEN_PER_LAYER). */
static void forward_layer(const LayerW *L,
                          size_t        seq,
                          const float  *h_in,
                          const float  *per_layer_input,
                          float        *h_out) {
    const int hd    = L->head_dim;
    const int q_out = L->q_out, kv_out = L->kv_out, inter = L->intermediate;

    float *h_pre = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    memcpy(h_pre, h_in, seq * HIDDEN * sizeof(float)); /* residual_1 */

    /* Attention sub-block */
    float *normed = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    rmsnorm_fp32(seq, HIDDEN, h_pre, L->input_ln_w, RMS_EPS, normed);

    float *q = (float *) xmalloc(seq * (size_t) q_out * sizeof(float));
    float *k = (float *) xmalloc(seq * (size_t) kv_out * sizeof(float));
    float *v = (float *) xmalloc(seq * (size_t) kv_out * sizeof(float));
    linear_fp32(seq, HIDDEN, q_out, normed, L->q_proj_w, nullptr, q);
    linear_fp32(seq, HIDDEN, kv_out, normed, L->k_proj_w, nullptr, k);
    linear_fp32(seq, HIDDEN, kv_out, normed, L->v_proj_w, nullptr, v);
    rmsnorm_fp32(seq * N_Q_HEADS, hd, q, L->q_norm_w, RMS_EPS, q);
    rmsnorm_fp32(seq * N_KV_HEADS, hd, k, L->k_norm_w, RMS_EPS, k);
    rmsnorm_fp32(seq * N_KV_HEADS, hd, v, nullptr, RMS_EPS, v);

    float *cos_b = (float *) xmalloc(seq * (size_t) hd * sizeof(float));
    float *sin_b = (float *) xmalloc(seq * (size_t) hd * sizeof(float));
    rope_compute(seq, hd, L->n_rotated_dims, L->rope_theta, cos_b, sin_b);
    rope_apply(seq, N_Q_HEADS, hd, q, cos_b, sin_b);
    rope_apply(seq, N_KV_HEADS, hd, k, cos_b, sin_b);

    float *attn_out = (float *) xmalloc(seq * (size_t) q_out * sizeof(float));
    attention_mqa_causal(
            seq, N_Q_HEADS, N_KV_HEADS, hd, (size_t) L->sliding_window, q, k, v, attn_out);

    float *o = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    linear_fp32(seq, q_out, HIDDEN, attn_out, L->o_proj_w, nullptr, o);

    /* post_attention_layernorm + residual_1 */
    float *post_attn = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    rmsnorm_fp32(seq, HIDDEN, o, L->post_attn_ln_w, RMS_EPS, post_attn);
    float *h_post_attn = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    add_fp32(seq * HIDDEN, h_pre, post_attn, h_post_attn);

    /* MLP sub-block */
    float *pre_ff = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    rmsnorm_fp32(seq, HIDDEN, h_post_attn, L->pre_ff_ln_w, RMS_EPS, pre_ff);
    float *gate = (float *) xmalloc(seq * (size_t) inter * sizeof(float));
    float *up   = (float *) xmalloc(seq * (size_t) inter * sizeof(float));
    linear_fp32(seq, HIDDEN, inter, pre_ff, L->gate_w, nullptr, gate);
    linear_fp32(seq, HIDDEN, inter, pre_ff, L->up_w, nullptr, up);
    gelu_tanh_fp32(seq * (size_t) inter, gate, gate);
    mul_fp32(seq * (size_t) inter, gate, up, gate);
    float *down = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    linear_fp32(seq, inter, HIDDEN, gate, L->down_w, nullptr, down);

    /* post_feedforward_layernorm + residual_2 */
    float *post_ff = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    rmsnorm_fp32(seq, HIDDEN, down, L->post_ff_ln_w, RMS_EPS, post_ff);
    float *h_post_ff = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    add_fp32(seq * HIDDEN, h_post_attn, post_ff, h_post_ff);

    /* PLE merge */
    float *gate_ple = (float *) xmalloc(seq * HIDDEN_PER_LAYER * sizeof(float));
    linear_fp32(seq, HIDDEN, HIDDEN_PER_LAYER, h_post_ff, L->per_layer_gate_w, nullptr, gate_ple);
    gelu_tanh_fp32(seq * HIDDEN_PER_LAYER, gate_ple, gate_ple);
    for (size_t t = 0; t < seq; t++) {
        float       *g = gate_ple + t * HIDDEN_PER_LAYER;
        const float *p = per_layer_input + t * HIDDEN_PER_LAYER;
        for (size_t i = 0; i < HIDDEN_PER_LAYER; i++)
            g[i] *= p[i];
    }
    float *proj_ple = (float *) xmalloc(seq * HIDDEN * sizeof(float));
    linear_fp32(seq, HIDDEN_PER_LAYER, HIDDEN, gate_ple, L->per_layer_proj_w, nullptr, proj_ple);
    rmsnorm_fp32(seq, HIDDEN, proj_ple, L->post_per_layer_norm_w, RMS_EPS, proj_ple);
    add_fp32(seq * HIDDEN, h_post_ff, proj_ple, h_out);

    /* layer_scalar */
    for (size_t i = 0; i < seq * HIDDEN; i++)
        h_out[i] *= L->layer_scalar;

    free(proj_ple);
    free(gate_ple);
    free(h_post_ff);
    free(post_ff);
    free(down);
    free(up);
    free(gate);
    free(pre_ff);
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
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 4, "<model.safetensors> <ids.bin> <out_prefix>");

    const char    *err = nullptr;
    struct st_ctx *ctx = st_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    /* Load model-level weights (embeddings + PLE pre-compute) */
    const struct st_tensor_t *embed = st_get(ctx, "model.language_model.embed_tokens.weight");
    const struct st_tensor_t *ple_table =
            st_get(ctx, "model.language_model.embed_tokens_per_layer.weight");
    float *model_proj_w      = load_bf16(ctx,
                                         "model.language_model.per_layer_model_projection.weight",
                                         (size_t) PLE_OUT * HIDDEN);
    float *model_proj_norm_w = load_bf16(
            ctx, "model.language_model.per_layer_projection_norm.weight", HIDDEN_PER_LAYER);
    if (!embed || !ple_table || !model_proj_w || !model_proj_norm_w) {
        fprintf(stderr, "model weight load failed\n");
        return 1;
    }

    /* Layer types for first 5 layers */
    const char *layer_types[5] = {
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
    };
    LayerW layers[5];
    for (int i = 0; i < 5; i++) {
        layers[i].layer_idx = i;
        if (!load_layer_w(ctx, &layers[i], layer_types[i])) {
            fprintf(stderr, "layer %d weights failed\n", i);
            return 1;
        }
    }

    size_t   n_ids = 0;
    int32_t *ids   = read_input_ids(argv[2], &n_ids);
    fprintf(stderr, "n_ids=%zu\n", n_ids);

    /* Embedding */
    const float     embed_scale = sqrtf((float) HIDDEN);
    float          *h           = (float *) xmalloc(n_ids * HIDDEN * sizeof(float));
    const uint16_t *etable      = (const uint16_t *) embed->data;
    for (size_t t = 0; t < n_ids; t++) {
        const uint16_t *row = etable + (size_t) ids[t] * HIDDEN;
        for (size_t i = 0; i < HIDDEN; i++)
            h[t * HIDDEN + i] = bf16_to_fp32(row[i]) * embed_scale;
    }

    /* PLE pre-compute */
    float          *ple_table_lookup = (float *) xmalloc(n_ids * PLE_OUT * sizeof(float));
    const uint16_t *ptable           = (const uint16_t *) ple_table->data;
    for (size_t t = 0; t < n_ids; t++) {
        const uint16_t *row = ptable + (size_t) ids[t] * PLE_OUT;
        for (size_t i = 0; i < PLE_OUT; i++)
            ple_table_lookup[t * PLE_OUT + i] = bf16_to_fp32(row[i]) * PLE_TABLE_SCALE;
    }
    float *ple_proj = (float *) xmalloc(n_ids * PLE_OUT * sizeof(float));
    linear_fp32(n_ids, HIDDEN, PLE_OUT, h, model_proj_w, nullptr, ple_proj);
    for (size_t i = 0; i < n_ids * PLE_OUT; i++)
        ple_proj[i] *= PLE_MODEL_PROJ_SCALE;
    rmsnorm_fp32(
            n_ids * NUM_LAYERS, HIDDEN_PER_LAYER, ple_proj, model_proj_norm_w, RMS_EPS, ple_proj);
    float *per_layer_inputs = (float *) xmalloc(n_ids * PLE_OUT * sizeof(float));
    for (size_t i = 0; i < n_ids * PLE_OUT; i++)
        per_layer_inputs[i] = (ple_proj[i] + ple_table_lookup[i]) * PLE_INPUT_SCALE;

    /* Forward through 5 layers, validating each */
    float *h_buf2 = (float *) xmalloc(n_ids * HIDDEN * sizeof(float));
    char   path[1024];
    for (int li = 0; li < 5; li++) {
        /* Slice per_layer_inputs[:, li, :] — extract into contiguous buffer. */
        float *ple_slice = (float *) xmalloc(n_ids * HIDDEN_PER_LAYER * sizeof(float));
        for (size_t t = 0; t < n_ids; t++) {
            const float *src = per_layer_inputs + t * PLE_OUT + (size_t) li * HIDDEN_PER_LAYER;
            float       *dst = ple_slice + t * HIDDEN_PER_LAYER;
            memcpy(dst, src, HIDDEN_PER_LAYER * sizeof(float));
        }
        forward_layer(&layers[li], n_ids, h, ple_slice, h_buf2);
        free(ple_slice);
        /* Save layer output for validation */
        snprintf(path, sizeof(path), "%s.layer_%02d.bin", argv[3], li);
        write_bin(path, h_buf2, n_ids * HIDDEN * sizeof(float));
        /* Swap: h_buf2 → h, allocate fresh h_buf2 next iteration */
        float *tmp = h;
        h          = h_buf2;
        h_buf2     = tmp;
    }
    fprintf(stderr, "wrote %s.layer_{00..04}.bin\n", argv[3]);

    free(h);
    free(h_buf2);
    free(per_layer_inputs);
    free(ple_proj);
    free(ple_table_lookup);
    free(model_proj_w);
    free(model_proj_norm_w);
    free(ids);
    for (int i = 0; i < 5; i++)
        free_layer_w(&layers[i]);
    st_close(ctx);
    return 0;
}
