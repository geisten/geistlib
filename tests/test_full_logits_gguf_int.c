/*
 * test_full_logits_gguf — Phase 2A: full forward from quantized GGUF: full forward + LM head +
 * softcap.
 *
 * Forwards through all 35 decoder layers (with KV-sharing for layers 15-34
 * and double-wide MLP for those layers), applies the final RMSNorm, the
 * LM head (tied with token embedding), and Gemma's tanh-softcap at 30.
 *
 * Validates against `dumps[T*]['logits']` (1, seq, 262144).
 */
#include "gguf_reader.h"
#include "quant.h"
#include "gguf_dequant.h"
#include "gemma4_kernels.h"
#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 1536
#define NUM_LAYERS 35
#define HIDDEN_PER_LAYER 256
#define PLE_OUT (NUM_LAYERS * HIDDEN_PER_LAYER)
#define N_Q_HEADS 8
#define N_KV_HEADS 1
#define VOCAB 262144
#define LOGIT_SOFTCAP 30.0f
#define RMS_EPS 1e-6f
#define PLE_INPUT_SCALE 0.7071067811865476f
#define PLE_MODEL_PROJ_SCALE 0.02551551815399144f
#define PLE_TABLE_SCALE 16.0f

/* Layer types for all 35 layers (4 sliding, 1 full, repeated 7 times). */
static const bool LAYER_IS_FULL[NUM_LAYERS] = {
        false, false, false, false, true, /* 0..4   */
        false, false, false, false, true, /* 5..9   */
        false, false, false, false, true, /* 10..14 */
        false, false, false, false, true, /* 15..19 (kv-shared from here) */
        false, false, false, false, true, /* 20..24 */
        false, false, false, false, true, /* 25..29 */
        false, false, false, false, true, /* 30..34 */
};
/* KV-shared layers reuse K/V from layer 13 (sliding source) or 14 (full source). */
#define KV_SLIDING_SRC 13
#define KV_FULL_SRC 14

typedef struct {
    int   layer_idx;
    bool  is_full;
    bool  is_kv_shared;
    int   head_dim;
    int   q_out;
    int   kv_out;
    int   intermediate; /* 6144 normal, 12288 if kv_shared (double-wide) */
    int   sliding_window;
    float rope_theta;
    int   n_rotated_dims;

    const float *input_ln_w;
    const float *q_proj_w;
    const float *k_proj_w; /* nullptr if kv_shared */
    const float *v_proj_w; /* nullptr if kv_shared */
    const float *o_proj_w;
    const float *q_norm_w;
    const float *k_norm_w; /* nullptr if kv_shared */
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

/* KV cache block: stores post-norm post-RoPE K and V for one source layer. */
typedef struct {
    int    head_dim;
    size_t seq;
    float *k; /* (seq, N_KV_HEADS, head_dim) */
    float *v; /* same */
} KVCache;

static int32_t *read_input_ids(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int32_t *ids = (int32_t *) malloc((size_t) sz);
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
static float *load_gguf(struct gguf_ctx *c, const char *n, size_t expected) {
    const struct gguf_tensor_t *t = gguf_get_tensor(c, n);
    if (!t) {
        fprintf(stderr, "miss: %s\n", n);
        return nullptr;
    }
    size_t elems = gguf_tensor_elem_count(t);
    if (elems != expected) {
        fprintf(stderr, "elem mismatch %s: %zu vs %zu\n", n, elems, expected);
        return nullptr;
    }
    float *fp32 = gguf_dequant_to_fp32(t);
    if (!fp32) {
        fprintf(stderr, "dequant failed for %s (dtype %s)\n", n, gguf_dtype_name(t->dtype));
        return nullptr;
    }
    return fp32;
}
static void layer_path(char *buf, size_t n, int idx, const char *suf) {
    snprintf(buf, n, "blk.%d.%s", idx, suf);
}

static bool load_layer_w(struct gguf_ctx *ctx, LayerW *L) {
    L->is_full        = LAYER_IS_FULL[L->layer_idx];
    L->is_kv_shared   = L->layer_idx >= 15;
    L->head_dim       = L->is_full ? 512 : 256;
    L->q_out          = N_Q_HEADS * L->head_dim;
    L->kv_out         = N_KV_HEADS * L->head_dim;
    L->intermediate   = L->is_kv_shared ? 12288 : 6144;
    L->sliding_window = L->is_full ? 0 : 512;
    L->rope_theta     = L->is_full ? 1000000.0f : 10000.0f;
    L->n_rotated_dims = L->is_full ? 128 : L->head_dim;

    char p[256];
    layer_path(p, sizeof(p), L->layer_idx, "attn_norm.weight");
    L->input_ln_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "attn_q.weight");
    L->q_proj_w = load_gguf(ctx, p, (size_t) L->q_out * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "attn_q_norm.weight");
    L->q_norm_w = load_gguf(ctx, p, L->head_dim);
    layer_path(p, sizeof(p), L->layer_idx, "attn_output.weight");
    L->o_proj_w = load_gguf(ctx, p, (size_t) HIDDEN * L->q_out);

    if (!L->is_kv_shared) {
        layer_path(p, sizeof(p), L->layer_idx, "attn_k.weight");
        L->k_proj_w = load_gguf(ctx, p, (size_t) L->kv_out * HIDDEN);
        layer_path(p, sizeof(p), L->layer_idx, "attn_v.weight");
        L->v_proj_w = load_gguf(ctx, p, (size_t) L->kv_out * HIDDEN);
        layer_path(p, sizeof(p), L->layer_idx, "attn_k_norm.weight");
        L->k_norm_w = load_gguf(ctx, p, L->head_dim);
    } else {
        L->k_proj_w = L->v_proj_w = L->k_norm_w = nullptr;
    }

    layer_path(p, sizeof(p), L->layer_idx, "post_attention_norm.weight");
    L->post_attn_ln_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "ffn_norm.weight");
    L->pre_ff_ln_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "ffn_gate.weight");
    L->gate_w = load_gguf(ctx, p, (size_t) L->intermediate * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "ffn_up.weight");
    L->up_w = load_gguf(ctx, p, (size_t) L->intermediate * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "ffn_down.weight");
    L->down_w = load_gguf(ctx, p, (size_t) HIDDEN * L->intermediate);
    layer_path(p, sizeof(p), L->layer_idx, "post_ffw_norm.weight");
    L->post_ff_ln_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "inp_gate.weight");
    L->per_layer_gate_w = load_gguf(ctx, p, (size_t) HIDDEN_PER_LAYER * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "proj.weight");
    L->per_layer_proj_w = load_gguf(ctx, p, (size_t) HIDDEN * HIDDEN_PER_LAYER);
    layer_path(p, sizeof(p), L->layer_idx, "post_norm.weight");
    L->post_per_layer_norm_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "layer_output_scale.weight");
    float *ls = load_gguf(ctx, p, 1);
    if (!ls)
        return false;
    L->layer_scalar = ls[0];
    free(ls);
    return L->input_ln_w && L->q_proj_w && L->q_norm_w && L->o_proj_w && L->post_attn_ln_w &&
           L->pre_ff_ln_w && L->gate_w && L->up_w && L->down_w && L->post_ff_ln_w &&
           L->per_layer_gate_w && L->per_layer_proj_w && L->post_per_layer_norm_w &&
           (L->is_kv_shared || (L->k_proj_w && L->v_proj_w && L->k_norm_w));
}
static void free_layer_w(LayerW *L) {
    free((void *) L->input_ln_w);
    free((void *) L->q_proj_w);
    free((void *) L->q_norm_w);
    free((void *) L->o_proj_w);
    free((void *) L->k_proj_w);
    free((void *) L->v_proj_w);
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

/* Forward one layer. If kv_use is non-null, skip K/V proj and use cached.
 * If kv_store is non-null, write the post-RoPE+norm K/V there. */
static void forward_layer(const LayerW  *L,
                          size_t         seq,
                          const float   *h_in,
                          const float   *per_layer_input,
                          KVCache       *kv_store,
                          const KVCache *kv_use,
                          float         *h_out) {
    const int hd    = L->head_dim;
    const int q_out = L->q_out, kv_out = L->kv_out, inter = L->intermediate;

    float *h_pre = (float *) malloc(seq * HIDDEN * sizeof(float));
    memcpy(h_pre, h_in, seq * HIDDEN * sizeof(float));

    float *normed = (float *) malloc(seq * HIDDEN * sizeof(float));
    rmsnorm_fp32(seq, HIDDEN, h_pre, L->input_ln_w, RMS_EPS, normed);

    /* Q always computed; per-head q_norm; RoPE on Q. */
    float *q = (float *) malloc(seq * (size_t) q_out * sizeof(float));
    linear_fp32(seq, HIDDEN, q_out, normed, L->q_proj_w, nullptr, q);
    rmsnorm_fp32(seq * N_Q_HEADS, hd, q, L->q_norm_w, RMS_EPS, q);

    float *cos_b = (float *) malloc(seq * (size_t) hd * sizeof(float));
    float *sin_b = (float *) malloc(seq * (size_t) hd * sizeof(float));
    rope_compute(seq, hd, L->n_rotated_dims, L->rope_theta, cos_b, sin_b);
    rope_apply(seq, N_Q_HEADS, hd, q, cos_b, sin_b);

    /* K/V either computed (and possibly cached) or pulled from cache. */
    const float *k_use;
    const float *v_use;
    float       *k_local = nullptr;
    float       *v_local = nullptr;
    if (kv_use) {
        k_use = kv_use->k;
        v_use = kv_use->v;
    } else {
        k_local = (float *) malloc(seq * (size_t) kv_out * sizeof(float));
        v_local = (float *) malloc(seq * (size_t) kv_out * sizeof(float));
        linear_fp32(seq, HIDDEN, kv_out, normed, L->k_proj_w, nullptr, k_local);
        linear_fp32(seq, HIDDEN, kv_out, normed, L->v_proj_w, nullptr, v_local);
        rmsnorm_fp32(seq * N_KV_HEADS, hd, k_local, L->k_norm_w, RMS_EPS, k_local);
        rmsnorm_fp32(seq * N_KV_HEADS, hd, v_local, nullptr, RMS_EPS, v_local);
        rope_apply(seq, N_KV_HEADS, hd, k_local, cos_b, sin_b);
        if (kv_store) {
            kv_store->head_dim = hd;
            kv_store->seq      = seq;
            kv_store->k        = (float *) malloc(seq * (size_t) kv_out * sizeof(float));
            kv_store->v        = (float *) malloc(seq * (size_t) kv_out * sizeof(float));
            memcpy(kv_store->k, k_local, seq * (size_t) kv_out * sizeof(float));
            memcpy(kv_store->v, v_local, seq * (size_t) kv_out * sizeof(float));
        }
        k_use = k_local;
        v_use = v_local;
    }

    float *attn_out = (float *) malloc(seq * (size_t) q_out * sizeof(float));
    attention_mqa_causal(
            seq, N_Q_HEADS, N_KV_HEADS, hd, (size_t) L->sliding_window, q, k_use, v_use, attn_out);

    float *o = (float *) malloc(seq * HIDDEN * sizeof(float));
    linear_fp32(seq, q_out, HIDDEN, attn_out, L->o_proj_w, nullptr, o);

    /* post_attn_norm + residual_1 */
    float *post_attn = (float *) malloc(seq * HIDDEN * sizeof(float));
    rmsnorm_fp32(seq, HIDDEN, o, L->post_attn_ln_w, RMS_EPS, post_attn);
    float *h_post_attn = (float *) malloc(seq * HIDDEN * sizeof(float));
    add_fp32(seq * HIDDEN, h_pre, post_attn, h_post_attn);

    /* MLP */
    float *pre_ff = (float *) malloc(seq * HIDDEN * sizeof(float));
    rmsnorm_fp32(seq, HIDDEN, h_post_attn, L->pre_ff_ln_w, RMS_EPS, pre_ff);
    float *gate = (float *) malloc(seq * (size_t) inter * sizeof(float));
    float *up   = (float *) malloc(seq * (size_t) inter * sizeof(float));
    linear_fp32(seq, HIDDEN, inter, pre_ff, L->gate_w, nullptr, gate);
    linear_fp32(seq, HIDDEN, inter, pre_ff, L->up_w, nullptr, up);
    gelu_tanh_fp32(seq * (size_t) inter, gate, gate);
    mul_fp32(seq * (size_t) inter, gate, up, gate);
    float *down = (float *) malloc(seq * HIDDEN * sizeof(float));
    linear_fp32(seq, inter, HIDDEN, gate, L->down_w, nullptr, down);

    /* post_ff_norm + residual_2 */
    float *post_ff = (float *) malloc(seq * HIDDEN * sizeof(float));
    rmsnorm_fp32(seq, HIDDEN, down, L->post_ff_ln_w, RMS_EPS, post_ff);
    float *h_post_ff = (float *) malloc(seq * HIDDEN * sizeof(float));
    add_fp32(seq * HIDDEN, h_post_attn, post_ff, h_post_ff);

    /* PLE merge + residual_3 */
    float *gate_ple = (float *) malloc(seq * HIDDEN_PER_LAYER * sizeof(float));
    linear_fp32(seq, HIDDEN, HIDDEN_PER_LAYER, h_post_ff, L->per_layer_gate_w, nullptr, gate_ple);
    gelu_tanh_fp32(seq * HIDDEN_PER_LAYER, gate_ple, gate_ple);
    for (size_t t = 0; t < seq; t++) {
        float       *g = gate_ple + t * HIDDEN_PER_LAYER;
        const float *p = per_layer_input + t * HIDDEN_PER_LAYER;
        for (size_t i = 0; i < HIDDEN_PER_LAYER; i++)
            g[i] *= p[i];
    }
    float *proj_ple = (float *) malloc(seq * HIDDEN * sizeof(float));
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
    free(k_local);
    free(v_local);
    free(normed);
    free(h_pre);
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 4, "<model.safetensors> <ids.bin> <out.bin>");

    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    const struct gguf_tensor_t *embed_t     = gguf_get_tensor(ctx, "token_embd.weight");
    const struct gguf_tensor_t *ple_table_t = gguf_get_tensor(ctx, "per_layer_token_embd.weight");
    if (!embed_t || !ple_table_t) {
        fprintf(stderr, "embed tensors missing\n");
        return 1;
    }
    fprintf(stderr,
            "Dequantizing embed_tokens (%s) and per_layer_token_embd (%s)...\n",
            gguf_dtype_name(embed_t->dtype),
            gguf_dtype_name(ple_table_t->dtype));
    float *embed_table    = gguf_dequant_to_fp32(embed_t);
    float *ple_table_data = gguf_dequant_to_fp32(ple_table_t);
    if (!embed_table || !ple_table_data) {
        fprintf(stderr, "embed dequant failed\n");
        return 1;
    }
    float *model_proj_w = load_gguf(ctx, "per_layer_model_proj.weight", (size_t) PLE_OUT * HIDDEN);
    float *model_proj_norm_w = load_gguf(ctx, "per_layer_proj_norm.weight", HIDDEN_PER_LAYER);
    float *final_norm_w      = load_gguf(ctx, "output_norm.weight", HIDDEN);
    /* lm_head shares weights with embed_tokens (tie_word_embeddings=true). */
    float *lm_head_w = (float *) malloc((size_t) VOCAB * HIDDEN * sizeof(float));
    memcpy(lm_head_w, embed_table, (size_t) VOCAB * HIDDEN * sizeof(float));
    if (!model_proj_w || !model_proj_norm_w || !final_norm_w || !lm_head_w) {
        fprintf(stderr, "model weights load failed\n");
        return 1;
    }

    fprintf(stderr, "Loading 35 layers...\n");
    LayerW layers[NUM_LAYERS];
    for (int i = 0; i < NUM_LAYERS; i++) {
        layers[i].layer_idx = i;
        if (!load_layer_w(ctx, &layers[i])) {
            fprintf(stderr, "layer %d weight load failed\n", i);
            return 1;
        }
    }
    fprintf(stderr, "All layers loaded.\n");

    size_t   n_ids = 0;
    int32_t *ids   = read_input_ids(argv[2], &n_ids);
    fprintf(stderr, "n_ids=%zu\n", n_ids);

    /* Embedding */
    const float embed_scale = sqrtf((float) HIDDEN);
    float      *h           = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    for (size_t t = 0; t < n_ids; t++) {
        const float *row = embed_table + (size_t) ids[t] * HIDDEN;
        for (size_t i = 0; i < HIDDEN; i++)
            h[t * HIDDEN + i] = row[i] * embed_scale;
    }

    /* PLE pre-compute */
    float *ple_table_lookup = (float *) malloc(n_ids * PLE_OUT * sizeof(float));
    for (size_t t = 0; t < n_ids; t++) {
        const float *row = ple_table_data + (size_t) ids[t] * PLE_OUT;
        for (size_t i = 0; i < PLE_OUT; i++)
            ple_table_lookup[t * PLE_OUT + i] = row[i] * PLE_TABLE_SCALE;
    }
    float *ple_proj = (float *) malloc(n_ids * PLE_OUT * sizeof(float));
    linear_fp32(n_ids, HIDDEN, PLE_OUT, h, model_proj_w, nullptr, ple_proj);
    for (size_t i = 0; i < n_ids * PLE_OUT; i++)
        ple_proj[i] *= PLE_MODEL_PROJ_SCALE;
    rmsnorm_fp32(
            n_ids * NUM_LAYERS, HIDDEN_PER_LAYER, ple_proj, model_proj_norm_w, RMS_EPS, ple_proj);
    float *per_layer_inputs = (float *) malloc(n_ids * PLE_OUT * sizeof(float));
    for (size_t i = 0; i < n_ids * PLE_OUT; i++)
        per_layer_inputs[i] = (ple_proj[i] + ple_table_lookup[i]) * PLE_INPUT_SCALE;
    free(ple_table_lookup);
    free(ple_proj);

    /* Forward through 35 layers */
    KVCache kv_sliding = {0}, kv_full = {0};
    float  *h_buf2    = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    float  *ple_slice = (float *) malloc(n_ids * HIDDEN_PER_LAYER * sizeof(float));
    for (int li = 0; li < NUM_LAYERS; li++) {
        for (size_t t = 0; t < n_ids; t++) {
            const float *src = per_layer_inputs + t * PLE_OUT + (size_t) li * HIDDEN_PER_LAYER;
            memcpy(ple_slice + t * HIDDEN_PER_LAYER, src, HIDDEN_PER_LAYER * sizeof(float));
        }
        KVCache       *store = nullptr;
        const KVCache *use   = nullptr;
        if (li == KV_SLIDING_SRC)
            store = &kv_sliding;
        else if (li == KV_FULL_SRC)
            store = &kv_full;
        else if (layers[li].is_kv_shared) {
            use = layers[li].is_full ? &kv_full : &kv_sliding;
        }
        if (li == 0) {
            FILE *dbg = fopen("/tmp/h_before_layer0.bin", "wb");
            xfwrite(h, sizeof(float), n_ids * HIDDEN, dbg);
            fclose(dbg);
            FILE *dbg2 = fopen("/tmp/ple_slice_layer0.bin", "wb");
            xfwrite(ple_slice, sizeof(float), n_ids * HIDDEN_PER_LAYER, dbg2);
            fclose(dbg2);
        }
        forward_layer(&layers[li], n_ids, h, ple_slice, store, use, h_buf2);
        float *tmp = h;
        h          = h_buf2;
        h_buf2     = tmp;
        if (li == 0) {
            FILE *dbg = fopen("/tmp/layer0_out_gguf.bin", "wb");
            xfwrite(h, sizeof(float), n_ids * HIDDEN, dbg);
            fclose(dbg);
            fprintf(stderr, "  DEBUG: dumped layer 0 inputs/output\n");
        }
        if (li % 7 == 6 || li == NUM_LAYERS - 1)
            fprintf(stderr, "  layer %d done\n", li);
    }
    free(ple_slice);
    free(h_buf2);
    free(kv_sliding.k);
    free(kv_sliding.v);
    free(kv_full.k);
    free(kv_full.v);

    /* Final norm */
    float *h_final = (float *) malloc(n_ids * HIDDEN * sizeof(float));
    rmsnorm_fp32(n_ids, HIDDEN, h, final_norm_w, RMS_EPS, h_final);

    /* LM head */
    float *logits = (float *) malloc(n_ids * (size_t) VOCAB * sizeof(float));
    fprintf(stderr, "LM head matmul (%zu x %d → %d)...\n", n_ids, HIDDEN, VOCAB);
    linear_fp32(n_ids, HIDDEN, VOCAB, h_final, lm_head_w, nullptr, logits);

    /* Logit softcap: tanh(x/cap) * cap */
    for (size_t i = 0; i < n_ids * (size_t) VOCAB; i++) {
        logits[i] = tanhf(logits[i] / LOGIT_SOFTCAP) * LOGIT_SOFTCAP;
    }

    /* Write logits */
    write_bin(argv[3], logits, n_ids * (size_t) VOCAB * sizeof(float));
    fprintf(stderr,
            "wrote %zu × %d logits = %.1f MB\n",
            n_ids,
            VOCAB,
            n_ids * (size_t) VOCAB * 4.0 / 1e6);

    free(logits);
    free(h_final);
    free(h);
    free(per_layer_inputs);
    free(model_proj_w);
    free(model_proj_norm_w);
    free(embed_table);
    free(ple_table_data);
    free(final_norm_w);
    free(lm_head_w);
    free(ids);
    for (int i = 0; i < NUM_LAYERS; i++)
        free_layer_w(&layers[i]);
    gguf_close(ctx);
    return 0;
}
