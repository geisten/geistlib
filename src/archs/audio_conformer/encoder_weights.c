/*
 * src/archs/audio_conformer/encoder_weights.c — weight loading, quantization, and teardown.
 *
 * Layer: ARCHITECTURE (audio_conformer). Split from the former
 * monolithic audio_encoder.c; pure moves, no behavior change.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "encoder_internal.h"

/* Intra-module forward decls (definition order preserved from the split). */
static void quantize_clippable_w8(struct ClippableLinear *cl, size_t out_dim, size_t in_dim);
static int  w8a8_layer_limit(void);

float *load_bf16(struct st_ctx *sf, const char *name, size_t expect_elems) {
    const struct st_tensor_t *t = st_get(sf, name);
    if (!t) {
        fprintf(stderr, "audio_encoder: missing %s\n", name);
        return nullptr;
    }
    if (t->dtype != ST_DTYPE_BF16) {
        fprintf(stderr, "audio_encoder: %s expected BF16, got %s\n", name, st_dtype_name(t->dtype));
        return nullptr;
    }
    size_t n = t->nbytes / 2;
    if (n != expect_elems) {
        fprintf(stderr, "audio_encoder: %s expected %zu elems, got %zu\n", name, expect_elems, n);
        return nullptr;
    }
    return bf16_alloc_fp32((const uint16_t *) t->data, n);
}

/* Read a scalar bf16 tensor (shape ()) → float. */
static bool load_bf16_scalar(struct st_ctx *sf, const char *name, float *out) {
    const struct st_tensor_t *t = st_get(sf, name);
    if (!t || t->dtype != ST_DTYPE_BF16 || t->nbytes != 2) {
        fprintf(stderr, "audio_encoder: bad scalar %s\n", name);
        return false;
    }
    /* Reuse bf16_to_fp32 from gemma4_kernels.h */
    *out = bf16_to_fp32(*(const uint16_t *) t->data);
    return true;
}

static bool load_clippable(struct st_ctx          *sf,
                           const char             *prefix,
                           struct ClippableLinear *cl,
                           size_t                  out_dim,
                           size_t                  in_dim,
                           enum audio_linear_prec  prec) {
    char buf[384];
    snprintf(buf, sizeof(buf), "%slinear.weight", prefix);
    cl->w = load_bf16(sf, buf, out_dim * in_dim);
    if (!cl->w)
        return false;
    snprintf(buf, sizeof(buf), "%sinput_min", prefix);
    if (!load_bf16_scalar(sf, buf, &cl->input_min))
        return false;
    snprintf(buf, sizeof(buf), "%sinput_max", prefix);
    if (!load_bf16_scalar(sf, buf, &cl->input_max))
        return false;
    snprintf(buf, sizeof(buf), "%soutput_min", prefix);
    if (!load_bf16_scalar(sf, buf, &cl->output_min))
        return false;
    snprintf(buf, sizeof(buf), "%soutput_max", prefix);
    if (!load_bf16_scalar(sf, buf, &cl->output_max))
        return false;

#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
    /* Apple Silicon's cblas_sgemm via Accelerate (AMX) beats hand-rolled
     * NEON W8 kernels for these matmul sizes — same lesson as the IQ
     * flat-decode cache. Default to FP32 on Mac; GEIST_AUDIO_FORCE_QUANT=1
     * opts back in for A/B benchmarks. Non-Apple keeps the per-class
     * precision (Pi 5 NEON quant beats naive fp32 loop ~3-4×). */
    if (getenv("GEIST_AUDIO_FORCE_QUANT") == nullptr)
        prec = AUDIO_PREC_FP32;
#endif

    cl->prec = prec;
    if (prec != AUDIO_PREC_FP32) {
        quantize_clippable_w8(cl, out_dim, in_dim);
#if !defined(GEIST_AUDIO_KEEP_FP32)
        safe_free((void **) &cl->w);
        cl->w = nullptr;
#endif
    }
    return true;
}

/* forward decl */

static bool load_ffn(struct st_ctx *sf, int layer_idx, const char *prefix, struct FFN *ffn) {
    char buf[384];
    /* FFN W8A8 is the default since f6156e74 (per-tensor mixed precision).
     * GEIST_AUDIO_W8A8_LAYER_LIMIT=N drops layers >= N back to FP32 for
     * mitigation experiments — without it FFN is the dominant drift
     * contributor across all 12 layers. */
    const bool                   layer_active = (layer_idx < w8a8_layer_limit());
    const enum audio_linear_prec ffn_prec     = layer_active ? AUDIO_PREC_W8A8 : AUDIO_PREC_FP32;
    snprintf(buf, sizeof(buf), "%sffw_layer_1.", prefix);
    if (!load_clippable(sf, buf, &ffn->ffw1, FFN_INTER, AUDIO_HIDDEN, ffn_prec))
        return false;
    snprintf(buf, sizeof(buf), "%sffw_layer_2.", prefix);
    if (!load_clippable(sf, buf, &ffn->ffw2, AUDIO_HIDDEN, FFN_INTER, ffn_prec))
        return false;
    snprintf(buf, sizeof(buf), "%spre_layer_norm.weight", prefix);
    ffn->pre_norm = load_bf16(sf, buf, AUDIO_HIDDEN);
    snprintf(buf, sizeof(buf), "%spost_layer_norm.weight", prefix);
    ffn->post_norm = load_bf16(sf, buf, AUDIO_HIDDEN);
    return ffn->pre_norm && ffn->post_norm;
}

/* GEIST_AUDIO_W8A8_LAYER_LIMIT=N restricts W8A8 (Attn and LConv) to layers
 * [0, N). Default N=N_LAYERS=12 → all layers W8A8 (legacy behaviour). Use
 * this to test the mitigation hypothesis that the structural drift at
 * positions [53, 69, 88, 114] is driven by cumulative quant-noise
 * resonance across all 12 Conformer layers - if dropping the last few
 * layers from W8A8 weakens or shifts the drift, the hypothesis holds. */
static int w8a8_layer_limit(void) {
    /* Read per call (not latched) — see load_attn. */
    const char *s     = getenv("GEIST_AUDIO_W8A8_LAYER_LIMIT");
    int         limit = (s != nullptr) ? atoi(s) : N_LAYERS;
    if (limit < 0)
        limit = 0;
    if (limit > N_LAYERS)
        limit = N_LAYERS;
    return limit;
}

static bool load_attn(struct st_ctx *sf, int layer_idx, const char *prefix, struct Attn *a) {
    char buf[384];
    /* GEIST_AUDIO_ATTN_W8A8=1 promotes Attn projections from W8A32 to W8A8.
     * Per bib.md A6 (4-bit Conformer with Native QAT, Google 2024) INT8-only
     * shows 0.87% WER loss without finetune — acceptable for the streaming
     * path where the next-token LM dominates output quality anyway. */
    /* Read per call (not latched): lets one process load A/B encoders for
     * the quant-parity test; the cost is nothing next to the weight load. */
    const char                  *attn_env     = getenv("GEIST_AUDIO_ATTN_W8A8");
    const int                    attn_w8a8    = (attn_env != nullptr && attn_env[0] == '1') ? 1 : 0;
    const bool                   layer_active = (layer_idx < w8a8_layer_limit());
    const enum audio_linear_prec attn_prec =
            (attn_w8a8 && layer_active) ? AUDIO_PREC_W8A8 : AUDIO_PREC_W8A32;

    snprintf(buf, sizeof(buf), "%sq_proj.", prefix);
    if (!load_clippable(sf, buf, &a->q_proj, AUDIO_HIDDEN, AUDIO_HIDDEN, attn_prec))
        return false;
    snprintf(buf, sizeof(buf), "%sk_proj.", prefix);
    if (!load_clippable(sf, buf, &a->k_proj, AUDIO_HIDDEN, AUDIO_HIDDEN, attn_prec))
        return false;
    snprintf(buf, sizeof(buf), "%sv_proj.", prefix);
    if (!load_clippable(sf, buf, &a->v_proj, AUDIO_HIDDEN, AUDIO_HIDDEN, attn_prec))
        return false;
    snprintf(buf, sizeof(buf), "%spost.", prefix);
    if (!load_clippable(sf, buf, &a->post, AUDIO_HIDDEN, AUDIO_HIDDEN, attn_prec))
        return false;
    snprintf(buf, sizeof(buf), "%srelative_k_proj.weight", prefix);
    a->relative_k_proj = load_bf16(sf, buf, (size_t) AUDIO_HIDDEN * AUDIO_HIDDEN);
    snprintf(buf, sizeof(buf), "%sper_dim_scale", prefix);
    a->per_dim_scale = load_bf16(sf, buf, HEAD_DIM);
    return a->relative_k_proj && a->per_dim_scale;
}

static bool load_lconv(struct st_ctx *sf, int layer_idx, const char *prefix, struct LConv *lc) {
    char buf[384];
    /* GEIST_AUDIO_LCONV_W8A8=1 promotes the LightConv linear_start/linear_end
     * from FP32 to W8A8 (depthwise conv stays FP32 — it's per-channel
     * causal and the kernel doesn't have a quant path). Same rationale
     * as Attn W8A8 (bib.md A6). */
    /* Read per call (not latched) — see load_attn. */
    const char                  *lconv_env  = getenv("GEIST_AUDIO_LCONV_W8A8");
    const int                    lconv_w8a8 = (lconv_env != nullptr && lconv_env[0] == '1') ? 1 : 0;
    const bool                   layer_active = (layer_idx < w8a8_layer_limit());
    const enum audio_linear_prec prec =
            (lconv_w8a8 && layer_active) ? AUDIO_PREC_W8A8 : AUDIO_PREC_FP32;

    snprintf(buf, sizeof(buf), "%slinear_start.", prefix);
    if (!load_clippable(sf, buf, &lc->linear_start, 2 * AUDIO_HIDDEN, AUDIO_HIDDEN, prec))
        return false;
    snprintf(buf, sizeof(buf), "%slinear_end.", prefix);
    if (!load_clippable(sf, buf, &lc->linear_end, AUDIO_HIDDEN, AUDIO_HIDDEN, prec))
        return false;
    snprintf(buf, sizeof(buf), "%sdepthwise_conv1d.weight", prefix);
    lc->depthwise = load_bf16(sf, buf, (size_t) AUDIO_HIDDEN * CONV_KERNEL);
    snprintf(buf, sizeof(buf), "%spre_layer_norm.weight", prefix);
    lc->pre_norm = load_bf16(sf, buf, AUDIO_HIDDEN);
    snprintf(buf, sizeof(buf), "%sconv_norm.weight", prefix);
    lc->conv_norm = load_bf16(sf, buf, AUDIO_HIDDEN);
    return lc->depthwise && lc->pre_norm && lc->conv_norm;
}

bool load_layer(struct st_ctx *sf, int layer_idx, struct ConformerLayer *L) {
    char p[64];
    snprintf(p, sizeof(p), "model.audio_tower.layers.%d.", layer_idx);
    char sub[384];
    snprintf(sub, sizeof(sub), "%sfeed_forward1.", p);
    if (!load_ffn(sf, layer_idx, sub, &L->ff1))
        return false;
    snprintf(sub, sizeof(sub), "%sfeed_forward2.", p);
    if (!load_ffn(sf, layer_idx, sub, &L->ff2))
        return false;
    snprintf(sub, sizeof(sub), "%sself_attn.", p);
    if (!load_attn(sf, layer_idx, sub, &L->attn))
        return false;
    snprintf(sub, sizeof(sub), "%slconv1d.", p);
    if (!load_lconv(sf, layer_idx, sub, &L->lconv))
        return false;
    snprintf(sub, sizeof(sub), "%snorm_pre_attn.weight", p);
    L->norm_pre_attn = load_bf16(sf, sub, AUDIO_HIDDEN);
    snprintf(sub, sizeof(sub), "%snorm_post_attn.weight", p);
    L->norm_post_attn = load_bf16(sf, sub, AUDIO_HIDDEN);
    snprintf(sub, sizeof(sub), "%snorm_out.weight", p);
    L->norm_out = load_bf16(sf, sub, AUDIO_HIDDEN);
    return L->norm_pre_attn && L->norm_post_attn && L->norm_out;
}

/* Quantize an FP32 weight matrix (out, in) into per-output-row symmetric INT8.
 * Stores the int8 weights and per-row fp32 scales into cl. Original FP32 stays
 * intact; the dispatcher in clip_linear_apply chooses the path at runtime. */
static void quantize_clippable_w8(struct ClippableLinear *cl, size_t out_dim, size_t in_dim) {
    cl->w_q8     = heap_alloc_array_aligned(int8_t, out_dim *in_dim);
    cl->w_scales = heap_alloc_array_aligned(float, out_dim);
    for (size_t c = 0; c < out_dim; c++) {
        const float *row  = cl->w + c * in_dim;
        float        amax = 0.0f;
        for (size_t k = 0; k < in_dim; k++) {
            float a = fabsf(row[k]);
            if (a > amax)
                amax = a;
        }
        float scale = amax / 127.0f;
        if (scale == 0.0f)
            scale = 1.0f;
        cl->w_scales[c]  = scale;
        const float inv  = 1.0f / scale;
        int8_t     *qrow = cl->w_q8 + c * in_dim;
        for (size_t k = 0; k < in_dim; k++)
            qrow[k] = (int8_t) lrintf(row[k] * inv);
    }
}

static void free_clippable(struct ClippableLinear *cl) {
    safe_free((void **) &cl->w);
    safe_free((void **) &cl->w_q8);
    safe_free((void **) &cl->w_scales);
    cl->w        = nullptr;
    cl->w_q8     = nullptr;
    cl->w_scales = nullptr;
}

static void free_ffn(struct FFN *f) {
    free_clippable(&f->ffw1);
    free_clippable(&f->ffw2);
    safe_free((void **) &f->pre_norm);
    safe_free((void **) &f->post_norm);
}

static void free_attn(struct Attn *a) {
    free_clippable(&a->q_proj);
    free_clippable(&a->k_proj);
    free_clippable(&a->v_proj);
    free_clippable(&a->post);
    safe_free((void **) &a->relative_k_proj);
    safe_free((void **) &a->per_dim_scale);
}

static void free_lconv(struct LConv *l) {
    free_clippable(&l->linear_start);
    free_clippable(&l->linear_end);
    safe_free((void **) &l->depthwise);
    safe_free((void **) &l->pre_norm);
    safe_free((void **) &l->conv_norm);
}

void free_layer(struct ConformerLayer *L) {
    free_ffn(&L->ff1);
    free_ffn(&L->ff2);
    free_attn(&L->attn);
    free_lconv(&L->lconv);
    safe_free((void **) &L->norm_pre_attn);
    safe_free((void **) &L->norm_post_attn);
    safe_free((void **) &L->norm_out);
}
