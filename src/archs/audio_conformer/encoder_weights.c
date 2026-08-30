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
    return bf16_alloc_fp32(n, t->data);
}

/* Read a scalar bf16 tensor (shape ()) → float. */
static bool load_bf16_scalar(struct st_ctx *sf, const char *name, float *out) {
    const struct st_tensor_t *t = st_get(sf, name);
    if (!t || t->dtype != ST_DTYPE_BF16 || t->nbytes != 2) {
        fprintf(stderr, "audio_encoder: bad scalar %s\n", name);
        return false;
    }
    /* memcpy, not *(const uint16_t *): tensor data sits at an arbitrary byte
     * offset in the mmap and may be odd-aligned. */
    uint16_t raw;
    memcpy(&raw, t->data, sizeof raw);
    *out = bf16_to_fp32(raw);
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

    cl->prec = prec;
    if (prec != AUDIO_PREC_FP32) {
        quantize_clippable_w8(cl, out_dim, in_dim);
        /* Drop the FP32 copy after quant — ~450 MB on a Pi 5. FP32 A/B
         * runs use GEIST_AUDIO_FORCE_QUANT / the per-class env knobs at
         * load time instead of keeping both copies resident. */
        safe_free((void **) &cl->w);
        cl->w = nullptr;
    }
    return true;
}

/* forward decl */

static bool env_flag(const char *name, bool fallback) {
    const char *s = getenv(name);
    if (s == nullptr || s[0] == '\0')
        return fallback;
    return s[0] == '1';
}

/* THE precision decision, resolved once per encoder create (#251) — env
 * mutated afterwards changes nothing, and the A/B parity tests get their
 * per-encoder configs by setting env between creates.
 *
 * force_fp32: Apple Silicon's cblas_sgemm via Accelerate (AMX) beats the
 * hand-rolled W8 kernels at these matmul sizes, so Apple defaults every
 * ClippableLinear to FP32; GEIST_AUDIO_FORCE_QUANT=1 opts back in.
 * attn/lconv W8A8 default ON since their quality gates went green
 * (parity, chat e2e, LibriSpeech WER — see PI5-audio.md); '0' opts out.
 * layer_limit: quantization restricted to layers [0, N) for the
 * quant-noise mitigation experiments. */
struct audio_prec_policy audio_prec_policy_resolve(void) {
    struct audio_prec_policy p = {
            .attn_w8a8   = env_flag("GEIST_AUDIO_ATTN_W8A8", true),
            .lconv_w8a8  = env_flag("GEIST_AUDIO_LCONV_W8A8", true),
            .force_fp32  = false,
            .layer_limit = N_LAYERS,
    };
#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
    p.force_fp32 = getenv("GEIST_AUDIO_FORCE_QUANT") == nullptr;
#endif
    const char *s = getenv("GEIST_AUDIO_W8A8_LAYER_LIMIT");
    if (s != nullptr) {
        p.layer_limit = atoi(s);
        if (p.layer_limit < 0)
            p.layer_limit = 0;
        if (p.layer_limit > N_LAYERS)
            p.layer_limit = N_LAYERS;
    }
    return p;
}

/* Pick the effective precision for one weight class on one layer. */
static enum audio_linear_prec pick_prec(const struct audio_prec_policy *pol,
                                        int                             layer_idx,
                                        bool                            class_on,
                                        enum audio_linear_prec          quant_prec,
                                        enum audio_linear_prec          plain_prec) {
    if (pol->force_fp32)
        return AUDIO_PREC_FP32;
    const bool active = class_on && layer_idx < pol->layer_limit;
    return active ? quant_prec : plain_prec;
}

static bool load_ffn(struct st_ctx                  *sf,
                     int                             layer_idx,
                     const char                     *prefix,
                     struct FFN                     *ffn,
                     const struct audio_prec_policy *pol) {
    char buf[384];
    /* FFN W8A8 is unconditional (modulo layer_limit/force_fp32) — it has
     * been the default since f6156e74 (per-tensor mixed precision). */
    const enum audio_linear_prec ffn_prec =
            pick_prec(pol, layer_idx, true, AUDIO_PREC_W8A8, AUDIO_PREC_FP32);
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
static bool load_attn(struct st_ctx                  *sf,
                      int                             layer_idx,
                      const char                     *prefix,
                      struct Attn                    *a,
                      const struct audio_prec_policy *pol) {
    char buf[384];
    /* Attn W8A8 vs W8A32 per the resolved policy. Per bib.md A6 (4-bit
     * Conformer with Native QAT, Google 2024) INT8-only shows 0.87% WER
     * loss without finetune. */
    const enum audio_linear_prec attn_prec =
            pick_prec(pol, layer_idx, pol->attn_w8a8, AUDIO_PREC_W8A8, AUDIO_PREC_W8A32);

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

static bool load_lconv(struct st_ctx                  *sf,
                       int                             layer_idx,
                       const char                     *prefix,
                       struct LConv                   *lc,
                       const struct audio_prec_policy *pol) {
    char buf[384];
    /* LightConv linear_start/linear_end W8A8 vs FP32 per the resolved
     * policy (depthwise conv stays FP32 — per-channel causal, no quant
     * kernel). */
    const enum audio_linear_prec prec =
            pick_prec(pol, layer_idx, pol->lconv_w8a8, AUDIO_PREC_W8A8, AUDIO_PREC_FP32);

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

bool load_layer(struct st_ctx                  *sf,
                int                             layer_idx,
                struct ConformerLayer          *L,
                const struct audio_prec_policy *pol) {
    char p[64];
    snprintf(p, sizeof(p), "model.audio_tower.layers.%d.", layer_idx);
    char sub[384];
    snprintf(sub, sizeof(sub), "%sfeed_forward1.", p);
    if (!load_ffn(sf, layer_idx, sub, &L->ff1, pol))
        return false;
    snprintf(sub, sizeof(sub), "%sfeed_forward2.", p);
    if (!load_ffn(sf, layer_idx, sub, &L->ff2, pol))
        return false;
    snprintf(sub, sizeof(sub), "%sself_attn.", p);
    if (!load_attn(sf, layer_idx, sub, &L->attn, pol))
        return false;
    snprintf(sub, sizeof(sub), "%slconv1d.", p);
    if (!load_lconv(sf, layer_idx, sub, &L->lconv, pol))
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
