/*
 * audio_encoder — Gemma 4 audio tower bringup module.
 *
 * Phase 2: only the subsample-conv-projection stage is wired up.
 *   input  : (T, 128)   log-mel
 *   conv1  : (1→128, 3×3, stride 2, pad 1)  + LayerNorm(128) + ReLU  → (128, T/2, 64)
 *   conv2  : (128→32, 3×3, stride 2, pad 1) + LayerNorm(32)  + ReLU  → (32,  T/4, 32)
 *   reshape: (T/4, 1024)
 *   linear : (1024, 1024) no bias                                     → (T/4, 1024)
 *
 * Weights are bf16 in the safetensors; we dequant to fp32 once at create
 * time and own the buffers thereafter.
 */
#include "audio_encoder.h"
#include "heap.h"

#include "audio_kernels.h"
#include "gemma4_kernels.h"
#include "mel_pipeline.h"
#include "safetensors_reader.h"

/* GCC -Wformat-truncation does worst-case analysis on `char*` prefixes
 * passed to snprintf — for our weight-name builders the prefixes are
 * bounded ~100 chars by construction (model.audio_tower.layers.NN....),
 * but GCC can't prove that and warns about hypothetical overflows.
 * Scoped-disable instead of growing buffers without limit. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* Encoder-stage timing accumulators. Compile-out by default; enable with
 * -DGEIST_AUDIO_PROFILE. Sums elapsed time per stage across one segment. */
#ifdef GEIST_AUDIO_PROFILE
struct ae_stage_timer {
    double   sum_ms;
    uint64_t count;
};
static struct ae_stage_timer g_ae_subsample, g_ae_pos_emb, g_ae_layer_total, g_ae_ffn1,
        g_ae_norm_pre_attn, g_ae_attn, g_ae_norm_post_attn, g_ae_lconv, g_ae_ffn2, g_ae_norm_out,
        g_ae_output_proj, g_ae_embed_proj;
static double ae_now_ms_(void) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double) tv.tv_sec * 1000.0 + (double) tv.tv_usec / 1000.0;
}
#define AE_TIC() const double _ae_t0 = ae_now_ms_()
#define AE_TOC(stg)                            \
    do {                                       \
        (stg).sum_ms += ae_now_ms_() - _ae_t0; \
        (stg).count++;                         \
    } while (0)
static void ae_profile_print_and_reset(void) {
    struct ae_stage_timer *s[] = {
            &g_ae_subsample,
            &g_ae_pos_emb,
            &g_ae_layer_total,
            &g_ae_ffn1,
            &g_ae_norm_pre_attn,
            &g_ae_attn,
            &g_ae_norm_post_attn,
            &g_ae_lconv,
            &g_ae_ffn2,
            &g_ae_norm_out,
            &g_ae_output_proj,
            &g_ae_embed_proj,
    };
    const char *names[] = {
            "subsample",
            "pos_emb+mask",
            "all_layers_TOTAL",
            "  ffn1",
            "  norm_pre_attn",
            "  attn",
            "  norm_post_attn",
            "  lconv",
            "  ffn2",
            "  norm_out",
            "output_proj",
            "embed_audio_proj",
    };
    double total = 0;
    for (size_t i = 0; i < sizeof s / sizeof s[0]; i++)
        total += s[i]->sum_ms;
    fprintf(stderr, "  audio-encoder profile (one segment):\n");
    for (size_t i = 0; i < sizeof s / sizeof s[0]; i++) {
        double pct = total > 0 ? 100.0 * s[i]->sum_ms / total : 0.0;
        fprintf(stderr,
                "    %-22s %8.2f ms  %5.1f%%  (%llu calls)\n",
                names[i],
                s[i]->sum_ms,
                pct,
                (unsigned long long) s[i]->count);
    }
    fprintf(stderr, "    %-22s %8.2f ms\n", "TOTAL (sum)", total);
    for (size_t i = 0; i < sizeof s / sizeof s[0]; i++) {
        s[i]->sum_ms = 0;
        s[i]->count  = 0;
    }
}
#else
#define AE_TIC() (void) 0
#define AE_TOC(stg) (void) 0
static inline void ae_profile_print_and_reset(void) {
}
#endif

#define AUDIO_HIDDEN 1024
#define MEL_DIM 128
#define LN_EPS 1e-6f
#define RMS_EPS 1e-6f
#define FFN_INTER 4096
#define N_HEADS 8
#define HEAD_DIM 128 /* AUDIO_HIDDEN / N_HEADS */
#define CHUNK_SIZE 12
#define MAX_PAST_HORIZON 12 /* attention_context_left - 1 */
#define MAX_FUTURE 0
#define CONTEXT_SIZE (CHUNK_SIZE + MAX_PAST_HORIZON + MAX_FUTURE) /* 24 */
#define POS_LEN (CONTEXT_SIZE / 2 + 1)                            /* 13 */
#define CONV_KERNEL 5
#define ATTN_SOFTCAP 50.0f
#define N_LAYERS 12

/* Per-tensor precision tag, set at load time based on the call-site class
 * (struct FFN macaron layers → W8A8, struct Attn projections → W8A32, struct LConv → FP32).
 * Pi-5 profiling (benchmark/BENCHMARK_PI5.md) shows struct FFN dominates encoder cost,
 * so W8A8 there. struct Attn is quant-sensitive (relative position embeddings,
 * softmax) → W8A32 keeps activations FP32 for safety. struct LConv stays FP32 —
 * depthwise convs don't benefit from the int8 dot path. */
enum audio_linear_prec {
    AUDIO_PREC_FP32  = 0,
    AUDIO_PREC_W8A32 = 1,
    AUDIO_PREC_W8A8  = 2,
};

struct ClippableLinear {
    /* `w` stays FP32 only when GEIST_AUDIO_KEEP_FP32 is defined at build
     * time (dev/A-B benchmarking). Default release builds drop it after
     * quant to save ~450 MB on Pi 5. `w_q8` / `w_scales` are populated
     * iff prec != AUDIO_PREC_FP32. */
    float                 *w;
    int8_t                *w_q8;
    float                 *w_scales;
    float                  input_min, input_max, output_min, output_max;
    enum audio_linear_prec prec;
};

struct FFN {
    struct ClippableLinear ffw1;      /* (FFN_INTER, AUDIO_HIDDEN) */
    struct ClippableLinear ffw2;      /* (AUDIO_HIDDEN, FFN_INTER) */
    float                 *pre_norm;  /* (AUDIO_HIDDEN,) */
    float                 *post_norm; /* (AUDIO_HIDDEN,) */
};

struct Attn {
    struct ClippableLinear q_proj, k_proj, v_proj, post;
    float                 *relative_k_proj; /* (AUDIO_HIDDEN, AUDIO_HIDDEN) — no clip */
    float                 *per_dim_scale;   /* (HEAD_DIM,) */
};

struct LConv {
    struct ClippableLinear linear_start; /* (2*AUDIO_HIDDEN, AUDIO_HIDDEN) */
    struct ClippableLinear linear_end;   /* (AUDIO_HIDDEN, AUDIO_HIDDEN) */
    float                 *depthwise;    /* (AUDIO_HIDDEN, CONV_KERNEL) — depthwise weights */
    float                 *pre_norm;     /* (AUDIO_HIDDEN,) */
    float                 *conv_norm;    /* (AUDIO_HIDDEN,) */
};

struct ConformerLayer {
    struct FFN   ff1, ff2;
    struct Attn  attn;
    struct LConv lconv;
    float       *norm_pre_attn; /* (AUDIO_HIDDEN,) */
    float       *norm_post_attn;
    float       *norm_out;
};

#define TEXT_HIDDEN 1536
#define OUTPUT_PROJ_DIMS AUDIO_SOFT_TOKEN_DIM /* 1536 */

/* PCM-buffer cap = 30s × 16 kHz = 480 000 samples (matches model's
 * audio_seq_length=750 × 40ms/token). */
#define PCM_BUFFER_CAP (16000 * 30)
/* Max mel frames for a PCM_BUFFER_CAP-second utterance (~30 s × 50 Hz frames + 1). */
#define MEL_BUF_CAP ((PCM_BUFFER_CAP / 160) + 4)
/* Max sub-tokens = mel cap / 4 (two stride-2 convs in subsample) + ceiling slack. */
#define MAX_SUB_TOKENS ((MEL_BUF_CAP / 4) + 16)

/* === Phase 8b chunk-streaming state (docs/audio-chunk-streaming/plan.md) ===
 *
 * Per-layer K/V caches and LConv depthwise-history. Lets push_pcm advance
 * Conformer layers block-by-block while audio is still arriving, instead of
 * running the whole encoder synchronously at end_input.
 *
 * Carrier ownership: one instance lives inside struct AudioEncoder, lazily
 * reset each segment. attn[*].n and lconv[*].hist content reflect the
 * processed sub-tokens for the current utterance. */
struct attn_kv_cache {
    float *k; /* (MAX_SUB_TOKENS, AUDIO_HIDDEN) — scaled K projections */
    float *v; /* (MAX_SUB_TOKENS, AUDIO_HIDDEN) — V projections */
    size_t n; /* sub-tokens cached so far */
};

struct lconv_state {
    /* Depthwise causal conv kernel=5 → keep the prior 4 sub-tokens' h after
     * GLU, stored row-major (CONV_KERNEL-1, AUDIO_HIDDEN). New entries
     * pushed at the tail via memmove. */
    float *hist;     /* ((CONV_KERNEL-1), AUDIO_HIDDEN) */
    size_t n_filled; /* 0..CONV_KERNEL-1 — how many history rows are populated */
};

/* Phase-3 incremental subsample state. Caches the (128, T_out0, 64) and
 * (32, T_out1, 32) intermediate conv2d outputs so that pushing more mel
 * extends them in-place instead of re-running the full subsample chain.
 * Conv2d outputs are stable for already-computed time positions as the
 * input grows - the kernel taps are at deterministic offsets from oh,
 * not anchored to h_in. */
#define SUBS_T_OUT0_CAP ((MEL_BUF_CAP / 2) + 2)
#define SUBS_T_OUT1_CAP ((SUBS_T_OUT0_CAP / 2) + 2)
#define SUBS_W_OUT0 64
#define SUBS_W_OUT1 32
#define SUBS_L0_CHANNELS 128
#define SUBS_L1_CHANNELS 32

struct subs_cache {
    /* Layer-0 conv intermediate, (128, T_out0, 64). Stores post-LN+ReLU
     * values - same as what subsample_layer's `out` buffer holds. */
    float *l0;
    size_t n_t_out0; /* # of valid time positions in l0 */
    /* Layer-1 conv intermediate, (32, T_out1, 32), post-LN+ReLU. */
    float *l1;
    size_t n_t_out1;
    size_t n_mel_seen; /* mel frames the cache reflects */
};

struct audio_stream_state {
    struct attn_kv_cache attn[N_LAYERS];
    struct lconv_state   lconv[N_LAYERS];

    /* Phase-3 cache: avoids re-running subsample on the full mel each push. */
    struct subs_cache subs;

    /* Accumulated subsample output (h_in for layer 0). Subsample re-runs on
     * the full mel each push for simplicity; sub-tokens before the new block
     * are already here and skipped at the Conformer-layer stage. */
    float *sub_buf;
    size_t n_sub_total; /* # sub-tokens currently in sub_buf */

    /* Accumulated soft tokens (output_proj + embed_audio applied). */
    float *soft; /* (MAX_SUB_TOKENS, AUDIO_SOFT_TOKEN_DIM) */
    size_t n_soft;
    size_t n_drained; /* pull drain pointer (Phase 2 worker path) */
};

struct AudioEncoder {
    struct st_ctx *sf;

    /* Subsample stage. */
    float *l0_conv;
    float *l0_norm;
    float *l1_conv;
    float *l1_norm;
    float *in_proj;

    /* 12 Conformer layers. */
    struct ConformerLayer layers[N_LAYERS];

    /* Final projections. */
    float *output_proj_w;    /* (1536, 1024) */
    float *output_proj_b;    /* (1536,) */
    float *embed_audio_proj; /* (1536, 1536) — no bias, no pre-norm scale */

    /* === Phase 8 streaming state === */
    pthread_mutex_t mtx;
    pthread_cond_t  cv;

    /* Internal mel pipeline + cached scratch (avoids reload per utterance). */
    struct MelState *mel;

    /* PCM buffer (int16, grows on push, capped at PCM_BUFFER_CAP). */
    int16_t *pcm_buf;
    size_t   pcm_len;
    size_t   pcm_cap;

    /* Streaming mel buffer — populated incrementally inside push_pcm as
     * each new 160-sample hop's worth of PCM arrives. compute_segment_locked
     * consumes this directly instead of re-running mel_frame_compute on the
     * full PCM buffer (Phase C: overlaps mel work with PCM capture). */
    float *mel_buf;        /* (MEL_BUF_CAP, MEL_N_MEL) */
    size_t mel_n_computed; /* number of mel frames already produced */
    size_t mel_cap;

    /* Soft-token output queue (filled on first pull after end_input). */
    float *soft_tokens; /* (n_soft, AUDIO_SOFT_TOKEN_DIM) */
    size_t n_soft;
    size_t n_emitted; /* drain pointer */

    bool end_input_flag; /* user signalled no more PCM */
    bool computed_flag;  /* audio_encoder_run has executed for this segment */
    bool shutdown_flag;  /* destroy/shutdown in progress */

    /* === Phase 8b chunk-streaming state (lazy-allocated, see plan doc). === */
    struct audio_stream_state *stream;

    /* === Phase 2: background worker thread for streaming compute. ===
     * Active iff GEIST_AUDIO_STREAM=1 at create. Drives stream_push on a
     * snapshot of mel_n_computed each time the caller signals (push_pcm
     * after enough new frames, or end_input). */
    bool      stream_enabled;
    bool      worker_active;
    bool      worker_kick;  /* push_pcm signaled new work */
    bool      worker_final; /* end_input signaled */
    pthread_t worker_tid;
    size_t    worker_last_mel; /* mel-frame count at last fire */
};

/* Forward decls — defined further down with the streaming impl. */
static struct audio_stream_state *audio_stream_state_create(void);
static void                       audio_stream_state_destroy(struct audio_stream_state *s);
static void                       audio_stream_state_reset(struct audio_stream_state *s);
static bool                       stream_worker_start(struct AudioEncoder *a);
static void                       stream_worker_stop(struct AudioEncoder *a);

static float *load_bf16(struct st_ctx *sf, const char *name, size_t expect_elems) {
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

static void quantize_clippable_w8(struct ClippableLinear *cl, size_t out_dim, size_t in_dim);

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

static int w8a8_layer_limit(void); /* forward decl */

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
    static int limit = -1;
    if (limit < 0) {
        const char *s = getenv("GEIST_AUDIO_W8A8_LAYER_LIMIT");
        limit         = (s != nullptr) ? atoi(s) : N_LAYERS;
        if (limit < 0)
            limit = 0;
        if (limit > N_LAYERS)
            limit = N_LAYERS;
    }
    return limit;
}

static bool load_attn(struct st_ctx *sf, int layer_idx, const char *prefix, struct Attn *a) {
    char buf[384];
    /* GEIST_AUDIO_ATTN_W8A8=1 promotes Attn projections from W8A32 to W8A8.
     * Per bib.md A6 (4-bit Conformer with Native QAT, Google 2024) INT8-only
     * shows 0.87% WER loss without finetune — acceptable for the streaming
     * path where the next-token LM dominates output quality anyway. */
    static int attn_w8a8 = -1;
    if (attn_w8a8 < 0) {
        const char *s = getenv("GEIST_AUDIO_ATTN_W8A8");
        attn_w8a8     = (s != nullptr && s[0] == '1') ? 1 : 0;
    }
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
    static int lconv_w8a8 = -1;
    if (lconv_w8a8 < 0) {
        const char *s = getenv("GEIST_AUDIO_LCONV_W8A8");
        lconv_w8a8    = (s != nullptr && s[0] == '1') ? 1 : 0;
    }
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

static bool load_layer(struct st_ctx *sf, int layer_idx, struct ConformerLayer *L) {
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

#if defined(__ARM_NEON)
/* W8A8 matmul: y[m,n] = scale_x * w_scales[n] * sum_k x_q8[m,k] * w_q8[n,k].
 * Activations are quantized once to int8 with a STATIC scale derived from
 * the precomputed clip range (input_min/max), then dotted against int8
 * weights via vdotq_s32. Caller passes scale_x_inv = 1 / scale_x. */
static void linear_w8a8(const int8_t *w_q8,
                        const float  *w_scales,
                        const float  *x,
                        float         scale_x_inv,
                        float         scale_x,
                        size_t        m,
                        size_t        in_dim,
                        size_t        out_dim,
                        float        *y) {
    int8_t           *x_q8  = heap_alloc_array_aligned(int8_t, m *in_dim);
    const float32x4_t inv_v = vdupq_n_f32(scale_x_inv);
    for (size_t i = 0; i < m; i++) {
        const float *xrow = x + i * in_dim;
        int8_t      *qrow = x_q8 + i * in_dim;
        size_t       k    = 0;
        for (; k + 16 <= in_dim; k += 16) {
            int32x4_t q0  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xrow + k + 0), inv_v));
            int32x4_t q1  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xrow + k + 4), inv_v));
            int32x4_t q2  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xrow + k + 8), inv_v));
            int32x4_t q3  = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xrow + k + 12), inv_v));
            int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            vst1q_s8(qrow + k, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
        }
        for (; k < in_dim; k++)
            qrow[k] = (int8_t) lrintf(xrow[k] * scale_x_inv);
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < out_dim; n++) {
        const int8_t *wrow   = w_q8 + n * in_dim;
        const float   wscale = w_scales[n];
        for (size_t i = 0; i < m; i++) {
            const int8_t *xrow = x_q8 + i * in_dim;
            int32x4_t     acc  = vdupq_n_s32(0);
            size_t        k    = 0;
            for (; k + 16 <= in_dim; k += 16) {
                acc = vdotq_s32(acc, vld1q_s8(wrow + k), vld1q_s8(xrow + k));
            }
            int32_t isum = vaddvq_s32(acc);
            for (; k < in_dim; k++)
                isum += (int32_t) wrow[k] * (int32_t) xrow[k];
            y[i * out_dim + n] = wscale * scale_x * (float) isum;
        }
    }
    safe_free((void **) &x_q8);
}

/* W8A32 matmul: y[m, n] = sum_k x[m, k] * (w_q8[n, k] * w_scales[n]).
 * w_q8 is row-major (out, in); x is row-major (m, in); y is row-major (m, out).
 * Each output row n reads its weight row once (in_dim int8 bytes) and dots
 * against M activation rows. NEON path converts int8 -> fp32 in-loop. */
static void linear_w8a32(const int8_t *w_q8,
                         const float  *w_scales,
                         const float  *x,
                         size_t        m,
                         size_t        in_dim,
                         size_t        out_dim,
                         float        *y) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < out_dim; n++) {
        const int8_t *wrow   = w_q8 + n * in_dim;
        const float   wscale = w_scales[n];
        for (size_t i = 0; i < m; i++) {
            const float *xrow = x + i * in_dim;
            float32x4_t  acc0 = vdupq_n_f32(0.0f);
            float32x4_t  acc1 = vdupq_n_f32(0.0f);
            float32x4_t  acc2 = vdupq_n_f32(0.0f);
            float32x4_t  acc3 = vdupq_n_f32(0.0f);
            size_t       k    = 0;
            for (; k + 16 <= in_dim; k += 16) {
                int8x16_t qv = vld1q_s8(wrow + k);
                /* int8x16 -> int16x8 x2 -> int32x4 x4 -> float32x4 x4 */
                int16x8_t   qa = vmovl_s8(vget_low_s8(qv));
                int16x8_t   qb = vmovl_s8(vget_high_s8(qv));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qa)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qa)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qb)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qb)));
                acc0           = vfmaq_f32(acc0, f0, vld1q_f32(xrow + k + 0));
                acc1           = vfmaq_f32(acc1, f1, vld1q_f32(xrow + k + 4));
                acc2           = vfmaq_f32(acc2, f2, vld1q_f32(xrow + k + 8));
                acc3           = vfmaq_f32(acc3, f3, vld1q_f32(xrow + k + 12));
            }
            float32x4_t sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float       dot = vaddvq_f32(sum);
            for (; k < in_dim; k++)
                dot += (float) wrow[k] * xrow[k];
            y[i * out_dim + n] = wscale * dot;
        }
    }
}
#else
static void linear_w8a32(const int8_t *w_q8,
                         const float  *w_scales,
                         const float  *x,
                         size_t        m,
                         size_t        in_dim,
                         size_t        out_dim,
                         float        *y) {
    for (size_t n = 0; n < out_dim; n++) {
        const float wscale = w_scales[n];
        for (size_t i = 0; i < m; i++) {
            float dot = 0.0f;
            for (size_t k = 0; k < in_dim; k++)
                dot += (float) w_q8[n * in_dim + k] * x[i * in_dim + k];
            y[i * out_dim + n] = wscale * dot;
        }
    }
}
#endif

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
static void free_layer(struct ConformerLayer *L) {
    free_ffn(&L->ff1);
    free_ffn(&L->ff2);
    free_attn(&L->attn);
    free_lconv(&L->lconv);
    safe_free((void **) &L->norm_pre_attn);
    safe_free((void **) &L->norm_post_attn);
    safe_free((void **) &L->norm_out);
}

struct AudioEncoder *audio_encoder_create(const char *safetensors_path) {
    const char    *err = nullptr;
    struct st_ctx *sf  = st_open(safetensors_path, &err);
    if (!sf) {
        fprintf(stderr, "audio_encoder: %s\n", err);
        return nullptr;
    }

    {
        const char *attn_env   = getenv("GEIST_AUDIO_ATTN_W8A8");
        const char *lconv_env  = getenv("GEIST_AUDIO_LCONV_W8A8");
        const char *attn_prec  = (attn_env && attn_env[0] == '1') ? "W8A8" : "W8A32";
        const char *lconv_prec = (lconv_env && lconv_env[0] == '1') ? "W8A8" : "FP32";
        fprintf(stderr,
                "audio_encoder: per-tensor precision (FFN=W8A8, Attn=%s, LConv=%s)"
#if defined(GEIST_AUDIO_KEEP_FP32)
                ", FP32 kept for A/B"
#endif
                "\n",
                attn_prec,
                lconv_prec);
    }

    struct AudioEncoder *a = heap_calloc_array_aligned(struct AudioEncoder, 1);
    a->sf                  = sf;

    const char *P = "model.audio_tower.subsample_conv_projection.";
    char        buf[384];
#define LOAD(field, suf, n)                         \
    do {                                            \
        snprintf(buf, sizeof(buf), "%s%s", P, suf); \
        a->field = load_bf16(sf, buf, n);           \
        if (!a->field) {                            \
            audio_encoder_destroy(a);               \
            return nullptr;                         \
        }                                           \
    } while (0)

    LOAD(l0_conv, "layer0.conv.weight", (size_t) 128 * 1 * 3 * 3);
    LOAD(l0_norm, "layer0.norm.weight", 128);
    LOAD(l1_conv, "layer1.conv.weight", (size_t) 32 * 128 * 3 * 3);
    LOAD(l1_norm, "layer1.norm.weight", 32);
    LOAD(in_proj, "input_proj_linear.weight", (size_t) AUDIO_HIDDEN * AUDIO_HIDDEN);

#undef LOAD

    /* Load all 12 Conformer layers' weights. */
    fprintf(stderr, "audio_encoder: loading %d Conformer layers...\n", N_LAYERS);
    for (int i = 0; i < N_LAYERS; i++) {
        if (!load_layer(sf, i, &a->layers[i])) {
            audio_encoder_destroy(a);
            return nullptr;
        }
    }
    fprintf(stderr, "audio_encoder: layers loaded.\n");

    /* Final projections. */
    a->output_proj_w = load_bf16(
            sf, "model.audio_tower.output_proj.weight", (size_t) OUTPUT_PROJ_DIMS * AUDIO_HIDDEN);
    a->output_proj_b    = load_bf16(sf, "model.audio_tower.output_proj.bias", OUTPUT_PROJ_DIMS);
    a->embed_audio_proj = load_bf16(sf,
                                    "model.embed_audio.embedding_projection.weight",
                                    (size_t) TEXT_HIDDEN * OUTPUT_PROJ_DIMS);
    if (!a->output_proj_w || !a->output_proj_b || !a->embed_audio_proj) {
        audio_encoder_destroy(a);
        return nullptr;
    }

    /* Streaming state: mutex/cv + PCM buffer. mel pipeline is created lazily
     * by the streaming path (audio_encoder_create only needs the safetensors
     * for weights; mel constants come from a separate file). */
    pthread_mutex_init(&a->mtx, nullptr);
    pthread_cond_init(&a->cv, nullptr);
    a->pcm_cap = PCM_BUFFER_CAP;
    a->pcm_buf = heap_alloc_array_aligned(int16_t, a->pcm_cap);
    a->mel_cap = MEL_BUF_CAP;
    a->mel_buf = heap_calloc_array_aligned(float, a->mel_cap *MEL_N_MEL);
    /* Phase 8b chunk-streaming state (per-layer K/V caches + LConv history).
     * Allocated eagerly so push_pcm doesn't pay heap-arena cost on the
     * audio path. Unused until the streaming forward (Phase 1b) lands. */
    a->stream = audio_stream_state_create();
    if (a->stream == nullptr) {
        audio_encoder_destroy(a);
        return nullptr;
    }
    /* Phase 2: opt-in streaming worker thread. Default off so existing
     * sync pull-after-end_input behaviour stays bit-identical until the
     * worker path is validated on Pi 5. */
    const char *env_stream = getenv("GEIST_AUDIO_STREAM");
    if (env_stream != nullptr && env_stream[0] == '1') {
        if (!stream_worker_start(a)) {
            fprintf(stderr,
                    "audio_encoder: GEIST_AUDIO_STREAM=1 set but worker "
                    "thread failed to start; falling back to sync path\n");
        }
    }
    return a;
}

void audio_encoder_destroy(struct AudioEncoder *a) {
    if (!a)
        return;
    /* Wake any blocked pulls before tearing down. */
    audio_encoder_shutdown(a);
    stream_worker_stop(a);
    pthread_mutex_destroy(&a->mtx);
    pthread_cond_destroy(&a->cv);
    safe_free((void **) &a->pcm_buf);
    safe_free((void **) &a->mel_buf);
    safe_free((void **) &a->soft_tokens);
    if (a->stream) {
        audio_stream_state_destroy(a->stream);
        a->stream = nullptr;
    }
    if (a->mel)
        mel_destroy(a->mel);
    safe_free((void **) &a->embed_audio_proj);
    safe_free((void **) &a->output_proj_b);
    safe_free((void **) &a->output_proj_w);
    for (int i = 0; i < N_LAYERS; i++)
        free_layer(&a->layers[i]);
    safe_free((void **) &a->in_proj);
    safe_free((void **) &a->l1_norm);
    safe_free((void **) &a->l1_conv);
    safe_free((void **) &a->l0_norm);
    safe_free((void **) &a->l0_conv);
    if (a->sf)
        st_close(a->sf);
    safe_free((void **) &a);
}

/* === Phase 8 streaming API ===================================================*/

/* Lazily create the mel pipeline. Constants path is a fixed default for now;
 * a future API can let callers override. */
static int ensure_mel(struct AudioEncoder *a) {
    if (a->mel)
        return 0;
    /* Constants live next to the audio_test_data dir of the bringup tree —
     * resolve relative to CWD to keep the C side tooling-agnostic. */
    const char *path = "audio_test_data/mel_constants.bin";
    a->mel           = mel_create(path);
    return a->mel ? 0 : -1;
}

int audio_encoder_push_pcm(struct AudioEncoder *a, const int16_t *samples, size_t n) {
    pthread_mutex_lock(&a->mtx);
    if (a->shutdown_flag || a->end_input_flag || a->pcm_len + n > a->pcm_cap) {
        pthread_mutex_unlock(&a->mtx);
        return -1;
    }
    memcpy(a->pcm_buf + a->pcm_len, samples, n * sizeof(int16_t));
    a->pcm_len += n;

    /* Advance mel pipeline as far as the now-larger PCM buffer allows.
     * Frame k uses padded[k*160 : k*160+320] where padded = 160 zeros +
     * pcm. So frame k requires pcm[0 .. (k+1)*160 - 1] available — i.e.
     * we need at least (k+1)*160 PCM samples to produce frame k. Phase
     * C: this work used to happen all-at-once after end_input; pushing
     * it inside push_pcm overlaps it with capture. */
    if (a->mel == nullptr)
        (void) ensure_mel(a); /* lazy first time */
    if (a->mel != nullptr) {
        float frame_pcm[MEL_FRAME_LENGTH];
        while (a->mel_n_computed < a->mel_cap && (a->mel_n_computed + 1) * 160 <= a->pcm_len) {
            const size_t k = a->mel_n_computed;
            for (size_t i = 0; i < MEL_FRAME_LENGTH; i++) {
                const long pi = (long) (k * 160 + i) - 160; /* unshift by left-pad */
                frame_pcm[i]  = (pi < 0) ? 0.0f : (float) a->pcm_buf[pi] / 32768.0f;
            }
            mel_frame_compute(a->mel, frame_pcm, a->mel_buf + k * MEL_N_MEL);
            a->mel_n_computed++;
        }
    }

    /* Phase 2: if the streaming worker is active and we've crossed a
     * sub-token-block boundary (~48 new mel frames = 12 sub-tokens) since
     * the last fire, kick the worker. This overlaps the Conformer compute
     * with the still-arriving PCM. */
    if (a->stream_enabled && a->worker_active && a->mel_n_computed >= a->worker_last_mel + 48) {
        a->worker_kick = true;
        pthread_cond_signal(&a->cv);
    }

    pthread_mutex_unlock(&a->mtx);
    return 0;
}

void audio_encoder_end_input(struct AudioEncoder *a) {
    pthread_mutex_lock(&a->mtx);
    a->end_input_flag = true;
    if (a->stream_enabled && a->worker_active) {
        a->worker_kick  = true;
        a->worker_final = true;
    }
    pthread_cond_broadcast(&a->cv);
    pthread_mutex_unlock(&a->mtx);
}

/* Run subsample + Conformer + projections on the pre-computed mel buffer
 * (populated incrementally by push_pcm — Phase C). Caller holds a->mtx;
 * we drop it during the heavy work and re-acquire afterwards. */
static int compute_segment_locked(struct AudioEncoder *a) {
    /* Snapshot mel frame count under the lock. PCM is needed only as a
     * fallback when something went wrong in the streaming path. */
    const size_t n_frames = a->mel_n_computed;
    const size_t n_mel    = n_frames + 1; /* +1 mirrors HF's padded extra frame */

    if (a->mel == nullptr || n_frames == 0) {
        /* No mel computed (e.g. push_pcm never ran the lazy mel init).
         * Skip — pull will surface zero soft-tokens. */
        a->computed_flag = true;
        pthread_cond_broadcast(&a->cv);
        return 0;
    }

    /* Hand the mel buffer + mask to the heavy stages. The mel buffer is
     * owned by struct AudioEncoder and stable across the unlock — only push_pcm
     * writes to it, and end_input has set end_input_flag which blocks
     * further pushes. */
    float *mel_view = a->mel_buf;
    bool  *mask     = heap_calloc_array_aligned(bool, n_mel);
    for (size_t i = 0; i < n_frames; i++)
        mask[i] = true;
    pthread_mutex_unlock(&a->mtx);

    const size_t max_soft = (n_mel + 3) / 4 + 4;
    float       *soft     = heap_alloc_array_aligned(float, max_soft *AUDIO_SOFT_TOKEN_DIM);
    size_t       n_soft   = audio_encoder_run(a, mel_view, mask, n_mel, soft);
    safe_free((void **) &mask);

    pthread_mutex_lock(&a->mtx);
    a->soft_tokens   = soft;
    a->n_soft        = n_soft;
    a->n_emitted     = 0;
    a->computed_flag = true;
    pthread_cond_broadcast(&a->cv);
    return 0;
}

/* Compute absolute deadline `timeout_ms` from now. */
static void make_deadline(struct timespec *ts, int timeout_ms) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    long long total_ns = (long long) tv.tv_usec * 1000 + (long long) timeout_ms * 1000000;
    ts->tv_sec         = tv.tv_sec + total_ns / 1000000000;
    ts->tv_nsec        = total_ns % 1000000000;
}

size_t
audio_encoder_pull_softtokens(struct AudioEncoder *a, float *out, size_t max_out, int timeout_ms) {
    pthread_mutex_lock(&a->mtx);
    while (true) {
        if (a->shutdown_flag) {
            pthread_mutex_unlock(&a->mtx);
            return 0;
        }

        /* Phase 2 streaming-worker path: drain incremental soft tokens from
         * state->soft as the worker produces them. The worker may emit
         * tokens BEFORE end_input arrives, so we can return early without
         * waiting for the full segment. */
        if (a->stream_enabled && a->worker_active && a->stream != nullptr) {
            const size_t avail = a->stream->n_soft - a->stream->n_drained;
            if (avail > 0) {
                const size_t take = avail < max_out ? avail : max_out;
                memcpy(out,
                       a->stream->soft + a->stream->n_drained * AUDIO_SOFT_TOKEN_DIM,
                       take * AUDIO_SOFT_TOKEN_DIM * sizeof(float));
                a->stream->n_drained += take;
                pthread_mutex_unlock(&a->mtx);
                return take;
            }
            /* No new tokens. If the worker has finalised, we're done. */
            if (a->computed_flag) {
                pthread_mutex_unlock(&a->mtx);
                return 0;
            }
            /* Otherwise fall through to wait. */
        } else {
            /* Sync path: drain queue if we have unread soft-tokens. */
            if (a->computed_flag && a->n_emitted < a->n_soft) {
                size_t avail = a->n_soft - a->n_emitted;
                size_t take  = avail < max_out ? avail : max_out;
                memcpy(out,
                       a->soft_tokens + a->n_emitted * AUDIO_SOFT_TOKEN_DIM,
                       take * AUDIO_SOFT_TOKEN_DIM * sizeof(float));
                a->n_emitted += take;
                pthread_mutex_unlock(&a->mtx);
                return take;
            }
            if (a->computed_flag && a->n_emitted >= a->n_soft) {
                pthread_mutex_unlock(&a->mtx);
                return 0;
            }
            /* Trigger sync compute on first pull after end_input. */
            if (!a->computed_flag && a->end_input_flag) {
                compute_segment_locked(a);
                continue;
            }
        }

        /* Nothing available, no end yet → wait or return per timeout. */
        if (timeout_ms == 0) {
            pthread_mutex_unlock(&a->mtx);
            return 0;
        }
        if (timeout_ms < 0) {
            pthread_cond_wait(&a->cv, &a->mtx);
        } else {
            struct timespec dl;
            make_deadline(&dl, timeout_ms);
            int rc = pthread_cond_timedwait(&a->cv, &a->mtx, &dl);
            if (rc == ETIMEDOUT) {
                pthread_mutex_unlock(&a->mtx);
                return 0;
            }
        }
    }
}

bool audio_encoder_segment_done(const struct AudioEncoder *a) {
    /* Read-only snapshot — race is OK; caller polls. */
    return a->end_input_flag && a->computed_flag && a->n_emitted >= a->n_soft;
}

void audio_encoder_reset(struct AudioEncoder *a) {
    pthread_mutex_lock(&a->mtx);
    a->pcm_len        = 0;
    a->mel_n_computed = 0;
    a->end_input_flag = false;
    a->computed_flag  = false;
    a->n_emitted      = 0;
    a->n_soft         = 0;
    safe_free((void **) &a->soft_tokens);
    a->soft_tokens = nullptr;
    if (a->stream)
        audio_stream_state_reset(a->stream);
    /* Phase 2 worker bookkeeping. */
    a->worker_kick     = false;
    a->worker_final    = false;
    a->worker_last_mel = 0;
    pthread_mutex_unlock(&a->mtx);
}

void audio_encoder_shutdown(struct AudioEncoder *a) {
    pthread_mutex_lock(&a->mtx);
    a->shutdown_flag = true;
    pthread_cond_broadcast(&a->cv);
    pthread_mutex_unlock(&a->mtx);
}

/* ----------------------- Conformer per-stage helpers ----------------------- */

/* Apply struct ClippableLinear: y = clamp(linear(clamp(x, in_min, in_max)), out_min, out_max).
 * x: (n, in_dim), y: (n, out_dim). x is mutated by the input-clamp pass. */
static void clip_linear_apply(const struct ClippableLinear *cl,
                              float                        *x,
                              size_t                        n,
                              size_t                        in_dim,
                              size_t                        out_dim,
                              float                        *y) {
    clamp_fp32(x, n * in_dim, cl->input_min, cl->input_max);

    enum audio_linear_prec prec = cl->prec;
#if defined(GEIST_AUDIO_KEEP_FP32)
    /* A/B-bench escape hatch — only meaningful if cl->w is still around. */
    if (cl->w != nullptr && getenv("GEIST_AUDIO_FORCE_FP32") != nullptr) {
        prec = AUDIO_PREC_FP32;
    }
#endif

    switch (prec) {
    case AUDIO_PREC_W8A8: {
#if defined(__ARM_NEON)
        /* Static activation scale derived from clip bounds applied above. */
        float amax    = fmaxf(fabsf(cl->input_min), fabsf(cl->input_max));
        float scale_x = amax / 127.0f;
        if (scale_x == 0.0f)
            scale_x = 1.0f;
        linear_w8a8(cl->w_q8, cl->w_scales, x, 1.0f / scale_x, scale_x, n, in_dim, out_dim, y);
#else
        /* ponytail: non-NEON falls through to W8A32 (correct, slower). x86
         * gets the int8 fast path once cpu_x86 lands the W8A8 GEMV. */
        linear_w8a32(cl->w_q8, cl->w_scales, x, n, in_dim, out_dim, y);
#endif
        break;
    }
    case AUDIO_PREC_W8A32:
        linear_w8a32(cl->w_q8, cl->w_scales, x, n, in_dim, out_dim, y);
        break;
    default:
        linear_fp32(x, cl->w, nullptr, n, in_dim, out_dim, y);
        break;
    }
    clamp_fp32(y, n * out_dim, cl->output_min, cl->output_max);
}

/* Macaron-style struct FFN: y = norm_post(SiLU(norm_pre(x) @ W1) @ W2) * 0.5 + x */
static void ffn_run(const struct FFN *ffn, float *h, size_t n) {
    const size_t hsize    = (size_t) n * AUDIO_HIDDEN;
    float       *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h, hsize * sizeof(float));

    rmsnorm_fp32(h, ffn->pre_norm, n, AUDIO_HIDDEN, RMS_EPS, h);

    float *mid = heap_alloc_array_aligned(float, (size_t) n *FFN_INTER);
    clip_linear_apply(&ffn->ffw1, h, n, AUDIO_HIDDEN, FFN_INTER, mid);
    silu_fp32(mid, (size_t) n * FFN_INTER);
    clip_linear_apply(&ffn->ffw2, mid, n, FFN_INTER, AUDIO_HIDDEN, h);
    safe_free((void **) &mid);

    rmsnorm_fp32(h, ffn->post_norm, n, AUDIO_HIDDEN, RMS_EPS, h);
    for (size_t i = 0; i < hsize; i++)
        h[i] = h[i] * 0.5f + residual[i];
    safe_free((void **) &residual);
}

/* NEON 16-wide unroll of the per-head dot product / axpy. HEAD_DIM=128, so
 * 8 iterations cover one head fully. The three attention triple-loops below
 * (matrix_ac, matrix_bd, attn@V) all collapse to these two primitives. */
#if defined(__ARM_NEON)
static inline float dot_head_fp32(const float *a, const float *b) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    for (int d = 0; d < HEAD_DIM; d += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + d + 0), vld1q_f32(b + d + 0));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + d + 4), vld1q_f32(b + d + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + d + 8), vld1q_f32(b + d + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + d + 12), vld1q_f32(b + d + 12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}
static inline void axpy_head_fp32(float *out, float w, const float *v) {
    float32x4_t vw = vdupq_n_f32(w);
    for (int d = 0; d < HEAD_DIM; d += 16) {
        float32x4_t o0 = vld1q_f32(out + d + 0), o1 = vld1q_f32(out + d + 4);
        float32x4_t o2 = vld1q_f32(out + d + 8), o3 = vld1q_f32(out + d + 12);
        o0 = vfmaq_f32(o0, vw, vld1q_f32(v + d + 0));
        o1 = vfmaq_f32(o1, vw, vld1q_f32(v + d + 4));
        o2 = vfmaq_f32(o2, vw, vld1q_f32(v + d + 8));
        o3 = vfmaq_f32(o3, vw, vld1q_f32(v + d + 12));
        vst1q_f32(out + d + 0, o0);
        vst1q_f32(out + d + 4, o1);
        vst1q_f32(out + d + 8, o2);
        vst1q_f32(out + d + 12, o3);
    }
}
static inline void zero_head_fp32(float *out) {
    const float32x4_t z = vdupq_n_f32(0.0f);
    for (int d = 0; d < HEAD_DIM; d += 16) {
        vst1q_f32(out + d + 0, z);
        vst1q_f32(out + d + 4, z);
        vst1q_f32(out + d + 8, z);
        vst1q_f32(out + d + 12, z);
    }
}
#else
static inline float dot_head_fp32(const float *a, const float *b) {
    float acc = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++)
        acc += a[d] * b[d];
    return acc;
}
static inline void axpy_head_fp32(float *out, float w, const float *v) {
    for (int d = 0; d < HEAD_DIM; d++)
        out[d] += w * v[d];
}
static inline void zero_head_fp32(float *out) {
    for (int d = 0; d < HEAD_DIM; d++)
        out[d] = 0.0f;
}
#endif

/* Chunked self-attention with relative position bias. Reads (n, 1024) hidden,
 * pos_emb (13, 1024), attn_mask (num_blocks, 12, 24) bool. Writes (n, 1024).
 *
 * Per HF Gemma4AudioAttention.forward — naive C, per-block, per-head loops.
 * Performance is not critical: 1024 hidden × 8 heads × ≤2-4 blocks per chunk. */
static void attn_run(const struct Attn *attn,
                     const float       *h,
                     size_t             n,
                     const float       *pos_emb,
                     const bool        *attn_mask,
                     float             *y) {
    const float q_scale = (1.0f / sqrtf((float) HEAD_DIM)) / logf(2.0f);
    /* M_E is glibc-specific (not in C23); pin the constant for portability. */
    const float k_scale    = log1pf(2.71828182845904523536f) / logf(2.0f); /* log(1+e) / log(2) */
    const int   num_blocks = ((int) n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const int   n_padded   = num_blocks * CHUNK_SIZE;

    /* 1. Q/K/V projections (with input/output clip). */
    float *h_clip_in = heap_alloc_array_aligned(float, n *AUDIO_HIDDEN);
    float *q         = heap_calloc_array_aligned(float, (size_t) n_padded *AUDIO_HIDDEN);
    float *k         = heap_alloc_array_aligned(float, (size_t) n *AUDIO_HIDDEN);
    float *v         = heap_alloc_array_aligned(float, (size_t) n *AUDIO_HIDDEN);

    memcpy(h_clip_in, h, n * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->q_proj, h_clip_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, q);
    memcpy(h_clip_in, h, n * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->k_proj, h_clip_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, k);
    memcpy(h_clip_in, h, n * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->v_proj, h_clip_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, v);
    safe_free((void **) &h_clip_in);

    /* 2. Apply scales: q *= q_scale * softplus(per_dim_scale), k *= k_scale.
     *    Layout: q is (n_padded, n_heads, head_dim). Q-pad rows beyond n stay zero.
     *    Fold q_scale into the per-dim table so the inner loop is one mul/lane. */
    float q_pds[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        q_pds[d] = q_scale * log1pf(expf(attn->per_dim_scale[d]));
    }
    for (int t = 0; t < (int) n; t++) {
        for (int head_i = 0; head_i < N_HEADS; head_i++) {
            float *qrow = q + ((size_t) t * N_HEADS + head_i) * HEAD_DIM;
#if defined(__ARM_NEON)
            for (int d = 0; d < HEAD_DIM; d += 16) {
                float32x4_t s0 = vld1q_f32(q_pds + d + 0), s1 = vld1q_f32(q_pds + d + 4);
                float32x4_t s2 = vld1q_f32(q_pds + d + 8), s3 = vld1q_f32(q_pds + d + 12);
                vst1q_f32(qrow + d + 0, vmulq_f32(vld1q_f32(qrow + d + 0), s0));
                vst1q_f32(qrow + d + 4, vmulq_f32(vld1q_f32(qrow + d + 4), s1));
                vst1q_f32(qrow + d + 8, vmulq_f32(vld1q_f32(qrow + d + 8), s2));
                vst1q_f32(qrow + d + 12, vmulq_f32(vld1q_f32(qrow + d + 12), s3));
            }
#else
            for (int d = 0; d < HEAD_DIM; d++)
                qrow[d] *= q_pds[d];
#endif
        }
    }
#if defined(__ARM_NEON)
    {
        const float32x4_t vks   = vdupq_n_f32(k_scale);
        const size_t      total = n * AUDIO_HIDDEN;
        size_t            i     = 0;
        for (; i + 16 <= total; i += 16) {
            vst1q_f32(k + i + 0, vmulq_f32(vld1q_f32(k + i + 0), vks));
            vst1q_f32(k + i + 4, vmulq_f32(vld1q_f32(k + i + 4), vks));
            vst1q_f32(k + i + 8, vmulq_f32(vld1q_f32(k + i + 8), vks));
            vst1q_f32(k + i + 12, vmulq_f32(vld1q_f32(k + i + 12), vks));
        }
        for (; i < total; i++)
            k[i] *= k_scale;
    }
#else
    for (size_t i = 0; i < n * AUDIO_HIDDEN; i++)
        k[i] *= k_scale;
#endif

    /* 3. Build context-windowed K/V: (num_blocks, CONTEXT_SIZE=24, n_heads, head_dim).
     *    HF: F.pad(K, (0,0, 0,0, max_past=12, chunk-1=11)), then unfold(size=24, stride=12).
     *    Block b's window covers padded source positions [b*12 .. b*12 + 23]
     *    where the source is K with 12 left zeros + n original + (chunk-1) right zeros.
     *    Mapped back: src_idx = (b*12 + ctx_pos) - 12; valid if src_idx in [0, n). */
    const size_t hd_per_t = (size_t) N_HEADS * HEAD_DIM;
    float *k_ctx = heap_calloc_array_aligned(float, (size_t) num_blocks * CONTEXT_SIZE * hd_per_t);
    float *v_ctx = heap_calloc_array_aligned(float, (size_t) num_blocks * CONTEXT_SIZE * hd_per_t);
    for (int b = 0; b < num_blocks; b++) {
        for (int c = 0; c < CONTEXT_SIZE; c++) {
            int src = b * CHUNK_SIZE + c - MAX_PAST_HORIZON;
            if (src < 0 || src >= (int) n)
                continue;
            memcpy(k_ctx + ((size_t) b * CONTEXT_SIZE + c) * hd_per_t,
                   k + (size_t) src * hd_per_t,
                   hd_per_t * sizeof(float));
            memcpy(v_ctx + ((size_t) b * CONTEXT_SIZE + c) * hd_per_t,
                   v + (size_t) src * hd_per_t,
                   hd_per_t * sizeof(float));
        }
    }
    safe_free((void **) &k);
    safe_free((void **) &v);

    /* 4. Relative-K projection: (POS_LEN=13, 1024) → (POS_LEN, 1024) → reshape (POS_LEN, 8, 128).
     */
    float *rel_k = heap_alloc_array_aligned(float, (size_t) POS_LEN *AUDIO_HIDDEN);
    linear_fp32(
            pos_emb, attn->relative_k_proj, nullptr, POS_LEN, AUDIO_HIDDEN, AUDIO_HIDDEN, rel_k);

    /* Output of attention before post-proj: (n_padded, 1024). */
    float *attn_out = heap_calloc_array_aligned(float, (size_t) n_padded *AUDIO_HIDDEN);

    /* Per-block, per-head attention. (b, hd) pairs are independent — parallelize.
     * Scratch arrays are declared inside the body so each thread has its own. */
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int b = 0; b < num_blocks; b++) {
        for (int hd = 0; hd < N_HEADS; hd++) {
            float scores_ac[CHUNK_SIZE * CONTEXT_SIZE];
            float scores_bd[CHUNK_SIZE * CONTEXT_SIZE];
            float bd_padded[CHUNK_SIZE * (CONTEXT_SIZE + 1)];
            /* Q rows for this block/head: (CHUNK_SIZE, HEAD_DIM). */
            const float *q_bh = q + ((size_t) b * CHUNK_SIZE) * hd_per_t + (size_t) hd * HEAD_DIM;
            /* K rows: (CONTEXT_SIZE, HEAD_DIM). */
            const float *k_bh =
                    k_ctx + ((size_t) b * CONTEXT_SIZE) * hd_per_t + (size_t) hd * HEAD_DIM;
            /* V rows: same layout as K. */
            const float *v_bh =
                    v_ctx + ((size_t) b * CONTEXT_SIZE) * hd_per_t + (size_t) hd * HEAD_DIM;
            /* Relative-K rows: (POS_LEN, HEAD_DIM). */
            const float *rk_h = rel_k + (size_t) hd * HEAD_DIM;
            /* Note: rel_k layout is (POS_LEN, N_HEADS, HEAD_DIM) flat as (POS_LEN, 1024).
             * To pick head `hd`, stride POS_LEN times by AUDIO_HIDDEN. */

            /* matrix_ac[i, j] = sum_d q[i, d] * k[j, d]   for i in chunk, j in context. */
            for (int i = 0; i < CHUNK_SIZE; i++) {
                const float *qi = q_bh + (size_t) i * hd_per_t;
                for (int j = 0; j < CONTEXT_SIZE; j++) {
                    const float *kj                 = k_bh + (size_t) j * hd_per_t;
                    scores_ac[i * CONTEXT_SIZE + j] = dot_head_fp32(qi, kj);
                }
            }

            /* matrix_bd[i, p] = sum_d q[i, d] * rel_k[p, d]   for i in chunk, p in pos_len.
             * Then rel_shift to (CHUNK_SIZE, CONTEXT_SIZE). */
            for (int i = 0; i < CHUNK_SIZE; i++) {
                const float *qi = q_bh + (size_t) i * hd_per_t;
                for (int p = 0; p < POS_LEN; p++) {
                    const float *rp                       = rk_h + (size_t) p * AUDIO_HIDDEN;
                    bd_padded[i * (CONTEXT_SIZE + 1) + p] = dot_head_fp32(qi, rp);
                }
                /* zero-pad the rest of this row (positions POS_LEN..CONTEXT_SIZE) */
                for (int p = POS_LEN; p <= CONTEXT_SIZE; p++) {
                    bd_padded[i * (CONTEXT_SIZE + 1) + p] = 0.0f;
                }
            }
            /* rel_shift: flatten (CHUNK_SIZE, CONTEXT_SIZE+1) → CHUNK_SIZE * (CONTEXT_SIZE+1)
             * then take the first CHUNK_SIZE * CONTEXT_SIZE entries, view as
             * (CHUNK_SIZE, CONTEXT_SIZE). */
            for (int idx = 0; idx < CHUNK_SIZE * CONTEXT_SIZE; idx++) {
                scores_bd[idx] = bd_padded[idx];
            }

            /* Combine + soft-cap + mask + softmax. */
            for (int i = 0; i < CHUNK_SIZE; i++) {
                for (int j = 0; j < CONTEXT_SIZE; j++) {
                    float s = scores_ac[i * CONTEXT_SIZE + j] + scores_bd[i * CONTEXT_SIZE + j];
                    s       = tanhf(s / ATTN_SOFTCAP) * ATTN_SOFTCAP;
                    if (attn_mask && !attn_mask[((size_t) b * CHUNK_SIZE + i) * CONTEXT_SIZE + j]) {
                        s = -1e9f;
                    }
                    scores_ac[i * CONTEXT_SIZE + j] = s; /* reuse scores_ac as combined */
                }
            }
            softmax_fp32(scores_ac, CHUNK_SIZE, CONTEXT_SIZE);

            /* attn @ V → (CHUNK_SIZE, HEAD_DIM) into attn_out at this block/head slot. */
            for (int i = 0; i < CHUNK_SIZE; i++) {
                const int t_out = b * CHUNK_SIZE + i;
                if (t_out >= n_padded)
                    break;
                float *out_row = attn_out + (size_t) t_out * AUDIO_HIDDEN + (size_t) hd * HEAD_DIM;
                zero_head_fp32(out_row);
                for (int j = 0; j < CONTEXT_SIZE; j++) {
                    const float  w  = scores_ac[i * CONTEXT_SIZE + j];
                    const float *vj = v_bh + (size_t) j * hd_per_t;
                    axpy_head_fp32(out_row, w, vj);
                }
            }
        }
    }
    safe_free((void **) &rel_k);
    safe_free((void **) &k_ctx);
    safe_free((void **) &v_ctx);
    safe_free((void **) &q);

    /* 5. Crop attn_out to first n rows, post-proj. */
    float *tmp_in = heap_alloc_array_aligned(float, n *AUDIO_HIDDEN);
    memcpy(tmp_in, attn_out, n * AUDIO_HIDDEN * sizeof(float));
    safe_free((void **) &attn_out);
    clip_linear_apply(&attn->post, tmp_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, y);
    safe_free((void **) &tmp_in);
}

/* LightConv1D: norm → linear_start (×2) → GLU → causal depthwise conv → conv_norm → SiLU →
 * linear_end → +residual */
static void lconv_run(const struct LConv *lc, float *h, size_t n) {
    const size_t hsize    = (size_t) n * AUDIO_HIDDEN;
    float       *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h, hsize * sizeof(float));

    rmsnorm_fp32(h, lc->pre_norm, n, AUDIO_HIDDEN, RMS_EPS, h);

    /* linear_start: (n, 1024) → (n, 2048) */
    float *doubled = heap_alloc_array_aligned(float, (size_t) n * 2 * AUDIO_HIDDEN);
    clip_linear_apply(&lc->linear_start, h, n, AUDIO_HIDDEN, 2 * AUDIO_HIDDEN, doubled);

    /* GLU: (n, 2048) → (n, 1024) */
    glu_fp32(doubled, n, AUDIO_HIDDEN, h);
    safe_free((void **) &doubled);

    /* Depthwise conv1d: input is (n, 1024). HF transposes (B, T, C) → (B, C, T),
     * runs conv, transposes back. We use (C, T) layout for the conv directly. */
    float *hct = heap_alloc_array_aligned(float, hsize);
    for (size_t t = 0; t < n; t++)
        for (int c = 0; c < AUDIO_HIDDEN; c++)
            hct[(size_t) c * n + t] = h[(size_t) t * AUDIO_HIDDEN + c];

    float *conv_out = heap_alloc_array_aligned(float, hsize);
    depthwise_conv1d_causal_fp32(hct, lc->depthwise, conv_out, AUDIO_HIDDEN, (int) n, CONV_KERNEL);
    safe_free((void **) &hct);

    /* Transpose back (C, T) → (T, C) */
    for (size_t t = 0; t < n; t++)
        for (int c = 0; c < AUDIO_HIDDEN; c++)
            h[(size_t) t * AUDIO_HIDDEN + c] = conv_out[(size_t) c * n + t];
    safe_free((void **) &conv_out);

    /* conv_norm + SiLU + linear_end + residual */
    rmsnorm_fp32(h, lc->conv_norm, n, AUDIO_HIDDEN, RMS_EPS, h);
    silu_fp32(h, hsize);
    float *end_in = heap_alloc_array_aligned(float, hsize);
    memcpy(end_in, h, hsize * sizeof(float));
    clip_linear_apply(&lc->linear_end, end_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, h);
    safe_free((void **) &end_in);

    for (size_t i = 0; i < hsize; i++)
        h[i] += residual[i];
    safe_free((void **) &residual);
}

void audio_encoder_layer_run(const struct AudioEncoder *a,
                             int                        layer_idx,
                             const float               *h_in,
                             size_t                     n,
                             const float               *pos_emb,
                             const bool                *attn_mask_5d,
                             float                     *h_out) {
    const struct ConformerLayer *L     = &a->layers[layer_idx];
    const size_t                 hsize = (size_t) n * AUDIO_HIDDEN;

    float *h = heap_alloc_array_aligned(float, hsize);
    memcpy(h, h_in, hsize * sizeof(float));

    /* 1. feed_forward1 (with internal residual + 0.5 scale) */
    {
        AE_TIC();
        ffn_run(&L->ff1, h, n);
        AE_TOC(g_ae_ffn1);
    }

    /* 2. residual + norm_pre_attn + self_attn + norm_post_attn + residual */
    float *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h, hsize * sizeof(float));
    {
        AE_TIC();
        rmsnorm_fp32(h, L->norm_pre_attn, n, AUDIO_HIDDEN, RMS_EPS, h);
        AE_TOC(g_ae_norm_pre_attn);
    }

    float *attn_out = heap_alloc_array_aligned(float, hsize);
    {
        AE_TIC();
        attn_run(&L->attn, h, n, pos_emb, attn_mask_5d, attn_out);
        AE_TOC(g_ae_attn);
    }
    memcpy(h, attn_out, hsize * sizeof(float));
    safe_free((void **) &attn_out);

    {
        AE_TIC();
        rmsnorm_fp32(h, L->norm_post_attn, n, AUDIO_HIDDEN, RMS_EPS, h);
        AE_TOC(g_ae_norm_post_attn);
    }
    for (size_t i = 0; i < hsize; i++)
        h[i] += residual[i];
    safe_free((void **) &residual);

    /* 3. lconv1d (internal residual) */
    {
        AE_TIC();
        lconv_run(&L->lconv, h, n);
        AE_TOC(g_ae_lconv);
    }

    /* 4. feed_forward2 (internal residual) */
    {
        AE_TIC();
        ffn_run(&L->ff2, h, n);
        AE_TOC(g_ae_ffn2);
    }

    /* 5. norm_out */
    {
        AE_TIC();
        rmsnorm_fp32(h, L->norm_out, n, AUDIO_HIDDEN, RMS_EPS, h_out);
        AE_TOC(g_ae_norm_out);
    }

    safe_free((void **) &h);
}

float *audio_encoder_compute_pos_emb(const struct AudioEncoder *a) {
    (void) a;
    /* Sinusoidal relative position embedding: (POS_LEN, 1024) with
     * position_ids = [POS_LEN-1, POS_LEN-2, ..., 0]. */
    float      *out            = heap_alloc_array_aligned(float, (size_t) POS_LEN *AUDIO_HIDDEN);
    const int   num_timescales = AUDIO_HIDDEN / 2;
    const float log_inc        = logf(10000.0f) / (float) (num_timescales - 1);
    for (int p = 0; p < POS_LEN; p++) {
        const int pos_id = POS_LEN - 1 - p; /* arange(context_size//2, -1, -1) */
        for (int i = 0; i < num_timescales; i++) {
            const float inv_ts                                  = expf((float) i * -log_inc);
            const float scaled                                  = (float) pos_id * inv_ts;
            out[(size_t) p * AUDIO_HIDDEN + i]                  = sinf(scaled);
            out[(size_t) p * AUDIO_HIDDEN + num_timescales + i] = cosf(scaled);
        }
    }
    return out;
}

bool *audio_encoder_compute_attn_mask(const struct AudioEncoder *a, size_t n) {
    (void) a;
    /* For each query at absolute position q_g = b*CHUNK + i, the visible keys
     * are k_g in [q_g - MAX_PAST_HORIZON, q_g + MAX_FUTURE] ∩ [0, n).
     * Context position c maps to k_g = b*CHUNK + c - MAX_PAST_HORIZON. */
    const int num_blocks = ((int) n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    bool     *m = heap_calloc_array_aligned(bool, (size_t) num_blocks * CHUNK_SIZE * CONTEXT_SIZE);
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < CHUNK_SIZE; i++) {
            const int q_g = b * CHUNK_SIZE + i;
            if (q_g >= (int) n)
                continue; /* invalid query row → mask all False */
            for (int c = 0; c < CONTEXT_SIZE; c++) {
                const int k_g = b * CHUNK_SIZE + c - MAX_PAST_HORIZON;
                if (k_g < 0 || k_g >= (int) n)
                    continue;
                const int off = k_g - q_g;
                /* sliding_window_mask_function((max_past_horizon, max_future)) — the
                 * effective window is `max_past_horizon` positions including the
                 * current one, i.e., off in [-(max_past_horizon-1), max_future]. */
                if (off < -(MAX_PAST_HORIZON - 1) || off > MAX_FUTURE)
                    continue;
                m[((size_t) b * CHUNK_SIZE + i) * CONTEXT_SIZE + c] = true;
            }
        }
    }
    return m;
}

/* HF subsample, B=1:
 *   conv_a(x_NCHW): (1, c_in, H, W) → (1, c_out, H_out, W_out)
 *   then permute (B,C,H,W) → (B,H,W,C), LayerNorm-ws over last axis (C),
 *   permute back → (B,C,H,W), ReLU.
 *
 * For our single-batch case we keep buffers in (C, H, W) layout, which is
 * what conv2d_fp32 expects directly. Permute-to-LayerNorm-and-back is
 * implemented by walking strides without an actual buffer transpose. */
/* Subsample stage = conv2d + LayerNorm (HWC over C) + ReLU.
 * Default start_h=0 runs the whole h-axis (used by audio_encoder_run).
 * The Phase-3 streaming path passes start_h > 0 to skip already-computed
 * h positions; conv2d outputs at h < start_h must already be in `out`
 * (cached from a previous call). */
static void subsample_layer_from(const float *w_conv,
                                 const float *w_norm,
                                 const float *in,
                                 int          c_in,
                                 int          h_in,
                                 int          w_in,
                                 int          c_out,
                                 float       *out,
                                 int         *h_out_p,
                                 int         *w_out_p,
                                 int          start_h) {
    const int kh = 3, kw = 3, sh = 2, sw = 2, ph = 1, pw = 1;
    const int h_out = (h_in + 2 * ph - kh) / sh + 1;
    const int w_out = (w_in + 2 * pw - kw) / sw + 1;
    *h_out_p        = h_out;
    *w_out_p        = w_out;
    if (start_h < 0)
        start_h = 0;
    if (start_h >= h_out)
        return;

    /* 1: conv → (c_out, h_out, w_out), only oh >= start_h written. */
    conv2d_fp32_from(in, w_conv, out, c_in, c_out, h_in, w_in, kh, kw, sh, sw, ph, pw, start_h);

    /* 2+3: LayerNorm over channel axis (c_out), per (h_out * w_out) "pixel".
     * Buffer is (c_out, h_out, w_out) — channels outer. LN normalises along
     * the C axis per (h, w) pixel, independent of other pixels. So we can
     * restrict to new h positions only. */
    const int    new_h     = h_out - start_h;
    const size_t n_pix_new = (size_t) new_h * w_out;
    float       *tmp       = heap_alloc_array_aligned(float, n_pix_new *(size_t) c_out);
    /* CHW(new slice) → HWC */
    for (int h = 0; h < new_h; h++) {
        for (int wpos = 0; wpos < w_out; wpos++) {
            for (int c = 0; c < c_out; c++) {
                tmp[((size_t) h * w_out + wpos) * c_out + c] =
                        out[((size_t) c * h_out + (start_h + h)) * w_out + wpos];
            }
        }
    }
    layernorm_fp32_ws(tmp, w_norm, n_pix_new, c_out, LN_EPS, tmp);
    /* HWC → CHW(new slice) */
    for (int h = 0; h < new_h; h++) {
        for (int wpos = 0; wpos < w_out; wpos++) {
            for (int c = 0; c < c_out; c++) {
                out[((size_t) c * h_out + (start_h + h)) * w_out + wpos] =
                        tmp[((size_t) h * w_out + wpos) * c_out + c];
            }
        }
    }
    safe_free((void **) &tmp);

    /* 4: ReLU on new h positions only. */
    for (int c = 0; c < c_out; c++) {
        float *slot = out + ((size_t) c * h_out + start_h) * w_out;
        for (int i = 0; i < new_h * w_out; i++) {
            if (slot[i] < 0.0f)
                slot[i] = 0.0f;
        }
    }
}

/* Thin wrapper for the existing call sites — full h_out range. */
static void subsample_layer(const float *w_conv,
                            const float *w_norm,
                            const float *in,
                            int          c_in,
                            int          h_in,
                            int          w_in,
                            int          c_out,
                            float       *out,
                            int         *h_out_p,
                            int         *w_out_p) {
    subsample_layer_from(w_conv, w_norm, in, c_in, h_in, w_in, c_out, out, h_out_p, w_out_p, 0);
}

/* Zero out time-positions in a (C, T, W) buffer where mask[t] == false. */
static void apply_time_mask(float *buf, const bool *mask, int c, int t, int w) {
    if (!mask)
        return;
    const size_t plane = (size_t) t * w;
    for (int ti = 0; ti < t; ti++) {
        if (mask[ti])
            continue;
        for (int ci = 0; ci < c; ci++) {
            float *row = buf + (size_t) ci * plane + (size_t) ti * w;
            for (int wi = 0; wi < w; wi++)
                row[wi] = 0.0f;
        }
    }
}

size_t audio_encoder_subsample_run(const struct AudioEncoder *a,
                                   const float               *mel_in,
                                   const bool                *mask_in,
                                   size_t                     n_mel_frames,
                                   float                     *out) {
    /* Treat (n_mel_frames, 128) as a (1, T, 128) image. In our (C, H, W)
     * scratch convention: c_in=1, h_in=T, w_in=128. */
    const int T = (int) n_mel_frames;
    const int W = MEL_DIM;

    /* Optional pre-conv-0 mask: HF multiplies the input by mask before conv.
     * We need a writable copy if a mask is provided. */
    const float *in_for_conv0 = mel_in;
    float       *mel_masked   = nullptr;
    if (mask_in) {
        mel_masked = heap_alloc_array_aligned(float, (size_t) T *W);
        memcpy(mel_masked, mel_in, (size_t) T * W * sizeof(float));
        apply_time_mask(mel_masked, mask_in, 1, T, W);
        in_for_conv0 = mel_masked;
    }

    /* Layer 0 output buffer: (128, T_out0, W_out0). */
    const int T_out0 = (T + 2 - 3) / 2 + 1;
    const int W_out0 = (W + 2 - 3) / 2 + 1;
    float    *l0     = heap_alloc_array_aligned(float, (size_t) 128 * T_out0 * W_out0);
    int       hh, ww;
    subsample_layer(a->l0_conv, a->l0_norm, in_for_conv0, 1, T, W, 128, l0, &hh, &ww);
    safe_free((void **) &mel_masked);

    /* Mask after layer 0 is mask[::2]. Apply before layer 1 conv. */
    if (mask_in) {
        bool *mask1 = heap_alloc_array_aligned(bool, (size_t) hh);
        for (int i = 0; i < hh; i++)
            mask1[i] = (i * 2 < T) ? mask_in[i * 2] : false;
        apply_time_mask(l0, mask1, 128, hh, ww);
        safe_free((void **) &mask1);
    }

    /* Layer 1 output: (32, T_out1, W_out1). */
    const int T_out1 = (hh + 2 - 3) / 2 + 1;
    const int W_out1 = (ww + 2 - 3) / 2 + 1;
    float    *l1     = heap_alloc_array_aligned(float, (size_t) 32 * T_out1 * W_out1);
    int       hh2, ww2;
    subsample_layer(a->l1_conv, a->l1_norm, l0, 128, hh, ww, 32, l1, &hh2, &ww2);
    safe_free((void **) &l0);

    /* Reshape (32, T_out1, 32) → (T_out1, 32, 32) → (T_out1, 1024).
     * HF: permute(0, 2, 3, 1).reshape(B, T_out1, -1)
     * The permute makes channels the LAST axis, then reshape concatenates
     * (W * C) per time step. So per (t, w, c) → flat[t * (W*C) + w * C + c]. */
    const int proj_in_dim = W_out1 * 32; /* = 32 * 32 = 1024 */
    float    *flat        = heap_alloc_array_aligned(float, (size_t) T_out1 *proj_in_dim);
    for (int t = 0; t < T_out1; t++) {
        for (int w = 0; w < W_out1; w++) {
            for (int c = 0; c < 32; c++) {
                flat[(size_t) t * proj_in_dim + w * 32 + c] =
                        l1[((size_t) c * T_out1 + t) * W_out1 + w];
            }
        }
    }
    safe_free((void **) &l1);

    /* Linear projection: (T_out1, 1024) @ in_proj^T (1024, 1024) → (T_out1, 1024). */
    linear_fp32(flat, a->in_proj, nullptr, (size_t) T_out1, AUDIO_HIDDEN, AUDIO_HIDDEN, out);
    safe_free((void **) &flat);

    return (size_t) T_out1;
}

/* Phase-3 incremental subsample. Extends the cached `subs->l0` and
 * `subs->l1` buffers with new time positions and projects the new l1
 * rows into `out_sub_buf` at the right offset. Returns the new total
 * number of sub-tokens (== T_out1 for the current n_mel_total).
 *
 * Conv2d outputs are stable for the time positions already computed
 * (kernel taps are at deterministic offsets from oh and the input
 * grows only at the right edge), so the old cached values for h <
 * subs->n_t_out0 stay correct. The Phase-3 `_from` family writes only
 * new h positions. */
static size_t audio_encoder_subsample_run_inc(const struct AudioEncoder *a,
                                              struct subs_cache         *subs,
                                              const float               *mel_in,
                                              const bool                *mask_in,
                                              size_t                     n_mel_total,
                                              float                     *out_sub_buf) {
    if (n_mel_total <= subs->n_mel_seen)
        return subs->n_t_out1;

    const int T          = (int) n_mel_total;
    const int W          = MEL_DIM;
    const int T_out0_new = (T + 2 - 3) / 2 + 1;
    const int W_out0     = (W + 2 - 3) / 2 + 1;
    const int T_out1_new = (T_out0_new + 2 - 3) / 2 + 1;
    const int W_out1     = (W_out0 + 2 - 3) / 2 + 1;

    if ((size_t) T_out0_new > SUBS_T_OUT0_CAP || (size_t) T_out1_new > SUBS_T_OUT1_CAP) {
        /* Fell off the cache cap - shouldn't happen for ≤30 s audio, but
         * fall back to a full re-run as a safety valve. */
        return audio_encoder_subsample_run(a, mel_in, mask_in, n_mel_total, out_sub_buf);
    }

    /* Optional mask. Apply on the entire mel (cheap - just zeroes rows). */
    const float *in_for_conv0 = mel_in;
    float       *mel_masked   = nullptr;
    if (mask_in) {
        mel_masked = heap_alloc_array_aligned(float, (size_t) T *W);
        memcpy(mel_masked, mel_in, (size_t) T * W * sizeof(float));
        apply_time_mask(mel_masked, mask_in, 1, T, W);
        in_for_conv0 = mel_masked;
    }

    /* l0 is dimensioned (128, SUBS_T_OUT0_CAP, 64). conv2d_fp32_from writes
     * outputs at oh ∈ [start_h, T_out0_new). The actual storage stride is
     * SUBS_T_OUT0_CAP, but conv2d expects stride == h_out_new. We resolve
     * by passing a per-channel sliced view = NOT possible with the
     * existing kernel signature. Workaround: write to a temporary buffer
     * sized (128, T_out0_new, 64) for the conv pass, then copy the *new*
     * h rows into the cache.
     *
     * The temporary allocation is bounded by T_out0_new * 128 * 64 * 4 B
     * = at most ~7 MB for 1 s of new audio. Heap-arena reuse keeps this
     * cheap compared to a re-run that would compute ALL h positions. */
    const int new_h0 = T_out0_new - (int) subs->n_t_out0;
    float    *l0_tmp =
            heap_alloc_array_aligned(float, (size_t) SUBS_L0_CHANNELS * T_out0_new * W_out0);
    /* Copy old l0 rows into the temp buffer at h=[0, n_t_out0). */
    for (int c = 0; c < SUBS_L0_CHANNELS; c++) {
        for (size_t h = 0; h < subs->n_t_out0; h++) {
            memcpy(l0_tmp + ((size_t) c * T_out0_new + h) * W_out0,
                   subs->l0 + ((size_t) c * SUBS_T_OUT0_CAP + h) * W_out0,
                   (size_t) W_out0 * sizeof(float));
        }
    }
    int hh, ww;
    subsample_layer_from(a->l0_conv,
                         a->l0_norm,
                         in_for_conv0,
                         1,
                         T,
                         W,
                         SUBS_L0_CHANNELS,
                         l0_tmp,
                         &hh,
                         &ww,
                         (int) subs->n_t_out0);
    safe_free((void **) &mel_masked);

    /* Mask on l0[::2] before conv1. Apply only to NEW rows. */
    if (mask_in) {
        bool *mask1 = heap_alloc_array_aligned(bool, (size_t) hh);
        for (int i = 0; i < hh; i++)
            mask1[i] = (i * 2 < T) ? mask_in[i * 2] : false;
        /* apply_time_mask runs on the full (C, hh, ww) buffer - the new rows
         * we just wrote are at the right end and any unchanged old rows we
         * also re-mask are bit-equivalent to their cached values. */
        apply_time_mask(l0_tmp, mask1, SUBS_L0_CHANNELS, hh, ww);
        safe_free((void **) &mask1);
    }

    /* Copy new l0 rows back into the cache. */
    for (int c = 0; c < SUBS_L0_CHANNELS; c++) {
        for (int h = (int) subs->n_t_out0; h < hh; h++) {
            memcpy(subs->l0 + ((size_t) c * SUBS_T_OUT0_CAP + h) * W_out0,
                   l0_tmp + ((size_t) c * T_out0_new + h) * W_out0,
                   (size_t) W_out0 * sizeof(float));
        }
    }

    /* Same trick for l1: gather full l0 slice, run subsample_layer_from
     * starting at the new T_out1 boundary, then store new rows. */
    float *l1_tmp =
            heap_alloc_array_aligned(float, (size_t) SUBS_L1_CHANNELS * T_out1_new * W_out1);
    /* Old l1 rows. */
    for (int c = 0; c < SUBS_L1_CHANNELS; c++) {
        for (size_t h = 0; h < subs->n_t_out1; h++) {
            memcpy(l1_tmp + ((size_t) c * T_out1_new + h) * W_out1,
                   subs->l1 + ((size_t) c * SUBS_T_OUT1_CAP + h) * W_out1,
                   (size_t) W_out1 * sizeof(float));
        }
    }
    int hh2, ww2;
    subsample_layer_from(a->l1_conv,
                         a->l1_norm,
                         l0_tmp,
                         SUBS_L0_CHANNELS,
                         hh,
                         ww,
                         SUBS_L1_CHANNELS,
                         l1_tmp,
                         &hh2,
                         &ww2,
                         (int) subs->n_t_out1);
    safe_free((void **) &l0_tmp);

    /* Copy new l1 rows into cache. */
    for (int c = 0; c < SUBS_L1_CHANNELS; c++) {
        for (int h = (int) subs->n_t_out1; h < hh2; h++) {
            memcpy(subs->l1 + ((size_t) c * SUBS_T_OUT1_CAP + h) * W_out1,
                   l1_tmp + ((size_t) c * T_out1_new + h) * W_out1,
                   (size_t) W_out1 * sizeof(float));
        }
    }
    (void) new_h0; /* not directly used after the boundary calc above */

    /* Reshape + project NEW l1 rows only into out_sub_buf at the right
     * offset, leaving older sub_buf positions untouched. */
    const int new_t1      = hh2 - (int) subs->n_t_out1;
    const int proj_in_dim = W_out1 * SUBS_L1_CHANNELS;
    float    *flat        = heap_alloc_array_aligned(float, (size_t) new_t1 *proj_in_dim);
    for (int t = 0; t < new_t1; t++) {
        const int abs_t = (int) subs->n_t_out1 + t;
        for (int wp = 0; wp < W_out1; wp++) {
            for (int c = 0; c < SUBS_L1_CHANNELS; c++) {
                flat[(size_t) t * proj_in_dim + wp * SUBS_L1_CHANNELS + c] =
                        l1_tmp[((size_t) c * T_out1_new + abs_t) * W_out1 + wp];
            }
        }
    }
    safe_free((void **) &l1_tmp);

    linear_fp32(flat,
                a->in_proj,
                nullptr,
                (size_t) new_t1,
                AUDIO_HIDDEN,
                AUDIO_HIDDEN,
                out_sub_buf + (size_t) subs->n_t_out1 * AUDIO_HIDDEN);
    safe_free((void **) &flat);

    /* Boundary-stability: with kernel=3, stride=2, pad=1, the last output
     * position reads input[2p+1] which falls in the zero-pad region when
     * h_in is odd. That last position will get DIFFERENT (correct) values
     * once h_in extends past 2p+1, so we must not cache it here - the
     * next push starts conv2d at this boundary and overwrites it. For
     * even h_in all output positions are stable. */
    const size_t stable_l0 = (T % 2 == 0) ? (size_t) hh : (size_t) (hh - 1);
    const size_t stable_l1 = (hh % 2 == 0) ? (size_t) hh2 : (size_t) (hh2 - 1);
    subs->n_t_out0         = stable_l0;
    subs->n_t_out1         = stable_l1;
    subs->n_mel_seen       = n_mel_total;
    return (size_t) hh2;
}

/* Full audio-tower forward: subsample + 12 Conformer layers + output_proj +
 * embed_audio. Caller passes padded mel + mask; output is the soft-token sequence. */
size_t audio_encoder_run(const struct AudioEncoder *a,
                         const float               *mel_in,
                         const bool                *mel_mask_in,
                         size_t                     n_mel_frames,
                         float                     *softtokens_out) {
    /* Step 1: subsample → (T_sub, 1024). */
    float *sub = heap_alloc_array_aligned(float, (n_mel_frames / 4 + 4) * AUDIO_HIDDEN);
    size_t n_sub;
    {
        AE_TIC();
        n_sub = audio_encoder_subsample_run(a, mel_in, mel_mask_in, n_mel_frames, sub);
        AE_TOC(g_ae_subsample);
    }

    /* Step 2: precompute pos_emb + attn_mask (constant across layers). */
    float *pos_emb;
    bool  *attn_mask;
    {
        AE_TIC();
        pos_emb   = audio_encoder_compute_pos_emb(a);
        attn_mask = audio_encoder_compute_attn_mask(a, n_sub);
        AE_TOC(g_ae_pos_emb);
    }

    /* Step 3: run 12 Conformer layers in sequence. Ping-pong two buffers. */
    float *h_a = sub; /* takes ownership of sub */
    float *h_b = heap_alloc_array_aligned(float, n_sub *AUDIO_HIDDEN);
    {
        AE_TIC();
        for (int li = 0; li < N_LAYERS; li++) {
            audio_encoder_layer_run(a, li, h_a, n_sub, pos_emb, attn_mask, h_b);
            float *tmp = h_a;
            h_a        = h_b;
            h_b        = tmp; /* result lands in h_a */
        }
        AE_TOC(g_ae_layer_total);
    }
    safe_free((void **) &h_b);
    safe_free((void **) &attn_mask);
    safe_free((void **) &pos_emb);

    /* Step 4: output_proj (1024 → 1536, with bias). */
    float *op = heap_alloc_array_aligned(float, n_sub *OUTPUT_PROJ_DIMS);
    {
        AE_TIC();
        linear_fp32(
                h_a, a->output_proj_w, a->output_proj_b, n_sub, AUDIO_HIDDEN, OUTPUT_PROJ_DIMS, op);
        AE_TOC(g_ae_output_proj);
    }
    safe_free((void **) &h_a);

    /* Step 5: embed_audio.embedding_pre_projection_norm (RMSNorm with_scale=False)
     *         then embedding_projection (1536 → 1536, no bias). */
    float *normed = heap_alloc_array_aligned(float, n_sub *OUTPUT_PROJ_DIMS);
    rmsnorm_fp32(op, nullptr, n_sub, OUTPUT_PROJ_DIMS, RMS_EPS, normed);
    safe_free((void **) &op);
    {
        AE_TIC();
        linear_fp32(normed,
                    a->embed_audio_proj,
                    nullptr,
                    n_sub,
                    OUTPUT_PROJ_DIMS,
                    TEXT_HIDDEN,
                    softtokens_out);
        AE_TOC(g_ae_embed_proj);
    }
    safe_free((void **) &normed);

    ae_profile_print_and_reset();
    return n_sub;
}

/* === Phase 8b chunk-streaming scaffolding (impl in subsequent commit). === */

static struct audio_stream_state *audio_stream_state_create(void) {
    struct audio_stream_state *s = heap_calloc_array_aligned(struct audio_stream_state, 1);
    if (s == nullptr)
        return nullptr;
    for (int li = 0; li < N_LAYERS; li++) {
        s->attn[li].k     = heap_calloc_array_aligned(float, MAX_SUB_TOKENS *AUDIO_HIDDEN);
        s->attn[li].v     = heap_calloc_array_aligned(float, MAX_SUB_TOKENS *AUDIO_HIDDEN);
        s->lconv[li].hist = heap_calloc_array_aligned(float, (CONV_KERNEL - 1) * AUDIO_HIDDEN);
        if (!s->attn[li].k || !s->attn[li].v || !s->lconv[li].hist) {
            audio_stream_state_destroy(s);
            return nullptr;
        }
    }
    s->sub_buf = heap_calloc_array_aligned(float, MAX_SUB_TOKENS *AUDIO_HIDDEN);
    s->soft    = heap_calloc_array_aligned(float, MAX_SUB_TOKENS *AUDIO_SOFT_TOKEN_DIM);
    /* Phase-3 subsample cache. l0 dominates the allocation - 128 channels
     * × T_out0 × 64 freq positions × 4 B = up to ~49 MB at the 30 s mel
     * cap. Allocated only when the stream worker may run; on Pi 5 we run
     * with 4 GB so this is comfortable. */
    s->subs.l0 = heap_calloc_array_aligned(
            float, (size_t) SUBS_L0_CHANNELS * SUBS_T_OUT0_CAP * SUBS_W_OUT0);
    s->subs.l1 = heap_calloc_array_aligned(
            float, (size_t) SUBS_L1_CHANNELS * SUBS_T_OUT1_CAP * SUBS_W_OUT1);
    if (!s->sub_buf || !s->soft || !s->subs.l0 || !s->subs.l1) {
        audio_stream_state_destroy(s);
        return nullptr;
    }
    return s;
}

static void audio_stream_state_destroy(struct audio_stream_state *s) {
    if (s == nullptr)
        return;
    for (int li = 0; li < N_LAYERS; li++) {
        safe_free((void **) &s->attn[li].k);
        safe_free((void **) &s->attn[li].v);
        safe_free((void **) &s->lconv[li].hist);
    }
    safe_free((void **) &s->sub_buf);
    safe_free((void **) &s->soft);
    safe_free((void **) &s->subs.l0);
    safe_free((void **) &s->subs.l1);
    safe_free((void **) &s);
}

static void audio_stream_state_reset(struct audio_stream_state *s) {
    if (s == nullptr)
        return;
    for (int li = 0; li < N_LAYERS; li++) {
        s->attn[li].n         = 0;
        s->lconv[li].n_filled = 0;
        /* No need to memset — n=0 / n_filled=0 guards readers. */
    }
    s->n_sub_total = 0;
    s->n_soft      = 0;
    /* Phase-3 cache reset: just clear the counters; the conv2d_fp32_from
     * overwrites the relevant cells before any subsequent reader. */
    s->subs.n_t_out0   = 0;
    s->subs.n_t_out1   = 0;
    s->subs.n_mel_seen = 0;
}

/* === Phase 1b: chunk-streaming forward (parity with audio_encoder_run). ===
 *
 * Process one block of new sub-tokens through one Conformer layer, threading
 * K/V cache (attention) and depthwise conv state (LConv) through state.
 *
 * h_chunk_io: (CHUNK_SIZE, AUDIO_HIDDEN) — input on entry, output on return.
 *             Always CHUNK_SIZE rows; if n_valid < CHUNK_SIZE the trailing
 *             rows are padding that must be zero-initialized by the caller.
 * n_valid:    actual sub-tokens in this block (≤ CHUNK_SIZE; last block
 *             of a final push may be shorter).
 * block_idx:  absolute block index (kv->n / CHUNK_SIZE BEFORE this call).
 * pos_emb:    (POS_LEN, AUDIO_HIDDEN) — constant across the whole utterance.
 *
 * Bit-for-bit equivalent to running the monolithic attn_run on the same
 * block's slice when the cache is read from the same positions. The chunked
 * attention layout has MAX_FUTURE=0, so per-block compute is causal and
 * does not depend on yet-unseen sub-tokens. */
static void attn_run_streaming_block(struct audio_stream_state *state,
                                     int                        layer_idx,
                                     const struct Attn         *attn,
                                     const float               *h_chunk_in,
                                     size_t                     n_valid,
                                     size_t                     block_idx,
                                     const float               *pos_emb,
                                     float                     *y_chunk_out) {
    const float           q_scale  = (1.0f / sqrtf((float) HEAD_DIM)) / logf(2.0f);
    const float           k_scale  = log1pf(2.71828182845904523536f) / logf(2.0f);
    const size_t          hd_per_t = (size_t) N_HEADS * HEAD_DIM;
    struct attn_kv_cache *kv       = &state->attn[layer_idx];

    /* 1. Q/K/V projection for the n_valid real rows. Q is padded to CHUNK_SIZE
     *    rows (zero pad) so the per-head attention loop has a stable shape. */
    float *h_clip = heap_alloc_array_aligned(float, (size_t) CHUNK_SIZE *AUDIO_HIDDEN);
    float *q      = heap_calloc_array_aligned(float, (size_t) CHUNK_SIZE *AUDIO_HIDDEN);
    float *k_new  = heap_alloc_array_aligned(float, n_valid *AUDIO_HIDDEN);
    float *v_new  = heap_alloc_array_aligned(float, n_valid *AUDIO_HIDDEN);

    memcpy(h_clip, h_chunk_in, n_valid * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->q_proj, h_clip, n_valid, AUDIO_HIDDEN, AUDIO_HIDDEN, q);
    memcpy(h_clip, h_chunk_in, n_valid * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->k_proj, h_clip, n_valid, AUDIO_HIDDEN, AUDIO_HIDDEN, k_new);
    memcpy(h_clip, h_chunk_in, n_valid * AUDIO_HIDDEN * sizeof(float));
    clip_linear_apply(&attn->v_proj, h_clip, n_valid, AUDIO_HIDDEN, AUDIO_HIDDEN, v_new);
    safe_free((void **) &h_clip);

    /* 2. Scale q and k. */
    float q_pds[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        q_pds[d] = q_scale * log1pf(expf(attn->per_dim_scale[d]));
    }
    for (size_t t = 0; t < n_valid; t++) {
        for (int head_i = 0; head_i < N_HEADS; head_i++) {
            float *qrow = q + ((size_t) t * N_HEADS + head_i) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++)
                qrow[d] *= q_pds[d];
        }
    }
    for (size_t i = 0; i < n_valid * AUDIO_HIDDEN; i++)
        k_new[i] *= k_scale;

    /* 3. Append scaled K, V to the cache. */
    memcpy(kv->k + kv->n * AUDIO_HIDDEN, k_new, n_valid * AUDIO_HIDDEN * sizeof(float));
    memcpy(kv->v + kv->n * AUDIO_HIDDEN, v_new, n_valid * AUDIO_HIDDEN * sizeof(float));
    safe_free((void **) &k_new);
    safe_free((void **) &v_new);
    const size_t kv_n_after = kv->n + n_valid;

    /* 4. Build K/V context window for this block from cache, matching the
     *    monolithic attn_run's b'th iteration: src = block_idx*12 + c - 12. */
    float *k_ctx = heap_calloc_array_aligned(float, (size_t) CONTEXT_SIZE *hd_per_t);
    float *v_ctx = heap_calloc_array_aligned(float, (size_t) CONTEXT_SIZE *hd_per_t);
    for (int c = 0; c < CONTEXT_SIZE; c++) {
        int src = (int) block_idx * CHUNK_SIZE + c - MAX_PAST_HORIZON;
        if (src < 0 || src >= (int) kv_n_after)
            continue;
        memcpy(k_ctx + (size_t) c * hd_per_t,
               kv->k + (size_t) src * hd_per_t,
               hd_per_t * sizeof(float));
        memcpy(v_ctx + (size_t) c * hd_per_t,
               kv->v + (size_t) src * hd_per_t,
               hd_per_t * sizeof(float));
    }

    /* 5. Compute the per-block mask locally. Same formula as
     *    audio_encoder_compute_attn_mask for the relevant block, with
     *    n = kv_n_after (total sub-tokens through layer 0 so far). */
    bool block_mask[CHUNK_SIZE * CONTEXT_SIZE] = {0};
    for (int i = 0; i < (int) n_valid; i++) {
        const int q_g = (int) block_idx * CHUNK_SIZE + i;
        if (q_g >= (int) kv_n_after)
            continue;
        for (int c = 0; c < CONTEXT_SIZE; c++) {
            const int k_g = (int) block_idx * CHUNK_SIZE + c - MAX_PAST_HORIZON;
            if (k_g < 0 || k_g >= (int) kv_n_after)
                continue;
            const int off = k_g - q_g;
            if (off < -(MAX_PAST_HORIZON - 1) || off > MAX_FUTURE)
                continue;
            block_mask[i * CONTEXT_SIZE + c] = true;
        }
    }

    /* 6. Relative-K projection — constant per encoder, recomputed per call
     *    (small: POS_LEN×1024). */
    float *rel_k = heap_alloc_array_aligned(float, (size_t) POS_LEN *AUDIO_HIDDEN);
    linear_fp32(
            pos_emb, attn->relative_k_proj, nullptr, POS_LEN, AUDIO_HIDDEN, AUDIO_HIDDEN, rel_k);

    /* 7. Attention output (CHUNK_SIZE, AUDIO_HIDDEN); only first n_valid used. */
    float *attn_out = heap_calloc_array_aligned(float, (size_t) CHUNK_SIZE *AUDIO_HIDDEN);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int hd = 0; hd < N_HEADS; hd++) {
        float scores_ac[CHUNK_SIZE * CONTEXT_SIZE];
        float scores_bd[CHUNK_SIZE * CONTEXT_SIZE];
        float bd_padded[CHUNK_SIZE * (CONTEXT_SIZE + 1)];

        const float *q_bh = q + (size_t) hd * HEAD_DIM;
        const float *k_bh = k_ctx + (size_t) hd * HEAD_DIM;
        const float *v_bh = v_ctx + (size_t) hd * HEAD_DIM;
        const float *rk_h = rel_k + (size_t) hd * HEAD_DIM;

        for (int i = 0; i < CHUNK_SIZE; i++) {
            const float *qi = q_bh + (size_t) i * hd_per_t;
            for (int j = 0; j < CONTEXT_SIZE; j++) {
                const float *kj                 = k_bh + (size_t) j * hd_per_t;
                scores_ac[i * CONTEXT_SIZE + j] = dot_head_fp32(qi, kj);
            }
        }
        for (int i = 0; i < CHUNK_SIZE; i++) {
            const float *qi = q_bh + (size_t) i * hd_per_t;
            for (int p = 0; p < POS_LEN; p++) {
                const float *rp                       = rk_h + (size_t) p * AUDIO_HIDDEN;
                bd_padded[i * (CONTEXT_SIZE + 1) + p] = dot_head_fp32(qi, rp);
            }
            for (int p = POS_LEN; p <= CONTEXT_SIZE; p++) {
                bd_padded[i * (CONTEXT_SIZE + 1) + p] = 0.0f;
            }
        }
        /* rel_shift — flat copy from bd_padded (stride CONTEXT_SIZE+1) into
         * scores_bd (stride CONTEXT_SIZE). This is a cyclic row shift that
         * aligns the relative-position bias across the chunk; same as the
         * monolithic attn_run. */
        for (int idx = 0; idx < CHUNK_SIZE * CONTEXT_SIZE; idx++) {
            scores_bd[idx] = bd_padded[idx];
        }
        for (int i = 0; i < CHUNK_SIZE; i++) {
            for (int j = 0; j < CONTEXT_SIZE; j++) {
                float s = scores_ac[i * CONTEXT_SIZE + j] + scores_bd[i * CONTEXT_SIZE + j];
                s       = tanhf(s / ATTN_SOFTCAP) * ATTN_SOFTCAP;
                if (!block_mask[i * CONTEXT_SIZE + j])
                    s = -1e9f;
                scores_ac[i * CONTEXT_SIZE + j] = s;
            }
        }
        softmax_fp32(scores_ac, CHUNK_SIZE, CONTEXT_SIZE);
        for (int i = 0; i < CHUNK_SIZE; i++) {
            float *out_row = attn_out + (size_t) i * AUDIO_HIDDEN + (size_t) hd * HEAD_DIM;
            zero_head_fp32(out_row);
            for (int j = 0; j < CONTEXT_SIZE; j++) {
                const float  w  = scores_ac[i * CONTEXT_SIZE + j];
                const float *vj = v_bh + (size_t) j * hd_per_t;
                axpy_head_fp32(out_row, w, vj);
            }
        }
    }

    safe_free((void **) &rel_k);
    safe_free((void **) &k_ctx);
    safe_free((void **) &v_ctx);
    safe_free((void **) &q);

    /* 8. Post-projection on the n_valid real rows only. */
    float *tmp_in = heap_alloc_array_aligned(float, n_valid *AUDIO_HIDDEN);
    memcpy(tmp_in, attn_out, n_valid * AUDIO_HIDDEN * sizeof(float));
    safe_free((void **) &attn_out);
    clip_linear_apply(&attn->post, tmp_in, n_valid, AUDIO_HIDDEN, AUDIO_HIDDEN, y_chunk_out);
    safe_free((void **) &tmp_in);

    /* 9. Advance cache. */
    kv->n = kv_n_after;
}

/* Streaming LConv: like lconv_run but the depthwise conv reads the last
 * (CONV_KERNEL-1) post-GLU rows from state for causal continuity, then
 * stores the last (CONV_KERNEL-1) of the new post-GLU h into state. */
static void lconv_run_streaming(struct audio_stream_state *state,
                                int                        layer_idx,
                                const struct LConv        *lc,
                                float                     *h,
                                size_t                     n) {
    const size_t        hsize = (size_t) n * AUDIO_HIDDEN;
    struct lconv_state *lcs   = &state->lconv[layer_idx];

    float *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h, hsize * sizeof(float));

    rmsnorm_fp32(h, lc->pre_norm, n, AUDIO_HIDDEN, RMS_EPS, h);

    float *doubled = heap_alloc_array_aligned(float, (size_t) n * 2 * AUDIO_HIDDEN);
    clip_linear_apply(&lc->linear_start, h, n, AUDIO_HIDDEN, 2 * AUDIO_HIDDEN, doubled);
    glu_fp32(doubled, n, AUDIO_HIDDEN, h);
    safe_free((void **) &doubled);

    /* Snapshot post-GLU h — both as conv input (with prepended history) and
     * as next call's state. */
    float *h_glu = heap_alloc_array_aligned(float, hsize);
    memcpy(h_glu, h, hsize * sizeof(float));

    /* Extended (T,C) input: zero-pad slot of (CONV_KERNEL-1) rows, with the
     * tail filled by lcs->hist (left-aligned to the boundary so n_filled<K-1
     * shows up as leading zeros — matches the implicit zero-pad of the
     * monolithic depthwise conv at sub-token 0). */
    const size_t prepend = (size_t) CONV_KERNEL - 1;
    const size_t ext_T   = prepend + n;
    float       *ext_in  = heap_calloc_array_aligned(float, ext_T *AUDIO_HIDDEN);
    if (lcs->n_filled > 0) {
        const size_t off = prepend - lcs->n_filled;
        memcpy(ext_in + off * AUDIO_HIDDEN,
               lcs->hist,
               lcs->n_filled * AUDIO_HIDDEN * sizeof(float));
    }
    memcpy(ext_in + prepend * AUDIO_HIDDEN, h_glu, hsize * sizeof(float));

    /* (T,C) → (C,T_ext) — depthwise conv expects per-channel rows. */
    float *ext_ct = heap_alloc_array_aligned(float, AUDIO_HIDDEN *ext_T);
    for (size_t t = 0; t < ext_T; t++) {
        for (int c = 0; c < AUDIO_HIDDEN; c++) {
            ext_ct[(size_t) c * ext_T + t] = ext_in[t * AUDIO_HIDDEN + c];
        }
    }
    safe_free((void **) &ext_in);

    float *conv_out = heap_alloc_array_aligned(float, AUDIO_HIDDEN *ext_T);
    depthwise_conv1d_causal_fp32(
            ext_ct, lc->depthwise, conv_out, AUDIO_HIDDEN, (int) ext_T, CONV_KERNEL);
    safe_free((void **) &ext_ct);

    /* Transpose back, taking only the n outputs corresponding to the new
     * sub-tokens (positions [prepend, prepend+n)). */
    for (size_t t = 0; t < n; t++) {
        for (int c = 0; c < AUDIO_HIDDEN; c++) {
            h[t * AUDIO_HIDDEN + c] = conv_out[(size_t) c * ext_T + (prepend + t)];
        }
    }
    safe_free((void **) &conv_out);

    /* Update lconv state: new hist = last min(K-1, n_filled+n) rows of
     * (lcs->hist || h_glu). Use a temp to avoid aliasing on lcs->hist. */
    const size_t total_avail  = lcs->n_filled + n;
    const size_t new_n_filled = total_avail < prepend ? total_avail : prepend;
    if (n >= new_n_filled) {
        memcpy(lcs->hist,
               h_glu + (n - new_n_filled) * AUDIO_HIDDEN,
               new_n_filled * AUDIO_HIDDEN * sizeof(float));
    } else {
        const size_t take_prior = new_n_filled - n;
        const size_t prior_off  = lcs->n_filled - take_prior;
        float       *tmp        = heap_alloc_array_aligned(float, new_n_filled *AUDIO_HIDDEN);
        memcpy(tmp,
               lcs->hist + prior_off * AUDIO_HIDDEN,
               take_prior * AUDIO_HIDDEN * sizeof(float));
        memcpy(tmp + take_prior * AUDIO_HIDDEN, h_glu, n * AUDIO_HIDDEN * sizeof(float));
        memcpy(lcs->hist, tmp, new_n_filled * AUDIO_HIDDEN * sizeof(float));
        safe_free((void **) &tmp);
    }
    lcs->n_filled = new_n_filled;
    safe_free((void **) &h_glu);

    rmsnorm_fp32(h, lc->conv_norm, n, AUDIO_HIDDEN, RMS_EPS, h);
    silu_fp32(h, hsize);
    float *end_in = heap_alloc_array_aligned(float, hsize);
    memcpy(end_in, h, hsize * sizeof(float));
    clip_linear_apply(&lc->linear_end, end_in, n, AUDIO_HIDDEN, AUDIO_HIDDEN, h);
    safe_free((void **) &end_in);

    for (size_t i = 0; i < hsize; i++)
        h[i] += residual[i];
    safe_free((void **) &residual);
}

/* Streaming per-layer forward. Mirrors audio_encoder_layer_run but threads
 * state through attention (K/V cache) and LConv (conv-state). */
static void audio_encoder_layer_run_streaming(const struct AudioEncoder *a,
                                              struct audio_stream_state *state,
                                              int                        layer_idx,
                                              float                     *h_io,
                                              size_t                     n_valid,
                                              size_t                     block_idx,
                                              const float               *pos_emb) {
    const struct ConformerLayer *L     = &a->layers[layer_idx];
    const size_t                 hsize = n_valid * AUDIO_HIDDEN;

    /* 1. FFN-1 (no state). */
    ffn_run(&L->ff1, h_io, n_valid);

    /* 2. norm_pre_attn + streaming attention + norm_post_attn + residual. */
    float *residual = heap_alloc_array_aligned(float, hsize);
    memcpy(residual, h_io, hsize * sizeof(float));
    rmsnorm_fp32(h_io, L->norm_pre_attn, n_valid, AUDIO_HIDDEN, RMS_EPS, h_io);

    float *attn_out = heap_alloc_array_aligned(float, hsize);
    attn_run_streaming_block(
            state, layer_idx, &L->attn, h_io, n_valid, block_idx, pos_emb, attn_out);
    memcpy(h_io, attn_out, hsize * sizeof(float));
    safe_free((void **) &attn_out);

    rmsnorm_fp32(h_io, L->norm_post_attn, n_valid, AUDIO_HIDDEN, RMS_EPS, h_io);
    for (size_t i = 0; i < hsize; i++)
        h_io[i] += residual[i];
    safe_free((void **) &residual);

    /* 3. LConv with conv-state. */
    lconv_run_streaming(state, layer_idx, &L->lconv, h_io, n_valid);

    /* 4. FFN-2 (no state). */
    ffn_run(&L->ff2, h_io, n_valid);

    /* 5. norm_out. */
    rmsnorm_fp32(h_io, L->norm_out, n_valid, AUDIO_HIDDEN, RMS_EPS, h_io);
}

/* Top-level streaming entry: drives chunk processing. Runs incremental
 * subsample on the cached l0/l1 intermediates (Phase 3 - extends the
 * cache with new time positions rather than recomputing from scratch),
 * identifies new sub-tokens, and pushes complete blocks (or the final
 * partial block if is_final) through the 12 streaming layers +
 * output_proj + embed_audio. Returns the number of NEW soft tokens
 * appended to state->soft. */
size_t audio_encoder_stream_push(const struct AudioEncoder *a,
                                 struct audio_stream_state *state,
                                 const float               *mel_full,
                                 const bool                *mel_mask,
                                 size_t                     n_mel_total,
                                 bool                       is_final) {
    if (state == nullptr || n_mel_total == 0)
        return 0;

    /* 1. Subsample. Default = full-mel re-run (Phase 2 behaviour, fastest
     *    for short push-to-talk clips where the alloc + memcpy overhead of
     *    the Phase-3 cache exceeds the conv2d savings). Opt-in to Phase-3
     *    incremental via GEIST_AUDIO_SUBSAMPLE_INC=1 for long-form audio
     *    where the conv2d savings win. */
    static int subs_inc = -1;
    if (subs_inc < 0) {
        const char *s = getenv("GEIST_AUDIO_SUBSAMPLE_INC");
        subs_inc      = (s != nullptr && s[0] == '1') ? 1 : 0;
    }
    const size_t n_sub_full =
            subs_inc ? audio_encoder_subsample_run_inc(
                               a, &state->subs, mel_full, mel_mask, n_mel_total, state->sub_buf)
                     : audio_encoder_subsample_run(
                               a, mel_full, mel_mask, n_mel_total, state->sub_buf);

    if (n_sub_full <= state->n_sub_total)
        return 0;

    /* 2. Determine block range to emit. Non-final: only full blocks.
     *    Final: include the partial trailing block (padded to CHUNK_SIZE). */
    const size_t block_start = state->n_sub_total / CHUNK_SIZE;
    const size_t block_end_excl =
            is_final ? (n_sub_full + CHUNK_SIZE - 1) / CHUNK_SIZE : n_sub_full / CHUNK_SIZE;
    if (block_end_excl <= block_start)
        return 0;

    /* 3. Constant pos_emb across blocks. */
    float *pos_emb = audio_encoder_compute_pos_emb(a);

    /* 4. Per-block × per-layer streaming. */
    float  h_chunk[CHUNK_SIZE * AUDIO_HIDDEN];
    size_t n_new_soft_total = 0;

    for (size_t b = block_start; b < block_end_excl; b++) {
        const size_t sub_start = b * CHUNK_SIZE;
        const size_t sub_end =
                (sub_start + CHUNK_SIZE) < n_sub_full ? (sub_start + CHUNK_SIZE) : n_sub_full;
        const size_t n_chunk = sub_end - sub_start;

        memset(h_chunk, 0, sizeof(h_chunk));
        memcpy(h_chunk,
               state->sub_buf + sub_start * AUDIO_HIDDEN,
               n_chunk * AUDIO_HIDDEN * sizeof(float));

        for (int li = 0; li < N_LAYERS; li++) {
            audio_encoder_layer_run_streaming(a, state, li, h_chunk, n_chunk, b, pos_emb);
        }

        /* output_proj + embed_audio on the n_chunk real rows. */
        float *op = heap_alloc_array_aligned(float, n_chunk *OUTPUT_PROJ_DIMS);
        linear_fp32(h_chunk,
                    a->output_proj_w,
                    a->output_proj_b,
                    n_chunk,
                    AUDIO_HIDDEN,
                    OUTPUT_PROJ_DIMS,
                    op);

        float *normed = heap_alloc_array_aligned(float, n_chunk *OUTPUT_PROJ_DIMS);
        rmsnorm_fp32(op, nullptr, n_chunk, OUTPUT_PROJ_DIMS, RMS_EPS, normed);
        safe_free((void **) &op);

        linear_fp32(normed,
                    a->embed_audio_proj,
                    nullptr,
                    n_chunk,
                    OUTPUT_PROJ_DIMS,
                    TEXT_HIDDEN,
                    state->soft + state->n_soft * AUDIO_SOFT_TOKEN_DIM);
        safe_free((void **) &normed);

        state->n_soft += n_chunk;
        state->n_sub_total = sub_end;
        n_new_soft_total += n_chunk;
    }

    safe_free((void **) &pos_emb);
    return n_new_soft_total;
}

/* Public accessor: return the current streaming state pointer (read-only
 * for tests; lifecycle is owned by audio_encoder_create/destroy/reset). */
struct audio_stream_state *audio_encoder_stream_state(struct AudioEncoder *a) {
    return a ? a->stream : nullptr;
}

const float *audio_stream_state_soft(const struct audio_stream_state *s) {
    return s ? s->soft : nullptr;
}

size_t audio_stream_state_n_soft(const struct audio_stream_state *s) {
    return s ? s->n_soft : 0;
}

/* === Phase 2: streaming worker thread driver. === */

/* Run one stream_push iteration. Caller has dropped a->mtx; we re-grab it
 * only to copy out the snapshot info, then drop again for the heavy
 * compute (which mutates state but not a->mel_buf — push_pcm continues
 * to append to mel_buf concurrently, but only the snapshot range is
 * read here so it's stable). */
static void worker_do_push(struct AudioEncoder *a, size_t mel_snap, bool is_final) {
    if (mel_snap == 0)
        return;
    /* Local mask — all-true for valid frames is fine since push_pcm has
     * already produced them via the mel pipeline (no silence-skip path
     * is wired through the streaming API). */
    bool *mask = heap_calloc_array_aligned(bool, mel_snap);
    for (size_t i = 0; i < mel_snap; i++)
        mask[i] = true;
    (void) audio_encoder_stream_push(a, a->stream, a->mel_buf, mask, mel_snap, is_final);
    safe_free((void **) &mask);
}

static void *audio_encoder_stream_worker(void *arg) {
    struct AudioEncoder *a = (struct AudioEncoder *) arg;
    pthread_mutex_lock(&a->mtx);
    while (true) {
        while (!a->worker_kick && !a->shutdown_flag) {
            pthread_cond_wait(&a->cv, &a->mtx);
        }
        if (a->shutdown_flag)
            break;
        a->worker_kick        = false;
        const size_t mel_snap = a->mel_n_computed;
        const bool   is_final = a->worker_final;
        a->worker_last_mel    = mel_snap;
        pthread_mutex_unlock(&a->mtx);

        worker_do_push(a, mel_snap, is_final);

        pthread_mutex_lock(&a->mtx);
        if (is_final) {
            a->computed_flag = true;
            a->worker_final  = false;
        }
        pthread_cond_broadcast(&a->cv);
    }
    pthread_mutex_unlock(&a->mtx);
    return nullptr;
}

/* Enable the streaming worker (called from audio_encoder_create when
 * GEIST_AUDIO_STREAM=1). Spawns the worker thread. */
static bool stream_worker_start(struct AudioEncoder *a) {
    a->stream_enabled  = true;
    a->worker_active   = true;
    a->worker_kick     = false;
    a->worker_final    = false;
    a->worker_last_mel = 0;
    if (pthread_create(&a->worker_tid, nullptr, audio_encoder_stream_worker, a) != 0) {
        a->worker_active  = false;
        a->stream_enabled = false;
        return false;
    }
    return true;
}

static void stream_worker_stop(struct AudioEncoder *a) {
    if (!a->worker_active)
        return;
    pthread_mutex_lock(&a->mtx);
    a->shutdown_flag = true;
    pthread_cond_broadcast(&a->cv);
    pthread_mutex_unlock(&a->mtx);
    pthread_join(a->worker_tid, nullptr);
    a->worker_active = false;
}
