/*
 * src/archs/audio_conformer/encoder_internal.h — shared types, config
 * macros, and cross-module prototypes of the Conformer audio encoder.
 *
 * Layer: ARCHITECTURE (audio_conformer, internal). The encoder is split
 * by responsibility: encoder_weights.c (load/quantize/free),
 * encoder_stream.c (worker thread + streaming API), encoder_forward.c
 * (Conformer stages), audio_encoder.c (create/destroy orchestration).
 */
#ifndef GEIST_INTERNAL_AUDIO_ENCODER_INTERNAL_H
#define GEIST_INTERNAL_AUDIO_ENCODER_INTERNAL_H
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
#include <stdatomic.h>
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
/* Defined in audio_encoder.c. */
extern struct ae_stage_timer g_ae_subsample, g_ae_pos_emb, g_ae_layer_total, g_ae_ffn1,
        g_ae_norm_pre_attn, g_ae_attn, g_ae_norm_post_attn, g_ae_lconv, g_ae_ffn2, g_ae_norm_out,
        g_ae_output_proj, g_ae_embed_proj;
static inline double ae_now_ms_(void) {
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
void ae_profile_print_and_reset(void);
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
    /* `w` is dropped after quantization (saves ~450 MB on a Pi 5);
     * it stays populated only for prec == AUDIO_PREC_FP32 layers.
     * `w_q8` / `w_scales` are populated iff prec != AUDIO_PREC_FP32. */
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

/* Upper bound on soft tokens for n_mel mel frames: the subsample stage
 * divides by 4, plus margin for partial blocks. THE single home of this
 * formula — session buffer sizing, the streaming path and the bench all
 * derive from it (#247). */
static inline size_t audio_soft_bound_from_mel(size_t n_mel) {
    return (n_mel + 3) / 4 + 4;
}

/* Mel framing constants shared by the one-shot (encode_pcm) and streaming
 * (push_pcm) paths: 10 ms hop, one 160-sample zero left-pad, and one
 * padded (mask=false) frame appended after the real ones — HF's
 * Gemma4AudioFeatureExtractor convention. The one-shot path used a 20 ms
 * hop with neither pad until the framing-parity fix; keep BOTH paths on
 * these constants so they can never diverge again. */
#define MEL_HOP 160
#define AUDIO_MAX_MEL_FRAMES 3000 /* 30 s at the 10 ms hop */

/* All-true mask for n_frames real frames, optionally with the extra
 * padded frame (mask=false). Caller frees. The single home of the
 * padded-frame convention — #235's missing-token bug and the one-shot
 * path's missing pad were both callers hand-rolling it. */
static inline bool *audio_mel_mask_alloc(size_t n_frames, bool pad_final, size_t *n_mel_out) {
    const size_t n_mel = pad_final ? n_frames + 1 : n_frames;
    bool        *mask  = heap_calloc_array_aligned(bool, n_mel);
    if (mask != nullptr) {
        for (size_t i = 0; i < n_frames; i++)
            mask[i] = true;
    }
    if (n_mel_out != nullptr)
        *n_mel_out = n_mel;
    return mask;
}

/* encoder_weights.c: shared "env var == '1'" read and the single
 * platform-precision decision (banner + loader read the same truth). */
bool audio_env_flag(const char *name);
bool audio_prec_forced_fp32(void);

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

    /* Constant (POS_LEN, AUDIO_HIDDEN) table, computed on first push and
     * reused across kicks — it never changes within an encoder. */
    float *pos_emb;

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

/* ---- Cross-module prototypes ------------------------------------------ */
float                     *load_bf16(struct st_ctx *sf, const char *name, size_t expect_elems);
bool                       load_layer(struct st_ctx *sf, int layer_idx, struct ConformerLayer *L);
void                       free_layer(struct ConformerLayer *L);
struct audio_stream_state *audio_stream_state_create(void);
void                       audio_stream_state_destroy(struct audio_stream_state *s);
bool                       stream_worker_start(struct AudioEncoder *a);
void                       stream_worker_stop(struct AudioEncoder *a);
void                       clip_linear_apply(const struct ClippableLinear *cl,
                                             float                        *x,
                                             size_t                        n,
                                             size_t                        in_dim,
                                             size_t                        out_dim,
                                             float                        *y);
void                       ffn_run(const struct FFN *ffn, float *h, size_t n);
float                      dot_head_fp32(const float *a, const float *b);
void                       axpy_head_fp32(float *out, float w, const float *v);
void                       zero_head_fp32(float *out);

void   attn_run(const struct Attn *attn,
                const float       *h,
                size_t             n,
                const float       *pos_emb,
                const bool        *attn_mask,
                float             *y);
void   lconv_run(const struct LConv *lc, float *h, size_t n);
void   audio_encoder_layer_run(const struct AudioEncoder *a,
                               int                        layer_idx,
                               const float               *h_in,
                               size_t                     n,
                               const float               *pos_emb,
                               const bool                *attn_mask_5d,
                               float                     *h_out);
float *audio_encoder_compute_pos_emb(const struct AudioEncoder *a);
bool  *audio_encoder_compute_attn_mask(const struct AudioEncoder *a, size_t n);
size_t audio_encoder_subsample_run(const struct AudioEncoder *a,
                                   const float               *mel_in,
                                   const bool                *mask_in,
                                   size_t                     n_mel_frames,
                                   float                     *out);
size_t audio_encoder_subsample_run_inc(const struct AudioEncoder *a,
                                       struct subs_cache         *subs,
                                       const float               *mel_in,
                                       const bool                *mask_in,
                                       size_t                     n_mel_total,
                                       float                     *out_sub_buf);

#endif /* GEIST_INTERNAL_AUDIO_ENCODER_INTERNAL_H */
