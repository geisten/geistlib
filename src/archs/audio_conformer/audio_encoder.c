/*
 * src/archs/audio_conformer/audio_encoder.c — encoder orchestration
 * (create/destroy) and the profiling accumulators.
 *
 * Layer: ARCHITECTURE (audio_conformer). The heavy lifting lives in
 * encoder_weights.c / encoder_stream.c / encoder_forward.c since the
 * module split; this file wires them together.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "encoder_internal.h"

#include "audio_linear.h"

#ifdef GEIST_AUDIO_PROFILE
struct ae_stage_timer g_ae_subsample, g_ae_pos_emb, g_ae_layer_total, g_ae_ffn1, g_ae_norm_pre_attn,
        g_ae_attn, g_ae_norm_post_attn, g_ae_lconv, g_ae_ffn2, g_ae_norm_out, g_ae_output_proj,
        g_ae_embed_proj;
#endif

#ifdef GEIST_AUDIO_PROFILE
void ae_profile_print_and_reset(void) {
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
#endif

struct AudioEncoder *audio_encoder_create(const char *safetensors_path) {
    const char    *err = nullptr;
    struct st_ctx *sf  = st_open(safetensors_path, &err);
    if (!sf) {
        fprintf(stderr, "audio_encoder: %s\n", err);
        return nullptr;
    }

    {
        /* Mirror load_clippable's Apple override — otherwise this line
         * claims W8A8 while the loader silently forced FP32, and the one
         * diagnostic users have lies about what was loaded. */
        bool forced_fp32 = false;
#if defined(__APPLE__) && defined(HAVE_ACCELERATE)
        forced_fp32 = getenv("GEIST_AUDIO_FORCE_QUANT") == nullptr;
#endif
        if (forced_fp32) {
            fprintf(stderr,
                    "audio_encoder: per-tensor precision: all FP32 (Accelerate "
                    "default; GEIST_AUDIO_FORCE_QUANT=1 to quantize)\n");
        } else {
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
    }

    /* Bind the quantized matmul kernels once, from the runtime probe —
     * every clip_linear_apply afterwards is a cached pointer read (#236). */
    fprintf(stderr, "audio_encoder: linear kernels: %s\n", audio_linear_bind()->name);

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
