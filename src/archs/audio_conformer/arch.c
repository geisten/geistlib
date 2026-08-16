/*
 * src/archs/audio_conformer/arch.c — Gemma 4 audio Conformer encoder.
 *
 * Layer: ARCHITECTURE.
 *
 * Phase B-5: wrap existing audio_encoder.c + mel_pipeline.c into the
 * engine's geist_arch_ops_encoder vtable.
 *
 * Search heuristics for aux files (audio_tower.safetensors + mel_constants.bin):
 *   - $GEIST_AUDIO_MODEL_PATH / $GEIST_MEL_CONSTANTS_PATH env overrides
 *   - <aux_search_root>/audio_tower.safetensors / mel_constants.bin
 *   - ./audio_bench/ and ./audio_test_data/ (project default locations)
 *   - ../gemma-4-E2B-it/audio_tower.safetensors
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch.h"

#include "audio_encoder.h"
#include "heap.h"
#include "mel_pipeline.h"

#include <geist.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct audio_conformer_state {
    struct AudioEncoder *enc; /* weights + Conformer layers */
    struct MelState     *mel; /* per-frame mel computation */
};

static char *find_file(const char        *env_name,
                       const char        *aux_root,
                       const char        *basename,
                       const char *const *fallbacks) {
    const char *env = env_name != nullptr ? getenv(env_name) : nullptr;
    if (env != nullptr && env[0] != '\0') {
        FILE *f = fopen(env, "rb");
        if (f != nullptr) {
            fclose(f);
            return strdup(env);
        }
    }
    if (aux_root != nullptr) {
        size_t need = strlen(aux_root) + 1 + strlen(basename) + 1;
        char  *cand = heap_alloc_aligned(need, alignof(char));
        if (cand != nullptr) {
            snprintf(cand, need, "%s/%s", aux_root, basename);
            FILE *f = fopen(cand, "rb");
            if (f != nullptr) {
                fclose(f);
                return cand;
            }
            safe_free((void **) &cand);
        }
    }
    if (fallbacks != nullptr) {
        for (size_t i = 0; fallbacks[i] != nullptr; i++) {
            FILE *f = fopen(fallbacks[i], "rb");
            if (f != nullptr) {
                fclose(f);
                return strdup(fallbacks[i]);
            }
        }
    }
    return nullptr;
}

static void *audio_conformer_state_create(struct geist_backend *be, const char *aux_root) {
    (void) be; /* B-5: audio encoder still uses its own allocation; B-6 cleanup will route. */

    static const char *audio_fallbacks[] = {
            "./audio_bench/audio_tower.safetensors",
            "../gemma-4-E2B-it/audio_tower.safetensors",
            "audio_tower.safetensors",
            nullptr,
    };
    static const char *mel_fallbacks[] = {
            "./audio_test_data/mel_constants.bin",
            "./mel_constants.bin",
            "../gemma-4-E2B-it/mel_constants.bin",
            nullptr,
    };

    char *audio_path = find_file(
            "GEIST_AUDIO_MODEL_PATH", aux_root, "audio_tower.safetensors", audio_fallbacks);
    if (audio_path == nullptr) {
        return nullptr;
    }
    char *mel_path =
            find_file("GEIST_MEL_CONSTANTS_PATH", aux_root, "mel_constants.bin", mel_fallbacks);
    if (mel_path == nullptr) {
        safe_free((void **) &audio_path);
        return nullptr;
    }

    struct AudioEncoder *enc = audio_encoder_create(audio_path);
    safe_free((void **) &audio_path);
    if (enc == nullptr) {
        safe_free((void **) &mel_path);
        return nullptr;
    }
    struct MelState *mel = mel_create(mel_path);
    safe_free((void **) &mel_path);
    if (mel == nullptr) {
        audio_encoder_destroy(enc);
        return nullptr;
    }

    struct audio_conformer_state *st =
            heap_alloc_aligned(sizeof(*st), alignof(struct audio_conformer_state));
    if (st == nullptr) {
        mel_destroy(mel);
        audio_encoder_destroy(enc);
        return nullptr;
    }
    *st = (struct audio_conformer_state) {.enc = enc, .mel = mel};
    return st;
}

static void audio_conformer_state_destroy(void *encoder_state) {
    if (encoder_state == nullptr) {
        return;
    }
    struct audio_conformer_state *st = encoder_state;
    if (st->mel != nullptr) {
        mel_destroy(st->mel);
    }
    if (st->enc != nullptr) {
        audio_encoder_destroy(st->enc);
    }
    safe_free(&encoder_state);
}

/* Compute mel frames from int16 PCM at 16 kHz. Returns n_frames on success,
 * 0 on bad input. PCM is processed in 320-sample windows (20 ms hop=full
 * frame); the existing mel pipeline matches HF's Gemma4AudioFeatureExtractor. */
static size_t pcm_to_mel(struct MelState *mel,
                         const int16_t   *pcm,
                         size_t           n_samples,
                         float           *mel_out,
                         bool            *mask_out,
                         size_t           max_frames) {
    if (mel == nullptr || pcm == nullptr) {
        return 0;
    }
    size_t n_frames = n_samples / MEL_FRAME_LENGTH;
    if (n_frames > max_frames) {
        n_frames = max_frames;
    }
    float frame_pcm[MEL_FRAME_LENGTH];
    for (size_t i = 0; i < n_frames; i++) {
        for (size_t k = 0; k < MEL_FRAME_LENGTH; k++) {
            frame_pcm[k] = (float) pcm[i * MEL_FRAME_LENGTH + k] / 32768.0f;
        }
        mel_frame_compute(mel, frame_pcm, mel_out + i * MEL_N_MEL);
        if (mask_out != nullptr) {
            mask_out[i] = true;
        }
    }
    return n_frames;
}

static size_t audio_conformer_encode_pcm(void          *encoder_state,
                                         const int16_t *pcm,
                                         size_t         n_samples,
                                         float         *out_soft,
                                         size_t         max_soft) {
    if (encoder_state == nullptr || pcm == nullptr || out_soft == nullptr || max_soft == 0) {
        return 0;
    }
    struct audio_conformer_state *st = encoder_state;

    /* Max mel frames = 1500 (30 seconds at 50 Hz frame rate). */
    static constexpr size_t MAX_MEL_FRAMES = 1500;
    if (n_samples > MAX_MEL_FRAMES * MEL_FRAME_LENGTH) {
        n_samples = MAX_MEL_FRAMES * MEL_FRAME_LENGTH;
    }
    size_t n_mel_frames = n_samples / MEL_FRAME_LENGTH;
    if (n_mel_frames == 0) {
        return 0;
    }

    float *mel  = heap_alloc_array_aligned(float, n_mel_frames *MEL_N_MEL);
    bool  *mask = heap_alloc_array_aligned(bool, n_mel_frames);
    if (mel == nullptr || mask == nullptr) {
        if (mel != nullptr)
            safe_free((void **) &mel);
        if (mask != nullptr)
            safe_free((void **) &mask);
        return 0;
    }

    size_t got = pcm_to_mel(st->mel, pcm, n_samples, mel, mask, n_mel_frames);
    if (got != n_mel_frames) {
        safe_free((void **) &mel);
        safe_free((void **) &mask);
        return 0;
    }

    /* struct AudioEncoder downsamples 4× then runs Conformer; for max_soft to be a
     * useful cap, the caller has to size their out_soft buffer accordingly.
     * audio_encoder_run produces soft tokens up to its internal limit. */
    size_t n_soft = audio_encoder_run(st->enc, mel, mask, n_mel_frames, out_soft);
    safe_free((void **) &mel);
    safe_free((void **) &mask);

    if (n_soft > max_soft) {
        /* Caller-supplied buffer too small — bytes beyond max_soft were
         * not actually written (audio_encoder_run overflowed); treat as
         * error to avoid misleading the caller. */
        return 0;
    }
    return n_soft;
}

static size_t audio_conformer_max_soft_tokens(const void *encoder_state, size_t n_samples) {
    (void) encoder_state;
    return audio_encoder_max_soft_tokens(n_samples);
}

static size_t audio_conformer_soft_token_dim(const void *encoder_state) {
    (void) encoder_state;
    return AUDIO_SOFT_TOKEN_DIM; /* 1536 for Gemma 4 */
}

const struct geist_arch_ops_encoder geist_arch_audio_conformer = {
        .name            = "audio_conformer",
        .state_create    = audio_conformer_state_create,
        .state_destroy   = audio_conformer_state_destroy,
        .encode_pcm      = audio_conformer_encode_pcm,
        .soft_token_dim  = audio_conformer_soft_token_dim,
        .max_soft_tokens = audio_conformer_max_soft_tokens,
};
