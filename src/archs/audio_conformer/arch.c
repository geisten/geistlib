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
#include "encoder_internal.h" /* MEL_HOP, AUDIO_MAX_MEL_FRAMES, mel-mask helper */
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
 * 0 on bad input. Framing per HF's Gemma4AudioFeatureExtractor and
 * identical to the streaming path (encoder_stream.c push_pcm): 10 ms hop
 * (MEL_HOP), one 160-sample zero left-pad. Frame i reads
 * padded[i*160 .. i*160+319] = pcm[i*160-160 .. i*160+159].
 * (This path ran a 20 ms hop with no pad until the framing-parity fix —
 * half the tokens of the reference for the same audio.) */
static size_t pcm_to_mel(struct MelState *mel,
                         const int16_t   *pcm,
                         size_t           n_samples,
                         float           *mel_out,
                         size_t           max_frames) {
    if (mel == nullptr || pcm == nullptr) {
        return 0;
    }
    size_t n_frames = n_samples / MEL_HOP;
    if (n_frames > max_frames) {
        n_frames = max_frames;
    }
    float frame_pcm[MEL_FRAME_LENGTH];
    for (size_t i = 0; i < n_frames; i++) {
        for (size_t k = 0; k < MEL_FRAME_LENGTH; k++) {
            const long pi = (long) (i * MEL_HOP + k) - MEL_HOP; /* unshift left-pad */
            frame_pcm[k]  = pi < 0 ? 0.0f : (float) pcm[pi] / 32768.0f;
        }
        mel_frame_compute(mel, frame_pcm, mel_out + i * MEL_N_MEL);
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

    /* Refuse (not silently truncate — #247) input beyond the 30 s the
     * model was built for. The session surfaces this as an error. */
    const size_t n_frames = n_samples / MEL_HOP;
    if (n_frames == 0 || n_frames > AUDIO_MAX_MEL_FRAMES) {
        return 0;
    }
    /* +1 padded (mask=false) frame — same convention as the streaming
     * path's final push. */
    size_t n_mel;
    bool  *mask = audio_mel_mask_alloc(n_frames, true, &n_mel);
    float *mel  = heap_calloc_array_aligned(float, n_mel *MEL_N_MEL); /* pad row = zeros */
    if (mel == nullptr || mask == nullptr) {
        safe_free((void **) &mel);
        safe_free((void **) &mask);
        return 0;
    }

    /* Capacity check BEFORE running: audio_encoder_run has no output cap,
     * so a too-small caller buffer must be refused up front, not detected
     * after the overflow. Exact subsample arithmetic: two stride-2 convs. */
    const size_t t_out0           = (n_mel - 1) / 2 + 1;
    const size_t n_soft_predicted = (t_out0 - 1) / 2 + 1;
    if (n_soft_predicted > max_soft) {
        safe_free((void **) &mel);
        safe_free((void **) &mask);
        return 0;
    }

    size_t got = pcm_to_mel(st->mel, pcm, n_samples, mel, n_frames);
    if (got != n_frames) {
        safe_free((void **) &mel);
        safe_free((void **) &mask);
        return 0;
    }

    size_t n_soft = audio_encoder_run(st->enc, mel, mask, n_mel, out_soft);

    /* GEIST_AUDIO_DEBUG_DUMP=<prefix>: write the mel and soft-token
     * stages as raw fp32 for reference-parity diagnosis (#268). */
    const char *dump = getenv("GEIST_AUDIO_DEBUG_DUMP");
    if (dump != nullptr && n_soft > 0) {
        char  path[512];
        FILE *f;
        snprintf(path, sizeof path, "%s.mel.bin", dump);
        if ((f = fopen(path, "wb")) != nullptr) {
            fwrite(mel, sizeof(float), n_mel * MEL_N_MEL, f);
            fclose(f);
        }
        snprintf(path, sizeof path, "%s.soft.bin", dump);
        if ((f = fopen(path, "wb")) != nullptr) {
            fwrite(out_soft, sizeof(float), n_soft * audio_encoder_soft_dim(st->enc), f);
            fclose(f);
        }
        fprintf(stderr,
                "audio dump: %zu mel frames, %zu soft tokens -> %s.*\n",
                n_mel,
                n_soft,
                dump);
    }
    safe_free((void **) &mel);
    safe_free((void **) &mask);
    return n_soft;
}

static bool audio_conformer_stream_begin(void *encoder_state) {
    struct audio_conformer_state *st = encoder_state;
    if (st == nullptr || st->enc == nullptr) {
        return false;
    }
    /* Lazy worker start: session-level streaming always overlaps encode
     * with the arriving PCM, no env needed. Reset clears any previous
     * utterance. */
    if (!stream_worker_start(st->enc)) {
        return false;
    }
    audio_encoder_reset(st->enc);
    return true;
}

static bool audio_conformer_stream_push(void *encoder_state, const int16_t *pcm, size_t n) {
    struct audio_conformer_state *st = encoder_state;
    if (st == nullptr || st->enc == nullptr || pcm == nullptr) {
        return false;
    }
    return audio_encoder_push_pcm(st->enc, pcm, n) == 0;
}

static size_t audio_conformer_stream_poll(void *encoder_state, float *out_soft, size_t max_soft) {
    struct audio_conformer_state *st = encoder_state;
    if (st == nullptr || st->enc == nullptr || out_soft == nullptr || max_soft == 0) {
        return 0;
    }
    /* timeout 0 = non-blocking: only tokens the worker already emitted. */
    return audio_encoder_pull_softtokens(st->enc, out_soft, max_soft, 0);
}

static size_t audio_conformer_stream_end(void *encoder_state, float *out_soft, size_t max_soft) {
    struct audio_conformer_state *st = encoder_state;
    if (st == nullptr || st->enc == nullptr || out_soft == nullptr || max_soft == 0) {
        return 0;
    }
    audio_encoder_end_input(st->enc);
    const size_t dim   = audio_encoder_soft_dim(st->enc);
    size_t       total = 0;
    size_t       got;
    while (total < max_soft &&
           (got = audio_encoder_pull_softtokens(
                    st->enc, out_soft + total * dim, max_soft - total, -1)) > 0) {
        total += got;
    }
    return total;
}

static size_t audio_conformer_max_soft_tokens(const void *encoder_state, size_t n_samples) {
    (void) encoder_state;
    return audio_encoder_max_soft_tokens(n_samples);
}

static size_t audio_conformer_soft_token_dim(const void *encoder_state) {
    const struct audio_conformer_state *st = encoder_state;
    /* Per-checkpoint: 1536 (E2B) or 2560 (E4B) — the text model's
     * residual-stream width, read from the tower safetensors (#258). */
    return audio_encoder_soft_dim(st != nullptr ? st->enc : nullptr);
}

const struct geist_arch_ops_encoder geist_arch_audio_conformer = {
        .name            = "audio_conformer",
        .state_create    = audio_conformer_state_create,
        .state_destroy   = audio_conformer_state_destroy,
        .encode_pcm      = audio_conformer_encode_pcm,
        .soft_token_dim  = audio_conformer_soft_token_dim,
        .stream_begin    = audio_conformer_stream_begin,
        .stream_push     = audio_conformer_stream_push,
        .stream_poll     = audio_conformer_stream_poll,
        .stream_end      = audio_conformer_stream_end,
        .max_soft_tokens = audio_conformer_max_soft_tokens,
};
