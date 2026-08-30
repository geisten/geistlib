/*
 * bench_audio_encode — standalone audio-encoder profile bench.
 *
 * Skips the LM model entirely so it fits on 4 GB Pi 5 (the audio
 * encoder + safetensors is ~600 MB FP32, far under the IQ2_M LM's
 * 2.8 GB). Runs one WAV file through the encoder + prints wall time;
 * when built with -DGEIST_AUDIO_PROFILE the encoder also prints its
 * per-stage breakdown to stderr.
 *
 *   bench_audio_encode <path/to/audio.wav>
 *
 * Resolves audio_tower.safetensors + mel_constants.bin via the normal
 * audio_conformer arch lookup (audio_bench/, audio_test_data/, etc.).
 */
#include "audio_test_util.h"
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/audio_conformer/audio_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

/* Chunk-walking WAV reader (audio_test_util.h) — the fixed-44-byte
 * shortcut mis-read ffmpeg WAVs with a LIST chunk (#268). */
static int16_t *read_wav_pcm(const char *path, size_t *n_samples_out, int *sample_rate_out) {
    return audio_test_read_wav(path, n_samples_out, sample_rate_out);
}

int main(int argc, char **argv) {
    /* Default to a shipped test asset so `make bench-audio` works from a
     * clean checkout. Override with explicit args for custom benchmarks. */
    const char *wav_path         = argc > 1 ? argv[1] : "audio_test_data/de_hello.wav";
    const char *audio_tower_path = argc > 2 ? argv[2] : "audio_bench/audio_tower.safetensors";
    size_t      n_samples;
    int         sample_rate;
    int16_t    *pcm = read_wav_pcm(wav_path, &n_samples, &sample_rate);
    if (pcm == nullptr) {
        fprintf(stderr, "could not read wav: %s\n", wav_path);
        return GEIST_TEST_FAIL;
    }
    if (sample_rate != 16000) {
        fprintf(stderr, "wav sample rate must be 16 kHz (got %d)\n", sample_rate);
        free(pcm);
        return GEIST_TEST_FAIL;
    }
    const double audio_seconds = (double) n_samples / 16000.0;
    printf("loaded %zu PCM samples = %.2f s of audio\n", n_samples, audio_seconds);

    const double         t_create_0 = now_ms();
    struct AudioEncoder *enc        = audio_encoder_create(audio_tower_path);
    const double         t_create   = now_ms() - t_create_0;
    if (enc == nullptr) {
        fprintf(stderr,
                "audio_encoder_create failed (missing "
                "audio_tower.safetensors or mel_constants.bin?)\n");
        free(pcm);
        return GEIST_TEST_FAIL;
    }
    printf("encoder loaded: %.0f ms (cold)\n", t_create);

    const double t_push_0 = now_ms();
    if (audio_encoder_push_pcm(enc, n_samples, pcm) != 0) {
        fprintf(stderr, "push_pcm failed\n");
        audio_encoder_destroy(enc);
        free(pcm);
        return GEIST_TEST_FAIL;
    }
    audio_encoder_end_input(enc);
    const double t_push = now_ms() - t_push_0;

    /* Pull soft-tokens — triggers full mel→subsample→12-layer→proj pipeline
     * on the first call after end_input. */
    const size_t soft_dim = 1536;
    const size_t max_soft = audio_encoder_max_soft_tokens(n_samples);
    float       *soft     = malloc(max_soft * soft_dim * sizeof(float));
    if (soft == nullptr) {
        audio_encoder_destroy(enc);
        free(pcm);
        return GEIST_TEST_FAIL;
    }

    const double t_encode_0 = now_ms();
    const size_t n_soft     = audio_encoder_pull_softtokens(enc, max_soft, soft, -1);
    const double t_encode   = now_ms() - t_encode_0;

    printf("\n--- bench_audio_encode results ---\n");
    printf("  audio length      : %.2f s\n", audio_seconds);
    printf("  push_pcm + end_in : %6.1f ms\n", t_push);
    printf("  pull (full pipe)  : %6.1f ms  (mel + subsample + 12-layer + projections)\n",
           t_encode);
    printf("  soft tokens out   : %zu × 1536 floats\n", n_soft);
    printf("  realtime factor   : %.2fx (encode wall / audio wall)\n",
           t_encode / 1e3 / audio_seconds);

    free(soft);
    audio_encoder_destroy(enc);
    free(pcm);
    return GEIST_TEST_PASS;
}
