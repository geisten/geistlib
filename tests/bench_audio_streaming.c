/*
 * bench_audio_streaming — simulate live-mic capture by pushing PCM in
 * 20 ms chunks at real-time rate, then measure the residual encode wall
 * after end_input(). With Phase C's incremental mel pipeline this drops
 * by the mel cost (mel runs concurrent with capture); without it the
 * residual is the full encode time.
 *
 *   bench_audio_streaming <audio.wav> [audio_tower.safetensors]
 *
 * Numbers reported:
 *   t_capture  : audio duration (= simulated capture wall)
 *   t_residual : end_input() → soft-tokens ready (this is the user-
 *                 perceived "compute tail" after they stop speaking)
 *   t_total    : first push → soft-tokens ready
 *
 * Compare to bench_audio_encode for the offline-no-realtime-gate baseline.
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

#define CHUNK_PCM_SAMPLES 320 /* 20 ms @ 16 kHz */
#define CHUNK_NS (20 * 1000 * 1000L)

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

    size_t   n_samples;
    int      sample_rate;
    int16_t *pcm = read_wav_pcm(wav_path, &n_samples, &sample_rate);
    if (pcm == nullptr || sample_rate != 16000) {
        fprintf(stderr, "wav read failed or sample rate != 16kHz\n");
        if (pcm)
            free(pcm);
        return GEIST_TEST_FAIL;
    }
    const double audio_seconds = (double) n_samples / 16000.0;
    printf("loaded %zu PCM samples = %.2f s of audio\n", n_samples, audio_seconds);

    struct AudioEncoder *enc = audio_encoder_create(audio_tower_path);
    if (enc == nullptr) {
        free(pcm);
        return GEIST_TEST_FAIL;
    }

    /* Simulate live capture: push 320-sample chunks every 20 ms. The chunk
     * boundary is wall-anchored at start (not cumulative across pushes) so
     * mel work inside push_pcm doesn't compound jitter. */
    const double t0_total = now_ms();
    for (size_t off = 0; off < n_samples; off += CHUNK_PCM_SAMPLES) {
        const size_t take =
                (n_samples - off) < CHUNK_PCM_SAMPLES ? (n_samples - off) : CHUNK_PCM_SAMPLES;
        audio_encoder_push_pcm(enc, pcm + off, take);

        const double target_ms = t0_total + (double) (off + take) / 16.0;
        const double now       = now_ms();
        const double remain_ms = target_ms - now;
        if (remain_ms > 0.0) {
            struct timespec ts = {
                    .tv_sec  = (time_t) (remain_ms / 1000.0),
                    .tv_nsec = (long) ((remain_ms - (long) (remain_ms / 1000.0) * 1000.0) * 1e6),
            };
            nanosleep(&ts, nullptr);
        }
    }
    const double t_after_capture = now_ms();

    /* End of utterance — measure residual encode tail. */
    audio_encoder_end_input(enc);
    const double t0_residual = now_ms();

    float *soft = malloc(256 * 1536 * sizeof(float));
    /* Drain until segment_done — the Phase 2 worker emits tokens
     * incrementally and pull returns as soon as ANY are available. */
    size_t n_soft = 0;
    while (!audio_encoder_segment_done(enc) && n_soft < 256) {
        size_t take = audio_encoder_pull_softtokens(enc, soft + n_soft * 1536, 256 - n_soft, -1);
        if (take == 0)
            break;
        n_soft += take;
    }
    const double t1 = now_ms();

    printf("\n--- bench_audio_streaming results ---\n");
    printf("  audio length        : %.2f s\n", audio_seconds);
    printf("  push loop (capture) : %6.1f ms  (simulated real-time rate)\n",
           t_after_capture - t0_total);
    printf("  residual after end  : %6.1f ms  (encode tail visible to user)\n", t1 - t0_residual);
    printf("  total wall          : %6.1f ms\n", t1 - t0_total);
    printf("  soft tokens out     : %zu\n", n_soft);
    printf("  residual / audio    : %.2fx  (lower is better — encode hidden in capture)\n",
           (t1 - t0_residual) / 1e3 / audio_seconds);

    free(soft);
    audio_encoder_destroy(enc);
    free(pcm);
    return GEIST_TEST_PASS;
}
