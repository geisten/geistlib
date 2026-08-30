/*
 * test_audio_stream_live_parity_int — the live streaming worker
 * (GEIST_AUDIO_STREAM=1, Phase 8b) must emit the same soft tokens as the
 * monolithic path for the same audio (#235).
 *
 * Two encoders in one process (the env is read per create): the reference
 * encodes after end_input in one go; the live one gets the same synthetic
 * sweep pushed in 20 ms chunks with the worker encoding mid-stream.
 * Asserts identical token count (the padded-frame fix) and value parity.
 *
 * SKIPs cleanly when audio_tower.safetensors / mel_constants.bin are
 * missing; the audio-smoke CI job runs it with fixtures mandatory.
 */
#define _POSIX_C_SOURCE 200809L /* setenv */

#include "audio_test_util.h"
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/audio_conformer/audio_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SECONDS 4 /* > one 48-frame worker kick, < the 256-token cap */
#define N_PCM ((size_t) AUDIO_TEST_SR * SECONDS)
#define PUSH_CHUNK 320 /* 20 ms — real push-to-talk granularity */

/* Push in 20 ms chunks, end input, drain. Returns token count. */
static size_t encode(const char *tower, const int16_t *pcm, float *soft, bool live) {
    if (live) {
        setenv("GEIST_AUDIO_STREAM", "1", 1);
    } else {
        unsetenv("GEIST_AUDIO_STREAM");
    }
    struct AudioEncoder *enc = audio_encoder_create(tower);
    if (enc == nullptr) {
        return 0;
    }
    for (size_t off = 0; off < N_PCM; off += PUSH_CHUNK) {
        size_t take = N_PCM - off < PUSH_CHUNK ? N_PCM - off : PUSH_CHUNK;
        if (audio_encoder_push_pcm(enc, take, pcm + off) != 0) {
            audio_encoder_destroy(enc);
            return 0;
        }
    }
    audio_encoder_end_input(enc);
    size_t total = 0;
    while (total < AUDIO_TEST_MAX_SOFT && !audio_encoder_segment_done(enc)) {
        size_t got = audio_encoder_pull_softtokens(
                enc, AUDIO_TEST_MAX_SOFT - total, soft + total * AUDIO_TEST_SOFT_DIM, -1);
        if (got == 0 && audio_encoder_segment_done(enc)) {
            break;
        }
        total += got;
    }
    audio_encoder_destroy(enc);
    return total;
}

int main(void) {
    const char *tower = audio_test_find_tower();
    GEIST_SKIP_IF(tower == nullptr, "audio_tower.safetensors not found (make fetch-audio-tower)");

    static int16_t pcm[N_PCM];
    audio_test_synth_sweep(pcm, N_PCM, SECONDS);

    static float soft_ref[AUDIO_TEST_MAX_SOFT * AUDIO_TEST_SOFT_DIM];
    static float soft_live[AUDIO_TEST_MAX_SOFT * AUDIO_TEST_SOFT_DIM];

    size_t n_ref = encode(tower, pcm, soft_ref, false);
    GEIST_SKIP_IF(n_ref == 0, "encoder load failed (mel_constants.bin missing?)");
    size_t n_live = encode(tower, pcm, soft_live, true);

    printf("monolithic: %zu tokens, live worker: %zu tokens\n", n_ref, n_live);
    if (n_live != n_ref) {
        fprintf(stderr, "FAIL: live worker token count differs (%zu vs %zu)\n", n_live, n_ref);
        return GEIST_TEST_FAIL;
    }

    size_t at      = 0;
    double max_abs = audio_test_max_abs_diff(soft_ref, soft_live, n_ref * AUDIO_TEST_SOFT_DIM, &at);
    printf("max|Δ| = %.3e at token %zu dim %zu\n",
           max_abs,
           at / AUDIO_TEST_SOFT_DIM,
           at % AUDIO_TEST_SOFT_DIM);
    /* The stream parity unit test measures ~1e-5 for chunk-scheduling
     * differences; 1e-3 leaves two orders of headroom while a real
     * divergence (missing frames, stale state) is O(1). */
    if (max_abs > 1e-3) {
        fprintf(stderr, "FAIL: live worker diverges from monolithic path\n");
        return GEIST_TEST_FAIL;
    }
    printf("PASS\n");
    return GEIST_TEST_PASS;
}
