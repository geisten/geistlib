/*
 * test_audio_long_clip_int — audio past ~10 s is no longer truncated (#247).
 *
 * A hardcoded max_soft = 256 (≈ 10.2 s at ~25 soft tokens/s) used to cap
 * the pipeline: a longer clip paid the full encode and silently lost
 * everything past the cap. The bound now derives from the audio length
 * (audio_encoder_max_soft_tokens / the encoder vtable's max_soft_tokens).
 *
 * Encodes a synthetic 12 s sweep and asserts the token count lands where
 * the duration says it must — above the old cap, and within ±15 % of the
 * ~25 tokens/s rate. Also pins the bound helper itself against the count.
 *
 * SKIPs cleanly when audio_tower.safetensors / mel_constants.bin are
 * missing; the audio-smoke CI job runs it with fixtures mandatory.
 */
#include "audio_test_util.h"
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/audio_conformer/audio_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SECONDS 12 /* > the old 256-token (~10.2 s) cap */
#define N_PCM ((size_t) AUDIO_TEST_SR * SECONDS)

int main(void) {
    const char *tower = audio_test_find_tower();
    GEIST_SKIP_IF(tower == nullptr, "audio_tower.safetensors not found (make fetch-audio-tower)");

    static int16_t pcm[N_PCM];
    audio_test_synth_sweep(pcm, N_PCM, SECONDS);

    const size_t bound = audio_encoder_max_soft_tokens(N_PCM);
    float       *soft  = calloc(bound * AUDIO_TEST_SOFT_DIM, sizeof(float));
    if (soft == nullptr) {
        return GEIST_TEST_ERROR;
    }

    struct AudioEncoder *enc = audio_encoder_create(tower);
    GEIST_SKIP_IF(enc == nullptr, "encoder load failed (mel_constants.bin missing?)");
    audio_encoder_push_pcm(enc, N_PCM, pcm);
    audio_encoder_end_input(enc);
    size_t total = 0;
    size_t got;
    while (total < bound &&
           (got = audio_encoder_pull_softtokens(
                    enc, bound - total, soft + total * AUDIO_TEST_SOFT_DIM, -1)) > 0) {
        total += got;
    }
    audio_encoder_destroy(enc);
    free(soft);

    const size_t expect = (size_t) (SECONDS * 25); /* ~25 soft tokens per second */
    printf("12 s clip: %zu soft tokens (bound %zu, expected ~%zu)\n", total, bound, expect);

    int fails = 0;
    if (total <= 256) {
        fprintf(stderr, "FAIL: still capped at the old 256-token limit\n");
        fails++;
    }
    if (total < expect * 85 / 100 || total > expect * 115 / 100) {
        fprintf(stderr, "FAIL: token count off the ~25/s rate (%zu vs ~%zu)\n", total, expect);
        fails++;
    }
    if (total > bound) {
        fprintf(stderr, "FAIL: bound helper under-estimates (%zu > %zu)\n", total, bound);
        fails++;
    }

    if (fails == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL (%d)\n", fails);
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
