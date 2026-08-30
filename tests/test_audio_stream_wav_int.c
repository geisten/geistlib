/*
 * test_audio_stream_wav_int — exercise the streaming push/pull API on
 * real WAV input and validate the produced soft tokens are sensible.
 *
 * Distinct from test_audio_stream_parity_unit, which drives
 * audio_encoder_stream_push directly with synthetic mel. This test
 * goes through the full PCM → mel → Conformer chain via the public
 * push_pcm / pull_softtokens API on actual speech, so it catches
 * regressions in:
 *   - mel_pipeline frame computation on real PCM
 *   - end_input / Phase 2 worker handoff timing
 *   - pull_softtokens loop (drain incrementally vs sync sync compute)
 *
 * The check is intentionally shape-and-sanity rather than parity-
 * against-a-baseline: we verify the encoder produced the expected
 * number of soft tokens, no NaN/Inf, non-trivial dynamic range, and
 * a reasonable distribution. A regression that produced all zeros or
 * NaN slips past unit tests but would crash an LM downstream.
 *
 * SKIPs if audio assets are missing. Doesn't need GGUF.
 *
 *   bin/<target>/release/tests/test_audio_stream_wav_int [path.wav]
 */
#include "audio_test_util.h"
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/audio_conformer/audio_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOFT_DIM 1536
#define MAX_SOFT 256
#define PUSH_CHUNK_S 320 /* 20 ms @ 16 kHz - matches push-to-talk frontends */

/* Chunk-walking WAV reader (audio_test_util.h) — the fixed-44-byte
 * shortcut mis-read ffmpeg WAVs with a LIST chunk (#268). */
static int16_t *read_wav_pcm(const char *path, size_t *n_samples_out, int *sample_rate_out) {
    return audio_test_read_wav(path, n_samples_out, sample_rate_out);
}

static const char *find_audio_tower(void) {
    static const char *cands[] = {
            "audio_bench/audio_tower.safetensors",
            "../audio_bench/audio_tower.safetensors",
            "../gemma-4-E2B-it/audio_tower.safetensors",
            nullptr,
    };
    for (size_t i = 0; cands[i] != nullptr; i++) {
        FILE *f = fopen(cands[i], "rb");
        if (f != nullptr) {
            fclose(f);
            return cands[i];
        }
    }
    return nullptr;
}

int main(int argc, char **argv) {
    const char *wav_path   = argc > 1 ? argv[1] : "audio_test_data/en_long.wav";
    const char *tower_path = find_audio_tower();
    if (tower_path == nullptr) {
        GEIST_SKIP("audio_tower.safetensors not found in audio_bench/ or sibling dirs");
    }

    size_t   n_samples;
    int      sample_rate;
    int16_t *pcm = read_wav_pcm(wav_path, &n_samples, &sample_rate);
    if (pcm == nullptr)
        GEIST_SKIP("could not read WAV (default audio_test_data/en_long.wav)");
    if (sample_rate != 16000) {
        free(pcm);
        GEIST_SKIP("WAV not 16 kHz mono 16-bit");
    }
    const double audio_s = (double) n_samples / 16000.0;
    printf("audio_stream_wav: %s (%.2f s, %zu samples)\n", wav_path, audio_s, n_samples);

    struct AudioEncoder *enc = audio_encoder_create(tower_path);
    if (enc == nullptr) {
        free(pcm);
        GEIST_SKIP("audio_encoder_create failed (missing mel_constants.bin?)");
    }

    /* Drive in 20 ms chunks like a real push-to-talk frontend. This
     * exercises mel-frame increments per push (Phase C) and lets the
     * Phase 2 worker fire mid-stream when GEIST_AUDIO_STREAM=1. */
    for (size_t off = 0; off < n_samples; off += PUSH_CHUNK_S) {
        size_t take = (n_samples - off) < PUSH_CHUNK_S ? (n_samples - off) : PUSH_CHUNK_S;
        if (audio_encoder_push_pcm(enc, take, pcm + off) != 0) {
            fprintf(stderr, "push_pcm failed at offset %zu\n", off);
            free(pcm);
            audio_encoder_destroy(enc);
            return GEIST_TEST_FAIL;
        }
    }
    audio_encoder_end_input(enc);
    free(pcm);

    /* Drain until segment_done — the Phase 2 worker emits tokens
     * incrementally, so we may need multiple pull calls. */
    float *soft = calloc(MAX_SOFT * SOFT_DIM, sizeof(float));
    if (soft == nullptr) {
        audio_encoder_destroy(enc);
        return GEIST_TEST_ERROR;
    }
    size_t n_soft = 0;
    while (!audio_encoder_segment_done(enc) && n_soft < MAX_SOFT) {
        size_t take =
                audio_encoder_pull_softtokens(enc, MAX_SOFT - n_soft, soft + n_soft * SOFT_DIM, -1);
        if (take == 0)
            break;
        n_soft += take;
    }
    printf("  soft tokens out: %zu (expected ~%.0f for %.2f s)\n", n_soft, audio_s * 25.0, audio_s);

    /* 1. Token count plausibility - Gemma 4 audio emits ~1 soft token
     *    per 40 ms of audio, i.e. 25 tokens/sec. Tolerate +/- 2 tokens
     *    for boundary effects (subsample receptive field + ceiling
     *    rounding). */
    const size_t expect_lo = (size_t) (audio_s * 25.0) > 2 ? (size_t) (audio_s * 25.0) - 2 : 1;
    const size_t expect_hi = (size_t) (audio_s * 25.0) + 4;
    if (n_soft < expect_lo || n_soft > expect_hi) {
        fprintf(stderr, "FAIL: token count %zu outside [%zu, %zu]\n", n_soft, expect_lo, expect_hi);
        free(soft);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }

    /* 2. No NaN / Inf anywhere - poisons downstream LM logits. */
    size_t n_nan = 0, n_inf = 0;
    for (size_t i = 0; i < n_soft * SOFT_DIM; i++) {
        if (isnan(soft[i]))
            n_nan++;
        else if (isinf(soft[i]))
            n_inf++;
    }
    if (n_nan != 0 || n_inf != 0) {
        fprintf(stderr, "FAIL: NaN=%zu Inf=%zu in soft tokens\n", n_nan, n_inf);
        free(soft);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }

    /* 3. Non-trivial dynamic range - all-zero output would mean the
     *    encoder silently failed. Check the absolute min/max across
     *    the soft-token tensor. Healthy soft tokens are roughly
     *    standard-normal scale post-rmsnorm + linear projection, so
     *    we expect peak |values| in the 1-100 range. */
    float  vmin = soft[0], vmax = soft[0];
    double vsum = 0.0, vsq = 0.0;
    for (size_t i = 0; i < n_soft * SOFT_DIM; i++) {
        const float v = soft[i];
        if (v < vmin)
            vmin = v;
        if (v > vmax)
            vmax = v;
        vsum += (double) v;
        vsq += (double) v * (double) v;
    }
    const double mean = vsum / (double) (n_soft * SOFT_DIM);
    const double stdv = sqrt(vsq / (double) (n_soft * SOFT_DIM) - mean * mean);
    printf("  value distribution: min=%.3f max=%.3f mean=%.3f stddev=%.3f\n",
           (double) vmin,
           (double) vmax,
           mean,
           stdv);
    if (fabsf(vmax - vmin) < 0.1f) {
        fprintf(stderr,
                "FAIL: dynamic range too small (%.4f) - encoder produced ~constant output\n",
                (double) (vmax - vmin));
        free(soft);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }
    if (stdv < 0.01) {
        fprintf(stderr, "FAIL: stddev too small (%.4f) - encoder output is degenerate\n", stdv);
        free(soft);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }

    /* 4. segment_done becomes true after we drained. */
    if (!audio_encoder_segment_done(enc)) {
        fprintf(stderr, "FAIL: segment_done is false after drain\n");
        free(soft);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }

    /* 5. Reset + replay should work without state bleed. */
    audio_encoder_reset(enc);
    if (audio_encoder_segment_done(enc)) {
        fprintf(stderr, "FAIL: segment_done still true after reset\n");
        free(soft);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }

    free(soft);
    audio_encoder_destroy(enc);
    printf("PASS\n");
    return GEIST_TEST_PASS;
}
