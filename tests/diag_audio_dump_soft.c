/*
 * diag_audio_dump_soft — diagnostic, NOT a test.
 *
 * Encodes a single WAV and dumps the soft-token tensor to a .bin
 * file. The companion Python script tools/compare_soft_tokens.py
 * reads two such dumps (one per encoder config) and computes the
 * per-token cosine similarity + L2 drift, isolating where W8A8
 * precision diverges from the FP32 baseline.
 *
 * Usage:
 *   diag_audio_dump_soft <input.wav> <output.bin>
 *
 * Output format (little-endian):
 *   uint32 n_tokens
 *   uint32 dim                  (=1536)
 *   float32[n_tokens * dim]     row-major soft tokens
 */
#include "audio_test_util.h"
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/audio_conformer/audio_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOFT_DIM 1536
#define MAX_SOFT 1024

/* Chunk-walking WAV reader (audio_test_util.h) — the fixed-44-byte
 * shortcut mis-read ffmpeg WAVs with a LIST chunk (#268). Rejects
 * non-16kHz input like the local reader it replaces. */
static int16_t *read_wav_pcm(const char *path, size_t *n_samples_out) {
    int      sr  = 0;
    int16_t *pcm = audio_test_read_wav(path, n_samples_out, &sr);
    if (pcm != nullptr && sr != 16000) {
        free(pcm);
        return nullptr;
    }
    return pcm;
}

static const char *find_tower(void) {
    static const char *c[] = {"audio_bench/audio_tower.safetensors", nullptr};
    FILE              *f   = fopen(c[0], "rb");
    if (f) {
        fclose(f);
        return c[0];
    }
    return nullptr;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: %s <input.wav> <output.bin>\n", argv[0]);
        return GEIST_TEST_SKIP;
    }
    const char *wav = argv[1];
    const char *out = argv[2];

    const char *tower = find_tower();
    if (!tower)
        GEIST_SKIP("audio_tower missing");

    struct AudioEncoder *enc = audio_encoder_create(tower);
    if (!enc)
        GEIST_SKIP("encoder create failed");

    size_t   n_samples;
    int16_t *pcm = read_wav_pcm(wav, &n_samples);
    if (!pcm) {
        audio_encoder_destroy(enc);
        GEIST_SKIP("wav read failed");
    }
    fprintf(stderr,
            "diag_audio_dump_soft: %s (%zu samples = %.2f s)\n",
            wav,
            n_samples,
            (double) n_samples / 16000.0);

    if (audio_encoder_push_pcm(enc, n_samples, pcm) != 0) {
        free(pcm);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }
    audio_encoder_end_input(enc);
    free(pcm);

    float *soft   = calloc(MAX_SOFT * SOFT_DIM, sizeof(float));
    size_t n_soft = 0;
    while (!audio_encoder_segment_done(enc) && n_soft < MAX_SOFT) {
        size_t take =
                audio_encoder_pull_softtokens(enc, MAX_SOFT - n_soft, soft + n_soft * SOFT_DIM, -1);
        if (!take)
            break;
        n_soft += take;
    }
    fprintf(stderr, "  encoded %zu soft tokens × %d dims\n", n_soft, SOFT_DIM);

    FILE *fo = fopen(out, "wb");
    if (!fo) {
        free(soft);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }
    uint32_t hdr[2] = {(uint32_t) n_soft, (uint32_t) SOFT_DIM};
    xfwrite(hdr, sizeof(uint32_t), 2, fo);
    xfwrite(soft, sizeof(float), n_soft * SOFT_DIM, fo);
    fclose(fo);
    fprintf(stderr, "  wrote %s\n", out);

    free(soft);
    audio_encoder_destroy(enc);
    return GEIST_TEST_PASS;
}
