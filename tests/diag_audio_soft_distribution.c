/*
 * diag_audio_soft_distribution — diagnostic, NOT a test.
 *
 * Encodes each WAV and reports per-token soft-vector norms across
 * positions, plus a cross-clip similarity matrix at fixed positions.
 * Used to identify boundary-token leakage at the start of every
 * Conformer encoder run.
 *
 * Bench only — not exercised by make test*.
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

static const char *CLIPS[] = {
        "audio_test_data/lampe_an.wav",
        "audio_test_data/de_hello.wav",
        "audio_test_data/hello_world.wav",
        "audio_test_data/de_question.wav",
        "audio_test_data/en_question.wav",
        "audio_test_data/en_long.wav",
        nullptr,
};

/* Chunk-walking WAV reader (audio_test_util.h) — the fixed-44-byte
 * shortcut mis-read ffmpeg WAVs with a LIST chunk (#268). */
static int16_t *read_wav_pcm(const char *path, size_t *n_samples_out, int *sample_rate_out) {
    return audio_test_read_wav(path, n_samples_out, sample_rate_out);
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

static double cos_sim(const float *a, const float *b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; i++) {
        ab += (double) a[i] * (double) b[i];
        aa += (double) a[i] * (double) a[i];
        bb += (double) b[i] * (double) b[i];
    }
    return (aa > 0 && bb > 0) ? ab / (sqrt(aa) * sqrt(bb)) : 0;
}

int main(void) {
    const char *tower = find_tower();
    if (!tower)
        GEIST_SKIP("audio_tower missing");

    struct AudioEncoder *enc = audio_encoder_create(tower);
    if (!enc)
        GEIST_SKIP("encoder create failed");

    /* Encode all clips, retain soft buffers. */
    size_t n_clips = 0;
    while (CLIPS[n_clips] != nullptr)
        n_clips++;

    float **softs  = calloc(n_clips, sizeof(float *));
    size_t *n_soft = calloc(n_clips, sizeof(size_t));
    double *durs   = calloc(n_clips, sizeof(double));

    for (size_t k = 0; k < n_clips; k++) {
        size_t   n_samples;
        int      sr;
        int16_t *pcm = read_wav_pcm(CLIPS[k], &n_samples, &sr);
        if (!pcm)
            continue;
        audio_encoder_reset(enc);
        audio_encoder_push_pcm(enc, n_samples, pcm);
        audio_encoder_end_input(enc);
        free(pcm);
        durs[k] = (double) n_samples / 16000.0;

        softs[k] = calloc(MAX_SOFT * SOFT_DIM, sizeof(float));
        while (!audio_encoder_segment_done(enc) && n_soft[k] < MAX_SOFT) {
            size_t take = audio_encoder_pull_softtokens(
                    enc, MAX_SOFT - n_soft[k], softs[k] + n_soft[k] * SOFT_DIM, -1);
            if (take == 0)
                break;
            n_soft[k] += take;
        }
    }

    /* === Cross-clip similarity at fixed token positions === */
    printf("\n=== cross-clip cosine similarity per token position ===\n");
    printf("Tokens at position P are clip-specific if cos_sim < 1.\n");
    printf("Identical (cos=1.0) means boundary/zero-pad noise.\n\n");
    printf("%-3s", "pos");
    for (size_t a = 0; a < n_clips; a++) {
        const char *name   = strrchr(CLIPS[a], '/');
        name               = name ? name + 1 : CLIPS[a];
        char short_name[8] = {0};
        strncpy(short_name, name, 6);
        printf(" %7s", short_name);
    }
    printf("\n");
    for (int pos = 0; pos < 8; pos++) {
        printf("%-3d", pos);
        for (size_t a = 0; a < n_clips; a++) {
            if (n_soft[a] <= (size_t) pos) {
                printf(" %7s", "-");
                continue;
            }
            /* Average sim to other clips at this position. */
            double sum = 0;
            int    cnt = 0;
            for (size_t b = 0; b < n_clips; b++) {
                if (a == b || n_soft[b] <= (size_t) pos)
                    continue;
                sum += cos_sim(softs[a] + pos * SOFT_DIM, softs[b] + pos * SOFT_DIM, SOFT_DIM);
                cnt++;
            }
            printf(" %7.4f", cnt ? sum / cnt : 0);
        }
        printf("\n");
    }

    printf("\n=== boundary-token threshold ===\n");
    printf("First position where avg cross-clip cos_sim drops below 0.99:\n");
    for (size_t a = 0; a < n_clips; a++) {
        const char *name = strrchr(CLIPS[a], '/');
        name             = name ? name + 1 : CLIPS[a];
        int first_unique = -1;
        for (size_t pos = 0; pos < n_soft[a]; pos++) {
            double sum = 0;
            int    cnt = 0;
            for (size_t b = 0; b < n_clips; b++) {
                if (a == b || n_soft[b] <= pos)
                    continue;
                sum += cos_sim(softs[a] + pos * SOFT_DIM, softs[b] + pos * SOFT_DIM, SOFT_DIM);
                cnt++;
            }
            if (cnt == 0)
                continue;
            if (sum / cnt < 0.99) {
                first_unique = (int) pos;
                break;
            }
        }
        printf("  %-24s dur=%.2fs n_soft=%zu  first_unique_pos=%d  signal_ratio=%.0f%%\n",
               name,
               durs[a],
               n_soft[a],
               first_unique,
               first_unique >= 0 ? 100.0 * (n_soft[a] - first_unique) / n_soft[a] : 0.0);
    }

    /* Per-token norm progression for shortest clip — show boundary effect at tail too. */
    printf("\n=== per-token L2 norms for lampe_an (shortest) ===\n");
    for (size_t t = 0; t < n_soft[0]; t++) {
        double s = 0;
        for (size_t d = 0; d < SOFT_DIM; d++) {
            float v = softs[0][t * SOFT_DIM + d];
            s += (double) v * v;
        }
        printf("  t=%2zu  ||x||=%.2f\n", t, sqrt(s));
    }

    for (size_t k = 0; k < n_clips; k++)
        free(softs[k]);
    free(softs);
    free(n_soft);
    free(durs);
    audio_encoder_destroy(enc);
    return GEIST_TEST_PASS;
}
