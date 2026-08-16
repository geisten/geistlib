/*
 * audio_test_util.h — shared scaffolding for the audio-tower tests.
 *
 * One home for the pieces the audio parity tests kept copy-pasting:
 * the synthetic sweep clip (kept in sync with tools/gen_test_wav.py —
 * this is the ONLY C statement of that formula), tower discovery, and
 * the soft-token comparison helpers.
 */
#ifndef GEIST_AUDIO_TEST_UTIL_H
#define GEIST_AUDIO_TEST_UTIL_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define AUDIO_TEST_SR 16000
#define AUDIO_TEST_SOFT_DIM 1536
#define AUDIO_TEST_MAX_SOFT 256

/* M_PI is not ISO C23; pin it like attn_run pins M_E. */
#define AUDIO_TEST_PI 3.14159265358979323846

/* 440->880 Hz sweep with a sine envelope — same formula as
 * tools/gen_test_wav.py, so CI-file and in-memory fixtures match. */
static inline void audio_test_synth_sweep(int16_t *pcm, size_t n, double seconds) {
    for (size_t i = 0; i < n; i++) {
        double t    = (double) i / AUDIO_TEST_SR;
        double freq = 440.0 * pow(2.0, t / seconds);
        double env  = sin(AUDIO_TEST_PI * t / seconds);
        pcm[i]      = (int16_t) (0.5 * env * sin(2.0 * AUDIO_TEST_PI * freq * t) * 32767.0);
    }
}

/* Tower discovery: env override, then the arch's default search location.
 * Returns nullptr when the file is absent — caller decides skip vs fail. */
static inline const char *audio_test_find_tower(void) {
    const char *tower = getenv("GEIST_AUDIO_MODEL_PATH");
    if (tower == nullptr) {
        tower = "audio_bench/audio_tower.safetensors";
    }
    FILE *f = fopen(tower, "rb");
    if (f == nullptr) {
        return nullptr;
    }
    fclose(f);
    return tower;
}

/* Max |a[i] - b[i]| over n elements; writes the argmax to *at (may be
 * nullptr). */
static inline double audio_test_max_abs_diff(const float *a, const float *b, size_t n, size_t *at) {
    double worst = 0.0;
    size_t idx   = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double) a[i] - (double) b[i]);
        if (d > worst) {
            worst = d;
            idx   = i;
        }
    }
    if (at != nullptr) {
        *at = idx;
    }
    return worst;
}

/* Cosine similarity of two AUDIO_TEST_SOFT_DIM-dim vectors. */
static inline double audio_test_token_cosine(const float *a, const float *b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t d = 0; d < AUDIO_TEST_SOFT_DIM; d++) {
        dot += (double) a[d] * (double) b[d];
        na += (double) a[d] * (double) a[d];
        nb += (double) b[d] * (double) b[d];
    }
    double denom = sqrt(na) * sqrt(nb);
    return denom > 0.0 ? dot / denom : 0.0;
}

#endif /* GEIST_AUDIO_TEST_UTIL_H */
