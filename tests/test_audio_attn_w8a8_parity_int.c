/*
 * test_audio_attn_w8a8_parity_int — W8A8 attention (and LConv) must produce
 * soft tokens equivalent to the W8A32/FP32 reference (#238).
 *
 * Loads the audio tower twice in one process — baseline precision, then
 * GEIST_AUDIO_ATTN_W8A8=1 (+GEIST_AUDIO_LCONV_W8A8=1) — encodes the same
 * synthetic 2 s sweep (same formula as tools/gen_test_wav.py, generated
 * in-memory: no fixture WAV), and asserts per-token cosine similarity of
 * the soft-token sequences.
 *
 * Thresholds are calibrated from measured runs (macOS + Pi 5, see #238):
 * real quantization noise lands at mean 1-cos ~3.4e-3 with worst tokens
 * ~0.93 cosine (start-of-clip boundary tokens carry the most drift); a
 * broken quant path (wrong scale, misapplied correction) collapses toward
 * 0. Semantic quality is separately pinned by test_audio_chat_e2e run
 * with GEIST_AUDIO_ATTN_W8A8=1.
 *
 * SKIPs cleanly when audio_tower.safetensors / mel_constants.bin are
 * missing; the audio-smoke CI job runs it with fixtures mandatory.
 */
#define _POSIX_C_SOURCE 200809L /* setenv */

#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/audio_conformer/audio_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SR 16000
#define SECONDS 2
#define N_PCM ((size_t) SR * SECONDS)
#define SOFT_DIM 1536
#define MAX_SOFT 256

/* M_PI is not ISO C23; pin it like attn_run pins M_E. */
#define PI 3.14159265358979323846

/* Same 440->880 Hz sweep with sine envelope as tools/gen_test_wav.py. */
static void synth_pcm(int16_t *pcm, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double t    = (double) i / SR;
        double freq = 440.0 * pow(2.0, t / (double) SECONDS);
        double env  = sin(PI * t / (double) SECONDS);
        pcm[i]      = (int16_t) (0.5 * env * sin(2.0 * PI * freq * t) * 32767.0);
    }
}

/* Encode `pcm` with a fresh encoder under the current env. Returns token
 * count, fills `soft` (MAX_SOFT × SOFT_DIM). 0 on load failure. */
static size_t encode_once(const char *tower_path, const int16_t *pcm, float *soft) {
    struct AudioEncoder *enc = audio_encoder_create(tower_path);
    if (enc == nullptr) {
        return 0;
    }
    audio_encoder_push_pcm(enc, pcm, N_PCM);
    audio_encoder_end_input(enc);
    size_t total = 0;
    size_t got;
    while (total < MAX_SOFT && (got = audio_encoder_pull_softtokens(
                                        enc, soft + total * SOFT_DIM, MAX_SOFT - total, -1)) > 0) {
        total += got;
    }
    audio_encoder_destroy(enc);
    return total;
}

static double token_cosine(const float *a, const float *b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t d = 0; d < SOFT_DIM; d++) {
        dot += (double) a[d] * (double) b[d];
        na += (double) a[d] * (double) a[d];
        nb += (double) b[d] * (double) b[d];
    }
    double denom = sqrt(na) * sqrt(nb);
    return denom > 0.0 ? dot / denom : 0.0;
}

int main(void) {
    /* Same discovery the arch uses: default location, env override. */
    const char *tower = getenv("GEIST_AUDIO_MODEL_PATH");
    if (tower == nullptr) {
        tower = "audio_bench/audio_tower.safetensors";
    }
    FILE *f = fopen(tower, "rb");
    GEIST_SKIP_IF(f == nullptr, "audio_tower.safetensors not found (make fetch-audio-tower)");
    fclose(f);

    static int16_t pcm[N_PCM];
    synth_pcm(pcm, N_PCM);

    static float soft_ref[MAX_SOFT * SOFT_DIM];
    static float soft_q[MAX_SOFT * SOFT_DIM];

    /* On Apple the loader defaults every ClippableLinear to FP32 (Accelerate
     * beats the quant kernels there) — force the quant paths so this test
     * exercises W8A8 on every platform, not just the Pi. */
    setenv("GEIST_AUDIO_FORCE_QUANT", "1", 1);
    unsetenv("GEIST_AUDIO_ATTN_W8A8");
    unsetenv("GEIST_AUDIO_LCONV_W8A8");
    size_t n_ref = encode_once(tower, pcm, soft_ref);
    GEIST_SKIP_IF(n_ref == 0, "encoder load failed (mel_constants.bin missing?)");

    int fails = 0;

    /* --- attention W8A8 vs reference -------------------------------------- */
    setenv("GEIST_AUDIO_ATTN_W8A8", "1", 1);
    size_t n_q = encode_once(tower, pcm, soft_q);
    if (n_q != n_ref) {
        fprintf(stderr, "FAIL: token count changed (%zu vs %zu)\n", n_q, n_ref);
        return GEIST_TEST_FAIL;
    }
    double worst = 1.0, mean = 0.0, max_abs = 0.0;
    for (size_t t = 0; t < n_ref; t++) {
        double c = token_cosine(soft_ref + t * SOFT_DIM, soft_q + t * SOFT_DIM);
        mean += c;
        if (c < worst)
            worst = c;
    }
    for (size_t i = 0; i < n_ref * SOFT_DIM; i++) {
        double d = fabs((double) soft_ref[i] - (double) soft_q[i]);
        if (d > max_abs)
            max_abs = d;
    }
    mean /= (double) n_ref;
    printf("attn-W8A8:  %zu tokens, 1-cos mean %.3e, worst %.3e, max|diff| %.3e\n",
           n_ref,
           1.0 - mean,
           1.0 - worst,
           max_abs);
    if (worst < 0.85 || mean < 0.99) {
        fprintf(stderr, "FAIL: attn-W8A8 drifts from reference\n");
        fails++;
    }

    /* --- attention + lconv W8A8 vs reference ------------------------------- */
    setenv("GEIST_AUDIO_LCONV_W8A8", "1", 1);
    n_q = encode_once(tower, pcm, soft_q);
    if (n_q != n_ref) {
        fprintf(stderr, "FAIL: token count changed with lconv (%zu vs %zu)\n", n_q, n_ref);
        return GEIST_TEST_FAIL;
    }
    worst = 1.0;
    mean  = 0.0;
    for (size_t t = 0; t < n_ref; t++) {
        double c = token_cosine(soft_ref + t * SOFT_DIM, soft_q + t * SOFT_DIM);
        mean += c;
        if (c < worst)
            worst = c;
    }
    mean /= (double) n_ref;
    printf("attn+lconv: %zu tokens, 1-cos mean %.3e, worst %.3e\n", n_ref, 1.0 - mean, 1.0 - worst);
    if (worst < 0.85 || mean < 0.99) {
        fprintf(stderr, "FAIL: attn+lconv W8A8 drifts from reference\n");
        fails++;
    }

    unsetenv("GEIST_AUDIO_ATTN_W8A8");
    unsetenv("GEIST_AUDIO_LCONV_W8A8");

    if (fails == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL (%d)\n", fails);
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
