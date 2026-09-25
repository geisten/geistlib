/*
 * test_audio_attn_w8a8_parity_int — W8A8 attention (and LConv) must produce
 * soft tokens equivalent to the W8A32/FP32 reference (#238).
 *
 * Loads the audio tower per config in one process (the precision env reads
 * are not latched), encodes the same synthetic 2 s sweep, and asserts
 * per-token cosine similarity against the shipping-precision reference.
 *
 * Thresholds are calibrated from measured runs on macOS, Pi 5 and x86_64
 * (#238; x86_64 added after geistkit's model profile ran this test on Zen 5
 * for the first time). Real quantization noise lands at mean 1-cos ~1e-3
 * with one outlier token per clip; a broken quant path (wrong scale,
 * misapplied correction) blows up the MEAN. Measured on Zen 5 with
 * AVX-512/VNNI, 51 tokens:
 *
 *   config      correct (mean / worst)   +128 correction removed
 *   attn        3.4e-03 / 5.5e-02        2.8e-02 / 3.1e-02
 *   attn+lconv  6.1e-03 / 1.6e-01        2.6e-01 / 2.8e-01
 *
 * Two things follow. The mean is what discriminates — it moves by a factor
 * of 8 to 43 — while the worst token of a BROKEN path can read *better*
 * than a correct one (3.1e-02 against 5.5e-02), so COS_WORST_MIN only
 * guards against a total collapse and is set with room to spare. And the
 * outlier is not a clip-boundary artefact as first assumed: it sits at
 * token 30 of 51, with the runner-up at 9.0e-02.
 *
 * The x86 kernels are not the reason this platform drifts further than the
 * ARM ones: GEIST_AUDIO_KERNEL=scalar and GEIST_FORCE_ISA=avx2 reproduce
 * the VNNI numbers bit for bit, because both kernels accumulate exactly in
 * int32. Semantic quality is separately pinned by test_audio_chat_e2e run
 * with GEIST_AUDIO_ATTN_W8A8=1, and geist-diktat measures 0.0 % WER on this
 * precision.
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

#define SECONDS 2
#define N_PCM ((size_t) AUDIO_TEST_SR * SECONDS)

/* 0.80: the measured x86_64 outlier is 0.842, and a collapsed path lands far
 * below either number — see the table above for why the mean carries the
 * discrimination and this bound only catches a collapse. */
#define COS_WORST_MIN 0.80
#define COS_MEAN_MIN 0.99

/* Encode `pcm` with a fresh encoder under the current env. Returns token
 * count, fills `soft`. 0 on load failure. */
static size_t encode_once(const char *tower_path, const int16_t *pcm, float *soft) {
    struct AudioEncoder *enc = audio_encoder_create(tower_path);
    if (enc == nullptr) {
        return 0;
    }
    audio_encoder_push_pcm(enc, N_PCM, pcm);
    audio_encoder_end_input(enc);
    size_t total = 0;
    size_t got;
    while (total < AUDIO_TEST_MAX_SOFT &&
           (got = audio_encoder_pull_softtokens(
                    enc, AUDIO_TEST_MAX_SOFT - total, soft + total * AUDIO_TEST_SOFT_DIM, -1)) >
                   0) {
        total += got;
    }
    audio_encoder_destroy(enc);
    return total;
}

/* Encode under the current env and compare against the reference.
 * Returns the number of failures (0 or 1). */
static int check_config(const char    *label,
                        const char    *tower,
                        const int16_t *pcm,
                        const float   *soft_ref,
                        size_t         n_ref,
                        float         *soft_q) {
    size_t n_q = encode_once(tower, pcm, soft_q);
    if (n_q != n_ref) {
        fprintf(stderr, "FAIL %s: token count changed (%zu vs %zu)\n", label, n_q, n_ref);
        return 1;
    }
    double worst = 1.0, mean = 0.0;
    for (size_t t = 0; t < n_ref; t++) {
        double c = audio_test_token_cosine(soft_ref + t * AUDIO_TEST_SOFT_DIM,
                                           soft_q + t * AUDIO_TEST_SOFT_DIM);
        mean += c;
        if (c < worst)
            worst = c;
    }
    mean /= (double) n_ref;
    double max_abs =
            audio_test_max_abs_diff(soft_ref, soft_q, n_ref * AUDIO_TEST_SOFT_DIM, nullptr);
    printf("%-11s %zu tokens, 1-cos mean %.3e, worst %.3e, max|diff| %.3e\n",
           label,
           n_ref,
           1.0 - mean,
           1.0 - worst,
           max_abs);
    if (worst < COS_WORST_MIN || mean < COS_MEAN_MIN) {
        fprintf(stderr, "FAIL %s: drifts from reference\n", label);
        return 1;
    }
    return 0;
}

int main(void) {
    const char *tower = audio_test_find_tower();
    GEIST_SKIP_IF(tower == nullptr, "audio_tower.safetensors not found (make fetch-audio-tower)");

    static int16_t pcm[N_PCM];
    audio_test_synth_sweep(pcm, N_PCM, SECONDS);

    static float soft_ref[AUDIO_TEST_MAX_SOFT * AUDIO_TEST_SOFT_DIM];
    static float soft_q[AUDIO_TEST_MAX_SOFT * AUDIO_TEST_SOFT_DIM];

    /* On Apple the loader defaults every ClippableLinear to FP32 (Accelerate
     * beats the quant kernels there) — force the quant paths so this test
     * exercises W8A8 on every platform, not just the Pi. */
    setenv("GEIST_AUDIO_FORCE_QUANT", "1", 1);
    /* W8A8 is the default since the quality gates went green — the
     * high-precision reference now has to opt OUT explicitly. */
    setenv("GEIST_AUDIO_ATTN_W8A8", "0", 1);
    setenv("GEIST_AUDIO_LCONV_W8A8", "0", 1);
    size_t n_ref = encode_once(tower, pcm, soft_ref);
    GEIST_SKIP_IF(n_ref == 0, "encoder load failed (mel_constants.bin missing?)");

    int fails = 0;
    setenv("GEIST_AUDIO_ATTN_W8A8", "1", 1);
    fails += check_config("attn-W8A8:", tower, pcm, soft_ref, n_ref, soft_q);
    setenv("GEIST_AUDIO_LCONV_W8A8", "1", 1);
    fails += check_config("attn+lconv:", tower, pcm, soft_ref, n_ref, soft_q);
    /* The defaults (both unset) must equal the fully quantized config. */
    unsetenv("GEIST_AUDIO_ATTN_W8A8");
    unsetenv("GEIST_AUDIO_LCONV_W8A8");
    fails += check_config("defaults:", tower, pcm, soft_ref, n_ref, soft_q);

    if (fails == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL (%d)\n", fails);
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
