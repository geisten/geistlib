/*
 * test_audio_stream_parity_unit — verify Phase 1b chunk-streaming
 * encoder produces the same soft tokens as the monolithic
 * audio_encoder_run on the same mel input.
 *
 * SKIP if audio_tower.safetensors is missing. Otherwise:
 *   1. Build a deterministic synthetic mel buffer (sine-mixed, 100 frames).
 *   2. Run audio_encoder_run(mel, n_mel) → ref_soft (T_sub × 1536).
 *   3. Reset state, feed mel chunked via audio_encoder_stream_push:
 *      - first push: half the mel, is_final=false
 *      - second push: remainder, is_final=true
 *   4. Compare ref_soft to state->soft elementwise.
 *
 * Tolerance 5e-4 max|Δ| — the streaming and monolithic paths execute
 * the same math but accumulate in different orders inside chunked
 * attention (K/V cache read vs. recompute), so a few ULP of fp32
 * non-associativity is expected. The K/V values themselves are
 * identical, so attention output should be bit-stable up to summation
 * order, which fp32 dot products handle within 1-2 ULP.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/audio_conformer/audio_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MEL_N_MEL 128
#define SOFT_DIM 1536
#define N_MEL_FRAMES 100 /* ~1 s of audio at 100 Hz frame rate */
#define CHUNK_SIZE 12    /* Conformer attention chunk (sub-tokens) */
#define SUB_DOWNSAMPLE 4 /* mel→sub-token stride (two stride-2 convs) */

static const char *find_audio_tower(void) {
    static const char *candidates[] = {
            "audio_bench/audio_tower.safetensors",
            "../audio_bench/audio_tower.safetensors",
            "../gemma-4-E2B-it/audio_tower.safetensors",
            nullptr,
    };
    for (size_t i = 0; candidates[i] != nullptr; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f != nullptr) {
            fclose(f);
            return candidates[i];
        }
    }
    return nullptr;
}

static void synth_mel(float *mel, bool *mask, size_t n_frames) {
    /* Deterministic broadband-ish signal: per-frame, per-bin sine mixture.
     * Keeps values bounded so the encoder doesn't hit clamp saturation. */
    for (size_t t = 0; t < n_frames; t++) {
        for (size_t b = 0; b < MEL_N_MEL; b++) {
            float phase            = (float) t * 0.07f + (float) b * 0.013f;
            float v                = 0.5f * sinf(phase) + 0.3f * sinf(2.3f * phase) - 1.0f;
            mel[t * MEL_N_MEL + b] = v;
        }
        mask[t] = true;
    }
}

int main(void) {
    const char *audio_tower = find_audio_tower();
    if (audio_tower == nullptr) {
        printf("SKIP: audio_tower.safetensors not found "
               "(audio_bench/, ../audio_bench/, ../gemma-4-E2B-it/)\n");
        return GEIST_TEST_SKIP;
    }

    struct AudioEncoder *enc = audio_encoder_create(audio_tower);
    if (enc == nullptr) {
        fprintf(stderr, "audio_encoder_create failed\n");
        return GEIST_TEST_FAIL;
    }

    float *mel  = calloc((size_t) N_MEL_FRAMES * MEL_N_MEL, sizeof(float));
    bool  *mask = calloc((size_t) N_MEL_FRAMES, sizeof(bool));
    if (mel == nullptr || mask == nullptr) {
        audio_encoder_destroy(enc);
        free(mel);
        free(mask);
        return GEIST_TEST_ERROR;
    }
    synth_mel(mel, mask, N_MEL_FRAMES);

    /* 1. Reference: monolithic audio_encoder_run. */
    const size_t max_sub  = N_MEL_FRAMES / SUB_DOWNSAMPLE + 8;
    float       *ref_soft = calloc(max_sub * SOFT_DIM, sizeof(float));
    if (ref_soft == nullptr) {
        audio_encoder_destroy(enc);
        free(mel);
        free(mask);
        return GEIST_TEST_ERROR;
    }
    const size_t n_soft_ref = audio_encoder_run(enc, N_MEL_FRAMES, mel, mask, ref_soft);
    printf("audio_stream_parity: monolithic produced %zu soft tokens\n", n_soft_ref);
    if (n_soft_ref == 0) {
        fprintf(stderr, "FAIL: monolithic encode returned 0 soft tokens\n");
        free(ref_soft);
        free(mel);
        free(mask);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }

    /* 2. Reset streaming state (audio_encoder_reset clears stream state). */
    audio_encoder_reset(enc);
    struct audio_stream_state *state = audio_encoder_stream_state(enc);
    if (state == nullptr) {
        fprintf(stderr, "FAIL: stream state is null\n");
        free(ref_soft);
        free(mel);
        free(mask);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }

    /* 3. Drive streaming in two halves to exercise cross-push state carry
     *    (K/V cache + LConv hist must survive between push calls). */
    const size_t n_half = N_MEL_FRAMES / 2;

    const size_t emit_a = audio_encoder_stream_push(enc, state, n_half, mel, mask, false);
    printf("audio_stream_parity: push#1 (%zu mel) emitted %zu soft tokens\n", n_half, emit_a);

    const size_t emit_b = audio_encoder_stream_push(enc, state, N_MEL_FRAMES, mel, mask, true);
    printf("audio_stream_parity: push#2 (%zu mel, final) emitted %zu soft\n",
           (size_t) N_MEL_FRAMES,
           emit_b);

    const size_t n_soft_stream = audio_stream_state_n_soft(state);
    const float *stream_soft   = audio_stream_state_soft(state);
    printf("audio_stream_parity: streaming produced %zu soft tokens\n", n_soft_stream);

    if (n_soft_stream != n_soft_ref) {
        fprintf(stderr,
                "FAIL: token count mismatch ref=%zu stream=%zu\n",
                n_soft_ref,
                n_soft_stream);
        free(ref_soft);
        free(mel);
        free(mask);
        audio_encoder_destroy(enc);
        return GEIST_TEST_FAIL;
    }

    /* 4. Element-wise compare. Tolerance 5e-4 — chunked attention with
     *    K/V cache reads vs. monolithic recompute can differ by a few
     *    ULP due to fp32 non-associativity in dot product summation. */
    float  max_abs = 0.0f;
    size_t max_at  = 0;
    for (size_t i = 0; i < n_soft_ref * SOFT_DIM; i++) {
        float d = fabsf(ref_soft[i] - stream_soft[i]);
        if (d > max_abs) {
            max_abs = d;
            max_at  = i;
        }
    }
    printf("audio_stream_parity: max|Δ| = %.6f at flat index %zu "
           "(token %zu, dim %zu)\n",
           (double) max_abs,
           max_at,
           max_at / SOFT_DIM,
           max_at % SOFT_DIM);

    free(ref_soft);
    free(mel);
    free(mask);
    audio_encoder_destroy(enc);

    const float tolerance = 5e-4f;
    if (max_abs > tolerance) {
        fprintf(stderr,
                "FAIL: max|Δ| %.6f exceeds tolerance %.6f\n",
                (double) max_abs,
                (double) tolerance);
        return GEIST_TEST_FAIL;
    }

    printf("PASS\n");
    return GEIST_TEST_PASS;
}
