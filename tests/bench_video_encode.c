/*
 * bench_video_encode — standalone video-tower bench (32 frame default).
 *
 * Synthesizes a deterministic time-varying gradient stack so the bench
 * is reproducible without bundling video assets. Each frame is run
 * sequentially through the vision tower (matches HF semantics —
 * Gemma4VisionModel.flatten(0,1) treats frames as batch elements) and
 * yields 70 soft tokens / frame.
 *
 *   bench_video_encode [<n_frames>] [<frame_h>] [<frame_w>]
 *
 * Defaults: 32 frames, 192×192. Reports per-frame and per-clip wall
 * time so the cost model scales cleanly from one image (bench_vision_encode)
 * to one video.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/vision_siglip/vision_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define V_SOFT 1536
#define BENCH_RUNS 3

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

static void synth_frame(int frame_idx, int n_frames, int h, int w, uint8_t *out) {
    const float phase = (float) frame_idx / (float) n_frames;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float    fx = (float) x / (w - 1);
            float    fy = (float) y / (h - 1);
            float    r  = 255.0f * (0.5f + 0.5f * sinf(6.28f * (fx + phase)));
            float    g  = 255.0f * (0.5f + 0.5f * sinf(6.28f * (fy + 1.7f * phase)));
            float    b  = 255.0f * (0.5f + 0.5f * sinf(6.28f * (fx + fy + 2.3f * phase)));
            uint8_t *p  = out + (y * w + x) * 3;
            p[0]        = (uint8_t) (r < 0 ? 0 : (r > 255 ? 255 : r));
            p[1]        = (uint8_t) (g < 0 ? 0 : (g > 255 ? 255 : g));
            p[2]        = (uint8_t) (b < 0 ? 0 : (b > 255 ? 255 : b));
        }
    }
}

static const char *find_weights(void) {
    static const char *candidates[] = {
            "./vision_bench/vision_tower.safetensors",
            "../gemma-4-E2B-it/vision_tower.safetensors",
            "vision_tower.safetensors",
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

int main(int argc, char **argv) {
    const int n_frames = argc > 1 ? atoi(argv[1]) : 32;
    const int frame_h  = argc > 2 ? atoi(argv[2]) : 192;
    const int frame_w  = argc > 3 ? atoi(argv[3]) : 192;
    if (n_frames < 1 || frame_h < 64 || frame_w < 64) {
        printf("SKIP: invalid frame dims (n=%d h=%d w=%d)\n", n_frames, frame_h, frame_w);
        return GEIST_TEST_SKIP;
    }

    const char *weights = find_weights();
    if (weights == nullptr) {
        printf("SKIP: vision_tower.safetensors not found "
               "(vision_bench/, ../gemma-4-E2B-it/, cwd)\n");
        return GEIST_TEST_SKIP;
    }
    printf("bench_video_encode: %d frames @ %dx%d, weights %s\n",
           n_frames,
           frame_h,
           frame_w,
           weights);

    const size_t frame_bytes = (size_t) frame_h * frame_w * 3;
    uint8_t     *frames      = malloc((size_t) n_frames * frame_bytes);
    if (frames == nullptr)
        return GEIST_TEST_ERROR;
    for (int f = 0; f < n_frames; f++) {
        synth_frame(f, n_frames, frame_h, frame_w, frames + (size_t) f * frame_bytes);
    }

    struct VisionEncoder *enc = vision_encoder_create(weights);
    if (enc == nullptr) {
        free(frames);
        return GEIST_TEST_ERROR;
    }

    float *soft =
            malloc((size_t) n_frames * VISION_SOFT_TOKENS_PER_VIDEO_FRAME * V_SOFT * sizeof(float));
    if (soft == nullptr) {
        vision_encoder_destroy(enc);
        free(frames);
        return GEIST_TEST_ERROR;
    }

    /* Warmup */
    size_t n_soft = vision_encoder_run_video(
            enc, (size_t) n_frames, (size_t) frame_h, (size_t) frame_w, frames, soft);
    if (n_soft == 0) {
        fprintf(stderr, "vision_encoder_run_video returned 0\n");
        free(soft);
        vision_encoder_destroy(enc);
        free(frames);
        return GEIST_TEST_FAIL;
    }

    double total = 0;
    double tmin = 1e30, tmax = 0;
    for (int r = 0; r < BENCH_RUNS; r++) {
        double t0 = now_ms();
        (void) vision_encoder_run_video(
                enc, (size_t) n_frames, (size_t) frame_h, (size_t) frame_w, frames, soft);
        double dt = now_ms() - t0;
        total += dt;
        if (dt < tmin)
            tmin = dt;
        if (dt > tmax)
            tmax = dt;
        printf("  run %d: %.1f ms total, %.1f ms/frame\n", r, dt, dt / n_frames);
    }
    const double mean = total / BENCH_RUNS;
    printf("bench_video_encode: %d frames @ %dx%d, soft_tokens=%zu, "
           "%d runs, mean=%.1f ms (%.1f ms/frame) "
           "[min=%.1f, max=%.1f]\n",
           n_frames,
           frame_h,
           frame_w,
           n_soft,
           BENCH_RUNS,
           mean,
           mean / n_frames,
           tmin,
           tmax);

    free(soft);
    vision_encoder_destroy(enc);
    free(frames);
    return GEIST_TEST_PASS;
}
