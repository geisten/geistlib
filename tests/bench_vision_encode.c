/*
 * bench_vision_encode — standalone vision tower / encoder bench.
 *
 * Skips the LM entirely so it fits on a 4 GB Pi 5 (vision_tower
 * safetensors is ~322 MB FP32 alone). Decodes a PNG via stb_image,
 * runs vision_encoder_run_image (preprocess → 16-layer ViT → kernel-3
 * pool → projector → 280 soft tokens), prints wall time + soft-token
 * count.
 *
 *   bench_vision_encode [<image.png>]
 *
 * Without argument, falls back to vision_bench/syn_320x224.png so the
 * bench is reproducible from a clean checkout. Resolves
 * vision_tower.safetensors via the normal vision_siglip lookup
 * (vision_bench/, ../gemma-4-E2B-it/, etc.) so this also doubles as a
 * smoke test for the tower load path.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/vision_siglip/vision_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include "stb_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define V_SOFT 1536
#define BENCH_RUNS 5

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
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
    const char *img_path = argc > 1 ? argv[1] : "vision_bench/syn_320x224.png";

    int      w = 0, h = 0, c = 0;
    uint8_t *rgb = stbi_load(img_path, &w, &h, &c, 3);
    if (rgb == nullptr) {
        printf("SKIP: stb_image failed to load %s\n", img_path);
        return GEIST_TEST_SKIP;
    }
    const char *weights = find_weights();
    if (weights == nullptr) {
        stbi_image_free(rgb);
        printf("SKIP: vision_tower.safetensors not found "
               "(vision_bench/, ../gemma-4-E2B-it/, cwd)\n");
        return GEIST_TEST_SKIP;
    }
    printf("bench_vision_encode: image %s (%dx%d), weights %s\n", img_path, h, w, weights);

    struct VisionEncoder *enc = vision_encoder_create(weights);
    if (enc == nullptr) {
        stbi_image_free(rgb);
        return GEIST_TEST_ERROR;
    }

    float *soft = malloc(VISION_SOFT_TOKENS_PER_IMAGE * V_SOFT * sizeof(float));
    if (soft == nullptr) {
        vision_encoder_destroy(enc);
        stbi_image_free(rgb);
        return GEIST_TEST_ERROR;
    }

    /* Warmup */
    size_t n_soft = vision_encoder_run_image(enc, (size_t) h, (size_t) w, rgb, soft);
    if (n_soft == 0) {
        fprintf(stderr, "vision_encoder_run_image returned 0\n");
        free(soft);
        vision_encoder_destroy(enc);
        stbi_image_free(rgb);
        return GEIST_TEST_FAIL;
    }

    /* Timed runs. */
    double total = 0;
    double tmin = 1e30, tmax = 0;
    for (int r = 0; r < BENCH_RUNS; r++) {
        double t0 = now_ms();
        (void) vision_encoder_run_image(enc, (size_t) h, (size_t) w, rgb, soft);
        double dt = now_ms() - t0;
        total += dt;
        if (dt < tmin)
            tmin = dt;
        if (dt > tmax)
            tmax = dt;
        printf("  run %d: %.1f ms\n", r, dt);
    }
    const double mean = total / BENCH_RUNS;
    printf("bench_vision_encode: image=%dx%d, soft_tokens=%zu, "
           "%d runs, mean=%.1f ms (min=%.1f, max=%.1f)\n",
           h,
           w,
           n_soft,
           BENCH_RUNS,
           mean,
           tmin,
           tmax);

    free(soft);
    vision_encoder_destroy(enc);
    stbi_image_free(rgb);
    return GEIST_TEST_PASS;
}
