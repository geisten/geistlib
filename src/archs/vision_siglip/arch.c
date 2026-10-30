/*
 * src/archs/vision_siglip/arch.c — Gemma 4 vision tower.
 *
 * Layer: ARCHITECTURE.
 *
 * Locates vision_tower.safetensors and runs the tower on images and
 * video frames.
 *
 * Search heuristics for vision_tower.safetensors:
 *   - $GEIST_VISION_MODEL_PATH env override
 *   - <aux_search_root>/vision_tower.safetensors  (typ. dir of GGUF)
 *   - ./vision_bench/vision_tower.safetensors
 *   - ./vision_tower.safetensors
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch.h"

#include "aux_file.h"
#include "heap.h"
#include "vision_encoder.h"

#include <geist.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The encoder state is the VisionEncoder itself. */

static void *vision_siglip_state_create(struct geist_backend *be, const char *aux_root) {
    (void) be; /* the tower runs on host memory, not backend buffers */

    static const char *vision_fallbacks[] = {
            "./vision_bench/vision_tower.safetensors",
            "vision_tower.safetensors",
            nullptr,
    };
    char *path = geist_find_aux_file(
            "GEIST_VISION_MODEL_PATH", aux_root, "vision_tower.safetensors", vision_fallbacks);
    if (path == nullptr) {
        return nullptr;
    }
    struct VisionEncoder *enc = vision_encoder_create(path);
    safe_free((void **) &path);
    return enc;
}

static void vision_siglip_state_destroy(void *encoder_state) {
    vision_encoder_destroy(encoder_state);
}

static size_t vision_siglip_encode_image(void          *encoder_state,
                                         size_t         height,
                                         size_t         width,
                                         size_t         max_soft,
                                         const uint8_t *rgb,
                                         float         *out_soft) {
    if (encoder_state == nullptr || rgb == nullptr || out_soft == nullptr || max_soft == 0 ||
        height == 0 || width == 0) {
        return 0;
    }
    size_t n_soft = vision_encoder_run_image(encoder_state, height, width, rgb, out_soft);
    if (n_soft > max_soft) {
        return 0;
    }
    return n_soft;
}

static size_t vision_siglip_encode_video(void          *encoder_state,
                                         size_t         n_frames,
                                         size_t         height,
                                         size_t         width,
                                         size_t         max_soft,
                                         const uint8_t *frames,
                                         float         *out_soft) {
    if (encoder_state == nullptr || frames == nullptr || out_soft == nullptr || max_soft == 0 ||
        n_frames == 0 || height == 0 || width == 0) {
        return 0;
    }
    size_t n_soft =
            vision_encoder_run_video(encoder_state, n_frames, height, width, frames, out_soft);
    if (n_soft > max_soft) {
        return 0;
    }
    return n_soft;
}

static size_t vision_siglip_soft_token_dim(const void *encoder_state) {
    /* Per-checkpoint: 1536 (E2B) or 2560 (E4B) — the text model's
     * residual-stream width, read from the tower safetensors (#258). */
    return vision_encoder_soft_dim(encoder_state);
}

const struct geist_arch_ops_vision geist_arch_vision_siglip = {
        .name           = "vision_siglip",
        .state_create   = vision_siglip_state_create,
        .state_destroy  = vision_siglip_state_destroy,
        .encode_image   = vision_siglip_encode_image,
        .encode_video   = vision_siglip_encode_video,
        .soft_token_dim = vision_siglip_soft_token_dim,
};
