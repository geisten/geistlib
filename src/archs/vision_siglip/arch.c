/*
 * src/archs/vision_siglip/arch.c — Gemma 4 vision tower.
 *
 * Layer: ARCHITECTURE.
 *
 * Phase P1: skeleton. Locates vision_tower.safetensors and opens it;
 * encode_image / encode_video return 0 until the per-block forward
 * lands in P3-P4.
 *
 * Search heuristics for vision_tower.safetensors:
 *   - $GEIST_VISION_MODEL_PATH env override
 *   - <aux_search_root>/vision_tower.safetensors  (typ. dir of GGUF)
 *   - ./vision_bench/vision_tower.safetensors
 *   - ../gemma-4-E2B-it/vision_tower.safetensors
 *   - ./vision_tower.safetensors
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch.h"

#include "heap.h"
#include "vision_encoder.h"

#include <geist.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct vision_siglip_state {
    struct VisionEncoder *enc;
};

/* Mirrors audio_conformer's find_file. Kept arch-local for now —
 * promotion to a shared helper waits for the third encoder. */
static char *find_file(const char        *env_name,
                       const char        *aux_root,
                       const char        *basename,
                       const char *const *fallbacks) {
    const char *env = env_name != nullptr ? getenv(env_name) : nullptr;
    if (env != nullptr && env[0] != '\0') {
        FILE *f = fopen(env, "rb");
        if (f != nullptr) {
            fclose(f);
            return strdup(env);
        }
    }
    if (aux_root != nullptr) {
        size_t need = strlen(aux_root) + 1 + strlen(basename) + 1;
        char  *cand = heap_alloc_aligned(need, alignof(char));
        if (cand != nullptr) {
            snprintf(cand, need, "%s/%s", aux_root, basename);
            FILE *f = fopen(cand, "rb");
            if (f != nullptr) {
                fclose(f);
                return cand;
            }
            safe_free((void **) &cand);
        }
    }
    if (fallbacks != nullptr) {
        for (size_t i = 0; fallbacks[i] != nullptr; i++) {
            FILE *f = fopen(fallbacks[i], "rb");
            if (f != nullptr) {
                fclose(f);
                return strdup(fallbacks[i]);
            }
        }
    }
    return nullptr;
}

static void *vision_siglip_state_create(struct geist_backend *be, const char *aux_root) {
    (void) be; /* P3+: tower weights will route through backend buffers. */

    static const char *vision_fallbacks[] = {
            "./vision_bench/vision_tower.safetensors",
            "../gemma-4-E2B-it/vision_tower.safetensors",
            "vision_tower.safetensors",
            nullptr,
    };
    char *path = find_file(
            "GEIST_VISION_MODEL_PATH", aux_root, "vision_tower.safetensors", vision_fallbacks);
    if (path == nullptr) {
        return nullptr;
    }
    struct VisionEncoder *enc = vision_encoder_create(path);
    safe_free((void **) &path);
    if (enc == nullptr) {
        return nullptr;
    }
    struct vision_siglip_state *st =
            heap_alloc_aligned(sizeof(*st), alignof(struct vision_siglip_state));
    if (st == nullptr) {
        vision_encoder_destroy(enc);
        return nullptr;
    }
    *st = (struct vision_siglip_state) {.enc = enc};
    return st;
}

static void vision_siglip_state_destroy(void *encoder_state) {
    if (encoder_state == nullptr) {
        return;
    }
    struct vision_siglip_state *st = encoder_state;
    if (st->enc != nullptr) {
        vision_encoder_destroy(st->enc);
    }
    safe_free(&encoder_state);
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
    struct vision_siglip_state *st = encoder_state;
    size_t n_soft = vision_encoder_run_image(st->enc, rgb, height, width, out_soft);
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
    struct vision_siglip_state *st = encoder_state;
    size_t n_soft = vision_encoder_run_video(st->enc, frames, n_frames, height, width, out_soft);
    if (n_soft > max_soft) {
        return 0;
    }
    return n_soft;
}

static size_t vision_siglip_soft_token_dim(const void *encoder_state) {
    const struct vision_siglip_state *st = encoder_state;
    /* Per-checkpoint: 1536 (E2B) or 2560 (E4B) — the text model's
     * residual-stream width, read from the tower safetensors (#258). */
    return vision_encoder_soft_dim(st != nullptr ? st->enc : nullptr);
}

const struct geist_arch_ops_vision geist_arch_vision_siglip = {
        .name           = "vision_siglip",
        .state_create   = vision_siglip_state_create,
        .state_destroy  = vision_siglip_state_destroy,
        .encode_image   = vision_siglip_encode_image,
        .encode_video   = vision_siglip_encode_video,
        .soft_token_dim = vision_siglip_soft_token_dim,
};
