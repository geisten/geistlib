/*
 * test_model_modalities_int — geist_model_modalities() vs. the attach calls.
 *
 * Pins the invariant the header promises: a set bit means the corresponding
 * geist_session_attach_* does NOT fail with GEIST_E_NOT_FOUND /
 * GEIST_E_UNSUPPORTED; a clear bit means it fails with exactly one of those.
 * Because the test asserts the invariant (not fixed per-model expectations),
 * it is meaningful against ANY GGUF the CI cache provides:
 *   - Gemma 4 with towers on disk  → audio/vision/video bits set
 *   - Gemma 4 without towers       → mask 0, attach returns NOT_FOUND
 *   - BitNet / Llama (text-only)   → mask 0, attach returns NOT_FOUND
 *
 * Additionally: GEIST_TEXT_ONLY=1 must force the mask to 0 on reload.
 *
 * SKIPs cleanly when no GGUF is found (GEIST_GGUF_PATH or default search).
 */
#define _POSIX_C_SOURCE 200809L /* setenv */

#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One second of 16 kHz silence — enough samples that a capable encoder
 * produces at least one soft token, so attach only fails for capability
 * reasons. */
#define N_PCM 16000
static int16_t pcm_silence[N_PCM];

/* Tiny 8x8 RGB frame; the vision encoder owns resize, any size is valid. */
#define IMG_DIM 8
static uint8_t rgb_gray[IMG_DIM * IMG_DIM * 3];

static bool is_capability_error(enum geist_status s) {
    return s == GEIST_E_NOT_FOUND || s == GEIST_E_UNSUPPORTED;
}

/* Assert the bit ⟺ attach-status invariant for one modality. Returns the
 * number of failures (0 or 1). Each attach gets a fresh session so a
 * successful soft-token injection can't disturb the next check. */
static int check_modality(struct geist_model   *model,
                          struct geist_backend *be,
                          unsigned              mask,
                          unsigned              bit,
                          const char           *name,
                          enum geist_status (*attach)(struct geist_session *)) {
    struct geist_session_opts opts = {.max_seq_len = 1024};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "FAIL %s: session_create failed\n", name);
        return 1;
    }
    enum geist_status s    = attach(sess);
    int               fail = 0;
    if (mask & bit) {
        if (is_capability_error(s)) {
            fprintf(stderr,
                    "FAIL %s: bit set but attach returned %s — %s\n",
                    name,
                    geist_status_to_string(s),
                    geist_session_errmsg(sess));
            fail = 1;
        }
    } else {
        if (!is_capability_error(s)) {
            fprintf(stderr,
                    "FAIL %s: bit clear but attach returned %s "
                    "(expected NOT_FOUND or UNSUPPORTED)\n",
                    name,
                    geist_status_to_string(s));
            fail = 1;
        }
    }
    printf("  %-6s bit=%d attach=%s%s\n",
           name,
           (mask & bit) ? 1 : 0,
           geist_status_to_string(s),
           fail ? "  <-- MISMATCH" : "");
    geist_session_destroy(sess);
    return fail;
}

static enum geist_status attach_audio_cb(struct geist_session *s) {
    return geist_session_attach_audio(s, N_PCM, pcm_silence, 16000);
}

static enum geist_status attach_image_cb(struct geist_session *s) {
    return geist_session_attach_image(s, IMG_DIM, IMG_DIM, rgb_gray);
}

static enum geist_status attach_video_cb(struct geist_session *s) {
    return geist_session_attach_video(s, 1, IMG_DIM, IMG_DIM, rgb_gray);
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    memset(rgb_gray, 128, sizeof(rgb_gray));

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    int fails = 0;

    /* nullptr is a valid query and answers 0. */
    if (geist_model_modalities(nullptr) != 0) {
        fprintf(stderr, "FAIL: geist_model_modalities(nullptr) != 0\n");
        fails++;
    }

    /* --- normal load: mask must match attach behaviour bit for bit ------- */
    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    unsigned mask = geist_model_modalities(model);
    printf("model: %s (arch %s), modalities mask = 0x%x\n",
           model_path,
           geist_model_arch(model),
           mask);

    if (mask & ~(unsigned) (GEIST_MOD_AUDIO | GEIST_MOD_VISION | GEIST_MOD_VIDEO)) {
        fprintf(stderr, "FAIL: unknown bits in mask 0x%x\n", mask);
        fails++;
    }

    fails += check_modality(model, be, mask, GEIST_MOD_AUDIO, "audio", attach_audio_cb);
    fails += check_modality(model, be, mask, GEIST_MOD_VISION, "vision", attach_image_cb);
    fails += check_modality(model, be, mask, GEIST_MOD_VIDEO, "video", attach_video_cb);
    geist_model_destroy(model);

    /* --- GEIST_TEXT_ONLY=1: same file must reload with mask 0 ------------ */
    setenv("GEIST_TEXT_ONLY", "1", 1);
    struct geist_model *text_only = nullptr;
    s                             = geist_model_load(model_path, be, &text_only);
    unsetenv("GEIST_TEXT_ONLY");
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load (GEIST_TEXT_ONLY=1) failed\n");
        fails++;
    } else {
        unsigned to_mask = geist_model_modalities(text_only);
        printf("GEIST_TEXT_ONLY=1 reload: mask = 0x%x\n", to_mask);
        if (to_mask != 0) {
            fprintf(stderr, "FAIL: GEIST_TEXT_ONLY=1 but mask is 0x%x\n", to_mask);
            fails++;
        }
        /* Invariant holds under text-only too: attach must now refuse. */
        fails += check_modality(text_only, be, to_mask, GEIST_MOD_AUDIO, "audio", attach_audio_cb);
        geist_model_destroy(text_only);
    }

    geist_backend_destroy(be);
    if (fails == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL (%d)\n", fails);
    }
    return fails == 0 ? 0 : GEIST_TEST_FAIL;
}
