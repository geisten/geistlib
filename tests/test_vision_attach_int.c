/*
 * test_vision_attach_int — end-to-end vision pipeline via geist_session
 * public API.
 *
 *   PNG (uint8 RGB HWC) → image_pipeline → vision_siglip tower → pool →
 *                       → projector → soft tokens (1536-dim)
 *                       → decoder prefill_image → decode
 *
 * SKIPs cleanly if any of the three required files is missing:
 *   - GGUF model (GEIST_GGUF_PATH or default search)
 *   - vision_tower.safetensors (auto-found near model)
 *   - input PNG (path passed as argv[1], optional; defaults to
 *     vision_bench/syn_320x224.png)
 *
 * Smoke test for the vision_siglip encoder arch.
 */
#include "test_helpers.h"

#include "stb_image.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    GEIST_REQUIRE_GGUF(model_path);

    const char *png_path = argc > 1 ? argv[1] : "vision_bench/syn_320x224.png";
    int         w = 0, h = 0, c = 0;
    uint8_t    *rgb = stbi_load(png_path, &w, &h, &c, 3);
    if (rgb == nullptr) {
        GEIST_SKIP("could not decode PNG (set first arg to a valid image, "
                   "default: vision_bench/syn_320x224.png)");
    }
    printf("loaded %s: %dx%d RGB (%d ch input → 3 forced)\n", png_path, h, w, c);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        stbi_image_free(rgb);
        return GEIST_TEST_ERROR;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        stbi_image_free(rgb);
        return GEIST_TEST_FAIL;
    }

    struct geist_session_opts opts = {.max_seq_len = 1024};
    struct geist_session     *sess = nullptr;
    s                              = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        stbi_image_free(rgb);
        return GEIST_TEST_FAIL;
    }

    s = geist_session_attach_image(sess, (size_t) h, (size_t) w, rgb);
    stbi_image_free(rgb);
    if (s == GEIST_E_NOT_FOUND) {
        printf("SKIP: vision_tower.safetensors not found\n");
        printf("  errmsg: %s\n", geist_session_errmsg(sess));
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    if (s != GEIST_OK) {
        fprintf(stderr,
                "attach_image failed: %s — %s\n",
                geist_status_to_string(s),
                geist_session_errmsg(sess));
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("image attached: 280 soft tokens injected into KV cache\n");

    /* Decode a handful of tokens after the vision context. We don't
     * expect coherent text from a bare image with no chat-template wrap
     * — this just validates the prefill_image path didn't poison the
     * KV cache or the logits head. */
    int           fails = 0;
    geist_token_t tok;
    printf("decoded:");
    for (int i = 0; i < 6; i++) {
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK) {
            fprintf(stderr, "\ndecode_step[%d] failed\n", i);
            fails++;
            break;
        }
        const char *text = geist_session_token_to_str(sess, tok);
        printf(" %d(%s)", tok, text != nullptr ? text : "?");
    }
    printf("\n");

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
