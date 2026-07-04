/*
 * test_session_lifecycle_int — verifies the engine-side session API
 * (model_load → session_create → prefill_tokens → decode_step → destroy).
 *
 * SKIPs cleanly if no GGUF model is available (GEIST_GGUF_PATH env or
 * default-path search). Otherwise loads the model, prefills 4 BOS-only
 * tokens, decodes 3 tokens, prints them.
 *
 * Phase B-4a smoke test — confirms the facade wiring works end-to-end.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        /* Fall back to cpu_scalar if neon not linked. */
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "model_load(%s) failed: %s — %s\n",
                model_path,
                geist_status_to_string(s),
                geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("loaded model from %s (arch=%s, backend=%s)\n",
           model_path,
           geist_model_arch(model),
           geist_backend_name(be));

    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    struct geist_session     *sess = nullptr;
    s                              = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed: %s\n", geist_status_to_string(s));
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Prefer set_prompt (uses tokenizer) when tokenizer.bin is reachable;
     * fall back to prefill_tokens with BOS only otherwise. */
    s = geist_session_set_prompt(sess, "Hello");
    if (s == GEIST_OK) {
        printf("set_prompt(\"Hello\") succeeded — tokenizer wired\n");
    } else if (s == GEIST_E_NOT_FOUND) {
        printf("set_prompt: no tokenizer.bin (expected on some hosts), "
               "falling back to prefill_tokens with BOS\n");
        const geist_token_t prompt_ids[] = {2 /* BOS */};
        s                                = geist_session_prefill_tokens(sess, 1, prompt_ids);
    }
    if (s != GEIST_OK) {
        fprintf(stderr,
                "prefill failed: %s — %s\n",
                geist_status_to_string(s),
                geist_session_errmsg(sess));
        goto cleanup_fail;
    }

    int           fails = 0;
    geist_token_t tok;
    for (int i = 0; i < 3; i++) {
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK) {
            fprintf(stderr, "decode_step[%d] failed: %s\n", i, geist_status_to_string(s));
            fails++;
            break;
        }
        const char *text = geist_session_token_to_str(sess, tok);
        printf("decoded token[%d] = %d (%s)\n", i, tok, text != nullptr ? text : "<unknown>");
    }

    struct geist_session_stats stats;
    if (geist_session_get_stats(sess, &stats) == GEIST_OK) {
        printf("stats: n_tokens_decoded=%llu  prefill_ms=%.2f  decode_ms=%.2f\n",
               (unsigned long long) stats.n_tokens_decoded,
               (double) stats.total_prefill_ns / 1e6,
               (double) stats.total_decode_ns / 1e6);
        if (stats.n_tokens_decoded != 3) {
            fprintf(stderr,
                    "expected n_tokens_decoded=3, got %llu\n",
                    (unsigned long long) stats.n_tokens_decoded);
            fails++;
        }
        if (stats.total_prefill_ns == 0) {
            fprintf(stderr, "expected total_prefill_ns > 0\n");
            fails++;
        }
        if (stats.total_decode_ns == 0) {
            fprintf(stderr, "expected total_decode_ns > 0\n");
            fails++;
        }
    }

    geist_session_destroy(sess);

    /* Capacity guard: a prefill that can't fit the session window must be
     * rejected up-front with TOO_MANY_TOKENS, not silently no-op'd (the
     * pp>=4096 decode bug). The over-large prefill is rejected before any
     * forward runs, so this is cheap. */
    {
        struct geist_session_opts small = {.max_seq_len = 512, .temperature = 0.0f};
        struct geist_session     *s2    = nullptr;
        s                               = geist_session_create(model, be, &small, &s2);
        if (s != GEIST_OK) {
            fprintf(stderr, "capacity: session_create failed\n");
            fails++;
        } else {
            const size_t   n_big = 8192; /* > any state-level default */
            geist_token_t *big   = malloc(n_big * sizeof(geist_token_t));
            for (size_t i = 0; i < n_big; i++) {
                big[i] = 2;
            }
            s = geist_session_prefill_tokens(s2, n_big, big);
            free(big);
            if (s != GEIST_E_TOO_MANY_TOKENS) {
                fprintf(stderr,
                        "capacity: expected GEIST_E_TOO_MANY_TOKENS, got %s\n",
                        geist_status_to_string(s));
                fails++;
            } else {
                printf("capacity: oversized prefill rejected (%s)\n", geist_session_errmsg(s2));
            }
            geist_session_destroy(s2);
        }
    }

    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS: session lifecycle end-to-end\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;

cleanup_fail:
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_FAIL;
}
