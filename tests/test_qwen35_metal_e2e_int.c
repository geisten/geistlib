/*
 * test_qwen35_metal_e2e_int — #296 hybrid Metal correctness gate.
 *
 * Runs the public API on the pinned Qwen3.5-0.8B Q8_0 fixture. The first
 * greedy continuation must match cpu_neon token-for-token, contain the
 * known answer, and reproduce after session_reset. This exercises Metal
 * Q8_0 projections/embeddings + SiLU, full-attention layers and GPU
 * DeltaNet prefill in one path.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_TOKENS = 5 };

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_QWEN35_GGUF_PATH");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    static const char *candidates[] = {
            "gguf_artifacts/qwen3.5-0.8b-q8_0.gguf",
            "./qwen3.5-0.8b-q8_0.gguf",
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

static enum geist_status run_once(const char   *path,
                                  const char   *backend_name,
                                  geist_token_t tokens[N_TOKENS],
                                  char         *decoded,
                                  size_t        decoded_cap,
                                  bool          check_reset) {
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create(backend_name, nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        return s;
    }
    struct geist_model *model = nullptr;
    s                         = geist_model_load(path, be, &model);
    if (s != GEIST_OK) {
        geist_backend_destroy(be);
        return s;
    }
    const struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    struct geist_session           *sess = nullptr;
    s                                    = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return s;
    }

    s           = geist_session_set_prompt(sess, "The capital of France is");
    size_t used = 0;
    for (size_t i = 0; s == GEIST_OK && i < N_TOKENS; i++) {
        s                 = geist_session_decode_step(sess, &tokens[i]);
        const char *piece = s == GEIST_OK ? geist_session_token_to_str(sess, tokens[i]) : nullptr;
        if (piece != nullptr) {
            const size_t n = strlen(piece);
            if (used + n + 1 < decoded_cap) {
                memcpy(decoded + used, piece, n);
                used += n;
                decoded[used] = '\0';
            }
        }
    }

    if (s == GEIST_OK && check_reset) {
        geist_token_t reset_tokens[N_TOKENS] = {0};
        geist_session_reset(sess);
        s = geist_session_set_prompt(sess, "The capital of France is");
        for (size_t i = 0; s == GEIST_OK && i < N_TOKENS; i++) {
            s = geist_session_decode_step(sess, &reset_tokens[i]);
        }
        if (s == GEIST_OK && memcmp(tokens, reset_tokens, sizeof(reset_tokens)) != 0) {
            s = GEIST_E_INTERNAL;
        }
    }

    if (s != GEIST_OK) {
        fprintf(stderr, "%s detail: %s\n", backend_name, geist_backend_errmsg(be));
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return s;
}

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no qwen35 GGUF. Run `make fetch-qwen35-model`, or set "
                           "GEIST_QWEN35_GGUF_PATH");
    }

    struct geist_backend   *probe = nullptr;
    const enum geist_status ps    = geist_backend_create("metal", nullptr, nullptr, &probe);
    if (ps == GEIST_E_NOT_FOUND || ps == GEIST_E_UNSUPPORTED) {
        GEIST_SKIP("Metal backend unavailable");
    }
    if (ps != GEIST_OK) {
        fprintf(stderr, "metal create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    geist_backend_destroy(probe);

    geist_token_t     cpu_tokens[N_TOKENS]   = {0};
    geist_token_t     metal_tokens[N_TOKENS] = {0};
    char              cpu_text[256]          = {0};
    char              metal_text[256]        = {0};
    enum geist_status s = run_once(path, "cpu_neon", cpu_tokens, cpu_text, sizeof(cpu_text), false);
    if (s != GEIST_OK) {
        fprintf(stderr, "cpu reference failed: %s\n", geist_status_to_string(s));
        return GEIST_TEST_FAIL;
    }
    s = run_once(path, "metal", metal_tokens, metal_text, sizeof(metal_text), true);
    if (s != GEIST_OK) {
        fprintf(stderr, "metal run/reset failed: %s\n", geist_status_to_string(s));
        return GEIST_TEST_FAIL;
    }
    if (memcmp(cpu_tokens, metal_tokens, sizeof(cpu_tokens)) != 0) {
        fprintf(stderr, "FAIL: Metal greedy tokens differ from cpu_neon\n");
        return GEIST_TEST_FAIL;
    }
    if (strstr(metal_text, "Paris") == nullptr) {
        fprintf(stderr, "FAIL: continuation lacks Paris: %s\n", metal_text);
        return GEIST_TEST_FAIL;
    }
    printf("cpu:   %s\nmetal: %s\nPASS: qwen35 Metal hybrid generation + reset parity\n",
           cpu_text,
           metal_text);
    return GEIST_TEST_PASS;
}
