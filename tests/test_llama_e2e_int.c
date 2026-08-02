/*
 * test_llama_e2e_int — P1.6 end-to-end check: geist_model_load →
 * geist_session_create → geist_session_set_prompt(real text) →
 * decode_step loop → decoded output text.
 *
 * Exercises the public engine API on a Llama-family GGUF. No
 * external tokenizer.bin needed — set_prompt auto-detects the
 * GGUF-embedded BPE tokenizer (P1.6 dispatch).
 *
 * SKIPs cleanly when no Llama GGUF is reachable; set
 * GEIST_LLAMA_GGUF_PATH to override the search.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ENGINE_LAYER
#include "src/engine/gguf_tokenizer.h"
#include "src/engine/model.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_LLAMA_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/smollm2-360m-instruct-q8_0.gguf",
            "./smollm2-360m-instruct-q8_0.gguf",
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

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no Llama GGUF. Run `make fetch-llama-model`, or set "
                           "GEIST_LLAMA_GGUF_PATH");
    }

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "model_load failed: %s — %s\n",
                geist_status_to_string(s),
                geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("loaded: arch=%s\n", geist_model_arch(model));

    /* Confirm the engine attached the GGUF-embedded tokenizer (not
     * the legacy sp_bpe path). */
    struct gguf_tokenizer *gtok = geist_model_internal_gguf_tokenizer(model);
    if (gtok == nullptr) {
        fprintf(stderr, "FAIL: model has no GGUF-embedded tokenizer attached\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("tokenizer: gguf-embedded, vocab=%zu, bos=%d, eos=%d\n",
           gtok->vocab_size,
           gtok->bos_id,
           gtok->eos_id);

    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    struct geist_session     *sess = nullptr;
    s                              = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed: %s\n", geist_status_to_string(s));
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Drive a real English prompt through the public API. set_prompt
     * encodes via the GGUF tokenizer (P1.5.g/.h hash-indexed), the
     * arch's prefill chunks through 32 Llama layers, and decode_step
     * emits next tokens. Print the decoded continuation. */
    int         fails       = 0;
    int32_t     out_ids[16] = {0};
    int         n_out       = 0;
    const char *prompt      = "The capital of France is";

    s = geist_session_set_prompt(sess, prompt);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "set_prompt FAIL: %s — %s\n",
                geist_status_to_string(s),
                geist_session_errmsg(sess));
        fails++;
    } else {
        printf("prompt: \"%s\"\n", prompt);
        const int N_STEPS = 12;
        for (int i = 0; i < N_STEPS; i++) {
            geist_token_t tok = -1;
            s                 = geist_session_decode_step(sess, &tok);
            if (s != GEIST_OK || tok < 0 || (size_t) tok >= gtok->vocab_size) {
                fprintf(stderr,
                        "decode_step[%d] FAIL: %s tok=%d\n",
                        i,
                        geist_status_to_string(s),
                        tok);
                fails++;
                break;
            }
            out_ids[n_out++] = (int32_t) tok;
            if (gtok->eos_id >= 0 && tok == gtok->eos_id)
                break;
        }
        char buf[256] = {0};
        gguf_tokenizer_decode(gtok, out_ids, (size_t) n_out, buf, sizeof buf);
        printf("output (%d tok): %s\n", n_out, buf);
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    if (fails == 0) {
        printf("PASS: end-to-end Llama via public API (prompt → tokens → forward → text)\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
