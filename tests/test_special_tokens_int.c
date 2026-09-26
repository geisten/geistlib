/*
 * test_special_tokens_int — public special-token accessors.
 *
 * Exercises geist_model_eos_token / _bos_token / _token_by_text (added so a
 * chat app can stop generation by token-id instead of string-matching the
 * decoded output). Prints the ids it finds and asserts the invariants that a
 * Gemma-style instruct GGUF must satisfy. SKIPs cleanly if no GGUF is given.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <string.h>

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("auto", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        fprintf(stderr, "model load: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    const geist_token_t bos = geist_model_bos_token(model);
    const geist_token_t eos = geist_model_eos_token(model);

    /* Render eos back to text via a session, so the log shows what the model
     * actually stops on (for this Gemma 4 GGUF it is "<turn|>", id 106 — the
     * end-of-turn marker doubles as eos, so `tok == eos` is the whole stop). */
    struct geist_session *sess = nullptr;
    (void) geist_session_create(model, be, nullptr, &sess);
    const char *eos_str = sess ? geist_session_token_to_str(sess, eos) : nullptr;
    printf("bos=%d  eos=%d (%s)\n", bos, eos, eos_str ? eos_str : "(control)");

    int fail = 0;
    /* A usable instruct GGUF exposes BOS and EOS. */
    if (eos == GEIST_TOKEN_NONE) {
        fprintf(stderr, "FAIL: no eos id\n");
        fail = 1;
    }
    /* Qwen GGUFs carry no BOS by design (the tokenizer adds none); only the
     * families that prepend one must expose it. */
    const char *arch = geist_model_arch(model);
    const bool  has_bos =
            arch != nullptr && (strcmp(arch, "gemma4") == 0 || strcmp(arch, "gemma3") == 0 ||
                                strcmp(arch, "llama") == 0);
    if (bos == GEIST_TOKEN_NONE && has_bos) {
        fprintf(stderr, "FAIL: no bos id for %s\n", arch);
        fail = 1;
    }

    /* Round-trip: looking up eos's own surface string returns eos again. This
     * exercises token_by_text positively without hardcoding model-specific
     * strings (skipped if eos renders as a control token). */
    if (eos_str != nullptr) {
        const geist_token_t round = geist_model_token_by_text(model, eos_str);
        printf("token_by_text(\"%s\") = %d\n", eos_str, round);
        if (round != eos) {
            fprintf(stderr, "FAIL: token_by_text round-trip %d != eos %d\n", round, eos);
            fail = 1;
        }
    }
    /* A non-token string resolves to NONE, not a stray id. */
    if (geist_model_token_by_text(model, "definitely not a single token") != GEIST_TOKEN_NONE) {
        fprintf(stderr, "FAIL: bogus text did not return GEIST_TOKEN_NONE\n");
        fail = 1;
    }
    /* nullptr model is handled, not crashed. */
    if (geist_model_eos_token(nullptr) != GEIST_TOKEN_NONE ||
        geist_model_token_by_text(nullptr, "x") != GEIST_TOKEN_NONE) {
        fprintf(stderr, "FAIL: nullptr model not handled\n");
        fail = 1;
    }

    if (sess) {
        geist_session_destroy(sess);
    }
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return fail ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
