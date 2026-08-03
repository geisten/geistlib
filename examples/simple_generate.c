/*
 * simple_generate.c — the smallest useful geist program.
 *
 * Loads a GGUF, prefills a text prompt, and greedy-decodes a continuation
 * to stdout. This exercises only the STABLE core of the public API
 * (include/geist.h): backend -> model -> session -> set_prompt ->
 * decode_step -> token_to_str.
 *
 * Build & run (from the repo root):
 *   make                       # build libgeist.a for the detected target
 *   make -C examples           # build this program against it
 *   OMP_WAIT_POLICY=active examples/simple_generate \
 *       gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "The capital of France is"
 */
#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GEIST_EMBED_MODEL
/* Release builds of `geist-bitnet` fold the GGUF into the binary via
 * .incbin (see release.yml); the weights alias zero-copy from .rodata,
 * demand-paged like an mmap — same RAM behavior, one file to ship. */
extern const unsigned char geist_model_start[], geist_model_end[];
#endif

int main(int argc, char **argv) {
#ifdef GEIST_EMBED_MODEL
    const char *prompt  = (argc > 1) ? argv[1] : "Hello, my name is";
    int         max_new = (argc > 2) ? atoi(argv[2]) : 256;
#else
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [prompt] [max_new_tokens]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *prompt     = (argc > 2) ? argv[2] : "Hello, my name is";
    /* Default: up to 256 tokens, stopping earlier at the model's EOS. A
     * plain completion (no chat template) can run to the end of the
     * context without ever emitting EOS, so "unbounded" is a footgun as a
     * default; pass an explicit count (or -1 for until-context-full). */
    int         max_new    = (argc > 3) ? atoi(argv[3]) : 256;
#endif

    /* "auto" picks the best backend compiled into this build for the host. */
    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend_create failed: %s\n", geist_last_create_error());
        return 1;
    }

    struct geist_model *model = nullptr;
#ifdef GEIST_EMBED_MODEL
    if (geist_model_load_from_memory(geist_model_start,
                                     (size_t) (geist_model_end - geist_model_start),
                                     be,
                                     &model) != GEIST_OK) {
        fprintf(stderr, "embedded model load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    fprintf(stderr, "loaded embedded model (arch: %s)\n", geist_model_arch(model));
#else
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    fprintf(stderr, "loaded %s (arch: %s)\n", model_path, geist_model_arch(model));
#endif

    /* Zero-initialized opts == greedy decode (temperature 0). */
    struct geist_session_opts opts = {0};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    if (geist_session_set_prompt(sess, prompt) != GEIST_OK) {
        fprintf(stderr, "set_prompt failed: %s\n", geist_session_errmsg(sess));
        return 1;
    }

    printf("%s", prompt);
    fflush(stdout);

    const geist_token_t eos = geist_model_eos_token(model);
    for (int i = 0; max_new < 0 || i < max_new; i++) {
        geist_token_t tok = 0;
        /* A failed step also ends the run cleanly when the context is full —
         * that is the cap when no max_new_tokens argument was given. */
        if (geist_session_decode_step(sess, &tok) != GEIST_OK) {
            fprintf(stderr, "\ndecode_step failed: %s\n", geist_session_errmsg(sess));
            break;
        }
        if (tok == eos) {
            break;
        }
        /* Tokens with no surface form (true control tokens) -> stop. Gemma
         * also emits bracketed specials like "<eos>" / "<end_of_turn>" that
         * DO carry a surface form; a real app would track the model's EOS id,
         * but for a self-contained demo we just stop at the first such token. */
        const char *piece = geist_session_token_to_str(sess, tok);
        if (piece == nullptr) {
            break;
        }
        size_t len = strlen(piece);
        if (len >= 2 && piece[0] == '<' && piece[len - 1] == '>') {
            break;
        }
        fputs(piece, stdout);
        fflush(stdout);
    }
    putchar('\n');

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
