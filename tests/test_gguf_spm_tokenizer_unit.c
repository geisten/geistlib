/*
 * test_gguf_spm_tokenizer_unit — GGUF-embedded SentencePiece (▁ BPE)
 * tokenizer regression test.
 *
 * Covers the gguf_tokenizer SPM path (model="gemma4"/"llama"/… with merges):
 *   1. Lossless round-trip — encode(text) then concatenate the decoded token
 *      strings must reproduce the input exactly, across ASCII, punctuation,
 *      code/newlines, multi-byte UTF-8, and runs of spaces. This exercises ▁
 *      normalization, the merge engine, and ▁→space / <0xXX>→byte decode.
 *   2. Reference token — for the gemma4-E2B model, "Hello" encodes to the
 *      single id 9259 (the value test_session_sampling_int documents as the
 *      peaked greedy prediction). Anchors the segmentation, not just losslessness.
 *
 * SKIPs cleanly when no GGUF model is reachable, or when the model carries no
 * GGUF-embedded tokenizer (set_prompt/tokenize return GEIST_E_NOT_FOUND).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

/* gemma4-E2B "Hello" → this single vocab id (see test_session_sampling_int). */
#define GEMMA4_HELLO_TOKEN 9259

/* Decode all ids by concatenating per-token strings into out (NUL-terminated). */
static void decode_concat(
        struct geist_session *s, const geist_token_t *ids, size_t n, char *out, size_t out_cap) {
    size_t w = 0;
    for (size_t i = 0; i < n; i++) {
        const char *p = geist_session_token_to_str(s, ids[i]);
        if (p == nullptr)
            continue;
        size_t l = strlen(p);
        if (w + l < out_cap) {
            memcpy(out + w, p, l);
            w += l;
        }
    }
    out[w < out_cap ? w : out_cap - 1] = '\0';
}

/* Returns: 0 = round-trip OK, 1 = mismatch, 77 = no tokenizer (skip). */
static int check_roundtrip(struct geist_session *s, const char *text) {
    geist_token_t     ids[1024];
    size_t            n  = 0;
    enum geist_status st = geist_session_tokenize(s, text, 1024, ids, &n);
    if (st == GEIST_E_NOT_FOUND)
        return 77;
    if (st != GEIST_OK) {
        fprintf(stderr, "  tokenize(\"%s\") failed: %s\n", text, geist_status_to_string(st));
        return 1;
    }
    char out[4096];
    decode_concat(s, ids, n, out, sizeof out);
    if (strcmp(out, text) != 0) {
        fprintf(stderr, "  roundtrip mismatch:\n    in : \"%s\"\n    out: \"%s\"\n", text, out);
        return 1;
    }
    return 0;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load(%s) failed: %s\n", model_path, geist_status_to_string(s));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    struct geist_session     *sess = nullptr;
    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    s                              = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed: %s\n", geist_status_to_string(s));
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    int fails = 0;

    /* ---- 1. Lossless round-trip across varied inputs ---- */
    static const char *cases[] = {
            "Hello world",
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "caf\xC3\xA9 r\xC3\xA9sum\xC3\xA9", /* café résumé (multi-byte) */
            "  leading and  double  spaces ",   /* space runs + trailing */
            "def f(x):\n    return x * 2",      /* code + newline + indent */
            "\xF0\x9F\x98\x80 emoji",           /* 😀 (4-byte UTF-8) */
            "1234567890",
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        int rc = check_roundtrip(sess, cases[i]);
        if (rc == 77) {
            /* No GGUF-embedded tokenizer (e.g. gpt2-only or external-bin model). */
            printf("SKIP: model has no GGUF-embedded SPM tokenizer\n");
            geist_session_destroy(sess);
            geist_model_destroy(model);
            geist_backend_destroy(be);
            return GEIST_TEST_SKIP;
        }
        fails += rc;
    }
    if (fails == 0)
        printf("roundtrip: %zu cases OK\n", sizeof cases / sizeof cases[0]);

    /* ---- 2. Reference token: "Hello" → 9259 (gemma4-E2B) ---- */
    {
        geist_token_t ids[16];
        size_t        n = 0;
        s               = geist_session_tokenize(sess, "Hello", 16, ids, &n);
        if (s != GEIST_OK || n < 1) {
            fprintf(stderr,
                    "tokenize(\"Hello\") failed: %s (n=%zu)\n",
                    geist_status_to_string(s),
                    n);
            fails++;
        } else if (ids[0] != GEMMA4_HELLO_TOKEN) {
            /* Model-specific anchor — only meaningful for gemma4-e2b-Q4_K_M. */
            fprintf(stderr,
                    "reference token mismatch: \"Hello\"[0] = %d, expected %d "
                    "(is this the gemma4-E2B model?)\n",
                    (int) ids[0],
                    GEMMA4_HELLO_TOKEN);
            fails++;
        } else {
            printf("reference: \"Hello\" -> %d OK\n", GEMMA4_HELLO_TOKEN);
        }
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS: GGUF SPM tokenizer round-trips + reference token\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
