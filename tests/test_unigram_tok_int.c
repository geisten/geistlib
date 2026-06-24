/*
 * test_unigram_tok_int — GGUF-embedded SentencePiece tokenizer, real model.
 *
 * Guards the shared SPM encode/decode path and, when GEIST_GGUF_PATH points at a
 * unigram model (no merges, scores only — e.g. BitNet's llama tokenizer), the
 * unigram merge algorithm specifically. The regression it pins: a score-driven
 * lattice collapses unigram output to single byte tokens (▁ → three <0xXX>),
 * so "The capital of France is" exploded to ~36 tokens. Correct subword merging
 * yields a handful. We assert a small token count (no byte-collapse) + a decode
 * round-trip that recovers the words. CI's reference model is SPM-with-merges,
 * so the unigram-specific branch is exercised locally (point GEIST_GGUF_PATH at
 * a unigram GGUF); CI still covers the shared encode/decode. No assert().
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <string.h>

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        GEIST_SKIP("backend_create failed");
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        GEIST_SKIP("model_load failed (set GEIST_GGUF_PATH)");
    }
    struct geist_session_opts opts = {0};
    struct geist_session     *s    = nullptr;
    if (geist_session_create(model, be, &opts, &s) != GEIST_OK) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP("session_create failed");
    }

    int               rc   = GEIST_TEST_PASS;
    const char       *text = "The capital of France is";
    geist_token_t     ids[64];
    size_t            n  = 0;
    enum geist_status st = geist_session_tokenize(s, text, 64, ids, &n);
    if (st != GEIST_OK) {
        /* No tokenizer at all -> the GGUF lacks one and has no sibling .bin. */
        fprintf(stderr, "tokenize failed: %s\n", geist_status_to_string(st));
        rc = GEIST_TEST_SKIP;
    } else {
        fprintf(stderr, "tokenized %zu ids\n", n);
        /* Byte-collapse would give ~36 for this 24-char string; subword ~6. */
        if (n == 0 || n > 12) {
            fprintf(stderr, "FAIL: expected subword tokens, got %zu (byte-level collapse?)\n", n);
            rc = GEIST_TEST_FAIL;
        }
        /* Decode round-trip: token pieces must recover the content words. */
        char   buf[512];
        size_t w = 0;
        for (size_t i = 0; i < n && w + 1 < sizeof buf; i++) {
            const char *piece = geist_session_token_to_str(s, ids[i]);
            if (!piece) {
                continue;
            }
            size_t pl = strlen(piece);
            if (w + pl >= sizeof buf) {
                break;
            }
            memcpy(buf + w, piece, pl);
            w += pl;
        }
        buf[w] = '\0';
        if (rc == GEIST_TEST_PASS &&
            (strstr(buf, "capital") == nullptr || strstr(buf, "France") == nullptr)) {
            fprintf(stderr, "FAIL: round-trip lost words: \"%s\"\n", buf);
            rc = GEIST_TEST_FAIL;
        }
        if (rc == GEIST_TEST_PASS) {
            printf("gguf tokenizer: subword encode + decode round-trip ok (%zu ids)\n", n);
        }
    }

    geist_session_destroy(s);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return rc;
}
