/*
 * test_qwen3_e2e_int — #275 end-to-end + tokenizer parity.
 *
 * Public-API path on a qwen3 GGUF:
 *   1. Tokenizer parity: geist_session_tokenize must reproduce the HF
 *      reference ids EXACTLY for a pinned fixture set covering the
 *      qwen2 pretokenizer's corners — contractions ('t), single-digit
 *      splits, punctuation-run merges ("(x", "**", ";y"), the
 *      trailing-whitespace rule ("  leading" → Ġ + Ġleading), CJK,
 *      umlauts, and chat-template special tokens INCLUDING a repeated
 *      special after a newline (the shredded-<|im_start|> bug).
 *      Reference ids: tokenizers 0.22 with Qwen/Qwen3-0.6B tokenizer.json.
 *   2. Generation smoke: greedy-decode "The capital of France is" and
 *      require "Paris" in the continuation.
 *
 * SKIPs cleanly when the fixture is absent (make fetch-qwen3-model).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_QWEN3_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/qwen3-0.6b-q8_0.gguf",
            "./qwen3-0.6b-q8_0.gguf",
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

struct parity_case {
    const char    *text;
    const int32_t *ids;
    size_t         n;
};

#define CASE(name, ...)                                \
    static const int32_t name##_ids[] = {__VA_ARGS__}; \
    static const size_t  name##_n     = sizeof name##_ids / sizeof name##_ids[0]

CASE(c0, 785, 6722, 315, 9625, 374);
CASE(c1, 9707, 11, 1879, 0, 220, 16, 17, 18, 19, 20);
CASE(c2, 220, 6388, 12621, 323, 3244, 3435, 198);
CASE(c3, 6464, 78940, 25, 220, 18, 11, 16, 19, 1959, 13785, 140448, 74764);
CASE(c4, 750, 282, 2075, 1648, 470, 856, 334, 17, 220, 671, 3980);
CASE(c5, 151644, 872, 198, 65835, 978, 14033, 5999, 1531, 30, 151645, 198, 151644, 77091, 198);
CASE(c6, 101059, 102819, 15767, 56833, 61803, 70534, 19182, 72858, 16744, 108704);
CASE(c7, 87, 28, 16, 70863, 28, 17, 26, 1350, 2075, 43010, 8);
CASE(c8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 15, 220, 24, 24, 24, 220, 15, 15, 15, 16, 16, 16);
CASE(c9, 15007, 944, 646, 944, 2765, 944, 432, 594);

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no qwen3 GGUF. Run `make fetch-qwen3-model`, or set "
                           "GEIST_QWEN3_GGUF_PATH");
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

    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    struct geist_session     *sess = nullptr;
    s                              = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed: %s\n", geist_status_to_string(s));
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    int fails = 0;

    /* ---- 1. Tokenizer parity against pinned HF reference ids. ---- */
    const struct parity_case cases[] = {
            {"The capital of France is", c0_ids, c0_n},
            {"Hello, world! 12345", c1_ids, c1_n},
            {"  leading spaces and\ttabs\n", c2_ids, c2_n},
            {"Gr\xc3\xb6\xc3\x9f"
             "e: 3,14 \xe2\x80\x94 \xc3\xbc"
             "berm\xc3\xa4\xc3\x9fig sch\xc3\xb6n",
             c3_ids,
             c3_n},
            {"def f(x): return x**2  # comment", c4_ids, c4_n},
            {"<|im_start|>user\nWie sp\xc3\xa4t ist es?<|im_end|>\n<|im_start|>assistant\n",
             c5_ids,
             c5_n},
            {"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x81\xae\xe3\x83\x86\xe3\x82\xad\xe3\x82"
             "\xb9\xe3\x83\x88\xe3\x81\xa8 \xe4\xb8\xad\xe6\x96\x87\xe6\x96\x87\xe6\x9c\xac",
             c6_ids,
             c6_n},
            {"x=1;y=2;print(x+y)", c7_ids, c7_n},
            {"1234567890 999 000111", c8_ids, c8_n},
            {"don't can't won't it's", c9_ids, c9_n},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        geist_token_t ids[256];
        size_t        n = 0;
        s               = geist_session_tokenize(sess, cases[i].text, 256, ids, &n);
        bool ok         = s == GEIST_OK && n == cases[i].n;
        for (size_t k = 0; ok && k < n; k++) {
            ok = ids[k] == (geist_token_t) cases[i].ids[k];
        }
        if (!ok) {
            fprintf(stderr, "FAIL parity case %zu: got [", i);
            for (size_t k = 0; k < n; k++)
                fprintf(stderr, "%d%s", (int) ids[k], k + 1 < n ? " " : "");
            fprintf(stderr, "] want [");
            for (size_t k = 0; k < cases[i].n; k++)
                fprintf(stderr, "%d%s", (int) cases[i].ids[k], k + 1 < cases[i].n ? " " : "");
            fprintf(stderr, "]\n");
            fails++;
        }
    }
    printf("tokenizer parity: %zu cases, %d failed\n", sizeof cases / sizeof cases[0], fails);

    /* ---- 2. Generation smoke. ---- */
    const char *prompt = "The capital of France is";
    s                  = geist_session_set_prompt(sess, prompt);
    if (s != GEIST_OK) {
        fprintf(stderr, "set_prompt FAIL: %s\n", geist_session_errmsg(sess));
        fails++;
    } else {
        char   decoded[512] = {0};
        size_t used         = 0;
        for (int i = 0; i < 12; i++) {
            geist_token_t tok = -1;
            if (geist_session_decode_step(sess, &tok) != GEIST_OK)
                break;
            const char *piece = geist_session_token_to_str(sess, tok);
            if (piece == nullptr)
                break;
            size_t plen = strlen(piece);
            if (used + plen < sizeof decoded - 1) {
                memcpy(decoded + used, piece, plen);
                used += plen;
            }
        }
        printf("continuation: \"%s\"\n", decoded);
        if (strstr(decoded, "Paris") == nullptr) {
            fprintf(stderr, "FAIL: continuation lacks \"Paris\"\n");
            fails++;
        }
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (fails > 0) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS\n");
    return GEIST_TEST_PASS;
}
