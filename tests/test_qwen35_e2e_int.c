/*
 * test_qwen35_e2e_int — #281 end-to-end: tokenizer parity + generation +
 * recurrent-state reset equivalence.
 *
 *   1. Tokenizer parity: pre = "qwen35" (qwen2 + \p{M} in the letter
 *      classes) must reproduce HF reference ids EXACTLY, incl. cases
 *      with combining marks. Reference: tokenizers 0.22 with
 *      Qwen/Qwen3.5-0.8B tokenizer.json.
 *   2. Generation smoke: greedy continuation of "The capital of France
 *      is" must contain "Paris".
 *   3. Reset equivalence: decode 5 tokens, session_reset, decode the
 *      same prompt again — identical tokens. Pins that the DeltaNet
 *      conv/delta state actually clears on reset (a stale S would
 *      diverge immediately).
 *
 * SKIPs cleanly when the fixture is absent (make fetch-qwen35-model).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_QWEN35_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
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

struct parity_case {
    const char    *text;
    const int32_t *ids;
    size_t         n;
};

#define CASE(name, ...)                                \
    static const int32_t name##_ids[] = {__VA_ARGS__}; \
    static const size_t  name##_n     = sizeof name##_ids / sizeof name##_ids[0]

CASE(c0, 760, 6511, 314, 9338, 369);
CASE(c1, 9419, 11, 1814, 0, 220, 16, 17, 18, 19, 20);
CASE(c2, 220, 6187, 12258, 321, 3139, 3325, 198);
CASE(c3, 6262, 76239, 25, 220, 18, 11, 16, 19, 1892, 13389, 198436, 72219);
CASE(c4, 727, 281, 2007, 1590, 460, 830, 332, 17, 220, 653, 3847);
CASE(c5, 248045, 846, 198, 63614, 196484, 5810, 1477, 30, 248046, 198, 248045, 74455, 198);
CASE(c6, 247359, 15303, 210342, 18602, 220, 99986, 109120);
CASE(c7, 14572, 914, 628, 914, 2677, 914, 424, 579);
CASE(c8, 26623, 157305, 50203, 91603, 571, 238976);
CASE(c9, 87, 52033, 127, 123, 33041, 3656);

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no qwen35 GGUF. Run `make fetch-qwen35-model`, or set "
                           "GEIST_QWEN35_GGUF_PATH");
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

    /* ---- 1. Tokenizer parity. ---- */
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
            {"don't can't won't it's", c7_ids, c7_n},
            {"Voil\xc3\xa0 caf\xc3\xa9 na\xc3\xafve r\xc3\xa9sum\xc3\xa9", c8_ids, c8_n},
            {"x\xcc\x81\xc3\xbf combining \xc3\xa0", c9_ids, c9_n},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        geist_token_t ids[128];
        size_t        n = 0;
        s               = geist_session_tokenize(sess, cases[i].text, 128, ids, &n);
        bool ok         = s == GEIST_OK && n == cases[i].n;
        for (size_t k = 0; ok && k < n; k++)
            ok = ids[k] == (geist_token_t) cases[i].ids[k];
        if (!ok) {
            fprintf(stderr, "FAIL parity case %zu\n", i);
            fails++;
        }
    }
    printf("tokenizer parity: %zu cases, %d failed\n", sizeof cases / sizeof cases[0], fails);

    /* ---- 2 + 3. Generation, then reset equivalence. ---- */
    const char   *prompt        = "The capital of France is";
    geist_token_t first_run[5]  = {0};
    geist_token_t second_run[5] = {0};
    char          decoded[256]  = {0};
    size_t        used          = 0;

    s = geist_session_set_prompt(sess, prompt);
    if (s != GEIST_OK) {
        fprintf(stderr, "set_prompt FAIL: %s\n", geist_session_errmsg(sess));
        fails++;
    } else {
        for (int i = 0; i < 5; i++) {
            if (geist_session_decode_step(sess, &first_run[i]) != GEIST_OK)
                break;
            const char *piece = geist_session_token_to_str(sess, first_run[i]);
            if (piece != nullptr) {
                size_t plen = strlen(piece);
                if (used + plen < sizeof decoded - 1) {
                    memcpy(decoded + used, piece, plen);
                    used += plen;
                }
            }
        }
        printf("continuation: \"%s\"\n", decoded);
        if (strstr(decoded, "Paris") == nullptr) {
            fprintf(stderr, "FAIL: continuation lacks \"Paris\"\n");
            fails++;
        }

        geist_session_reset(sess);
        s = geist_session_set_prompt(sess, prompt);
        if (s != GEIST_OK) {
            fprintf(stderr, "post-reset set_prompt FAIL\n");
            fails++;
        } else {
            for (int i = 0; i < 5; i++) {
                if (geist_session_decode_step(sess, &second_run[i]) != GEIST_OK)
                    break;
            }
            if (memcmp(first_run, second_run, sizeof first_run) != 0) {
                fprintf(stderr,
                        "FAIL: reset not equivalent (stale DeltaNet state?) "
                        "[%d %d %d %d %d] vs [%d %d %d %d %d]\n",
                        first_run[0],
                        first_run[1],
                        first_run[2],
                        first_run[3],
                        first_run[4],
                        second_run[0],
                        second_run[1],
                        second_run[2],
                        second_run[3],
                        second_run[4]);
                fails++;
            } else {
                printf("reset equivalence: PASS\n");
            }
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
