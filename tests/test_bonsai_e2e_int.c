/*
 * test_bonsai_e2e_int — Ternary-Bonsai-2-27B (PQ2_0 + prism.hadamard) end
 * to end on cpu_neon, against goldens from the PrismML llama.cpp fork
 * (PrismML-Eng/llama.cpp 01ae597, CPU and Metal agree on all of them):
 *
 *   1. the chat prompt below tokenizes to the fork's ids;
 *   2. the top-5 next-token ids after it are the fork's, in order;
 *   3. the first 16 greedy tokens are the fork's.
 *
 * geist's per-row int8 activations and the fork's blocked ones drift
 * apart after ~24 tokens on this prompt, so the greedy check stops well
 * short of that; a missing or misplaced rotation diverges at token 0.
 *
 * 27B: needs ~8 GB and a minute. SKIPs cleanly without the fixture
 * (GEIST_BONSAI_GGUF_PATH or gguf_artifacts/) or without cpu_neon.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_BONSAI_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/Ternary-Bonsai-2-27B-PQ2_0.gguf",
            "./Ternary-Bonsai-2-27B-PQ2_0.gguf",
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

static const char PROMPT[] = "<|im_start|>user\nExplain in two sentences why the sky is blue."
                             "<|im_end|>\n<|im_start|>assistant\n";
static const geist_token_t WANT_IDS[]    = {248045,
                                            846,
                                            198,
                                            814,
                                            20139,
                                            303,
                                            1330,
                                            22157,
                                            3069,
                                            279,
                                            12515,
                                            369,
                                            6105,
                                            13,
                                            248046,
                                            198,
                                            248045,
                                            74455,
                                            198};
static const geist_token_t WANT_TOP5[]   = {248068, 248046, 760, 29108, 37728};
static const geist_token_t WANT_GREEDY[] = {248068,
                                            198,
                                            760,
                                            1156,
                                            6587,
                                            264,
                                            1330,
                                            1284,
                                            17834,
                                            15673,
                                            314,
                                            3069,
                                            279,
                                            12515,
                                            369,
                                            6105};

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("Ternary-Bonsai-2-27B-PQ2_0.gguf not found (GEIST_BONSAI_GGUF_PATH)");
    }
    struct geist_backend *be = nullptr;
    GEIST_SKIP_IF(geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK,
                  "cpu_neon backend not compiled in");
    struct geist_model *m = nullptr;
    if (geist_model_load(path, be, &m) != GEIST_OK) {
        fprintf(stderr, "FAIL: load: %s\n", geist_last_create_error());
        return GEIST_TEST_FAIL;
    }
    struct geist_session_opts o = {.temperature = 0.0f};
    struct geist_session     *s = nullptr;
    if (geist_session_create(m, be, &o, &s) != GEIST_OK) {
        fprintf(stderr, "FAIL: session_create\n");
        return GEIST_TEST_FAIL;
    }

    int           fails = 0;
    geist_token_t ids[64];
    size_t        n = 0;
    fails += geist_expect(geist_session_tokenize(s, PROMPT, 64, ids, &n) == GEIST_OK, "tokenize");
    fails += geist_expect(n == sizeof WANT_IDS / sizeof WANT_IDS[0] &&
                                  memcmp(ids, WANT_IDS, sizeof WANT_IDS) == 0,
                          "prompt ids == fork");
    fails += geist_expect(geist_session_prefill_tokens(s, n, ids) == GEIST_OK, "prefill");

    size_t       nv = 0;
    const float *lg = geist_session_peek_logits(&nv, s);
    fails += geist_expect(lg != nullptr && nv > 0, "logits");
    for (size_t k = 0; lg != nullptr && k < 5; k++) {
        /* k-th best: count how many logits beat the expected id. */
        const float v    = lg[WANT_TOP5[k]];
        size_t      rank = 0;
        for (size_t t = 0; t < nv; t++) {
            rank += lg[t] > v;
        }
        char what[64];
        snprintf(what, sizeof what, "top-%zu id %d == fork", k + 1, (int) WANT_TOP5[k]);
        fails += geist_expect(rank == k, what);
    }

    for (size_t i = 0; i < sizeof WANT_GREEDY / sizeof WANT_GREEDY[0]; i++) {
        geist_token_t t = -1;
        if (geist_session_decode_step(s, &t) != GEIST_OK) {
            fails += geist_expect(false, "decode_step");
            break;
        }
        if (t != WANT_GREEDY[i]) {
            fprintf(stderr,
                    "FAIL: greedy token %zu: %d, fork %d\n",
                    i,
                    (int) t,
                    (int) WANT_GREEDY[i]);
            fails++;
            break;
        }
    }

    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    if (fails == 0) {
        printf("PASS test_bonsai_e2e_int\n");
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
