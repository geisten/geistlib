/*
 * test_gemma3_decode_parity_int — a Gemma 3 decode step embeds the token
 * the way prefill does.
 *
 * Gemma multiplies the token embedding by sqrt(d_model). Prefill gated that
 * scale on has_embed_scale, the decode step on has_ple; Gemma 3 has the
 * scale and no per-layer embeddings, so every decoded token entered the
 * residual stream ~25x too small (d_model 640) while the perplexity test,
 * which feeds tokens through prefill, stayed green.
 *
 * Greedy-decode a few tokens, then feed the same tokens through prefill one
 * at a time and compare the logits at every position. Both paths run the
 * same weights on the same backend, so they agree to kernel rounding; the
 * missing scale moved the logits by whole units and changed the argmax.
 *
 * GEIST_GEMMA3_GGUF_PATH names the GGUF (`make fetch-gemma3-model` puts
 * one at gguf_artifacts/gemma-3-270m-it-Q4_K_M.gguf, tried next); skips
 * without.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t N_DECODE = 8;
constexpr size_t CAP      = 256;
constexpr float  TOL      = 0.05f; /* max |logit difference| over the vocab */

static const char *gemma3_path(void) {
    const char *env = getenv("GEIST_GEMMA3_GGUF_PATH");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    const char *fixture = "gguf_artifacts/gemma-3-270m-it-Q4_K_M.gguf";
    FILE       *f       = fopen(fixture, "rb");
    if (f == nullptr) {
        return nullptr;
    }
    fclose(f);
    return fixture;
}

int main(void) {
    const char *path = gemma3_path();
    if (path == nullptr) {
        GEIST_SKIP("set GEIST_GEMMA3_GGUF_PATH or run make fetch-gemma3-model");
    }
    struct geist_backend *be = nullptr;
    if (geist_backend_create(GEIST_TEST_CPU_BACKEND, nullptr, nullptr, &be) != GEIST_OK &&
        geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_model             *m    = nullptr;
    struct geist_session           *s    = nullptr;
    const struct geist_session_opts opts = {.max_seq_len = CAP, .top_p = 1.0f};
    if (geist_model_load(path, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &opts, &s) != GEIST_OK) {
        fprintf(stderr, "load failed: %s\n", geist_backend_errmsg(be));
        return GEIST_TEST_FAIL;
    }
    GEIST_REQUIRE_ARCH(geist_model_arch(m), "gemma3");

    static geist_token_t prompt[CAP];
    size_t               n_prompt = 0;
    prompt[0]                     = geist_model_bos_token(m);
    if (geist_session_tokenize(s,
                               "The capital of France is Paris. The capital of Italy is",
                               CAP - 1 - N_DECODE,
                               prompt + 1,
                               &n_prompt) != GEIST_OK ||
        n_prompt == 0) {
        fprintf(stderr, "tokenize failed\n");
        return GEIST_TEST_FAIL;
    }
    n_prompt++;

    /* Decode path: greedy tokens and the logits after each step. */
    size_t        nv = 0;
    const float  *lg = nullptr;
    geist_token_t gen[N_DECODE];
    static float  dec_logits[N_DECODE][262144];
    if (geist_session_prefill_tokens(s, n_prompt, prompt) != GEIST_OK) {
        fprintf(stderr, "prefill failed: %s\n", geist_session_errmsg(s));
        return GEIST_TEST_FAIL;
    }
    for (size_t i = 0; i < N_DECODE; i++) {
        if (geist_session_decode_step(s, &gen[i]) != GEIST_OK ||
            (lg = geist_session_peek_logits(&nv, s)) == nullptr ||
            nv > sizeof dec_logits[0] / sizeof dec_logits[0][0]) {
            fprintf(stderr, "decode step %zu failed\n", i);
            return GEIST_TEST_FAIL;
        }
        memcpy(dec_logits[i], lg, nv * sizeof(float));
    }

    /* Prefill path: the same tokens, one prefill call each. */
    float worst = 0.0f;
    if (geist_session_reset(s) != GEIST_OK ||
        geist_session_prefill_tokens(s, n_prompt, prompt) != GEIST_OK) {
        fprintf(stderr, "re-prefill failed\n");
        return GEIST_TEST_FAIL;
    }
    for (size_t i = 0; i < N_DECODE; i++) {
        size_t nv2 = 0;
        if (geist_session_prefill_tokens(s, 1, &gen[i]) != GEIST_OK ||
            (lg = geist_session_peek_logits(&nv2, s)) == nullptr || nv2 != nv) {
            fprintf(stderr, "prefill of token %zu failed\n", i);
            return GEIST_TEST_FAIL;
        }
        for (size_t v = 0; v < nv; v++) {
            const float d = fabsf(lg[v] - dec_logits[i][v]);
            worst         = d > worst ? d : worst;
        }
    }
    printf("  %zu decoded tokens, max |decode - prefill| logit %.4g (bound %.2g)\n",
           N_DECODE,
           (double) worst,
           (double) TOL);

    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    if (!(worst <= TOL)) {
        printf("FAIL: gemma3 decode and prefill disagree\n");
        return GEIST_TEST_FAIL;
    }
    printf("PASS: gemma3 decode matches prefill\n");
    return GEIST_TEST_PASS;
}
