/*
 * test_awq_int — verifies AWQ scale loading + application.
 *
 * Strategy: run two sessions on the same model, one with AWQ disabled
 * (default) and one with awq_scales_path set. Decode N tokens from each;
 * assert decode succeeds in both and the two passes produce DIFFERENT
 * tokens. The "different tokens" check proves the AWQ scales actually
 * affect the forward pass (not silently ignored).
 *
 * If AWQ either crashed, was no-op, or produced garbage (all -1), the
 * test fails.
 *
 * SKIPs cleanly if either the GGUF model or the
 * gemma4-e2b.awq_scales.bin file is missing.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_DECODE 6

static int decode_tokens(struct geist_model              *model,
                         struct geist_backend            *be,
                         const struct geist_session_opts *opts,
                         geist_token_t                   *out_tokens,
                         int                              n) {
    struct geist_session *sess = nullptr;
    enum geist_status     s    = geist_session_create(model, be, opts, &sess);
    if (s != GEIST_OK)
        return -1;

    const geist_token_t bos[1] = {2};
    s                          = geist_session_prefill_tokens(sess, 1, bos);
    if (s != GEIST_OK) {
        geist_session_destroy(sess);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        s = geist_session_decode_step(sess, &out_tokens[i]);
        if (s != GEIST_OK) {
            geist_session_destroy(sess);
            return -1;
        }
    }
    geist_session_destroy(sess);
    return 0;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    /* Locate the AWQ scales file alongside the model. */
    const char *awq_env = getenv("GEIST_AWQ_PATH");
    const char *awq_path =
            awq_env != nullptr ? awq_env : "gguf_artifacts/gemma4-e2b.awq_scales.bin";

    /* Quick existence check via fopen — keeps failure mode clear. */
    {
        FILE *f = fopen(awq_path, "rb");
        if (f == nullptr) {
            printf("SKIP: awq scales file not found at %s\n", awq_path);
            return GEIST_TEST_SKIP;
        }
        fclose(f);
    }

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
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Pass A: greedy decode without AWQ. */
    struct geist_session_opts opts_no_awq = {
            .max_seq_len = 1024,
            .temperature = 0.0f,
    };
    geist_token_t tokens_no_awq[N_DECODE];
    if (decode_tokens(model, be, &opts_no_awq, tokens_no_awq, N_DECODE) != 0) {
        fprintf(stderr, "no-AWQ decode failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("no-AWQ tokens:");
    for (int i = 0; i < N_DECODE; i++)
        printf(" %d", tokens_no_awq[i]);
    printf("\n");

    /* Pass B: greedy decode with AWQ scales applied. */
    struct geist_session_opts opts_with_awq = {
            .max_seq_len     = 1024,
            .temperature     = 0.0f,
            .awq_scales_path = awq_path,
    };
    geist_token_t tokens_with_awq[N_DECODE];
    if (decode_tokens(model, be, &opts_with_awq, tokens_with_awq, N_DECODE) != 0) {
        fprintf(stderr, "AWQ decode failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("with-AWQ tokens:");
    for (int i = 0; i < N_DECODE; i++)
        printf(" %d", tokens_with_awq[i]);
    printf("\n");

    /* Validity: no decode returned -1 and tokens are in plausible vocab range. */
    int fails = 0;
    for (int i = 0; i < N_DECODE; i++) {
        if (tokens_no_awq[i] < 0 || tokens_with_awq[i] < 0) {
            fprintf(stderr,
                    "invalid token at [%d]: no_awq=%d with_awq=%d\n",
                    i,
                    tokens_no_awq[i],
                    tokens_with_awq[i]);
            fails++;
        }
    }

    /* Effect: at least one token must differ (otherwise AWQ silently did
     * nothing). Pure equality would mean either the scales were all 1.0
     * or the wiring is broken. */
    int n_same = 0;
    for (int i = 0; i < N_DECODE; i++) {
        if (tokens_no_awq[i] == tokens_with_awq[i])
            n_same++;
    }
    if (n_same == N_DECODE) {
        fprintf(stderr, "AWQ had no effect on decoded tokens\n");
        fails++;
    }

    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS: AWQ load + apply changes decode output without crashing\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
