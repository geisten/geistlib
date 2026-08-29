/*
 * test_pin_prefix_int — verifies geist_session_pin_prefix() semantics.
 *
 * Workflow under test:
 *   pin_prefix(P)  → cache holds P (kv_len = prefix_length)
 *   prefill(Q)     → cache holds P + Q
 *   decode N       → token sequence A
 *   reset()        → kv_len truncates to prefix_length (P stays in cache)
 *   prefill(Q)     → cache holds P + Q again
 *   decode N       → token sequence B
 *
 * A and B must be identical: proves the prefix KV state survived reset
 * and the second prefill(Q) appended only Q (not re-running P).
 *
 * Phase B-4f smoke test. SKIPs cleanly if no GGUF model is available.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_DECODE 4

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
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    struct geist_session     *sess = nullptr;
    s                              = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Prefix and query are arbitrary but plausible token ids. Use ids
     * inside the Gemma 4 vocab to avoid OOB indices in lm_head. */
    const geist_token_t prefix_ids[] = {2, 105, 106, 107}; /* <bos>+specials */
    const size_t        prefix_n     = sizeof(prefix_ids) / sizeof(prefix_ids[0]);
    const geist_token_t query_ids[]  = {1408, 236743, 244549};
    const size_t        query_n      = sizeof(query_ids) / sizeof(query_ids[0]);

    s = geist_session_pin_prefix(sess, prefix_n, prefix_ids);
    if (s == GEIST_E_UNSUPPORTED) {
        printf("SKIP: arch does not support pin_prefix\n");
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    if (s != GEIST_OK) {
        fprintf(stderr,
                "pin_prefix failed: %s — %s\n",
                geist_status_to_string(s),
                geist_session_errmsg(sess));
        goto fail;
    }
    printf("pinned %zu-token prefix\n", prefix_n);

    /* Pass A: prefill(query), decode N. */
    s = geist_session_prefill_tokens(sess, query_n, query_ids);
    if (s != GEIST_OK) {
        fprintf(stderr, "pass A prefill_tokens failed: %s\n", geist_status_to_string(s));
        goto fail;
    }
    geist_token_t tokens_a[N_DECODE];
    for (int i = 0; i < N_DECODE; i++) {
        s = geist_session_decode_step(sess, &tokens_a[i]);
        if (s != GEIST_OK) {
            fprintf(stderr, "pass A decode[%d] failed\n", i);
            goto fail;
        }
    }
    printf("pass A:");
    for (int i = 0; i < N_DECODE; i++)
        printf(" %d", tokens_a[i]);
    printf("\n");

    /* Reset must truncate to prefix_length (NOT 0); the prefix KV stays. */
    s = geist_session_reset(sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "reset failed: %s\n", geist_status_to_string(s));
        goto fail;
    }

    /* Pass B: same prefill(query) — must append on top of pinned prefix. */
    s = geist_session_prefill_tokens(sess, query_n, query_ids);
    if (s != GEIST_OK) {
        fprintf(stderr, "pass B prefill_tokens failed: %s\n", geist_status_to_string(s));
        goto fail;
    }
    geist_token_t tokens_b[N_DECODE];
    for (int i = 0; i < N_DECODE; i++) {
        s = geist_session_decode_step(sess, &tokens_b[i]);
        if (s != GEIST_OK) {
            fprintf(stderr, "pass B decode[%d] failed\n", i);
            goto fail;
        }
    }
    printf("pass B:");
    for (int i = 0; i < N_DECODE; i++)
        printf(" %d", tokens_b[i]);
    printf("\n");

    int fails = 0;
    for (int i = 0; i < N_DECODE; i++) {
        if (tokens_a[i] != tokens_b[i]) {
            fprintf(stderr, "MISMATCH at [%d]: A=%d B=%d\n", i, tokens_a[i], tokens_b[i]);
            fails++;
        }
    }

    /* Failure propagation: pinning past max_seq_len must report the error,
     * not a silent GEIST_OK with an empty cache. Runs last — a failed pin
     * leaves the session truncated. */
    static geist_token_t too_many[1025];
    for (size_t i = 0; i < sizeof(too_many) / sizeof(too_many[0]); i++) {
        too_many[i] = 2;
    }
    s = geist_session_pin_prefix(sess, sizeof(too_many) / sizeof(too_many[0]), too_many);
    if (s == GEIST_OK) {
        fprintf(stderr, "MISMATCH: pin_prefix of 1025 tokens into a 1024-slot KV returned OK\n");
        fails++;
    } else {
        printf("overflow pin rejected: %s — %s\n",
               geist_status_to_string(s),
               geist_session_errmsg(sess));
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS: pin_prefix + reset preserves prefix KV across replay\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;

fail:
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_FAIL;
}
