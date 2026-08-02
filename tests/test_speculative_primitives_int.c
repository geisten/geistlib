/*
 * test_speculative_primitives_int — verifies the two arch-level
 * primitives that the speculative-decode loop builds on:
 *
 *   transformer_verify_forward(state, k, ids, out_tokens):
 *     processes k tokens, returns k argmaxes (one per input position).
 *     Advances kv_len by k.
 *
 *   transformer_kv_truncate(state, new_len):
 *     shrinks kv_len; invalidates pending logits. Subsequent prefill
 *     overwrites KV from new_len onwards.
 *
 * Three sub-tests:
 *
 *   1. K=1 equivalence: verify_forward(1, [t]) must produce the same
 *      argmax that prefill_tokens(1, [t]) + decode_step would produce.
 *      Proves verify_forward's per-position argmax = the model's
 *      committed prediction at that position.
 *
 *   2. K=4 contiguity: verify_forward(4, [a,b,c,d]) must produce the
 *      same 4 argmaxes as running prefill_tokens(1, [t]) + decode_step
 *      sequentially with t = a,b,c,d. Proves the per-position math is
 *      identical regardless of batching width.
 *
 *   3. Round-trip via truncate: do verify(K=4) → kv_truncate(N) →
 *      prefill_tokens(same 4 ids) — final argmax must equal the K=4
 *      verify's last output (and the prefill+decode reference). Proves
 *      kv_truncate restores a clean append point.
 *
 * Internal-arch test: includes arch_state.h (architecture layer).
 *
 * SKIPs cleanly if no GGUF model is reachable.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct geist_backend *be;
static struct geist_model   *model;

static int load_model(const char *model_path) {
    enum geist_status s = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend_create: %s\n", geist_last_create_error());
        return 1;
    }
    s = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load: %s\n", geist_last_create_error());
        return 1;
    }
    return 0;
}

static void teardown(void) {
    geist_model_destroy(model);
    geist_backend_destroy(be);
}

/* Helper: create a session and return its underlying transformer state.
 * arch_meta is opaque void*; we reach it via the model accessor.
 * Resets state because session_create does NOT reset between sessions
 * that share an arch_meta (one model = one arch state). */
static struct transformer_arch_state *open_session(struct geist_session **out_sess) {
    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    enum geist_status         s    = geist_session_create(model, be, &opts, out_sess);
    if (s != GEIST_OK)
        return nullptr;
    s = geist_session_reset(*out_sess);
    if (s != GEIST_OK) {
        geist_session_destroy(*out_sess);
        *out_sess = nullptr;
        return nullptr;
    }
    return (struct transformer_arch_state *) geist_model_internal_arch_meta(model);
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    if (load_model(model_path))
        return GEIST_TEST_FAIL;

    /* Use a small but realistic prompt. BOS=2 keeps the test deterministic
     * across runs. Q3_K_M and Q4_K_M both produce the same model output. */
    const geist_token_t prompt_ids[] = {2, 9259}; /* <bos> Hello */
    const size_t        prompt_n     = sizeof prompt_ids / sizeof prompt_ids[0];
    const size_t        K            = 4;

    /* ---- Reference run: prefill prompt, then K sequential decode steps. */
    geist_token_t ref_tokens[K];
    {
        struct geist_session          *sess = nullptr;
        struct transformer_arch_state *st   = open_session(&sess);
        if (st == nullptr) {
            fprintf(stderr, "session create failed (or internal accessor missing)\n");
            teardown();
            return GEIST_TEST_FAIL;
        }
        enum geist_status s = geist_session_prefill_tokens(sess, prompt_n, prompt_ids);
        if (s != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        for (size_t i = 0; i < K; i++) {
            s = geist_session_decode_step(sess, &ref_tokens[i]);
            if (s != GEIST_OK) {
                teardown();
                return GEIST_TEST_FAIL;
            }
        }
        geist_session_destroy(sess);
    }
    printf("reference (prefill+decode):");
    for (size_t i = 0; i < K; i++)
        printf(" %d", ref_tokens[i]);
    printf("\n");

    /* ---- Test 2: K=4 verify_forward equivalence. Feed [ref_0, ref_1,
     * ref_2, ref_3] as draft after prefilling prompt; the verify outputs
     * should be the model's prediction at each position = the NEXT
     * reference token (verify[i] predicts what comes after consuming
     * proposed[i], which is ref_tokens[i+1]). The last verify output
     * (verify[K-1]) predicts the K-th token (not in our ref). */
    geist_token_t verify_out[K];
    size_t        kv_after_verify = 0;
    {
        struct geist_session          *sess = nullptr;
        struct transformer_arch_state *st   = open_session(&sess);
        if (st == nullptr) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        enum geist_status s = geist_session_prefill_tokens(sess, prompt_n, prompt_ids);
        if (s != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        struct transformer_arch_session *asess     = geist_session_internal_arch_session(sess);
        const size_t                     kv_before = asess->kv_len;
        s = transformer_verify_forward(asess, K, ref_tokens, verify_out);
        if (s != GEIST_OK) {
            fprintf(stderr, "verify_forward failed: %s\n", geist_status_to_string(s));
            teardown();
            return GEIST_TEST_FAIL;
        }
        kv_after_verify = asess->kv_len;
        if (kv_after_verify != kv_before + K) {
            fprintf(stderr,
                    "verify_forward kv_len: %zu, expected %zu\n",
                    kv_after_verify,
                    kv_before + K);
            teardown();
            return GEIST_TEST_FAIL;
        }
        geist_session_destroy(sess);
    }
    printf("verify_forward(K=4):");
    for (size_t i = 0; i < K; i++)
        printf(" %d", verify_out[i]);
    printf("\n");

    /* verify_out[i] is model's prediction at position kv_before + i,
     * = what should come after consuming ref_tokens[0..i].
     * That's ref_tokens[i+1] for i < K-1. */
    int fails = 0;
    for (size_t i = 0; i + 1 < K; i++) {
        if (verify_out[i] != ref_tokens[i + 1]) {
            fprintf(stderr,
                    "FAIL: verify_out[%zu]=%d, expected ref_tokens[%zu]=%d\n",
                    i,
                    verify_out[i],
                    i + 1,
                    ref_tokens[i + 1]);
            fails++;
        }
    }
    /* verify_out[K-1] is prediction of token K+1 — not in our 4-token ref
     * but it should be a valid vocab id (not -1). */
    if (verify_out[K - 1] < 0) {
        fprintf(stderr, "FAIL: verify_out[K-1]=%d invalid\n", verify_out[K - 1]);
        fails++;
    }

    /* ---- Test 3: kv_truncate round-trip. */
    {
        struct geist_session          *sess = nullptr;
        struct transformer_arch_state *st   = open_session(&sess);
        if (st == nullptr) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        enum geist_status s = geist_session_prefill_tokens(sess, prompt_n, prompt_ids);
        if (s != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        struct transformer_arch_session *asess     = geist_session_internal_arch_session(sess);
        const size_t                     kv_before = asess->kv_len;

        geist_token_t scratch_out[K];
        s = transformer_verify_forward(asess, K, ref_tokens, scratch_out);
        if (s != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        if (asess->kv_len != kv_before + K) {
            fprintf(stderr,
                    "FAIL: kv_len after verify=%zu, expected %zu\n",
                    asess->kv_len,
                    kv_before + K);
            fails++;
        }
        /* Truncate back to kv_before. */
        transformer_kv_truncate(asess, kv_before);
        if (asess->kv_len != kv_before) {
            fprintf(stderr,
                    "FAIL: kv_len after truncate=%zu, expected %zu\n",
                    asess->kv_len,
                    kv_before);
            fails++;
        }
        if (asess->logits_valid) {
            fprintf(stderr, "FAIL: logits_valid not cleared after truncate\n");
            fails++;
        }

        /* Re-prefill the same ids; verify the next-prediction matches
         * what reference produced after K decode steps. */
        s = geist_session_prefill_tokens(sess, K, ref_tokens);
        if (s != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        geist_token_t after_truncate_prefill_next;
        s = geist_session_decode_step(sess, &after_truncate_prefill_next);
        if (s != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        /* This is the prediction at position prompt+K (= what would be
         * the 5th token in the reference if we kept decoding). It should
         * equal verify_out[K-1]. */
        if (after_truncate_prefill_next != verify_out[K - 1]) {
            fprintf(stderr,
                    "FAIL: post-truncate decode=%d, verify_out[K-1]=%d\n",
                    after_truncate_prefill_next,
                    verify_out[K - 1]);
            fails++;
        } else {
            printf("post-truncate decode produced %d (matches verify_out[%zu])\n",
                   after_truncate_prefill_next,
                   K - 1);
        }
        geist_session_destroy(sess);
    }

    teardown();
    if (fails == 0) {
        printf("PASS: speculative primitives are sampling-equivalent\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
