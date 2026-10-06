/*
 * test_session_sampling_int — exercises the geist_session_opts sampling
 * fields (temperature, top_p, top_k, random_seed) end-to-end.
 *
 *   - Greedy (temperature=0) → must match the canonical session
 *     argmax behaviour (decoded tokens identical run-to-run).
 *   - Top-k temperature sample → at least one decode within 8 must
 *     differ from greedy (proves sampling is engaged and reads opts).
 *   - Same seed twice → identical token sequence (RNG is deterministic).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_DECODE 8

/* Returns GEIST_OK on success and fills out_tokens[N_DECODE]; otherwise the
 * failing status. GEIST_E_NOT_FOUND from set_prompt means no usable tokenizer
 * (no tokenizer.bin, and the GGUF tokenizer is a non-gpt2 SPM model geist's
 * encode path doesn't implement) — the caller turns that into a clean SKIP. */
static enum geist_status run_decode(const char                      *model_path,
                                    struct geist_backend            *be,
                                    const struct geist_session_opts *opts,
                                    geist_token_t                    out_tokens[static N_DECODE]) {
    struct geist_model *model = nullptr;
    enum geist_status   s     = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_status_to_string(s));
        return s;
    }
    struct geist_session *sess = nullptr;
    s                          = geist_session_create(model, be, opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed: %s\n", geist_status_to_string(s));
        geist_model_destroy(model);
        return s;
    }
    s = geist_session_set_prompt(sess, "Hello");
    if (s != GEIST_OK) {
        if (s != GEIST_E_NOT_FOUND) {
            fprintf(stderr,
                    "set_prompt failed: %s — %s\n",
                    geist_status_to_string(s),
                    geist_session_errmsg(sess));
        }
        geist_session_destroy(sess);
        geist_model_destroy(model);
        return s;
    }
    for (int i = 0; i < N_DECODE; i++) {
        s = geist_session_decode_step(sess, &out_tokens[i]);
        if (s != GEIST_OK) {
            fprintf(stderr, "decode_step[%d] failed: %s\n", i, geist_status_to_string(s));
            geist_session_destroy(sess);
            geist_model_destroy(model);
            return s;
        }
    }
    geist_session_destroy(sess);
    geist_model_destroy(model);
    return GEIST_OK;
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

    int fails = 0;

    /* ---- Test 1: greedy (temperature=0) reproduces canonical ---- */
    geist_token_t             greedy_a[N_DECODE], greedy_b[N_DECODE];
    struct geist_session_opts greedy_opts = {.temperature = 0.0f};
    enum geist_status         rc          = run_decode(model_path, be, &greedy_opts, greedy_a);
    if (rc == GEIST_E_NOT_FOUND) {
        printf("SKIP: no tokenizer for set_prompt path "
               "(needs tokenizer.bin or a gpt2-BPE GGUF)\n");
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    if (rc != GEIST_OK) {
        goto fail;
    }
    if (run_decode(model_path, be, &greedy_opts, greedy_b) != GEIST_OK) {
        goto fail;
    }
    if (memcmp(greedy_a, greedy_b, sizeof greedy_a) != 0) {
        fprintf(stderr, "FAIL: greedy is non-deterministic\n");
        fails++;
    } else {
        printf("greedy[8] :");
        for (int i = 0; i < N_DECODE; i++)
            printf(" %d", (int) greedy_a[i]);
        printf("\n");
    }

    /* ---- Test 2: high-temperature top-k sample diverges from greedy ----
     *
     * Use a very hot temperature so the softmax flattens enough for top-k
     * to actually pick something other than argmax. The "Hello" prompt
     * has a strongly peaked distribution on 9259 so we need temperature
     * well above 1.0 to break out of it deterministically. */
    geist_token_t             sampled_a[N_DECODE], sampled_b[N_DECODE];
    struct geist_session_opts sample_opts = {
            .temperature = 10.0f,
            .top_k       = 200,
            .top_p       = 1.0f,
            .random_seed = 0xDEADBEEFu,
    };
    if (run_decode(model_path, be, &sample_opts, sampled_a) != GEIST_OK) {
        goto fail;
    }
    if (memcmp(sampled_a, greedy_a, sizeof sampled_a) == 0) {
        fprintf(stderr,
                "FAIL: top-k 200 / temp 10.0 produced the greedy "
                "sequence — sampler not engaged?\n");
        fails++;
    } else {
        printf("sample[8] :");
        for (int i = 0; i < N_DECODE; i++)
            printf(" %d", (int) sampled_a[i]);
        printf("  (seed=0x%llx)\n", (unsigned long long) sample_opts.random_seed);
    }

    /* ---- Test 3: same seed → identical sequence ---- */
    if (run_decode(model_path, be, &sample_opts, sampled_b) != GEIST_OK) {
        goto fail;
    }
    if (memcmp(sampled_a, sampled_b, sizeof sampled_a) != 0) {
        fprintf(stderr, "FAIL: same-seed runs produced different sequences\n");
        fprintf(stderr, "  A:");
        for (int i = 0; i < N_DECODE; i++)
            fprintf(stderr, " %d", (int) sampled_a[i]);
        fprintf(stderr, "\n  B:");
        for (int i = 0; i < N_DECODE; i++)
            fprintf(stderr, " %d", (int) sampled_b[i]);
        fprintf(stderr, "\n");
        fails++;
    } else {
        printf("seed-determinism: PASS (same 8 tokens with seed=0x%llx)\n",
               (unsigned long long) sample_opts.random_seed);
    }

    geist_backend_destroy(be);
    if (fails == 0) {
        printf("PASS: greedy + top-k/top-p + seed determinism all work\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;

fail:
    geist_backend_destroy(be);
    return GEIST_TEST_FAIL;
}
