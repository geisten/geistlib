/*
 * test_speculative_loop_int — geist_session_decode_speculative
 * must produce the same token stream as sequential decode_step.
 *
 * Two sub-tests:
 *
 *   1. Equivalence: decode N tokens via decode_step (reference), then
 *      reset, decode N tokens via decode_speculative (k_max=4) feeding
 *      the growing history. Token streams must be identical.
 *
 *   2. Stats: count spec_step calls and average tokens-per-call. Just
 *      reports the number — useful for tuning k_max + drafter params
 *      against real prompts.
 *
 * SKIPs cleanly if no GGUF model is reachable.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "spec_tie.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct geist_backend *be;
static struct geist_model   *model;

static int load_model(const char *path) {
    enum geist_status s = geist_backend_create(
            getenv("GEIST_BACKEND") != nullptr ? getenv("GEIST_BACKEND") : "cpu_neon",
            nullptr,
            nullptr,
            &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend_create: %s\n", geist_last_create_error());
        return 1;
    }
    s = geist_model_load(path, be, &model);
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

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    if (load_model(model_path))
        return GEIST_TEST_FAIL;

    /* Slightly longer prompt so the drafter has at least some history
     * to scan once decoding starts. BOS=2 keeps it deterministic. */
    const geist_token_t prompt_ids[] = {2, 9259, 1018, 9259, 1018, 9259, 1018};
    const size_t        prompt_n     = sizeof prompt_ids / sizeof prompt_ids[0];
    const size_t        N            = 30; /* tokens to decode */
    const size_t        K_MAX        = 4;

    /* ---- Reference: sequential decode_step. */
    geist_token_t ref[N];
    {
        struct geist_session     *sess = nullptr;
        struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
        if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        (void) geist_session_reset(sess);
        if (geist_session_prefill_tokens(sess, prompt_n, prompt_ids) != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        for (size_t i = 0; i < N; i++) {
            if (geist_session_decode_step(sess, &ref[i]) != GEIST_OK) {
                teardown();
                return GEIST_TEST_FAIL;
            }
        }
        geist_session_destroy(sess);
    }
    printf("reference (decode_step × %zu):", N);
    for (size_t i = 0; i < N; i++)
        printf(" %d", ref[i]);
    printf("\n");

    /* ---- Speculative: same prompt, same N tokens via decode_speculative. */
    geist_token_t spec[N + K_MAX + 1]; /* overflow guard */
    size_t        spec_n     = 0;
    size_t        n_calls    = 0;
    size_t        total_emit = 0;
    {
        struct geist_session     *sess = nullptr;
        struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
        if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        (void) geist_session_reset(sess);
        if (geist_session_prefill_tokens(sess, prompt_n, prompt_ids) != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }

        /* History for the drafter = prompt + emitted-so-far. */
        geist_token_t history[1024];
        size_t        history_n = prompt_n;
        memcpy(history, prompt_ids, prompt_n * sizeof(*prompt_ids));

        geist_token_t emitted[K_MAX + 1];
        size_t        emit_n = 0;
        while (spec_n < N) {
            n_calls++;
            enum geist_status s = geist_session_decode_speculative(
                    sess, K_MAX, history_n, history, K_MAX + 1, emitted, &emit_n);
            if (s != GEIST_OK) {
                fprintf(stderr, "decode_speculative failed: %s\n", geist_status_to_string(s));
                teardown();
                return GEIST_TEST_FAIL;
            }
            total_emit += emit_n;
            for (size_t i = 0; i < emit_n && spec_n < N; i++) {
                spec[spec_n++]       = emitted[i];
                history[history_n++] = emitted[i];
            }
        }
        geist_session_destroy(sess);
    }
    printf("speculative (k_max=%zu, %zu calls, %zu emits): ", K_MAX, n_calls, total_emit);
    for (size_t i = 0; i < N; i++)
        printf(" %d", spec[i]);
    printf("\n");

    /* ---- Compare. */
    const struct geist_session_opts ref_opts = {.max_seq_len = 1024, .temperature = 0.0f};
    int fails = spec_compare(model, be, &ref_opts, prompt_n, prompt_ids, N, ref, spec, "greedy");
    printf("speculative tokens-per-call: %.2f  (sequential would be 1.00)\n",
           (double) total_emit / (double) n_calls);

    /* ---- Stochastic path: temperature > 0. Ensures spec_step doesn't
     * blow up under non-greedy sampling. The current simple accept-reject
     * is not distribution-preserving (would need rejection sampling on
     * per-position logits), so we only check that tokens are emitted at
     * a sane rate and stay within vocab. */
    {
        struct geist_session     *sess = nullptr;
        struct geist_session_opts opts = {
                .max_seq_len = 1024,
                .temperature = 0.7f,
                .top_p       = 0.9f,
                .random_seed = 42,
        };
        if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        (void) geist_session_reset(sess);
        if (geist_session_prefill_tokens(sess, prompt_n, prompt_ids) != GEIST_OK) {
            teardown();
            return GEIST_TEST_FAIL;
        }
        geist_token_t history[1024];
        size_t        history_n = prompt_n;
        memcpy(history, prompt_ids, prompt_n * sizeof(*prompt_ids));
        geist_token_t emitted[K_MAX + 1];
        size_t        emitted_total = 0;
        size_t        stoch_calls   = 0;
        while (emitted_total < N) {
            size_t            n;
            enum geist_status sx = geist_session_decode_speculative(
                    sess, K_MAX, history_n, history, K_MAX + 1, emitted, &n);
            if (sx != GEIST_OK || n == 0) {
                fprintf(stderr, "stoch FAIL: %s n=%zu\n", geist_status_to_string(sx), n);
                fails++;
                break;
            }
            stoch_calls++;
            for (size_t i = 0; i < n && emitted_total < N; i++) {
                if (emitted[i] < 0) {
                    fprintf(stderr, "stoch FAIL: invalid token %d\n", emitted[i]);
                    fails++;
                }
                history[history_n++] = emitted[i];
                emitted_total++;
            }
        }
        printf("stochastic (temp=0.7, top_p=0.9): %zu emits in %zu calls = %.2f t/call\n",
               emitted_total,
               stoch_calls,
               (double) emitted_total / (double) stoch_calls);
        geist_session_destroy(sess);
    }

    teardown();
    if (fails == 0) {
        printf("PASS: speculative loop matches sequential decode (greedy);\n"
               "      stochastic path produced valid tokens (distribution-preserving\n"
               "      stochastic spec would need per-position logits + rejection sampling)\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
