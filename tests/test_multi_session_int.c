/*
 * test_multi_session_int — verifies P1.2.f multi-session-per-model.
 *
 * Loads one model, creates two sessions on it with different prompts,
 * interleaves decode steps, and checks that each session keeps its own
 * KV state (decoding on one doesn't bleed into the other).
 *
 * Scenarios:
 *   1. session A on "Hello", session B on "Goodbye" — first decoded
 *      token from each must differ (different prompts produce
 *      different next tokens).
 *   2. Decode another step on A, then another on B, and re-decode
 *      everything from a fresh control session per prompt. Each
 *      session's tokens must match its own control stream — proving
 *      the interleave didn't corrupt state.
 *   3. Destroy B first, then A. Both must finish cleanly; the model
 *      stays loadable for further sessions.
 *
 * SKIPs cleanly if no GGUF model + tokenizer is reachable.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_DECODE 3

static enum geist_status decode_stream(struct geist_session *sess, geist_token_t out[N_DECODE]) {
    for (int i = 0; i < N_DECODE; i++) {
        const enum geist_status s = geist_session_decode_step(sess, &out[i]);
        if (s != GEIST_OK)
            return s;
    }
    return GEIST_OK;
}

/* Build a fresh single-session reference stream for a given prompt.
 * Used as the oracle for the interleaved test. */
static enum geist_status reference_stream(struct geist_model   *model,
                                          struct geist_backend *be,
                                          const char           *prompt,
                                          geist_token_t         out[N_DECODE]) {
    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    struct geist_session     *sess = nullptr;
    enum geist_status         s    = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK)
        return s;
    s = geist_session_set_prompt(sess, prompt);
    if (s == GEIST_OK) {
        s = decode_stream(sess, out);
    }
    geist_session_destroy(sess);
    return s;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    /* Best available CPU backend, in kernel-speed order (see the parallel
     * variant for why cpu_x86 belongs in this chain). */
    enum geist_status s = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_x86", nullptr, nullptr, &be);
    }
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
        fprintf(stderr,
                "model_load(%s) failed: %s — %s\n",
                model_path,
                geist_status_to_string(s),
                geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Reference streams — one session at a time, fully decoded.
     * These are the per-prompt deterministic stream the interleaved
     * runs must reproduce. */
    geist_token_t ref_A[N_DECODE], ref_B[N_DECODE];
    s = reference_stream(model, be, "Hello", ref_A);
    if (s == GEIST_E_NOT_FOUND) {
        printf("SKIP: tokenizer.bin not reachable for set_prompt path\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "reference_stream(A) failed: %s\n", geist_status_to_string(s));
        goto fail;
    }
    s = reference_stream(model, be, "Goodbye", ref_B);
    if (s != GEIST_OK) {
        fprintf(stderr, "reference_stream(B) failed: %s\n", geist_status_to_string(s));
        goto fail;
    }
    printf("reference A: [%d %d %d]\n", ref_A[0], ref_A[1], ref_A[2]);
    printf("reference B: [%d %d %d]\n", ref_B[0], ref_B[1], ref_B[2]);

    /* Streams-differ check: same model + different prompts must give
     * different continuations SOMEWHERE in the window — otherwise the
     * prompt isn't reaching the session's KV cache. Compared over the
     * whole window, not just token 0: a weak model may legitimately
     * predict the same first token for both prompts (bitnet-large gives
     * 29892 for both "Hello" and "Goodbye", then diverges). */
    if (memcmp(ref_A, ref_B, sizeof ref_A) == 0) {
        fprintf(stderr,
                "FAIL: ref_A == ref_B over all %d tokens — different prompts "
                "produced identical continuations (prompt didn't reach "
                "session KV)\n",
                N_DECODE);
        goto fail;
    }

    /* Interleaved run: two live sessions, alternating decode_step. */
    struct geist_session_opts opts   = {.max_seq_len = 1024, .temperature = 0.0f};
    struct geist_session     *sess_A = nullptr;
    struct geist_session     *sess_B = nullptr;

    s = geist_session_create(model, be, &opts, &sess_A);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create(A) failed: %s\n", geist_status_to_string(s));
        goto fail;
    }
    s = geist_session_create(model, be, &opts, &sess_B);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create(B) failed: %s\n", geist_status_to_string(s));
        geist_session_destroy(sess_A);
        goto fail;
    }

    /* Set prompts in the order A, B — exercising the attach-swap
     * between sessions. */
    s = geist_session_set_prompt(sess_A, "Hello");
    if (s != GEIST_OK) {
        fprintf(stderr, "set_prompt(A) failed\n");
        goto cleanup_fail;
    }
    s = geist_session_set_prompt(sess_B, "Goodbye");
    if (s != GEIST_OK) {
        fprintf(stderr, "set_prompt(B) failed\n");
        goto cleanup_fail;
    }

    /* Interleave: A.step, B.step, A.step, B.step, A.step, B.step. */
    geist_token_t live_A[N_DECODE], live_B[N_DECODE];
    int           fails = 0;
    for (int i = 0; i < N_DECODE; i++) {
        s = geist_session_decode_step(sess_A, &live_A[i]);
        if (s != GEIST_OK) {
            fprintf(stderr, "decode A[%d]\n", i);
            fails++;
            break;
        }
        s = geist_session_decode_step(sess_B, &live_B[i]);
        if (s != GEIST_OK) {
            fprintf(stderr, "decode B[%d]\n", i);
            fails++;
            break;
        }
    }
    printf("live    A: [%d %d %d]\n", live_A[0], live_A[1], live_A[2]);
    printf("live    B: [%d %d %d]\n", live_B[0], live_B[1], live_B[2]);

    /* Each interleaved stream must match its own reference exactly. If
     * the sessions shared state, decoding on B would corrupt A's KV
     * cache and live_A[1] / live_A[2] would diverge from ref_A. */
    for (int i = 0; i < N_DECODE; i++) {
        if (live_A[i] != ref_A[i]) {
            fprintf(stderr,
                    "FAIL: live_A[%d]=%d != ref_A[%d]=%d (session A's KV "
                    "got corrupted by interleaved B decode)\n",
                    i,
                    live_A[i],
                    i,
                    ref_A[i]);
            fails++;
        }
        if (live_B[i] != ref_B[i]) {
            fprintf(stderr,
                    "FAIL: live_B[%d]=%d != ref_B[%d]=%d (session B's KV "
                    "got corrupted by interleaved A decode)\n",
                    i,
                    live_B[i],
                    i,
                    ref_B[i]);
            fails++;
        }
    }

    /* Destroy B first; A should still be usable. */
    geist_session_destroy(sess_B);
    sess_B = nullptr;

    /* One more decode on A — confirms its arch_session survived B's
     * destroy. The token isn't checked against a reference (we'd need
     * a 4-token oracle); the success condition is that decode_step
     * returns GEIST_OK and produces a non-(-1) token. */
    geist_token_t tail = -1;
    s                  = geist_session_decode_step(sess_A, &tail);
    if (s != GEIST_OK || tail < 0) {
        fprintf(stderr,
                "FAIL: post-B-destroy decode on A failed (s=%s, tok=%d)\n",
                geist_status_to_string(s),
                tail);
        fails++;
    } else {
        printf("post-B-destroy decode on A produced token %d\n", tail);
    }

    geist_session_destroy(sess_A);
    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS: multi-session interleave keeps per-session KV "
               "independent (A=%s, B=%s)\n",
               "Hello",
               "Goodbye");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;

cleanup_fail:
    geist_session_destroy(sess_A);
    geist_session_destroy(sess_B);
fail:
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_FAIL;
}
