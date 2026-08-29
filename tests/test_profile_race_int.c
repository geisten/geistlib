/*
 * test_profile_race_int — the shared diagnostics and the lazy policy
 * caches, exercised concurrently (issue #335).
 *
 * test_multi_session_parallel_int already runs concurrent sessions, but
 * with every environment-gated path at its default and profiling off —
 * which is exactly the state in which these particular races do not
 * happen. The forward profiler's counters are file-scope, shared by all
 * sessions, and were incremented with a plain `+=`; its `registered` flag
 * was written under a mutex and read outside one; and the first-use
 * caches in the NEON Q4_K/Q6_K kernels, the vision and audio softmaxes,
 * and the speculative decoder were plain ints initialized by whichever
 * thread arrived first.
 *
 * So this test arms all of it BEFORE any session exists — profiling on,
 * every policy cache explicitly set — and then has several threads reach
 * the caches for the first time simultaneously. Under MODE=tsan that is
 * the race report; in any other mode it is a cheap smoke test that the
 * armed paths still produce answers.
 *
 * The env values chosen are the defaults, so behaviour is unchanged and
 * only the initialization path is under test.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_THREADS 4
#define PROMPT_LEN 8
/* Short on purpose: this test is about first-use initialization racing,
 * not throughput. The window it needs is the first forward pass. */
#define N_DECODE 4

static const geist_token_t PROMPTS[N_THREADS][PROMPT_LEN] = {
        {12, 345, 678, 910, 1112, 1314, 1516, 1718},
        {21, 543, 876, 109, 2111, 4131, 6151, 8171},
        {33, 999, 111, 222, 3333, 1044, 2055, 3066},
        {47, 250, 460, 870, 1180, 2290, 3300, 4410},
};

struct worker {
    struct geist_session *sess;
    const geist_token_t  *prompt;
    int                   failed;
};

static void *worker_main(void *arg) {
    struct worker *w = arg;
    if (geist_session_prefill_tokens(w->sess, PROMPT_LEN, w->prompt) != GEIST_OK) {
        w->failed = 1;
        return nullptr;
    }
    for (int i = 0; i < N_DECODE; i++) {
        geist_token_t t = 0;
        if (geist_session_decode_step(w->sess, &t) != GEIST_OK) {
            w->failed = 2;
            return nullptr;
        }
    }
    return nullptr;
}

/* Arm every lazily-initialized switch the forward path consults. Set
 * before the first session so the caches are still cold when the threads
 * start; the values are the defaults, so nothing about the computation
 * changes — only that the initialization happens under contention. */
static void arm_env_gated_paths(void) {
    static const char *const VARS[][2] = {
            {"GEIST_PROFILE_PREFILL", "1"}, /* forward + per-stage profilers */
            {"GEIST_PROFILE_QUANT", "1"},   /* NEON activation-quant profiler */
            {"GEIST_Q4K_PACK_ACT", "1"},    /* Q4_K prefill activation packing */
            {"GEIST_Q6K_PACK_ACT", "1"},    /* Q6_K prefill activation packing */
            {"GEIST_PP", "0"},              /* Q4_K predecode parallel-for */
            {"GEIST_FAST_TANH", "0"},       /* vision/audio softmax exp path */
            {"GEIST_SPEC_MIN_L", "2"},      /* speculative min match length */
    };
    for (size_t i = 0; i < sizeof(VARS) / sizeof(VARS[0]); i++) {
        setenv(VARS[i][0], VARS[i][1], 1);
    }
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    arm_env_gated_paths();

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Sessions are created serially — the threading contract is one
     * thread per session in steady state, not concurrent setup. */
    struct geist_session_opts opts               = {.max_seq_len = 512};
    struct worker             workers[N_THREADS] = {0};
    int                       created            = 0;
    for (int i = 0; i < N_THREADS; i++) {
        if (geist_session_create(model, be, &opts, &workers[i].sess) != GEIST_OK) {
            break;
        }
        workers[i].prompt = PROMPTS[i];
        created++;
    }
    if (created != N_THREADS) {
        fprintf(stderr, "only %d/%d sessions created\n", created, N_THREADS);
        for (int i = 0; i < created; i++) {
            geist_session_destroy(workers[i].sess);
        }
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    pthread_t tids[N_THREADS];
    for (int i = 0; i < N_THREADS; i++) {
        if (pthread_create(&tids[i], nullptr, worker_main, &workers[i]) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            return GEIST_TEST_ERROR;
        }
    }
    int fails = 0;
    for (int i = 0; i < N_THREADS; i++) {
        pthread_join(tids[i], nullptr);
        if (workers[i].failed != 0) {
            fprintf(stderr, "worker %d failed (%d)\n", i, workers[i].failed);
            fails++;
        }
    }

    for (int i = 0; i < N_THREADS; i++) {
        geist_session_destroy(workers[i].sess);
    }
    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (fails != 0) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: %d concurrent sessions with profiling and every env-gated policy "
           "cache armed (a profile dump follows at exit)\n",
           N_THREADS);
    return GEIST_TEST_PASS;
}
