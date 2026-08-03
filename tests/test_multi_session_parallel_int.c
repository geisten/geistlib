/*
 * test_multi_session_parallel_int — concurrent sessions on one model.
 *
 * The session-threading refactor's contract: steady-state per-session ops
 * may run concurrently across different sessions of one model, one thread
 * per session. This test exercises exactly that — 4 sessions created
 * serially (setup is single-threaded per the contract), then 4 pthreads
 * each prefill a distinct pre-tokenized prompt and greedily decode
 * N_DECODE tokens; a second round resets and repeats, exercising
 * state_reset under concurrency.
 *
 * Two nets:
 *   - Under MODE=tsan, ThreadSanitizer flags any data race on the shared
 *     model state (weights, RoPE tables, spec sketch, backend).
 *   - In any mode, the outputs are compared token-for-token against a
 *     serial rerun of the same prompts on fresh sessions — greedy decode
 *     is deterministic, so cross-session KV contamination shows up as a
 *     mismatch even when it doesn't race.
 *
 * Prompts are pre-tokenized ids (fixed arrays): the tokenizers are
 * caller-serialized in v1 and stay out of the parallel section.
 *
 * SKIPs cleanly if no GGUF fixture is reachable.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h> /* prefill_tokens / decode_step */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_SESSIONS 4
#define N_DECODE 32
#define N_ROUNDS 2

/* Distinct short prompts, pre-tokenized with small in-vocab ids (every
 * model fixture has vocab >> 4096). The token VALUES only need to be
 * valid ids — the test checks determinism, not linguistics. */
static const geist_token_t PROMPTS[N_SESSIONS][8] = {
        {12, 345, 678, 910, 1112, 1314, 1516, 1718},
        {21, 543, 876, 109, 2111, 4131, 6151, 8171},
        {33, 999, 111, 222, 3333, 1044, 2055, 3066},
        {47, 250, 460, 870, 1180, 2290, 3300, 4410},
};
#define PROMPT_LEN 8

struct worker {
    struct geist_session *sess;
    const geist_token_t  *prompt;
    geist_token_t         out[N_ROUNDS][N_DECODE];
    int                   failed; /* nonzero = prefill/decode error */
};

static void *worker_main(void *arg) {
    struct worker *w = arg;
    for (int round = 0; round < N_ROUNDS; round++) {
        if (geist_session_reset(w->sess) != GEIST_OK) {
            w->failed = 1;
            return nullptr;
        }
        if (geist_session_prefill_tokens(w->sess, PROMPT_LEN, w->prompt) != GEIST_OK) {
            w->failed = 2;
            return nullptr;
        }
        for (int i = 0; i < N_DECODE; i++) {
            if (geist_session_decode_step(w->sess, &w->out[round][i]) != GEIST_OK) {
                w->failed = 3;
                return nullptr;
            }
        }
    }
    return nullptr;
}

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/bitnet_b1_58-large-TQ2_0.gguf",
            "gguf_artifacts/gemma4-e2b-Q4_K_M.gguf",
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

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        printf("SKIP: no GGUF fixture found (set GEIST_GGUF_PATH)\n");
        return GEIST_TEST_SKIP;
    }

    struct geist_backend *be = nullptr;
    /* Best available CPU backend, in kernel-speed order. cpu_x86 matters
     * twice on the TSan leg: it is the backend whose per-thread workspace
     * this test gates, and the scalar fallback it silently took before was
     * both blind to that code and slow enough to blow the job timeout. */
    enum geist_status s = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_x86", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_model *model = nullptr;
    s                         = geist_model_load(path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }

    const struct geist_session_opts opts  = {.max_seq_len = 256, .temperature = 0.0f};
    int                             fails = 0;

    /* ---- Parallel phase: sessions created serially, decoded in parallel. */
    struct worker workers[N_SESSIONS] = {0};
    for (int i = 0; i < N_SESSIONS; i++) {
        if (geist_session_create(model, be, &opts, &workers[i].sess) != GEIST_OK) {
            fprintf(stderr, "session_create %d failed\n", i);
            return GEIST_TEST_ERROR;
        }
        workers[i].prompt = PROMPTS[i];
    }
    pthread_t tids[N_SESSIONS];
    for (int i = 0; i < N_SESSIONS; i++) {
        pthread_create(&tids[i], nullptr, worker_main, &workers[i]);
    }
    for (int i = 0; i < N_SESSIONS; i++) {
        pthread_join(tids[i], nullptr);
    }
    for (int i = 0; i < N_SESSIONS; i++) {
        fails += geist_expect(workers[i].failed == 0, "parallel worker completed without error");
    }

    /* ---- Serial rerun on fresh sessions: greedy decode is deterministic,
     * so any cross-session contamination in the parallel phase shows up
     * as a token mismatch here. */
    for (int i = 0; i < N_SESSIONS; i++) {
        struct geist_session *ref = nullptr;
        if (geist_session_create(model, be, &opts, &ref) != GEIST_OK) {
            fprintf(stderr, "serial session_create %d failed\n", i);
            return GEIST_TEST_ERROR;
        }
        geist_token_t ref_out[N_DECODE];
        int           ok = 1;
        if (geist_session_prefill_tokens(ref, PROMPT_LEN, PROMPTS[i]) != GEIST_OK) {
            ok = 0;
        }
        for (int t = 0; ok && t < N_DECODE; t++) {
            if (geist_session_decode_step(ref, &ref_out[t]) != GEIST_OK) {
                ok = 0;
            }
        }
        fails += geist_expect(ok, "serial reference decode succeeded");
        if (ok) {
            for (int round = 0; round < N_ROUNDS; round++) {
                const bool same = memcmp(workers[i].out[round], ref_out, sizeof ref_out) == 0;
                if (!same) {
                    fprintf(stderr,
                            "session %d round %d: parallel tokens diverge from serial "
                            "(first at ",
                            i,
                            round);
                    for (int t = 0; t < N_DECODE; t++) {
                        if (workers[i].out[round][t] != ref_out[t]) {
                            fprintf(stderr,
                                    "pos %d: %d != %d)\n",
                                    t,
                                    workers[i].out[round][t],
                                    ref_out[t]);
                            break;
                        }
                    }
                }
                fails += geist_expect(same, "parallel decode bit-identical to serial");
            }
        }
        geist_session_destroy(ref);
    }

    for (int i = 0; i < N_SESSIONS; i++) {
        geist_session_destroy(workers[i].sess);
    }
    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("multi-session parallel: %d sessions × %d rounds × %d tokens, "
           "bit-identical to serial\n",
           N_SESSIONS,
           N_ROUNDS,
           N_DECODE);
    return GEIST_TEST_PASS;
}
