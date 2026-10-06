#define _POSIX_C_SOURCE 200809L /* nanosleep */
/*
 * test_idle_cpu_int — a model that has finished decoding leaves the CPU
 * idle (#651).
 *
 * Loads a model on the first CPU backend linked, decodes 16 tokens, sleeps
 * 1 s, then measures the process's CPU time (user + system) over a further
 * 2 s of sleep. More than 10 % of one core fails: OpenMP workers still
 * spinning after the last parallel region (OMP_WAIT_POLICY=active with an
 * unbounded KMP_BLOCKTIME measured ~440 % here).
 *
 * Model: GEIST_LLAMA_GGUF_PATH, else gguf_artifacts/smollm2-360m-instruct-
 * q8_0.gguf, else GEIST_GGUF_PATH (the macOS CI leg's qwen3.5 fixture).
 * Skips without one. An explicit KMP_BLOCKTIME / OMP_WAIT_POLICY in the
 * environment is the caller's choice and skips the check.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_LLAMA_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    const char *dflt = "gguf_artifacts/smollm2-360m-instruct-q8_0.gguf";
    FILE       *f    = fopen(dflt, "rb");
    if (f != nullptr) {
        fclose(f);
        return dflt;
    }
    env = getenv("GEIST_GGUF_PATH");
    return env != nullptr && env[0] != '\0' ? env : nullptr;
}

static double cpu_seconds(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (double) (r.ru_utime.tv_sec + r.ru_stime.tv_sec) +
           (double) (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6;
}

static void sleep_s(time_t s) {
    struct timespec t = {.tv_sec = s};
    while (nanosleep(&t, &t) != 0) {
    }
}

int main(void) {
    if (getenv("KMP_BLOCKTIME") != nullptr || getenv("OMP_WAIT_POLICY") != nullptr) {
        GEIST_SKIP("KMP_BLOCKTIME / OMP_WAIT_POLICY set by the caller");
    }
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no model. Run `make fetch-llama-model`, or set GEIST_LLAMA_GGUF_PATH");
    }

    static const char *const backends[] = {"cpu_neon", "cpu_x86", "cpu_scalar"};
    struct geist_backend    *be         = nullptr;
    for (size_t i = 0; i < sizeof backends / sizeof backends[0] && be == nullptr; i++) {
        if (geist_backend_create(backends[i], nullptr, nullptr, &be) != GEIST_OK)
            be = nullptr;
    }
    if (be == nullptr) {
        fprintf(stderr, "no CPU backend: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct geist_model   *m  = nullptr;
    struct geist_session *s  = nullptr;
    int                   rc = GEIST_TEST_ERROR;
    if (geist_model_load(path, be, &m) != GEIST_OK) {
        fprintf(stderr, "load %s failed\n", path);
        goto out;
    }
    struct geist_session_opts opts = {.top_p = 1.0f};
    if (geist_session_create(m, be, &opts, &s) != GEIST_OK) {
        fprintf(stderr, "session create failed\n");
        goto out;
    }
    const geist_token_t prompt[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    geist_token_t       tok       = 0;
    if (geist_session_prefill_tokens(s, 8, prompt) != GEIST_OK) {
        fprintf(stderr, "prefill failed\n");
        goto out;
    }
    for (int i = 0; i < 16; i++) {
        if (geist_session_decode_step(s, &tok) != GEIST_OK) {
            fprintf(stderr, "decode step %d failed\n", i);
            goto out;
        }
    }

    sleep_s(1);
    const double c0 = cpu_seconds();
    sleep_s(2);
    const double used = cpu_seconds() - c0;
    const double pct  = used / 2.0 * 100.0;
    printf("%s on %s: %.1f %% of one core over 2 s idle\n", path, geist_backend_name(be), pct);
    rc = geist_expect(pct <= 10.0, "idle CPU after decode <= 10 % of one core") == 0
                 ? GEIST_TEST_PASS
                 : GEIST_TEST_FAIL;
    if (rc == GEIST_TEST_PASS)
        printf("PASS: the CPU goes idle after decode\n");

out:
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return rc;
}
