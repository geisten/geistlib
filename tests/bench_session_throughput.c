/*
 * bench_session_throughput — measures end-to-end prefill + decode wall
 * through the new geist_session_* public API.
 *
 * Workload:
 *   - Load model (cold; once-per-run cost)
 *   - Prefill 64 tokens (warm-up)
 *   - Prefill 200 fresh tokens (measured)
 *   - Decode 50 tokens (measured)
 *
 * Reports: model_load ms, prefill 200t ms (and ms/tok), decode 50t ms
 * (and ms/tok), throughput tok/s for both phases.
 *
 * Compares directly to the existing eval_geist baseline since both
 * paths call into the same LM*; numbers should match within noise.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

/* Workload sizes are env-overridable so the run can be matched exactly to an
 * external reference (e.g. `llama-bench -p 512 -n 128`): set GEIST_BENCH_PP and
 * GEIST_BENCH_TG. Defaults keep the historical 200/50 workload. */
static size_t env_size(const char *name, size_t fallback) {
    const char *raw = getenv(name);
    if (raw == nullptr || raw[0] == '\0')
        return fallback;
    long v = atol(raw);
    return v > 0 ? (size_t) v : fallback;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    /* ---- Setup ---- */
    const char *backend_name = getenv("GEIST_BENCH_BACKEND");
    if (backend_name == nullptr || backend_name[0] == '\0') {
        backend_name = "cpu_neon";
    }
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create(backend_name, nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    double              t0    = monotonic_ms();
    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    double t_load             = monotonic_ms() - t0;
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Size the session window to the requested workload — positions past
     * max_seq_len are rejected with GEIST_E_TOO_MANY_TOKENS. Keep the
     * historical 2048 floor for the default 200/50 run. */
    const size_t prefill_n = env_size("GEIST_BENCH_PP", 200);
    const int    decode_n  = (int) env_size("GEIST_BENCH_TG", 50);
    const size_t need      = prefill_n + (size_t) decode_n;

    struct geist_session_opts opts = {.max_seq_len = need > 2048 ? need : 2048,
                                      .temperature = 0.0f};
    struct geist_session     *sess = nullptr;
    s                              = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* ---- Warm-up: prefill 64 tokens ---- */
    const size_t   warm_n   = 64;
    geist_token_t *warm_ids = malloc(warm_n * sizeof(geist_token_t));
    for (size_t i = 0; i < warm_n; i++) {
        warm_ids[i] = 2 + (geist_token_t) (i & 0xff);
    }
    s = geist_session_prefill_tokens(sess, warm_n, warm_ids);
    if (s != GEIST_OK) {
        fprintf(stderr, "warmup prefill failed: %s\n", geist_session_errmsg(sess));
        goto fail;
    }
    free(warm_ids);

    s = geist_session_reset(sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "reset failed\n");
        goto fail;
    }

    /* ---- Measured: prefill (GEIST_BENCH_PP, default 200) ---- */
    geist_token_t *prefill_ids = malloc(prefill_n * sizeof(geist_token_t));
    for (size_t i = 0; i < prefill_n; i++) {
        prefill_ids[i] = 2 + (geist_token_t) ((i * 37) & 0xff);
    }
    t0               = monotonic_ms();
    s                = geist_session_prefill_tokens(sess, prefill_n, prefill_ids);
    double t_prefill = monotonic_ms() - t0;
    if (s != GEIST_OK) {
        fprintf(stderr, "prefill failed: %s\n", geist_session_errmsg(sess));
        free(prefill_ids);
        goto fail;
    }
    free(prefill_ids);

    /* ---- Measured: decode (GEIST_BENCH_TG, default 50) ---- */
    geist_token_t tok = 0;
    t0                = monotonic_ms();
    for (int i = 0; i < decode_n; i++) {
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK) {
            fprintf(stderr, "decode_step[%d] failed\n", i);
            goto fail;
        }
    }
    double t_decode = monotonic_ms() - t0;

    /* ---- Report ---- */
    printf("Backend: %s  Model: %s\n", geist_backend_name(be), model_path);
    printf("  model_load:        %8.1f ms  (cold)\n", t_load);
    printf("  prefill (%zu tok):  %8.1f ms  =  %5.2f ms/tok  =  %6.1f tok/s\n",
           prefill_n,
           t_prefill,
           t_prefill / (double) prefill_n,
           (double) prefill_n / (t_prefill / 1e3));
    printf("  decode  (%d tok):   %8.1f ms  =  %5.2f ms/tok  =  %6.1f tok/s\n",
           decode_n,
           t_decode,
           t_decode / (double) decode_n,
           (double) decode_n / (t_decode / 1e3));

    /* Total (prefill+decode) end-to-end throughput — the user-facing number:
     * how fast a request of (PP prompt + TG generated) tokens completes.
     * total tok/s = (PP + TG) / (prefill_time + decode_time). This combines
     * the fast prefill and slow decode by the actual workload mix, so it is
     * directly comparable to `llama-bench -pg PP,TG`. */
    const double t_total = t_prefill + t_decode;
    const size_t n_total = prefill_n + (size_t) decode_n;
    printf("  total   (%zu tok):  %8.1f ms  =  %5.2f ms/tok  =  %6.1f tok/s  (pp%zu+tg%d, "
           "user-facing)\n",
           n_total,
           t_total,
           t_total / (double) n_total,
           (double) n_total / (t_total / 1e3),
           prefill_n,
           decode_n);

    /* Sampler-state sanity check via stats. */
    struct geist_session_stats stats;
    geist_session_get_stats(sess, &stats);
    if (stats.n_tokens_decoded != (uint64_t) decode_n) {
        fprintf(stderr,
                "stats.n_tokens_decoded = %llu, expected %d\n",
                (unsigned long long) stats.n_tokens_decoded,
                decode_n);
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;

fail:
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_FAIL;
}
