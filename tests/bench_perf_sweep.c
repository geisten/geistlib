/*
 * tests/bench_perf_sweep — sequence-length-sweep perf bench for geist.
 *
 * For each (seq_len, decode_n) point in the sweep:
 *   1. Reset session, build a `seq_len`-length pseudo-random prompt.
 *   2. Prefill it — measure wall-clock ms.
 *   3. Decode `decode_n` tokens — measure wall-clock ms.
 *   4. Emit one JSON line per row.
 *
 * Args:
 *   --gguf PATH                GGUF model path (overrides GEIST_GGUF_PATH /
 *                              fixture lookup; consumed by bench_standard.py)
 *   --seq-lens 32,128,...      comma-separated prefill lengths (required)
 *   --decode-n N               decode steps after each prefill (default 64)
 *   --warmup N                 warmup prefill+decode run before any measurement
 *                              (default 64); this run is DISCARDED — it pages the
 *                              weights resident, resolves backend kernels, and
 *                              spins up the OpenMP pool so timings reflect steady
 *                              state, not cold-start.
 *   --repeats N                measured repeats per seq_len; JSON core metrics
 *                              report the MEAN over the repeats, plus best/worst
 *                              (default 10)
 *   --m-max N                  session prefill chunk cap (default arch cap)
 *   --vocab N                  pseudo-random token range upper bound
 *                              (default 32000 — works for Llama2 SP and Llama3 BPE)
 *   --threads N                informational only; the active thread count is
 *                              controlled by OMP_NUM_THREADS. We echo it back
 *                              in the JSONL for reproducibility.
 *   --emit-jsonl               accepted alias; output is JSONL by default.
 *
 * Output: one JSON object per line on stdout, e.g.
 *   {"seq_len":256,"decode_n":64,"prefill_ms":1234.5,"decode_ms":890.1,
 *    "total_ms":2124.6,"prefill_tps":207.4,"decode_tps":71.9,
 *    "total_tps":150.6,"rss_mb":4321.2,"threads":4}
 *
 * Bench-only; not a correctness test. Exits 77 (SKIP) if no GGUF found.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#else
#include <sys/resource.h>
#endif

static double process_rss_mb(void) {
#if defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t      count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info, &count) !=
        KERN_SUCCESS) {
        return 0.0;
    }
    return (double) info.resident_size / (1024.0 * 1024.0);
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return 0.0;
    }
    /* Linux reports ru_maxrss in KB; macOS in B (handled above). */
    return (double) ru.ru_maxrss / 1024.0;
#endif
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

static void parse_csv_ints(const char *csv, int *out, int max, int *n_out) {
    int         n = 0;
    const char *p = csv;
    while (*p && n < max) {
        char *end;
        long  v = strtol(p, &end, 10);
        if (end == p)
            break;
        out[n++] = (int) v;
        p        = end;
        if (*p == ',')
            p++;
    }
    *n_out = n;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *) a;
    const double db = *(const double *) b;
    return (da > db) - (da < db);
}

static double mean_of(const double *v, int n) {
    if (n <= 0)
        return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += v[i];
    return sum / (double) n;
}

int main(int argc, char **argv) {
    int         seq_lens[16];
    int         n_seq_lens = 0;
    int         decode_n   = 64;
    int         warmup     = 64;
    int         repeats    = 10;
    int         m_max      = 0;
    int         vocab_cap  = 32000;
    int         threads    = 0; /* informational; the runtime reads OMP_NUM_THREADS. */
    const char *gguf_arg   = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq-lens") == 0 && i + 1 < argc) {
            parse_csv_ints(argv[++i], seq_lens, 16, &n_seq_lens);
        } else if (strcmp(argv[i], "--decode-n") == 0 && i + 1 < argc) {
            decode_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--m-max") == 0 && i + 1 < argc) {
            m_max = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_cap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) {
            gguf_arg = argv[++i];
        } else if (strcmp(argv[i], "--emit-jsonl") == 0) {
            /* Accepted alias; stdout is already JSONL. */
        }
    }
    if (n_seq_lens == 0) {
        fprintf(stderr,
                "usage: %s [--gguf PATH] --seq-lens 32,128,256,512,1024,2048 "
                "[--decode-n 64] [--warmup 64] [--repeats 10] [--vocab 32000] "
                "[--threads N] [--emit-jsonl]\n",
                argv[0]);
        return GEIST_TEST_ERROR;
    }
    if (repeats < 1) {
        repeats = 1;
    }

    const char *model_path = gguf_arg;
    if (model_path == nullptr) {
        model_path = geist_test_find_gguf();
        GEIST_SKIP_IF(model_path == nullptr,
                      "GGUF model not found. Set GEIST_GGUF_PATH, pass "
                      "--gguf PATH, or place model in ./, ../models/");
    }

    /* This benchmark measures text decoder throughput. Loading optional
     * audio/vision towers can add hundreds of MB and OOM 4 GB Pi 5 runs
     * without affecting the metric being measured. */
    setenv("GEIST_TEXT_ONLY", "1", 0);

    /* "auto" = registry preference order (cpu_neon/cpu_x86 before cpu_scalar).
     * The old default of "cpu_neon" + silent cpu_scalar fallback measured the
     * scalar path on x86 builds unless GEIST_BENCH_BACKEND was set (#102). */
    const char *backend_name = getenv("GEIST_BENCH_BACKEND");
    if (backend_name == nullptr || backend_name[0] == '\0') {
        backend_name = "auto";
    }
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create(backend_name, nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    fprintf(stderr, "[bench] backend: %s\n", geist_backend_name(be));

    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Find the maximum seq_len in the sweep so we size the session right. */
    int max_seq = 0;
    for (int i = 0; i < n_seq_lens; i++) {
        if (seq_lens[i] > max_seq)
            max_seq = seq_lens[i];
    }
    const int                 session_seq = max_seq + decode_n + warmup + 32;
    struct geist_session_opts opts        = {
            .max_seq_len = (size_t) session_seq,
            .temperature = 0.0f,
            .m_max       = m_max > 0 ? (size_t) m_max : 0,
    };
    struct geist_session *sess = nullptr;
    s                          = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Pre-generate token buffer for the largest prefill. Pseudo-random in
     * [1, vocab_cap) — avoid token 0 (often <pad> / <unk>). */
    geist_token_t *ids = (geist_token_t *) malloc((size_t) max_seq * sizeof(*ids));
    for (int i = 0; i < max_seq; i++) {
        ids[i] = 1 + (geist_token_t) ((i * 2654435761u) % (unsigned) (vocab_cap - 1));
    }

    /* WARMUP PHASE (discarded, not measured) — one prefill + a few decode
     * steps to page the weights resident, resolve backend kernels, and spin up
     * the OpenMP pool, so the measured repeats reflect steady state rather than
     * cold-start. Announced on stderr; stdout stays clean JSONL. */
    const int warmup_n = (warmup > max_seq) ? max_seq : warmup;
    if (warmup_n > 0) {
        fprintf(stderr,
                "[bench] warmup phase (discarded): %d-token prefill + 4 decode "
                "steps\n",
                warmup_n);
        (void) geist_session_reset(sess);
        (void) geist_session_prefill_tokens(sess, (size_t) warmup_n, ids);
        geist_token_t junk = 0;
        for (int i = 0; i < 4; i++)
            (void) geist_session_decode_step(sess, &junk);
    }
    fprintf(stderr,
            "[bench] measuring %d repeat(s) per seq_len; reporting the MEAN "
            "(with best/worst)\n",
            repeats);

    for (int idx = 0; idx < n_seq_lens; idx++) {
        const int n_p            = seq_lens[idx];
        double   *prefill_ms     = (double *) malloc((size_t) repeats * sizeof(*prefill_ms));
        double   *decode_ms      = (double *) malloc((size_t) repeats * sizeof(*decode_ms));
        double *total_ms_repeats = (double *) malloc((size_t) repeats * sizeof(*total_ms_repeats));
        if (prefill_ms == nullptr || decode_ms == nullptr || total_ms_repeats == nullptr) {
            free(prefill_ms);
            free(decode_ms);
            free(total_ms_repeats);
            fprintf(stderr, "alloc failed for repeats=%d\n", repeats);
            break;
        }

        int measured = 0;
        for (int r = 0; r < repeats; r++) {
            s = geist_session_reset(sess);
            if (s != GEIST_OK) {
                fprintf(stderr, "reset failed at seq_len=%d repeat=%d\n", n_p, r);
                continue;
            }

            const double t0        = monotonic_ms();
            s                      = geist_session_prefill_tokens(sess, (size_t) n_p, ids);
            const double t_prefill = monotonic_ms() - t0;
            if (s != GEIST_OK) {
                fprintf(stderr, "prefill failed at seq_len=%d repeat=%d\n", n_p, r);
                continue;
            }

            const double  t1  = monotonic_ms();
            geist_token_t out = 0;
            for (int j = 0; j < decode_n; j++) {
                (void) geist_session_decode_step(sess, &out);
            }
            const double t_decode = monotonic_ms() - t1;

            prefill_ms[measured]       = t_prefill;
            decode_ms[measured]        = t_decode;
            total_ms_repeats[measured] = t_prefill + t_decode;
            measured++;
        }

        if (measured == 0) {
            free(prefill_ms);
            free(decode_ms);
            free(total_ms_repeats);
            continue;
        }

        /* Core metric is the MEAN over the measured repeats; sort only to pull
         * best (fastest) / worst (slowest) for the spread. */
        const double t_prefill = mean_of(prefill_ms, measured);
        const double t_decode  = mean_of(decode_ms, measured);
        qsort(prefill_ms, (size_t) measured, sizeof(*prefill_ms), cmp_double);
        qsort(decode_ms, (size_t) measured, sizeof(*decode_ms), cmp_double);
        qsort(total_ms_repeats, (size_t) measured, sizeof(*total_ms_repeats), cmp_double);

        const double pre_tps         = (double) n_p * 1000.0 / t_prefill;
        const double dec_tps         = (double) decode_n * 1000.0 / t_decode;
        const double total_ms        = t_prefill + t_decode;
        const double total_tps       = (double) (n_p + decode_n) * 1000.0 / total_ms;
        const double pre_best        = prefill_ms[0];
        const double pre_worst       = prefill_ms[measured - 1];
        const double dec_best        = decode_ms[0];
        const double dec_worst       = decode_ms[measured - 1];
        const double total_best      = total_ms_repeats[0];
        const double total_worst     = total_ms_repeats[measured - 1];
        const double total_tps_best  = (double) (n_p + decode_n) * 1000.0 / total_best;
        const double total_tps_worst = (double) (n_p + decode_n) * 1000.0 / total_worst;

        printf("{\"seq_len\":%d,\"decode_n\":%d,"
               "\"prefill_ms\":%.2f,\"decode_ms\":%.2f,\"total_ms\":%.2f,"
               "\"prefill_tps\":%.3f,\"decode_tps\":%.3f,\"total_tps\":%.3f,"
               "\"prefill_ms_best\":%.2f,\"prefill_ms_worst\":%.2f,"
               "\"decode_ms_best\":%.2f,\"decode_ms_worst\":%.2f,"
               "\"total_ms_best\":%.2f,\"total_ms_worst\":%.2f,"
               "\"total_tps_best\":%.3f,\"total_tps_worst\":%.3f,"
               "\"agg\":\"mean\",\"repeats\":%d,\"warmup\":%d,"
               "\"rss_mb\":%.2f,\"threads\":%d}\n",
               n_p,
               decode_n,
               t_prefill,
               t_decode,
               total_ms,
               pre_tps,
               dec_tps,
               total_tps,
               pre_best,
               pre_worst,
               dec_best,
               dec_worst,
               total_best,
               total_worst,
               total_tps_best,
               total_tps_worst,
               measured,
               warmup_n,
               process_rss_mb(),
               threads);
        fflush(stdout);
        free(prefill_ms);
        free(decode_ms);
        free(total_ms_repeats);
    }

    free(ids);
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;
}
