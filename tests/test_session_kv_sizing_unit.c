/*
 * test_session_kv_sizing_unit — a session's KV cache is sized from its own
 * max_seq_len, not the model's cap (#577).
 *
 * docs/API_CONTRACT.md promises that sessions taking a smaller cap than the
 * model get KV caches sized from it. On cpu_scalar, for each KV mode:
 *
 *   - a session with max_seq_len 64 adds far less resident memory than one
 *     at the model cap (the KV is memset at create, so it is resident at
 *     once; VmRSS, Linux only);
 *   - both sessions decode the same tokens;
 *   - the small session refuses a prompt past its 64 rows.
 */
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

enum { MODEL_CAP = 32768, SMALL_CAP = 64, STEPS = 6 };

/* VmRSS of this process in KiB, or -1 where /proc is not there. */
static long rss_kib(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (f == nullptr) {
        return -1;
    }
    char line[256];
    long kib = -1;
    while (fgets(line, sizeof line, f) != nullptr) {
        if (sscanf(line, "VmRSS: %ld kB", &kib) == 1) {
            break;
        }
    }
    fclose(f);
    return kib;
}

/* Create a session with `cap` rows, report the VmRSS it added, prefill one
 * token and decode STEPS greedy tokens. false on any failure. */
static bool run(struct geist_model   *m,
                struct geist_backend *be,
                enum geist_kv_mode    mode,
                size_t                cap,
                long                 *added_kib,
                geist_token_t         out[STEPS]) {
    const struct geist_session_opts o      = {.kv_mode = mode, .top_p = 1.0f, .max_seq_len = cap};
    struct geist_session           *s      = nullptr;
    static const geist_token_t      prompt = 7;
    const long                      before = rss_kib();
    bool                            ok     = geist_session_create(m, be, &o, &s) == GEIST_OK;
    const long                      after  = rss_kib();
    *added_kib                             = before >= 0 && after >= 0 ? after - before : -1;
    ok = ok && geist_session_prefill_tokens(s, 1, &prompt) == GEIST_OK;
    for (int i = 0; ok && i < STEPS; i++) {
        ok = geist_session_decode_step(s, &out[i]) == GEIST_OK;
    }
    if (s != nullptr) {
        geist_session_destroy(s);
    }
    return ok;
}

/* A prompt one row past the small cap is refused. */
static bool
refuses_overflow(struct geist_model *m, struct geist_backend *be, enum geist_kv_mode mode) {
    const struct geist_session_opts o = {.kv_mode = mode, .top_p = 1.0f, .max_seq_len = SMALL_CAP};
    struct geist_session           *s = nullptr;
    if (geist_session_create(m, be, &o, &s) != GEIST_OK) {
        return false;
    }
    geist_token_t prompt[SMALL_CAP + 1];
    for (size_t i = 0; i < SMALL_CAP + 1; i++) {
        prompt[i] = (geist_token_t) (i % 60 + 1);
    }
    const bool refused = geist_session_prefill_tokens(s, SMALL_CAP + 1, prompt) != GEIST_OK;
    geist_session_destroy(s);
    return refused;
}

int main(void) {
#if defined(__GLIBC__)
    /* Fixed thresholds: every large block is its own mapping, returned at
     * free, so a later session cannot reuse an earlier one's pages unseen
     * (glibc otherwise raises the mmap threshold after the first free). */
    mallopt(M_MMAP_THRESHOLD, 256 * 1024);
    mallopt(M_TRIM_THRESHOLD, 256 * 1024);
#endif
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_scalar backend not in this build\n");
        return GEIST_TEST_SKIP;
    }
    /* 2 attention layers x 2 KV heads x head_dim 64: FP32 KV at 32768 rows
     * is 2 x 2 x 32768 x 2 x 64 x 4 B = 64 MiB per session. */
    struct tf_buf                   g  = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                                           .d_model  = 256,
                                                                           .heads    = 4,
                                                                           .kv_heads = 2,
                                                                           .ffn      = 256,
                                                                           .vocab    = 64,
                                                                           .context  = 64,
                                                                           .seed     = 5});
    const struct geist_session_opts lo = {.max_seq_len = MODEL_CAP};
    struct geist_model             *m  = nullptr;
    if (geist_model_load_from_memory_with_opts(g.b, g.n, be, &lo, &m) != GEIST_OK) {
        fprintf(stderr, "ERROR: model load: %s\n", geist_last_create_error());
        free(g.b);
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }

    static const struct {
        enum geist_kv_mode mode;
        const char        *name;
        size_t             full_kib; /* KV bytes at MODEL_CAP, KiB (lower bound) */
    } modes[] = {
            {GEIST_KV_FP32, "fp32", 65536},
            {GEIST_KV_INT8, "int8", 16384},
            {GEIST_KV_INT4, "int4", 8192},
            {GEIST_KV_KIVI, "kivi", 4096},
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof modes / sizeof modes[0]; i++) {
        long          small_kib = 0, full_kib = 0;
        geist_token_t a[STEPS] = {0}, b[STEPS] = {0};
        const bool    ok_small = run(m, be, modes[i].mode, SMALL_CAP, &small_kib, a);
        const bool    ok_full  = run(m, be, modes[i].mode, 0, &full_kib, b);
        char          msg[192];
        snprintf(msg, sizeof msg, "%s: both sessions run", modes[i].name);
        fails += geist_expect(ok_small && ok_full, msg);
        snprintf(msg,
                 sizeof msg,
                 "%s: a %d-row session decodes as a %d-row one",
                 modes[i].name,
                 SMALL_CAP,
                 MODEL_CAP);
        fails += geist_expect(memcmp(a, b, sizeof a) == 0, msg);
#if defined(__linux__)
        snprintf(msg,
                 sizeof msg,
                 "%s: session RSS %ld KiB at %d rows vs %ld KiB at %d rows",
                 modes[i].name,
                 small_kib,
                 SMALL_CAP,
                 full_kib,
                 MODEL_CAP);
        /* The full session holds at least the whole KV; the small one at
         * most an eighth of it (scratch and the rest are cap-independent). */
        fails += geist_expect(small_kib >= 0 && full_kib >= (long) modes[i].full_kib &&
                                      small_kib < full_kib - (long) modes[i].full_kib * 7 / 8,
                              msg);
#endif
        snprintf(msg,
                 sizeof msg,
                 "%s: a %d-row session refuses %d rows",
                 modes[i].name,
                 SMALL_CAP,
                 SMALL_CAP + 1);
        fails += geist_expect(refuses_overflow(m, be, modes[i].mode), msg);
    }
    geist_model_destroy(m);
    free(g.b);
    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("session kv sizing: pass\n");
    return GEIST_TEST_PASS;
}
