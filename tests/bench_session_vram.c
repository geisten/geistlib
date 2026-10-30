/*
 * bench_session_vram — what one more session costs on a GPU backend, for a
 * mixture of experts whose hot experts fill the device (#245 follow-up).
 *
 *   bench_session_vram model.gguf backend [max_sessions [max_seq_len [m_max]]]
 *
 * GEIST_BENCH_BATCH_SESSIONS=<k> loads with geist_session_opts.batch_sessions
 * = k: the hot-expert budget then leaves room for k sessions, and all k must
 * be created.
 * Loads the model with the given window and row cap, then creates sessions
 * one at a time; each takes one prefill token and one decode step so its
 * buffers exist. After the load and after each session it prints the
 * device's free bytes (geist_backend_memory_info) and stops at the first
 * failure. GEIST_MOE_HOT etc. apply as usual.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>

enum { MAXS = 64 };

static double free_mib(struct geist_backend *be) {
    struct geist_backend_memory m;
    return geist_backend_memory_info(be, &m) == GEIST_OK ? (double) m.free_bytes / 1048576.0 : -1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: bench_session_vram model.gguf backend [max_sessions [max_seq_len "
                "[m_max]]]\n");
        return 2;
    }
    const size_t          n   = argc > 3 ? strtoul(argv[3], nullptr, 10) : 16;
    const size_t          win = argc > 4 ? strtoul(argv[4], nullptr, 10) : 2048;
    const size_t          mm  = argc > 5 ? strtoul(argv[5], nullptr, 10) : 0;
    struct geist_backend *be  = nullptr;
    if (geist_backend_create(argv[2], nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend: %s\n", geist_last_create_error());
        return 2;
    }
    const double              f0 = free_mib(be);
    struct geist_session_opts so = {.max_seq_len = win, .top_p = 1.0f, .m_max = mm};
    if (getenv("GEIST_BENCH_BATCH_SESSIONS") != nullptr)
        so.batch_sessions = strtoul(getenv("GEIST_BENCH_BATCH_SESSIONS"), nullptr, 10);
    struct geist_model *m = nullptr;
    if (geist_model_load_with_opts(argv[1], be, &so, &m) != GEIST_OK) {
        fprintf(stderr, "load: %s\n", geist_last_create_error());
        return 2;
    }
    double prev = free_mib(be);
    printf("{\"free_before_load_mib\":%.0f,\"free_after_load_mib\":%.0f}\n", f0, prev);
    struct geist_session *ss[MAXS] = {};
    size_t                k        = 0;
    for (; k < n && k < MAXS; k++) {
        geist_token_t     tok = 100, out = 0;
        enum geist_status s = geist_session_create(m, be, &so, &ss[k]);
        if (s == GEIST_OK)
            s = geist_session_prefill_tokens(ss[k], 1, &tok);
        if (s == GEIST_OK)
            s = geist_session_decode_step(ss[k], &out);
        const double f = free_mib(be);
        printf("{\"session\":%zu,\"status\":\"%s\",\"free_mib\":%.0f,\"cost_mib\":%.0f}\n",
               k + 1,
               geist_status_to_string(s),
               f,
               prev - f);
        prev = f;
        if (s != GEIST_OK)
            break;
    }
    for (size_t i = 0; i < MAXS; i++)
        geist_session_destroy(ss[i]);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return 0;
}
