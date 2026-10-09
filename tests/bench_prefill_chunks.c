/*
 * bench_prefill_chunks — #628 part 2 diagnostic (not merged): one long
 * prefill fed as one call vs fixed-size calls. A per-call cost that grows
 * with the cached length shows up as a lower rate for small chunks.
 * Usage: bench_prefill_chunks <model.gguf> <n_tokens> <chunk>...  (0 = one call)
 * Backend: GEIST_BENCH_BACKEND (default "auto").
 */
#define _POSIX_C_SOURCE 200809L

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf n_tokens chunk...\n", argv[0]);
        return 2;
    }
    const size_t n   = strtoul(argv[2], nullptr, 10);
    const char  *bn  = getenv("GEIST_BENCH_BACKEND");
    struct geist_backend *be = nullptr;
    struct geist_model   *m  = nullptr;
    if (geist_backend_create(bn != nullptr ? bn : "auto", nullptr, nullptr, &be) != GEIST_OK ||
        geist_model_load(argv[1], be, &m) != GEIST_OK) {
        fprintf(stderr, "load failed: %s\n", geist_last_create_error());
        return 1;
    }
    geist_token_t *ids = malloc(n * sizeof *ids);
    for (size_t i = 0; i < n; i++) {
        ids[i] = (geist_token_t) (100 + (i * 7919u) % 20000u);
    }
    for (int a = 3; a < argc; a++) {
        const size_t chunk = strtoul(argv[a], nullptr, 10);
        struct geist_session_opts o = {.max_seq_len = n + 16, .temperature = 0.0f};
        struct geist_session     *s = nullptr;
        if (geist_session_create(m, be, &o, &s) != GEIST_OK) {
            fprintf(stderr, "session_create failed\n");
            return 1;
        }
        const double t0 = now_s();
        size_t       calls = 0;
        for (size_t off = 0; off < n; calls++) {
            const size_t c = chunk == 0 || n - off < chunk ? n - off : chunk;
            if (geist_session_prefill_tokens(s, c, ids + off) != GEIST_OK) {
                fprintf(stderr, "prefill failed at %zu: %s\n", off, geist_session_errmsg(s));
                return 1;
            }
            off += c;
        }
        const double dt = now_s() - t0;
        printf("chunk %5zu  calls %4zu  %7.2f s  %7.1f tok/s\n", chunk, calls, dt, (double) n / dt);
        fflush(stdout);
        geist_session_destroy(s);
    }
    free(ids);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return 0;
}
