/*
 * bench_pin_prefix — #578 diagnostic (not merged): HELIO's pattern. A
 * prompt of n tokens; the first p are a constant prefix. Compares, over r
 * requests, (a) reset to empty + full prompt, and (b) pin the prefix once,
 * then reset (to the pin) + the remainder. Both feed in `chunk`-token
 * prefill calls, as HELIO did (64).
 * Usage: bench_pin_prefix <model.gguf> <n> <p> <chunk> <r>
 * Backend: GEIST_BENCH_BACKEND (default "auto").
 */
#define _POSIX_C_SOURCE 200809L

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec * 1e-6;
}

static int feed(struct geist_session *s, size_t n, const geist_token_t *ids, size_t chunk) {
    for (size_t off = 0; off < n;) {
        const size_t c = n - off < chunk ? n - off : chunk;
        if (geist_session_prefill_tokens(s, c, ids + off) != GEIST_OK) {
            fprintf(stderr, "prefill failed: %s\n", geist_session_errmsg(s));
            return 1;
        }
        off += c;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s model.gguf n p chunk r\n", argv[0]);
        return 2;
    }
    const size_t n = strtoul(argv[2], nullptr, 10), p = strtoul(argv[3], nullptr, 10);
    const size_t chunk = strtoul(argv[4], nullptr, 10), r = strtoul(argv[5], nullptr, 10);
    const char  *bn = getenv("GEIST_BENCH_BACKEND");
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
    struct geist_session_opts o = {.max_seq_len = n + 16, .temperature = 0.0f};
    struct geist_session     *a = nullptr, *b = nullptr;
    if (geist_session_create(m, be, &o, &a) != GEIST_OK ||
        geist_session_create(m, be, &o, &b) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        return 1;
    }
    double t = now_ms();
    if (feed(a, n, ids, chunk) != 0) {
        return 1;
    }
    printf("cold full prompt (%zu tok, chunk %zu): %.0f ms\n", n, chunk, now_ms() - t);
    t = now_ms();
    if (geist_session_pin_prefix(b, p, ids) != GEIST_OK) {
        fprintf(stderr, "pin_prefix failed: %s\n", geist_session_errmsg(b));
        return 1;
    }
    printf("pin_prefix (%zu tok): %.0f ms\n", p, now_ms() - t);
    for (size_t i = 0; i < r; i++) {
        t = now_ms();
        if (geist_session_reset(a) != GEIST_OK || feed(a, n, ids, chunk) != 0) {
            return 1;
        }
        const double full = now_ms() - t;
        t = now_ms();
        if (geist_session_reset(b) != GEIST_OK || feed(b, n - p, ids + p, chunk) != 0) {
            return 1;
        }
        printf("req %zu: full %.0f ms | pinned remainder (%zu tok) %.0f ms\n", i, full, n - p,
               now_ms() - t);
        fflush(stdout);
    }
    geist_session_destroy(a);
    geist_session_destroy(b);
    free(ids);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return 0;
}
