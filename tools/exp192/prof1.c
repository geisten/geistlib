/* #192 step 1: one row through decode_step vs prefill_tokens(1), N times after a prompt.
 * usage: prof1 MODEL PROMPTS BACKEND MODE N   (MODE: none | decode | prefillK, K = rows per call)
 * Run with the backend's profiler on; the per-call cost is (MODE - none) / N. */
#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const char *mode = argv[4]; const int N = atoi(argv[5]);
    FILE *f = fopen(argv[2], "rb"); if (!f) return 2;
    static char all[1 << 20]; size_t len = fread(all, 1, sizeof all - 1, f); all[len] = 0; fclose(f);
    char *e = strchr(all, '\x1e'); if (e) *e = 0;
    struct geist_backend *be = nullptr;
    if (geist_backend_create(argv[3], nullptr, nullptr, &be) != GEIST_OK) return 1;
    struct geist_session_opts opts = {.max_seq_len = 8192, .temperature = 0.0f};
    struct geist_model *m = nullptr; struct geist_session *s = nullptr;
    if (geist_model_load_with_opts(argv[1], be, &opts, &m) != GEIST_OK || geist_session_create(m, be, &opts, &s) != GEIST_OK) return 1;
    static geist_token_t ids[8192]; size_t n = 0;
    if (geist_session_tokenize(s, all, 4096, ids, &n) != GEIST_OK || geist_session_prefill_tokens(s, n, ids) != GEIST_OK) return 1;
    geist_token_t t;
    if (strcmp(mode, "decode") == 0 && geist_session_decode_step(s, &t) != GEIST_OK) return 1; /* the pending token: no forward */
    double t0 = ms();
    for (int i = 0; i < N; i++) {
        enum geist_status st = GEIST_OK;
        if (strcmp(mode, "decode") == 0) st = geist_session_decode_step(s, &t);
        else if (strncmp(mode, "prefill", 7) == 0) {
            const size_t k = (size_t) atoi(mode + 7);
            st = geist_session_prefill_tokens(s, k, ids + (i * k) % (n - k));
        }
        if (st != GEIST_OK) { fprintf(stderr, "%s %d: %s\n", mode, i, geist_session_errmsg(s)); return 1; }
    }
    if (strcmp(mode, "decode") == 0) (void) geist_session_peek_logits(&len, s); /* settle the last deferred forward */
    printf("%s  prompt %zu  N %d  %.2f ms/call\n", mode, n, N, (ms() - t0) / (N > 0 ? N : 1));
    geist_session_destroy(s); geist_model_destroy(m); geist_backend_destroy(be);
    return 0;
}
