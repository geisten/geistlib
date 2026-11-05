/* Verify-cost ablation: ARMS="none;CLS,CLS;..." (GEIST_X_SKIP values), each arm
 * a fresh session + prompt prefill (no skip), then CALLS verify_drafts calls
 * with N_DRAFTS rejected drafts under the arm's skip. Rounds alternate order.
 * usage: vprof MODEL PROMPT N_DRAFTS CALLS ROUNDS */
#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void *a, const void *b) { double x = *(const double *) a, y = *(const double *) b; return (x > y) - (x < y); }
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const size_t nd = (size_t) atoi(argv[3]); const int N = atoi(argv[4]), R = atoi(argv[5]);
    static char arms_s[4096]; snprintf(arms_s, sizeof arms_s, "%s", getenv("ARMS") ? getenv("ARMS") : "none");
    char *arms[32]; int na = 0;
    for (char *t = strtok(arms_s, ";"); t && na < 32; t = strtok(nullptr, ";")) arms[na++] = t;
    FILE *f = fopen(argv[2], "rb"); if (!f) return 2;
    static char all[1 << 20]; size_t len = fread(all, 1, sizeof all - 1, f); all[len] = 0; fclose(f);
    char *e = strchr(all, '\x1e'); if (e) *e = 0;
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK) return 1;
    struct geist_session_opts opts = {.max_seq_len = 8192, .temperature = 0.0f};
    struct geist_model *m = nullptr;
    if (geist_model_load_with_opts(argv[1], be, &opts, &m) != GEIST_OK) return 1;
    static geist_token_t ids[8192]; size_t n = 0;
    geist_token_t drafts[16], out[18]; size_t n_out = 0;
    for (size_t i = 0; i < nd; i++) drafts[i] = 7u + (geist_token_t) i;
    for (int r = 0; r < R; r++) {
        for (int k = 0; k < na; k++) {
            const int a = (r % 2 == 0) ? k : na - 1 - k;
            unsetenv("GEIST_X_SKIP");
            struct geist_session *s = nullptr;
            if (geist_session_create(m, be, &opts, &s) != GEIST_OK) return 1;
            if (geist_session_tokenize(s, all, 4096, ids, &n) != GEIST_OK || geist_session_prefill_tokens(s, n, ids) != GEIST_OK) return 1;
            if (strcmp(arms[a], "none") != 0) setenv("GEIST_X_SKIP", arms[a], 1);
            double v[256];
            for (int i = -2; i < N; i++) {
                const double t0 = ms();
                if (geist_session_verify_drafts(s, nd, drafts, sizeof out / sizeof out[0], out, &n_out) != GEIST_OK) {
                    fprintf(stderr, "arm %s verify %d: %s\n", arms[a], i, geist_session_errmsg(s)); return 1;
                }
                if (i >= 0) v[i] = ms() - t0;
            }
            unsetenv("GEIST_X_SKIP");
            qsort(v, (size_t) N, sizeof v[0], cmpd);
            printf("round %d arm %-28s median %7.2f ms  (p25 %.2f p75 %.2f)\n", r, arms[a], v[N / 2], v[N / 4], v[3 * N / 4]);
            fflush(stdout);
            geist_session_destroy(s);
        }
    }
    geist_model_destroy(m); geist_backend_destroy(be);
    return 0;
}
