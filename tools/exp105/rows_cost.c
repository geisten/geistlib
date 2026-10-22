/* Cost of an m-row pass (prefill_tokens(m)) against one decode step, Bonsai 2 27B (#105).
 * usage: rows_cost MODEL PROMPTS BACKEND REPS — first prompt only; median ms per m. */
#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp(const void *a, const void *b) { double d = *(const double *) a - *(const double *) b; return (d > 0) - (d < 0); }
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    const int R = atoi(argv[4]);
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
    geist_token_t t; double v[9][64];
    for (int r = 0; r < R; r++) {
        double t0 = ms(); enum geist_status ds = geist_session_decode_step(s, &t); v[0][r] = ms() - t0; if (ds != GEIST_OK) fprintf(stderr, "decode_step: %s %s\n", geist_status_to_string(ds), geist_session_errmsg(s));
        for (int k = 1; k <= 8; k++) {
            t0 = ms();
            if (geist_session_prefill_tokens(s, (size_t) k, ids + 16 * r + k) != GEIST_OK) { fprintf(stderr, "prefill: %s\n", geist_session_errmsg(s)); return 1; }
            v[k][r] = ms() - t0;
        }
    }
    printf("%s  prompt %zu  reps %d  median ms\n  decode_step %.1f\n", geist_backend_name(be), n, R, (qsort(v[0], R, sizeof(double), cmp), v[0][R / 2]));
    for (int k = 1; k <= 8; k++) { qsort(v[k], R, sizeof(double), cmp); printf("  m=%d  %.1f\n", k, v[k][R / 2]); }
    return 0;
}
