/* Acceptance of geistlib's speculative decode on given prompts (experiment for geistlib#105).
 * Prompts: a file of raw chat texts separated by the byte 0x1e. For each prompt: prefill, then greedy
 * geist_session_decode_speculative(k_max) until N tokens are emitted; prints tokens per call and time.
 * The drafter is the model's MTP head with GEIST_MTP=1 (if the GGUF has one), else geistlib's n-gram lookup.
 * usage: spec_accept MODEL.gguf PROMPTS BACKEND N KMAX */
#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAXH 16384
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: spec_accept MODEL PROMPTS BACKEND N KMAX\n"); return 2; }
    const size_t N = (size_t) atol(argv[4]), K = (size_t) atol(argv[5]);
    FILE *f = fopen(argv[2], "rb"); if (!f) return 2;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *all = malloc((size_t) len + 1); if (fread(all, 1, (size_t) len, f) != (size_t) len) return 2; all[len] = 0; fclose(f);
    struct geist_backend *be = nullptr;
    if (geist_backend_create(argv[3], nullptr, nullptr, &be) != GEIST_OK) { fprintf(stderr, "backend: %s\n", geist_last_create_error()); return 1; }
    struct geist_session_opts opts = {.max_seq_len = 8192, .temperature = 0.0f};
    struct geist_model *m = nullptr;
    if (geist_model_load_with_opts(argv[1], be, &opts, &m) != GEIST_OK) { fprintf(stderr, "load: %s\n", geist_last_create_error()); return 1; }
    static geist_token_t hist[MAXH];
    geist_token_t buf[17];
    size_t tc = 0, te = 0; double tt = 0;
    printf("backend %s  k_max %zu  tokens %zu  GEIST_MTP=%s SEQ=%s\n", geist_backend_name(be), K, N, getenv("GEIST_MTP") ? getenv("GEIST_MTP") : "0", getenv("SEQ") ? "1" : "0");
    int idx = 0;
    FILE *dump = getenv("DUMP") ? fopen(getenv("DUMP"), "w") : nullptr; /* prompt and emitted token ids per prompt */
    for (char *p = all; p && *p; idx++) {
        char *e = strchr(p, '\x1e'); if (e) *e = 0;
        struct geist_session *s = nullptr;
        if (geist_session_create(m, be, &opts, &s) != GEIST_OK) { fprintf(stderr, "session: %s\n", geist_last_create_error()); return 1; }
        size_t hn = 0;
        if (geist_session_tokenize(s, p, MAXH - N - 32, hist, &hn) != GEIST_OK ||
            geist_session_prefill_tokens(s, hn, hist) != GEIST_OK) { fprintf(stderr, "prefill %d: %s\n", idx, geist_session_errmsg(s)); return 1; }
        size_t calls = 0, emits = 0; double t0 = ms();
        while (emits < N) {
            size_t n = 0;
            enum geist_status st = getenv("SEQ") ? geist_session_decode_step(s, &buf[0]) : /* SEQ=1: plain decode */
                                   geist_session_decode_speculative(s, K, hn, hist, K + 1, buf, &n);
            if (getenv("SEQ")) n = 1;
            if (st != GEIST_OK || n == 0) { fprintf(stderr, "spec %d: %s\n", idx, geist_status_to_string(st)); return 1; }
            calls++;
            for (size_t i = 0; i < n; i++) { emits++; if (hn < MAXH) hist[hn++] = buf[i]; }
        }
        double dt = ms() - t0;
        if (dump) { fprintf(dump, "{\"prompt\":%d,\"n_prompt\":%zu,\"ids\":[", idx, hn - emits);
                    for (size_t i = 0; i < hn; i++) fprintf(dump, i ? ",%d" : "%d", (int) hist[i]); fprintf(dump, "]}\n"); fflush(dump); }
        printf("prompt %2d  prompt_tokens %5zu  calls %4zu  emitted %4zu  tokens/call %.2f  %.1f tok/s\n", idx, hn - emits, calls, emits, (double) emits / calls, emits / (dt / 1e3));
        fflush(stdout);
        tc += calls; te += emits; tt += dt;
        geist_session_destroy(s);
        p = e ? e + 1 : nullptr;
    }
    printf("TOTAL  calls %zu  emitted %zu  tokens/call %.3f  %.2f tok/s\n", tc, te, (double) te / tc, te / (tt / 1e3));
    geist_model_destroy(m); geist_backend_destroy(be);
    return 0;
}
