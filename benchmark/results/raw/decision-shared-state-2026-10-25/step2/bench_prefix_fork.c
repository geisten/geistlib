/* mb26 — #26 phase-1 micro-benchmark (scratch tool, not for the repo).
 *
 * mb26 <gguf> <backend> <prompts.txt> <align> <gen_n>
 *
 * prompts.txt: one question per line, space-separated token ids; all lines
 * share a common prefix (one MMLU subject's 5-shot exemplars).
 *
 * 1. Isolation: per question, an independent session (reset, prefill prefix,
 *    prefill suffix, then gen_n greedy tokens) is the reference. A worker
 *    session prefills the prefix once, snapshots, and answers every question
 *    by restore + suffix + gen_n tokens, in forward, reverse and interleaved
 *    order. Logits after the suffix and after every generated token must be
 *    bit-identical, tokens identical. With the cut on a chunk multiple the
 *    suffix logits must also equal a single prefill of the joined prompt.
 * 2. Cost: full prefill vs restore + suffix per question; snapshot size,
 *    snapshot and restore time.
 * 3. m-row pass cost after a real prompt: prefill_tokens(m) for m = 1..8
 *    against one decode step.
 */
#include <geist.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAXQ = 64, MAXT = 4096 };

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(size_t n, double v[static n]) {
    qsort(v, n, sizeof v[0], cmp_d);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

#define CHECK(expr)                                                                         \
    do {                                                                                    \
        enum geist_status st_ = (expr);                                                     \
        if (st_ != GEIST_OK) {                                                              \
            fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #expr, (int)st_);       \
            exit(2);                                                                        \
        }                                                                                   \
    } while (0)

static size_t         nq;
static size_t         qlen[MAXQ];
static geist_token_t *qids[MAXQ];
static size_t         vocab;

static float *logits_copy(struct geist_session *s) {
    size_t       n = 0;
    const float *l = geist_session_peek_logits(&n, s);
    if (l == nullptr || n == 0) {
        fprintf(stderr, "no logits\n");
        exit(2);
    }
    vocab   = n;
    float *c = malloc(n * sizeof *c);
    memcpy(c, l, n * sizeof *c);
    return c;
}

static bool logits_eq(struct geist_session *s, const float *ref) {
    size_t       n = 0;
    const float *l = geist_session_peek_logits(&n, s);
    return l != nullptr && n == vocab && memcmp(l, ref, n * sizeof *l) == 0;
}

static float max_abs_diff(const float *a, const float *b) {
    float m = 0;
    for (size_t i = 0; i < vocab; i++) {
        float d = a[i] - b[i];
        d       = d < 0 ? -d : d;
        m       = d > m ? d : m;
    }
    return m;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: mb26 gguf backend prompts align gen_n\n");
        return 1;
    }
    const size_t align = strtoul(argv[4], nullptr, 10);
    const size_t gen_n = strtoul(argv[5], nullptr, 10);
    FILE        *f     = fopen(argv[3], "r");
    if (f == nullptr)
        return 1;
    static char line[1 << 20];
    while (nq < MAXQ && fgets(line, sizeof line, f) != nullptr) {
        qids[nq] = malloc(MAXT * sizeof(geist_token_t));
        size_t n = 0;
        for (char *p = strtok(line, " \n"); p != nullptr && n < MAXT; p = strtok(nullptr, " \n"))
            qids[nq][n++] = (geist_token_t)strtol(p, nullptr, 10);
        qlen[nq++] = n;
    }
    fclose(f);
    size_t lcp = qlen[0];
    for (size_t q = 1; q < nq; q++) {
        size_t k = 0;
        while (k < lcp && k < qlen[q] && qids[q][k] == qids[0][k])
            k++;
        lcp = k;
    }
    /* cut strictly before the last shared token (text may merge there) */
    size_t cut = lcp - 1;
    if (align > 1)
        cut = cut / align * align;

    struct geist_backend *be = nullptr;
    struct geist_model   *m  = nullptr;
    CHECK(geist_backend_create(argv[2], nullptr, nullptr, &be));
    struct geist_session_opts so = {.max_seq_len = MAXT, .top_p = 1.0f, .kv_mode = GEIST_KV_FP32};
    CHECK(geist_model_load_with_opts(argv[1], be, &so, &m));
    struct geist_session *ref = nullptr, *w = nullptr;
    CHECK(geist_session_create(m, be, &so, &ref));
    CHECK(geist_session_create(m, be, &so, &w));

    printf("questions %zu, prompt tokens", nq);
    for (size_t q = 0; q < nq; q++)
        printf(" %zu", qlen[q]);
    printf("\nshared prefix %zu tokens, cut at %zu (align %zu)\n", lcp, cut, align);

    /* warm-up: one full prompt */
    CHECK(geist_session_prefill_tokens(ref, qlen[0], qids[0]));
    CHECK(geist_session_reset(ref));

    /* ---- references ---------------------------------------------------- */
    static float         *ref_suffix[MAXQ], *ref_joined[MAXQ], *ref_gen_last[MAXQ];
    static geist_token_t *ref_tok[MAXQ];
    double                t_full[MAXQ], t_split_ref[MAXQ];
    for (size_t q = 0; q < nq; q++) {
        CHECK(geist_session_reset(ref));
        double t0 = now_ms();
        CHECK(geist_session_prefill_tokens(ref, qlen[q], qids[q]));
        t_full[q]     = now_ms() - t0;
        ref_joined[q] = logits_copy(ref);

        CHECK(geist_session_reset(ref));
        t0 = now_ms();
        CHECK(geist_session_prefill_tokens(ref, cut, qids[q]));
        CHECK(geist_session_prefill_tokens(ref, qlen[q] - cut, qids[q] + cut));
        t_split_ref[q] = now_ms() - t0;
        ref_suffix[q]  = logits_copy(ref);
        ref_tok[q]     = malloc((gen_n + 1) * sizeof(geist_token_t));
        for (size_t i = 0; i < gen_n; i++)
            CHECK(geist_session_decode_step(ref, &ref_tok[q][i]));
        ref_gen_last[q] = gen_n ? logits_copy(ref) : nullptr;
    }

    /* ---- worker: prefix once, snapshot -------------------------------- */
    CHECK(geist_session_reset(w));
    double t0 = now_ms();
    CHECK(geist_session_prefill_tokens(w, cut, qids[0]));
    const double t_prefix = now_ms() - t0;
    size_t       snap_n   = 0;
    CHECK(geist_session_snapshot_size(&snap_n, w));
    void *img = malloc(snap_n);
    t0        = now_ms();
    size_t got = 0;
    CHECK(geist_session_snapshot(&got, snap_n, img, w));
    const double t_snap = now_ms() - t0;
    printf("prefix prefill %.1f ms; snapshot %.1f MiB in %.2f ms\n", t_prefix,
           (double)got / (1 << 20), t_snap);

    size_t order[3][MAXQ];
    for (size_t q = 0; q < nq; q++) {
        order[0][q] = q;
        order[1][q] = nq - 1 - q;
        order[2][q] = q % 2 ? nq - 1 - q / 2 : q / 2;
    }
    const char *oname[3] = {"forward", "reverse", "interleaved"};
    int         fails    = 0;
    double      t_restore[3 * MAXQ], t_suffix[3 * MAXQ];
    size_t      nr = 0;
    for (int o = 0; o < 3; o++) {
        int ok_l = 0, ok_j = 0, ok_g = 0;
        for (size_t k = 0; k < nq; k++) {
            const size_t q = order[o][k];
            t0             = now_ms();
            CHECK(geist_session_restore(got, img, w));
            t_restore[nr] = now_ms() - t0;
            t0            = now_ms();
            CHECK(geist_session_prefill_tokens(w, qlen[q] - cut, qids[q] + cut));
            t_suffix[nr++] = now_ms() - t0;
            ok_l += logits_eq(w, ref_suffix[q]);
            ok_j += logits_eq(w, ref_joined[q]);
            if (o == 0 && !logits_eq(w, ref_joined[q])) {
                size_t       n = 0;
                const float *l = geist_session_peek_logits(&n, w);
                printf("  q%zu restore+suffix vs joined single prefill: max |diff| %.3g\n", q,
                       (double)max_abs_diff(l, ref_joined[q]));
            }
            bool g = true;
            for (size_t i = 0; i < gen_n; i++) {
                geist_token_t t = 0;
                CHECK(geist_session_decode_step(w, &t));
                g = g && t == ref_tok[q][i];
            }
            if (gen_n)
                g = g && logits_eq(w, ref_gen_last[q]);
            ok_g += g;
        }
        printf("%-11s suffix logits bit-equal to independent split %d/%zu, to joined %d/%zu; "
               "%zu generated tokens + final logits equal %d/%zu\n",
               oname[o], ok_l, nq, ok_j, nq, gen_n, ok_g, nq);
        fails += (ok_l != (int)nq) + (ok_g != (int)nq);
    }
    double sum_full = 0, sum_shared = t_prefix;
    for (size_t q = 0; q < nq; q++)
        sum_full += t_full[q];
    for (size_t k = 0; k < nq; k++)
        sum_shared += t_restore[k] + t_suffix[k];
    printf("per question: full prefill median %.1f ms; restore median %.2f ms; suffix median %.1f "
           "ms\n",
           median(nq, t_full), median(nr, t_restore), median(nr, t_suffix));
    printf("all %zu questions: independent %.0f ms, shared prefix %.0f ms (%.2fx)\n", nq, sum_full,
           sum_shared, sum_shared / sum_full);

    /* ---- m-row pass cost ------------------------------------------------ */
    CHECK(geist_session_restore(got, img, w));
    CHECK(geist_session_prefill_tokens(w, qlen[0] - cut, qids[0] + cut));
    size_t n2 = 0;
    CHECK(geist_session_snapshot_size(&n2, w));
    void  *img2 = malloc(n2);
    size_t got2 = 0;
    CHECK(geist_session_snapshot(&got2, n2, img2, w));
    enum { REP = 9, STEPS = 16 };
    double v[REP];
    for (int r = 0; r < REP; r++) {
        CHECK(geist_session_restore(got2, img2, w));
        geist_token_t t = 0;
        CHECK(geist_session_decode_step(w, &t)); /* first step: sample only */
        t0 = now_ms();
        for (int i = 0; i < STEPS; i++)
            CHECK(geist_session_decode_step(w, &t));
        v[r] = (now_ms() - t0) / STEPS;
    }
    printf("decode step after %zu tokens: %.2f ms\n", qlen[0], median(REP, v));
    const size_t ms[] = {1, 2, 3, 4, 6, 8, 16};
    for (size_t j = 0; j < sizeof ms / sizeof ms[0]; j++) {
        for (int r = 0; r < REP; r++) {
            CHECK(geist_session_restore(got2, img2, w));
            t0 = now_ms();
            CHECK(geist_session_prefill_tokens(w, ms[j], qids[1] + 100));
            v[r] = now_ms() - t0;
        }
        printf("prefill_tokens(m=%zu) pass: %.2f ms\n", ms[j], median(REP, v));
    }
    printf("RESULT %s\n", fails ? "FAIL" : "PASS");
    geist_session_destroy(w);
    geist_session_destroy(ref);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return fails ? 3 : 0;
}
