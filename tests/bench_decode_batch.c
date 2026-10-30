/*
 * bench_decode_batch — geist_session_decode_batch (#26) on a real GGUF:
 * isolation, agreement with decode_step and aggregate tokens/s.
 *
 *   bench_decode_batch model.gguf backend prompts_a.txt prompts_b.txt [n_check [n_time]]
 *
 * prompts_*.txt: one prompt per line as space-separated token IDs, 8 lines
 * each (two subjects). Every prompt is prefilled once and its session image
 * kept (geist_session_snapshot), so every run starts from the same state.
 *
 *   1. Reference: per prompt of A, n_check greedy decode_step steps; the
 *      tokens and the logits pending before each step.
 *   2. Agreement: decode_batch with n = 1, 2, 4, 8 sessions A0..A(n-1) for
 *      n_check steps against their references (first divergent token, steps
 *      with bit-identical logits, max |logit difference| before divergence).
 *   3. Isolation: A0's trace in slot 0 among A1..A7 and in slot 3 among
 *      B0..B6 must be bit-identical. A pair with B0 is reported only: another
 *      n may select other row kernels.
 *   4. Speed: n_time steps of decode_step and of decode_batch at n = 2, 4, 8,
 *      three interleaved rounds; aggregate tokens/s, best round.
 *
 * Exit 3 when an isolation check fails; 77 when the model cannot batch.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { NQ = 8, MAXT = 4096, MAXB = 8 };

struct prompt {
    size_t         n;
    geist_token_t *ids;
    void          *img;
    size_t         img_n;
};

struct trace {
    geist_token_t *tok;    /* [steps] */
    float         *logits; /* [steps][vocab], pending before each step */
};

static size_t vocab;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec * 1e3 + (double) t.tv_nsec / 1e6;
}

static void check(enum geist_status s, const char *what) {
    if (s != GEIST_OK) {
        fprintf(stderr, "%s: %s\n", what, geist_status_to_string(s));
        exit(2);
    }
}

static size_t read_prompts(const char *path, struct prompt p[static NQ]) {
    FILE *f = fopen(path, "r");
    if (f == nullptr) {
        perror(path);
        exit(2);
    }
    static char line[1 << 20];
    size_t      n = 0;
    while (n < NQ && fgets(line, sizeof line, f) != nullptr) {
        p[n].ids = calloc(MAXT, sizeof *p[n].ids);
        size_t k = 0;
        for (char *t = strtok(line, " \n"); t != nullptr && k < MAXT; t = strtok(nullptr, " \n"))
            p[n].ids[k++] = (geist_token_t) strtol(t, nullptr, 10);
        p[n++].n = k;
    }
    fclose(f);
    return n;
}

static void prepare(struct geist_session *s, struct prompt *p) {
    check(geist_session_reset(s), "reset");
    check(geist_session_prefill_tokens(s, p->n, p->ids), "prefill");
    size_t n = 0;
    if (geist_session_peek_logits(&n, s) == nullptr)
        check(GEIST_E_BACKEND, "peek");
    vocab = n;
    check(geist_session_snapshot_size(&p->img_n, s), "snapshot size");
    p->img = malloc(p->img_n);
    check(geist_session_snapshot(&p->img_n, p->img_n, p->img, s), "snapshot");
}

static void trace_alloc(struct trace *t, size_t steps) {
    t->tok    = calloc(steps, sizeof *t->tok);
    t->logits = malloc(steps * vocab * sizeof(float));
    if (t->tok == nullptr || t->logits == nullptr)
        check(GEIST_E_OOM, "trace");
}

static void pending(struct geist_session *s, float *dst) {
    size_t       n = 0;
    const float *l = geist_session_peek_logits(&n, s);
    if (l == nullptr || n != vocab)
        check(GEIST_E_BACKEND, "peek");
    memcpy(dst, l, vocab * sizeof(float));
}

static void reference(struct geist_session *s, struct prompt *p, size_t steps, struct trace *t) {
    check(geist_session_restore(p->img_n, p->img, s), "restore");
    for (size_t i = 0; i < steps; i++) {
        pending(s, t->logits + i * vocab);
        check(geist_session_decode_step(s, &t->tok[i]), "decode_step");
    }
}

static void batched(size_t                n,
                    struct geist_session *ss[static n],
                    struct prompt        *ps[static n],
                    size_t                steps,
                    struct trace          tr[static n]) {
    for (size_t b = 0; b < n; b++)
        check(geist_session_restore(ps[b]->img_n, ps[b]->img, ss[b]), "restore");
    geist_token_t out[MAXB];
    for (size_t i = 0; i < steps; i++) {
        for (size_t b = 0; b < n; b++)
            pending(ss[b], tr[b].logits + i * vocab);
        check(geist_session_decode_batch(n, ss, out), "decode_batch");
        for (size_t b = 0; b < n; b++)
            tr[b].tok[i] = out[b];
    }
}

struct cmp {
    size_t first_token_diff; /* = steps when none */
    size_t bit_identical_steps;
    float  max_abs;
};

/* Logits only up to the first divergent token: after it the inputs differ. */
static struct cmp compare(const struct trace *a, const struct trace *b, size_t steps) {
    struct cmp c = {.first_token_diff = steps};
    for (size_t i = 0; i < steps; i++) {
        const float *x = a->logits + i * vocab, *y = b->logits + i * vocab;
        c.bit_identical_steps += memcmp(x, y, vocab * sizeof(float)) == 0;
        for (size_t k = 0; k < vocab; k++) {
            const float d = fabsf(x[k] - y[k]);
            c.max_abs     = d > c.max_abs ? d : c.max_abs;
        }
        if (a->tok[i] != b->tok[i]) {
            c.first_token_diff = i;
            break;
        }
    }
    return c;
}

static double time_single(struct geist_session *s, struct prompt *p, size_t steps) {
    check(geist_session_restore(p->img_n, p->img, s), "restore");
    geist_token_t t = 0;
    for (int w = 0; w < 4; w++)
        check(geist_session_decode_step(s, &t), "decode_step");
    const double t0 = now_ms();
    for (size_t i = 0; i < steps; i++)
        check(geist_session_decode_step(s, &t), "decode_step");
    return now_ms() - t0;
}

static double time_batched(size_t                n,
                           struct geist_session *ss[static n],
                           struct prompt        *ps[static n],
                           size_t                steps) {
    for (size_t b = 0; b < n; b++)
        check(geist_session_restore(ps[b]->img_n, ps[b]->img, ss[b]), "restore");
    geist_token_t out[MAXB];
    for (int w = 0; w < 4; w++)
        check(geist_session_decode_batch(n, ss, out), "decode_batch");
    const double t0 = now_ms();
    for (size_t i = 0; i < steps; i++)
        check(geist_session_decode_batch(n, ss, out), "decode_batch");
    return now_ms() - t0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: bench_decode_batch model.gguf backend a.txt b.txt [n_check [n_time]]\n");
        return 2;
    }
    const size_t  n_check = argc > 5 ? strtoul(argv[5], nullptr, 10) : 32;
    const size_t  n_time  = argc > 6 ? strtoul(argv[6], nullptr, 10) : 64;
    struct prompt A[NQ] = {}, Bp[NQ] = {};
    if (read_prompts(argv[3], A) != NQ || read_prompts(argv[4], Bp) != NQ) {
        fprintf(stderr, "need %d prompts per file\n", NQ);
        return 2;
    }
    struct geist_backend *be = nullptr;
    struct geist_model   *m  = nullptr;
    check(geist_backend_create(argv[2], nullptr, nullptr, &be), "backend");
    struct geist_session_opts so = {.max_seq_len = 2048, .top_p = 1.0f, .kv_mode = GEIST_KV_FP32};
    /* GEIST_BENCH_M_MAX: the sessions' row cap (#245: nine sessions of a
     * mixture of experts on an 11 GB card need the small one). */
    if (getenv("GEIST_BENCH_M_MAX") != nullptr)
        so.m_max = strtoul(getenv("GEIST_BENCH_M_MAX"), nullptr, 10);
    check(geist_model_load_with_opts(argv[1], be, &so, &m), "load");
    const size_t max = geist_model_decode_batch_max(m);
    printf("decode_batch_max %zu\n", max);
    if (max < MAXB)
        GEIST_SKIP("the model cannot batch 8 sessions here");
    struct geist_session *ref = nullptr, *ss[MAXB] = {};
    check(geist_session_create(m, be, &so, &ref), "session");
    for (size_t b = 0; b < MAXB; b++)
        check(geist_session_create(m, be, &so, &ss[b]), "session");
    const double tp = now_ms();
    for (size_t q = 0; q < NQ; q++) {
        prepare(ref, &A[q]);
        prepare(ref, &Bp[q]);
    }
    printf("prepared 16 prompts in %.1f s\n", (now_ms() - tp) / 1e3);

    struct trace refA[NQ], tr[MAXB], a0[3];
    for (size_t q = 0; q < NQ; q++) {
        trace_alloc(&refA[q], n_check);
        reference(ref, &A[q], n_check, &refA[q]);
    }
    for (size_t b = 0; b < MAXB; b++)
        trace_alloc(&tr[b], n_check);
    for (size_t k = 0; k < 3; k++)
        trace_alloc(&a0[k], n_check);
    int          fails = 0;
    const size_t Bs[]  = {1, 2, 4, 8};
    for (size_t k = 0; k < 4; k++) {
        const size_t   n = Bs[k];
        struct prompt *ps[MAXB];
        for (size_t b = 0; b < n; b++)
            ps[b] = &A[b];
        batched(n, ss, ps, n_check, tr);
        size_t agree = 0, bit_all = 0, earliest = n_check;
        float  worst = 0;
        for (size_t b = 0; b < n; b++) {
            const struct cmp c = compare(&refA[b], &tr[b], n_check);
            agree += c.first_token_diff == n_check;
            bit_all += c.bit_identical_steps == n_check;
            worst    = c.max_abs > worst ? c.max_abs : worst;
            earliest = c.first_token_diff < earliest ? c.first_token_diff : earliest;
        }
        printf("agreement n=%zu: %zu/%zu sessions token-identical to decode_step over %zu steps "
               "(earliest divergence %zu), %zu/%zu bit-identical throughout, max |logit diff| "
               "%.4g\n",
               n,
               agree,
               n,
               n_check,
               earliest,
               bit_all,
               n,
               (double) worst);
        if (n == 1 && bit_all != 1) {
            printf("  n=1 differs from decode_step\n");
            fails++;
        }
        if (n == 8) {
            memcpy(a0[0].tok, tr[0].tok, n_check * sizeof *tr[0].tok);
            memcpy(a0[0].logits, tr[0].logits, n_check * vocab * sizeof(float));
        }
    }
    {
        struct prompt *py[MAXB] = {&Bp[0], &Bp[1], &Bp[2], &A[0], &Bp[3], &Bp[4], &Bp[5], &Bp[6]};
        batched(8, ss, py, n_check, tr);
        const struct cmp c8    = compare(&a0[0], &tr[3], n_check);
        struct prompt   *pz[2] = {&Bp[0], &A[0]};
        batched(2, ss, pz, n_check, tr);
        const struct cmp c2 = compare(&a0[0], &tr[1], n_check);
        printf("isolation: A0 in slot 0 of A0..A7 vs slot 3 among B0..B6: %zu/%zu steps "
               "bit-identical; vs slot 1 of a pair with B0: %zu/%zu\n",
               c8.bit_identical_steps,
               n_check,
               c2.bit_identical_steps,
               n_check);
        fails += c8.bit_identical_steps != n_check; /* the pair is another n: reported only */
    }
    double       single[3], multi[3][3];
    const size_t Ts[] = {2, 4, 8};
    for (int r = 0; r < 3; r++) {
        single[r] = time_single(ref, &A[r], n_time);
        for (size_t k = 0; k < 3; k++) {
            struct prompt *ps[MAXB];
            for (size_t b = 0; b < Ts[k]; b++)
                ps[b] = &A[(b + (size_t) r) % NQ];
            multi[r][k] = time_batched(Ts[k], ss, ps, n_time);
        }
    }
    double       s_best = fmin(single[0], fmin(single[1], single[2]));
    const double s_rate = (double) n_time / (s_best / 1e3);
    printf("speed, %zu steps, best of 3 rounds:\n  decode_step: %.1f ms/step, %.2f tok/s "
           "(rounds %.0f/%.0f/%.0f ms)\n",
           n_time,
           s_best / (double) n_time,
           s_rate,
           single[0],
           single[1],
           single[2]);
    for (size_t k = 0; k < 3; k++) {
        const double best = fmin(multi[0][k], fmin(multi[1][k], multi[2][k]));
        const double rate = (double) (Ts[k] * n_time) / (best / 1e3);
        printf("  decode_batch n=%zu: %.1f ms/step, %.2f tok/s, %.2fx (rounds %.0f/%.0f/%.0f ms)\n",
               Ts[k],
               best / (double) n_time,
               rate,
               rate / s_rate,
               multi[0][k],
               multi[1][k],
               multi[2][k]);
    }
    printf("ISOLATION: %s\n", fails ? "FAIL" : "PASS");
    for (size_t b = 0; b < MAXB; b++)
        geist_session_destroy(ss[b]);
    geist_session_destroy(ref);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return fails ? 3 : 0;
}
