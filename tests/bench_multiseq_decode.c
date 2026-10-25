/*
 * bench_multiseq_decode — #26 step 3 gate: B independent sequences stepped as
 * one pass (transformer_multi_decode_step, a research prototype, no public
 * API) against single-sequence decode, on a real Qwen3.5-family GGUF (Bonsai 2
 * 27B) on Metal.
 *
 *   bench_multiseq_decode model.gguf prompts_a.txt prompts_b.txt [n_check [n_time]]
 *
 * prompts_*.txt: one prompt per line as space-separated token IDs, 8 lines
 * each (A and B are two subjects). Every prompt is prefilled once and its
 * complete session image kept (geist_session_snapshot), so every run below
 * starts from exactly the same state.
 *
 *   1. Reference: per prompt, single-sequence greedy decode (the engine's own
 *      transformer_decode_step) for n_check steps; tokens and dense logits.
 *   2. Agreement: B = 1, 2, 4, 8 sequences A0..A(B-1) in one pass for n_check
 *      steps against their references (first divergent token, max |logit
 *      difference|, bit-identical steps).
 *   3. Companion invariance: A0 in slot 0 among A1..A7 versus A0 in slot 3
 *      among B0..B6 (other questions, other lengths); and at B = 4 among
 *      A1..A3 versus B0..B2. A0's tokens and logits must be bit-identical.
 *   4. Speed: n_time steps, logits left on the device, for single-sequence
 *      decode and B = 1, 2, 4, 8, interleaved over three rounds; aggregate
 *      tokens/s. Gate: B = 8 at >= 2x the single-sequence rate.
 *
 * Build (not part of make test):
 *   cc -std=c23 -O2 -DGEIST_INTERNAL_ARCH_LAYER -Iinclude -I. -Isrc/base -Isrc/quant
 * -Isrc/backends/common \
 *      -Isrc/formats/gguf -Isrc/engine tests/bench_multiseq_decode.c \
 *      lib/mac-omp/release/libgeist.a -framework Accelerate -framework Metal \
 *      -framework Foundation -L/opt/homebrew/opt/libomp/lib -lomp -lc++ -lm \
 *      -o bench_multiseq_decode
 */
#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"

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
    geist_token_t  first; /* greedy token after the prompt */
};

static size_t vocab;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec * 1e3 + (double) t.tv_nsec / 1e6;
}

static void die(const char *what, enum geist_status s) {
    fprintf(stderr, "%s: %s\n", what, geist_status_to_string(s));
    exit(2);
}
#define CHECK(expr)                           \
    do {                                      \
        const enum geist_status st_ = (expr); \
        if (st_ != GEIST_OK)                  \
            die(#expr, st_);                  \
    } while (0)

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

static geist_token_t argmax(const float *row) {
    size_t best = 0;
    for (size_t i = 1; i < vocab; i++)
        best = row[i] > row[best] ? i : best;
    return (geist_token_t) best;
}

/* Prefill, remember the first greedy token, keep the complete image. */
static void prepare(struct geist_session *s, struct prompt *p) {
    CHECK(geist_session_reset(s));
    CHECK(geist_session_prefill_tokens(s, p->n, p->ids));
    size_t       n = 0;
    const float *l = geist_session_peek_logits(&n, s);
    if (l == nullptr) {
        die("peek", GEIST_E_BACKEND);
    }
    vocab    = n;
    p->first = argmax(l);
    CHECK(geist_session_snapshot_size(&p->img_n, s));
    p->img = malloc(p->img_n);
    CHECK(geist_session_snapshot(&p->img_n, p->img_n, p->img, s));
}

struct trace {
    geist_token_t *tok;    /* [steps] */
    float         *logits; /* [steps][vocab] */
};

static void trace_alloc(struct trace *t, size_t steps) {
    t->tok    = calloc(steps, sizeof *t->tok);
    t->logits = malloc(steps * vocab * sizeof(float));
    if (t->tok == nullptr || t->logits == nullptr) {
        die("trace alloc", GEIST_E_OOM);
    }
}

static void reference(struct geist_session *s, struct prompt *p, size_t steps, struct trace *t) {
    struct transformer_arch_session *a = geist_session_internal_arch_session(s);
    CHECK(geist_session_restore(p->img_n, p->img, s));
    geist_token_t in = p->first;
    for (size_t i = 0; i < steps; i++) {
        geist_token_t out = 0;
        CHECK(transformer_decode_step(a, in, &out));
        size_t       n = 0;
        const float *l = geist_session_peek_logits(&n, s);
        if (l == nullptr || n != vocab) {
            die("reference peek", GEIST_E_BACKEND);
        }
        memcpy(t->logits + i * vocab, l, vocab * sizeof(float));
        t->tok[i] = argmax(l);
        if (t->tok[i] != out) {
            fprintf(stderr, "note: decode_step token %d != dense argmax %d\n", out, t->tok[i]);
        }
        in = t->tok[i];
    }
}

/* B sequences from prompts ps[] in sessions ss[], steps with logits. */
static void batched(struct transformer_arch_session *carrier,
                    size_t                           B,
                    struct geist_session            *ss[static B],
                    struct prompt                   *ps[static B],
                    size_t                           steps,
                    struct trace                     tr[static B]) {
    struct transformer_arch_session *seqs[MAXB];
    geist_token_t                    in[MAXB], out[MAXB];
    float                           *logits = malloc(B * vocab * sizeof(float));
    for (size_t b = 0; b < B; b++) {
        CHECK(geist_session_restore(ps[b]->img_n, ps[b]->img, ss[b]));
        seqs[b] = geist_session_internal_arch_session(ss[b]);
        in[b]   = ps[b]->first;
    }
    for (size_t i = 0; i < steps; i++) {
        CHECK(transformer_multi_decode_step(carrier, B, seqs, in, out, logits));
        for (size_t b = 0; b < B; b++) {
            memcpy(tr[b].logits + i * vocab, logits + b * vocab, vocab * sizeof(float));
            tr[b].tok[i] = out[b];
            in[b]        = out[b];
        }
    }
    free(logits);
}

struct cmp {
    size_t first_token_diff; /* steps if none */
    size_t bit_identical_steps;
    float  max_abs;
};

/* Logits compared only up to the first token divergence (after it the
 * inputs differ and the comparison means nothing). */
static struct cmp compare(const struct trace *a, const struct trace *b, size_t steps) {
    struct cmp c = {.first_token_diff = steps};
    for (size_t i = 0; i < steps; i++) {
        const float *x = a->logits + i * vocab, *y = b->logits + i * vocab;
        if (memcmp(x, y, vocab * sizeof(float)) == 0)
            c.bit_identical_steps++;
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
    struct transformer_arch_session *a = geist_session_internal_arch_session(s);
    CHECK(geist_session_restore(p->img_n, p->img, s));
    geist_token_t in = p->first, out = 0;
    for (int w = 0; w < 4; w++, in = out)
        CHECK(transformer_decode_step(a, in, &out));
    const double t0 = now_ms();
    for (size_t i = 0; i < steps; i++, in = out)
        CHECK(transformer_decode_step(a, in, &out));
    return now_ms() - t0;
}

static double time_batched(struct transformer_arch_session *carrier,
                           size_t                           B,
                           struct geist_session            *ss[static B],
                           struct prompt                   *ps[static B],
                           size_t                           steps) {
    struct transformer_arch_session *seqs[MAXB];
    geist_token_t                    in[MAXB], out[MAXB];
    for (size_t b = 0; b < B; b++) {
        CHECK(geist_session_restore(ps[b]->img_n, ps[b]->img, ss[b]));
        seqs[b] = geist_session_internal_arch_session(ss[b]);
        in[b]   = ps[b]->first;
    }
    for (int w = 0; w < 4; w++) {
        CHECK(transformer_multi_decode_step(carrier, B, seqs, in, out, nullptr));
        memcpy(in, out, B * sizeof *in);
    }
    const double t0 = now_ms();
    for (size_t i = 0; i < steps; i++) {
        CHECK(transformer_multi_decode_step(carrier, B, seqs, in, out, nullptr));
        memcpy(in, out, B * sizeof *in);
    }
    return now_ms() - t0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: bench_multiseq_decode model.gguf a.txt b.txt [n_check [n_time]]\n");
        return 2;
    }
    const size_t  n_check = argc > 4 ? strtoul(argv[4], nullptr, 10) : 32;
    const size_t  n_time  = argc > 5 ? strtoul(argv[5], nullptr, 10) : 64;
    struct prompt A[NQ] = {}, Bp[NQ] = {};
    if (read_prompts(argv[2], A) != NQ || read_prompts(argv[3], Bp) != NQ) {
        fprintf(stderr, "need %d prompts per file\n", NQ);
        return 2;
    }
    struct geist_backend *be = nullptr;
    struct geist_model   *m  = nullptr;
    CHECK(geist_backend_create("metal", nullptr, nullptr, &be));
    struct geist_session_opts so = {.max_seq_len = 2048, .top_p = 1.0f, .kv_mode = GEIST_KV_FP32};
    CHECK(geist_model_load_with_opts(argv[1], be, &so, &m));
    struct geist_session *carrier = nullptr, *ref = nullptr, *ss[MAXB] = {};
    CHECK(geist_session_create(m, be, &so, &carrier));
    CHECK(geist_session_create(m, be, &so, &ref));
    for (size_t b = 0; b < MAXB; b++)
        CHECK(geist_session_create(m, be, &so, &ss[b]));
    struct transformer_arch_session *ca = geist_session_internal_arch_session(carrier);
    printf("carrier m_max %zu\n", ca->m_max);

    const double tp = now_ms();
    for (size_t q = 0; q < NQ; q++) {
        prepare(ref, &A[q]);
        prepare(ref, &Bp[q]);
    }
    printf("prepared 16 prompts in %.1f s (A %zu..%zu tokens)\n",
           (now_ms() - tp) / 1e3,
           A[0].n,
           A[7].n);

    /* 1. references */
    struct trace refA[NQ];
    for (size_t q = 0; q < NQ; q++) {
        trace_alloc(&refA[q], n_check);
        reference(ref, &A[q], n_check, &refA[q]);
    }
    /* 2. agreement */
    int          fails = 0;
    const size_t Bs[]  = {1, 2, 4, 8};
    struct trace tr[MAXB];
    for (size_t b = 0; b < MAXB; b++)
        trace_alloc(&tr[b], n_check);
    struct trace a0_in_A8 = {};
    trace_alloc(&a0_in_A8, n_check);
    struct trace a0_in_A4 = {};
    trace_alloc(&a0_in_A4, n_check);
    for (size_t k = 0; k < 4; k++) {
        const size_t   B = Bs[k];
        struct prompt *ps[MAXB];
        for (size_t b = 0; b < B; b++)
            ps[b] = &A[b];
        batched(ca, B, ss, ps, n_check, tr);
        size_t agree = 0, bit_all = 0;
        float  worst   = 0;
        size_t min_div = n_check;
        for (size_t b = 0; b < B; b++) {
            const struct cmp c = compare(&refA[b], &tr[b], n_check);
            agree += c.first_token_diff == n_check;
            bit_all += c.bit_identical_steps == n_check;
            worst   = c.max_abs > worst ? c.max_abs : worst;
            min_div = c.first_token_diff < min_div ? c.first_token_diff : min_div;
        }
        printf("agreement B=%zu: %zu/%zu sequences token-identical over %zu steps (earliest "
               "divergence "
               "step %zu), %zu/%zu bit-identical logits at every step, max |logit diff| before "
               "divergence %.4g\n",
               B,
               agree,
               B,
               n_check,
               min_div,
               bit_all,
               B,
               (double) worst);
        if (B == 1 && bit_all != 1) {
            printf("  B=1 is not bit-identical to single-sequence decode\n");
            fails++;
        }
        if (B == 8) {
            memcpy(a0_in_A8.tok, tr[0].tok, n_check * sizeof *tr[0].tok);
            memcpy(a0_in_A8.logits, tr[0].logits, n_check * vocab * sizeof(float));
        }
        if (B == 4) {
            memcpy(a0_in_A4.tok, tr[0].tok, n_check * sizeof *tr[0].tok);
            memcpy(a0_in_A4.logits, tr[0].logits, n_check * vocab * sizeof(float));
        }
    }
    /* 3. companion invariance */
    {
        struct prompt *ps[MAXB] = {&Bp[0], &Bp[1], &Bp[2], &A[0], &Bp[3], &Bp[4], &Bp[5], &Bp[6]};
        batched(ca, 8, ss, ps, n_check, tr);
        const struct cmp c = compare(&a0_in_A8, &tr[3], n_check);
        printf("invariance B=8: A0 in slot 0 among A1..A7 vs slot 3 among B0..B6: tokens identical "
               "%s, bit-identical logits %zu/%zu steps (max |diff| %.3g)\n",
               c.first_token_diff == n_check ? "yes" : "NO",
               c.bit_identical_steps,
               n_check,
               (double) c.max_abs);
        fails += c.bit_identical_steps != n_check;
        struct prompt *p4[MAXB] = {&Bp[0], &Bp[1], &A[0], &Bp[2]};
        batched(ca, 4, ss, p4, n_check, tr);
        const struct cmp c4 = compare(&a0_in_A4, &tr[2], n_check);
        printf("invariance B=4: A0 in slot 0 among A1..A3 vs slot 2 among B0..B2: tokens identical "
               "%s, bit-identical logits %zu/%zu steps (max |diff| %.3g)\n",
               c4.first_token_diff == n_check ? "yes" : "NO",
               c4.bit_identical_steps,
               n_check,
               (double) c4.max_abs);
        fails += c4.bit_identical_steps != n_check;
    }
    /* 4. speed, three interleaved rounds */
    double single[3], multi[3][4];
    for (int r = 0; r < 3; r++) {
        single[r] = time_single(ref, &A[r], n_time);
        for (size_t k = 0; k < 4; k++) {
            struct prompt *ps[MAXB];
            for (size_t b = 0; b < Bs[k]; b++)
                ps[b] = &A[(b + (size_t) r) % NQ];
            multi[r][k] = time_batched(ca, Bs[k], ss, ps, n_time);
        }
    }
    double s_best = single[0];
    for (int r = 1; r < 3; r++)
        s_best = single[r] < s_best ? single[r] : s_best;
    const double s_rate = (double) n_time / (s_best / 1e3);
    printf("speed (%zu steps, best of 3 rounds):\n  single-sequence decode: %.1f ms/step, %.2f "
           "tok/s "
           "(rounds %.0f/%.0f/%.0f ms)\n",
           n_time,
           s_best / (double) n_time,
           s_rate,
           single[0],
           single[1],
           single[2]);
    double ratio8 = 0;
    for (size_t k = 0; k < 4; k++) {
        double best = multi[0][k];
        for (int r = 1; r < 3; r++)
            best = multi[r][k] < best ? multi[r][k] : best;
        const double rate = (double) (Bs[k] * n_time) / (best / 1e3);
        printf("  B=%zu: %.1f ms/step, %.2f tok/s aggregate, %.2fx single (rounds %.0f/%.0f/%.0f "
               "ms)\n",
               Bs[k],
               best / (double) n_time,
               rate,
               rate / s_rate,
               multi[0][k],
               multi[1][k],
               multi[2][k]);
        if (Bs[k] == 8)
            ratio8 = rate / s_rate;
    }
    printf("GATE (B=8 >= 2.0x single): %s (%.2fx)\n", ratio8 >= 2.0 ? "PASS" : "FAIL", ratio8);
    printf("ISOLATION: %s\n", fails ? "FAIL" : "PASS");
    for (size_t b = 0; b < MAXB; b++)
        geist_session_destroy(ss[b]);
    geist_session_destroy(ref);
    geist_session_destroy(carrier);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return fails ? 3 : 0;
}
