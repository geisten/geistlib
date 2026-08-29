/*
 * tools/zo_tune.c — forward-only fine-tuning of a geist model (MeZO/QZO).
 *
 * Trains the per-tensor gains that geist_model_gains exposes against
 * teacher-forced token NLL. There is no backward pass, no autodiff and no
 * optimizer state: each step is two forward passes and one scalar. Memory
 * is inference memory plus 2*n_gains floats.
 *
 * Why this is tractable on a ternary model: the trits stay frozen and only
 * the gains move, so the search dimension is n_gains (a few hundred), not
 * the parameter count. Zeroth-order optimization degrades with dimension —
 * shrinking the dimension is the whole trick.
 *
 * The gradient estimate is the classic two-sided SPSA form used by MeZO,
 *
 *     d = (L(θ + εz) − L(θ − εz)) / 2ε ,   θ ← θ − lr·d·z ,
 *
 * with z a Rademacher vector regenerated from a seed rather than stored,
 * and QZO's directional-derivative clipping on d to stop one bad batch
 * from throwing the run.
 *
 * Build: make clean && make EXTRA_CFLAGS=-DGEIST_TUNE
 * Run:   bin/<target>/release/tools/zo_tune --gguf M.gguf --data train.jsonl
 * Self-check (no model needed): zo_tune --self-check
 *
 * Data format — one JSON object per line, two string fields:
 *     {"prompt": "turn on the kitchen light", "completion": "light.kitchen"}
 * The loss is the NLL of `completion` given BOS + `prompt`. Prompt and
 * completion are tokenized separately, so the split lands on a token
 * boundary by construction — write the completion the way the model should
 * emit it, leading space included if that is what the tokenizer produces.
 */
#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LINE_CAP  = 1 << 16, /* bytes per JSONL line */
    FIELD_CAP = 1 << 15, /* bytes per decoded string field */
    TOKEN_CAP = 2048,    /* tokens per field */
    BATCH_CAP = 32,
};

/* ---- Rademacher perturbation from a seed (never stored) ---------------- */

static inline uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

/* g[i] = base[i] + a * z_i, with z_i in {-1,+1} drawn from `seed`.
 *
 * Every use of the perturbation goes through this one function — both
 * probes, and the update — so the same seed always reproduces the same z
 * and nothing has to be kept. Writing g from `base` rather than adding to
 * g in place also means no float rounding accumulates across steps: base
 * is the only running state. In-place (g == base) is fine, the mapping is
 * elementwise. */
static void gains_set(float *g, const float *base, size_t n, uint64_t seed, float a) {
    uint64_t s = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        const float z = (xs64(&s) & 1u) ? 1.0f : -1.0f;
        g[i]          = base[i] + a * z;
    }
}

/* ---- Minimal JSON string-field reader ---------------------------------- */

/* Find "key" at the top level of `line` and decode its string value into
 * `out`. Returns the decoded length, or -1 if the key is absent, the value
 * is not a string, it does not fit, or an escape is malformed.
 *
 * Deliberately not a JSON parser: it reads the two fields this tool needs.
 * But it does decode escapes properly — training data that silently loses
 * its umlauts would poison the run in a way no loss curve would reveal. */
static long json_string_field(const char *line, const char *key, char *out, size_t cap) {
    char      pattern[64];
    const int pn = snprintf(pattern, sizeof pattern, "\"%s\"", key);
    if (pn <= 0 || (size_t) pn >= sizeof pattern) {
        return -1;
    }
    const char *p = strstr(line, pattern);
    if (p == nullptr) {
        return -1;
    }
    p += (size_t) pn;
    while (*p == ' ' || *p == '\t') {
        p++;
    }
    if (*p != ':') {
        return -1;
    }
    p++;
    while (*p == ' ' || *p == '\t') {
        p++;
    }
    if (*p != '"') {
        return -1;
    }
    p++;

    size_t w = 0;
    while (*p != '"') {
        if (*p == '\0' || w + 4 >= cap) {
            return -1;
        }
        if (*p != '\\') {
            out[w++] = *p++;
            continue;
        }
        p++;
        switch (*p) {
        case '"':
            out[w++] = '"';
            p++;
            break;
        case '\\':
            out[w++] = '\\';
            p++;
            break;
        case '/':
            out[w++] = '/';
            p++;
            break;
        case 'b':
            out[w++] = '\b';
            p++;
            break;
        case 'f':
            out[w++] = '\f';
            p++;
            break;
        case 'n':
            out[w++] = '\n';
            p++;
            break;
        case 'r':
            out[w++] = '\r';
            p++;
            break;
        case 't':
            out[w++] = '\t';
            p++;
            break;
        case 'u': {
            unsigned cp = 0;
            p++;
            for (int k = 0; k < 4; k++) {
                const char c = p[k];
                unsigned   d;
                if (c >= '0' && c <= '9') {
                    d = (unsigned) (c - '0');
                } else if (c >= 'a' && c <= 'f') {
                    d = (unsigned) (c - 'a') + 10u;
                } else if (c >= 'A' && c <= 'F') {
                    d = (unsigned) (c - 'A') + 10u;
                } else {
                    return -1;
                }
                cp = cp * 16u + d;
            }
            p += 4;
            /* Surrogate half: a correct decoder would pair it up. This
             * one refuses rather than emitting a lone half — a visible
             * failure beats silently corrupted training text. */
            if (cp >= 0xD800u && cp <= 0xDFFFu) {
                return -1;
            }
            if (cp < 0x80u) {
                out[w++] = (char) cp;
            } else if (cp < 0x800u) {
                out[w++] = (char) (0xC0u | (cp >> 6));
                out[w++] = (char) (0x80u | (cp & 0x3Fu));
            } else {
                out[w++] = (char) (0xE0u | (cp >> 12));
                out[w++] = (char) (0x80u | ((cp >> 6) & 0x3Fu));
                out[w++] = (char) (0x80u | (cp & 0x3Fu));
            }
            break;
        }
        default:
            return -1;
        }
    }
    out[w] = '\0';
    return (long) w;
}

/* ---- Examples ---------------------------------------------------------- */

struct example {
    geist_token_t *ctx;
    size_t         n_ctx;
    geist_token_t *tgt;
    size_t         n_tgt;
};

static void examples_free(struct example *ex, size_t n) {
    if (ex == nullptr) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        free(ex[i].ctx);
        free(ex[i].tgt);
    }
    free(ex);
}

static geist_token_t *tokens_dup(const geist_token_t *src, size_t n) {
    geist_token_t *d = malloc(n * sizeof *d);
    if (d != nullptr) {
        memcpy(d, src, n * sizeof *d);
    }
    return d;
}

/* Read a JSONL file into tokenized examples. Blank lines are skipped; a
 * malformed line is reported with its number and aborts the load rather
 * than being dropped, since a silently shorter training set is the kind of
 * thing you only notice as an unexplained plateau. */
static struct example *
load_jsonl(struct geist_session *s, struct geist_model *m, const char *path, size_t *n_out) {
    *n_out = 0;

    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
        fprintf(stderr, "zo_tune: cannot open %s\n", path);
        return nullptr;
    }

    char *line  = malloc(LINE_CAP);
    char *field = malloc(FIELD_CAP);
    if (line == nullptr || field == nullptr) {
        free(line);
        free(field);
        fclose(f);
        return nullptr;
    }

    const geist_token_t bos = geist_model_bos_token(m);
    geist_token_t      *tok = malloc(TOKEN_CAP * sizeof *tok);

    struct example *ex = nullptr;
    size_t          n = 0, cap = 0;
    size_t          lineno = 0;
    bool            bad    = false;

    while (!bad && fgets(line, LINE_CAP, f) != nullptr) {
        lineno++;
        const char *scan = line;
        while (*scan == ' ' || *scan == '\t' || *scan == '\r' || *scan == '\n') {
            scan++;
        }
        if (*scan == '\0') {
            continue;
        }
        if (n == cap) {
            const size_t ncap = cap ? cap * 2 : 64;
            void        *p    = realloc(ex, ncap * sizeof *ex);
            if (p == nullptr) {
                bad = true;
                break;
            }
            ex  = p;
            cap = ncap;
        }
        struct example e = {0};

        /* Context: BOS + prompt, matching what set_prompt would build. */
        if (json_string_field(line, "prompt", field, FIELD_CAP) < 0) {
            fprintf(stderr, "zo_tune: %s:%zu — no usable \"prompt\" field\n", path, lineno);
            bad = true;
            break;
        }
        size_t n_tok = 0;
        if (geist_session_tokenize(s, field, TOKEN_CAP, tok, &n_tok) != GEIST_OK || n_tok == 0) {
            fprintf(stderr, "zo_tune: %s:%zu — prompt did not tokenize\n", path, lineno);
            bad = true;
            break;
        }
        const bool add_bos = bos != GEIST_TOKEN_NONE && tok[0] != bos;
        e.n_ctx            = n_tok + (add_bos ? 1u : 0u);
        e.ctx              = malloc(e.n_ctx * sizeof *e.ctx);
        if (e.ctx == nullptr) {
            bad = true;
            break;
        }
        if (add_bos) {
            e.ctx[0] = bos;
        }
        memcpy(e.ctx + (add_bos ? 1u : 0u), tok, n_tok * sizeof *tok);

        /* Target: the completion, scored token by token. */
        if (json_string_field(line, "completion", field, FIELD_CAP) < 0) {
            fprintf(stderr, "zo_tune: %s:%zu — no usable \"completion\" field\n", path, lineno);
            free(e.ctx);
            bad = true;
            break;
        }
        if (geist_session_tokenize(s, field, TOKEN_CAP, tok, &n_tok) != GEIST_OK || n_tok == 0) {
            fprintf(stderr, "zo_tune: %s:%zu — completion did not tokenize\n", path, lineno);
            free(e.ctx);
            bad = true;
            break;
        }
        e.n_tgt = n_tok;
        e.tgt   = tokens_dup(tok, n_tok);
        if (e.tgt == nullptr) {
            free(e.ctx);
            bad = true;
            break;
        }
        ex[n++] = e;
    }

    free(tok);
    free(field);
    free(line);
    fclose(f);

    if (bad) {
        examples_free(ex, n);
        return nullptr;
    }
    if (n == 0) {
        fprintf(stderr, "zo_tune: %s holds no examples\n", path);
        free(ex);
        return nullptr;
    }
    *n_out = n;
    return ex;
}

/* ---- Loss: teacher-forced NLL ------------------------------------------ */

static double logsumexp(const float *v, size_t n) {
    float mx = v[0];
    for (size_t i = 1; i < n; i++) {
        if (v[i] > mx) {
            mx = v[i];
        }
    }
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        acc += exp((double) (v[i] - mx));
    }
    return (double) mx + log(acc);
}

/* Mean NLL per target token. One prefill for the context, then one
 * single-token prefill per further target token — those run at decode cost
 * against a warm KV cache, so an example costs about one prefill.
 *
 * `pin_len` is how many leading context tokens are already resident as a
 * pinned KV prefix; reset() truncates back to them rather than to empty, so
 * only the suffix is re-prefilled. Every example shares that work instead
 * of redoing it, which for a fixed system prompt is most of the forward
 * pass.
 *
 * pin_len MUST match what is actually pinned on the session. It is not a
 * "use the pin or don't" switch: reset() restores to the pinned prefix
 * either way, so passing 0 while a prefix is pinned prefills the shared
 * context a SECOND time on top of itself. That produces a plausible but
 * wrong loss — measured, it moved the holdout from 2.79 to 3.26. Pass 0
 * only when nothing is pinned. */
static double nll_one(struct geist_session *s, const struct example *e, size_t pin_len) {
    if (geist_session_reset(s) != GEIST_OK) {
        return NAN;
    }
    /* pin_len is capped below so a suffix is never empty: peek_logits needs
     * a prefill to have produced pending logits. */
    if (geist_session_prefill_tokens(s, e->n_ctx - pin_len, e->ctx + pin_len) != GEIST_OK) {
        return NAN;
    }
    double acc = 0.0;
    for (size_t k = 0; k < e->n_tgt; k++) {
        size_t       nv = 0;
        const float *lg = geist_session_peek_logits(s, &nv);
        if (lg == nullptr || nv == 0 || (size_t) e->tgt[k] >= nv) {
            return NAN;
        }
        acc += logsumexp(lg, nv) - (double) lg[e->tgt[k]];
        if (k + 1 < e->n_tgt) {
            if (geist_session_prefill_tokens(s, 1, &e->tgt[k]) != GEIST_OK) {
                return NAN;
            }
        }
    }
    return acc / (double) e->n_tgt;
}

static double nll_indexed(struct geist_session *s,
                          const struct example *ex,
                          const size_t         *idx,
                          size_t                n,
                          size_t                pin_len) {
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double v = nll_one(s, &ex[idx[i]], pin_len);
        if (isnan(v)) {
            return NAN;
        }
        acc += v;
    }
    return acc / (double) n;
}

static double
nll_range(struct geist_session *s, const struct example *ex, size_t lo, size_t hi, size_t pin_len) {
    double acc = 0.0;
    for (size_t i = lo; i < hi; i++) {
        const double v = nll_one(s, &ex[i], pin_len);
        if (isnan(v)) {
            return NAN;
        }
        acc += v;
    }
    return (hi > lo) ? acc / (double) (hi - lo) : NAN;
}

/* ---- Shared-prefix pinning --------------------------------------------- */

/* Longest token prefix common to every example's context, capped one short
 * of the shortest context so each example still has a suffix to prefill —
 * peek_logits needs a prefill to have produced pending logits. */
static size_t common_prefix_len(const struct example *ex, size_t n) {
    size_t lcp = ex[0].n_ctx;
    for (size_t i = 1; i < n; i++) {
        size_t k = 0;
        while (k < lcp && k < ex[i].n_ctx && ex[i].ctx[k] == ex[0].ctx[k]) {
            k++;
        }
        lcp = k;
    }
    for (size_t i = 0; i < n; i++) {
        if (ex[i].n_ctx <= lcp) {
            lcp = ex[i].n_ctx - 1;
        }
    }
    return lcp;
}

/* How far the pinned and unpinned loss for the same example may differ
 * before pinning is rejected. In NATS, absolute — not relative.
 *
 * Both sides were measured on BitNet 2B-4T with a 68-token shared system
 * prompt:
 *
 *   pinned vs unpinned, same example : 4.270556 vs 4.281214   (Δ 0.011)
 *   context actually missing         : 3.6280   vs 9.8156     (Δ 6.19)
 *
 * The first is prefill chunking: arch_ops.c prefills in m_max(=64)-token
 * sub-batches, so 76 tokens in one call splits 64+12 while pinned-68 plus
 * an 8-token suffix splits 64+4 then 8. Different GEMM batch shapes sum in
 * a different order and the last bits move. Benign, and unavoidable short
 * of pinning at a multiple of m_max.
 *
 * The second is the failure this gate exists to catch. 0.1 nats sits ~10x
 * above the noise and ~60x below the fault.
 *
 * Absolute rather than relative because the noise is roughly absolute while
 * the baseline is not: once tuning drives the loss to ~0.02, that same
 * 0.003-nat wobble reads as 11% and a relative gate rejects a pin that is
 * perfectly fine. Measured, not hypothetical. */
#define PIN_LOSS_TOLERANCE_NATS 0.1

/* How often the pinned prefix is recomputed during training.
 *
 * The pinned KV is a function of the WEIGHTS, and the gains are weights: the
 * cached prefix reflects whatever gains were live when it was pinned, while
 * the suffix and target are scored with the current ones. Training makes the
 * cache stale by construction.
 *
 * It is not fatal — within a step both probes see the same prefix, so the
 * finite difference is taken at a consistent (if slightly displaced)
 * operating point, and a run tuned this way still generalizes when
 * re-verified from a fresh process. But it is measurable: after 400 steps
 * the in-run holdout read 0.1556 against a prefix pinned at gains=1.0, and
 * 0.2249 once re-pinned at the tuned gains. That is 0.07 nats of pure
 * bookkeeping error, and it flatters the number being reported.
 *
 * Re-pinning costs one prefill of the shared prefix. Every 25 steps that is
 * far below the ~89% the pin saves, and it bounds the drift instead of
 * letting it accumulate over the whole run. */
#define REPIN_EVERY_STEPS 25

/* Pin the shared prefix, and CHECK it changed nothing that matters: one
 * example scored with a full prefill against the same example scored on top
 * of the pinned prefix.
 *
 * The check is not paranoia. geist_session_pin_prefix returns GEIST_OK even
 * when the arch's pin_prefix failed underneath (session.c discards that
 * status), so an unnoticed failure would mean training on truncated
 * contexts behind a loss curve that still looks plausible.
 *
 * Returns the usable pin length, or 0 to prefill in full. */
static size_t pin_shared_prefix(struct geist_session *s, const struct example *ex, size_t n) {
    const size_t lcp = common_prefix_len(ex, n);
    if (lcp == 0) {
        fprintf(stderr, "pin: examples share no context prefix — pinning off\n");
        return 0;
    }
    const double unpinned = nll_one(s, &ex[0], 0);
    if (isnan(unpinned)) {
        return 0;
    }
    if (geist_session_pin_prefix(s, lcp, ex[0].ctx) != GEIST_OK) {
        fprintf(stderr, "pin: %s — pinning off\n", geist_session_errmsg(s));
        return 0;
    }
    const double pinned = nll_one(s, &ex[0], lcp);
    const double delta  = isnan(pinned) ? INFINITY : fabs(pinned - unpinned);
    if (delta > PIN_LOSS_TOLERANCE_NATS) {
        fprintf(stderr,
                "pin: %zu-token prefix moves the loss by %.4f nats (%.6f vs "
                "%.6f) — pinning off, the context is not intact\n",
                lcp,
                delta,
                pinned,
                unpinned);
        (void) geist_session_pin_prefix(s, 0, ex[0].ctx); /* unpin */
        return 0;
    }
    fprintf(stderr, "pin: %zu-token prefix verified (loss delta %.4f nats)\n", lcp, delta);
    return lcp;
}

/* ---- The ZO step ------------------------------------------------------- */

struct zo_cfg {
    float  eps;
    float  lr;
    float  clip;
    size_t batch;
};

/* Two forward passes, one scalar update. `g` is the live view geist reads
 * during the forward pass; `base` is the running parameter vector. */
static double zo_step(struct geist_session *s,
                      float                *g,
                      float                *base,
                      size_t                n,
                      const struct example *ex,
                      const size_t         *idx,
                      const struct zo_cfg  *c,
                      size_t                pin_len,
                      uint64_t             *rng) {
    const uint64_t seed = xs64(rng);

    gains_set(g, base, n, seed, +c->eps);
    const double lp = nll_indexed(s, ex, idx, c->batch, pin_len);

    gains_set(g, base, n, seed, -c->eps);
    const double lm = nll_indexed(s, ex, idx, c->batch, pin_len);

    if (isnan(lp) || isnan(lm)) {
        memcpy(g, base, n * sizeof *g);
        return NAN;
    }

    double d = (lp - lm) / (2.0 * (double) c->eps);
    /* QZO's directional-derivative clipping: with a batch this small the
     * estimate is noisy, and one outlier scaled by 1/2ε is large enough to
     * wreck the vector. Clipping bounds the damage. */
    if (d > (double) c->clip) {
        d = (double) c->clip;
    }
    if (d < -(double) c->clip) {
        d = -(double) c->clip;
    }

    gains_set(base, base, n, seed, (float) (-(double) c->lr * d));
    memcpy(g, base, n * sizeof *g);

    return 0.5 * (lp + lm);
}

/* ---- Self-check -------------------------------------------------------- */

/* Not assert(): the shipped build is -DNDEBUG, which would compile every
 * check away and leave --self-check reporting success on no evidence. */
static int checks_failed = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "self-check FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            checks_failed++;                                                           \
        }                                                                              \
    } while (0)

static int self_check(void) {
    /* Perturbation: |z| == 1, seed-reproducible, and the two probes
     * straddle base — the three properties zo_step's arithmetic rests on. */
    float base[16], a[16], b[16];
    for (size_t i = 0; i < 16; i++) {
        base[i] = 1.0f + 0.01f * (float) i;
    }
    gains_set(a, base, 16, 12345u, 0.25f);
    gains_set(b, base, 16, 12345u, -0.25f);
    for (size_t i = 0; i < 16; i++) {
        CHECK(fabsf(fabsf(a[i] - base[i]) - 0.25f) < 1e-6f);
        CHECK(fabsf((a[i] + b[i]) * 0.5f - base[i]) < 1e-6f);
    }
    /* A different seed must give a different direction somewhere —
     * otherwise every step would probe the same axis forever. */
    float c2[16];
    gains_set(c2, base, 16, 999u, 0.25f);
    bool differs = false;
    for (size_t i = 0; i < 16; i++) {
        differs = differs || (c2[i] != a[i]);
    }
    CHECK(differs);

    /* A zero-amplitude perturbation is exactly base (the restore path). */
    gains_set(a, base, 16, 12345u, 0.0f);
    for (size_t i = 0; i < 16; i++) {
        CHECK(a[i] == base[i]);
    }

    const float v[3] = {0.0f, 1.0f, 0.0f};
    CHECK(fabs(logsumexp(v, 3) - log(2.0 + M_E)) < 1e-9);

    /* Shared-prefix length. The cap matters more than the match: every
     * example must keep at least one token to prefill, or peek_logits has
     * no pending logits to read. */
    {
        geist_token_t  t0[] = {1, 2, 3, 4, 5};
        geist_token_t  t1[] = {1, 2, 3, 9, 9};
        geist_token_t  t2[] = {1, 2, 3};
        geist_token_t  t3[] = {7, 2, 3, 4, 5};
        struct example a2[] = {{t0, 5, nullptr, 0}, {t1, 5, nullptr, 0}};
        struct example a3[] = {{t0, 5, nullptr, 0}, {t1, 5, nullptr, 0}, {t2, 3, nullptr, 0}};
        struct example a1[] = {{t0, 5, nullptr, 0}};
        struct example ad[] = {{t0, 5, nullptr, 0}, {t3, 5, nullptr, 0}};
        CHECK(common_prefix_len(a2, 2) == 3); /* diverges at index 3 */
        CHECK(common_prefix_len(a3, 3) == 2); /* capped by the 3-token one */
        CHECK(common_prefix_len(a1, 1) == 4); /* single example: all but one */
        CHECK(common_prefix_len(ad, 2) == 0); /* differ at index 0 */
    }

    /* JSON field decoding, including the escapes that would otherwise
     * corrupt German training text silently. */
    char o[256];
    CHECK(json_string_field("{\"prompt\": \"hi\", \"completion\": \"x\"}", "prompt", o, sizeof o) ==
          2);
    CHECK(strcmp(o, "hi") == 0);
    CHECK(json_string_field("{\"completion\":\"a\\nb\"}", "completion", o, sizeof o) == 3);
    CHECK(strcmp(o, "a\nb") == 0);
    CHECK(json_string_field("{\"p\":\"K\\u00fcche\"}", "p", o, sizeof o) == 6); /* ü is 2 bytes */
    /* Split: "\xbc" would otherwise swallow the following 'c' as a third
     * hex digit and overflow the escape. */
    CHECK(strcmp(o,
                 "K\xc3\xbc"
                 "che") == 0);
    CHECK(json_string_field("{\"p\":\"say \\\"hi\\\"\"}", "p", o, sizeof o) == 8);
    CHECK(strcmp(o, "say \"hi\"") == 0);
    CHECK(json_string_field("{\"p\":\"x\"}", "q", o, sizeof o) == -1);       /* absent */
    CHECK(json_string_field("{\"p\":42}", "p", o, sizeof o) == -1);          /* not a string */
    CHECK(json_string_field("{\"p\":\"\\ud83d\"}", "p", o, sizeof o) == -1); /* lone surrogate */
    CHECK(json_string_field("{\"p\":\"oops", "p", o, sizeof o) == -1);       /* unterminated */

    if (checks_failed != 0) {
        fprintf(stderr, "self-check: %d failure(s)\n", checks_failed);
        return 1;
    }
    printf("self-check ok\n");
    return 0;
}

/* ---- main -------------------------------------------------------------- */

static void usage(void) {
    fprintf(stderr,
            "usage: zo_tune --gguf M.gguf --data train.jsonl [options]\n"
            "       zo_tune --self-check\n"
            "  --out PATH      gains sidecar to write        (gains.bin)\n"
            "  --init PATH     start from this sidecar       (all 1.0)\n"
            "  --steps N       ZO steps                      (2000)\n"
            "  --batch N       examples per step, 1..32      (1)\n"
            "  --lr F          learning rate                 (1e-4)\n"
            "  --eps F         perturbation radius           (1e-3)\n"
            "  --clip F        directional-derivative clip   (10.0)\n"
            "  --holdout N     last N examples held out      (10%%)\n"
            "  --eval-every N  holdout eval interval, 0=off  (200)\n"
            "  --seed N        RNG seed                      (1)\n"
            "  --no-pin        do not pin the shared prefix  (pin on)\n");
}

int main(int argc, char **argv) {
    const char   *gguf = nullptr, *data = nullptr, *out = "gains.bin", *init = nullptr;
    size_t        steps = 2000, eval_every = 200;
    long          holdout = -1;
    bool          no_pin  = false;
    uint64_t      seed    = 1;
    struct zo_cfg c       = {.eps = 1e-3f, .lr = 1e-4f, .clip = 10.0f, .batch = 1};

    for (int i = 1; i < argc; i++) {
        const char *a    = argv[i];
        const bool  more = (i + 1 < argc);
        if (!strcmp(a, "--self-check")) {
            return self_check();
        } else if (!strcmp(a, "--gguf") && more) {
            gguf = argv[++i];
        } else if (!strcmp(a, "--data") && more) {
            data = argv[++i];
        } else if (!strcmp(a, "--out") && more) {
            out = argv[++i];
        } else if (!strcmp(a, "--init") && more) {
            init = argv[++i];
        } else if (!strcmp(a, "--steps") && more) {
            steps = strtoul(argv[++i], nullptr, 10);
        } else if (!strcmp(a, "--batch") && more) {
            c.batch = strtoul(argv[++i], nullptr, 10);
        } else if (!strcmp(a, "--lr") && more) {
            c.lr = strtof(argv[++i], nullptr);
        } else if (!strcmp(a, "--eps") && more) {
            c.eps = strtof(argv[++i], nullptr);
        } else if (!strcmp(a, "--clip") && more) {
            c.clip = strtof(argv[++i], nullptr);
        } else if (!strcmp(a, "--holdout") && more) {
            holdout = strtol(argv[++i], nullptr, 10);
        } else if (!strcmp(a, "--eval-every") && more) {
            eval_every = strtoul(argv[++i], nullptr, 10);
        } else if (!strcmp(a, "--no-pin")) {
            no_pin = true;
        } else if (!strcmp(a, "--seed") && more) {
            seed = strtoull(argv[++i], nullptr, 10);
        } else {
            usage();
            return 2;
        }
    }
    if (gguf == nullptr || data == nullptr) {
        usage();
        return 2;
    }
    if (c.batch < 1 || c.batch > BATCH_CAP) {
        fprintf(stderr, "zo_tune: --batch must be 1..%d\n", BATCH_CAP);
        return 2;
    }
    if (!(c.eps > 0.0f)) {
        fprintf(stderr, "zo_tune: --eps must be > 0\n");
        return 2;
    }

    int                   rc    = 1;
    struct geist_backend *be    = nullptr;
    struct geist_model   *model = nullptr;
    struct geist_session *sess  = nullptr;
    struct example       *ex    = nullptr;
    size_t                n_ex  = 0;
    float                *base  = nullptr;
    size_t               *idx   = nullptr;

    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "zo_tune: backend_create: %s\n", geist_last_create_error());
        goto done;
    }
    if (geist_model_load(gguf, be, &model) != GEIST_OK) {
        fprintf(stderr, "zo_tune: model_load(%s) failed\n", gguf);
        goto done;
    }

    float            *g       = nullptr;
    size_t            n_gains = 0;
    enum geist_status gs      = geist_model_gains(model, &g, &n_gains);
    if (gs != GEIST_OK) {
        fprintf(stderr,
                "zo_tune: gains unavailable (%s): %s\n",
                geist_status_to_string(gs),
                geist_backend_errmsg(be)); /* op_gains writes the backend slot */
        goto done;
    }

    struct geist_session_opts opts = {0}; /* greedy — the loss reads logits */
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "zo_tune: session_create failed\n");
        goto done;
    }

    ex = load_jsonl(sess, model, data, &n_ex);
    if (ex == nullptr) {
        goto done;
    }
    if (holdout < 0) {
        holdout = (long) (n_ex / 10u);
    }
    if ((size_t) holdout >= n_ex) {
        fprintf(stderr, "zo_tune: --holdout %ld leaves no training data\n", holdout);
        goto done;
    }
    const size_t n_train = n_ex - (size_t) holdout;
    if (c.batch > n_train) {
        c.batch = n_train;
    }

    base = malloc(n_gains * sizeof *base);
    idx  = malloc(c.batch * sizeof *idx);
    if (base == nullptr || idx == nullptr) {
        goto done;
    }
    /* --init: continue from an existing sidecar. This is also the load
     * path a serving process uses — geist_model_gains hands out the live
     * array and a read straight into it takes effect on the next forward
     * pass, which is why swapping tuning profiles needs no reload. */
    if (init != nullptr) {
        FILE *f = fopen(init, "rb");
        if (f == nullptr || fread(g, sizeof *g, n_gains, f) != n_gains) {
            fprintf(stderr,
                    "zo_tune: reading %zu gains from %s failed (wrong model?)\n",
                    n_gains,
                    init);
            if (f != nullptr) {
                fclose(f);
            }
            goto done;
        }
        fclose(f);
    }
    memcpy(base, g, n_gains * sizeof *base);

    /* Pin whatever context the examples share. Runs after --init so the
     * equality check that validates the pin sees the gains the run will
     * actually use. */
    const size_t pin_len = no_pin ? 0 : pin_shared_prefix(sess, ex, n_ex);

    fprintf(stderr,
            "zo_tune: %zu gains, %zu train + %ld holdout examples, "
            "batch %zu, lr %g, eps %g, %zu steps\n",
            n_gains,
            n_train,
            holdout,
            c.batch,
            (double) c.lr,
            (double) c.eps,
            steps);
    if (pin_len > 0) {
        size_t total = 0;
        for (size_t i = 0; i < n_ex; i++) {
            total += ex[i].n_ctx;
        }
        fprintf(stderr,
                "zo_tune: pinned %zu shared context tokens, prefill drops "
                "%.0f%% (mean ctx %.1f -> %.1f tokens)\n",
                pin_len,
                100.0 * (double) pin_len / ((double) total / (double) n_ex),
                (double) total / (double) n_ex,
                (double) total / (double) n_ex - (double) pin_len);
    }

    if (holdout > 0) {
        const double h0 = nll_range(sess, ex, n_train, n_ex, pin_len);
        fprintf(stderr, "step 0  holdout %.4f (initial)\n", h0);
    }

    uint64_t rng   = seed | 1u;
    size_t   n_bad = 0;
    for (size_t t = 0; t < steps; t++) {
        /* Refresh the cached prefix against the gains the run has reached.
         * Safe here and only here: between steps g == base, so the prefix
         * is recomputed at the current parameters rather than at a probe. */
        if (pin_len > 0 && t > 0 && t % REPIN_EVERY_STEPS == 0) {
            (void) geist_session_pin_prefix(sess, pin_len, ex[0].ctx);
        }
        for (size_t b = 0; b < c.batch; b++) {
            idx[b] = (size_t) (xs64(&rng) % n_train);
        }
        const double loss = zo_step(sess, g, base, n_gains, ex, idx, &c, pin_len, &rng);
        if (isnan(loss)) {
            if (++n_bad > 8) {
                fprintf(stderr, "zo_tune: too many failed steps, aborting\n");
                goto done;
            }
            continue;
        }
        if (t % 50 == 0) {
            fprintf(stderr, "step %zu  train %.4f\n", t, loss);
        }
        if (eval_every > 0 && holdout > 0 && t > 0 && t % eval_every == 0) {
            /* Evaluate the current vector, not a perturbed one — zo_step
             * leaves g == base on exit. Re-pin first so the reported number
             * is not measured against a prefix cached at older gains. */
            if (pin_len > 0) {
                (void) geist_session_pin_prefix(sess, pin_len, ex[0].ctx);
            }
            const double h = nll_range(sess, ex, n_train, n_ex, pin_len);
            fprintf(stderr, "step %zu  holdout %.4f\n", t, h);
        }
    }

    if (holdout > 0) {
        if (pin_len > 0) {
            (void) geist_session_pin_prefix(sess, pin_len, ex[0].ctx);
        }
        const double h1 = nll_range(sess, ex, n_train, n_ex, pin_len);
        fprintf(stderr, "step %zu  holdout %.4f (final)\n", steps, h1);
    }

    {
        FILE *f = fopen(out, "wb");
        if (f == nullptr || fwrite(base, sizeof *base, n_gains, f) != n_gains) {
            fprintf(stderr, "zo_tune: writing %s failed\n", out);
            if (f != nullptr) {
                fclose(f);
            }
            goto done;
        }
        fclose(f);
    }
    fprintf(stderr, "wrote %zu gains (%zu bytes) -> %s\n", n_gains, n_gains * sizeof *base, out);
    rc = 0;

done:
    free(idx);
    free(base);
    examples_free(ex, n_ex);
    if (sess != nullptr) {
        geist_session_destroy(sess);
    }
    if (model != nullptr) {
        geist_model_destroy(model);
    }
    if (be != nullptr) {
        geist_backend_destroy(be);
    }
    return rc;
}
