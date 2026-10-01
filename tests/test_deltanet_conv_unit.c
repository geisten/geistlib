/*
 * test_deltanet_conv_unit — the chunked DeltaNet prefill's causal conv.
 *
 * transformer_dn_conv_silu_row (layer_deltanet.c) picks a token's K input
 * rows from the K - 1 history rows and the chunk's own rows, then runs a
 * channel loop that the compiler vectorizes: K == 4 has a path of its
 * own, any other K adds tap after tap. Two checks:
 *
 *   - the row against a double-precision reference, for K = 1..5, channel
 *     counts that leave vector tails, and every token of chunks shorter
 *     and longer than the history, inputs large enough to reach both
 *     branches of silu;
 *   - a chunked prefill against the sequential recurrence
 *     (GEIST_DN_SEQ_PREFILL=1, read at model load), whose conv is
 *     separate code, on a Qwen3.5-style hybrid (model_fixtures.h) with a
 *     conv kernel of 4 and of 3: a 12-token prompt in chunks of 5, 5 and
 *     2, and one whose first chunk, 2 tokens, is shorter than the
 *     history. The logits after the prompt must agree. This is
 *     test_deltanet_chunk_int's oracle without a model to fetch.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "src/archs/transformer/forward/internal.h"

#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rng = 0x2545F4914F6CDD1Dull;

/* Uniform in [-s, s). */
static float urand(float s) {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    const unsigned long long r = rng * 0x2545F4914F6CDD1Dull;
    return (float) ((double) (r >> 11) / (double) (1ull << 53) * 2.0 - 1.0) * s;
}

/* Worst |y - ref| over a chunk of seq tokens, relative to |ref| plus the
 * taps' own size sum_j |w_j x_j|: float adds lose bits to the summands,
 * not to a sum that cancels. */
static double row_case(size_t K, size_t convd, size_t seq) {
    const size_t hist = K - 1;
    float       *old  = malloc((hist + 1) * convd * sizeof *old);
    float       *qkv  = malloc(seq * convd * sizeof *qkv);
    float       *w    = malloc(convd * K * sizeof *w);
    float       *y    = malloc(convd * sizeof *y);
    if (old == nullptr || qkv == nullptr || w == nullptr || y == nullptr) {
        fprintf(stderr, "FAIL: alloc\n");
        exit(1);
    }
    for (size_t i = 0; i < hist * convd; i++)
        old[i] = urand(8.0f);
    for (size_t i = 0; i < seq * convd; i++)
        qkv[i] = urand(8.0f);
    for (size_t i = 0; i < convd * K; i++)
        w[i] = urand(1.5f);

    double worst = 0.0;
    for (size_t t = 0; t < seq; t++) {
        transformer_dn_conv_silu_row(t, K, convd, old, qkv, w, y);
        for (size_t c = 0; c < convd; c++) {
            double acc = 0.0, mag = 0.0;
            for (size_t j = 0; j < K; j++) {
                const size_t i = t + j;
                const double x = i < hist ? old[i * convd + c] : qkv[(i - hist) * convd + c];
                acc += (double) w[c * K + j] * x;
                mag += fabs((double) w[c * K + j] * x);
            }
            const double ref = acc / (1.0 + exp(-acc));
            const double err = fabs((double) y[c] - ref) / fmax(fabs(ref) + mag, 1e-30);
            worst            = fmax(worst, err);
        }
    }
    free(old);
    free(qkv);
    free(w);
    free(y);
    return worst;
}

/* ---- chunked vs sequential prefill on the fixture -------------------- */

constexpr size_t PROMPT = 12;

static const geist_token_t prompt[PROMPT] = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45};

/* The logits after prefilling the prompt, first `first` tokens on their
 * own (0: all at once). */
static bool
prefill_logits(struct geist_session *s, size_t first, size_t vocab, float out[static vocab]) {
    if (geist_session_reset(s) != GEIST_OK)
        return false;
    if (first > 0 && geist_session_prefill_tokens(s, first, prompt) != GEIST_OK)
        return false;
    if (geist_session_prefill_tokens(s, PROMPT - first, prompt + first) != GEIST_OK)
        return false;
    size_t       n = 0;
    const float *p = geist_session_peek_logits(&n, s);
    if (p == nullptr || n != vocab)
        return false;
    memcpy(out, p, vocab * sizeof *out);
    return true;
}

static struct geist_session *open_session(const struct tf_buf   *g,
                                          const char            *backend,
                                          struct geist_backend **be,
                                          struct geist_model   **m) {
    struct geist_session *s = nullptr;
    if (geist_backend_create(backend, nullptr, nullptr, be) != GEIST_OK || *be == nullptr)
        return nullptr;
    const struct geist_session_opts o = {.top_p = 1.0f, .m_max = 5};
    if (geist_model_load_from_memory(g->b, g->n, *be, m) != GEIST_OK ||
        geist_session_create(*m, *be, &o, &s) != GEIST_OK)
        return nullptr;
    return s;
}

/* Worst |chunked - sequential| over the logits, relative to the largest
 * sequential logit; -1 if the backend is not in this build. */
static double prefill_case(const char *backend, uint32_t dn_conv, size_t first) {
    struct tf_vocab       v     = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf         g     = mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
                                                                      .interval   = 4,
                                                                      .d_model    = 64,
                                                                      .heads      = 4,
                                                                      .kv_heads   = 2,
                                                                      .head_dim   = 16,
                                                                      .rope_dims  = 8,
                                                                      .ffn        = 128,
                                                                      .dn_k_heads = 2,
                                                                      .dn_v_heads = 4,
                                                                      .dn_head_k  = 16,
                                                                      .dn_head_v  = 16,
                                                                      .dn_conv    = dn_conv,
                                                                      .seed       = 7,
                                                                      .tok        = &v});
    const size_t          vocab = v.n_tok;
    float                *lc    = malloc(vocab * sizeof *lc);
    float                *ls    = malloc(vocab * sizeof *ls);
    struct geist_backend *bc = nullptr, *bs = nullptr;
    struct geist_model   *mc = nullptr, *ms = nullptr;

    unsetenv("GEIST_DN_SEQ_PREFILL");
    struct geist_session *sc = open_session(&g, backend, &bc, &mc);
    setenv("GEIST_DN_SEQ_PREFILL", "1", 1);
    struct geist_session *ss = sc != nullptr ? open_session(&g, backend, &bs, &ms) : nullptr;
    unsetenv("GEIST_DN_SEQ_PREFILL");

    double rel = -1.0;
    if (sc == nullptr && bc == nullptr) {
        rel = -1.0; /* not in this build */
    } else if (lc == nullptr || ls == nullptr || sc == nullptr || ss == nullptr ||
               !prefill_logits(sc, first, vocab, lc) || !prefill_logits(ss, first, vocab, ls)) {
        fprintf(stderr, "FAIL: %s conv %u: model, session or prefill\n", backend, dn_conv);
        rel = 1e30;
    } else {
        double md = 0.0, scale = 1e-6;
        for (size_t i = 0; i < vocab; i++) {
            md    = fmax(md, fabs((double) lc[i] - ls[i]));
            scale = fmax(scale, fabs((double) ls[i]));
        }
        rel = md / scale;
    }
    geist_session_destroy(sc);
    geist_session_destroy(ss);
    geist_model_destroy(mc);
    geist_model_destroy(ms);
    geist_backend_destroy(bc);
    geist_backend_destroy(bs);
    free(lc);
    free(ls);
    free(g.b);
    tf_free_vocab(&v);
    return rel;
}

/* The taps' float adds and silu through a vector expf (and a reciprocal
 * and Newton step for the division under -ffast-math) are a few ulp:
 * 1.2e-7 measured with gcc 14 on x86-64. A wrong row, tap or weight is
 * off by percent or more. */
constexpr double ROW_TOL = 2e-6;

/* The two prefills round differently through 4 layers: up to 1.0e-6 of
 * the largest logit on cpu_scalar and cpu_neon; cpu_x86 quantizes the
 * activations to int8, which rounds most of the difference away. A conv
 * reading the wrong token's rows moved them by 0.6. */
constexpr double PREFILL_TOL = 1e-4;

int main(void) {
    int fails = 0;

    static const size_t Ks[]     = {1, 2, 3, 4, 5};
    static const size_t convds[] = {1, 9, 64, 131};
    static const size_t seqs[]   = {1, 2, 5};
    double              worst    = 0.0;
    for (size_t a = 0; a < sizeof Ks / sizeof Ks[0]; a++)
        for (size_t b = 0; b < sizeof convds / sizeof convds[0]; b++)
            for (size_t c = 0; c < sizeof seqs / sizeof seqs[0]; c++) {
                const double e = row_case(Ks[a], convds[b], seqs[c]);
                worst          = fmax(worst, e);
                if (e > ROW_TOL) {
                    fprintf(stderr,
                            "FAIL: conv row K=%zu convd=%zu seq=%zu: rel err %.2e > %.0e\n",
                            Ks[a],
                            convds[b],
                            seqs[c],
                            e,
                            ROW_TOL);
                    fails++;
                }
            }
    printf("conv row: worst rel err %.2e (K 1..5, convd 1..131, every token)\n", worst);

    static const char *const backends[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};
    static const uint32_t    convs[]    = {4, 3};
    static const size_t      firsts[]   = {0, 2};
    int                      ran        = 0;
    for (size_t i = 0; i < sizeof backends / sizeof backends[0]; i++)
        for (size_t k = 0; k < sizeof convs / sizeof convs[0]; k++)
            for (size_t f = 0; f < sizeof firsts / sizeof firsts[0]; f++) {
                const double rel = prefill_case(backends[i], convs[k], firsts[f]);
                if (rel < 0.0)
                    continue;
                ran++;
                printf("%s conv %u, first chunk %zu: chunked vs sequential logits rel %.2e\n",
                       backends[i],
                       convs[k],
                       firsts[f] != 0 ? firsts[f] : (size_t) 5,
                       rel);
                if (rel > PREFILL_TOL) {
                    fprintf(stderr, "FAIL: chunked prefill != sequential (rel %.2e)\n", rel);
                    fails++;
                }
            }
    if (ran == 0)
        printf("note: no CPU backend in this build ran the prefill check\n");
    if (fails == 0)
        printf("PASS: DeltaNet conv row matches the reference; chunked prefill matches "
               "sequential\n");
    return fails == 0 ? 0 : 1;
}
