/*
 * test_embed_e2e — geist_session_embed against a real embedding checkpoint.
 *
 * Point GEIST_EMBED_GGUF_PATH at microsoft/bitnet-embedding-0.6b (see
 * test_bitlinear_subln_e2e for the file name). SKIPs without it.
 *
 * The checks are chosen for what can actually break, not for what is easy
 * to assert:
 *
 *   - CHUNK INVARIANCE. Prefill runs in chunks of m_max and each chunk
 *     overwrites the residual scratch, so the pooling sum lives outside it.
 *     Get that wrong and short texts still look perfect while anything past
 *     one chunk silently averages the wrong rows. Two sessions, one chunked
 *     and one not, must agree to the last bit the arithmetic allows.
 *   - STATELESSNESS. embed resets the session coming and going; a
 *     generation started afterwards must not notice it ran.
 *   - SEMANTIC ORDER. A forward that is subtly wrong still produces finite,
 *     unit-length vectors. It does not produce a space where a paraphrase
 *     outranks an unrelated sentence.
 *   - The n_dim contract, determinism, and unit length.
 *
 *   - THE VENDOR'S PUBLISHED VECTOR. The model card prints the first seven
 *     components of the embedding for one exact prompt, and that is the only
 *     external ground truth there is: microsoft's own llama-embedding cannot
 *     run these checkpoints (its main branch ignores the *_norm_in tensors
 *     and emits all-NaN), so there is no binary to diff against.
 *
 *     The card attributes those numbers to its 270M sibling by defaulting
 *     the snippet's MODEL variable there. They are this model's: it carries
 *     a massive activation of 0.47 at component 1 on ordinary text and the
 *     published vector has 0.68 in that slot, where the 270M's spectrum is
 *     flat (largest component 2% of the energy). Measured here: RMSE 8.5e-4.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_TOK = 1024, MAX_DIM = 4096 };

/* Tokenize `text` and wrap it in the model's BOS/EOS, which is what an
 * embedding model is trained on and what geist_session_tokenize leaves to
 * the caller by design. */
static size_t tokenize_wrapped(struct geist_session *s,
                               struct geist_model   *m,
                               const char           *text,
                               geist_token_t         out[static MAX_TOK]) {
    geist_token_t raw[MAX_TOK];
    size_t        n_raw = 0;
    if (geist_session_tokenize(s, text, MAX_TOK, raw, &n_raw) != GEIST_OK) {
        return 0;
    }
    size_t              k   = 0;
    const geist_token_t bos = geist_model_bos_token(m);
    const geist_token_t eos = geist_model_eos_token(m);
    if (bos != GEIST_TOKEN_NONE) {
        out[k++] = bos;
    }
    if (n_raw > MAX_TOK - k - 1) {
        n_raw = MAX_TOK - k - 1;
    }
    memcpy(out + k, raw, n_raw * sizeof(geist_token_t));
    k += n_raw;
    if (eos != GEIST_TOKEN_NONE) {
        out[k++] = eos;
    }
    return k;
}

static double cosine(size_t n, const float a[static n], const float b[static n]) {
    double d = 0.0;
    for (size_t i = 0; i < n; i++) {
        d += (double) a[i] * (double) b[i];
    }
    return d;
}

static double max_abs_diff(size_t n, const float a[static n], const float b[static n]) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double) a[i] - (double) b[i]);
        worst          = d > worst ? d : worst;
    }
    return worst;
}

int main(void) {
    const char *path = getenv("GEIST_EMBED_GGUF_PATH");
    if (path == nullptr || path[0] == '\0') {
        printf("SKIP: set GEIST_EMBED_GGUF_PATH to a microsoft/bitnet-embedding-* GGUF\n");
        return GEIST_TEST_SKIP;
    }

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_model *m = nullptr;
    if (geist_model_load(path, be, &m) != GEIST_OK) {
        fprintf(stderr, "load failed: %s\n", geist_backend_errmsg(be));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    const size_t dim   = geist_model_embed_dim(m);
    int          fails = 0;
    fails +=
            geist_expect(dim > 0 && dim <= MAX_DIM, "geist_model_embed_dim reports a usable width");
    if (dim == 0 || dim > MAX_DIM) {
        geist_model_destroy(m);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("  %s embed_dim=%zu\n", geist_model_arch(m), dim);

    /* Two sessions over the same model: one whose prefill chunk is a single
     * row, one that swallows the whole text in one chunk. */
    struct geist_session *chunked = nullptr;
    struct geist_session *whole   = nullptr;
    /* m_max = 1 forces one chunk per token; 0 takes the model default, which
     * the backend caps (caps.max_m) — asking for more than that is refused,
     * so "unchunked" here means "the widest chunk this backend allows". */
    const struct geist_session_opts o_chunked = {.max_seq_len = 512, .m_max = 1};
    const struct geist_session_opts o_whole   = {.max_seq_len = 512};
    if (geist_session_create(m, be, &o_chunked, &chunked) != GEIST_OK ||
        geist_session_create(m, be, &o_whole, &whole) != GEIST_OK) {
        fprintf(stderr, "session create failed: %s\n", geist_backend_errmsg(be));
        geist_model_destroy(m);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    static float  a[MAX_DIM], b[MAX_DIM], d[MAX_DIM];
    geist_token_t ids[MAX_TOK];

    /* ---- n_dim contract ---------------------------------------------- */
    {
        const size_t n = tokenize_wrapped(whole, m, "hello", ids);
        fails += geist_expect(n > 0, "tokenizer produced ids");
        s = geist_session_embed(whole, dim + 1, n, ids, a);
        fails += geist_expect(s == GEIST_E_INVALID_ARG,
                              "a wrong n_dim is refused, not silently truncated");
    }

    /* ---- determinism + unit length ------------------------------------ */
    {
        const size_t n = tokenize_wrapped(whole, m, "The capital of France is Paris.", ids);
        s              = geist_session_embed(whole, dim, n, ids, a);
        fails += geist_expect(s == GEIST_OK, "embed succeeds");
        s = geist_session_embed(whole, dim, n, ids, b);
        fails += geist_expect(s == GEIST_OK, "embed succeeds again");
        fails += geist_expect(memcmp(a, b, dim * sizeof(float)) == 0,
                              "the same ids give a bit-identical vector");
        fails += geist_expect(fabs(cosine(dim, a, a) - 1.0) < 1e-4, "the result is L2-normalized");
    }

    /* ---- chunk-boundary invariance ------------------------------------ */
    {
        /* Long enough that m_max=1 forces many chunks and m_max=256 forces
         * exactly one — the pooling accumulator is the only thing that can
         * tell the two apart. */
        static const char LONG_TEXT[] =
                "Retrieval augmented generation stitches a language model to a document store. "
                "The store is queried with an embedding of the user's question, the nearest "
                "passages are pasted into the prompt, and the model answers from them. "
                "Most of the engineering is in the store, not the model.";
        const size_t n = tokenize_wrapped(whole, m, LONG_TEXT, ids);
        printf("  chunk-invariance text: %zu tokens\n", n);
        fails += geist_expect(n > 32, "the long text really is longer than one chunk");

        s = geist_session_embed(whole, dim, n, ids, a);
        fails += geist_expect(s == GEIST_OK, "unchunked embed succeeds");
        s = geist_session_embed(chunked, dim, n, ids, b);
        fails += geist_expect(s == GEIST_OK, "chunked embed succeeds");

        const double worst = max_abs_diff(dim, a, b);
        const double cos   = cosine(dim, a, b);
        printf("  default m_max vs m_max=1: max|diff|=%.2e cos=%.9f\n", worst, cos);
        /* The tolerance is loose ON PURPOSE, and it is worth knowing why.
         * geist's prefill is ALREADY not chunk-invariant for quantized
         * weights — the GEMM's M dimension selects the kernel, and the
         * kernel decides the accumulation order. Measured on this tree,
         * m_max=1 vs the default over the same 56 tokens moves the final
         * logits by mean 1.46 on gemma4-e2b Q4_K_M and 0.33 on qwen3-0.6b
         * Q8_0, both stock models, both unchanged by the embed work. A
         * pooled vector inherits that.
         *
         * So this bound cannot be tightened to bit-equality without
         * fixing the kernel dispatch first. It is still the check that
         * matters: a broken accumulator — last chunk only, rows counted
         * twice, wrong offset — lands far below 0.99, while the honest
         * numerical spread sits at 0.9990 (bitnet-embedding-270m) and
         * exactly 1.0 (bitnet-embedding-0.6b, whose I2_S path happens to
         * be M-invariant and is therefore the control for this test). */
        fails += geist_expect(cos > 0.99, "chunking does not change the pooled vector");
    }

    /* ---- the vendor's published vector --------------------------------- */
    fails += geist_expect(dim == 1024,
                          "GEIST_EMBED_GGUF_PATH points at bitnet-embedding-0.6b — the "
                          "reference vector below is that checkpoint's");
    if (dim == 1024) {
        /* microsoft/bitnet-embedding-0.6b's card, verbatim:
         *   llama-embedding -p "query: What is BitNet?" --embd-normalize 2
         *   [[0.0239517, 0.6826404, -0.0000000, -0.0644535, 0.0613754,
         *     0.0473094, 0.0114330, ...]]
         * Seven components is all it prints; it is still enough to separate
         * mean pooling (RMSE 8.5e-4) from last-token (0.24), which is the
         * question the checkpoint's own documentation leaves open — the GGUF
         * key says mean, the prose says last-token. The file is right. */
        static const float CARD[] = {0.0239517f,
                                     0.6826404f,
                                     -0.0000000f,
                                     -0.0644535f,
                                     0.0613754f,
                                     0.0473094f,
                                     0.0114330f};
        enum { N_CARD = sizeof CARD / sizeof CARD[0] };

        const size_t n = tokenize_wrapped(whole, m, "query: What is BitNet?", ids);
        s              = geist_session_embed(whole, dim, n, ids, a);
        fails += geist_expect(s == GEIST_OK, "reference prompt embeds");

        double sq = 0.0;
        for (size_t i = 0; i < N_CARD; i++) {
            const double d_i = (double) a[i] - (double) CARD[i];
            sq += d_i * d_i;
        }
        const double rmse = sqrt(sq / (double) N_CARD);
        printf("  vs published vector: rmse=%.2e  (mine:", rmse);
        for (size_t i = 0; i < N_CARD; i++) {
            printf(" %.6f", (double) a[i]);
        }
        printf(")\n");
        /* 5e-3 leaves room for the vendor's kernels accumulating I2_S and
         * F16 in a different order than ours; it is two orders of magnitude
         * below the gap to the wrong pooling. */
        fails += geist_expect(rmse < 5e-3,
                              "the pooled vector reproduces the vendor's published embedding");
    }

    /* ---- retrieval ------------------------------------------------------ *
     * The strongest cheap quality signal. A forward with a misplaced norm
     * still yields finite, unit-length, deterministic, chunk-invariant
     * vectors — everything above passes. What it does NOT yield is a space
     * where each query's own answer is its nearest neighbour among five
     * plausible distractors. Retrieval@1 must be perfect on a set this easy;
     * anything less means the embeddings are decorative. */
    {
        static const char *const QUERIES[] = {
                "How do I make bread rise?",
                "What causes tides in the ocean?",
                "Why is my laptop battery draining fast?",
                "When should I prune apple trees?",
                "How is interest on a mortgage calculated?",
        };
        static const char *const DOCS[] = {
                "Yeast ferments the sugars in dough and the carbon dioxide it releases "
                "expands the gluten network, lifting the loaf.",
                "The moon's gravity pulls the oceans toward it, and the earth's rotation "
                "carries coastlines through the resulting bulges twice a day.",
                "Background processes, a bright display and an aging cell all shorten "
                "runtime; check which applications are consuming power.",
                "Late winter, while the tree is dormant and before the buds swell, is the "
                "right moment to cut back an apple tree.",
                "Each payment covers the interest accrued on the outstanding balance since "
                "the last one, and whatever remains reduces the principal.",
        };
        enum { N_PAIRS = sizeof QUERIES / sizeof QUERIES[0] };
        static float doc_vec[N_PAIRS][MAX_DIM];

        for (size_t i = 0; i < N_PAIRS && s == GEIST_OK; i++) {
            const size_t n = tokenize_wrapped(whole, m, DOCS[i], ids);
            s              = geist_session_embed(whole, dim, n, ids, doc_vec[i]);
        }
        fails += geist_expect(s == GEIST_OK, "document embeds succeed");

        size_t hits         = 0;
        double worst_margin = 1.0;
        for (size_t q = 0; q < N_PAIRS && s == GEIST_OK; q++) {
            const size_t n = tokenize_wrapped(whole, m, QUERIES[q], ids);
            s              = geist_session_embed(whole, dim, n, ids, a);
            if (s != GEIST_OK) {
                break;
            }
            size_t best       = 0;
            double best_score = -2.0, runner_up = -2.0;
            for (size_t d_i = 0; d_i < N_PAIRS; d_i++) {
                const double sc = cosine(dim, a, doc_vec[d_i]);
                if (sc > best_score) {
                    runner_up  = best_score;
                    best_score = sc;
                    best       = d_i;
                } else if (sc > runner_up) {
                    runner_up = sc;
                }
            }
            if (best == q) {
                hits++;
                const double margin = best_score - runner_up;
                worst_margin        = margin < worst_margin ? margin : worst_margin;
            } else {
                printf("  MISS q%zu -> doc%zu (cos %.4f)\n", q, best, best_score);
            }
        }
        fails += geist_expect(s == GEIST_OK, "query embeds succeed");
        printf("  retrieval@1: %zu/%d  worst margin over the runner-up: %.4f\n",
               hits,
               (int) N_PAIRS,
               worst_margin);
        fails += geist_expect(hits == N_PAIRS,
                              "every query retrieves its own answer first — the pooled space "
                              "is semantic, not merely well-formed");
    }

    /* ---- statelessness -------------------------------------------------- */
    {
        geist_token_t t_clean = 0, t_after = 0;
        s = geist_session_reset(whole);
        s = (s == GEIST_OK) ? geist_session_set_prompt(whole, "The capital of France is") : s;
        s = (s == GEIST_OK) ? geist_session_decode_step(whole, &t_clean) : s;

        const size_t n = tokenize_wrapped(whole, m, "an unrelated sentence", ids);
        s              = (s == GEIST_OK) ? geist_session_embed(whole, dim, n, ids, d) : s;

        s = (s == GEIST_OK) ? geist_session_reset(whole) : s;
        s = (s == GEIST_OK) ? geist_session_set_prompt(whole, "The capital of France is") : s;
        s = (s == GEIST_OK) ? geist_session_decode_step(whole, &t_after) : s;
        fails += geist_expect(s == GEIST_OK, "generation still works around an embed");
        fails += geist_expect(t_clean == t_after,
                              "an embed leaves no trace in the session's decode state");
    }

    geist_session_destroy(chunked);
    geist_session_destroy(whole);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("embed: chunk-invariant, stateless, semantically ordered\n");
    return GEIST_TEST_PASS;
}
