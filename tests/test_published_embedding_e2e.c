/*
 * test_published_embedding_e2e — the checkpoints as upstream ships them.
 *
 * tools/convert_bitnet_embedding.py exists because geistlib needed a GGUF it
 * could read. Microsoft also PUBLISHES one, ready-made, on the model page:
 *
 *   microsoft/bitnet-embedding-0.6b  bitnet-embeddings-0.6b-bf16-i2_s.gguf
 *
 * Point GEIST_EMBED_GGUF_PATH at it. SKIPs without it — the file is 428 MB
 * and not in CI.
 *
 * That file differs from ours in three ways that each used to be fatal: its
 * norm gammas are F16, its blocks carry 20 tensors, and it spells the
 * embedding metadata as gguf-py's numeric `{arch}.pooling_type` rather than
 * our `bitnet.embedding.*` keys. This test is the end-to-end statement that
 * none of that matters any more.
 *
 * The gate is the vendor's own published embedding. Their model card prints
 * the first seven components of the vector for one exact prompt, and that is
 * the only external ground truth available: upstream's llama-embedding
 * cannot run these checkpoints (its main branch ignores the *_norm_in
 * tensors and emits all-NaN), so there is no binary to diff against.
 *
 * Seven components is few, but it is enough to separate the two poolings —
 * mean lands at RMSE 8.5e-4 and last-token at 0.24 — which is the question
 * the checkpoint's own documentation leaves open: the card says last-token
 * in prose, the GGUF says mean, and the card's reproducible invocation
 * passes no override, so what it printed is what the key says.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_TOK = 512, MAX_DIM = 4096 };

/* Tokenize and wrap in whatever specials the model's tokenizer metadata
 * says it was trained with. geist_session_tokenize returns content tokens
 * by design, so the wrapping is the caller's — and for a pooled vector it
 * is not cosmetic: pooling over a sequence one token short of upstream's is
 * a different vector. */
static size_t prefill_wrapped(struct geist_session *s,
                              struct geist_model   *m,
                              const char           *text,
                              enum geist_status    *out_status) {
    geist_token_t ids[MAX_TOK];
    size_t        n = 0;
    *out_status     = geist_session_reset(s);
    if (*out_status != GEIST_OK) {
        return 0;
    }
    const geist_token_t bos = geist_model_bos_token(m);
    const geist_token_t eos = geist_model_eos_token(m);
    size_t              k   = 0;
    if (geist_model_add_bos(m) && bos != GEIST_TOKEN_NONE) {
        ids[k++] = bos;
    }
    *out_status = geist_session_tokenize(s, text, MAX_TOK - k - 1, ids + k, &n);
    if (*out_status != GEIST_OK) {
        return 0;
    }
    k += n;
    if (geist_model_add_eos(m) && eos != GEIST_TOKEN_NONE) {
        ids[k++] = eos;
    }
    *out_status = geist_session_prefill_tokens(s, k, ids);
    return k;
}

static bool embed(struct geist_session *s,
                  struct geist_model   *m,
                  const char           *text,
                  size_t                dim,
                  float                 out[static MAX_DIM]) {
    enum geist_status st = GEIST_OK;
    if (prefill_wrapped(s, m, text, &st) == 0 || st != GEIST_OK) {
        return false;
    }
    size_t       n = 0;
    const float *v = geist_session_peek_embedding(&n, s);
    if (v == nullptr || n != dim) {
        return false;
    }
    memcpy(out, v, dim * sizeof(float));
    return true;
}

static double cosine(size_t n, const float a[static n], const float b[static n]) {
    double d = 0.0;
    for (size_t i = 0; i < n; i++) {
        d += (double) a[i] * (double) b[i];
    }
    return d;
}

int main(void) {
    const char *path = getenv("GEIST_EMBED_GGUF_PATH");
    if (path == nullptr || path[0] == '\0') {
        printf("SKIP: set GEIST_EMBED_GGUF_PATH to upstream's published "
               "bitnet-embeddings-0.6b-bf16-i2_s.gguf\n");
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

    /* The load itself is a check: F16 gammas and a 20-tensor block both used
     * to fail here, before any embedding was computed. */
    struct geist_model *m = nullptr;
    s                     = geist_model_load(path, be, &m);
    if (s != GEIST_OK) {
        fprintf(stderr, "load failed (%d): %s\n", (int) s, geist_backend_errmsg(be));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("  loaded %s (arch=%s)\n", path, geist_model_arch(m));

    int                             fails = 0;
    struct geist_session           *sess  = nullptr;
    const struct geist_session_opts opts  = {.max_seq_len = 512};
    if (geist_session_create(m, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "session create failed: %s\n", geist_backend_errmsg(be));
        geist_model_destroy(m);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* The width comes from the first embedding: peek_embedding reports it,
     * and an embedding model that reports none is the failure this whole
     * test exists to catch. */
    size_t            dim = 0;
    enum geist_status st  = GEIST_OK;
    if (prefill_wrapped(sess, m, "query: What is BitNet?", &st) == 0 || st != GEIST_OK) {
        fprintf(stderr, "prefill failed: %s\n", geist_session_errmsg(sess));
        goto teardown_fail;
    }
    if (geist_session_peek_embedding(&dim, sess) == nullptr || dim == 0) {
        fprintf(stderr, "no pooled embedding — is the pooling metadata being read?\n");
        goto teardown_fail;
    }
    printf("  embedding dim=%zu\n", dim);
    fails = geist_expect(dim == 1024,
                         "GEIST_EMBED_GGUF_PATH is bitnet-embedding-0.6b — the reference "
                         "vector below is that checkpoint's");
    if (dim != 1024) {
        goto teardown_fail;
    }

    static float a[MAX_DIM], b[MAX_DIM];

    /* ---- the vendor's published vector -------------------------------- */
    {
        /* microsoft/bitnet-embedding-0.6b's card, verbatim:
         *   llama-embedding -p "query: What is BitNet?" --embd-normalize 2
         *   [[0.0239517, 0.6826404, -0.0000000, -0.0644535, 0.0613754,
         *     0.0473094, 0.0114330, ...]] */
        static const float CARD[] = {0.0239517f,
                                     0.6826404f,
                                     -0.0000000f,
                                     -0.0644535f,
                                     0.0613754f,
                                     0.0473094f,
                                     0.0114330f};
        enum { N_CARD = sizeof CARD / sizeof CARD[0] };

        fails += geist_expect(embed(sess, m, "query: What is BitNet?", dim, a),
                              "the reference prompt embeds");
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
        /* 5e-3 leaves room for upstream's kernels accumulating I2_S and F16
         * in a different order than ours; it is two orders of magnitude
         * below the gap to the wrong pooling. */
        fails += geist_expect(rmse < 5e-3,
                              "the pooled vector reproduces the vendor's published embedding");
    }

    /* ---- retrieval ----------------------------------------------------- *
     * A forward with a misplaced norm still yields a finite, unit-length,
     * deterministic vector. It does not yield a space where each question
     * finds its own answer among plausible distractors. */
    {
        static const char *const QUERIES[] = {
                "why does my loaf not rise?",
                "what makes the sea level change twice a day?",
                "how much of my instalment pays down the debt?",
        };
        static const char *const DOCS[] = {
                "Yeast ferments the sugars in dough and the carbon dioxide it releases "
                "inflates the gluten network, which is what lifts a loaf.",
                "The moon's gravity pulls the oceans toward it, and the earth's rotation "
                "carries every coastline through the resulting bulges twice a day.",
                "Each payment first covers the interest accrued on the outstanding "
                "balance; whatever is left reduces the principal.",
        };
        enum { N = sizeof QUERIES / sizeof QUERIES[0] };
        static float docs[N][MAX_DIM];

        bool ok = true;
        for (size_t i = 0; i < N && ok; i++) {
            ok = embed(sess, m, DOCS[i], dim, docs[i]);
        }
        fails += geist_expect(ok, "the documents embed");

        size_t hits = 0;
        for (size_t q = 0; q < N && ok; q++) {
            if (!embed(sess, m, QUERIES[q], dim, b)) {
                ok = false;
                break;
            }
            size_t best  = 0;
            double score = -2.0;
            for (size_t d = 0; d < N; d++) {
                const double sc = cosine(dim, b, docs[d]);
                if (sc > score) {
                    score = sc;
                    best  = d;
                }
            }
            if (best == q) {
                hits++;
            } else {
                printf("  MISS q%zu -> doc%zu (cos %.4f)\n", q, best, score);
            }
        }
        printf("  retrieval@1: %zu/%d\n", hits, (int) N);
        fails += geist_expect(hits == N,
                              "every question retrieves its own answer — the pooled space is "
                              "semantic, not merely well-formed");
    }

    /* ---- determinism and unit length ----------------------------------- */
    {
        fails += geist_expect(embed(sess, m, "query: What is BitNet?", dim, b),
                              "the reference prompt embeds again");
        fails += geist_expect(memcmp(a, b, dim * sizeof(float)) == 0,
                              "the same tokens give a bit-identical vector");
        fails += geist_expect(fabs(cosine(dim, b, b) - 1.0) < 1e-4, "the result is L2-normalized");
    }

    geist_session_destroy(sess);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("published embedding GGUF: loads, pools as the vendor does, retrieves\n");
    return GEIST_TEST_PASS;

teardown_fail:
    geist_session_destroy(sess);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return GEIST_TEST_FAIL;
}
