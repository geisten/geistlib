/*
 * bench_nll — teacher-forced negative log-likelihood of a text (#78 phase 3).
 *
 * Feeds the text one token at a time (decode shapes) and adds up
 * -log p(next token) from the logits after each one; prints the mean NLL
 * and the perplexity. Two placements of the same model agree on quality
 * when their perplexities agree — a sharper signal than greedy tokens for a
 * mixture-of-experts model, whose routing turns tiny numeric differences
 * into different continuations.
 *
 *   bench_nll <model.gguf> <text file> [max tokens, default 512]
 *
 * GEIST_BENCH_BACKEND picks the backend (default: the engine's choice);
 * GEIST_BENCH_SPLIT_BACKEND (+ GEIST_BENCH_SPLIT_LAYERS, "auto" or a count)
 * adds a split backend, which for a mixture-of-experts model holds its
 * hottest experts. Output: one JSON line on stdout.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr)
        return nullptr;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = n >= 0 ? malloc((size_t) n + 1) : nullptr;
    if (buf != nullptr && fread(buf, 1, (size_t) n, f) == (size_t) n)
        buf[n] = '\0';
    else {
        free(buf);
        buf = nullptr;
    }
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: bench_nll <model.gguf> <text file> [max tokens]\n");
        return GEIST_TEST_ERROR;
    }
    const size_t max_tokens = argc > 3 ? (size_t) strtoull(argv[3], nullptr, 10) : 512;
    char        *text       = read_file(argv[2]);
    if (text == nullptr) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return GEIST_TEST_ERROR;
    }
    struct geist_backend *be    = nullptr;
    const char           *bname = getenv("GEIST_BENCH_BACKEND");
    if (geist_backend_create(bname, nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_session_opts lo    = {0};
    const char               *sname = getenv("GEIST_BENCH_SPLIT_BACKEND"),
               *sn                  = getenv("GEIST_BENCH_SPLIT_LAYERS");
    if (sname != nullptr && sname[0] != '\0' && sn != nullptr) {
        if (geist_backend_create(sname, nullptr, nullptr, &lo.split_backend) != GEIST_OK) {
            fprintf(stderr, "split backend create failed: %s\n", geist_last_create_error());
            return GEIST_TEST_ERROR;
        }
        lo.split_layers =
                strcmp(sn, "auto") == 0 ? GEIST_SPLIT_AUTO : (size_t) strtoull(sn, nullptr, 10);
    }
    struct geist_model             *m   = nullptr;
    struct geist_session           *s   = nullptr;
    geist_token_t                  *ids = malloc((max_tokens + 1) * sizeof *ids);
    size_t                          n   = 0;
    const struct geist_session_opts so  = {.top_p = 1.0f, .max_seq_len = max_tokens + 8};
    if (ids == nullptr || geist_model_load_with_opts(argv[1], be, &lo, &m) != GEIST_OK ||
        geist_session_create(m, be, &so, &s) != GEIST_OK) {
        fprintf(stderr, "load failed: %s\n", geist_backend_errmsg(be));
        return GEIST_TEST_ERROR;
    }
    /* No more tokens than bytes, plus room for BOS/EOS. */
    size_t         total = 0;
    geist_token_t *all   = malloc((strlen(text) + 16) * sizeof *all);
    if (all == nullptr ||
        geist_session_tokenize(s, text, strlen(text) + 16, all, &total) != GEIST_OK) {
        fprintf(stderr, "tokenize failed\n");
        return GEIST_TEST_ERROR;
    }
    n = total < max_tokens ? total : max_tokens;
    memcpy(ids, all, n * sizeof *ids);

    double sum = 0.0;
    size_t cnt = 0, nl = 0;
    for (size_t i = 0; i + 1 < n; i++) {
        if (geist_session_prefill_tokens(s, 1, &ids[i]) != GEIST_OK) {
            fprintf(stderr, "prefill failed at %zu: %s\n", i, geist_backend_errmsg(be));
            return GEIST_TEST_ERROR;
        }
        const float *lg = geist_session_peek_logits(&nl, s);
        if (lg == nullptr || (size_t) ids[i + 1] >= nl) {
            fprintf(stderr, "no logits at %zu\n", i);
            return GEIST_TEST_ERROR;
        }
        double mx = lg[0];
        for (size_t v = 1; v < nl; v++)
            mx = lg[v] > mx ? lg[v] : mx;
        double z = 0.0;
        for (size_t v = 0; v < nl; v++)
            z += exp((double) lg[v] - mx);
        sum += mx + log(z) - (double) lg[ids[i + 1]];
        cnt++;
    }
    printf("{\"tokens\":%zu,\"nll\":%.6f,\"ppl\":%.4f}\n",
           cnt,
           sum / (double) cnt,
           exp(sum / (double) cnt));
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    if (lo.split_backend != nullptr)
        geist_backend_destroy(lo.split_backend);
    free(ids);
    free(all);
    free(text);
    return GEIST_TEST_PASS;
}
