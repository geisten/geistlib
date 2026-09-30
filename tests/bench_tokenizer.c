/*
 * bench_tokenizer — encode time against input length for each merge-driven
 * tokenizer: gguf_tokenizer in gpt2 mode (ordinary words, and one unbroken
 * run of letters, which the pre-tokenizer keeps as a single chunk), SPM
 * mode (the whole text between specials is one chunk), unigram mode, and
 * sp_bpe_tokenizer. Synthetic vocabularies from tokenizer_fixtures.h.
 *
 * Each row prints the median encode time and an FNV-1a hash of the ids:
 * two builds that print the same hashes produced the same tokens.
 *
 * Usage: bench_tokenizer [max_chars]   (default 65536)
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "gguf_reader.h"
#include "gguf_tokenizer.h"
#include "heap.h"
#include "sp_bpe_tokenizer.h"
#include "tokenizer_fixtures.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec * 1e-6;
}

static uint64_t fnv_ids(size_t n, const void *ids, size_t elem) {
    uint64_t       h = 0xcbf29ce484222325ULL;
    const uint8_t *p = ids;
    for (size_t i = 0; i < n * elem; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

struct row {
    double   ms;
    size_t   n_ids;
    uint64_t hash;
};

/* One gguf_tokenizer mode over one text: median of up to 5 runs (1 when a
 * run takes over a second). */
static struct row time_gguf(const struct gguf_tokenizer *tok, const char *text, size_t n) {
    struct row r   = {.ms = -1.0};
    int32_t   *ids = malloc((n + 16) * sizeof *ids);
    double     t[5];
    int        runs = 0;
    for (; runs < 5; runs++) {
        size_t       n_ids = 0;
        const double t0    = now_ms();
        if (gguf_tokenizer_encode(tok, text, n + 16, ids, &n_ids) != GEIST_OK) {
            free(ids);
            return r;
        }
        t[runs] = now_ms() - t0;
        r.n_ids = n_ids;
        if (t[runs] > 1000.0) {
            runs++;
            break;
        }
    }
    qsort(t, (size_t) runs, sizeof *t, cmp_double);
    r.ms   = t[runs / 2];
    r.hash = fnv_ids(r.n_ids, ids, sizeof *ids);
    free(ids);
    return r;
}

static struct row time_sp_bpe(const struct sp_bpe_tokenizer *tok, const char *text) {
    struct row r = {.ms = -1.0};
    double     t[5];
    int        runs = 0;
    for (; runs < 5; runs++) {
        uint32_t    *ids   = nullptr;
        size_t       n_ids = 0;
        const double t0    = now_ms();
        if (!sp_bpe_tokenizer_encode(tok, text, &ids, &n_ids)) {
            return r;
        }
        t[runs] = now_ms() - t0;
        r.n_ids = n_ids;
        r.hash  = fnv_ids(n_ids, ids, sizeof *ids);
        safe_free((void **) &ids);
        if (t[runs] > 1000.0) {
            runs++;
            break;
        }
    }
    qsort(t, (size_t) runs, sizeof *t, cmp_double);
    r.ms = t[runs / 2];
    return r;
}

static bool load_gguf(struct gguf_tokenizer *tok, const struct tf_buf *g) {
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open_memory(g->b, g->n, &err);
    if (ctx == nullptr) {
        fprintf(stderr, "gguf_open_memory: %s\n", err != nullptr ? err : "?");
        return false;
    }
    const bool ok = gguf_tokenizer_load_copy(tok, ctx);
    gguf_close(ctx);
    return ok;
}

int main(int argc, char **argv) {
    const size_t max_chars = argc > 1 && atol(argv[1]) > 0 ? (size_t) atol(argv[1]) : 65536;

    struct tf_vocab spm_v  = tf_make_vocab("\xe2\x96\x81", true); /* ▁ */
    struct tf_vocab gpt2_v = tf_make_vocab("\xc4\xa0", false);    /* Ġ */
    struct tf_buf   g_gpt2 = tf_gguf(&gpt2_v, "gpt2", true);
    struct tf_buf   g_spm  = tf_gguf(&spm_v, "llama", true);
    struct tf_buf   g_uni  = tf_gguf(&spm_v, "llama", false);
    struct tf_buf   bin    = tf_sp_bpe_bin(&spm_v);

    struct gguf_tokenizer t_gpt2, t_spm, t_uni;
    if (!load_gguf(&t_gpt2, &g_gpt2) || !load_gguf(&t_spm, &g_spm) || !load_gguf(&t_uni, &g_uni)) {
        fprintf(stderr, "tokenizer load failed\n");
        return 1;
    }
    char path[] = "/tmp/bench_tokenizer_XXXXXX";
    int  fd     = mkstemp(path);
    if (fd < 0 || write(fd, bin.b, bin.n) != (ssize_t) bin.n) {
        fprintf(stderr, "cannot write %s\n", path);
        return 1;
    }
    close(fd);
    struct sp_bpe_tokenizer *t_sp  = nullptr;
    const bool               sp_ok = sp_bpe_tokenizer_load(&t_sp, path);
    unlink(path);
    if (!sp_ok) {
        fprintf(stderr, "sp_bpe load failed\n");
        return 1;
    }

    printf("{\"bench\":\"tokenizer\",\"rows\":[\n");
    bool first = true;
    for (size_t n = 1024; n <= max_chars; n *= 4) {
        char *words   = tf_text(n, 0x9E3779B9u ^ (uint32_t) n, false);
        char *letters = tf_text(n, 0x85EBCA6Bu ^ (uint32_t) n, true);
        const struct {
            const char *name;
            struct row  r;
        } rows[] = {
                {"gpt2 words", time_gguf(&t_gpt2, words, n)},
                {"gpt2 one run", time_gguf(&t_gpt2, letters, n)},
                {"spm words", time_gguf(&t_spm, words, n)},
                {"unigram words", time_gguf(&t_uni, words, n)},
                {"sp_bpe words", time_sp_bpe(t_sp, words)},
        };
        for (size_t i = 0; i < sizeof rows / sizeof *rows; i++) {
            printf("%s {\"mode\":\"%s\",\"chars\":%zu,\"ids\":%zu,\"ms\":%.3f,\"us_per_char\":%.4f,"
                   "\"hash\":\"%016llx\"}",
                   first ? " " : ",\n ",
                   rows[i].name,
                   n,
                   rows[i].r.n_ids,
                   rows[i].r.ms,
                   rows[i].r.ms * 1e3 / (double) n,
                   (unsigned long long) rows[i].r.hash);
            first = false;
        }
        free(words);
        free(letters);
    }
    printf("\n]}\n");

    sp_bpe_tokenizer_free(t_sp);
    gguf_tokenizer_unload(&t_gpt2);
    gguf_tokenizer_unload(&t_spm);
    gguf_tokenizer_unload(&t_uni);
    free(g_gpt2.b);
    free(g_spm.b);
    free(g_uni.b);
    free(bin.b);
    tf_free_vocab(&spm_v);
    tf_free_vocab(&gpt2_v);
    return 0;
}
