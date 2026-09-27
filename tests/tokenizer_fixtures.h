/*
 * tokenizer_fixtures.h — synthetic tokenizers for the merge-engine test and
 * bench_tokenizer: in-memory GGUFs for the gpt2 (byte-level BPE), SPM
 * (merges over ▁-normalized text) and unigram (scores, no merges) modes of
 * gguf_tokenizer, and a tokenizer.bin for sp_bpe_tokenizer.
 *
 * The vocabulary is built like a trained BPE's: every word of a fixed list
 * is spelled out as a chain of merges from its first character (with the
 * word-start marker) to the whole word, so ordinary text merges down to one
 * token per word and a run of letters exercises long merge chains. Unigram
 * scores favour longer pieces. Header-only; each includer uses what it
 * needs.
 */
#ifndef GEIST_TESTS_TOKENIZER_FIXTURES_H
#define GEIST_TESTS_TOKENIZER_FIXTURES_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const TF_WORDS[] = {
        "the",   "of",    "and",    "to",     "in",    "is",    "that",  "for",   "it",
        "as",    "was",   "with",   "be",     "by",    "on",    "not",   "he",    "this",
        "are",   "or",    "his",    "from",   "at",    "which", "but",   "have",  "an",
        "had",   "they",  "you",    "were",   "their", "one",   "all",   "we",    "can",
        "her",   "has",   "there",  "been",   "if",    "more",  "when",  "will",  "would",
        "who",   "so",    "no",     "tokens", "merge", "rank",  "heap",  "model", "text",
        "input", "sting", "string", "then",   "these", "those", "other", "into",  "over",
};
enum { TF_N_WORDS = sizeof TF_WORDS / sizeof TF_WORDS[0] };

/* ---- growable byte buffer ------------------------------------------------ */

struct tf_buf {
    uint8_t *b;
    size_t   n, cap;
};

static void tf_put(struct tf_buf *o, const void *p, size_t n) {
    if (o->n + n > o->cap) {
        size_t cap = o->cap ? o->cap : 4096;
        while (cap < o->n + n) {
            cap *= 2;
        }
        uint8_t *nb = realloc(o->b, cap);
        if (nb == nullptr) {
            fprintf(stderr, "tokenizer_fixtures: out of memory\n");
            exit(1);
        }
        o->b   = nb;
        o->cap = cap;
    }
    memcpy(o->b + o->n, p, n);
    o->n += n;
}
static void tf_u16(struct tf_buf *o, uint16_t v) {
    tf_put(o, &v, 2);
}
static void tf_u32(struct tf_buf *o, uint32_t v) {
    tf_put(o, &v, 4);
}
static void tf_u64(struct tf_buf *o, uint64_t v) {
    tf_put(o, &v, 8);
}
static void tf_f32(struct tf_buf *o, float v) {
    tf_put(o, &v, 4);
}
static void tf_gstr(struct tf_buf *o, const char *s, size_t n) {
    tf_u64(o, n);
    tf_put(o, s, n);
}

/* ---- the token and merge lists ------------------------------------------- */

struct tf_vocab {
    char   **tok; /* NUL-terminated token strings */
    float   *score;
    size_t   n_tok;
    char   **merge_l, **merge_r; /* merge m joins merge_l[m] + merge_r[m] */
    size_t   n_merge;
    uint32_t unk;
};

static void tf_free_vocab(struct tf_vocab *v) {
    for (size_t i = 0; i < v->n_tok; i++) {
        free(v->tok[i]);
    }
    for (size_t i = 0; i < v->n_merge; i++) {
        free(v->merge_l[i]);
        free(v->merge_r[i]);
    }
    free(v->tok);
    free(v->score);
    free(v->merge_l);
    free(v->merge_r);
    memset(v, 0, sizeof *v);
}

static char *tf_strndup(const char *s, size_t n) {
    char *d = malloc(n + 1);
    if (d == nullptr) {
        exit(1);
    }
    memcpy(d, s, n);
    d[n] = '\0';
    return d;
}

static bool tf_has_tok(const struct tf_vocab *v, const char *s) {
    for (size_t i = 0; i < v->n_tok; i++) {
        if (strcmp(v->tok[i], s) == 0) {
            return true;
        }
    }
    return false;
}

static void tf_add_tok(struct tf_vocab *v, const char *s, size_t n, float score) {
    char *t = tf_strndup(s, n);
    if (tf_has_tok(v, t)) {
        free(t);
        return;
    }
    v->tok             = realloc(v->tok, (v->n_tok + 1) * sizeof *v->tok);
    v->score           = realloc(v->score, (v->n_tok + 1) * sizeof *v->score);
    v->tok[v->n_tok]   = t;
    v->score[v->n_tok] = score;
    v->n_tok += 1;
}

/* Appends merge (l, r) unless it is already listed: ranks follow first use. */
static void tf_add_merge(struct tf_vocab *v, const char *l, size_t ll, const char *r, size_t rl) {
    for (size_t m = 0; m < v->n_merge; m++) {
        if (strlen(v->merge_l[m]) == ll && memcmp(v->merge_l[m], l, ll) == 0 &&
            strlen(v->merge_r[m]) == rl && memcmp(v->merge_r[m], r, rl) == 0) {
            return;
        }
    }
    v->merge_l             = realloc(v->merge_l, (v->n_merge + 1) * sizeof *v->merge_l);
    v->merge_r             = realloc(v->merge_r, (v->n_merge + 1) * sizeof *v->merge_r);
    v->merge_l[v->n_merge] = tf_strndup(l, ll);
    v->merge_r[v->n_merge] = tf_strndup(r, rl);
    v->n_merge += 1;
}

/* The chain that builds s[0, n) left to right from s[0, first): merges
 * (s[0,first), s[first]), (s[0,first+1), s[first+1]), ..., and each prefix
 * as a token. Unigram scores grow with length, so the greedy pair merge
 * prefers the longer pieces. */
static void tf_add_chain(struct tf_vocab *v, const char *s, size_t first, size_t n) {
    for (size_t end = first + 1; end <= n; end++) {
        tf_add_merge(v, s, end - 1, s + end - 1, 1);
        tf_add_tok(v, s, end, -10.0f + (float) end);
    }
}

/* marker: the word-start symbol ("▁" for SentencePiece, "Ġ" for gpt2).
 * byte_tokens: add the <0xXX> byte-fallback tokens (SentencePiece). */
static struct tf_vocab tf_make_vocab(const char *marker, bool byte_tokens) {
    struct tf_vocab v = {0};
    tf_add_tok(&v, "<unk>", 5, 0.0f);
    tf_add_tok(&v, "<s>", 3, 0.0f);
    tf_add_tok(&v, "</s>", 4, 0.0f);
    v.unk = 0;
    if (byte_tokens) {
        for (int b = 0; b < 256; b++) {
            char s[8];
            snprintf(s, sizeof s, "<0x%02X>", b);
            tf_add_tok(&v, s, 6, -100.0f);
        }
    }
    tf_add_tok(&v, marker, strlen(marker), -10.0f);
    for (char c = 'a'; c <= 'z'; c++) {
        tf_add_tok(&v, &c, 1, -10.0f);
    }
    const size_t mlen = strlen(marker);
    for (size_t w = 0; w < TF_N_WORDS; w++) {
        /* marker + word: ▁t, ▁th, ▁the */
        char         full[64];
        const size_t wl = strlen(TF_WORDS[w]);
        memcpy(full, marker, mlen);
        memcpy(full + mlen, TF_WORDS[w], wl);
        tf_add_chain(&v, full, mlen, mlen + wl);
    }
    /* The bare words too: a run of letters without the marker (one gpt2
     * pre-tokenizer chunk) merges as well. */
    for (size_t w = 0; w < TF_N_WORDS; w++) {
        tf_add_chain(&v, TF_WORDS[w], 1, strlen(TF_WORDS[w]));
    }
    return v;
}

/* ---- GGUF ---------------------------------------------------------------- */

enum { TF_GGUF_UINT32 = 4, TF_GGUF_FLOAT32 = 6, TF_GGUF_STRING = 8, TF_GGUF_ARRAY = 9 };

/* A GGUF holding only tokenizer metadata. model is "gpt2" or "llama";
 * with_merges selects BPE (gpt2 / SPM) over unigram (scores only). */
static struct tf_buf tf_gguf(const struct tf_vocab *v, const char *model, bool with_merges) {
    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, 1); /* tensors: the reader wants at least one */
    tf_u64(&o, 4); /* metadata keys */
    tf_gstr(&o, "tokenizer.ggml.model", 20);
    tf_u32(&o, TF_GGUF_STRING);
    tf_gstr(&o, model, strlen(model));
    tf_gstr(&o, "tokenizer.ggml.tokens", 21);
    tf_u32(&o, TF_GGUF_ARRAY);
    tf_u32(&o, TF_GGUF_STRING);
    tf_u64(&o, v->n_tok);
    for (size_t i = 0; i < v->n_tok; i++) {
        tf_gstr(&o, v->tok[i], strlen(v->tok[i]));
    }
    if (with_merges) {
        tf_gstr(&o, "tokenizer.ggml.merges", 21);
        tf_u32(&o, TF_GGUF_ARRAY);
        tf_u32(&o, TF_GGUF_STRING);
        tf_u64(&o, v->n_merge);
        for (size_t m = 0; m < v->n_merge; m++) {
            char         s[128];
            const size_t n = (size_t) snprintf(s, sizeof s, "%s %s", v->merge_l[m], v->merge_r[m]);
            tf_gstr(&o, s, n);
        }
    } else {
        tf_gstr(&o, "tokenizer.ggml.scores", 21);
        tf_u32(&o, TF_GGUF_ARRAY);
        tf_u32(&o, TF_GGUF_FLOAT32);
        tf_u64(&o, v->n_tok);
        for (size_t i = 0; i < v->n_tok; i++) {
            tf_f32(&o, v->score[i]);
        }
    }
    tf_gstr(&o, "tokenizer.ggml.unknown_token_id", 31);
    tf_u32(&o, TF_GGUF_UINT32);
    tf_u32(&o, v->unk);
    /* One dummy 1-element f32 tensor: name, dims, dtype, offset. */
    tf_gstr(&o, "t", 1);
    tf_u32(&o, 1);
    tf_u64(&o, 1);
    tf_u32(&o, 0);
    tf_u64(&o, 0);
    while (o.n % 32 != 0) {
        tf_put(&o, "", 1);
    }
    tf_f32(&o, 0.0f);
    return o;
}

/* ---- tokenizer.bin (sp_bpe_tokenizer) ------------------------------------ */

static struct tf_buf tf_sp_bpe_bin(const struct tf_vocab *v) {
    struct tf_buf o = {0};
    tf_u32(&o, 0x4B544D47u); /* "GMTK" */
    tf_u32(&o, 1u);
    tf_u32(&o, (uint32_t) v->n_tok);
    tf_u32(&o, (uint32_t) v->n_merge);
    tf_u32(&o, 0u); /* specials */
    tf_u32(&o, 1u); /* bos */
    tf_u32(&o, 2u); /* eos */
    tf_u32(&o, 0u); /* pad */
    tf_u32(&o, v->unk);
    for (size_t i = 0; i < v->n_tok; i++) {
        tf_u16(&o, (uint16_t) strlen(v->tok[i]));
        tf_put(&o, v->tok[i], strlen(v->tok[i]));
    }
    for (size_t m = 0; m < v->n_merge; m++) {
        tf_u16(&o, (uint16_t) strlen(v->merge_l[m]));
        tf_put(&o, v->merge_l[m], strlen(v->merge_l[m]));
        tf_u16(&o, (uint16_t) strlen(v->merge_r[m]));
        tf_put(&o, v->merge_r[m], strlen(v->merge_r[m]));
    }
    return o;
}

/* ---- text ---------------------------------------------------------------- */

/* n bytes of words separated by single spaces, NUL-terminated. With
 * letters_only, one unbroken run of the words' letters (one pre-tokenizer
 * chunk even for gpt2). */
static char *tf_text(size_t n, uint32_t seed, bool letters_only) {
    char  *t = malloc(n + 1);
    size_t w = 0;
    while (w < n) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        const char  *word = TF_WORDS[seed % TF_N_WORDS];
        const size_t l    = strlen(word);
        for (size_t i = 0; i < l && w < n; i++) {
            t[w++] = word[i];
        }
        if (!letters_only && w < n) {
            t[w++] = ' ';
        }
    }
    t[n] = '\0';
    return t;
}

#endif /* GEIST_TESTS_TOKENIZER_FIXTURES_H */
