/*
 * src/engine/dry_breakers.c — DRY sequence breakers (#695).
 *
 * Layer: ENGINE.
 *
 * A port of llama.cpp's get_overlapping_token_sequences and the breaker
 * loop of llama_sampler_init_dry (src/llama-sampler.cpp; originally
 * koboldcpp #982). Runs once at geist_session_create, only when DRY is on:
 * one decode per vocabulary entry and breaker, a tokenize per partial match.
 *
 * llama.cpp keeps the sequences in a multimap and does not remove a repeated
 * single-token entry; neither does this. Duplicates change nothing: the
 * sampler takes the longest matching tail and asks only whether an empty
 * one exists.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "dry_breakers.h"
#include "gguf_tokenizer.h"

#include "checked.h"
#include "heap.h"

#include <stdint.h>
#include <string.h>

const char *const geist_dry_default_breakers[4] = {"\n", ":", "\"", "*"};

constexpr size_t DRY_MAX_CHAR_LEN = 40; /* llama.cpp's MAX_CHAR_LEN */
constexpr size_t DRY_MAX_SEQ_LEN  = 20; /* llama.cpp's MAX_SEQ_LEN: tail tokens */

/* A growing array of packed words. heap.h has no realloc (AGENT.md §3), so
 * growth copies. */
struct packed {
    geist_token_t *w;
    size_t         n;
    size_t         cap;
};

[[nodiscard]] static bool packed_push(struct packed *p, size_t n, const geist_token_t *src) {
    if (n > p->cap - p->n) {
        size_t cap = p->cap > 0 ? p->cap : 256;
        while (n > cap - p->n) {
            if (ckd_mul(&cap, cap, 2)) {
                return false;
            }
        }
        geist_token_t *w = heap_alloc_array_aligned(geist_token_t, cap);
        if (w == nullptr) {
            return false;
        }
        if (p->n > 0) {
            memcpy(w, p->w, p->n * sizeof *w);
        }
        safe_free((void **) &p->w);
        p->w   = w;
        p->cap = cap;
    }
    memcpy(p->w + p->n, src, n * sizeof *src);
    p->n += n;
    return true;
}

static bool contains(size_t n_word, const char *word, size_t n_str, const char *str) {
    if (n_str > n_word) {
        return false;
    }
    for (size_t i = 0; i + n_str <= n_word; i++) {
        if (memcmp(word + i, str, n_str) == 0) {
            return true;
        }
    }
    return false;
}

/* One breaker string against the whole vocabulary. `word` has room for the
 * longest decoded token; `tail` for a NUL-terminated copy of `str`. */
[[nodiscard]] static enum geist_status overlapping_sequences(struct packed               *out,
                                                             const struct gguf_tokenizer *tok,
                                                             size_t                       n_str,
                                                             const char                  *str,
                                                             size_t                       word_cap,
                                                             char                        *word,
                                                             char                        *tail) {
    geist_token_t seq[2 + 64];
    for (size_t id = 0; id < tok->vocab_size; id++) {
        const int32_t tid    = (int32_t) id;
        const size_t  n_word = gguf_tokenizer_decode(tok, &tid, 1, word, word_cap);
        if (n_word > word_cap) {
            return GEIST_E_INTERNAL; /* decoding never lengthens a token */
        }
        if (contains(n_word, word, n_str, str)) {
            seq[0] = 1;
            seq[1] = tid;
            if (!packed_push(out, 2, seq)) {
                return GEIST_E_OOM;
            }
            continue;
        }
        /* Each place the token's text ends in a prefix of str: the rest of
         * str, tokenized, is the tail that completes the breaker. */
        for (size_t pos = 0; pos < n_word; pos++) {
            if (word[pos] != str[0]) {
                continue;
            }
            size_t i     = 1;
            bool   match = true;
            for (; i < n_str && i + pos < n_word; i++) {
                if (word[pos + i] != str[i]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            memcpy(tail, str + i, n_str - i);
            tail[n_str - i]       = '\0';
            size_t            n_t = 0;
            enum geist_status s   = gguf_tokenizer_encode(tok, tail, 64, seq + 2, &n_t);
            if (s != GEIST_OK) {
                return s;
            }
            if (n_t > DRY_MAX_SEQ_LEN) {
                n_t = DRY_MAX_SEQ_LEN;
            }
            seq[0] = (geist_token_t) (1 + n_t);
            seq[1] = tid;
            if (!packed_push(out, 2 + n_t, seq)) {
                return GEIST_E_OOM;
            }
        }
    }
    return GEIST_OK;
}

enum geist_status geist_dry_breakers_resolve(size_t                      *n_words,
                                             geist_token_t              **out,
                                             const struct gguf_tokenizer *tok,
                                             size_t                       n_strs,
                                             const char *const           *strs) {
    *n_words = 0;
    *out     = nullptr;
    if (tok == nullptr || n_strs == 0) {
        return GEIST_OK;
    }
    if (strs == nullptr || tok->vocab_size > (size_t) INT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    size_t word_cap = 1;
    for (size_t id = 0; id < tok->vocab_size; id++) {
        word_cap = tok->token_len[id] > word_cap ? tok->token_len[id] : word_cap;
    }
    char *word = heap_alloc_array_aligned(char, word_cap);
    char  tail[DRY_MAX_CHAR_LEN + 1];
    if (word == nullptr) {
        return GEIST_E_OOM;
    }
    struct packed     p = {0};
    enum geist_status s = GEIST_OK;
    for (size_t b = 0; b < n_strs && s == GEIST_OK; b++) {
        if (strs[b] == nullptr || strs[b][0] == '\0') {
            continue; /* llama.cpp skips them with a warning */
        }
        size_t n_str = strlen(strs[b]);
        n_str        = n_str < DRY_MAX_CHAR_LEN ? n_str : DRY_MAX_CHAR_LEN;
        s            = overlapping_sequences(&p, tok, n_str, strs[b], word_cap, word, tail);
    }
    safe_free((void **) &word);
    if (s != GEIST_OK) {
        safe_free((void **) &p.w);
        return s;
    }
    *n_words = p.n;
    *out     = p.w;
    return GEIST_OK;
}
