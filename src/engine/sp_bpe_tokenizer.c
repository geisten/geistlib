#include "sp_bpe_tokenizer.h"
#include "heap.h"
#include "pair_merge.h"

#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* SentencePiece space marker U+2581 in UTF-8. */
static const char SP_MARKER[3] = {(char) 0xE2, (char) 0x96, (char) 0x81};
#define SP_MARKER_LEN 3

#define MAGIC_LE 0x4B544D47u /* "GMTK" little-endian */
#define VERSION 1u

#define VOCAB_LOAD_FACTOR_DENOM 2 /* hash cap = vocab_size * 2 */

/* ---- public struct -------------------------------------------------------- */

struct sp_token_entry {
    const char *text; /* points into mmap */
    uint32_t    len;
};

struct sp_vocab_hash {
    uint32_t    id;   /* UINT32_MAX = empty */
    const char *text; /* points into mmap */
    uint32_t    len;
};

struct sp_merge_hash {
    uint32_t    rank; /* UINT32_MAX = empty */
    const char *left;
    const char *right;
    uint32_t    left_len;
    uint32_t    right_len;
};

struct sp_special {
    uint32_t    id;
    const char *text;
    uint32_t    len;
};

struct sp_bpe_tokenizer {
    int    fd;
    void  *map;
    size_t map_size;

    uint32_t vocab_size;
    uint32_t merges_count;
    uint32_t specials_count;
    uint32_t bos_id, eos_id, pad_id, unk_id;

    struct sp_token_entry *id_to_token; /* size = vocab_size */

    struct sp_vocab_hash *vocab_hash;
    size_t                vocab_hash_cap;

    struct sp_merge_hash *merge_hash;
    size_t                merge_hash_cap;

    struct sp_special *specials; /* sorted by len descending for longest-match */

    /* Byte-fallback: precomputed map from byte value to vocab ID of <0xXX>.
     * UINT32_MAX if that byte's fallback token is not in vocab. */
    uint32_t byte_to_id[256];
};

/* ---- helpers -------------------------------------------------------------- */

static uint64_t fnv1a(const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *) data;
    uint64_t       h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t fnv1a_two(const void *a, size_t na, const void *b, size_t nb) {
    uint64_t       h = 14695981039346656037ULL;
    const uint8_t *p = (const uint8_t *) a;
    for (size_t i = 0; i < na; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    p = (const uint8_t *) b;
    for (size_t i = 0; i < nb; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

/* ---- vocab hash table ----------------------------------------------------- */

static void vocab_hash_insert(
        struct sp_vocab_hash *tab, size_t cap, const char *text, uint32_t len, uint32_t id) {
    size_t mask = cap - 1;
    size_t i    = (size_t) fnv1a(text, len) & mask;
    while (tab[i].id != UINT32_MAX) {
        i = (i + 1) & mask;
    }
    tab[i].id   = id;
    tab[i].text = text;
    tab[i].len  = len;
}

static uint32_t
vocab_hash_lookup(const struct sp_vocab_hash *tab, size_t cap, const char *text, uint32_t len) {
    size_t mask = cap - 1;
    size_t i    = (size_t) fnv1a(text, len) & mask;
    while (tab[i].id != UINT32_MAX) {
        if (tab[i].len == len && memcmp(tab[i].text, text, len) == 0) {
            return tab[i].id;
        }
        i = (i + 1) & mask;
    }
    return UINT32_MAX;
}

/* ---- merge hash table ----------------------------------------------------- */

static void merge_hash_insert(struct sp_merge_hash *tab,
                              size_t                cap,
                              const char           *left,
                              uint32_t              left_len,
                              const char           *right,
                              uint32_t              right_len,
                              uint32_t              rank) {
    size_t mask = cap - 1;
    size_t i    = (size_t) fnv1a_two(left, left_len, right, right_len) & mask;
    while (tab[i].rank != UINT32_MAX) {
        i = (i + 1) & mask;
    }
    tab[i].rank      = rank;
    tab[i].left      = left;
    tab[i].right     = right;
    tab[i].left_len  = left_len;
    tab[i].right_len = right_len;
}

static uint32_t merge_hash_lookup(const struct sp_merge_hash *tab,
                                  size_t                      cap,
                                  const char                 *left,
                                  uint32_t                    left_len,
                                  const char                 *right,
                                  uint32_t                    right_len) {
    size_t mask = cap - 1;
    size_t i    = (size_t) fnv1a_two(left, left_len, right, right_len) & mask;
    while (tab[i].rank != UINT32_MAX) {
        if (tab[i].left_len == left_len && tab[i].right_len == right_len &&
            memcmp(tab[i].left, left, left_len) == 0 &&
            memcmp(tab[i].right, right, right_len) == 0) {
            return tab[i].rank;
        }
        i = (i + 1) & mask;
    }
    return UINT32_MAX;
}

/* ---- loader --------------------------------------------------------------- */

static int specials_cmp_desc_len(const void *a, const void *b) {
    const struct sp_special *sa = (const struct sp_special *) a;
    const struct sp_special *sb = (const struct sp_special *) b;
    if (sa->len != sb->len)
        return (int) sb->len - (int) sa->len;
    return 0;
}

bool sp_bpe_tokenizer_load(struct sp_bpe_tokenizer **out, const char *path) {
    if (!out || !path)
        return false;
    *out = nullptr;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        close(fd);
        return false;
    }
    size_t fsize = (size_t) sb.st_size;

    void *map = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return false;
    }

    const uint8_t *p   = (const uint8_t *) map;
    const uint8_t *end = p + fsize;

    if (fsize < 36)
        goto bad_format;
    uint32_t magic_le, version, vocab_size, merges_count, specials_count;
    memcpy(&magic_le, p, 4);
    p += 4;
    memcpy(&version, p, 4);
    p += 4;
    memcpy(&vocab_size, p, 4);
    p += 4;
    memcpy(&merges_count, p, 4);
    p += 4;
    memcpy(&specials_count, p, 4);
    p += 4;
    if (magic_le != MAGIC_LE) {
        fprintf(stderr, "sp_bpe_tokenizer: bad magic %08x\n", magic_le);
        goto bad;
    }
    if (version != VERSION) {
        fprintf(stderr, "sp_bpe_tokenizer: unsupported version %u\n", version);
        goto bad;
    }

    uint32_t bos, eos, pad, unk;
    memcpy(&bos, p, 4);
    p += 4;
    memcpy(&eos, p, 4);
    p += 4;
    memcpy(&pad, p, 4);
    p += 4;
    memcpy(&unk, p, 4);
    p += 4;

    /* Allocate context */
    struct sp_bpe_tokenizer *tok = heap_calloc_array_aligned(struct sp_bpe_tokenizer, 1);
    if (!tok)
        goto bad;
    tok->fd             = fd;
    tok->map            = map;
    tok->map_size       = fsize;
    tok->vocab_size     = vocab_size;
    tok->merges_count   = merges_count;
    tok->specials_count = specials_count;
    tok->bos_id         = bos;
    tok->eos_id         = eos;
    tok->pad_id         = pad;
    tok->unk_id         = unk;

    /* Vocab section: vocab_size entries of (len:u16, bytes) */
    tok->id_to_token    = heap_calloc_array_aligned(struct sp_token_entry, vocab_size);
    tok->vocab_hash_cap = next_pow2((size_t) vocab_size * VOCAB_LOAD_FACTOR_DENOM);
    tok->vocab_hash     = heap_alloc_array_aligned(struct sp_vocab_hash, tok->vocab_hash_cap);
    if (!tok->id_to_token || !tok->vocab_hash)
        goto bad_alloc;
    for (size_t i = 0; i < tok->vocab_hash_cap; i++)
        tok->vocab_hash[i].id = UINT32_MAX;

    for (uint32_t id = 0; id < vocab_size; id++) {
        if (p + 2 > end)
            goto bad_format_with_tok;
        uint16_t len;
        memcpy(&len, p, 2);
        p += 2;
        if (p + len > end)
            goto bad_format_with_tok;
        tok->id_to_token[id].text = (const char *) p;
        tok->id_to_token[id].len  = len;
        if (len > 0) {
            vocab_hash_insert(tok->vocab_hash, tok->vocab_hash_cap, (const char *) p, len, id);
        }
        p += len;
    }

    /* Merges section: merges_count entries of (l_len:u16, l_bytes, r_len:u16, r_bytes) */
    tok->merge_hash_cap = next_pow2((size_t) merges_count * VOCAB_LOAD_FACTOR_DENOM);
    tok->merge_hash     = heap_alloc_array_aligned(struct sp_merge_hash, tok->merge_hash_cap);
    if (!tok->merge_hash)
        goto bad_alloc;
    for (size_t i = 0; i < tok->merge_hash_cap; i++)
        tok->merge_hash[i].rank = UINT32_MAX;

    for (uint32_t rank = 0; rank < merges_count; rank++) {
        if (p + 2 > end)
            goto bad_format_with_tok;
        uint16_t llen;
        memcpy(&llen, p, 2);
        p += 2;
        if (p + llen > end)
            goto bad_format_with_tok;
        const char *lbytes = (const char *) p;
        p += llen;
        if (p + 2 > end)
            goto bad_format_with_tok;
        uint16_t rlen;
        memcpy(&rlen, p, 2);
        p += 2;
        if (p + rlen > end)
            goto bad_format_with_tok;
        const char *rbytes = (const char *) p;
        p += rlen;
        merge_hash_insert(tok->merge_hash, tok->merge_hash_cap, lbytes, llen, rbytes, rlen, rank);
    }

    /* Special tokens section */
    tok->specials = heap_calloc_array_aligned(struct sp_special, specials_count);
    if (!tok->specials && specials_count > 0)
        goto bad_alloc;

    for (uint32_t i = 0; i < specials_count; i++) {
        if (p + 6 > end)
            goto bad_format_with_tok;
        uint32_t id;
        memcpy(&id, p, 4);
        p += 4;
        uint16_t len;
        memcpy(&len, p, 2);
        p += 2;
        if (p + len > end)
            goto bad_format_with_tok;
        tok->specials[i].id   = id;
        tok->specials[i].text = (const char *) p;
        tok->specials[i].len  = len;
        p += len;
    }
    /* Sort by length desc so longest-match wins during scan. */
    qsort(tok->specials, specials_count, sizeof(*tok->specials), specials_cmp_desc_len);

    /* Pre-compute byte-fallback map: <0x00>..<0xFF> -> vocab IDs. */
    for (uint32_t b = 0; b < 256; b++) {
        char buf[7]; /* "<0xAB>" */
        snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tok->byte_to_id[b] = vocab_hash_lookup(tok->vocab_hash, tok->vocab_hash_cap, buf, 6);
    }

    *out = tok;
    return true;

bad_format_with_tok:
    fprintf(stderr, "sp_bpe_tokenizer: truncated file\n");
    sp_bpe_tokenizer_free(tok);
    return false;
bad_alloc:
    fprintf(stderr, "sp_bpe_tokenizer: allocation failed\n");
    sp_bpe_tokenizer_free(tok);
    return false;
bad_format:
    fprintf(stderr, "sp_bpe_tokenizer: file too small for header\n");
bad:
    munmap(map, fsize);
    close(fd);
    return false;
}

void sp_bpe_tokenizer_free(struct sp_bpe_tokenizer *tok) {
    if (!tok)
        return;
    safe_free((void **) &tok->id_to_token);
    safe_free((void **) &tok->vocab_hash);
    safe_free((void **) &tok->merge_hash);
    safe_free((void **) &tok->specials);
    if (tok->map && tok->map != MAP_FAILED)
        munmap(tok->map, tok->map_size);
    if (tok->fd >= 0)
        close(tok->fd);
    safe_free((void **) &tok);
}

/* ---- accessors ------------------------------------------------------------ */

uint32_t sp_bpe_tokenizer_vocab_size(const struct sp_bpe_tokenizer *tok) {
    return tok ? tok->vocab_size : 0;
}
uint32_t sp_bpe_tokenizer_bos_id(const struct sp_bpe_tokenizer *tok) {
    return tok ? tok->bos_id : 0;
}
uint32_t sp_bpe_tokenizer_eos_id(const struct sp_bpe_tokenizer *tok) {
    return tok ? tok->eos_id : 0;
}
uint32_t sp_bpe_tokenizer_pad_id(const struct sp_bpe_tokenizer *tok) {
    return tok ? tok->pad_id : 0;
}
uint32_t sp_bpe_tokenizer_unk_id(const struct sp_bpe_tokenizer *tok) {
    return tok ? tok->unk_id : 0;
}
uint32_t sp_bpe_tokenizer_specials_count(const struct sp_bpe_tokenizer *tok) {
    return tok ? tok->specials_count : 0;
}

const char *
sp_bpe_tokenizer_id_to_text(const struct sp_bpe_tokenizer *tok, uint32_t id, size_t *len_out) {
    if (!tok || id >= tok->vocab_size) {
        if (len_out)
            *len_out = 0;
        return nullptr;
    }
    if (len_out)
        *len_out = tok->id_to_token[id].len;
    return tok->id_to_token[id].text;
}

/* ---- encoding ------------------------------------------------------------- */

/* UTF-8 codepoint length from leading byte (1..4), or 1 on invalid leading. */
static inline size_t utf8_cp_len(uint8_t b) {
    if (b < 0x80)
        return 1;
    if ((b & 0xE0) == 0xC0)
        return 2;
    if ((b & 0xF0) == 0xE0)
        return 3;
    if ((b & 0xF8) == 0xF0)
        return 4;
    return 1; /* invalid leading byte; treat as single byte */
}

/* Append a token ID to dynamic array; returns false on alloc failure.
 * heap.h has no realloc-equivalent, so this grows via explicit
 * alloc-new + memcpy + safe_free-old. The grow happens at most
 * log2(N) times during encode, so the extra cost is negligible. */
static bool append_id(uint32_t **arr, size_t *count, size_t *cap, uint32_t id) {
    if (*count == *cap) {
        size_t    new_cap = *cap ? *cap * 2 : 16;
        uint32_t *na      = heap_alloc_array_aligned(uint32_t, new_cap);
        if (!na)
            return false;
        if (*arr) {
            memcpy(na, *arr, *count * sizeof(uint32_t));
            safe_free((void **) arr);
        }
        *arr = na;
        *cap = new_cap;
    }
    (*arr)[(*count)++] = id;
    return true;
}

/* Try longest special-token match at text+pos. Returns match length (0 if none).
 * On match, writes special's ID to *id_out. */
static size_t match_special(const struct sp_bpe_tokenizer *tok,
                            const char                    *text,
                            size_t                         remaining,
                            uint32_t                      *id_out) {
    /* specials are sorted by length desc */
    for (uint32_t i = 0; i < tok->specials_count; i++) {
        const struct sp_special *s = &tok->specials[i];
        if (s->len > remaining)
            continue;
        if (memcmp(text, s->text, s->len) == 0) {
            *id_out = s->id;
            return s->len;
        }
    }
    return 0;
}

/* Merge-engine key (pair_merge.h): the merge's rank, lowest first. */
static bool sp_merge_key(
        const void *ctx, const char *buf, size_t off, size_t llen, size_t rlen, uint64_t *key) {
    const struct sp_bpe_tokenizer *tok  = ctx;
    const uint32_t                 rank = merge_hash_lookup(tok->merge_hash,
                                                            tok->merge_hash_cap,
                                                            buf + off,
                                                            (uint32_t) llen,
                                                            buf + off + llen,
                                                            (uint32_t) rlen);
    *key                                = rank;
    return rank != UINT32_MAX;
}

/* BPE-encode a single chunk (no specials inside). The chunk is normalized
 * (" " -> "▁") into a stack/heap buffer, then greedy-merged. Output token
 * IDs are appended to (*tokens). */
static bool encode_chunk(const struct sp_bpe_tokenizer *tok,
                         const char                    *in,
                         size_t                         in_len,
                         uint32_t                     **tokens,
                         size_t                        *count,
                         size_t                        *cap) {
    if (in_len == 0)
        return true;

    /* Step 1: normalize. Each ASCII space (1 byte) becomes ▁ (3 bytes). */
    size_t n_spaces = 0;
    for (size_t i = 0; i < in_len; i++)
        if (in[i] == ' ')
            n_spaces++;
    size_t buf_len = in_len + n_spaces * (SP_MARKER_LEN - 1);
    if (buf_len > (size_t) INT_MAX)
        return false; /* symbol indices are int (pair_merge.h) */
    char *buf = heap_alloc_array_aligned(char, buf_len);
    if (!buf)
        return false;
    {
        size_t w = 0;
        for (size_t r = 0; r < in_len; r++) {
            if (in[r] == ' ') {
                memcpy(buf + w, SP_MARKER, SP_MARKER_LEN);
                w += SP_MARKER_LEN;
            } else {
                buf[w++] = in[r];
            }
        }
    }

    /* Step 2: split into UTF-8 codepoints (initial symbols), each a slice
     * of buf. Worst case: each byte is its own codepoint (ASCII). */
    struct pair_merge_sym *syms = heap_alloc_array_aligned(struct pair_merge_sym, buf_len);
    if (!syms) {
        safe_free((void **) &buf);
        return false;
    }
    size_t n_syms = 0;
    {
        size_t i = 0;
        while (i < buf_len) {
            size_t len = utf8_cp_len((uint8_t) buf[i]);
            if (i + len > buf_len)
                len = buf_len - i;
            syms[n_syms] = (struct pair_merge_sym) {
                    .off = i, .len = len, .prev = (int) n_syms - 1, .next = (int) n_syms + 1};
            n_syms++;
            i += len;
        }
        syms[n_syms - 1].next = -1;
    }

    /* Step 3: greedy lowest-rank merge, leftmost among equal ranks, until
     * no merges are available (pair_merge.h, O(n log n)). */
    if (!pair_merge_run(n_syms, syms, buf, sp_merge_key, tok)) {
        safe_free((void **) &syms);
        safe_free((void **) &buf);
        return false;
    }

    /* Step 4: lookup each symbol in vocab. If missing, byte-fallback
     * (emit one <0xXX> token per byte); if that fails too, emit unk_id. */
    bool ok = true;
    for (int i = 0; i >= 0; i = syms[i].next) {
        const char *t    = buf + syms[i].off;
        uint32_t    tlen = (uint32_t) syms[i].len;
        uint32_t    id   = vocab_hash_lookup(tok->vocab_hash, tok->vocab_hash_cap, t, tlen);
        if (id != UINT32_MAX) {
            if (!append_id(tokens, count, cap, id)) {
                ok = false;
                break;
            }
        } else {
            for (uint32_t b = 0; b < tlen; b++) {
                uint32_t bid = tok->byte_to_id[(uint8_t) t[b]];
                if (bid == UINT32_MAX)
                    bid = tok->unk_id;
                if (!append_id(tokens, count, cap, bid)) {
                    ok = false;
                    break;
                }
            }
            if (!ok)
                break;
        }
    }

    safe_free((void **) &syms);
    safe_free((void **) &buf);
    return ok;
}

bool sp_bpe_tokenizer_encode(const struct sp_bpe_tokenizer *tok,
                             const char                    *text,
                             uint32_t                     **tokens_out,
                             size_t                        *count_out) {
    if (!tok || !text || !tokens_out || !count_out)
        return false;

    uint32_t *tokens = nullptr;
    size_t    count = 0, cap = 0;

    size_t pos         = 0;
    size_t len         = strlen(text);
    size_t chunk_start = 0;

    while (pos < len) {
        uint32_t sp_id  = UINT32_MAX;
        size_t   sp_len = match_special(tok, text + pos, len - pos, &sp_id);
        if (sp_len > 0) {
            /* Flush pending non-special chunk */
            if (chunk_start < pos) {
                if (!encode_chunk(
                            tok, text + chunk_start, pos - chunk_start, &tokens, &count, &cap)) {
                    safe_free((void **) &tokens);
                    return false;
                }
            }
            if (!append_id(&tokens, &count, &cap, sp_id)) {
                safe_free((void **) &tokens);
                return false;
            }
            pos += sp_len;
            chunk_start = pos;
        } else {
            pos++;
        }
    }
    /* Trailing non-special chunk */
    if (chunk_start < len) {
        if (!encode_chunk(tok, text + chunk_start, len - chunk_start, &tokens, &count, &cap)) {
            safe_free((void **) &tokens);
            return false;
        }
    }

    *tokens_out = tokens;
    *count_out  = count;
    return true;
}
