/*
 * test_tokenizer_merge_unit — the O(n log n) merge engine (pair_merge.h)
 * makes the same merges as the quadratic rescan it replaced, and the
 * encoders built on it produce the same tokens.
 *
 * 1. Engine against the rescan. Random symbol sequences, merged by a key
 *    that depends only on the two symbols' bytes (as the rank and the
 *    score lookups do), with few distinct keys so that equal-key pairs —
 *    where only the leftmost may merge — are everywhere, and with long
 *    runs of one symbol (the "aaaa" case: every pair has the same key).
 *    The final segmentation must match the reference rescan exactly.
 *
 * 2. Encoders against the rescan. Synthetic SPM (merges) and unigram
 *    (scores) tokenizers from tokenizer_fixtures.h, and sp_bpe_tokenizer
 *    on the same vocabulary. The reference here re-implements the old
 *    encode of a text without specials: ▁ normalization (plus the leading
 *    ▁ of add_space_prefix for the GGUF SPM modes), one symbol per byte of
 *    this ASCII text, the rescan, vocab lookup. Texts of words, of one
 *    unbroken run of letters, and random letters, up to 3000 bytes — far
 *    past the gpt2 rescan cutoff, so the heap path is what runs.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "gguf_tokenizer.h"
#include "pair_merge.h"
#include "sp_bpe_tokenizer.h"
#include "tokenizer_fixtures.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint32_t g_rng = 0x2545F491u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

/* ---- 1. engine against the rescan ---------------------------------------- */

/* A key from the two symbols' bytes alone; no merge for about a quarter
 * of the pairs and at most 5 distinct keys, so ties are common. */
static bool
toy_key(const void *ctx, const char *buf, size_t off, size_t llen, size_t rlen, uint64_t *key) {
    (void) ctx;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < llen; i++) {
        h = (h ^ (uint8_t) buf[off + i]) * 16777619u;
    }
    h = (h ^ 0xFFu) * 16777619u; /* the split point matters */
    for (size_t i = 0; i < rlen; i++) {
        h = (h ^ (uint8_t) buf[off + llen + i]) * 16777619u;
    }
    if (h % 4 == 0 || llen + rlen > 12) {
        return false;
    }
    *key = (h >> 8) % 5;
    return true;
}

static void init_syms(size_t n, struct pair_merge_sym *syms) {
    for (size_t i = 0; i < n; i++) {
        syms[i] = (struct pair_merge_sym) {
                .off = i, .len = 1, .prev = (int) i - 1, .next = i + 1 < n ? (int) i + 1 : -1};
    }
}

/* The loop pair_merge_run replaced: rescan every pair, merge the lowest
 * key, leftmost among equal keys. */
static void rescan_merge(size_t n, struct pair_merge_sym *syms, const char *buf) {
    if (n < 2) {
        return;
    }
    while (true) {
        int      best     = -1;
        uint64_t best_key = UINT64_MAX;
        for (int i = 0; syms[i].next >= 0; i = syms[i].next) {
            uint64_t k = 0;
            if (toy_key(nullptr, buf, syms[i].off, syms[i].len, syms[syms[i].next].len, &k) &&
                (best < 0 || k < best_key)) {
                best     = i;
                best_key = k;
            }
        }
        if (best < 0) {
            return;
        }
        const int r = syms[best].next;
        syms[best].len += syms[r].len;
        syms[best].next = syms[r].next;
        if (syms[r].next >= 0) {
            syms[syms[r].next].prev = best;
        }
    }
}

static int check_engine(void) {
    enum { MAX_N = 700 };
    static char                  buf[MAX_N];
    static struct pair_merge_sym a[MAX_N], b[MAX_N];
    int                          fails = 0;
    for (int trial = 0; trial < 3000; trial++) {
        const size_t n = 1 + next_u32() % (trial < 2000 ? 64 : MAX_N);
        /* alphabets of 1-4 letters make runs and repeats */
        const uint32_t alpha = 1 + next_u32() % 4;
        for (size_t i = 0; i < n; i++) {
            buf[i] = (char) ('a' + next_u32() % alpha);
        }
        init_syms(n, a);
        init_syms(n, b);
        rescan_merge(n, a, buf);
        if (!pair_merge_run(n, b, buf, toy_key, nullptr)) {
            fprintf(stderr, "FAIL: pair_merge_run allocation (n=%zu)\n", n);
            return 1;
        }
        bool same = true;
        int  i = 0, j = 0;
        for (; i >= 0 && j >= 0; i = a[i].next, j = b[j].next) {
            same = same && a[i].off == b[j].off && a[i].len == b[j].len;
        }
        same = same && i < 0 && j < 0;
        if (!same) {
            fprintf(stderr,
                    "FAIL: engine differs from the rescan (trial %d, n=%zu, alpha=%u)\n",
                    trial,
                    n,
                    alpha);
            fails++;
        }
    }
    printf("  engine: 3000 random sequences (n <= %d), same merges as the rescan\n", MAX_N);
    return fails;
}

/* ---- 2. encoders against the rescan -------------------------------------- */

static const char SPM_MARKER_BYTES[] = "\xe2\x96\x81";

/* Exact-match string -> value map for the reference's lookups (the linear
 * scans would make it too slow to be a unit test). First insert wins. */
struct ref_map {
    size_t   cap; /* power of two */
    char   **key;
    size_t  *klen;
    int64_t *val;
};

static uint64_t ref_hash(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ (uint8_t) s[i]) * 1099511628211ULL;
    }
    return h;
}

static void ref_map_init(struct ref_map *m, size_t n) {
    m->cap = 16;
    while (m->cap < 2 * n) {
        m->cap *= 2;
    }
    m->key  = calloc(m->cap, sizeof *m->key);
    m->klen = calloc(m->cap, sizeof *m->klen);
    m->val  = calloc(m->cap, sizeof *m->val);
}

static void ref_map_put(struct ref_map *m, const char *k, size_t n, int64_t v) {
    size_t i = ref_hash(k, n) & (m->cap - 1);
    while (m->key[i] != nullptr) {
        if (m->klen[i] == n && memcmp(m->key[i], k, n) == 0) {
            return;
        }
        i = (i + 1) & (m->cap - 1);
    }
    m->key[i] = malloc(n + 1);
    memcpy(m->key[i], k, n);
    m->klen[i] = n;
    m->val[i]  = v;
}

static int64_t ref_map_get(const struct ref_map *m, const char *k, size_t n) {
    for (size_t i = ref_hash(k, n) & (m->cap - 1); m->key[i] != nullptr;
         i        = (i + 1) & (m->cap - 1)) {
        if (m->klen[i] == n && memcmp(m->key[i], k, n) == 0) {
            return m->val[i];
        }
    }
    return -1;
}

static void ref_map_free(struct ref_map *m) {
    for (size_t i = 0; i < m->cap; i++) {
        free(m->key[i]);
    }
    free(m->key);
    free(m->klen);
    free(m->val);
}

struct ref_vocab {
    const struct tf_vocab *v;
    bool                   by_score; /* unigram: pairs forming a token, highest score */
    struct ref_map         tok;      /* token -> id */
    struct ref_map         merge;    /* left "\x01" right -> rank */
};

static void ref_vocab_init(struct ref_vocab *rv, const struct tf_vocab *v, bool by_score) {
    rv->v        = v;
    rv->by_score = by_score;
    ref_map_init(&rv->tok, v->n_tok);
    for (size_t i = 0; i < v->n_tok; i++) {
        ref_map_put(&rv->tok, v->tok[i], strlen(v->tok[i]), (int64_t) i);
    }
    ref_map_init(&rv->merge, v->n_merge);
    for (size_t m = 0; m < v->n_merge; m++) {
        char         k[160];
        const size_t n = (size_t) snprintf(k, sizeof k, "%s\x01%s", v->merge_l[m], v->merge_r[m]);
        ref_map_put(&rv->merge, k, n, (int64_t) m);
    }
}

static int32_t ref_tok_id(const struct ref_vocab *rv, const char *s, size_t n) {
    return (int32_t) ref_map_get(&rv->tok, s, n);
}

static int64_t
ref_rank(const struct ref_vocab *rv, const char *l, size_t ll, const char *r, size_t rl) {
    char k[160];
    if (ll + rl + 1 > sizeof k) {
        return -1;
    }
    memcpy(k, l, ll);
    k[ll] = '\x01';
    memcpy(k + ll + 1, r, rl);
    return ref_map_get(&rv->merge, k, ll + 1 + rl);
}

/* The old encode of an ASCII text without specials into ids; returns count. */
static size_t
ref_encode(const struct ref_vocab *rv, const char *text, bool space_prefix, int32_t *ids) {
    const struct tf_vocab *v   = rv->v;
    const size_t           tl  = strlen(text);
    char                  *buf = malloc(3 * tl + 3 + 1);
    size_t                 w   = 0;
    if (space_prefix) {
        memcpy(buf, SPM_MARKER_BYTES, 3);
        w = 3;
    }
    for (size_t i = 0; i < tl; i++) {
        if (text[i] == ' ') {
            memcpy(buf + w, SPM_MARKER_BYTES, 3);
            w += 3;
        } else {
            buf[w++] = text[i];
        }
    }
    /* symbols: ▁ is one 3-byte codepoint, letters one byte each */
    size_t                 n    = 0;
    struct pair_merge_sym *syms = malloc(w * sizeof *syms);
    for (size_t k = 0; k < w;) {
        const size_t len = (uint8_t) buf[k] == 0xE2 ? 3 : 1;
        syms[n]          = (struct pair_merge_sym) {
                .off = k, .len = len, .prev = (int) n - 1, .next = (int) n + 1};
        n++;
        k += len;
    }
    syms[n - 1].next = -1;
    while (true) {
        int     best     = -1;
        int64_t best_key = 0;
        float   best_sc  = 0.0f;
        for (int i = 0; syms[i].next >= 0; i = syms[i].next) {
            const struct pair_merge_sym *l = &syms[i], *r = &syms[syms[i].next];
            if (rv->by_score) {
                const int32_t id = ref_tok_id(rv, buf + l->off, l->len + r->len);
                if (id >= 0 && (best < 0 || v->score[id] > best_sc)) {
                    best    = i;
                    best_sc = v->score[id];
                }
            } else {
                const int64_t m = ref_rank(rv, buf + l->off, l->len, buf + r->off, r->len);
                if (m >= 0 && (best < 0 || m < best_key)) {
                    best     = i;
                    best_key = m;
                }
            }
        }
        if (best < 0) {
            break;
        }
        const int r = syms[best].next;
        syms[best].len += syms[r].len;
        syms[best].next = syms[r].next;
        if (syms[r].next >= 0) {
            syms[syms[r].next].prev = best;
        }
    }
    size_t n_ids = 0;
    for (int i = 0; i >= 0; i = syms[i].next) {
        const int32_t id = ref_tok_id(rv, buf + syms[i].off, syms[i].len);
        ids[n_ids++]     = id >= 0 ? id : (int32_t) v->unk; /* ASCII: no byte fallback needed */
    }
    free(syms);
    free(buf);
    return n_ids;
}

static bool load_gguf(struct gguf_tokenizer *tok, const struct tf_buf *g) {
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open_memory(g->b, g->n, &err);
    if (ctx == nullptr) {
        return false;
    }
    const bool ok = gguf_tokenizer_load_copy(tok, ctx);
    gguf_close(ctx);
    return ok;
}

static char *random_letters(size_t n) {
    char *t = malloc(n + 1);
    for (size_t i = 0; i < n; i++) {
        const uint32_t r = next_u32() % 32;
        t[i]             = r < 26 ? (char) ('a' + r) : ' ';
    }
    t[n] = '\0';
    return t;
}

static int check_encoders(void) {
    struct tf_vocab v  = tf_make_vocab(SPM_MARKER_BYTES, true);
    struct tf_buf   gs = tf_gguf(&v, "llama", true), gu = tf_gguf(&v, "llama", false);
    struct tf_buf   bin = tf_sp_bpe_bin(&v);

    struct gguf_tokenizer spm, uni;
    int                   fails = 0;
    if (!load_gguf(&spm, &gs) || !load_gguf(&uni, &gu)) {
        fprintf(stderr, "FAIL: fixture tokenizer did not load\n");
        return 1;
    }
    char path[] = "/tmp/test_tokenizer_merge_XXXXXX";
    int  fd     = mkstemp(path);
    if (fd < 0 || write(fd, bin.b, bin.n) != (ssize_t) bin.n) {
        fprintf(stderr, "FAIL: cannot write %s\n", path);
        return 1;
    }
    close(fd);
    struct sp_bpe_tokenizer *sp    = nullptr;
    const bool               sp_ok = sp_bpe_tokenizer_load(&sp, path);
    unlink(path);
    if (!sp_ok) {
        fprintf(stderr, "FAIL: sp_bpe fixture did not load\n");
        return 1;
    }

    struct ref_vocab by_rank, by_sc;
    ref_vocab_init(&by_rank, &v, false);
    ref_vocab_init(&by_sc, &v, true);
    int32_t *got   = malloc(8192 * sizeof *got);
    int32_t *want  = malloc(8192 * sizeof *want);
    int      texts = 0;
    for (int trial = 0; trial < 60; trial++) {
        const size_t n    = 1 + next_u32() % 3000;
        char        *text = trial % 3 == 0   ? tf_text(n, next_u32(), false)
                            : trial % 3 == 1 ? tf_text(n, next_u32(), true)
                                             : random_letters(n);
        /* GGUF SPM (add_space_prefix on by default) and unigram */
        for (int mode = 0; mode < 2; mode++) {
            size_t n_got = 0;
            if (gguf_tokenizer_encode(mode == 0 ? &spm : &uni, text, 8192, got, &n_got) !=
                GEIST_OK) {
                fprintf(stderr, "FAIL: encode failed\n");
                fails++;
                continue;
            }
            const size_t n_want = ref_encode(mode == 0 ? &by_rank : &by_sc, text, true, want);
            if (n_got != n_want || memcmp(got, want, n_got * sizeof *got) != 0) {
                fprintf(stderr,
                        "FAIL: %s differs from the rescan (%zu bytes, %zu vs %zu ids)\n",
                        mode == 0 ? "SPM" : "unigram",
                        n,
                        n_got,
                        n_want);
                fails++;
            }
        }
        /* sp_bpe: no leading ▁ */
        uint32_t *sp_ids = nullptr;
        size_t    n_sp   = 0;
        if (!sp_bpe_tokenizer_encode(sp, text, &sp_ids, &n_sp)) {
            fprintf(stderr, "FAIL: sp_bpe encode failed\n");
            fails++;
        } else {
            const size_t n_want = ref_encode(&by_rank, text, false, want);
            bool         same   = n_sp == n_want;
            for (size_t i = 0; same && i < n_sp; i++) {
                same = sp_ids[i] == (uint32_t) want[i];
            }
            if (!same) {
                fprintf(stderr, "FAIL: sp_bpe differs from the rescan (%zu bytes)\n", n);
                fails++;
            }
            safe_free((void **) &sp_ids);
        }
        free(text);
        texts++;
    }
    printf("  encoders: %d texts (words, one run, random letters; up to 3000 bytes), SPM, "
           "unigram and sp_bpe as the rescan\n",
           texts);
    free(got);
    free(want);
    ref_map_free(&by_rank.tok);
    ref_map_free(&by_rank.merge);
    ref_map_free(&by_sc.tok);
    ref_map_free(&by_sc.merge);
    sp_bpe_tokenizer_free(sp);
    gguf_tokenizer_unload(&spm);
    gguf_tokenizer_unload(&uni);
    free(gs.b);
    free(gu.b);
    free(bin.b);
    tf_free_vocab(&v);
    return fails;
}

int main(void) {
    int fails = check_engine();
    fails += check_encoders();
    if (fails != 0) {
        fprintf(stderr, "FAIL: %d check(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: the merge heap makes the rescan's merges\n");
    return GEIST_TEST_PASS;
}
