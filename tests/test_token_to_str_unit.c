/*
 * test_token_to_str_unit — geist_session_token_to_str returns each token's
 * surface form as a NUL-terminated string that stays valid for the
 * session's lifetime, as include/geist.h promises (STABLE since 0.1.0).
 *
 * It did not. For a GGUF-embedded tokenizer it decoded into one thread-local
 * 256-byte buffer: each call overwrote the string the call before had
 * returned, and a token longer than 254 bytes came back cut short. For a
 * tokenizer.bin it returned the piece's bytes inside the file's mapping,
 * which are not NUL-terminated, so strlen ran on into the next entry's
 * length field (the "trailer byte" test_audio_chat_e2e strips).
 *
 * One-layer llama GGUFs with F32 weights carry the tokenizer_fixtures vocab,
 * plus long tokens (255 to 70000 bytes), as every tokenizer the loader
 * takes: gpt2 byte-level BPE, SentencePiece BPE and unigram embedded in the
 * GGUF (loaded from memory), and a tokenizer.bin beside a GGUF without one
 * (loaded from a file). Every id's string must be its decoded surface form,
 * with the right length; a repeat call must return that same string; and it
 * must still be that after all other ids were converted, after
 * geist_session_reset, and after a second session came and went.
 */
#include "test_helpers.h"
#include "model_fixtures.h"
#include "tokenizer_fixtures.h"

#include <geist.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { D = 64, FFN = 64 };

/* Letters only, so every kind decodes them to themselves: either side of a
 * 256-byte buffer, past the old 254-byte cut, and past 4 KB and 64 KB. A
 * tokenizer.bin stores 16-bit lengths, so it gets all but the last. */
static const size_t LONG_TOKENS[] = {255, 256, 300, 5000, 70000};
enum { N_LONG = sizeof LONG_TOKENS / sizeof LONG_TOKENS[0] };

enum kind { GPT2, SPM, UNIGRAM, TOKENIZER_BIN };
static const char *const KIND_NAME[] = {"gpt2", "SentencePiece BPE", "unigram", "tokenizer.bin"};

static const char GPT2_MARKER[] = "\xc4\xa0";     /* Ġ */
static const char SPM_MARKER[]  = "\xe2\x96\x81"; /* ▁ */

/* A one-layer llama, d_model 64 in one head (model_fixtures.h). With
 * tok_model set it carries v as that GGUF tokenizer, else none. */
static struct tf_buf model_gguf(const struct tf_vocab *v, const char *tok_model, bool with_merges) {
    return mf_llama_gguf(&(struct mf_llama) {.layers     = 1,
                                             .d_model    = D,
                                             .heads      = 1,
                                             .kv_heads   = 1,
                                             .ffn        = FFN,
                                             .vocab      = (uint32_t) v->n_tok,
                                             .context    = 64,
                                             .tok        = tok_model != nullptr ? v : nullptr,
                                             .tok_model  = tok_model,
                                             .tok_merges = with_merges});
}

/* The fixture vocab for a kind, then its long tokens. */
static struct tf_vocab vocab(enum kind k, size_t *n_long) {
    struct tf_vocab v = tf_make_vocab(k == GPT2 ? GPT2_MARKER : SPM_MARKER, k != GPT2);
    *n_long           = k == TOKENIZER_BIN ? N_LONG - 1 : N_LONG;
    for (size_t i = 0; i < *n_long; i++) {
        char *s = xmalloc(LONG_TOKENS[i]);
        memset(s, 'x', LONG_TOKENS[i]);
        tf_add_tok(&v, s, LONG_TOKENS[i], -50.0f);
        free(s);
    }
    return v;
}

/* The string token_to_str must return for vocab entry tok: gpt2 maps Ġ back
 * to a space; the GGUF SentencePiece modes map ▁ to a space and <0xXX> to
 * that byte; a tokenizer.bin piece comes back as it is stored. Writes it to
 * out (strlen(tok) + 2 bytes) and returns its length as a C string (<0x00>
 * is the empty string). */
static size_t surface(enum kind k, const char *tok, char *out) {
    const size_t n = strlen(tok);
    if ((k == SPM || k == UNIGRAM) && n == 6 && memcmp(tok, "<0x", 3) == 0 && tok[5] == '>') {
        out[0] = (char) strtoul(tok + 3, nullptr, 16);
        out[1] = '\0';
        return strlen(out);
    }
    const char  *marker = k == GPT2 ? GPT2_MARKER : SPM_MARKER;
    const size_t ml     = strlen(marker);
    size_t       w      = 0;
    for (size_t i = 0; i < n;) {
        if (k != TOKENIZER_BIN && n - i >= ml && memcmp(tok + i, marker, ml) == 0) {
            out[w++] = ' ';
            i += ml;
        } else {
            out[w++] = tok[i++];
        }
    }
    out[w] = '\0';
    return w;
}

static bool is_surface(const char *got, enum kind k, const char *tok) {
    char        *want = xmalloc(strlen(tok) + 2);
    const size_t n    = surface(k, tok, want);
    const bool   ok   = got != nullptr && strlen(got) == n && memcmp(got, want, n) == 0;
    free(want);
    return ok;
}

/* How many of the strings in got are no longer their token's. */
static size_t n_wrong(const char *const *got, enum kind k, const struct tf_vocab *v) {
    size_t bad = 0;
    for (size_t i = 0; i < v->n_tok; i++) {
        bad += !is_surface(got[i], k, v->tok[i]);
    }
    return bad;
}

static int expect_none_wrong(size_t bad, size_t n, enum kind k, const char *when) {
    char what[160];
    snprintf(what,
             sizeof what,
             "%s: all %zu strings right %s (%zu wrong)",
             KIND_NAME[k],
             n,
             when,
             bad);
    return geist_expect(bad == 0, what);
}

static int check_model(struct geist_backend  *be,
                       struct geist_model    *m,
                       enum kind              k,
                       const struct tf_vocab *v,
                       size_t                 n_long) {
    struct geist_session *s = nullptr;
    if (geist_session_create(m, be, nullptr, &s) != GEIST_OK) {
        fprintf(stderr, "FAIL: %s: session_create\n", KIND_NAME[k]);
        return 1;
    }
    const size_t n     = v->n_tok;
    const char **got   = xmalloc(n * sizeof *got);
    int          fails = 0;
    for (size_t i = 0; i < n; i++) {
        got[i] = geist_session_token_to_str(s, (geist_token_t) i);
    }
    /* Checked only now: every string must have outlived the calls after it. */
    fails += expect_none_wrong(n_wrong(got, k, v), n, k, "after converting every id");
    char what[160];
    for (size_t i = n - n_long; i < n; i++) {
        snprintf(what,
                 sizeof what,
                 "%s: the %zu-byte token comes back whole",
                 KIND_NAME[k],
                 strlen(v->tok[i]));
        fails += geist_expect(is_surface(got[i], k, v->tok[i]), what);
    }
    size_t moved = 0;
    for (size_t i = 0; i < n; i++) {
        moved += geist_session_token_to_str(s, (geist_token_t) i) != got[i];
    }
    snprintf(what,
             sizeof what,
             "%s: a repeat returns the string it returned before (%zu did not)",
             KIND_NAME[k],
             moved);
    fails += geist_expect(moved == 0, what);

    fails += geist_expect(geist_session_reset(s) == GEIST_OK, "session_reset");
    fails += expect_none_wrong(n_wrong(got, k, v), n, k, "after geist_session_reset");

    struct geist_session *s2 = nullptr;
    if (geist_session_create(m, be, nullptr, &s2) == GEIST_OK) {
        const char **got2 = xmalloc(n * sizeof *got2);
        for (size_t i = 0; i < n; i++) {
            got2[i] = geist_session_token_to_str(s2, (geist_token_t) i);
        }
        fails += expect_none_wrong(n_wrong(got2, k, v), n, k, "in a second session");
        free(got2);
        geist_session_destroy(s2);
    } else {
        fails += geist_expect(false, "second session_create");
    }
    fails += expect_none_wrong(n_wrong(got, k, v), n, k, "after the second session ended");

    /* Out of range: the GGUF tokenizer decodes it as "<unk>", a tokenizer.bin
     * has nothing; a negative id is nothing everywhere. Unchanged behaviour. */
    const char *past = geist_session_token_to_str(s, (geist_token_t) n);
    snprintf(what,
             sizeof what,
             "%s: an id past the vocab gives %s",
             KIND_NAME[k],
             k == TOKENIZER_BIN ? "nullptr" : "\"<unk>\"");
    fails += geist_expect(k == TOKENIZER_BIN ? past == nullptr
                                             : past != nullptr && strcmp(past, "<unk>") == 0,
                          what);
    snprintf(what, sizeof what, "%s: a negative id gives nullptr", KIND_NAME[k]);
    fails += geist_expect(geist_session_token_to_str(s, -1) == nullptr, what);

    free(got);
    geist_session_destroy(s);
    return fails;
}

static int run_embedded(struct geist_backend *be, enum kind k) {
    size_t              n_long = 0;
    struct tf_vocab     v      = vocab(k, &n_long);
    struct tf_buf       g      = model_gguf(&v, k == GPT2 ? "gpt2" : "llama", k != UNIGRAM);
    struct geist_model *m      = nullptr;
    int                 fails;
    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK) {
        fprintf(stderr,
                "FAIL: %s: model did not load: %s\n",
                KIND_NAME[k],
                geist_last_create_error());
        fails = 1;
    } else {
        fails = check_model(be, m, k, &v, n_long);
        geist_model_destroy(m);
    }
    free(g.b);
    tf_free_vocab(&v);
    return fails;
}

static bool write_file(const char *path, const struct tf_buf *b) {
    FILE *f = fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = fwrite(b->b, 1, b->n, f) == b->n;
    return fclose(f) == 0 && ok;
}

/* A GGUF without a tokenizer and the vocab as tokenizer.bin beside it. */
static int run_tokenizer_bin(struct geist_backend *be) {
    size_t          n_long = 0;
    struct tf_vocab v      = vocab(TOKENIZER_BIN, &n_long);
    struct tf_buf   g      = model_gguf(&v, nullptr, false);
    struct tf_buf   bin    = tf_sp_bpe_bin(&v);
    char            dir[]  = "/tmp/test_token_to_str_XXXXXX";
    char            gpath[64], tpath[64];
    int             fails = 1;
    if (mkdtemp(dir) == nullptr) {
        fprintf(stderr, "FAIL: mkdtemp\n");
    } else {
        snprintf(gpath, sizeof gpath, "%s/model.gguf", dir);
        snprintf(tpath, sizeof tpath, "%s/tokenizer.bin", dir);
        /* The loader tries GEIST_TOKENIZER_PATH first: point it here, not at
         * whatever the environment names. */
        if (!write_file(gpath, &g) || !write_file(tpath, &bin) ||
            setenv("GEIST_TOKENIZER_PATH", tpath, 1) != 0) {
            fprintf(stderr, "FAIL: cannot write the fixture files in %s\n", dir);
        } else {
            struct geist_model *m = nullptr;
            if (geist_model_load(gpath, be, &m) != GEIST_OK) {
                fprintf(stderr,
                        "FAIL: tokenizer.bin: model did not load: %s\n",
                        geist_last_create_error());
            } else {
                fails = check_model(be, m, TOKENIZER_BIN, &v, n_long);
                geist_model_destroy(m);
            }
        }
        unlink(gpath);
        unlink(tpath);
        rmdir(dir);
    }
    free(g.b);
    free(bin.b);
    tf_free_vocab(&v);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_scalar backend did not register\n");
        return GEIST_TEST_SKIP;
    }
    int fails = 0;
    fails += run_embedded(be, GPT2);
    fails += run_embedded(be, SPM);
    fails += run_embedded(be, UNIGRAM);
    fails += run_tokenizer_bin(be);
    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: token strings are NUL-terminated and outlive the calls after them\n");
    return GEIST_TEST_PASS;
}
