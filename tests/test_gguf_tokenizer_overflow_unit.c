/*
 * test_gguf_tokenizer_overflow_unit — a text that needs more ids than the
 * caller has room for is an error, not a shortened encoding (AGENT.md §5).
 *
 * geist_session_tokenize documents GEIST_E_INVALID_ARG on overflow. Its
 * sp_bpe path did that; its GGUF-embedded path returned GEIST_OK with the
 * first out_capacity ids, because gguf_tokenizer_encode stopped at its cap
 * and still reported success. Pinned here on synthetic tokenizers for all
 * three encode paths — SPM (merges), unigram (scores, no merges) and gpt2 —
 * with a text that ends in ordinary pieces and one that ends in a special
 * token, whose write had a truncating guard of its own. No fixture needed.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "gguf_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---- in-memory GGUF carrying only a tokenizer --------------------------- */

struct blob {
    uint8_t b[1024];
    size_t  n;
    bool    overflow;
};

static void put(struct blob *o, const void *p, size_t n) {
    if (n > sizeof o->b - o->n) {
        o->overflow = true;
        return;
    }
    memcpy(o->b + o->n, p, n);
    o->n += n;
}
static void put_u32(struct blob *o, uint32_t v) {
    put(o, &v, 4);
}
static void put_u64(struct blob *o, uint64_t v) {
    put(o, &v, 8);
}
static void put_str(struct blob *o, const char *s) {
    put_u64(o, strlen(s));
    put(o, s, strlen(s));
}

enum { VT_I32 = 5, VT_F32 = 6, VT_STRING = 8, VT_ARRAY = 9 };

static void put_arr(struct blob *o, const char *key, uint32_t elem_vt, uint64_t n) {
    put_str(o, key);
    put_u32(o, VT_ARRAY);
    put_u32(o, elem_vt);
    put_u64(o, n);
}

/* "▁" is SentencePiece's space, "Ġ" gpt2's; "</s>" is the one CONTROL token,
 * so the encoders match it as a special. */
static const char *const VOCAB[] = {"<unk>", "</s>", "\xe2\x96\x81", "\xc4\xa0", "a", "b", "ab"};
constexpr size_t         N_VOCAB = sizeof VOCAB / sizeof VOCAB[0];

/* model "llama" with merges selects SPM, without merges (scores only)
 * unigram; "gpt2" selects byte-level BPE. */
static void build(struct blob *o, const char *model, bool merges) {
    *o = (struct blob) {0};
    put(o, "GGUF", 4);
    put_u32(o, 3);              /* version */
    put_u64(o, 1);              /* n_tensors — the reader refuses 0 */
    put_u64(o, merges ? 5 : 4); /* n_kv */

    put_str(o, "tokenizer.ggml.model");
    put_u32(o, VT_STRING);
    put_str(o, model);

    put_arr(o, "tokenizer.ggml.tokens", VT_STRING, N_VOCAB);
    for (size_t i = 0; i < N_VOCAB; i++)
        put_str(o, VOCAB[i]);

    put_arr(o, "tokenizer.ggml.scores", VT_F32, N_VOCAB);
    for (size_t i = 0; i < N_VOCAB; i++) {
        const float score = -(float) i;
        put(o, &score, 4);
    }

    put_arr(o, "tokenizer.ggml.token_type", VT_I32, N_VOCAB);
    for (size_t i = 0; i < N_VOCAB; i++)
        put_u32(o, i == 1 ? 3 : 1); /* CONTROL for "</s>", NORMAL else */

    if (merges) {
        put_arr(o, "tokenizer.ggml.merges", VT_STRING, 1);
        put_str(o, "a b");
    }

    put_str(o, "t"); /* a 1-element f32 tensor: name, n_dims, dim, dtype, offset */
    put_u32(o, 1);
    put_u64(o, 1);
    put_u32(o, 0);
    put_u64(o, 0);
    static const uint8_t zeros[32] = {0};
    put(o, zeros, (32 - o->n % 32) % 32); /* the data section is 32-byte aligned */
    put(o, zeros, 4);                     /* the tensor's 4 bytes */
}

/* ---- the checks --------------------------------------------------------- */

static int expect(bool cond, const char *mode, const char *text, const char *what) {
    if (!cond)
        fprintf(stderr, "FAIL: %s \"%s\": %s\n", mode, text, what);
    return cond ? 0 : 1;
}

static int check(const char *mode, const struct gguf_tokenizer *tok, const char *text) {
    int32_t ids[64];
    size_t  n_all = 0;
    if (gguf_tokenizer_encode(tok, text, sizeof ids / sizeof ids[0], ids, &n_all) != GEIST_OK ||
        n_all < 2)
        return expect(false, mode, text, "reference encode needs >= 2 ids");

    int    fails = 0;
    size_t n     = SIZE_MAX;
    fails += expect(gguf_tokenizer_encode(tok, text, n_all, ids, &n) == GEIST_OK && n == n_all,
                    mode,
                    text,
                    "exact fit is OK");

    ids[0] = -1;
    n      = SIZE_MAX;
    fails += expect(gguf_tokenizer_encode(tok, text, n_all - 1, ids, &n) == GEIST_E_INVALID_ARG,
                    mode,
                    text,
                    "one id short is GEIST_E_INVALID_ARG");
    fails += expect(n == 0 && ids[0] == -1, mode, text, "one id short writes nothing");

    n = SIZE_MAX;
    fails += expect(gguf_tokenizer_encode(tok, text, 0, ids, &n) == GEIST_E_INVALID_ARG && n == 0,
                    mode,
                    text,
                    "cap 0 is GEIST_E_INVALID_ARG");
    return fails;
}

int main(void) {
    static const struct {
        const char              *mode, *model;
        bool                     merges;
        enum gguf_tokenizer_mode want;
    } variants[] = {
            {"spm", "llama", true, GGUF_TOK_MODE_SPM},
            {"unigram", "llama", false, GGUF_TOK_MODE_UNIGRAM},
            {"gpt2", "gpt2", true, GGUF_TOK_MODE_GPT2},
    };
    static const char *const texts[] = {"ab ab", "ab</s>"};

    int fails = 0;
    for (size_t v = 0; v < sizeof variants / sizeof variants[0]; v++) {
        struct blob o;
        build(&o, variants[v].model, variants[v].merges);
        if (o.overflow) {
            fprintf(stderr, "blob buffer too small\n");
            return GEIST_TEST_ERROR;
        }
        const char      *err = nullptr;
        struct gguf_ctx *ctx = gguf_open_memory(o.b, o.n, &err);
        if (ctx == nullptr) {
            fprintf(stderr, "gguf_open_memory: %s\n", err != nullptr ? err : "?");
            return GEIST_TEST_ERROR;
        }
        struct gguf_tokenizer tok;
        const bool            loaded = gguf_tokenizer_load_copy(&tok, ctx);
        gguf_close(ctx);
        if (!loaded) {
            fprintf(stderr, "%s: tokenizer load failed\n", variants[v].mode);
            return GEIST_TEST_ERROR;
        }
        fails += expect(tok.mode == variants[v].want, variants[v].mode, "", "selects its mode");

        for (size_t t = 0; t < sizeof texts / sizeof texts[0]; t++)
            fails += check(variants[v].mode, &tok, texts[t]);

        /* Nothing to encode needs no room: cap 0 with no buffer is OK. */
        size_t n = SIZE_MAX;
        fails += expect(gguf_tokenizer_encode(&tok, "", 0, nullptr, &n) == GEIST_OK && n == 0,
                        variants[v].mode,
                        "",
                        "empty text, cap 0, no buffer is OK");

        gguf_tokenizer_unload(&tok);
    }

    if (fails == 0)
        printf("PASS: gguf_tokenizer_encode fails instead of truncating (spm, unigram, gpt2)\n");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
