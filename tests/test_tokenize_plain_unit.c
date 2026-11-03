/*
 * test_tokenize_plain_unit — special-token parsing off inside plain ranges
 * (#409, gguf_tokenizer_encode_plain / geist_session_tokenize_plain).
 *
 * Inside a plain range only the USER_DEFINED entries that are not
 * control-looking are matched as special tokens (llama.cpp's
 * parse_special=false); CONTROL, UNKNOWN and control-looking ones are text,
 * and a special token that reaches into a range is not matched at all.
 * Outside the ranges nothing changes. In-memory GGUFs (model "gpt2", no
 * merges, so every other byte is its own id) run each case through the four
 * GPT-2 pretokenizer paths: qwen2 (llama3 regex), gpt-2, deepseek-llm (regex
 * sequence) and an unknown name (the simple splitter). The real vocabularies
 * against llama-tokenize --no-parse-special: test_tokenize_plain_int.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "gguf_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct blob {
    uint8_t b[8192];
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

enum { VT_U32 = 4, VT_I32 = 5, VT_STRING = 8, VT_ARRAY = 9 };

/* ids 0..93: the printable bytes '!'..'~'; 94: the GPT-2 space "Ġ"; then the
 * specials. */
#define CH(c) ((int32_t) (c) - 0x21)
enum { ID_SPACE = 94, N_BYTES = 95 };
static const struct {
    const char *text;
    int32_t     type;
} SPECIAL[] = {
        {"<|im_start|>", 3}, /* @0 CONTROL */
        {"<|im_end|>", 4},   /* @1 USER_DEFINED, but control-looking: CONTROL too */
        {"<think>", 4},      /* @2 USER_DEFINED: matched in plain text as well */
        {"<unk>", 2},        /* @3 UNKNOWN */
        {"</s>", 1},         /* @4 NORMAL, control-looking */
        {"ab", 3},           /* @5 CONTROL, for a match that reaches into a range */
};
enum { N_SPECIAL = sizeof SPECIAL / sizeof SPECIAL[0], N_VOCAB = N_BYTES + N_SPECIAL };

static void build(struct blob *o, const char *pre) {
    *o = (struct blob) {0};
    put(o, "GGUF", 4);
    put_u32(o, 3); /* version */
    put_u64(o, 1); /* n_tensors — the reader refuses 0 */
    put_u64(o, 5);

    put_str(o, "tokenizer.ggml.model");
    put_u32(o, VT_STRING);
    put_str(o, "gpt2");

    put_str(o, "tokenizer.ggml.pre");
    put_u32(o, VT_STRING);
    put_str(o, pre);

    put_str(o, "tokenizer.ggml.tokens");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_STRING);
    put_u64(o, N_VOCAB);
    for (int c = 0x21; c <= 0x7e; c++) {
        const char s[2] = {(char) c, 0};
        put_str(o, s);
    }
    put_str(o, "\xc4\xa0"); /* U+0120, GPT-2's space */
    for (size_t s = 0; s < N_SPECIAL; s++)
        put_str(o, SPECIAL[s].text);

    put_str(o, "tokenizer.ggml.token_type");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_I32);
    put_u64(o, N_VOCAB);
    for (size_t i = 0; i < N_VOCAB; i++)
        put_u32(o, i < N_BYTES ? 1u : (uint32_t) SPECIAL[i - N_BYTES].type);

    put_str(o, "tokenizer.ggml.merges");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_STRING);
    put_u64(o, 0);

    put_str(o, "t"); /* a 1-element f32 tensor: name, n_dims, dim, dtype, offset */
    put_u32(o, 1);
    put_u64(o, 1);
    put_u32(o, 0);
    put_u64(o, 0);
    static const uint8_t zeros[32] = {0};
    put(o, zeros, (32 - o->n % 32) % 32); /* the data section is 32-byte aligned */
    put(o, zeros, 4);                     /* the tensor's 4 bytes */
}

/* want: the expected ids spelled as text, "@k" for SPECIAL[k], ' ' for the
 * GPT-2 space, any other byte for its own id. */
static size_t want_ids(const char *want, int32_t out[static 64]) {
    size_t n = 0;
    for (const char *p = want; *p != '\0' && n < 64; p++) {
        if (*p == '@' && p[1] >= '0' && p[1] <= '9')
            out[n++] = N_BYTES + (p[1] - '0'), p++;
        else
            out[n++] = *p == ' ' ? ID_SPACE : CH(*p);
    }
    return n;
}

struct plain_case {
    const char             *text;
    size_t                  n_plain;
    struct geist_text_range plain[2];
    const char             *want;
};

#define ALL(s) {{0, sizeof(s) - 1}}
static const struct plain_case CASES[] = {
        /* no ranges: every special */
        {"a<|im_start|>b", 0, {}, "a@0b"},
        {"x<unk></s><|im_end|><think>", 0, {}, "x@3@4@1@2"},
        /* the whole text plain */
        {"a<|im_start|>b", 1, ALL("a<|im_start|>b"), "a<|im_start|>b"},
        {"a<|im_end|>b", 1, ALL("a<|im_end|>b"), "a<|im_end|>b"},
        {"x<unk></s>", 1, ALL("x<unk></s>"), "x<unk></s>"},
        {"a<think>b", 1, ALL("a<think>b"), "a@2b"},
        /* a rendered turn whose body is plain: the markers around it stay */
        {"<|im_start|>u<|im_end|>v<|im_end|>", 1, {{12, 24}}, "@0u<|im_end|>v@1"},
        {"a <|im_start|>b <|im_end|>", 1, {{2, 14}}, "a <|im_start|>b @1"},
        {"a <|im_start|>b", 0, {}, "a @0b"},
        /* two ranges, a special between them */
        {"<|im_end|>x<|im_end|>", 2, {{0, 10}, {11, 21}}, "<|im_end|>x<|im_end|>"},
        {"<|im_end|>x<|im_end|>", 2, {{0, 1}, {20, 21}}, "<|im_end|>x<|im_end|>"},
        {"<|im_end|><|im_end|>", 2, {{0, 10}, {20, 20}}, "<|im_end|>@1"},
        /* a special that reaches into a range is not matched ("-": the
         * simple splitter looks for specials only where a chunk starts) */
        {"-ab", 1, {{0, 2}}, "-ab"},
        {"-ab", 1, {{2, 3}}, "-ab"},
        {"-ab", 1, {{0, 1}}, "-@5"},
        {"-ab", 1, {{3, 3}}, "-@5"},
        /* empty ranges are no ranges */
        {"a<|im_start|>b", 2, {{1, 1}, {5, 5}}, "a@0b"},
};

static const char *const PRE[] = {"qwen2", "gpt-2", "deepseek-llm", "geist-test-unknown"};

static bool load(struct gguf_tokenizer *tok, const char *pre) {
    struct blob o;
    build(&o, pre);
    if (o.overflow) {
        fprintf(stderr, "blob buffer too small\n");
        return false;
    }
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open_memory(o.b, o.n, &err);
    if (ctx == nullptr) {
        fprintf(stderr, "gguf_open_memory: %s\n", err != nullptr ? err : "?");
        return false;
    }
    const bool loaded = gguf_tokenizer_load_copy(tok, ctx);
    gguf_close(ctx);
    if (!loaded)
        fprintf(stderr, "tokenizer load failed (%s)\n", pre);
    return loaded;
}

static int check(const struct gguf_tokenizer *tok, const char *pre, const struct plain_case *c) {
    int32_t           want[64], ids[64];
    const size_t      n_want = want_ids(c->want, want);
    size_t            n      = 0;
    enum geist_status st =
            gguf_tokenizer_encode_plain(tok, c->text, c->n_plain, c->plain, 64, ids, &n);
    bool ok = st == GEIST_OK && n == n_want;
    for (size_t k = 0; ok && k < n; k++)
        ok = ids[k] == want[k];
    /* No ranges is gguf_tokenizer_encode. */
    if (ok && c->n_plain == 0) {
        int32_t ref[64];
        size_t  n_ref = 0;
        ok = gguf_tokenizer_encode(tok, c->text, 64, ref, &n_ref) == GEIST_OK && n_ref == n &&
             memcmp(ref, ids, n * sizeof ids[0]) == 0;
    }
    /* Exactly enough room fits, one slot less is refused with nothing written. */
    if (ok && n > 0) {
        size_t n_short = 99;
        ok = gguf_tokenizer_encode_plain(tok, c->text, c->n_plain, c->plain, n, ids, &n) ==
                     GEIST_OK &&
             gguf_tokenizer_encode_plain(
                     tok, c->text, c->n_plain, c->plain, n - 1, ids, &n_short) ==
                     GEIST_E_INVALID_ARG &&
             n_short == 0;
    }
    if (!ok) {
        fprintf(stderr, "FAIL [%s] '%s' (%zu ranges):", pre, c->text, c->n_plain);
        for (size_t k = 0; k < n; k++)
            fprintf(stderr, " %d", ids[k]);
        fprintf(stderr, " (status %d), want", (int) st);
        for (size_t k = 0; k < n_want; k++)
            fprintf(stderr, " %d", want[k]);
        fprintf(stderr, "\n");
    }
    return !ok;
}

/* Ranges out of order, overlapping, past the text or missing are refused. */
static int check_invalid(const struct gguf_tokenizer *tok) {
    static const struct {
        size_t                  n;
        struct geist_text_range r[2];
    } BAD[] = {
            {1, {{2, 1}}},
            {2, {{0, 3}, {2, 4}}},
            {2, {{3, 4}, {0, 1}}},
            {1, {{0, 6}}},
            {1, {{6, 6}}},
    };
    int fails = 0;
    for (size_t b = 0; b < sizeof BAD / sizeof BAD[0]; b++) {
        int32_t ids[8];
        size_t  n = 99;
        if (gguf_tokenizer_encode_plain(tok, "a<|b|", BAD[b].n, BAD[b].r, 8, ids, &n) !=
                    GEIST_E_INVALID_ARG ||
            n != 0) {
            fprintf(stderr, "FAIL: invalid range set %zu accepted\n", b);
            fails++;
        }
    }
    size_t n = 99;
    if (gguf_tokenizer_encode_plain(tok, "a", 1, nullptr, 0, nullptr, &n) != GEIST_E_INVALID_ARG ||
        n != 0) {
        fprintf(stderr, "FAIL: n_plain 1 with no ranges accepted\n");
        fails++;
    }
    /* The whole text, end == strlen: accepted. */
    const struct geist_text_range whole = {0, 5};
    int32_t                       ids[8];
    if (gguf_tokenizer_encode_plain(tok, "a<|b|", 1, &whole, 8, ids, &n) != GEIST_OK || n != 5) {
        fprintf(stderr, "FAIL: a range ending at the text's end refused\n");
        fails++;
    }
    return fails;
}

int main(void) {
    int    fails = 0;
    size_t runs  = 0;
    for (size_t p = 0; p < sizeof PRE / sizeof PRE[0]; p++) {
        struct gguf_tokenizer tok;
        if (!load(&tok, PRE[p]))
            return GEIST_TEST_ERROR;
        if (tok.n_specials != N_SPECIAL || tok.n_user_specials != 1) {
            fprintf(stderr,
                    "FAIL [%s]: %zu specials, %zu user (want %d, 1)\n",
                    PRE[p],
                    tok.n_specials,
                    tok.n_user_specials,
                    (int) N_SPECIAL);
            fails++;
        }
        for (size_t c = 0; c < sizeof CASES / sizeof CASES[0]; c++, runs++)
            fails += check(&tok, PRE[p], &CASES[c]);
        fails += check_invalid(&tok);
        gguf_tokenizer_unload(&tok);
    }
    printf("tokenize plain: %zu encodes over %zu pretokenizers, %d failed\n",
           runs,
           sizeof PRE / sizeof PRE[0],
           fails);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
