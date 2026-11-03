/*
 * test_llama3_pretok_unit — the llama3 / qwen2 / qwen35 pretokenizer
 * scanner and what reaches it (#399).
 *
 * Llama 3 GGUFs ("gpt2" + "llama-bpe") went through geist's simplified GPT-2
 * splitter, so " 2000" became " 2" "000" instead of llama.cpp's " " "200"
 * "0", " ..." became " ." "." ".", and the wikitext-2 test text left
 * llama-tokenize at token 39. The qwen2 scanner, which now serves both, had
 * its own corners: \p{L} / \p{N} approximated (², ½ counted as letters), and
 * any whitespace codepoint, not only a space, taken as the lead of a
 * punctuation run.
 *
 *   1. gguf_tokenizer_llama3_chunk_end against chunk lists worked out from
 *      llama.cpp's unicode_regex_split_custom_llama3 / _qwen2 / _qwen35
 *      (chunks separated by '\x01').
 *   2. gguf_tokenizer_encode on in-memory GGUFs: the pretokenizer names
 *      reach the scanner, Llama 3 skips the merges for a chunk that is a
 *      vocab entry (ignore_merges), and a whitespace run before a special
 *      token ends at it.
 *
 * Parity with llama.cpp on the real vocabularies (wikitext-2 test text plus
 * unicode / code / number / whitespace / contraction cases) is checked
 * against llama-tokenize outside the unit suite (#399 PR).
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "gguf_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---- 1. the scanner ------------------------------------------------------ */

struct chunk_case {
    size_t      max_digits;
    bool        marks;
    const char *spec;
};

static const struct chunk_case CHUNK_CASES[] = {
        /* llama3 */
        {3, false, "Hello\x01 world"},
        {3,
         false,
         "200\x01"
         "0"},
        {3,
         false,
         " \x01"
         "200\x01"
         "0"},
        {3,
         false,
         "123\x01"
         "456\x01"
         "7"},
        {3,
         false,
         "x\x01 =\x01 \x01"
         "1\x01;\n"},
        {3, false, "end\x01 ...\x01 \r\n"},
        {3, false, "!!!\n\n"},
        {3, false, "don\x01't"},
        {3, false, "I\x01'M\x01 HERE"},
        {3, false, "they\x01'LL"},
        {3, false, "we\x01'Ve"},
        {3, false, "(Hello"},
        {3, false, "\n\x01Hello"},
        {3, false, "a\x01 \x01 b"},
        {3, false, "a\x01 \x01 !"},
        {3, false, "\t\x01\tif"},
        {3, false, "\t\x01!"},
        {3, false, "x\x01 \n \n\x01 y"},
        {3, false, "trailing\x01   "},
        {3, false, "\xE3\x80\x80\x01're"},           /* U+3000 then 're */
        {3, false, "x\x01\xC2\xB2\xC2\xB3\xC2\xB9"}, /* x²³¹: ² is \p{N} */
        {3, false, "\xC2\xBD\x01's"},                /* ½'s */
        {3, false, "cafe\x01\xCC\x81\x01 x"},        /* U+0301 is no letter */
        {3, false, "\xC2\xA0word"},                  /* NBSP prefixes a word */
        {3, false, " \x01\xC2\xA0word"},
        {3,
         false,
         "a\x01\xC2\x85"
         "b"},                                              /* U+0085 is \s */
        {3, false, "\xD9\xA1\xD9\xA2\xD9\xA3\x01\xD9\xA4"}, /* Arabic-Indic digits */
        {3, false, "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E"}, /* 日本語 */
        {3, false, "\xF0\x9F\x9A\x80\x01 ok"},              /* 🚀 ok */
        /* qwen2: single digits */
        {1,
         false,
         " \x01"
         "2\x01"
         "0\x01"
         "0\x01"
         "0"},
        {1, false, "x\x01\xC2\xB2"},
        {1, false, "cafe\x01\xCC\x81\x01 x"},
        /* qwen35: \p{M} joins the letter run */
        {1, true, "cafe\xCC\x81\x01 x"},
        {1, true, "\xCC\x81\xCC\x81x"},
        {1, true, " \xCC\x81x"},
};

static int check_chunks(const struct chunk_case *c) {
    char        text[256], want[16][64];
    int         n_want = 0;
    size_t      tl     = 0;
    const char *p      = c->spec;
    while (*p != '\0') {
        const char  *e = strchr(p, '\x01');
        const size_t n = e != nullptr ? (size_t) (e - p) : strlen(p);
        memcpy(want[n_want], p, n);
        want[n_want++][n] = '\0';
        memcpy(text + tl, p, n);
        tl += n;
        p += n + (e != nullptr);
    }
    text[tl] = '\0';
    size_t i = 0;
    for (int k = 0; k < n_want; k++) {
        const size_t end =
                i < tl ? gguf_tokenizer_llama3_chunk_end(tl, i, c->max_digits, c->marks, text) : i;
        const size_t len = strlen(want[k]);
        if (end != i + len || memcmp(text + i, want[k], len) != 0) {
            fprintf(stderr,
                    "FAIL: digits %zu marks %d '%s': chunk %d is '%.*s', want '%s'\n",
                    c->max_digits,
                    c->marks,
                    text,
                    k,
                    (int) (end - i),
                    text + i,
                    want[k]);
            return 1;
        }
        i = end;
    }
    return geist_expect(i == tl, "the chunks cover the text");
}

/* ---- 2. through the encoder ---------------------------------------------- */

struct blob {
    uint8_t b[2048];
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

enum { VT_I32 = 5, VT_STRING = 8, VT_ARRAY = 9 };

/* GPT-2 byte-level pieces: \xc4\xa0 is the space (Ġ). No merges, so every
 * multi-byte id below comes from ignore_merges alone. */
static const char *const VOCAB[] = {
        "2",
        "0",
        "200",
        "a",
        "b",
        "ab",
        "\xc4\xa0",
        "\xc4\xa0\xc4\xa0",
        "<|eot_id|>",
};
enum { ID_2, ID_0, ID_200, ID_A, ID_B, ID_AB, ID_SP, ID_SP2, ID_EOT, N_VOCAB };

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
    for (size_t i = 0; i < N_VOCAB; i++)
        put_str(o, VOCAB[i]);

    put_str(o, "tokenizer.ggml.token_type");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_I32);
    put_u64(o, N_VOCAB);
    for (size_t i = 0; i < N_VOCAB; i++)
        put_u32(o, i == ID_EOT ? 3 : 1); /* CONTROL : NORMAL */

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

struct encode_case {
    const char *pre, *text;
    size_t      n;
    int32_t     want[8];
};

static const struct encode_case ENCODE_CASES[] = {
        {"llama-bpe", "2000", 2, {ID_200, ID_0}},
        {"llama3", "2000", 2, {ID_200, ID_0}},
        {"falcon3", "2000", 2, {ID_200, ID_0}},
        {"qwen2", "2000", 4, {ID_2, ID_0, ID_0, ID_0}},
        {"llama-bpe", "ab", 1, {ID_AB}},
        {"qwen2", "ab", 2, {ID_A, ID_B}},
        {"llama-bpe", "a  <|eot_id|>", 3, {ID_A, ID_SP2, ID_EOT}},
        {"llama-bpe", "a  ", 2, {ID_A, ID_SP2}},
};

static int check_encode(const struct encode_case *c, bool *error) {
    struct blob o;
    build(&o, c->pre);
    if (o.overflow) {
        fprintf(stderr, "blob buffer too small\n");
        *error = true;
        return 0;
    }
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open_memory(o.b, o.n, &err);
    if (ctx == nullptr) {
        fprintf(stderr, "gguf_open_memory: %s\n", err != nullptr ? err : "?");
        *error = true;
        return 0;
    }
    struct gguf_tokenizer tok;
    const bool            loaded = gguf_tokenizer_load_copy(&tok, ctx);
    gguf_close(ctx);
    if (!loaded) {
        fprintf(stderr, "%s: tokenizer load failed\n", c->pre);
        *error = true;
        return 0;
    }
    int32_t           ids[16];
    size_t            n  = 0;
    enum geist_status st = gguf_tokenizer_encode(&tok, c->text, 16, ids, &n);
    gguf_tokenizer_unload(&tok);
    bool ok = st == GEIST_OK && n == c->n;
    for (size_t k = 0; ok && k < n; k++)
        ok = ids[k] == c->want[k];
    if (!ok) {
        fprintf(stderr, "FAIL: pre %s '%s':", c->pre, c->text);
        for (size_t k = 0; k < n; k++)
            fprintf(stderr, " %d", ids[k]);
        fprintf(stderr, " (status %d), want", (int) st);
        for (size_t k = 0; k < c->n; k++)
            fprintf(stderr, " %d", c->want[k]);
        fprintf(stderr, "\n");
    }
    return !ok;
}

int main(void) {
    int  fails = 0;
    bool error = false;
    for (size_t c = 0; c < sizeof CHUNK_CASES / sizeof CHUNK_CASES[0]; c++)
        fails += check_chunks(&CHUNK_CASES[c]);
    for (size_t c = 0; c < sizeof ENCODE_CASES / sizeof ENCODE_CASES[0]; c++)
        fails += check_encode(&ENCODE_CASES[c], &error);
    if (error)
        return GEIST_TEST_ERROR;
    printf("llama3 pretokenizer: %zu chunk cases, %zu encodes, %d failed\n",
           sizeof CHUNK_CASES / sizeof CHUNK_CASES[0],
           sizeof ENCODE_CASES / sizeof ENCODE_CASES[0],
           fails);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
