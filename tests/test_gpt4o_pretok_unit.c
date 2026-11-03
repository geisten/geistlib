/*
 * test_gpt4o_pretok_unit — the gpt-4o pretokenizer scanner (#243).
 *
 * gguf_tokenizer_gpt4o_chunk_end splits text as llama.cpp's
 * LLAMA_VOCAB_PRE_TYPE_GPT4O regex does on its collapsed text (GPT-OSS
 * "gpt-4o", Llama 4 "llama4"). Each case lists the chunks, separated by
 * '\x01', worked out by hand from the regex's alternatives in order:
 * case runs ("CamelCase" splits, "HTTPServer" does not), contractions
 * kept on the word, digits in threes, punctuation runs that take a lead
 * space and trailing newlines, whitespace runs that leave their last
 * codepoint to the next word. A non-ASCII letter counts as upper and lower
 * case at once, as in llama.cpp, so "ÉCOLE" splits after É. Parity with
 * llama.cpp on the real vocabulary is checked against llama-tokenize
 * outside the unit suite (#243 PR).
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "gguf_tokenizer.h"

#include <stdio.h>
#include <string.h>

static const char *const CASES[] = {
        "Hello\x01 world",
        "don't\x01 stop",
        "HELLO'S",
        "We'RE\x01 here",
        "Camel\x01"
        "Case",
        "HTTPServer",
        "i\x01Phone",
        "123\x01"
        "456\x01"
        "7",
        "12\x01"
        "abc",
        "a\x01,b",
        "x\x01 =\x01 \x01"
        "1\x01;\n",
        "path\x01/to\x01/file",
        "end\x01.\n\n",
        "a\x01 \x01 b",
        "a\x01 \n\x01 b",
        "trailing\x01   ",
        "\t\x01\tif",
        "\xC3\x89\x01"
        "COLE",                                 /* ÉCOLE */
        "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E", /* 日本語 */
        "na\xC3\xAFve",                         /* naïve */
        "cafe\x01\xCC\x81\x01 x",               /* cafe + U+0301 (a mark) */
        "\xF0\x9F\x98\x80!",                    /* emoji + '!' */
        "x\x01\xC2\xB2",                        /* x² (² is \p{N}) */
        " \x01\xC2\xA0word",                    /* NBSP prefixes a word */
        "!!\x01 ...\x01 \r\n",
        "a\x01 \x01 \x01"
        "42",
};

static int check(const char *spec) {
    char        text[256], want[16][64];
    int         n_want = 0;
    size_t      tl     = 0;
    const char *p      = spec;
    while (*p != '\0') {
        const char  *e = strchr(p, '\x01');
        const size_t n = e != nullptr ? (size_t) (e - p) : strlen(p);
        memcpy(want[n_want], p, n);
        want[n_want++][n] = '\0';
        memcpy(text + tl, p, n);
        tl += n;
        p += n + (e != nullptr);
    }
    text[tl]     = '\0';
    int    fails = 0;
    size_t i     = 0;
    for (int c = 0; c < n_want; c++) {
        const size_t end = i < tl ? gguf_tokenizer_gpt4o_chunk_end(tl, i, text) : i;
        const size_t len = strlen(want[c]);
        if (end != i + len || memcmp(text + i, want[c], len) != 0) {
            fprintf(stderr,
                    "  '%s': chunk %d is '%.*s', want '%s'\n",
                    text,
                    c,
                    (int) (end - i),
                    text + i,
                    want[c]);
            return 1;
        }
        i = end;
    }
    fails += geist_expect(i == tl, "the chunks cover the text");
    return fails;
}

int main(void) {
    int fails = 0;
    for (size_t c = 0; c < sizeof CASES / sizeof CASES[0]; c++)
        fails += check(CASES[c]);
    printf("gpt4o pretokenizer: %zu cases, %d failed\n", sizeof CASES / sizeof CASES[0], fails);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
