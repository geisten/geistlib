/*
 * test_deepseek_pretok_unit — the regex-sequence pretokenizers (#242).
 *
 * gguf_tokenizer_seq_split splits text as llama.cpp's regex lists for
 * tokenizer.ggml.pre "deepseek-llm" (DeepSeek-V2) and "deepseek-v3" do:
 * each regex runs over the pieces the previous one left, matches and the
 * gaps between them becoming pieces. Each case lists the pieces, separated
 * by '\x01', worked out by hand from the regexes in order. deepseek-llm:
 * newlines alone, a letter run from the fixed range list with one leading
 * whitespace, a punctuation run likewise, trailing whitespace, CJK runs,
 * number runs (and what no regex claims stays a piece of its own).
 * deepseek-v3: numbers in threes, kana/CJK runs, then punctuation that
 * takes an ASCII word, letter-or-mark runs with a one-codepoint prefix,
 * punctuation runs with trailing newlines, and the whitespace rules.
 * Parity with llama.cpp on the real vocabularies is checked against
 * llama-tokenize outside the unit suite (#242 PR).
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "gguf_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct pcase {
    enum gguf_pre_seq kind;
    const char       *pieces;
};

static const struct pcase CASES[] = {
        {GGUF_PRE_DEEPSEEK_LLM, "Hello\x01 world"},
        {GGUF_PRE_DEEPSEEK_LLM, "don\x01'\x01t\x01 stop"},
        {GGUF_PRE_DEEPSEEK_LLM, "12345\x01 abc"},
        {GGUF_PRE_DEEPSEEK_LLM,
         "a\x01\n\x01\n\x01"
         "b"},
        {GGUF_PRE_DEEPSEEK_LLM, "x\x01  "},
        {GGUF_PRE_DEEPSEEK_LLM, "hi\x01 \x01 there"},
        {GGUF_PRE_DEEPSEEK_LLM,
         "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x83\x86\xE3\x82\xB9"
         "\xE3\x83\x88"}, /* 日本語テスト */
        {GGUF_PRE_DEEPSEEK_LLM,
         "a\x01,\x01"
         "b\x01!?\x01 c"},
        {GGUF_PRE_DEEPSEEK_LLM, "\xC2\xB5\x01\xC2\xB2"}, /* µ² */
        {GGUF_PRE_DEEPSEEK_LLM,
         "\xE2\x82\xAC\x01"
         "5"}, /* €5 */
        {GGUF_PRE_DEEPSEEK_V3, "Hello\x01 world"},
        {GGUF_PRE_DEEPSEEK_V3,
         "123\x01"
         "456\x01"
         "789"},
        {GGUF_PRE_DEEPSEEK_V3, "don\x01't"},
        {GGUF_PRE_DEEPSEEK_V3, "x\x01.y"},
        {GGUF_PRE_DEEPSEEK_V3, "a\x01 \x01 b"},
        {GGUF_PRE_DEEPSEEK_V3,
         "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\x01"
         "abc"},
        {GGUF_PRE_DEEPSEEK_V3, "hi\x01!!\n\n\x01ok"},
        {GGUF_PRE_DEEPSEEK_V3, " \n\x01 x"},
        {GGUF_PRE_DEEPSEEK_V3, "na\xC3\xAFve\x01 caf\xC3\xA9"},
        {GGUF_PRE_DEEPSEEK_V3, "e\xCC\x81x"}, /* e + U+0301 + x: letters and a mark */
        {GGUF_PRE_DEEPSEEK_V3,
         "12\x01"
         "abc"},
        {GGUF_PRE_DEEPSEEK_V3, "end\x01   "},
};

/* UTF-8 to codepoints (valid input only); returns the count. */
static size_t decode(const char *s, size_t n, uint32_t out[static 256]) {
    size_t k = 0;
    for (size_t i = 0; i < n && k < 256;) {
        const unsigned char c   = (unsigned char) s[i];
        const size_t        len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        uint32_t            cp  = len == 1 ? c : c & (0x3Fu >> (len - 1));
        for (size_t j = 1; j < len; j++)
            cp = (cp << 6) | ((unsigned char) s[i + j] & 0x3Fu);
        out[k++] = cp;
        i += len;
    }
    return k;
}

int main(void) {
    int fails = 0;
    for (size_t c = 0; c < sizeof CASES / sizeof CASES[0]; c++) {
        /* The text, and the expected piece lengths in codepoints. */
        char        text[256];
        size_t      want[64], n_want = 0, tlen = 0;
        const char *p = CASES[c].pieces;
        while (true) {
            const char  *sep = strchr(p, '\x01');
            const size_t len = sep != nullptr ? (size_t) (sep - p) : strlen(p);
            uint32_t     tmp[256];
            want[n_want++] = decode(p, len, tmp);
            memcpy(text + tlen, p, len);
            tlen += len;
            if (sep == nullptr)
                break;
            p = sep + 1;
        }
        uint32_t     cps[256];
        size_t       pieces[512];
        const size_t n   = decode(text, tlen, cps);
        const size_t got = gguf_tokenizer_seq_split(CASES[c].kind, n, cps, pieces);
        bool         ok  = got == n_want;
        for (size_t i = 0; ok && i < got; i++)
            ok = pieces[i] == want[i];
        if (!ok) {
            fprintf(stderr,
                    "FAIL case %zu (%s):",
                    c,
                    CASES[c].kind == GGUF_PRE_DEEPSEEK_V3 ? "deepseek-v3" : "deepseek-llm");
            for (size_t i = 0; i < got; i++)
                fprintf(stderr, " %zu", pieces[i]);
            fprintf(stderr, " | want");
            for (size_t i = 0; i < n_want; i++)
                fprintf(stderr, " %zu", want[i]);
            fprintf(stderr, "\n");
            fails++;
        }
    }
    if (fails) {
        fprintf(stderr, "FAIL: %d case(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: deepseek-llm and deepseek-v3 pieces match the hand-derived splits (%zu cases)\n",
           sizeof CASES / sizeof CASES[0]);
    return GEIST_TEST_PASS;
}
