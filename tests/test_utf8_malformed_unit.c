/*
 * test_utf8_malformed_unit — malformed UTF-8 read the way llama.cpp reads it
 * (#427).
 *
 * llama.cpp's unicode_cpts_from_utf8 (unicode.cpp at 8642541b2, the
 * llama-tokenize 0.8.0 build) turns a byte that starts no valid sequence —
 * a stray continuation byte, a truncated sequence, an overlong form, a lead
 * past U+10FFFF, 0xF8..0xFF — into U+FFFD and moves on by one byte; a
 * surrogate's three bytes pass. Its byte-level BPE pretokenizers classify
 * those codepoints (U+FFFD is a symbol) and rebuild their words from them,
 * so the merges see U+FFFD's three bytes where the text had the bad one.
 *
 * 1. gguf_tokenizer_utf8_cp walks each text into llama.cpp's codepoints:
 *    REBUILT is the text llama.cpp rebuilds (malformed byte -> U+FFFD).
 * 2. Every pretokenizer scanner that reads codepoints — gpt-2, smollm,
 *    llama3, qwen2, qwen35, gpt-4o, deepseek-llm, deepseek-v3 — splits each
 *    text into PIECES_*: the words of llama.cpp's unicode_regex_split with
 *    that pretokenizer's regex list (llama-vocab.cpp at 8642541b2), mapped
 *    back to the bytes of the text, separated by '\x01'. They were produced
 *    by a harness compiled from that unicode.cpp, not worked out by hand.
 * 3. gguf_tokenizer_encode on a synthetic vocabulary of the 256 GPT-2 byte
 *    symbols and no merges, for each of those pretokenizers and the simple
 *    splitter ("deepseek-coder"): one id per byte of REBUILT. The same for a
 *    "gemma4" vocabulary (raw UTF-8, one token per codepoint): one id per
 *    codepoint of REBUILT with spaces as U+2581.
 * 4. SentencePiece ("llama"; #454): llama.cpp's llm_tokenizer_spm_session
 *    does not decode at all. After llama_escape_whitespace (' ' -> U+2581)
 *    it cuts the text into symbols by unicode_len_utf8 — the lead byte's
 *    high nibble alone: 0x0..0xB 1 byte (a stray continuation byte too),
 *    0xC..0xD 2, 0xE 3, 0xF 4 (0xF8..0xFF too), clipped to the end, the
 *    continuation bytes never checked — so a malformed sequence swallows
 *    the bytes after it, U+2581's included; a symbol that forms no token
 *    is written as its <0xXX> byte tokens. SYMBOLS_SPM is that split, from
 *    the same harness. gguf_tokenizer_encode on a vocabulary of the byte
 *    tokens, the ASCII characters, U+2581 and every symbol that is one
 *    valid codepoint, with no pair to merge, as unigram (scores) and as
 *    SPM (a merge nothing uses): one id per symbol, or its bytes.
 * On the real vocabularies (ggml-vocab-*.gguf of llama.cpp, gpt-oss-20b,
 * Llama 4 Scout, DeepSeek-V2-Lite, Gemma 3 270M) the ids of llama-tokenize
 * 0.8.0 and of gguf_tokenizer_encode agree on these texts for every one of
 * these pretokenizers, gemma4 and the SentencePiece vocabularies llama-spm,
 * phi-3, baichuan and Gemma 3 (#427, #454 PRs).
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "gguf_tokenizer.h"
#include "tokenizer_fixtures.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_TEXTS = 17 };

static const char *const TEXTS[N_TEXTS] = {
        "a\xFF"
        "b \xFF\xFE!",
        "Hello\x80world",
        "x \xC3(y",
        "\xE2\x82 euro",
        "\xF0\x9F\x98 smile",
        "abc\xF8\x88"
        "def",
        "123\xFF"
        "4567",
        "  \xFF  x",
        "it\xFF's \xFFOK",
        "\xC0\xAF slash \xE0\x80\x80",
        "\xF4\x90\x80\x80 big",
        "\xED\xA0\x80 surr",
        "end\xE4",
        "\xFF\n\n\xFE",
        "\xFF"
        "abc \xFF"
        "123",
        "CJK\xE6\x97\xA5\xFF\xE6\x9C\xAC",
        ".\xFF.",
};
static const char *const REBUILT[N_TEXTS] = {
        "a\xEF\xBF\xBD"
        "b \xEF\xBF\xBD\xEF\xBF\xBD!",
        "Hello\xEF\xBF\xBDworld",
        "x \xEF\xBF\xBD(y",
        "\xEF\xBF\xBD\xEF\xBF\xBD euro",
        "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD smile",
        "abc\xEF\xBF\xBD\xEF\xBF\xBD"
        "def",
        "123\xEF\xBF\xBD"
        "4567",
        "  \xEF\xBF\xBD  x",
        "it\xEF\xBF\xBD's \xEF\xBF\xBDOK",
        "\xEF\xBF\xBD\xEF\xBF\xBD slash \xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD",
        "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD big",
        "\xED\xA0\x80 surr",
        "end\xEF\xBF\xBD",
        "\xEF\xBF\xBD\n\n\xEF\xBF\xBD",
        "\xEF\xBF\xBD"
        "abc \xEF\xBF\xBD"
        "123",
        "CJK\xE6\x97\xA5\xEF\xBF\xBD\xE6\x9C\xAC",
        ".\xEF\xBF\xBD.",
};
static const char *const PIECES_GPT2[N_TEXTS] = {
        "a\x01\xFF\x01"
        "b\x01 \xFF\xFE!",
        "Hello\x01\x80\x01world",
        "x\x01 \xC3(\x01y",
        "\xE2\x82\x01 euro",
        "\xF0\x9F\x98\x01 smile",
        "abc\x01\xF8\x88\x01"
        "def",
        "123\x01\xFF\x01"
        "4567",
        " \x01 \xFF\x01 \x01 x",
        "it\x01\xFF'\x01s\x01 \xFF\x01OK",
        "\xC0\xAF\x01 slash\x01 \xE0\x80\x80",
        "\xF4\x90\x80\x80\x01 big",
        "\xED\xA0\x80\x01 surr",
        "end\x01\xE4",
        "\xFF\x01\n\x01\n\x01\xFE",
        "\xFF\x01"
        "abc\x01 \xFF\x01"
        "123",
        "CJK\xE6\x97\xA5\x01\xFF\x01\xE6\x9C\xAC",
        ".\xFF.",
};
static const char *const PIECES_SMOLLM[N_TEXTS] = {
        "a\x01\xFF\x01"
        "b\x01 \xFF\xFE!",
        "Hello\x01\x80\x01world",
        "x\x01 \xC3(\x01y",
        "\xE2\x82\x01 euro",
        "\xF0\x9F\x98\x01 smile",
        "abc\x01\xF8\x88\x01"
        "def",
        "1\x01"
        "2\x01"
        "3\x01\xFF\x01"
        "4\x01"
        "5\x01"
        "6\x01"
        "7",
        " \x01 \xFF\x01 \x01 x",
        "it\x01\xFF'\x01s\x01 \xFF\x01OK",
        "\xC0\xAF\x01 slash\x01 \xE0\x80\x80",
        "\xF4\x90\x80\x80\x01 big",
        "\xED\xA0\x80\x01 surr",
        "end\x01\xE4",
        "\xFF\x01\n\x01\n\x01\xFE",
        "\xFF\x01"
        "abc\x01 \xFF\x01"
        "1\x01"
        "2\x01"
        "3",
        "CJK\xE6\x97\xA5\x01\xFF\x01\xE6\x9C\xAC",
        ".\xFF.",
};
static const char *const PIECES_LLAMA3[N_TEXTS] = {
        "a\x01\xFF"
        "b\x01 \xFF\xFE!",
        "Hello\x01\x80world",
        "x\x01 \xC3(\x01y",
        "\xE2\x82\x01 euro",
        "\xF0\x9F\x98\x01 smile",
        "abc\x01\xF8\x88\x01"
        "def",
        "123\x01\xFF\x01"
        "456\x01"
        "7",
        " \x01 \xFF\x01 \x01 x",
        "it\x01\xFF'\x01s\x01 \xFF\x01OK",
        "\xC0\xAF\x01 slash\x01 \xE0\x80\x80",
        "\xF4\x90\x80\x80\x01 big",
        "\xED\xA0\x80\x01 surr",
        "end\x01\xE4",
        "\xFF\n\n\x01\xFE",
        "\xFF"
        "abc\x01 \xFF\x01"
        "123",
        "CJK\xE6\x97\xA5\x01\xFF\xE6\x9C\xAC",
        ".\xFF.",
};
static const char *const PIECES_QWEN2[N_TEXTS] = {
        "a\x01\xFF"
        "b\x01 \xFF\xFE!",
        "Hello\x01\x80world",
        "x\x01 \xC3(\x01y",
        "\xE2\x82\x01 euro",
        "\xF0\x9F\x98\x01 smile",
        "abc\x01\xF8\x88\x01"
        "def",
        "1\x01"
        "2\x01"
        "3\x01\xFF\x01"
        "4\x01"
        "5\x01"
        "6\x01"
        "7",
        " \x01 \xFF\x01 \x01 x",
        "it\x01\xFF'\x01s\x01 \xFF\x01OK",
        "\xC0\xAF\x01 slash\x01 \xE0\x80\x80",
        "\xF4\x90\x80\x80\x01 big",
        "\xED\xA0\x80\x01 surr",
        "end\x01\xE4",
        "\xFF\n\n\x01\xFE",
        "\xFF"
        "abc\x01 \xFF\x01"
        "1\x01"
        "2\x01"
        "3",
        "CJK\xE6\x97\xA5\x01\xFF\xE6\x9C\xAC",
        ".\xFF.",
};
static const char *const PIECES_QWEN35[N_TEXTS] = {
        "a\x01\xFF"
        "b\x01 \xFF\xFE!",
        "Hello\x01\x80world",
        "x\x01 \xC3(\x01y",
        "\xE2\x82\x01 euro",
        "\xF0\x9F\x98\x01 smile",
        "abc\x01\xF8\x88\x01"
        "def",
        "1\x01"
        "2\x01"
        "3\x01\xFF\x01"
        "4\x01"
        "5\x01"
        "6\x01"
        "7",
        " \x01 \xFF\x01 \x01 x",
        "it\x01\xFF'\x01s\x01 \xFF\x01OK",
        "\xC0\xAF\x01 slash\x01 \xE0\x80\x80",
        "\xF4\x90\x80\x80\x01 big",
        "\xED\xA0\x80\x01 surr",
        "end\x01\xE4",
        "\xFF\n\n\x01\xFE",
        "\xFF"
        "abc\x01 \xFF\x01"
        "1\x01"
        "2\x01"
        "3",
        "CJK\xE6\x97\xA5\x01\xFF\xE6\x9C\xAC",
        ".\xFF.",
};
static const char *const PIECES_GPT4O[N_TEXTS] = {
        "a\x01\xFF"
        "b\x01 \xFF\xFE!",
        "Hello\x01\x80world",
        "x\x01 \xC3(\x01y",
        "\xE2\x82\x01 euro",
        "\xF0\x9F\x98\x01 smile",
        "abc\x01\xF8\x88\x01"
        "def",
        "123\x01\xFF\x01"
        "456\x01"
        "7",
        " \x01 \xFF\x01 \x01 x",
        "it\x01\xFF'\x01s\x01 \xFF\x01OK",
        "\xC0\xAF\x01 slash\x01 \xE0\x80\x80",
        "\xF4\x90\x80\x80\x01 big",
        "\xED\xA0\x80\x01 surr",
        "end\x01\xE4",
        "\xFF\n\n\x01\xFE",
        "\xFF"
        "abc\x01 \xFF\x01"
        "123",
        "CJK\xE6\x97\xA5\x01\xFF\xE6\x9C\xAC",
        ".\xFF.",
};
static const char *const PIECES_DSLLM[N_TEXTS] = {
        "a\x01\xFF\x01"
        "b\x01 \xFF\xFE\x01!",
        "Hello\x01\x80\x01world",
        "x\x01 \xC3\x01(\x01y",
        "\xE2\x82\x01 euro",
        "\xF0\x9F\x98\x01 smile",
        "abc\x01\xF8\x88\x01"
        "def",
        "123\x01\xFF\x01"
        "4567",
        "  \xFF\x01 \x01 x",
        "it\x01\xFF\x01'\x01s\x01 \xFF\x01OK",
        "\xC0\xAF\x01 slash\x01 \xE0\x80\x80",
        "\xF4\x90\x80\x80\x01 big",
        "\xED\xA0\x80\x01 surr",
        "end\x01\xE4",
        "\xFF\x01\n\x01\n\x01\xFE",
        "\xFF\x01"
        "abc\x01 \xFF\x01"
        "123",
        "CJK\x01\xE6\x97\xA5\x01\xFF\x01\xE6\x9C\xAC",
        ".\x01\xFF\x01.",
};
static const char *const PIECES_DSV3[N_TEXTS] = {
        "a\x01\xFF\x01"
        "b\x01 \xFF\xFE!",
        "Hello\x01\x80\x01world",
        "x\x01 \xC3(\x01y",
        "\xE2\x82\x01 euro",
        "\xF0\x9F\x98\x01 smile",
        "abc\x01\xF8\x88\x01"
        "def",
        "123\x01\xFF\x01"
        "456\x01"
        "7",
        " \x01 \xFF\x01 \x01 x",
        "it\x01\xFF'\x01s\x01 \xFF\x01OK",
        "\xC0\xAF\x01 slash\x01 \xE0\x80\x80",
        "\xF4\x90\x80\x80\x01 big",
        "\xED\xA0\x80\x01 surr",
        "end\x01\xE4",
        "\xFF\n\n\x01\xFE",
        "\xFF\x01"
        "abc\x01 \xFF\x01"
        "123",
        "CJK\x01\xE6\x97\xA5\x01\xFF\x01\xE6\x9C\xAC",
        ".\xFF.",
};

static const char *const SYMBOLS_SPM[N_TEXTS] = {
        "a\x01\xFF"
        "b\xE2\x96\x01\x81\x01\xFF\xFE!",
        "H\x01"
        "e\x01l\x01l\x01o\x01\x80\x01w\x01o\x01r\x01l\x01"
        "d",
        "x\x01\xE2\x96\x81\x01\xC3(\x01y",
        "\xE2\x82\xE2\x01\x96\x01\x81\x01"
        "e\x01u\x01r\x01o",
        "\xF0\x9F\x98\xE2\x01\x96\x01\x81\x01s\x01m\x01i\x01l\x01"
        "e",
        "a\x01"
        "b\x01"
        "c\x01\xF8\x88"
        "de\x01"
        "f",
        "1\x01"
        "2\x01"
        "3\x01\xFF"
        "456\x01"
        "7",
        "\xE2\x96\x81\x01\xE2\x96\x81\x01\xFF\xE2\x96\x81\x01\xE2\x96\x81\x01x",
        "i\x01t\x01\xFF's\xE2\x01\x96\x01\x81\x01\xFFOK",
        "\xC0\xAF\x01\xE2\x96\x81\x01s\x01l\x01"
        "a\x01s\x01h\x01\xE2\x96\x81\x01\xE0\x80\x80",
        "\xF4\x90\x80\x80\x01\xE2\x96\x81\x01"
        "b\x01i\x01g",
        "\xED\xA0\x80\x01\xE2\x96\x81\x01s\x01u\x01r\x01r",
        "e\x01n\x01"
        "d\x01\xE4",
        "\xFF\n\n\xFE",
        "\xFF"
        "abc\x01\xE2\x96\x81\x01\xFF"
        "123",
        "C\x01J\x01K\x01\xE6\x97\xA5\x01\xFF\xE6\x9C\xAC",
        ".\x01\xFF.",
};

/* Valid UTF-8 (REBUILT) to codepoints with their lengths; returns the count. */
static size_t decode_valid(const char *s, uint32_t cps[static 64], size_t lens[static 64]) {
    size_t k = 0;
    for (size_t i = 0; s[i] != '\0' && k < 64; k++) {
        const unsigned char c   = (unsigned char) s[i];
        const size_t        len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        uint32_t            cp  = len == 1 ? c : c & (0x3Fu >> (len - 1));
        for (size_t j = 1; j < len; j++)
            cp = (cp << 6) | ((unsigned char) s[i + j] & 0x3Fu);
        cps[k]  = cp;
        lens[k] = len;
        i += len;
    }
    return k;
}

/* 1. The decoder against llama.cpp's codepoints. None of the texts holds a
 * real U+FFFD, so each one in REBUILT stands for one byte of the text. */
static int check_decode(size_t t) {
    const char  *text = TEXTS[t];
    const size_t tlen = strlen(text);
    uint32_t     cps[64];
    size_t       lens[64];
    const size_t n = decode_valid(REBUILT[t], cps, lens);
    size_t       i = 0, k = 0;
    for (; i < tlen && k < n; k++) {
        size_t         adv;
        const uint32_t cp   = gguf_tokenizer_utf8_cp(tlen, i, text, &adv);
        const size_t   want = cps[k] == 0xFFFD ? 1 : lens[k];
        if (cp != cps[k] || adv != want) {
            fprintf(stderr,
                    "  text %zu byte %zu: U+%04X over %zu bytes, want U+%04X over %zu\n",
                    t,
                    i,
                    cp,
                    adv,
                    cps[k],
                    want);
            return 1;
        }
        i += adv;
    }
    return geist_expect(i == tlen && k == n, "the codepoints cover the text");
}

enum scanner { SC_GPT2, SC_SMOLLM, SC_LLAMA3, SC_QWEN2, SC_QWEN35, SC_GPT4O, SC_DSLLM, SC_DSV3 };

struct family {
    const char        *pre;
    enum scanner       sc;
    const char *const *pieces;
};

static const struct family FAMILIES[] = {
        {"gpt-2", SC_GPT2, PIECES_GPT2},
        {"smollm", SC_SMOLLM, PIECES_SMOLLM},
        {"llama-bpe", SC_LLAMA3, PIECES_LLAMA3},
        {"qwen2", SC_QWEN2, PIECES_QWEN2},
        {"qwen35", SC_QWEN35, PIECES_QWEN35},
        {"gpt-4o", SC_GPT4O, PIECES_GPT4O},
        {"deepseek-llm", SC_DSLLM, PIECES_DSLLM},
        {"deepseek-v3", SC_DSV3, PIECES_DSV3},
};
enum { N_FAMILIES = sizeof FAMILIES / sizeof FAMILIES[0] };

/* The byte ends of the pieces the scanner cuts text into; returns the count. */
static size_t
scan(enum scanner sc, size_t tlen, const char text[static tlen], size_t ends[static tlen]) {
    size_t n = 0;
    if (sc == SC_DSLLM || sc == SC_DSV3) {
        uint32_t cps[64];
        size_t   off[65], pieces[128], n_cp = 0;
        for (size_t i = 0; i < tlen && n_cp < 64; n_cp++) {
            size_t adv;
            cps[n_cp] = gguf_tokenizer_utf8_cp(tlen, i, text, &adv);
            off[n_cp] = i;
            i += adv;
        }
        off[n_cp]          = tlen;
        const size_t n_pcs = gguf_tokenizer_seq_split(
                sc == SC_DSV3 ? GGUF_PRE_DEEPSEEK_V3 : GGUF_PRE_DEEPSEEK_LLM, n_cp, cps, pieces);
        size_t at = 0;
        for (size_t p = 0; p < n_pcs; p++) {
            at += pieces[p];
            ends[n++] = off[at];
        }
        return n;
    }
    for (size_t i = 0; i < tlen; n++) {
        switch (sc) {
        case SC_GPT2:
        case SC_SMOLLM:
            i = gguf_tokenizer_gpt2_chunk_end(tlen, i, sc == SC_SMOLLM, text);
            break;
        case SC_LLAMA3:
            i = gguf_tokenizer_llama3_chunk_end(tlen, i, 3, false, text);
            break;
        case SC_QWEN2:
        case SC_QWEN35:
            i = gguf_tokenizer_llama3_chunk_end(tlen, i, 1, sc == SC_QWEN35, text);
            break;
        case SC_GPT4O:
            i = gguf_tokenizer_gpt4o_chunk_end(tlen, i, text);
            break;
        case SC_DSLLM:
        case SC_DSV3:
            break;
        }
        ends[n] = i;
    }
    return n;
}

/* 2. The scanner's pieces against llama.cpp's. */
static int check_split(const struct family *f, size_t t) {
    const char  *text = TEXTS[t];
    const size_t tlen = strlen(text);
    size_t       want[64], n_want = 0, at = 0;
    for (const char *p = f->pieces[t];;) {
        const char *sep = strchr(p, '\x01');
        at += sep != nullptr ? (size_t) (sep - p) : strlen(p);
        want[n_want++] = at;
        if (sep == nullptr)
            break;
        p = sep + 1;
    }
    size_t       got[64];
    const size_t n  = scan(f->sc, tlen, text, got);
    bool         ok = n == n_want && at == tlen;
    for (size_t k = 0; ok && k < n; k++)
        ok = got[k] == want[k];
    if (!ok) {
        fprintf(stderr, "  %s text %zu: piece ends", f->pre, t);
        for (size_t k = 0; k < n; k++)
            fprintf(stderr, " %zu", got[k]);
        fprintf(stderr, ", want");
        for (size_t k = 0; k < n_want; k++)
            fprintf(stderr, " %zu", want[k]);
        fprintf(stderr, "\n");
    }
    return !ok;
}

/* ---- 3. encode on synthetic vocabularies --------------------------------- */

/* GPT-2's bytes_to_unicode: the codepoint byte b stands for. */
static uint32_t gpt2_cp(uint32_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)
        return b;
    return b <= 32 ? 256 + b : b == 173 ? 323 : 289 + (b - 127);
}

static size_t put_utf8(uint32_t cp, char out[static 4]) {
    if (cp < 0x80) {
        out[0] = (char) cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char) (0xC0 | (cp >> 6));
        out[1] = (char) (0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char) (0xE0 | (cp >> 12));
        out[1] = (char) (0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char) (0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char) (0xF0 | (cp >> 18));
    out[1] = (char) (0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char) (0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char) (0x80 | (cp & 0x3F));
    return 4;
}

/* A tokenizer-only GGUF: model, pre (unless null), the tokens with their
 * scores, the merges ("l r" strings) and, for gemma4 and llama,
 * add_space_prefix = false. */
static struct tf_buf make_gguf(const char *model, const char *pre, const struct tf_vocab *v) {
    const bool    gemma4 = strcmp(model, "gemma4") == 0 || strcmp(model, "llama") == 0;
    struct tf_buf o      = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, 1); /* tensors: the reader wants at least one */
    tf_u64(&o, 4 + (pre != nullptr) + gemma4);
    tf_gstr(&o, "tokenizer.ggml.model", 20);
    tf_u32(&o, TF_GGUF_STRING);
    tf_gstr(&o, model, strlen(model));
    if (pre != nullptr) {
        tf_gstr(&o, "tokenizer.ggml.pre", 18);
        tf_u32(&o, TF_GGUF_STRING);
        tf_gstr(&o, pre, strlen(pre));
    }
    tf_gstr(&o, "tokenizer.ggml.tokens", 21);
    tf_u32(&o, TF_GGUF_ARRAY);
    tf_u32(&o, TF_GGUF_STRING);
    tf_u64(&o, v->n_tok);
    for (size_t i = 0; i < v->n_tok; i++)
        tf_gstr(&o, v->tok[i], strlen(v->tok[i]));
    tf_gstr(&o, "tokenizer.ggml.scores", 21);
    tf_u32(&o, TF_GGUF_ARRAY);
    tf_u32(&o, TF_GGUF_FLOAT32);
    tf_u64(&o, v->n_tok);
    for (size_t i = 0; i < v->n_tok; i++)
        tf_f32(&o, v->score[i]);
    tf_gstr(&o, "tokenizer.ggml.merges", 21);
    tf_u32(&o, TF_GGUF_ARRAY);
    tf_u32(&o, TF_GGUF_STRING);
    tf_u64(&o, v->n_merge);
    for (size_t m = 0; m < v->n_merge; m++) {
        char         s[64];
        const size_t n = (size_t) snprintf(s, sizeof s, "%s %s", v->merge_l[m], v->merge_r[m]);
        tf_gstr(&o, s, n);
    }
    if (gemma4) {
        tf_gstr(&o, "tokenizer.ggml.add_space_prefix", 31);
        tf_u32(&o, 7); /* bool */
        tf_put(&o, "", 1);
    }
    tf_gstr(&o, "t", 1); /* one dummy 1-element f32 tensor */
    tf_u32(&o, 1);
    tf_u64(&o, 1);
    tf_u32(&o, 0);
    tf_u64(&o, 0);
    while (o.n % 32 != 0)
        tf_put(&o, "", 1);
    tf_f32(&o, 0.0f);
    return o;
}

static bool load(struct gguf_tokenizer *tok, const struct tf_buf *g) {
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open_memory(g->b, g->n, &err);
    if (ctx == nullptr)
        return false;
    const bool ok = gguf_tokenizer_load_copy(tok, ctx);
    gguf_close(ctx);
    return ok;
}

static int compare_ids(const char   *what,
                       size_t        t,
                       size_t        n,
                       const int32_t got[static n],
                       size_t        n_want,
                       const int32_t want[static n_want]) {
    bool ok = n == n_want;
    for (size_t k = 0; ok && k < n; k++)
        ok = got[k] == want[k];
    if (ok)
        return 0;
    fprintf(stderr, "  %s text %zu: ids", what, t);
    for (size_t k = 0; k < n; k++)
        fprintf(stderr, " %d", got[k]);
    fprintf(stderr, ", want");
    for (size_t k = 0; k < n_want; k++)
        fprintf(stderr, " %d", want[k]);
    fprintf(stderr, "\n");
    return 1;
}

/* Byte-level BPE: token b is byte b's GPT-2 symbol, so with no merges the
 * ids are the bytes of the text the merges see — REBUILT. */
static int check_encode_gpt2(const char *pre) {
    struct tf_vocab v = {0};
    for (uint32_t b = 0; b < 256; b++) {
        char s[4];
        tf_add_tok(&v, s, put_utf8(gpt2_cp(b), s), 0.0f);
    }
    struct tf_buf         g = make_gguf("gpt2", pre, &v);
    struct gguf_tokenizer tok;
    int                   fails = geist_expect(load(&tok, &g), "gpt2 vocabulary loads");
    for (size_t t = 0; fails == 0 && t < N_TEXTS; t++) {
        int32_t ids[64], want[64];
        size_t  n = 0, n_want = 0;
        fails += geist_expect(gguf_tokenizer_encode(&tok, TEXTS[t], 64, ids, &n) == GEIST_OK,
                              "encode succeeds");
        for (const char *p = REBUILT[t]; *p != '\0'; p++)
            want[n_want++] = (unsigned char) *p;
        fails += compare_ids(pre, t, n, ids, n_want, want);
    }
    if (fails == 0)
        gguf_tokenizer_unload(&tok);
    free(g.b);
    tf_free_vocab(&v);
    return fails;
}

/* gemma4: raw UTF-8 with spaces as U+2581, one token per codepoint of
 * REBUILT (and the <0xXX> byte tokens the old encoder fell back to); one
 * merge nothing here can use, so the file is a merge vocabulary. */
static int check_encode_gemma4(void) {
    static const char MARK[] = "\xE2\x96\x81";
    struct tf_vocab   v      = {0};
    for (int b = 0; b < 256; b++) {
        char s[8];
        snprintf(s, sizeof s, "<0x%02X>", b);
        tf_add_tok(&v, s, 6, 0.0f);
    }
    tf_add_tok(&v, "Q", 1, 0.0f);
    tf_add_tok(&v, "QQ", 2, 0.0f);
    tf_add_merge(&v, "Q", 1, "Q", 1);
    for (size_t t = 0; t < N_TEXTS; t++) {
        uint32_t cps[64];
        size_t   lens[64];
        size_t   off = 0;
        for (size_t k = 0, n = decode_valid(REBUILT[t], cps, lens); k < n; off += lens[k++]) {
            if (cps[k] == ' ')
                tf_add_tok(&v, MARK, 3, 0.0f);
            else
                tf_add_tok(&v, REBUILT[t] + off, lens[k], 0.0f);
        }
    }
    struct tf_buf         g = make_gguf("gemma4", nullptr, &v);
    struct gguf_tokenizer tok;
    int                   fails = geist_expect(load(&tok, &g), "gemma4 vocabulary loads");
    for (size_t t = 0; fails == 0 && t < N_TEXTS; t++) {
        int32_t  ids[64], want[64];
        size_t   n = 0, n_want = 0, off = 0, lens[64];
        uint32_t cps[64];
        fails += geist_expect(gguf_tokenizer_encode(&tok, TEXTS[t], 64, ids, &n) == GEIST_OK,
                              "encode succeeds");
        for (size_t k = 0, nc = decode_valid(REBUILT[t], cps, lens); k < nc; off += lens[k++]) {
            const char  *s   = cps[k] == ' ' ? MARK : REBUILT[t] + off;
            const size_t len = cps[k] == ' ' ? 3 : lens[k];
            for (size_t id = 0; id < v.n_tok; id++)
                if (strlen(v.tok[id]) == len && memcmp(v.tok[id], s, len) == 0)
                    want[n_want++] = (int32_t) id;
        }
        fails += compare_ids("gemma4", t, n, ids, n_want, want);
    }
    if (fails == 0)
        gguf_tokenizer_unload(&tok);
    free(g.b);
    tf_free_vocab(&v);
    return fails;
}

/* SentencePiece: the ids of llama.cpp's symbols (SYMBOLS_SPM) on a
 * vocabulary in which no two symbols merge — unigram, or with merges an SPM
 * vocabulary whose only merge nothing uses. */
static int check_encode_spm(bool with_merges) {
    static const char MARK[] = "\xE2\x96\x81";
    struct tf_vocab   v      = {0};
    for (int b = 0; b < 256; b++) {
        char s[8];
        snprintf(s, sizeof s, "<0x%02X>", b);
        tf_add_tok(&v, s, 6, -100.0f);
    }
    for (char c = '!'; c <= '~'; c++)
        tf_add_tok(&v, &c, 1, -1.0f);
    tf_add_tok(&v, "\n", 1, -1.0f);
    tf_add_tok(&v, MARK, 3, -1.0f);
    for (size_t t = 0; t < N_TEXTS; t++) {
        for (const char *p = SYMBOLS_SPM[t];;) {
            const char    *sep = strchr(p, '\x01');
            const size_t   len = sep != nullptr ? (size_t) (sep - p) : strlen(p);
            size_t         adv;
            const uint32_t cp = gguf_tokenizer_utf8_cp(len, 0, p, &adv);
            if (adv == len && !(cp == 0xFFFD && adv == 1))
                tf_add_tok(&v, p, len, -1.0f); /* one valid codepoint */
            if (sep == nullptr)
                break;
            p = sep + 1;
        }
    }
    if (with_merges) {
        tf_add_tok(&v, "QQ", 2, -1.0f);
        tf_add_merge(&v, "Q", 1, "Q", 1);
    }
    const char           *what = with_merges ? "spm" : "unigram";
    struct tf_buf         g    = make_gguf("llama", nullptr, &v);
    struct gguf_tokenizer tok;
    int                   fails = geist_expect(load(&tok, &g), "llama vocabulary loads");
    for (size_t t = 0; fails == 0 && t < N_TEXTS; t++) {
        int32_t ids[64], want[64];
        size_t  n = 0, n_want = 0;
        fails += geist_expect(gguf_tokenizer_encode(&tok, TEXTS[t], 64, ids, &n) == GEIST_OK,
                              "encode succeeds");
        for (const char *p = SYMBOLS_SPM[t];;) {
            const char  *sep = strchr(p, '\x01');
            const size_t len = sep != nullptr ? (size_t) (sep - p) : strlen(p);
            int32_t      id  = -1;
            for (size_t i = 0; id < 0 && i < v.n_tok; i++)
                if (strlen(v.tok[i]) == len && memcmp(v.tok[i], p, len) == 0)
                    id = (int32_t) i;
            if (id >= 0)
                want[n_want++] = id;
            else /* no token: its bytes, <0xXX> = id XX */
                for (size_t b = 0; b < len; b++)
                    want[n_want++] = (unsigned char) p[b];
            if (sep == nullptr)
                break;
            p = sep + 1;
        }
        fails += compare_ids(what, t, n, ids, n_want, want);
    }
    if (fails == 0)
        gguf_tokenizer_unload(&tok);
    free(g.b);
    tf_free_vocab(&v);
    return fails;
}

int main(void) {
    int fails = 0;
    for (size_t t = 0; t < N_TEXTS; t++)
        fails += check_decode(t);
    for (size_t f = 0; f < N_FAMILIES; f++)
        for (size_t t = 0; t < N_TEXTS; t++)
            fails += check_split(&FAMILIES[f], t);
    for (size_t f = 0; f < N_FAMILIES; f++)
        fails += check_encode_gpt2(FAMILIES[f].pre);
    fails += check_encode_gemma4();
    fails += check_encode_gpt2("deepseek-coder"); /* the simple splitter */
    fails += check_encode_spm(false);
    fails += check_encode_spm(true);
    printf("malformed UTF-8 vs llama.cpp: %d texts x %d pretokenizers + gemma4, spm, unigram; "
           "%d failed\n",
           (int) N_TEXTS,
           (int) N_FAMILIES,
           fails);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
