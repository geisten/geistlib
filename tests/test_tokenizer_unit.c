/*
 * test_tokenizer — encodes TEST_STRINGS with sp_bpe_tokenizer and prints the
 * ids in a diff-friendly form, for a manual parity check against a reference
 * tokenizer (e.g. HuggingFace's on the same strings). Fails only if an
 * encode fails; skips without a tokenizer.bin argument.
 *
 * Output format (per input, stdout):
 *   INPUT: <hex-encoded-utf8>
 *     ids:        id1,id2,...
 *
 * Run:
 *   ./test_tokenizer ../gemma-4-E2B-it/tokenizer.bin
 */
#include "sp_bpe_tokenizer.h"
#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TEST_STRINGS[] = {
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "a",
        "",
        " ",
        "  leading double space",
        "trailing space ",
        "multiple    spaces between",
        "123 4567 89",
        "x = 42 + 3.14",
        "for (int i = 0; i < 10; i++) { sum += arr[i]; }",
        "def f(x): return x ** 2",
        "Hallo Welt — wie geht es dir?",
        "中文测试: 你好世界",
        "日本語のテスト",
        "Привет мир",
        "🎉 Emoji works! 🚀✨",
        "\n\nDouble newline\n",
        "tab\there",
        "Mixed\nline\rbreaks\r\n",
        "quote\"with'apostrophes",
        "<bos>",
        "<|image|>",
        "Just text with <|audio|> in middle",
        "This is a longer paragraph with multiple sentences. "
        "This is a longer paragraph with multiple sentences. "
        "This is a longer paragraph with multiple sentences. "
        "This is a longer paragraph with multiple sentences. "
        "This is a longer paragraph with multiple sentences. ",
        "\xe2\x96\x81",
        "word\xe2\x96\x81with\xe2\x96\x81markers",
        "  multiple  \xe2\x96\x81  marker  mix",
};
static const size_t TEST_COUNT = sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]);

static void print_hex(const char *s) {
    const unsigned char *u = (const unsigned char *) s;
    while (*u)
        printf("%02x", *u++);
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 2, "<tokenizer.bin>");

    struct sp_bpe_tokenizer *tok = nullptr;
    if (!sp_bpe_tokenizer_load(&tok, argv[1])) {
        fprintf(stderr, "sp_bpe_tokenizer_load failed for %s\n", argv[1]);
        return 1;
    }
    fprintf(stderr,
            "# loaded vocab_size=%u, specials=%u, "
            "bos=%u eos=%u pad=%u unk=%u\n",
            sp_bpe_tokenizer_vocab_size(tok),
            sp_bpe_tokenizer_specials_count(tok),
            sp_bpe_tokenizer_bos_id(tok),
            sp_bpe_tokenizer_eos_id(tok),
            sp_bpe_tokenizer_pad_id(tok),
            sp_bpe_tokenizer_unk_id(tok));

    int rc = 0;
    for (size_t i = 0; i < TEST_COUNT; i++) {
        const char *s = TEST_STRINGS[i];
        printf("INPUT: ");
        print_hex(s);
        printf("\n");

        uint32_t *ids = nullptr;
        size_t    n   = 0;
        if (!sp_bpe_tokenizer_encode(tok, s, &ids, &n)) {
            printf("  ids:        ENCODE_FAILED\n\n");
            rc = 1;
            continue;
        }
        printf("  ids:        ");
        for (size_t j = 0; j < n; j++) {
            if (j > 0)
                printf(",");
            printf("%u", ids[j]);
        }
        printf("\n\n");
        free(ids);
    }

    sp_bpe_tokenizer_free(tok);
    return rc;
}
