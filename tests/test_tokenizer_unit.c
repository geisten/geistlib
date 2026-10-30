/*
 * test_tokenizer — encodes TEST_STRINGS with a tokenizer.bin
 * (gguf_tokenizer_load_bin) and prints the
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
#define GEIST_INTERNAL_ENGINE_LAYER

#include "gguf_tokenizer.h"
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

    struct gguf_tokenizer tok;
    if (!gguf_tokenizer_load_bin(&tok, argv[1])) {
        fprintf(stderr, "gguf_tokenizer_load_bin failed for %s\n", argv[1]);
        return 1;
    }
    fprintf(stderr,
            "# loaded vocab_size=%zu, merges=%zu, specials=%zu, bos=%d eos=%d unk=%d\n",
            tok.vocab_size,
            tok.n_merges,
            tok.n_specials,
            tok.bos_id,
            tok.eos_id,
            tok.unk_id);

    int      rc  = 0;
    size_t   cap = 0;
    int32_t *ids = nullptr;
    for (size_t i = 0; i < TEST_COUNT; i++) {
        const char *s = TEST_STRINGS[i];
        printf("INPUT: ");
        print_hex(s);
        printf("\n");

        size_t n = 0;
        cap      = strlen(s) * 3 + 1; /* a space may cost 3 byte ids */
        ids      = realloc(ids, cap * sizeof *ids);
        if (ids == nullptr || gguf_tokenizer_encode(&tok, s, cap, ids, &n) != GEIST_OK) {
            printf("  ids:        ENCODE_FAILED\n\n");
            rc = 1;
            continue;
        }
        printf("  ids:        ");
        for (size_t j = 0; j < n; j++) {
            if (j > 0)
                printf(",");
            printf("%d", ids[j]);
        }
        printf("\n\n");
    }
    free(ids);
    gguf_tokenizer_unload(&tok);
    return rc;
}
