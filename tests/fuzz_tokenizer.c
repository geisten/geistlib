/*
 * fuzz_tokenizer — fuzz the GGUF tokenizer: vocab, merges and the encoder.
 *
 * The token table, the scores, the token types and the BPE merge list all come
 * out of the model file, so they are attacker-controlled in exactly the sense
 * gguf_reader.c:153 means. The tokenizer then builds a hash table sized from
 * that vocab count, splits every merge entry on its first space, and runs an
 * encoder over caller text — length arithmetic on untrusted data at every step,
 * which is exactly where overflow and bounds bugs live.
 *
 * The input is one GGUF. Its tail doubles as the text to encode, so a single
 * mutation surface reaches both the loader and the encoder. Decode runs on the
 * IDs that come back plus deliberately out-of-range IDs, because the header
 * promises those are written as <unk> rather than read out of bounds.
 *
 * Copy mode (gguf_tokenizer_load_copy) is the one geist_model_load uses: it
 * memcpys every vocab and merge byte out of the mapping, so it is the path
 * where a bad length becomes a bad memcpy.
 */
/* gguf_tokenizer.h is engine-internal on purpose; a fuzzer of the loader has
 * to reach in the same way the engine does. */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "gguf_reader.h"
#include "gguf_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The seed builder serves the standalone driver only: libFuzzer reads the
 * corpus file that `--seed` wrote, so under -fsanitize=fuzzer these helpers
 * are dead code and clang says so under -Werror. */
#ifdef GEIST_FUZZ_STANDALONE

/* ---- append-only byte buffer for building the seed -------------------- */

struct buf {
    uint8_t b[1024];
    size_t  n;
};

static void put_bytes(struct buf *o, const void *p, size_t n) {
    if (o->n + n > sizeof(o->b)) {
        fprintf(stderr, "fuzz_tokenizer: seed buffer overflow\n");
        exit(1);
    }
    memcpy(o->b + o->n, p, n);
    o->n += n;
}
static void put_u32(struct buf *o, uint32_t v) {
    put_bytes(o, &v, 4);
}
static void put_u64(struct buf *o, uint64_t v) {
    put_bytes(o, &v, 8);
}
static void put_f32(struct buf *o, float v) {
    put_bytes(o, &v, 4);
}
static void put_i32(struct buf *o, int32_t v) {
    put_bytes(o, &v, 4);
}
static void put_gstr(struct buf *o, const char *s) {
    const size_t len = strlen(s);
    put_u64(o, (uint64_t) len);
    put_bytes(o, s, len);
}

enum { VT_U32 = 4, VT_I32 = 5, VT_F32 = 6, VT_STRING = 8, VT_ARRAY = 9 };

#define GGUF_MAGIC 0x46554747u

/* Vocab small enough to leave room for mutation, wide enough that the
 * encoder's merge and byte-fallback paths are both reachable. */
static const char *const VOCAB[] = {"<unk>", "<s>", "</s>", "\xe2\x96\x81the", "he", "t", "h", "e"};
#define VOCAB_N ((uint64_t) (sizeof VOCAB / sizeof VOCAB[0]))

/* The seed's last 64 bytes, which the target encodes: merges, specials, a
 * literal ▁, multi-byte and invalid UTF-8, runs of whitespace. */
static const char SEED_TEXT[] =
        "the hehe teeth <s>th</s> \xe2\x96\x81the\t\xc3\xa9\xf0\x9f\x98\x80 "
        "\xff\xc3  eh\nthe heath the thee";
static_assert(sizeof SEED_TEXT - 1 == 64, "the target encodes the input's last 64 bytes");

/* A GGUF carrying a loadable SPM tokenizer: model + tokens are the two keys
 * gguf_tokenizer_load requires, the rest exercise the optional paths. */
static void seed_gguf(struct buf *o) {
    o->n = 0;
    put_u32(o, GGUF_MAGIC);
    put_u32(o, 3); /* version */
    put_u64(o, 1); /* n_tensors — the reader refuses 0, see below */
    put_u64(o, 7); /* n_meta */

    put_gstr(o, "general.alignment");
    put_u32(o, VT_U32);
    put_u32(o, 32);

    put_gstr(o, "tokenizer.ggml.model");
    put_u32(o, VT_STRING);
    put_gstr(o, "llama");

    put_gstr(o, "tokenizer.ggml.tokens");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_STRING);
    put_u64(o, VOCAB_N);
    for (uint64_t i = 0; i < VOCAB_N; i++)
        put_gstr(o, VOCAB[i]);

    put_gstr(o, "tokenizer.ggml.scores");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_F32);
    put_u64(o, VOCAB_N);
    for (uint64_t i = 0; i < VOCAB_N; i++)
        put_f32(o, -(float) i);

    put_gstr(o, "tokenizer.ggml.token_type");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_I32);
    put_u64(o, VOCAB_N);
    for (uint64_t i = 0; i < VOCAB_N; i++)
        put_i32(o, i < 3 ? 3 : 1); /* CONTROL for the specials, NORMAL else */

    put_gstr(o, "tokenizer.ggml.merges");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_STRING);
    put_u64(o, 3);
    put_gstr(o, "t h");
    put_gstr(o, "he e");
    put_gstr(o, "the"); /* no space: the loader's empty-right-half path */

    put_gstr(o, "tokenizer.ggml.bos_token_id");
    put_u32(o, VT_U32);
    put_u32(o, 1);

    /* The tokenizer reads no tensor, but gguf_open_memory refuses a file
     * without one, and a refused input never reaches the tokenizer. Its
     * data ends the file, so it carries the text. */
    put_gstr(o, "t"); /* a 16-element f32 tensor: name, n_dims, dim, dtype, offset */
    put_u32(o, 1);
    put_u64(o, (sizeof SEED_TEXT - 1) / 4);
    put_u32(o, 0);
    put_u64(o, 0);
    static const uint8_t zeros[32] = {0};
    put_bytes(o, zeros, (32 - o->n % 32) % 32); /* general.alignment above */
    put_bytes(o, SEED_TEXT, sizeof SEED_TEXT - 1);
}

#endif /* GEIST_FUZZ_STANDALONE */

/* ---- the target ------------------------------------------------------- */

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

static volatile size_t sink;

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    uint8_t *copy = malloc(size ? size : 1);
    if (copy == nullptr)
        return 0;
    memcpy(copy, data, size);

    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open_memory(copy, size, &err);
    if (ctx == nullptr) {
        free(copy);
        return 0;
    }

    struct gguf_tokenizer tok;
    if (!gguf_tokenizer_load_copy(&tok, ctx)) {
        gguf_close(ctx);
        free(copy);
        return 0;
    }
    /* Copy mode promises independence from the mapping. Close and free the
     * GGUF here on purpose: a tokenizer that still points into it turns every
     * use below into a use-after-free that ASan will name. */
    gguf_close(ctx);
    free(copy);
    copy = nullptr;

    /* The input's tail is the text to encode: same mutation surface, and it
     * carries whatever bytes the fuzzer found interesting — including invalid
     * UTF-8, which the encoder must survive. */
    char         text[65];
    const size_t tail = size < 64 ? size : 64;
    memcpy(text, data + (size - tail), tail);
    text[tail] = '\0';

    int32_t ids[128];
    size_t  n_ids = 0;
    if (gguf_tokenizer_encode(&tok, text, sizeof ids / sizeof ids[0], ids, &n_ids) == GEIST_OK) {
        char out[512];
        sink = gguf_tokenizer_decode(&tok, ids, n_ids, out, sizeof out);
        /* A capacity the answer cannot fit in: the header promises a truncated
         * write and the would-be total, not a write past the end. */
        char small[4];
        sink = gguf_tokenizer_decode(&tok, ids, n_ids, small, sizeof small);
    }

    /* IDs outside the vocab must decode as placeholders. vocab_size comes from
     * the parsed file, not from the seed's constant: the fuzzer changes it. */
    const int32_t rogue[] = {-1, 0, INT32_MAX, (int32_t) (tok.vocab_size + 1000)};
    char          out2[64];
    sink = gguf_tokenizer_decode(&tok, rogue, sizeof rogue / sizeof rogue[0], out2, sizeof out2);

    (void) gguf_tokenizer_id_for_text(&tok, text);
    (void) gguf_tokenizer_id_for_text(&tok, "");

    gguf_tokenizer_unload(&tok);
    return 0;
}

/* ---- standalone driver ------------------------------------------------ */

#ifdef GEIST_FUZZ_STANDALONE
static uint64_t prng(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int main(int argc, char **argv) {
    struct buf seed;
    seed_gguf(&seed);

    if (argc == 2 && strcmp(argv[1], "--seed") == 0) {
        fwrite(seed.b, 1, seed.n, stdout);
        return 0;
    }

    /* A seed the reader refuses never reaches the tokenizer: every run below
     * would fuzz the reader alone and still report no crash. */
    const char           *err = nullptr;
    struct gguf_ctx      *ctx = gguf_open_memory(seed.b, seed.n, &err);
    struct gguf_tokenizer tok;
    if (ctx == nullptr || !gguf_tokenizer_load_copy(&tok, ctx)) {
        fprintf(stderr,
                "fuzz_tokenizer: the seed loads no tokenizer (%s)\n",
                err != nullptr ? err : "gguf_tokenizer_load_copy failed");
        gguf_close(ctx);
        return 1;
    }
    gguf_tokenizer_unload(&tok);
    gguf_close(ctx);

    const long runs = (argc == 2) ? strtol(argv[1], nullptr, 10) : 3000;
    uint64_t   s    = 0xC0FFEEull;

    LLVMFuzzerTestOneInput(seed.b, seed.n);

    for (long i = 0; i < runs; i++) {
        uint8_t input[sizeof seed.b];
        /* Full length: the reader refuses every prefix of the seed, so a cut
         * input never reaches the tokenizer. Cutting is fuzz_gguf's job. */
        const size_t size = seed.n;
        memcpy(input, seed.b, size);
        const unsigned pokes = 1u + (unsigned) (prng(&s) % 8u);
        for (unsigned p = 0; p < pokes; p++)
            input[prng(&s) % size] = (uint8_t) prng(&s);
        LLVMFuzzerTestOneInput(input, size);
    }
    printf("fuzz_tokenizer: %ld runs, no crash\n", runs);
    return 0;
}
#endif
