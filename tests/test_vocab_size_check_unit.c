/*
 * test_vocab_size_check_unit — a GGUF that gives no vocabulary size is
 * refused at load, and one that gives it only through its tokenizer is not.
 *
 * Many converters write no <arch>.vocab_size key; every family then takes
 * the tokenizer.ggml.tokens count (family_vocab_size). With neither, the
 * vocabulary was 0 and the load went on to size the embedding table, the
 * LM head and the token-id checks by it. state_create now refuses that and
 * names the family.
 *
 * Minimal in-memory llama GGUFs (metadata plus one dummy tensor, as in
 * test_head_count_check_unit): the one with neither must be refused naming
 * the vocabulary; the ones with the key or only the token list must get past
 * the check (they then fail for the missing weights, which is not what this
 * test is about).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { N_TOKENS = 32 };

static size_t put_u32(uint8_t *p, uint32_t v) {
    memcpy(p, &v, 4);
    return 4;
}

static size_t put_u64(uint8_t *p, uint64_t v) {
    memcpy(p, &v, 8);
    return 8;
}

static size_t put_str(uint8_t *p, const char *s) {
    const uint64_t n = strlen(s);
    size_t         o = put_u64(p, n);
    memcpy(p + o, s, n);
    return o + n;
}

static size_t kv_u32(uint8_t *p, const char *key, uint32_t v) {
    size_t o = put_str(p, key);
    o += put_u32(p + o, 4); /* GGUF value type: uint32 */
    return o + put_u32(p + o, v);
}

/* tokenizer.ggml.tokens: an array of N_TOKENS one-byte strings. */
static size_t kv_tokens(uint8_t *p) {
    size_t o = put_str(p, "tokenizer.ggml.tokens");
    o += put_u32(p + o, 9); /* array */
    o += put_u32(p + o, 8); /* of strings */
    o += put_u64(p + o, N_TOKENS);
    for (int i = 0; i < N_TOKENS; i++) {
        const char s[2] = {(char) ('A' + i), '\0'};
        o += put_str(p + o, s);
    }
    return o;
}

/* One-layer llama, 2 heads of 32; the vocabulary from the key, the token
 * list, or nowhere. */
static size_t build_gguf(uint8_t *buf, bool vocab_key, bool tokens) {
    size_t o = 0;
    memcpy(buf + o, "GGUF", 4);
    o += 4;
    o += put_u32(buf + o, 3);                                          /* version */
    o += put_u64(buf + o, 1);                                          /* n_tensors */
    o += put_u64(buf + o, 7 + (vocab_key ? 1 : 0) + (tokens ? 1 : 0)); /* n_kv */
    o += put_str(buf + o, "general.architecture");
    o += put_u32(buf + o, 8); /* string */
    o += put_str(buf + o, "llama");
    o += kv_u32(buf + o, "llama.block_count", 1);
    o += kv_u32(buf + o, "llama.embedding_length", 64);
    o += kv_u32(buf + o, "llama.attention.head_count", 2);
    o += kv_u32(buf + o, "llama.attention.head_count_kv", 2);
    o += kv_u32(buf + o, "llama.feed_forward_length", 128);
    o += kv_u32(buf + o, "llama.context_length", 64);
    if (vocab_key) {
        o += kv_u32(buf + o, "llama.vocab_size", N_TOKENS);
    }
    if (tokens) {
        o += kv_tokens(buf + o);
    }
    o += put_str(buf + o, "t"); /* tensor info: name, dims, dtype, offset */
    o += put_u32(buf + o, 1);
    o += put_u64(buf + o, 1);
    o += put_u32(buf + o, 0); /* f32 */
    o += put_u64(buf + o, 0);
    while (o % 32 != 0) {
        buf[o++] = 0;
    }
    memset(buf + o, 0, 4); /* the tensor's 4 data bytes */
    return o + 4;
}

/* Loads the GGUF; returns the error text (nullptr when there is none). */
static const char *load(struct geist_backend *be, bool vocab_key, bool tokens, bool *loaded) {
    uint8_t             buf[2048];
    const size_t        n = build_gguf(buf, vocab_key, tokens);
    struct geist_model *m = nullptr;
    *loaded               = geist_model_load_from_memory(buf, n, be, &m) == GEIST_OK;
    if (m != nullptr) {
        geist_model_destroy(m);
    }
    return geist_backend_errmsg(be);
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_scalar backend did not register\n");
        return GEIST_TEST_SKIP;
    }
    int  fails  = 0;
    bool loaded = false;

    const char *e = load(be, false, false, &loaded);
    fails += geist_expect(!loaded && e != nullptr && strstr(e, "no vocabulary size") != nullptr,
                          "a GGUF with no vocab_size and no token list is refused naming it");
    if (e != nullptr && strstr(e, "no vocabulary size") == nullptr) {
        fprintf(stderr, "  error was: %s\n", e);
    }
    /* No weights in these GGUFs: the load fails later, for that reason. */
    e = load(be, false, true, &loaded);
    fails += geist_expect(e == nullptr || strstr(e, "vocabulary") == nullptr,
                          "the token list alone gives the vocabulary");
    e = load(be, true, false, &loaded);
    fails += geist_expect(e == nullptr || strstr(e, "vocabulary") == nullptr,
                          "the vocab_size key gives the vocabulary");

    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: a GGUF without a vocabulary size is refused at load\n");
    return GEIST_TEST_PASS;
}
