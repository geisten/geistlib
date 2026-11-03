/*
 * test_bos_rule_unit — whether a GGUF tokenizer adds BOS, decided as
 * llama.cpp decides it (#385).
 *
 * llama.cpp (llama_vocab::impl::load) starts from the default of the
 * vocabulary type and the pretokenizer, lets tokenizer.ggml.add_bos_token
 * replace it, and forces it on for Gemma 4; llama-perplexity puts BOS at the
 * start of every chunk exactly when the result is true. Llama 3.x GGUFs
 * ("gpt2" + "llama-bpe") often carry no flag: geist used to answer false for
 * every byte-BPE file without one, so eval_geist's BOS command, and every
 * harness that asks it, scored Llama 3.2 without BOS while llama.cpp scored
 * it with one (+10 % perplexity at 512 tokens).
 *
 *   1. gguf_tokenizer_add_bos_rule against a table of llama.cpp's outcomes.
 *   2. gguf_tokenizer_load on in-memory GGUFs: the rule reaches add_bos, and
 *      a file without the flag and without a bos_id adds nothing.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "gguf_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---- 1. the rule -------------------------------------------------------- */

struct rule_case {
    const char *model, *pre; /* pre null: no tokenizer.ggml.pre */
    int         flag;        /* -1: no tokenizer.ggml.add_bos_token */
    bool        fallback, want;
};

static const struct rule_case RULE_CASES[] = {
        /* Llama 3.x without the flag (Llama 3.2 3B Instruct Q4_K_M): BOS. */
        {"gpt2", "llama-bpe", -1, false, true},
        {"gpt2", "llama3", -1, false, true},
        {"gpt2", "llama-v3", -1, false, true},
        {"gpt2", "tekken", -1, false, true},
        {"gpt2", "chameleon", -1, false, true},
        {"gpt2", "falcon3", -1, false, true},
        /* An explicit flag wins over the pretokenizer's default. */
        {"gpt2", "llama-bpe", 0, false, false},
        {"gpt2", "qwen2", 1, false, true},
        /* Qwen, SmolLM, GPT-OSS, Llama 4 without the flag: none (#281). */
        {"gpt2", "qwen2", -1, false, false},
        {"gpt2", "qwen35", -1, false, false},
        {"gpt2", "smollm", -1, false, false},
        {"gpt2", "gpt-4o", -1, false, false},
        {"gpt2", "llama4", -1, false, false},
        {"gpt2", nullptr, -1, false, false},
        /* The fallback is only for names llama.cpp does not know. */
        {"gpt2", "llama-bpe", -1, true, true},
        {"gpt2", "qwen2", -1, true, false},
        /* SentencePiece and WordPiece default to BOS; the flag still wins. */
        {"llama", nullptr, -1, false, true},
        {"llama", "default", 0, true, false},
        {"bert", nullptr, -1, false, true},
        /* T5 (UGM), RWKV, PLaMo-2: no BOS by default. */
        {"t5", nullptr, -1, true, false},
        {"rwkv", nullptr, -1, true, false},
        {"plamo2", nullptr, -1, true, false},
        {"t5", nullptr, 1, false, true},
        /* Gemma 4: on even against the flag. */
        {"gemma4", nullptr, 0, false, true},
        {"gemma4", nullptr, -1, false, true},
        {"gpt2", "gemma4", 0, false, true},
        {"gpt2", "granite-embed-multi-311m", 0, false, true},
        /* ... but only as a byte-BPE vocabulary: SPM keeps the flag. */
        {"llama", "gemma4", 0, false, false},
        /* Unknown to llama.cpp: geist's fallback answers. */
        {"mystery", nullptr, -1, true, true},
        {"mystery", nullptr, -1, false, false},
        {"mystery", nullptr, 0, true, false},
        /* Prefixes are not names. */
        {"gpt2", "llama-bp", -1, false, false},
        {"gpt", "llama-bpe", -1, false, false},
};

static int check_rule(void) {
    int fails = 0;
    for (size_t i = 0; i < sizeof RULE_CASES / sizeof RULE_CASES[0]; i++) {
        const struct rule_case *c = &RULE_CASES[i];
        /* Not NUL-terminated in a GGUF: pass a copy with trailing junk. */
        char model[32], pre[48];
        snprintf(model, sizeof model, "%sXYZ", c->model);
        if (c->pre != nullptr)
            snprintf(pre, sizeof pre, "%sXYZ", c->pre);
        const bool got = gguf_tokenizer_add_bos_rule(strlen(c->model),
                                                     c->pre != nullptr ? strlen(c->pre) : 0,
                                                     c->flag,
                                                     c->fallback,
                                                     model,
                                                     c->pre != nullptr ? pre : nullptr);
        if (got != c->want) {
            fprintf(stderr,
                    "FAIL: rule model=%s pre=%s flag=%d fallback=%d: got %d, want %d\n",
                    c->model,
                    c->pre != nullptr ? c->pre : "(none)",
                    c->flag,
                    c->fallback,
                    got,
                    c->want);
            fails++;
        }
    }
    return fails;
}

/* ---- 2. through the loader ---------------------------------------------- */

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

enum { VT_U32 = 4, VT_F32 = 6, VT_BOOL = 7, VT_STRING = 8, VT_ARRAY = 9 };

static const char *const VOCAB[] = {"<unk>", "<s>", "\xc4\xa0", "a", "b", "ab"};
constexpr size_t         N_VOCAB = sizeof VOCAB / sizeof VOCAB[0];

struct load_case {
    const char *what, *model, *pre;
    int         flag;   /* -1: absent */
    int         bos_id; /* -1: absent */
    bool        want;
};

static const struct load_case LOAD_CASES[] = {
        {"Llama 3 byte-BPE, no flag", "gpt2", "llama-bpe", -1, 1, true},
        {"Llama 3 byte-BPE, flag false", "gpt2", "llama-bpe", 0, 1, false},
        {"Llama 3 byte-BPE, no flag, no bos_id", "gpt2", "llama-bpe", -1, -1, false},
        {"Qwen3.5 byte-BPE with a bos_id, no flag", "gpt2", "qwen35", -1, 1, false},
        {"SentencePiece, no flag", "llama", nullptr, -1, 1, true},
        {"SentencePiece, flag false", "llama", nullptr, 0, 1, false},
        {"Gemma 4, flag false", "gemma4", nullptr, 0, 1, true},
        {"T5 unigram with a bos_id, no flag", "t5", nullptr, -1, 1, false},
};

/* A tokenizer-only GGUF with merges (gpt2 / SPM; T5 too, geist does not
 * care) plus the optional pre, bos_token_id and add_bos_token keys. */
static void build(struct blob *o, const struct load_case *c) {
    *o = (struct blob) {0};
    put(o, "GGUF", 4);
    put_u32(o, 3); /* version */
    put_u64(o, 1); /* n_tensors — the reader refuses 0 */
    put_u64(o, 4 + (c->pre != nullptr) + (c->bos_id >= 0) + (c->flag >= 0));

    put_str(o, "tokenizer.ggml.model");
    put_u32(o, VT_STRING);
    put_str(o, c->model);

    put_str(o, "tokenizer.ggml.tokens");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_STRING);
    put_u64(o, N_VOCAB);
    for (size_t i = 0; i < N_VOCAB; i++)
        put_str(o, VOCAB[i]);

    put_str(o, "tokenizer.ggml.scores");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_F32);
    put_u64(o, N_VOCAB);
    for (size_t i = 0; i < N_VOCAB; i++) {
        const float score = -(float) i;
        put(o, &score, 4);
    }

    put_str(o, "tokenizer.ggml.merges");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_STRING);
    put_u64(o, 1);
    put_str(o, "a b");

    if (c->pre != nullptr) {
        put_str(o, "tokenizer.ggml.pre");
        put_u32(o, VT_STRING);
        put_str(o, c->pre);
    }
    if (c->bos_id >= 0) {
        put_str(o, "tokenizer.ggml.bos_token_id");
        put_u32(o, VT_U32);
        put_u32(o, (uint32_t) c->bos_id);
    }
    if (c->flag >= 0) {
        put_str(o, "tokenizer.ggml.add_bos_token");
        put_u32(o, VT_BOOL);
        const uint8_t b = (uint8_t) c->flag;
        put(o, &b, 1);
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

static int check_load(bool *error) {
    int fails = 0;
    for (size_t i = 0; i < sizeof LOAD_CASES / sizeof LOAD_CASES[0]; i++) {
        const struct load_case *c = &LOAD_CASES[i];
        struct blob             o;
        build(&o, c);
        if (o.overflow) {
            fprintf(stderr, "blob buffer too small\n");
            *error = true;
            return fails;
        }
        const char      *err = nullptr;
        struct gguf_ctx *ctx = gguf_open_memory(o.b, o.n, &err);
        if (ctx == nullptr) {
            fprintf(stderr, "%s: gguf_open_memory: %s\n", c->what, err != nullptr ? err : "?");
            *error = true;
            return fails;
        }
        struct gguf_tokenizer tok;
        const bool            loaded = gguf_tokenizer_load_copy(&tok, ctx);
        gguf_close(ctx);
        if (!loaded) {
            fprintf(stderr, "%s: tokenizer load failed\n", c->what);
            *error = true;
            return fails;
        }
        if (tok.add_bos != c->want) {
            fprintf(stderr, "FAIL: load %s: add_bos %d, want %d\n", c->what, tok.add_bos, c->want);
            fails++;
        }
        gguf_tokenizer_unload(&tok);
    }
    return fails;
}

int main(void) {
    bool error = false;
    int  fails = check_rule() + check_load(&error);
    if (error)
        return GEIST_TEST_ERROR;
    if (fails == 0)
        printf("PASS: add_bos follows llama.cpp's rule (%zu rule cases, %zu loads)\n",
               sizeof RULE_CASES / sizeof RULE_CASES[0],
               sizeof LOAD_CASES / sizeof LOAD_CASES[0]);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
