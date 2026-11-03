/*
 * test_control_looking_unit — the vocab entries matched as special tokens in
 * the text (#404).
 *
 * llama.cpp marks a list of control-looking strings CONTROL whatever their
 * tokenizer.ggml.token_type ("</s>", "<eos>", "<|im_end|>", ..., and the FIM
 * names when the file has no id for that role), so with special parsing on,
 * the text "</s>" is Qwen3's NORMAL token 128247, not "<" "/" "s" ">". Gemma 4
 * keeps "</s>" NORMAL (it has "<|tool_response>"). UNKNOWN entries ("<unk>")
 * are matched too. In-memory GGUFs ("gpt2" + "qwen2", no merges) check each
 * rule; parity on the real vocabularies (every entry as input) is checked
 * against llama-tokenize outside the unit suite (#404 PR).
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "test_helpers.h"

#include "gguf_reader.h"
#include "gguf_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

enum { VT_U32 = 4, VT_I32 = 5, VT_STRING = 8, VT_ARRAY = 9 };

static const char *const VOCAB[] = {"<",
                                    "/",
                                    "s",
                                    ">",
                                    "x",
                                    "P",
                                    "R",
                                    "E",
                                    "</s>",
                                    "<eos>",
                                    "<unk>",
                                    "<PRE>",
                                    "<|tool_response>"};
enum {
    ID_LT,
    ID_SL,
    ID_S,
    ID_GT,
    ID_X,
    ID_P,
    ID_R,
    ID_E,
    ID_CLOSE_S,
    ID_EOS,
    ID_UNK,
    ID_PRE,
    ID_TOOL,
    N_VOCAB
};

/* types: write token_type (all NORMAL but <unk>, UNKNOWN); tool: the last
 * entry is "<|tool_response>" (USER_DEFINED, as in Gemma 4), else "y";
 * fim_pre: the file gives <PRE> as tokenizer.ggml.fim_pre_token_id. */
struct variant {
    bool types, tool, fim_pre;
};

static void build(struct blob *o, struct variant v) {
    *o = (struct blob) {0};
    put(o, "GGUF", 4);
    put_u32(o, 3); /* version */
    put_u64(o, 1); /* n_tensors — the reader refuses 0 */
    put_u64(o, 4 + v.types + v.fim_pre);

    put_str(o, "tokenizer.ggml.model");
    put_u32(o, VT_STRING);
    put_str(o, "gpt2");

    put_str(o, "tokenizer.ggml.pre");
    put_u32(o, VT_STRING);
    put_str(o, "qwen2");

    put_str(o, "tokenizer.ggml.tokens");
    put_u32(o, VT_ARRAY);
    put_u32(o, VT_STRING);
    put_u64(o, N_VOCAB);
    for (size_t i = 0; i < N_VOCAB; i++)
        put_str(o, i == ID_TOOL && !v.tool ? "y" : VOCAB[i]);

    if (v.types) {
        put_str(o, "tokenizer.ggml.token_type");
        put_u32(o, VT_ARRAY);
        put_u32(o, VT_I32);
        put_u64(o, N_VOCAB);
        for (size_t i = 0; i < N_VOCAB; i++)
            put_u32(o, i == ID_UNK ? 2 : i == ID_TOOL && v.tool ? 4 : 1);
    }
    if (v.fim_pre) {
        put_str(o, "tokenizer.ggml.fim_pre_token_id");
        put_u32(o, VT_U32);
        put_u32(o, ID_PRE);
    }

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
    struct variant v;
    const char    *text;
    size_t         n;
    int32_t        want[8];
};

static const struct encode_case CASES[] = {
        /* NORMAL by type, special by text */
        {{true, false, false}, "x</s>x", 3, {ID_X, ID_CLOSE_S, ID_X}},
        {{true, false, false}, "<eos>", 1, {ID_EOS}},
        {{true, false, false}, "<PRE>", 1, {ID_PRE}},
        /* UNKNOWN */
        {{true, false, false}, "x<unk>", 2, {ID_X, ID_UNK}},
        /* without a token_type array too */
        {{false, false, false}, "</s>", 1, {ID_CLOSE_S}},
        /* the Gemma 4 workaround: "</s>" stays text, "<eos>" does not */
        {{true, true, false}, "</s>", 4, {ID_LT, ID_SL, ID_S, ID_GT}},
        {{true, true, false}, "<eos>x", 2, {ID_EOS, ID_X}},
        {{true, true, false}, "x<|tool_response>", 2, {ID_X, ID_TOOL}},
        /* a FIM name is promoted only when the file has no id for its role */
        {{true, false, true}, "<PRE>", 5, {ID_LT, ID_P, ID_R, ID_E, ID_GT}},
        {{true, false, true}, "</s>", 1, {ID_CLOSE_S}},
};

static int check(const struct encode_case *c, bool *error) {
    struct blob o;
    build(&o, c->v);
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
        fprintf(stderr, "tokenizer load failed\n");
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
        fprintf(stderr,
                "FAIL: types %d tool %d fim_pre %d '%s':",
                c->v.types,
                c->v.tool,
                c->v.fim_pre,
                c->text);
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
    for (size_t c = 0; c < sizeof CASES / sizeof CASES[0]; c++)
        fails += check(&CASES[c], &error);
    if (error)
        return GEIST_TEST_ERROR;
    printf("control-looking specials: %zu encodes, %d failed\n",
           sizeof CASES / sizeof CASES[0],
           fails);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
