/*
 * model_fixtures.h — whole llama models built in memory, for tests that
 * need a session without a model file: the tensors and metadata of
 * tools/gen_synth_gguf.py's llama, F32 weights, and optionally a GGUF
 * tokenizer from tokenizer_fixtures.h. Load the buffer with
 * geist_model_load_from_memory, or write it out for geist_model_load.
 * Header-only; each includer uses what it needs.
 */
#ifndef GEIST_TESTS_MODEL_FIXTURES_H
#define GEIST_TESTS_MODEL_FIXTURES_H

#include "tokenizer_fixtures.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mf_llama {
    uint32_t layers, d_model, heads, kv_heads, ffn, vocab, context;
    /* 0: every weight 0.01; otherwise small values from this seed, so that
     * decoding moves through different tokens. Norms are 1. */
    uint64_t seed;
    /* A GGUF tokenizer (tf_tokenizer_kv) when tok is set; vocab must then
     * be tok->n_tok. */
    const struct tf_vocab *tok;
    const char            *tok_model;
    bool                   tok_merges;
};

static inline void mf_key(struct tf_buf *o, const char *key, uint32_t type) {
    tf_gstr(o, key, strlen(key));
    tf_u32(o, type);
}

static inline float mf_weight(uint64_t *s) {
    if (*s == 0) {
        return 0.01f;
    }
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return ((float) (*s >> 40) / (float) (1u << 24) - 0.5f) * 0.2f;
}

static inline struct tf_buf mf_llama_gguf(const struct mf_llama *c) {
    const uint32_t hd = c->d_model / c->heads;
    enum { PER_LAYER = 9 };
    const size_t n_t = 2 + (size_t) PER_LAYER * c->layers;
    struct {
        char     name[48];
        uint64_t ne0, ne1; /* ne1 0: one dimension */
        bool     norm;
    } *t = malloc(n_t * sizeof *t);
    if (t == nullptr) {
        fprintf(stderr, "model_fixtures: out of memory\n");
        exit(1);
    }
    size_t n = 0;
#define MF_T(ne0_, ne1_, norm_, ...)                        \
    do {                                                    \
        snprintf(t[n].name, sizeof t[n].name, __VA_ARGS__); \
        t[n].ne0  = (ne0_);                                 \
        t[n].ne1  = (ne1_);                                 \
        t[n].norm = (norm_);                                \
        n++;                                                \
    } while (0)
    MF_T(c->d_model, c->vocab, false, "token_embd.weight");
    MF_T(c->d_model, 0, true, "output_norm.weight");
    for (uint32_t l = 0; l < c->layers; l++) {
        MF_T(c->d_model, 0, true, "blk.%u.attn_norm.weight", l);
        MF_T(c->d_model, c->heads * hd, false, "blk.%u.attn_q.weight", l);
        MF_T(c->d_model, c->kv_heads * hd, false, "blk.%u.attn_k.weight", l);
        MF_T(c->d_model, c->kv_heads * hd, false, "blk.%u.attn_v.weight", l);
        MF_T(c->heads * hd, c->d_model, false, "blk.%u.attn_output.weight", l);
        MF_T(c->d_model, 0, true, "blk.%u.ffn_norm.weight", l);
        MF_T(c->d_model, c->ffn, false, "blk.%u.ffn_gate.weight", l);
        MF_T(c->d_model, c->ffn, false, "blk.%u.ffn_up.weight", l);
        MF_T(c->ffn, c->d_model, false, "blk.%u.ffn_down.weight", l);
    }
#undef MF_T
    const struct {
        const char *key;
        uint32_t    v;
    } u32[] = {
            {"llama.block_count", c->layers},
            {"llama.embedding_length", c->d_model},
            {"llama.feed_forward_length", c->ffn},
            {"llama.attention.head_count", c->heads},
            {"llama.attention.head_count_kv", c->kv_heads},
            {"llama.context_length", c->context},
            {"llama.rope.dimension_count", hd},
            {"llama.vocab_size", c->vocab},
    };
    const size_t n_u32 = sizeof u32 / sizeof u32[0];

    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, n_t);
    tf_u64(&o, 1 + n_u32 + 2 + (c->tok != nullptr ? TF_TOKENIZER_KEYS : 0));
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, "llama", 5);
    for (size_t i = 0; i < n_u32; i++) {
        mf_key(&o, u32[i].key, TF_GGUF_UINT32);
        tf_u32(&o, u32[i].v);
    }
    mf_key(&o, "llama.rope.freq_base", TF_GGUF_FLOAT32);
    tf_f32(&o, 10000.0f);
    mf_key(&o, "llama.attention.layer_norm_rms_epsilon", TF_GGUF_FLOAT32);
    tf_f32(&o, 1e-5f);
    if (c->tok != nullptr) {
        tf_tokenizer_kv(&o, c->tok, c->tok_model, c->tok_merges);
    }
    uint64_t off = 0;
    for (size_t i = 0; i < n_t; i++) {
        tf_gstr(&o, t[i].name, strlen(t[i].name));
        tf_u32(&o, t[i].ne1 != 0 ? 2 : 1);
        tf_u64(&o, t[i].ne0);
        if (t[i].ne1 != 0) {
            tf_u64(&o, t[i].ne1);
        }
        tf_u32(&o, 0); /* F32 */
        tf_u64(&o, off);
        off += (t[i].ne0 * (t[i].ne1 != 0 ? t[i].ne1 : 1) * 4 + 31) / 32 * 32;
    }
    uint64_t s = c->seed;
    for (size_t i = 0; i < n_t; i++) {
        while (o.n % 32 != 0) {
            tf_put(&o, "", 1);
        }
        for (uint64_t e = 0; e < t[i].ne0 * (t[i].ne1 != 0 ? t[i].ne1 : 1); e++) {
            tf_f32(&o, t[i].norm ? 1.0f : mf_weight(&s));
        }
    }
    free(t);
    return o;
}

#endif /* GEIST_TESTS_MODEL_FIXTURES_H */
