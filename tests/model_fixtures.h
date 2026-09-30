/*
 * model_fixtures.h — whole models built in memory, for tests that need a
 * session without a model file, F32 weights throughout:
 *
 *   mf_llama_gguf   the tensors and metadata of tools/gen_synth_gguf.py's
 *                   llama, optionally with a GGUF tokenizer from
 *                   tokenizer_fixtures.h;
 *   mf_qwen35_gguf  a Qwen3.5-style hybrid: gated-DeltaNet blocks with a
 *                   softmax attention block every `interval` blocks.
 *
 * Load the buffer with geist_model_load_from_memory, or write it out for
 * geist_model_load. Header-only; each includer uses what it needs.
 */
#ifndef GEIST_TESTS_MODEL_FIXTURES_H
#define GEIST_TESTS_MODEL_FIXTURES_H

#include "tokenizer_fixtures.h"

#include <stdarg.h>
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

/* Qwen3.5 geometry (qwen35.* keys). Block i is attention when
 * (i + 1) % interval == 0, gated DeltaNet otherwise. Attention: GQA with
 * QK norms, RoPE on the first rope_dims of each head, and the joint
 * query+gate projection. DeltaNet: dn_k_heads key heads of dn_head_k and
 * dn_v_heads value heads of dn_head_v (a multiple of 4: the NEON
 * recurrence steps 4 lanes), a causal conv of dn_conv taps. The vocabulary
 * is the tokenizer's, as the family has no vocab_size key, so tok is
 * required; weights as mf_llama, with the DeltaNet decay ssm_a at -1 so the
 * state shrinks rather than grows. */
struct mf_qwen35 {
    uint32_t               layers, interval, d_model, heads, kv_heads, head_dim, rope_dims, ffn;
    uint32_t               dn_k_heads, dn_v_heads, dn_head_k, dn_head_v, dn_conv;
    uint64_t               seed;
    const struct tf_vocab *tok;
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

/* ---- F32 tensors ---------------------------------------------------------- */

enum mf_fill { MF_RAND, MF_ONE, MF_NEG_ONE, MF_ZERO };

struct mf_tensor {
    char         name[48];
    uint64_t     ne0, ne1; /* ne1 0: one dimension */
    enum mf_fill fill;
};

struct mf_tensors {
    struct mf_tensor *t;
    size_t            n, cap;
};

static inline void
mf_add(struct mf_tensors *ts, uint64_t ne0, uint64_t ne1, enum mf_fill fill, const char *fmt, ...) {
    if (ts->n == ts->cap) {
        ts->cap = ts->cap != 0 ? 2 * ts->cap : 32;
        ts->t   = realloc(ts->t, ts->cap * sizeof *ts->t);
        if (ts->t == nullptr) {
            fprintf(stderr, "model_fixtures: out of memory\n");
            exit(1);
        }
    }
    struct mf_tensor *t = &ts->t[ts->n++];
    va_list           ap;
    va_start(ap, fmt);
    vsnprintf(t->name, sizeof t->name, fmt, ap);
    va_end(ap);
    t->ne0  = ne0;
    t->ne1  = ne1;
    t->fill = fill;
}

/* The tensor infos, then the data, after the metadata: F32, each tensor
 * 32-byte aligned, MF_RAND values drawn in tensor order from `seed`. Frees
 * the list. */
static inline void mf_write_tensors(struct tf_buf *o, struct mf_tensors *ts, uint64_t seed) {
    uint64_t off = 0;
    for (size_t i = 0; i < ts->n; i++) {
        const struct mf_tensor *t = &ts->t[i];
        tf_gstr(o, t->name, strlen(t->name));
        tf_u32(o, t->ne1 != 0 ? 2 : 1);
        tf_u64(o, t->ne0);
        if (t->ne1 != 0) {
            tf_u64(o, t->ne1);
        }
        tf_u32(o, 0); /* F32 */
        tf_u64(o, off);
        off += (t->ne0 * (t->ne1 != 0 ? t->ne1 : 1) * 4 + 31) / 32 * 32;
    }
    uint64_t s = seed;
    for (size_t i = 0; i < ts->n; i++) {
        const struct mf_tensor *t = &ts->t[i];
        while (o->n % 32 != 0) {
            tf_put(o, "", 1);
        }
        const float c = t->fill == MF_ONE ? 1.0f : t->fill == MF_NEG_ONE ? -1.0f : 0.0f;
        for (uint64_t e = 0; e < t->ne0 * (t->ne1 != 0 ? t->ne1 : 1); e++) {
            tf_f32(o, t->fill == MF_RAND ? mf_weight(&s) : c);
        }
    }
    free(ts->t);
    *ts = (struct mf_tensors) {0};
}

/* ---- llama ---------------------------------------------------------------- */

static inline struct tf_buf mf_llama_gguf(const struct mf_llama *c) {
    const uint32_t    hd = c->d_model / c->heads;
    struct mf_tensors ts = {0};
    mf_add(&ts, c->d_model, c->vocab, MF_RAND, "token_embd.weight");
    mf_add(&ts, c->d_model, 0, MF_ONE, "output_norm.weight");
    for (uint32_t l = 0; l < c->layers; l++) {
        mf_add(&ts, c->d_model, 0, MF_ONE, "blk.%u.attn_norm.weight", l);
        mf_add(&ts, c->d_model, c->heads * hd, MF_RAND, "blk.%u.attn_q.weight", l);
        mf_add(&ts, c->d_model, c->kv_heads * hd, MF_RAND, "blk.%u.attn_k.weight", l);
        mf_add(&ts, c->d_model, c->kv_heads * hd, MF_RAND, "blk.%u.attn_v.weight", l);
        mf_add(&ts, c->heads * hd, c->d_model, MF_RAND, "blk.%u.attn_output.weight", l);
        mf_add(&ts, c->d_model, 0, MF_ONE, "blk.%u.ffn_norm.weight", l);
        mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_gate.weight", l);
        mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_up.weight", l);
        mf_add(&ts, c->ffn, c->d_model, MF_RAND, "blk.%u.ffn_down.weight", l);
    }
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
    tf_u64(&o, ts.n);
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
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

/* ---- qwen35 --------------------------------------------------------------- */

static inline struct tf_buf mf_qwen35_gguf(const struct mf_qwen35 *c) {
    const uint64_t    vocab     = c->tok->n_tok;
    const uint64_t    q_out     = (uint64_t) c->heads * c->head_dim;
    const uint64_t    kv_out    = (uint64_t) c->kv_heads * c->head_dim;
    const uint64_t    key_dim   = (uint64_t) c->dn_k_heads * c->dn_head_k;
    const uint64_t    value_dim = (uint64_t) c->dn_v_heads * c->dn_head_v;
    const uint64_t    conv_dim  = 2 * key_dim + value_dim;
    struct mf_tensors ts        = {0};
    mf_add(&ts, c->d_model, vocab, MF_RAND, "token_embd.weight"); /* tied lm_head */
    mf_add(&ts, c->d_model, 0, MF_ONE, "output_norm.weight");
    for (uint32_t l = 0; l < c->layers; l++) {
        mf_add(&ts, c->d_model, 0, MF_ONE, "blk.%u.attn_norm.weight", l);
        /* The pre-FFN norm, under its HF name in this family. */
        mf_add(&ts, c->d_model, 0, MF_ONE, "blk.%u.post_attention_norm.weight", l);
        if ((l + 1) % c->interval == 0) {
            mf_add(&ts, c->head_dim, 0, MF_ONE, "blk.%u.attn_q_norm.weight", l);
            mf_add(&ts, c->head_dim, 0, MF_ONE, "blk.%u.attn_k_norm.weight", l);
            mf_add(&ts, c->d_model, 2 * q_out, MF_RAND, "blk.%u.attn_q.weight", l); /* + gate */
            mf_add(&ts, c->d_model, kv_out, MF_RAND, "blk.%u.attn_k.weight", l);
            mf_add(&ts, c->d_model, kv_out, MF_RAND, "blk.%u.attn_v.weight", l);
            mf_add(&ts, q_out, c->d_model, MF_RAND, "blk.%u.attn_output.weight", l);
        } else {
            mf_add(&ts, c->d_model, conv_dim, MF_RAND, "blk.%u.attn_qkv.weight", l);
            mf_add(&ts, c->d_model, value_dim, MF_RAND, "blk.%u.attn_gate.weight", l);
            mf_add(&ts, c->dn_conv, conv_dim, MF_RAND, "blk.%u.ssm_conv1d.weight", l);
            mf_add(&ts, c->d_model, c->dn_v_heads, MF_RAND, "blk.%u.ssm_beta.weight", l);
            mf_add(&ts, c->d_model, c->dn_v_heads, MF_RAND, "blk.%u.ssm_alpha.weight", l);
            mf_add(&ts, c->dn_v_heads, 0, MF_NEG_ONE, "blk.%u.ssm_a", l);
            mf_add(&ts, c->dn_v_heads, 0, MF_ZERO, "blk.%u.ssm_dt.bias", l);
            mf_add(&ts, c->dn_head_v, 0, MF_ONE, "blk.%u.ssm_norm.weight", l);
            mf_add(&ts, value_dim, c->d_model, MF_RAND, "blk.%u.ssm_out.weight", l);
        }
        mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_gate.weight", l);
        mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_up.weight", l);
        mf_add(&ts, c->ffn, c->d_model, MF_RAND, "blk.%u.ffn_down.weight", l);
    }
    const struct {
        const char *key;
        uint32_t    v;
    } u32[] = {
            {"qwen35.block_count", c->layers},
            {"qwen35.embedding_length", c->d_model},
            {"qwen35.feed_forward_length", c->ffn},
            {"qwen35.attention.head_count", c->heads},
            {"qwen35.attention.head_count_kv", c->kv_heads},
            {"qwen35.attention.key_length", c->head_dim},
            {"qwen35.rope.dimension_count", c->rope_dims},
            {"qwen35.full_attention_interval", c->interval},
            {"qwen35.ssm.group_count", c->dn_k_heads},
            {"qwen35.ssm.time_step_rank", c->dn_v_heads},
            {"qwen35.ssm.state_size", c->dn_head_k},
            {"qwen35.ssm.inner_size", c->dn_v_heads * c->dn_head_v},
            {"qwen35.ssm.conv_kernel", c->dn_conv},
    };
    const size_t n_u32 = sizeof u32 / sizeof u32[0];

    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, ts.n);
    tf_u64(&o, 1 + n_u32 + 2 + TF_TOKENIZER_KEYS);
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, "qwen35", 6);
    for (size_t i = 0; i < n_u32; i++) {
        mf_key(&o, u32[i].key, TF_GGUF_UINT32);
        tf_u32(&o, u32[i].v);
    }
    mf_key(&o, "qwen35.rope.freq_base", TF_GGUF_FLOAT32);
    tf_f32(&o, 10000000.0f);
    mf_key(&o, "qwen35.attention.layer_norm_rms_epsilon", TF_GGUF_FLOAT32);
    tf_f32(&o, 1e-6f);
    tf_tokenizer_kv(&o, c->tok, "gpt2", true);
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

#endif /* GEIST_TESTS_MODEL_FIXTURES_H */
