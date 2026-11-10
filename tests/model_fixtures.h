/*
 * model_fixtures.h — whole models built in memory, for tests that need a
 * session without a model file, F32 weights throughout:
 *
 *   mf_llama_gguf   the tensors and metadata of tools/gen_synth_gguf.py's
 *                   llama, optionally with a GGUF tokenizer from
 *                   tokenizer_fixtures.h;
 *   mf_qwen35_gguf  a Qwen3.5-style hybrid: gated-DeltaNet blocks with a
 *                   softmax attention block every `interval` blocks;
 *   mf_qwen3moe_gguf a Qwen3-MoE: Qwen3 attention, a mixture-of-experts FFN.
 *
 * Load the buffer with geist_model_load_from_memory, or write it out for
 * geist_model_load. Header-only; each includer uses what it needs.
 */
#ifndef GEIST_TESTS_MODEL_FIXTURES_H
#define GEIST_TESTS_MODEL_FIXTURES_H

#include "tokenizer_fixtures.h"

#include <math.h>
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
    /* With tok: "<s>" is the BOS and a CONTROL token (tf_tokenizer_bos_kv). */
    bool tok_bos;
    /* attn_q stored as F16 instead of F32 (backends that widen or refuse
     * half-precision projections). */
    bool f16_q;
    /* ffn_down stored as IQ2_S (all-zero blocks; ffn a multiple of 256), a
     * dtype the CPU backends run and metal does not (Q5_0 was, until #376). */
    bool iq2s_down;
    /* A separate output.weight (lm_head), so token_embd is a lookup-only
     * table; tied otherwise. */
    bool untied;
    /* token_embd stored as F16 instead of F32 (a tied F16 lm_head, the
     * dtype the speculative output head sketches). */
    bool f16_embd;
    /* tokenizer.chat_template when set; its length in chat_template_len, so
     * a template may contain NUL bytes (0: strlen). */
    const char *chat_template;
    size_t      chat_template_len;
    /* rope_freqs.weight (Llama 3.x frequency factors, #379) with this many
     * elements, 1 + i / 2 (all 1 with rope_freqs_ones); 0: none. */
    uint32_t rope_freqs;
    bool     rope_freqs_ones;
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
    /* qwen35moe (#169) when experts > 0: the FFN is a router over
     * `experts` SwiGLU experts of width ffn, the `used` best per token, and
     * a shared expert of width ffn. With `dense_twin` the router, the
     * routed experts and the shared gate are zero and the shared expert
     * draws the dense FFN's values (down doubled): the same logits as the
     * dense qwen35 of the same seed, since sigmoid(0) = 1/2. `gate_ramp`
     * fills the shared gate with mf_ramp instead (a non-constant weight
     * sigmoid(x . g) a test computes itself). */
    uint32_t experts, used;
    bool     dense_twin, gate_ramp;
    bool     no_shared_key; /* no expert_shared_feed_forward_length */
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

/* MF_RAMP: element i is mf_ramp(i), mixed signs, no draw from the seed.
 * MF_STEP: element i is 1 + i / 2 (positive factors, e.g. rope_freqs).
 * MF_KB, MF_VB: a DeepSeek attn_kv_b split as llama.cpp's converter splits
 * it (#375), two consecutive tensors: MF_KB [nope, rank, heads] (kvb_pair:
 * the value dims) draws the whole MF_RAND kv_b [heads * (nope + v), rank]
 * where kv_b would and writes each head's k rows transposed, MF_VB [rank,
 * v, heads] (kvb_pair: nope) its v rows. The draws are kv_b's, so every
 * tensor after them has the values of the unsplit file. */
enum mf_fill { MF_RAND, MF_ONE, MF_NEG_ONE, MF_ZERO, MF_RAMP, MF_STEP, MF_KB, MF_VB };

static inline float mf_ramp(uint64_t i) {
    return ((float) (i % 7) - 3.0f) * 0.25f;
}

struct mf_tensor {
    char         name[48];
    uint64_t     ne0, ne1; /* ne1 0: one dimension */
    uint64_t     ne2;      /* 0: at most two dimensions */
    enum mf_fill fill;
    float        scale;    /* MF_RAND values times this (mf_add: 1) */
    bool         f16;      /* stored as F16; F32 otherwise */
    bool         iq2s;     /* stored as zero IQ2_S blocks (ne0 a multiple of 256) */
    bool         q8_0;     /* stored as Q8_0 of its values (ne0 a multiple of 32) */
    bool         q4k;      /* stored as Q4_K blocks of random codes (ne0 a multiple of 256) */
    uint64_t     kvb_pair; /* MF_KB: the value dims; MF_VB: nope */
};

struct mf_tensors {
    struct mf_tensor *t;
    size_t            n, cap;
};

[[gnu::format(printf, 5, 6)]] static inline void
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
    t->ne0      = ne0;
    t->ne1      = ne1;
    t->ne2      = 0;
    t->fill     = fill;
    t->scale    = 1.0f;
    t->f16      = false;
    t->iq2s     = false;
    t->q8_0     = false;
    t->q4k      = false;
    t->kvb_pair = 0;
}

static inline uint64_t mf_elems(const struct mf_tensor *t) {
    return t->ne0 * (t->ne1 != 0 ? t->ne1 : 1) * (t->ne2 != 0 ? t->ne2 : 1);
}

/* Bytes of a tensor's data: F32, F16, 82 per 256 elements for IQ2_S, 144
 * per 256 for Q4_K, or 34 per 32 for Q8_0. */
static inline uint64_t mf_bytes(const struct mf_tensor *t) {
    return t->iq2s   ? mf_elems(t) / 256 * 82
           : t->q4k  ? mf_elems(t) / 256 * 144
           : t->q8_0 ? mf_elems(t) / 32 * 34
                     : mf_elems(t) * (t->f16 ? 2 : 4);
}

/* The tensor infos, then the data, after the metadata: F32 (F16, zero
 * IQ2_S blocks or Q8_0 where the tensor says so), each tensor 32-byte
 * aligned, MF_RAND values drawn in tensor order from `seed`. Q8_0 is
 * ggml's: per 32 values d = max|v| / 127 as F16, then round(v / d) as int8.
 * Frees the list. */
static inline void mf_write_tensors(struct tf_buf *o, struct mf_tensors *ts, uint64_t seed) {
    uint64_t off = 0;
    for (size_t i = 0; i < ts->n; i++) {
        const struct mf_tensor *t = &ts->t[i];
        tf_gstr(o, t->name, strlen(t->name));
        tf_u32(o, t->ne2 != 0 ? 3 : t->ne1 != 0 ? 2 : 1);
        tf_u64(o, t->ne0);
        if (t->ne1 != 0) {
            tf_u64(o, t->ne1);
        }
        if (t->ne2 != 0) {
            tf_u64(o, t->ne2);
        }
        /* IQ2_S : Q4_K : Q8_0 : F16 : F32 */
        tf_u32(o, t->iq2s ? 22 : t->q4k ? 12 : t->q8_0 ? 8 : t->f16 ? 1 : 0);
        tf_u64(o, off);
        off += (mf_bytes(t) + 31) / 32 * 32;
    }
    uint64_t s   = seed;
    float   *kvb = nullptr; /* MF_KB's draws, until MF_VB has used them */
    for (size_t i = 0; i < ts->n; i++) {
        const struct mf_tensor *t = &ts->t[i];
        while (o->n % 32 != 0) {
            tf_put(o, "", 1);
        }
        if (t->iq2s) {
            for (uint64_t b = 0; b < mf_bytes(t); b++) {
                tf_put(o, "", 1);
            }
            continue;
        }
        if (t->q4k) {
            /* Per 256 values: d and dmin as F16 in [1/2048, 1/1024), then
             * the 12 scale bytes and 128 code bytes drawn from the seed. */
            for (uint64_t b = 0; b < mf_elems(t) / 256; b++) {
                for (int h = 0; h < 2; h++) {
                    const _Float16 dv = (_Float16) ((1.0f + fabsf(mf_weight(&s))) / 2048.0f);
                    tf_put(o, &dv, 2);
                }
                for (int k = 0; k < 140; k++) {
                    mf_weight(&s);
                    const uint8_t byte = (uint8_t) (s >> 24);
                    tf_put(o, &byte, 1);
                }
            }
            continue;
        }
        /* kv_b row (h, j) of nope + v rows per head, column r. */
        const bool     kb   = t->fill == MF_KB;
        const uint64_t nope = kb ? t->ne0 : t->kvb_pair, vd = kb ? t->kvb_pair : t->ne1;
        const uint64_t rank = kb ? t->ne1 : t->ne0;
        if (kb) {
            kvb = malloc(t->ne2 * (nope + vd) * rank * sizeof *kvb);
            if (kvb == nullptr) {
                fprintf(stderr, "model_fixtures: out of memory\n");
                exit(1);
            }
            for (uint64_t e = 0; e < t->ne2 * (nope + vd) * rank; e++)
                kvb[e] = mf_weight(&s) * t->scale;
        }
        const float c = t->fill == MF_ONE ? 1.0f : t->fill == MF_NEG_ONE ? -1.0f : 0.0f;
        float       q8[32];
        for (uint64_t e = 0; e < mf_elems(t); e++) {
            const bool     sp = kb || t->fill == MF_VB; /* 3-D: element (a, b, h) */
            const uint64_t a = sp ? e % t->ne0 : 0, b = sp ? e / t->ne0 % t->ne1 : 0;
            const uint64_t h = sp ? e / (t->ne0 * t->ne1) : 0;
            const float    v = t->fill == MF_RAND   ? mf_weight(&s) * t->scale
                               : t->fill == MF_RAMP ? mf_ramp(e)
                               : t->fill == MF_STEP ? 1.0f + 0.5f * (float) e
                               : kb                 ? kvb[(h * (nope + vd) + a) * rank + b]
                               : t->fill == MF_VB   ? kvb[(h * (nope + vd) + nope + b) * rank + a]
                                                    : c;
            if (t->q8_0) {
                q8[e % 32] = v;
                if (e % 32 != 31)
                    continue;
                float amax = 0.0f;
                for (size_t j = 0; j < 32; j++)
                    amax = fabsf(q8[j]) > amax ? fabsf(q8[j]) : amax;
                const _Float16 d = (_Float16) (amax / 127.0f);
                tf_put(o, &d, 2);
                for (size_t j = 0; j < 32; j++) {
                    const int8_t qv = (int8_t) (amax > 0.0f ? lrintf(q8[j] / (float) d) : 0);
                    tf_put(o, &qv, 1);
                }
            } else if (t->f16) {
                const _Float16 hv = (_Float16) v;
                tf_put(o, &hv, 2);
            } else {
                tf_f32(o, v);
            }
        }
        if (t->fill == MF_VB) {
            free(kvb);
            kvb = nullptr;
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
    ts.t[ts.n - 1].f16 = c->f16_embd;
    mf_add(&ts, c->d_model, 0, MF_ONE, "output_norm.weight");
    if (c->rope_freqs != 0) {
        mf_add(&ts, c->rope_freqs, 0, c->rope_freqs_ones ? MF_ONE : MF_STEP, "rope_freqs.weight");
    }
    if (c->untied) {
        mf_add(&ts, c->d_model, c->vocab, MF_RAND, "output.weight");
    }
    for (uint32_t l = 0; l < c->layers; l++) {
        mf_add(&ts, c->d_model, 0, MF_ONE, "blk.%u.attn_norm.weight", l);
        mf_add(&ts, c->d_model, c->heads * hd, MF_RAND, "blk.%u.attn_q.weight", l);
        ts.t[ts.n - 1].f16 = c->f16_q;
        mf_add(&ts, c->d_model, c->kv_heads * hd, MF_RAND, "blk.%u.attn_k.weight", l);
        mf_add(&ts, c->d_model, c->kv_heads * hd, MF_RAND, "blk.%u.attn_v.weight", l);
        mf_add(&ts, c->heads * hd, c->d_model, MF_RAND, "blk.%u.attn_output.weight", l);
        mf_add(&ts, c->d_model, 0, MF_ONE, "blk.%u.ffn_norm.weight", l);
        mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_gate.weight", l);
        mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_up.weight", l);
        mf_add(&ts, c->ffn, c->d_model, MF_RAND, "blk.%u.ffn_down.weight", l);
        ts.t[ts.n - 1].iq2s = c->iq2s_down;
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
    tf_u64(&o,
           1 + n_u32 + 2 + (c->tok != nullptr ? TF_TOKENIZER_KEYS : 0) +
                   (c->tok != nullptr && c->tok_bos ? TF_TOKENIZER_BOS_KEYS : 0) +
                   (c->chat_template != nullptr ? 1 : 0));
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, "llama", 5);
    if (c->chat_template != nullptr) {
        mf_key(&o, "tokenizer.chat_template", TF_GGUF_STRING);
        tf_gstr(&o,
                c->chat_template,
                c->chat_template_len ? c->chat_template_len : strlen(c->chat_template));
    }
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
        if (c->tok_bos) {
            tf_tokenizer_bos_kv(&o, c->tok);
        }
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
        if (c->experts == 0) {
            mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_gate.weight", l);
            mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_up.weight", l);
            mf_add(&ts, c->ffn, c->d_model, MF_RAND, "blk.%u.ffn_down.weight", l);
            continue;
        }
        const enum mf_fill routed = c->dense_twin ? MF_ZERO : MF_RAND;
        mf_add(&ts, c->d_model, c->experts, routed, "blk.%u.ffn_gate_inp.weight", l);
        mf_add(&ts, c->d_model, c->ffn, routed, "blk.%u.ffn_gate_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, c->d_model, c->ffn, routed, "blk.%u.ffn_up_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, c->ffn, c->d_model, routed, "blk.%u.ffn_down_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts,
               c->d_model,
               0,
               c->gate_ramp ? MF_RAMP : routed,
               "blk.%u.ffn_gate_inp_shexp.weight",
               l);
        mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_gate_shexp.weight", l);
        mf_add(&ts, c->d_model, c->ffn, MF_RAND, "blk.%u.ffn_up_shexp.weight", l);
        mf_add(&ts, c->ffn, c->d_model, MF_RAND, "blk.%u.ffn_down_shexp.weight", l);
        ts.t[ts.n - 1].scale = c->dense_twin ? 2.0f : 1.0f;
    }
    const char *const arch = c->experts > 0 ? "qwen35moe" : "qwen35";
    const struct {
        const char *key;
        uint32_t    v;
    } u32[] = {
            {"block_count", c->layers},
            {"embedding_length", c->d_model},
            {"attention.head_count", c->heads},
            {"attention.head_count_kv", c->kv_heads},
            {"attention.key_length", c->head_dim},
            {"rope.dimension_count", c->rope_dims},
            {"full_attention_interval", c->interval},
            {"ssm.group_count", c->dn_k_heads},
            {"ssm.time_step_rank", c->dn_v_heads},
            {"ssm.state_size", c->dn_head_k},
            {"ssm.inner_size", c->dn_v_heads * c->dn_head_v},
            {"ssm.conv_kernel", c->dn_conv},
            {"feed_forward_length", c->ffn}, /* dense; the MoE's below */
            {"expert_feed_forward_length", c->ffn},
            {"expert_shared_feed_forward_length", c->ffn},
            {"expert_count", c->experts},
            {"expert_used_count", c->used},
    };
    const size_t n_u32 = c->experts > 0 ? sizeof u32 / sizeof u32[0] - 1 - c->no_shared_key : 13;
    char         key[96];

    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, ts.n);
    tf_u64(&o, 1 + n_u32 + 2 + TF_TOKENIZER_KEYS);
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, arch, strlen(arch));
    for (size_t i = 0, j = 0; j < n_u32; i++) {
        if (c->experts > 0 && (i == 12 || (i == 14 && c->no_shared_key)))
            continue; /* no dense width; the shared width left out */
        snprintf(key, sizeof key, "%s.%s", arch, u32[i].key);
        mf_key(&o, key, TF_GGUF_UINT32);
        tf_u32(&o, u32[i].v);
        j++;
    }
    snprintf(key, sizeof key, "%s.rope.freq_base", arch);
    mf_key(&o, key, TF_GGUF_FLOAT32);
    tf_f32(&o, 10000000.0f);
    snprintf(key, sizeof key, "%s.attention.layer_norm_rms_epsilon", arch);
    mf_key(&o, key, TF_GGUF_FLOAT32);
    tf_f32(&o, 1e-6f);
    tf_tokenizer_kv(&o, c->tok, "gpt2", true);
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

/* ---- qwen3moe ------------------------------------------------------------ */

/* Qwen3-MoE geometry (qwen3moe.* keys): GQA with per-head QK norms and a
 * head_dim of its own; each layer's FFN is a router over `experts` SwiGLU
 * experts of width expert_ff, the `used` best per token. Weights as
 * mf_llama. For the load-refusal tests: `drop` names a tensor to leave out,
 * `gating` (when nonzero) writes qwen3moe.expert_gating_func, and
 * `shared_expert` adds a shared-expert router the family does not run.
 * `olmoe` writes the same model as OLMoE (olmoe.* keys): QK norms over the
 * whole row, the expert width in feed_forward_length, no key_length;
 * `bad_qk_norm` then leaves those norms at head_dim. `mixtral` writes it as
 * Mixtral (llama.* keys): no QK norms, the same keys as OLMoE; `split_experts`
 * then adds the per-expert tensors of old files, ahead of the merged ones.
 * `granite` writes it as Granite-MoE (granitemoe.* keys, shaped like Mixtral)
 * with the four scales below, each written when nonzero, and `shared_ff` as
 * the width of a shared expert. */
struct mf_qwen3moe {
    uint32_t    layers, d_model, heads, kv_heads, head_dim, experts, used, expert_ff, vocab;
    uint64_t    seed;
    const char *drop;
    uint32_t    gating;
    bool        shared_expert;
    bool        olmoe;
    bool        bad_qk_norm;
    bool        mixtral;
    bool        split_experts;
    bool        granite;
    float       attention_scale, embedding_scale, residual_scale, logit_scale;
    uint32_t    shared_ff;
    /* #482: the routed experts stored as Q8_0 (d_model and expert_ff
     * multiples of 32), a dtype the device's grouped GEMMs take. */
    bool q8_experts;
    /* #374: stored as Q4_K of random codes instead (d_model and expert_ff
     * multiples of 256): cpu_x86 repacks them. */
    bool q4k_experts;
};

static inline struct tf_buf mf_qwen3moe_gguf(const struct mf_qwen3moe *c) {
    const uint32_t    d     = c->d_model;
    const bool        no_qk = c->mixtral || c->granite;
    const bool        ol    = c->olmoe || no_qk; /* no key_length, width in feed_forward */
    const char *const arch  = c->mixtral   ? "llama"
                              : c->olmoe   ? "olmoe"
                              : c->granite ? "granitemoe"
                                           : "qwen3moe";
    const uint32_t    q_n   = c->olmoe && !c->bad_qk_norm ? c->heads * c->head_dim : c->head_dim;
    const uint32_t    k_n   = c->olmoe && !c->bad_qk_norm ? c->kv_heads * c->head_dim : c->head_dim;
    struct mf_tensors ts    = {0};
    mf_add(&ts, d, c->vocab, MF_RAND, "token_embd.weight");
    mf_add(&ts, d, 0, MF_ONE, "output_norm.weight");
    for (uint32_t l = 0; l < c->layers; l++) {
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.attn_norm.weight", l);
        mf_add(&ts, d, c->heads * c->head_dim, MF_RAND, "blk.%u.attn_q.weight", l);
        mf_add(&ts, d, c->kv_heads * c->head_dim, MF_RAND, "blk.%u.attn_k.weight", l);
        mf_add(&ts, d, c->kv_heads * c->head_dim, MF_RAND, "blk.%u.attn_v.weight", l);
        mf_add(&ts, c->heads * c->head_dim, d, MF_RAND, "blk.%u.attn_output.weight", l);
        if (!no_qk) {
            mf_add(&ts, q_n, 0, MF_ONE, "blk.%u.attn_q_norm.weight", l);
            mf_add(&ts, k_n, 0, MF_ONE, "blk.%u.attn_k_norm.weight", l);
        }
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.ffn_norm.weight", l);
        mf_add(&ts, d, c->experts, MF_RAND, "blk.%u.ffn_gate_inp.weight", l);
        if (c->split_experts) {
            mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_gate.0.weight", l);
        }
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_gate_exps.weight", l);
        ts.t[ts.n - 1].ne2  = c->experts;
        ts.t[ts.n - 1].q8_0 = c->q8_experts;
        ts.t[ts.n - 1].q4k  = c->q4k_experts;
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_up_exps.weight", l);
        ts.t[ts.n - 1].ne2  = c->experts;
        ts.t[ts.n - 1].q8_0 = c->q8_experts;
        ts.t[ts.n - 1].q4k  = c->q4k_experts;
        mf_add(&ts, c->expert_ff, d, MF_RAND, "blk.%u.ffn_down_exps.weight", l);
        ts.t[ts.n - 1].ne2  = c->experts;
        ts.t[ts.n - 1].q8_0 = c->q8_experts;
        ts.t[ts.n - 1].q4k  = c->q4k_experts;
        if (c->shared_expert) {
            mf_add(&ts, d, 0, MF_RAND, "blk.%u.ffn_gate_inp_shexp.weight", l);
        }
    }
    if (c->drop != nullptr) {
        for (size_t i = 0; i < ts.n; i++) {
            if (strcmp(ts.t[i].name, c->drop) == 0) {
                ts.t[i] = ts.t[--ts.n];
                break;
            }
        }
    }
    const struct {
        const char *key;
        uint32_t    v;
    } u32[] = {
            {"block_count", c->layers},
            {"embedding_length", d},
            {"attention.head_count", c->heads},
            {"attention.head_count_kv", c->kv_heads},
            {"expert_count", c->experts},
            {"expert_used_count", c->used},
            {"context_length", 4096},
            {"vocab_size", c->vocab},
            {ol ? "feed_forward_length" : "expert_feed_forward_length", c->expert_ff},
            {"attention.key_length", c->head_dim},
            {"expert_gating_func", c->gating},
            {"expert_shared_feed_forward_length", c->shared_ff},
    };
    /* OLMoE and Mixtral have no key_length; the gating key is written only
     * when asked. */
    const bool   write_u32[] = {true,
                                true,
                                true,
                                true,
                                true,
                                true,
                                true,
                                true,
                                true,
                                !ol,
                                c->gating != 0,
                                c->shared_ff != 0};
    const size_t n_u32       = sizeof u32 / sizeof u32[0];
    size_t       n_written   = 0;
    for (size_t i = 0; i < n_u32; i++) {
        n_written += write_u32[i] ? 1 : 0;
    }
    const struct {
        const char *key;
        float       v;
    } f32[] = {
            {"attention.scale", c->attention_scale},
            {"embedding_scale", c->embedding_scale},
            {"residual_scale", c->residual_scale},
            {"logit_scale", c->logit_scale},
    };
    for (size_t i = 0; i < sizeof f32 / sizeof f32[0]; i++) {
        n_written += f32[i].v != 0.0f ? 1 : 0;
    }

    char          key[64];
    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, ts.n);
    tf_u64(&o, 1 + n_written + 2);
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, arch, strlen(arch));
    for (size_t i = 0; i < n_u32; i++) {
        if (write_u32[i]) {
            snprintf(key, sizeof key, "%s.%s", arch, u32[i].key);
            mf_key(&o, key, TF_GGUF_UINT32);
            tf_u32(&o, u32[i].v);
        }
    }
    for (size_t i = 0; i < sizeof f32 / sizeof f32[0]; i++) {
        if (f32[i].v != 0.0f) {
            snprintf(key, sizeof key, "%s.%s", arch, f32[i].key);
            mf_key(&o, key, TF_GGUF_FLOAT32);
            tf_f32(&o, f32[i].v);
        }
    }
    snprintf(key, sizeof key, "%s.rope.freq_base", arch);
    mf_key(&o, key, TF_GGUF_FLOAT32);
    tf_f32(&o, 1000000.0f);
    snprintf(key, sizeof key, "%s.attention.layer_norm_rms_epsilon", arch);
    mf_key(&o, key, TF_GGUF_FLOAT32);
    tf_f32(&o, 1e-6f);
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

/* GPT-OSS (#243): q/k/v/o biases, a sink logit per head (attn_sinks,
 * drawn at sink_scale times the usual range), even layers sliding over
 * `window` keys, YaRN RoPE unless no_yarn, and an MoE whose router and
 * experts carry biases. F32 weights, tensors and keys under "gpt-oss". */
struct mf_gpt_oss {
    uint32_t layers, d_model, heads, kv_heads, head_dim, experts, used, expert_ff, vocab, window;
    uint64_t seed;
    float    sink_scale;
    bool     no_yarn;
    float    gate_bias_scale; /* the experts' gate biases times this (0: 1) */
};

static inline struct tf_buf mf_gpt_oss_gguf(const struct mf_gpt_oss *c) {
    const uint32_t    d  = c->d_model;
    const uint32_t    q  = c->heads * c->head_dim;
    const uint32_t    kv = c->kv_heads * c->head_dim;
    struct mf_tensors ts = {0};
    mf_add(&ts, d, c->vocab, MF_RAND, "token_embd.weight");
    mf_add(&ts, d, c->vocab, MF_RAND, "output.weight");
    mf_add(&ts, d, 0, MF_ONE, "output_norm.weight");
    for (uint32_t l = 0; l < c->layers; l++) {
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.attn_norm.weight", l);
        mf_add(&ts, d, q, MF_RAND, "blk.%u.attn_q.weight", l);
        mf_add(&ts, q, 0, MF_RAND, "blk.%u.attn_q.bias", l);
        mf_add(&ts, d, kv, MF_RAND, "blk.%u.attn_k.weight", l);
        mf_add(&ts, kv, 0, MF_RAND, "blk.%u.attn_k.bias", l);
        mf_add(&ts, d, kv, MF_RAND, "blk.%u.attn_v.weight", l);
        mf_add(&ts, kv, 0, MF_RAND, "blk.%u.attn_v.bias", l);
        mf_add(&ts, q, d, MF_RAND, "blk.%u.attn_output.weight", l);
        mf_add(&ts, d, 0, MF_RAND, "blk.%u.attn_output.bias", l);
        mf_add(&ts, c->heads, 0, MF_RAND, "blk.%u.attn_sinks.weight", l);
        ts.t[ts.n - 1].scale = c->sink_scale;
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.post_attention_norm.weight", l);
        mf_add(&ts, d, c->experts, MF_RAND, "blk.%u.ffn_gate_inp.weight", l);
        mf_add(&ts, c->experts, 0, MF_RAND, "blk.%u.ffn_gate_inp.bias", l);
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_gate_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, c->expert_ff, c->experts, MF_RAND, "blk.%u.ffn_gate_exps.bias", l);
        if (c->gate_bias_scale != 0.0f)
            ts.t[ts.n - 1].scale = c->gate_bias_scale;
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_up_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, c->expert_ff, c->experts, MF_RAND, "blk.%u.ffn_up_exps.bias", l);
        mf_add(&ts, c->expert_ff, d, MF_RAND, "blk.%u.ffn_down_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, d, c->experts, MF_RAND, "blk.%u.ffn_down_exps.bias", l);
    }
    const struct {
        const char *key;
        uint32_t    v;
    } u32[] = {
            {"block_count", c->layers},
            {"embedding_length", d},
            {"attention.head_count", c->heads},
            {"attention.head_count_kv", c->kv_heads},
            {"expert_count", c->experts},
            {"expert_used_count", c->used},
            {"context_length", 4096},
            {"vocab_size", c->vocab},
            {"expert_feed_forward_length", c->expert_ff},
            {"attention.key_length", c->head_dim},
            {"attention.sliding_window", c->window},
            {"rope.scaling.original_context_length", 64},
    };
    const struct {
        const char *key;
        float       v;
    } f32[] = {
            {"rope.freq_base", 150000.0f},
            {"attention.layer_norm_rms_epsilon", 1e-5f},
            {"rope.scaling.factor", 32.0f},
            {"rope.scaling.yarn_beta_fast", 32.0f},
            {"rope.scaling.yarn_beta_slow", 1.0f},
    };
    const size_t  n_u32 = sizeof u32 / sizeof u32[0], n_f32 = sizeof f32 / sizeof f32[0];
    const char    arch[] = "gpt-oss";
    char          key[80];
    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, ts.n);
    tf_u64(&o, 1 + n_u32 + n_f32 + (c->no_yarn ? 0 : 1));
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, arch, strlen(arch));
    for (size_t i = 0; i < n_u32; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, u32[i].key);
        mf_key(&o, key, TF_GGUF_UINT32);
        tf_u32(&o, u32[i].v);
    }
    for (size_t i = 0; i < n_f32; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, f32[i].key);
        mf_key(&o, key, TF_GGUF_FLOAT32);
        tf_f32(&o, f32[i].v);
    }
    if (!c->no_yarn) {
        snprintf(key, sizeof key, "%s.rope.scaling.type", arch);
        mf_key(&o, key, TF_GGUF_STRING);
        tf_gstr(&o, "yarn", 4);
    }
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

/* DeepSeek-V2 (#242), the unsplit attn_kv_b layout, or with `split` the
 * same weights as llama.cpp's converter writes them today (#375): k_b /
 * v_b (MF_KB / MF_VB), one KV head of key_length rank + rope and
 * value_length rank, the per-head sizes in key_length_mla /
 * value_length_mla. `lead` dense layers,
 * then MoE layers with expert_shared_count shared experts. q_rank > 0
 * writes q_a / q_a_norm / q_b (V3) instead of attn_q. YaRN with DeepSeek's
 * log multiplier unless no_yarn. v3 writes a sigmoid router (gating 2) with
 * exp_probs_b, n_group groups (topk_group used), weights renormalised and
 * scaled by w_scale. F32 weights. */
struct mf_deepseek2 {
    uint32_t layers, lead, d_model, heads, nope, rope, v_dim, kv_rank, q_rank, ff;
    uint32_t experts, used, expert_ff, shared, vocab;
    uint64_t seed;
    bool     no_yarn, v3;
    bool     split; /* attn_k_b / attn_v_b and the absorbed-MLA metadata (#375) */
    uint32_t n_group, topk_group;
    float    w_scale;
};

static inline struct tf_buf mf_deepseek2_gguf(const struct mf_deepseek2 *c) {
    const uint32_t    d  = c->d_model;
    const uint32_t    hk = c->nope + c->rope;
    struct mf_tensors ts = {0};
    mf_add(&ts, d, c->vocab, MF_RAND, "token_embd.weight");
    mf_add(&ts, d, c->vocab, MF_RAND, "output.weight");
    mf_add(&ts, d, 0, MF_ONE, "output_norm.weight");
    for (uint32_t l = 0; l < c->layers; l++) {
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.attn_norm.weight", l);
        if (c->q_rank > 0) {
            mf_add(&ts, d, c->q_rank, MF_RAND, "blk.%u.attn_q_a.weight", l);
            mf_add(&ts, c->q_rank, 0, MF_ONE, "blk.%u.attn_q_a_norm.weight", l);
            mf_add(&ts, c->q_rank, c->heads * hk, MF_RAND, "blk.%u.attn_q_b.weight", l);
        } else {
            mf_add(&ts, d, c->heads * hk, MF_RAND, "blk.%u.attn_q.weight", l);
        }
        mf_add(&ts, d, c->kv_rank + c->rope, MF_RAND, "blk.%u.attn_kv_a_mqa.weight", l);
        mf_add(&ts, c->kv_rank, 0, MF_RAMP, "blk.%u.attn_kv_a_norm.weight", l);
        if (c->split) {
            mf_add(&ts, c->nope, c->kv_rank, MF_KB, "blk.%u.attn_k_b.weight", l);
            ts.t[ts.n - 1].ne2      = c->heads;
            ts.t[ts.n - 1].kvb_pair = c->v_dim;
            mf_add(&ts, c->kv_rank, c->v_dim, MF_VB, "blk.%u.attn_v_b.weight", l);
            ts.t[ts.n - 1].ne2      = c->heads;
            ts.t[ts.n - 1].kvb_pair = c->nope;
        } else {
            mf_add(&ts,
                   c->kv_rank,
                   c->heads * (c->nope + c->v_dim),
                   MF_RAND,
                   "blk.%u.attn_kv_b.weight",
                   l);
        }
        mf_add(&ts, c->heads * c->v_dim, d, MF_RAND, "blk.%u.attn_output.weight", l);
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.ffn_norm.weight", l);
        if (l < c->lead) {
            mf_add(&ts, d, c->ff, MF_RAND, "blk.%u.ffn_gate.weight", l);
            mf_add(&ts, d, c->ff, MF_RAND, "blk.%u.ffn_up.weight", l);
            mf_add(&ts, c->ff, d, MF_RAND, "blk.%u.ffn_down.weight", l);
            continue;
        }
        mf_add(&ts, d, c->experts, MF_RAND, "blk.%u.ffn_gate_inp.weight", l);
        ts.t[ts.n - 1].scale = 10.0f; /* routing with clear margins */
        if (c->v3) {
            mf_add(&ts, c->experts, 0, MF_RAND, "blk.%u.exp_probs_b.bias", l);
        }
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_gate_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_up_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, c->expert_ff, d, MF_RAND, "blk.%u.ffn_down_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, d, c->shared * c->expert_ff, MF_RAND, "blk.%u.ffn_gate_shexp.weight", l);
        mf_add(&ts, d, c->shared * c->expert_ff, MF_RAND, "blk.%u.ffn_up_shexp.weight", l);
        mf_add(&ts, c->shared * c->expert_ff, d, MF_RAND, "blk.%u.ffn_down_shexp.weight", l);
    }
    struct {
        const char *key;
        uint32_t    v;
    } u32[16] = {
            {"block_count", c->layers},
            {"embedding_length", d},
            {"attention.head_count", c->heads},
            {"attention.head_count_kv", c->split ? 1 : c->heads},
            {"expert_count", c->experts},
            {"expert_used_count", c->used},
            {"context_length", 4096},
            {"vocab_size", c->vocab},
            {"feed_forward_length", c->ff},
            {"expert_feed_forward_length", c->expert_ff},
            {"expert_shared_count", c->shared},
            {"leading_dense_block_count", c->lead},
            {"attention.kv_lora_rank", c->kv_rank},
            {"attention.key_length", c->split ? c->kv_rank + c->rope : hk},
            {"attention.value_length", c->split ? c->kv_rank : c->v_dim},
            {"rope.dimension_count", c->rope},
    };
    size_t n_u32 = 16;
    struct {
        const char *key;
        uint32_t    v;
    } extra[7]     = {0};
    size_t n_extra = 0;
    if (c->split) {
        extra[n_extra++] = (typeof(extra[0])) {"attention.key_length_mla", hk};
        extra[n_extra++] = (typeof(extra[0])) {"attention.value_length_mla", c->v_dim};
    }
    if (c->q_rank > 0)
        extra[n_extra++] = (typeof(extra[0])) {"attention.q_lora_rank", c->q_rank};
    if (!c->no_yarn)
        extra[n_extra++] = (typeof(extra[0])) {"rope.scaling.original_context_length", 64};
    if (c->v3) {
        extra[n_extra++] = (typeof(extra[0])) {"expert_gating_func", 2};
        extra[n_extra++] = (typeof(extra[0])) {"expert_group_count", c->n_group};
        extra[n_extra++] = (typeof(extra[0])) {"expert_group_used_count", c->topk_group};
    }
    const struct {
        const char *key;
        float       v;
    } f32[] = {
            {"rope.freq_base", 10000.0f},
            {"attention.layer_norm_rms_epsilon", 1e-6f},
            {"rope.scaling.factor", 40.0f},
            {"rope.scaling.yarn_log_multiplier", 0.0707f},
            {"expert_weights_scale", c->v3 ? c->w_scale : 1.0f},
    };
    const size_t  n_f32  = c->no_yarn ? 2 : sizeof f32 / sizeof f32[0];
    const char    arch[] = "deepseek2";
    char          key[80];
    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, ts.n);
    tf_u64(&o,
           1 + n_u32 + n_extra + n_f32 + (c->no_yarn ? 0 : 1) + (c->v3 ? 1 : 0) +
                   (c->no_yarn && c->v3 ? 1 : 0));
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, arch, strlen(arch));
    for (size_t i = 0; i < n_u32; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, u32[i].key);
        mf_key(&o, key, TF_GGUF_UINT32);
        tf_u32(&o, u32[i].v);
    }
    for (size_t i = 0; i < n_extra; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, extra[i].key);
        mf_key(&o, key, TF_GGUF_UINT32);
        tf_u32(&o, extra[i].v);
    }
    for (size_t i = 0; i < n_f32; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, f32[i].key);
        mf_key(&o, key, TF_GGUF_FLOAT32);
        tf_f32(&o, f32[i].v);
    }
    if (c->no_yarn && c->v3) { /* the scale still applies without YaRN */
        snprintf(key, sizeof key, "%s.expert_weights_scale", arch);
        mf_key(&o, key, TF_GGUF_FLOAT32);
        tf_f32(&o, c->w_scale);
    }
    if (!c->no_yarn) {
        snprintf(key, sizeof key, "%s.rope.scaling.type", arch);
        mf_key(&o, key, TF_GGUF_STRING);
        tf_gstr(&o, "yarn", 4);
    }
    if (c->v3) {
        snprintf(key, sizeof key, "%s.expert_weights_norm", arch);
        mf_key(&o, key, 7 /* bool */);
        tf_put(&o, "\1", 1);
    }
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

/* GLM-4.5 (#242, architecture glm4moe): q/k/v biases, per-head QK norms
 * unless no_qk_norm (the Air has none), NEOX RoPE on the first `rot` dims
 * of each head, the pre-FFN norm named post_attention_norm, `lead` dense
 * layers, then a sigmoid router with exp_probs_b, renormalised and scaled
 * by w_scale, plus `shared` shared experts. One NextN layer follows the
 * stack in block_count, as the converter writes it. F32 weights. */
struct mf_glm4moe {
    uint32_t layers, lead, d_model, heads, kv_heads, head_dim, rot, ff;
    uint32_t experts, used, expert_ff, shared, vocab;
    uint64_t seed;
    bool     no_qk_norm;
    float    w_scale;
};

static inline struct tf_buf mf_glm4moe_gguf(const struct mf_glm4moe *c) {
    const uint32_t    d  = c->d_model;
    const uint32_t    qo = c->heads * c->head_dim, ko = c->kv_heads * c->head_dim;
    struct mf_tensors ts = {0};
    mf_add(&ts, d, c->vocab, MF_RAND, "token_embd.weight");
    mf_add(&ts, d, c->vocab, MF_RAND, "output.weight");
    mf_add(&ts, d, 0, MF_ONE, "output_norm.weight");
    for (uint32_t l = 0; l < c->layers; l++) {
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.attn_norm.weight", l);
        mf_add(&ts, d, qo, MF_RAND, "blk.%u.attn_q.weight", l);
        mf_add(&ts, qo, 0, MF_RAND, "blk.%u.attn_q.bias", l);
        ts.t[ts.n - 1].scale = 5.0f;
        mf_add(&ts, d, ko, MF_RAND, "blk.%u.attn_k.weight", l);
        mf_add(&ts, ko, 0, MF_RAND, "blk.%u.attn_k.bias", l);
        ts.t[ts.n - 1].scale = 5.0f;
        mf_add(&ts, d, ko, MF_RAND, "blk.%u.attn_v.weight", l);
        mf_add(&ts, ko, 0, MF_RAND, "blk.%u.attn_v.bias", l);
        ts.t[ts.n - 1].scale = 5.0f;
        mf_add(&ts, qo, d, MF_RAND, "blk.%u.attn_output.weight", l);
        if (!c->no_qk_norm) {
            mf_add(&ts, c->head_dim, 0, MF_RAMP, "blk.%u.attn_q_norm.weight", l);
            mf_add(&ts, c->head_dim, 0, MF_RAMP, "blk.%u.attn_k_norm.weight", l);
        }
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.post_attention_norm.weight", l);
        if (l < c->lead) {
            mf_add(&ts, d, c->ff, MF_RAND, "blk.%u.ffn_gate.weight", l);
            mf_add(&ts, d, c->ff, MF_RAND, "blk.%u.ffn_up.weight", l);
            mf_add(&ts, c->ff, d, MF_RAND, "blk.%u.ffn_down.weight", l);
            continue;
        }
        mf_add(&ts, d, c->experts, MF_RAND, "blk.%u.ffn_gate_inp.weight", l);
        ts.t[ts.n - 1].scale = 10.0f; /* routing with clear margins */
        mf_add(&ts, c->experts, 0, MF_RAND, "blk.%u.exp_probs_b.bias", l);
        ts.t[ts.n - 1].scale = 10.0f; /* shifts the selection by up to 0.5 */
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_gate_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_up_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, c->expert_ff, d, MF_RAND, "blk.%u.ffn_down_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, d, c->shared * c->expert_ff, MF_RAND, "blk.%u.ffn_gate_shexp.weight", l);
        mf_add(&ts, d, c->shared * c->expert_ff, MF_RAND, "blk.%u.ffn_up_shexp.weight", l);
        mf_add(&ts, c->shared * c->expert_ff, d, MF_RAND, "blk.%u.ffn_down_shexp.weight", l);
    }
    /* The NextN layer's own tensors; the stack must not read them. */
    mf_add(&ts, 2 * d, d, MF_RAND, "blk.%u.nextn.eh_proj.weight", c->layers);
    mf_add(&ts, d, 0, MF_ONE, "blk.%u.nextn.enorm.weight", c->layers);
    mf_add(&ts, d, 0, MF_ONE, "blk.%u.nextn.hnorm.weight", c->layers);
    const struct {
        const char *key;
        uint32_t    v;
    } u32[] = {
            {"block_count", c->layers + 1},
            {"nextn_predict_layers", 1},
            {"embedding_length", d},
            {"attention.head_count", c->heads},
            {"attention.head_count_kv", c->kv_heads},
            {"attention.key_length", c->head_dim},
            {"attention.value_length", c->head_dim},
            {"rope.dimension_count", c->rot},
            {"expert_count", c->experts},
            {"expert_used_count", c->used},
            {"context_length", 4096},
            {"vocab_size", c->vocab},
            {"feed_forward_length", c->ff},
            {"expert_feed_forward_length", c->expert_ff},
            {"expert_shared_count", c->shared},
            {"leading_dense_block_count", c->lead},
            {"expert_gating_func", 2},
    };
    const struct {
        const char *key;
        float       v;
    } f32[] = {
            {"rope.freq_base", 10000.0f},
            {"attention.layer_norm_rms_epsilon", 1e-6f},
            {"expert_weights_scale", c->w_scale},
    };
    const size_t  n_u32 = sizeof u32 / sizeof u32[0], n_f32 = sizeof f32 / sizeof f32[0];
    const char    arch[] = "glm4moe";
    char          key[80];
    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, ts.n);
    tf_u64(&o, 1 + n_u32 + n_f32 + 1);
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, arch, strlen(arch));
    for (size_t i = 0; i < n_u32; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, u32[i].key);
        mf_key(&o, key, TF_GGUF_UINT32);
        tf_u32(&o, u32[i].v);
    }
    for (size_t i = 0; i < n_f32; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, f32[i].key);
        mf_key(&o, key, TF_GGUF_FLOAT32);
        tf_f32(&o, f32[i].v);
    }
    snprintf(key, sizeof key, "%s.expert_weights_norm", arch);
    mf_key(&o, key, 7 /* bool */);
    tf_put(&o, "\1", 1);
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

/* Phi-MoE (#523): LayerNorms with a bias for the attention, FFN and output
 * norms (weights MF_RAMP, so not all ones), q/k/v/o biases, an lm_head bias
 * unless no_lm_bias, GQA with head_dim = d_model / heads (the file has no
 * key_length), and an MoE of `experts` experts, `used` per token, routed by
 * SparseMixer. attention.sliding_window is stated, as real files state it,
 * and not applied. F32 weights, tensors and keys under "phimoe". */
struct mf_phimoe {
    uint32_t layers, d_model, heads, kv_heads, experts, used, expert_ff, vocab;
    uint64_t seed;
    bool     no_lm_bias, longrope;
};

static inline struct tf_buf mf_phimoe_gguf(const struct mf_phimoe *c) {
    const uint32_t    d  = c->d_model;
    const uint32_t    hd = d / c->heads;
    const uint32_t    kv = c->kv_heads * hd;
    struct mf_tensors ts = {0};
    mf_add(&ts, d, c->vocab, MF_RAND, "token_embd.weight");
    mf_add(&ts, d, c->vocab, MF_RAND, "output.weight");
    if (!c->no_lm_bias) {
        mf_add(&ts, c->vocab, 0, MF_RAND, "output.bias");
        ts.t[ts.n - 1].scale = 5.0f;
    }
    mf_add(&ts, d, 0, MF_RAMP, "output_norm.weight");
    mf_add(&ts, d, 0, MF_RAND, "output_norm.bias");
    ts.t[ts.n - 1].scale = 5.0f;
    for (uint32_t l = 0; l < c->layers; l++) {
        const struct {
            const char *name;
            uint32_t    ne0, ne1;
            float       scale; /* 0: MF_RAMP */
        } t[] = {
                {"attn_norm.weight", d, 0, 0.0f},
                {"attn_norm.bias", d, 0, 5.0f},
                {"attn_q.weight", d, d, 1.0f},
                {"attn_q.bias", d, 0, 5.0f},
                {"attn_k.weight", d, kv, 1.0f},
                {"attn_k.bias", kv, 0, 5.0f},
                {"attn_v.weight", d, kv, 1.0f},
                {"attn_v.bias", kv, 0, 5.0f},
                {"attn_output.weight", d, d, 1.0f},
                {"attn_output.bias", d, 0, 5.0f},
                {"ffn_norm.weight", d, 0, 0.0f},
                {"ffn_norm.bias", d, 0, 5.0f},
                {"ffn_gate_inp.weight", d, c->experts, 1.0f},
        };
        for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
            mf_add(&ts,
                   t[i].ne0,
                   t[i].ne1,
                   t[i].scale == 0.0f ? MF_RAMP : MF_RAND,
                   "blk.%u.%s",
                   l,
                   t[i].name);
            ts.t[ts.n - 1].scale = t[i].scale;
        }
        /* Three times the usual range each, so the experts' sum weighs
         * against the biases and the router's weights show in the logits. */
        const struct {
            const char *name;
            uint32_t    ne0, ne1;
        } e[] = {{"ffn_gate_exps", d, c->expert_ff},
                 {"ffn_up_exps", d, c->expert_ff},
                 {"ffn_down_exps", c->expert_ff, d}};
        for (size_t i = 0; i < sizeof e / sizeof e[0]; i++) {
            mf_add(&ts, e[i].ne0, e[i].ne1, MF_RAND, "blk.%u.%s.weight", l, e[i].name);
            ts.t[ts.n - 1].ne2   = c->experts;
            ts.t[ts.n - 1].scale = 3.0f;
        }
        if (c->longrope)
            mf_add(&ts, hd / 2, 0, MF_ONE, "blk.%u.rope_factors_long.weight", l);
    }
    const struct {
        const char *key;
        uint32_t    v;
    } u32[] = {
            {"block_count", c->layers},
            {"embedding_length", d},
            {"attention.head_count", c->heads},
            {"attention.head_count_kv", c->kv_heads},
            {"expert_count", c->experts},
            {"expert_used_count", c->used},
            {"context_length", 4096},
            {"vocab_size", c->vocab},
            {"feed_forward_length", c->expert_ff},
            {"rope.dimension_count", hd},
            {"attention.sliding_window", 7},
    };
    const struct {
        const char *key;
        float       v;
    } f32[] = {
            {"rope.freq_base", 10000.0f},
            {"attention.layer_norm_rms_epsilon", 1e-5f},
    };
    const size_t  n_u32 = sizeof u32 / sizeof u32[0], n_f32 = sizeof f32 / sizeof f32[0];
    const char    arch[] = "phimoe";
    char          key[80];
    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, ts.n);
    tf_u64(&o, 1 + n_u32 + n_f32);
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, arch, strlen(arch));
    for (size_t i = 0; i < n_u32; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, u32[i].key);
        mf_key(&o, key, TF_GGUF_UINT32);
        tf_u32(&o, u32[i].v);
    }
    for (size_t i = 0; i < n_f32; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, f32[i].key);
        mf_key(&o, key, TF_GGUF_FLOAT32);
        tf_f32(&o, f32[i].v);
    }
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

/* Llama 4 (#244): every moe_step-th layer (1-based) a mixture of experts
 * plus a shared expert, the rest dense; rope_freqs.weight factors (1, 1.5,
 * 2, ...); attention.chunk_size / temperature_length shrink llama.cpp's
 * 8192 so a short prompt crosses chunks. F32 weights. */
struct mf_llama4 {
    uint32_t layers, moe_step, d_model, heads, kv_heads, head_dim, ff;
    uint32_t experts, used, expert_ff, vocab, chunk, temp_length;
    uint64_t seed;
};

static inline struct tf_buf mf_llama4_gguf(const struct mf_llama4 *c) {
    const uint32_t    d  = c->d_model;
    struct mf_tensors ts = {0};
    mf_add(&ts, d, c->vocab, MF_RAND, "token_embd.weight");
    mf_add(&ts, d, c->vocab, MF_RAND, "output.weight");
    mf_add(&ts, d, 0, MF_ONE, "output_norm.weight");
    mf_add(&ts, c->head_dim / 2, 0, MF_STEP, "rope_freqs.weight");
    for (uint32_t l = 0; l < c->layers; l++) {
        const bool moe = (l + 1) % c->moe_step == 0;
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.attn_norm.weight", l);
        mf_add(&ts, d, c->heads * c->head_dim, MF_RAND, "blk.%u.attn_q.weight", l);
        mf_add(&ts, d, c->kv_heads * c->head_dim, MF_RAND, "blk.%u.attn_k.weight", l);
        mf_add(&ts, d, c->kv_heads * c->head_dim, MF_RAND, "blk.%u.attn_v.weight", l);
        mf_add(&ts, c->heads * c->head_dim, d, MF_RAND, "blk.%u.attn_output.weight", l);
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.ffn_norm.weight", l);
        if (!moe) {
            mf_add(&ts, d, c->ff, MF_RAND, "blk.%u.ffn_gate.weight", l);
            mf_add(&ts, d, c->ff, MF_RAND, "blk.%u.ffn_up.weight", l);
            mf_add(&ts, c->ff, d, MF_RAND, "blk.%u.ffn_down.weight", l);
            continue;
        }
        mf_add(&ts, d, c->experts, MF_RAND, "blk.%u.ffn_gate_inp.weight", l);
        ts.t[ts.n - 1].scale = 2.0f; /* weights inside sigmoid's bend */
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_gate_exps.weight", l);
        ts.t[ts.n - 1].scale = 4.0f; /* silu well past its linear part */
        ts.t[ts.n - 1].ne2   = c->experts;
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_up_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, c->expert_ff, d, MF_RAND, "blk.%u.ffn_down_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_gate_shexp.weight", l);
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_up_shexp.weight", l);
        mf_add(&ts, c->expert_ff, d, MF_RAND, "blk.%u.ffn_down_shexp.weight", l);
    }
    const struct {
        const char *key;
        uint32_t    v;
    } u32[] = {
            {"block_count", c->layers},
            {"embedding_length", d},
            {"attention.head_count", c->heads},
            {"attention.head_count_kv", c->kv_heads},
            {"expert_count", c->experts},
            {"expert_used_count", c->used},
            {"context_length", 4096},
            {"vocab_size", c->vocab},
            {"feed_forward_length", c->ff},
            {"expert_feed_forward_length", c->expert_ff},
            {"interleave_moe_layer_step", c->moe_step},
            {"attention.key_length", c->head_dim},
            {"rope.dimension_count", c->head_dim},
            {"attention.chunk_size", c->chunk},
            {"attention.temperature_length", c->temp_length},
    };
    const size_t  n_u32  = sizeof u32 / sizeof u32[0];
    const char    arch[] = "llama4";
    char          key[80];
    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, ts.n);
    tf_u64(&o, 1 + n_u32 + 2);
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, arch, strlen(arch));
    for (size_t i = 0; i < n_u32; i++) {
        snprintf(key, sizeof key, "%s.%s", arch, u32[i].key);
        mf_key(&o, key, TF_GGUF_UINT32);
        tf_u32(&o, u32[i].v);
    }
    snprintf(key, sizeof key, "%s.rope.freq_base", arch);
    mf_key(&o, key, TF_GGUF_FLOAT32);
    tf_f32(&o, 500000.0f);
    snprintf(key, sizeof key, "%s.attention.layer_norm_rms_epsilon", arch);
    mf_key(&o, key, TF_GGUF_FLOAT32);
    tf_f32(&o, 1e-5f);
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

#endif /* GEIST_TESTS_MODEL_FIXTURES_H */
