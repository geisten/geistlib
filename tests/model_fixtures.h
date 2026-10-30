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
    /* ffn_down stored as Q5_0 (all-zero blocks; ffn a multiple of 32), a
     * dtype the CPU backends run and metal does not. */
    bool q5_0_down;
    /* A separate output.weight (lm_head), so token_embd is a lookup-only
     * table; tied otherwise. */
    bool untied;
    /* tokenizer.chat_template when set; its length in chat_template_len, so
     * a template may contain NUL bytes (0: strlen). */
    const char *chat_template;
    size_t      chat_template_len;
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

/* MF_RAMP: element i is mf_ramp(i), mixed signs, no draw from the seed. */
enum mf_fill { MF_RAND, MF_ONE, MF_NEG_ONE, MF_ZERO, MF_RAMP };

static inline float mf_ramp(uint64_t i) {
    return ((float) (i % 7) - 3.0f) * 0.25f;
}

struct mf_tensor {
    char         name[48];
    uint64_t     ne0, ne1; /* ne1 0: one dimension */
    uint64_t     ne2;      /* 0: at most two dimensions */
    enum mf_fill fill;
    float        scale; /* MF_RAND values times this (mf_add: 1) */
    bool         f16;   /* stored as F16; F32 otherwise */
    bool         q5_0;  /* stored as zero Q5_0 blocks (ne0 a multiple of 32) */
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
    t->ne0   = ne0;
    t->ne1   = ne1;
    t->ne2   = 0;
    t->fill  = fill;
    t->scale = 1.0f;
    t->f16   = false;
    t->q5_0  = false;
}

static inline uint64_t mf_elems(const struct mf_tensor *t) {
    return t->ne0 * (t->ne1 != 0 ? t->ne1 : 1) * (t->ne2 != 0 ? t->ne2 : 1);
}

/* Bytes of a tensor's data: F32, F16, or 22 per 32 elements for Q5_0. */
static inline uint64_t mf_bytes(const struct mf_tensor *t) {
    return t->q5_0 ? mf_elems(t) / 32 * 22 : mf_elems(t) * (t->f16 ? 2 : 4);
}

/* The tensor infos, then the data, after the metadata: F32 (F16 or zero
 * Q5_0 blocks where the tensor says so), each tensor 32-byte aligned,
 * MF_RAND values drawn in tensor order from `seed`. Frees the list. */
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
        tf_u32(o, t->q5_0 ? 6 : t->f16 ? 1 : 0); /* Q5_0 : F16 : F32 */
        tf_u64(o, off);
        off += (mf_bytes(t) + 31) / 32 * 32;
    }
    uint64_t s = seed;
    for (size_t i = 0; i < ts->n; i++) {
        const struct mf_tensor *t = &ts->t[i];
        while (o->n % 32 != 0) {
            tf_put(o, "", 1);
        }
        if (t->q5_0) {
            for (uint64_t b = 0; b < mf_bytes(t); b++) {
                tf_put(o, "", 1);
            }
            continue;
        }
        const float c = t->fill == MF_ONE ? 1.0f : t->fill == MF_NEG_ONE ? -1.0f : 0.0f;
        for (uint64_t e = 0; e < mf_elems(t); e++) {
            const float v = t->fill == MF_RAND   ? mf_weight(&s) * t->scale
                            : t->fill == MF_RAMP ? mf_ramp(e)
                                                 : c;
            if (t->f16) {
                const _Float16 h = (_Float16) v;
                tf_put(o, &h, 2);
            } else {
                tf_f32(o, v);
            }
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
        ts.t[ts.n - 1].q5_0 = c->q5_0_down;
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
 * `shared_expert` adds a shared-expert router the family does not run. */
struct mf_qwen3moe {
    uint32_t    layers, d_model, heads, kv_heads, head_dim, experts, used, expert_ff, vocab;
    uint64_t    seed;
    const char *drop;
    uint32_t    gating;
    bool        shared_expert;
};

static inline struct tf_buf mf_qwen3moe_gguf(const struct mf_qwen3moe *c) {
    const uint32_t    d  = c->d_model;
    struct mf_tensors ts = {0};
    mf_add(&ts, d, c->vocab, MF_RAND, "token_embd.weight");
    mf_add(&ts, d, 0, MF_ONE, "output_norm.weight");
    for (uint32_t l = 0; l < c->layers; l++) {
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.attn_norm.weight", l);
        mf_add(&ts, d, c->heads * c->head_dim, MF_RAND, "blk.%u.attn_q.weight", l);
        mf_add(&ts, d, c->kv_heads * c->head_dim, MF_RAND, "blk.%u.attn_k.weight", l);
        mf_add(&ts, d, c->kv_heads * c->head_dim, MF_RAND, "blk.%u.attn_v.weight", l);
        mf_add(&ts, c->heads * c->head_dim, d, MF_RAND, "blk.%u.attn_output.weight", l);
        mf_add(&ts, c->head_dim, 0, MF_ONE, "blk.%u.attn_q_norm.weight", l);
        mf_add(&ts, c->head_dim, 0, MF_ONE, "blk.%u.attn_k_norm.weight", l);
        mf_add(&ts, d, 0, MF_ONE, "blk.%u.ffn_norm.weight", l);
        mf_add(&ts, d, c->experts, MF_RAND, "blk.%u.ffn_gate_inp.weight", l);
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_gate_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, d, c->expert_ff, MF_RAND, "blk.%u.ffn_up_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
        mf_add(&ts, c->expert_ff, d, MF_RAND, "blk.%u.ffn_down_exps.weight", l);
        ts.t[ts.n - 1].ne2 = c->experts;
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
            {"qwen3moe.block_count", c->layers},
            {"qwen3moe.embedding_length", d},
            {"qwen3moe.attention.head_count", c->heads},
            {"qwen3moe.attention.head_count_kv", c->kv_heads},
            {"qwen3moe.attention.key_length", c->head_dim},
            {"qwen3moe.expert_count", c->experts},
            {"qwen3moe.expert_used_count", c->used},
            {"qwen3moe.expert_feed_forward_length", c->expert_ff},
            {"qwen3moe.context_length", 4096},
            {"qwen3moe.vocab_size", c->vocab},
            {"qwen3moe.expert_gating_func", c->gating},
    };
    const size_t n_u32 = sizeof u32 / sizeof u32[0] - (c->gating == 0 ? 1 : 0);

    struct tf_buf o = {0};
    tf_put(&o, "GGUF", 4);
    tf_u32(&o, 3);
    tf_u64(&o, ts.n);
    tf_u64(&o, 1 + n_u32 + 2);
    mf_key(&o, "general.architecture", TF_GGUF_STRING);
    tf_gstr(&o, "qwen3moe", 8);
    for (size_t i = 0; i < n_u32; i++) {
        mf_key(&o, u32[i].key, TF_GGUF_UINT32);
        tf_u32(&o, u32[i].v);
    }
    mf_key(&o, "qwen3moe.rope.freq_base", TF_GGUF_FLOAT32);
    tf_f32(&o, 1000000.0f);
    mf_key(&o, "qwen3moe.attention.layer_norm_rms_epsilon", TF_GGUF_FLOAT32);
    tf_f32(&o, 1e-6f);
    mf_write_tensors(&o, &ts, c->seed);
    return o;
}

#endif /* GEIST_TESTS_MODEL_FIXTURES_H */
