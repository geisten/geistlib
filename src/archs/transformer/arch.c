/*
 * src/archs/transformer/arch.c — transformer decoder arch_ops impl.
 *
 * Layer: ARCHITECTURE.
 *
 * Phase B-4e production swap (this file): the arch_state is the v2 state
 * built in arch_state.{h,c}. Every op routes through backend->vtbl->*
 * via transformer_decode_step. The legacy `LM*` delegation is gone;
 * the lm.c module is now only used by tests in tests/legacy/.
 *
 * Phase B-4f: prefill_audio and pin_prefix wired through v2. peek_logits
 * wired through to scratch_logits for the public session_peek_logits API.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch.h"
#include "arch_state.h"
#include "forward.h"

#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stddef.h>

static void *op_state_create(struct geist_backend            *be,
                             const char                      *gguf_path,
                             const struct geist_session_opts *opts) {
    if (be == nullptr || gguf_path == nullptr) {
        return nullptr;
    }
    struct transformer_arch_state *st = nullptr;
    enum geist_status              s  = transformer_state_create(be, gguf_path, opts, &st);
    if (s != GEIST_OK) {
        return nullptr;
    }
    return st;
}

static void *op_state_create_from_memory(struct geist_backend            *be,
                                         const void                      *data,
                                         size_t                           size,
                                         const struct geist_session_opts *opts) {
    if (be == nullptr || data == nullptr) {
        return nullptr;
    }
    struct transformer_arch_state *st = nullptr;
    enum geist_status s = transformer_state_create_from_memory(be, data, size, opts, &st);
    if (s != GEIST_OK) {
        return nullptr;
    }
    return st;
}

static void op_state_destroy(void *arch_state) {
    transformer_state_destroy(arch_state);
}

static void op_state_reset(void *arch_state) {
    transformer_state_reset(arch_state);
}

static void op_set_session_opts(void *arch_state, const struct geist_session_opts *opts) {
    transformer_state_apply_opts(arch_state, opts);
}

/* Append a sequence of tokens to the KV cache via the batched seq>1
 * path. After the call, arch_state holds logits for the next position;
 * ops->decode_step will return the argmax of those logits on its first
 * invocation. */
static void op_prefill(void *arch_state, size_t n, const geist_token_t ids[static n]) {
    struct transformer_arch_state *st = arch_state;
    if (st == nullptr || n == 0) {
        return;
    }
    (void) transformer_prefill_text_batch(st, n, ids);
    /* On failure kv_len doesn't advance by n; the engine detects the
     * shortfall and surfaces the backend's recorded error. */
}

/* Append `n` audio soft-tokens to the KV cache via the batched seq>1
 * path. Each soft-token is a HIDDEN-dim FP32 vector produced upstream
 * by the audio encoder. */
static void op_prefill_audio(void *arch_state, size_t n, const float *soft_tokens) {
    struct transformer_arch_state *st = arch_state;
    if (st == nullptr || soft_tokens == nullptr || n == 0) {
        return;
    }
    (void) transformer_prefill_audio_batch(st, n, soft_tokens);
}

/* Vision soft-tokens follow the same wire format as audio (1536-dim
 * fp32 per token). The transformer side just memcpys them into the
 * residual stream and runs the layer loop — no embedding lookup, no
 * scale. Delegate to the audio prefill batch path. */
static void op_prefill_image(void *arch_state, size_t n, const float *soft_tokens) {
    struct transformer_arch_state *st = arch_state;
    if (st == nullptr || soft_tokens == nullptr || n == 0) {
        return;
    }
    (void) transformer_prefill_audio_batch(st, n, soft_tokens);
}

/* Pin `n` prefix tokens into the KV cache. Truncates cache, prefills the
 * prefix once, snapshots the resulting kv_len as the reset target.
 * Subsequent state_reset calls truncate kv_len back to the pinned length,
 * keeping the prefix's KV state across conversation turns. */
static void op_pin_prefix(void *arch_state, size_t n, const geist_token_t ids[static n]) {
    struct transformer_arch_state *st = arch_state;
    if (st == nullptr) {
        return;
    }
    (void) transformer_pin_prefix(st, n, ids);
}

/* Greedy one-token autoregressive step. Returns the prediction computed
 * by the prior prefill/decode call, then advances the KV cache with that
 * prediction so the next call's pending value is the prediction for the
 * following position. Mirrors lm.c::lm_decode_step's "return-then-advance"
 * cadence. */
static geist_token_t op_decode_step(void *arch_state) {
    struct transformer_arch_state *st = arch_state;
    if (st == nullptr || !st->sess->logits_valid) {
        return -1;
    }
    const geist_token_t prev    = st->sess->next_token_pending;
    geist_token_t       scratch = -1;
    if (transformer_decode_step(st, prev, &scratch) != GEIST_OK) {
        return -1;
    }
    return prev;
}

static geist_token_t op_peek_next_token(void *arch_state) {
    const struct transformer_arch_state *st = arch_state;
    if (st == nullptr || !st->sess->logits_valid)
        return -1;
    return st->sess->next_token_pending;
}

/* Returns a pointer to the pending next-position logits and the vocab
 * size in *n_logits. nullptr (with *n_logits=0) if no logits are pending.
 * CPU-only: pointer aliases the backend buffer and is valid until the
 * next mutating call on the session. */
static const float *op_peek_logits(void *arch_state, size_t *n_logits) {
    struct transformer_arch_state *st = arch_state;
    if (n_logits == nullptr)
        return nullptr;
    if (st == nullptr || !st->sess->logits_valid || st->sess->scratch_logits == nullptr) {
        *n_logits = 0;
        return nullptr;
    }
    /* The spec-head fast path leaves scratch_logits SPARSE (-inf off its
     * top-K) — right for greedy, wrong for value consumers. Recompute the
     * dense head lazily, once, from the hidden still in scratch_h_a. */
    if (st->sess->logits_sparse && transformer_head_dense_recompute(st) != GEIST_OK) {
        *n_logits = 0;
        return nullptr;
    }
    *n_logits = (size_t) st->vocab_size;
    float *p  = (float *) st->backend->desc->vtbl->buffer_map(st->sess->scratch_logits);
    /* The greedy argmax path skips the Gemma final-logit softcap (monotonic,
     * so the argmax is invariant and it saves ~262k tanhf/token on decode).
     * peek_logits exposes the VALUES to scoring/perplexity consumers, which
     * need the model-conformant softcapped logits — apply it lazily, once. */
    if (p != nullptr && st->config.logit_softcap > 0.0f && !st->sess->logits_softcapped) {
        const float c = st->config.logit_softcap;
        for (size_t i = 0; i < (size_t) st->vocab_size; i++) {
            p[i] = tanhf(p[i] / c) * c;
        }
        st->sess->logits_softcapped = true;
    }
    return (const float *) p;
}

static enum geist_status op_verify_forward(void               *arch_state,
                                           size_t              k,
                                           const geist_token_t ids[static k],
                                           geist_token_t       out_tokens[static k]) {
    return transformer_verify_forward(
            (struct transformer_arch_state *) arch_state, k, ids, out_tokens);
}

static void op_kv_truncate(void *arch_state, size_t new_len) {
    transformer_kv_truncate((struct transformer_arch_state *) arch_state, new_len);
}

static size_t op_kv_len(const void *arch_state) {
    const struct transformer_arch_state *st = arch_state;
    return st != nullptr ? st->sess->kv_len : 0;
}

/* ---- Multi-session vtable hooks (P1.2.f) ------------------------------ */

static void *op_session_alloc(void *arch_state, const struct geist_session_opts *opts) {
    return transformer_session_alloc(arch_state, opts);
}

static void op_session_free(void *arch_state, void *session_meta) {
    transformer_session_free(arch_state, session_meta);
}

static void op_session_attach(void *arch_state, void *session_meta) {
    transformer_session_attach(arch_state, session_meta);
}

const struct geist_arch_ops_decoder geist_arch_transformer = {
        .name                     = "transformer",
        .state_create             = op_state_create,
        .state_create_from_memory = op_state_create_from_memory,
        .state_destroy            = op_state_destroy,
        .state_reset              = op_state_reset,
        .set_session_opts         = op_set_session_opts,
        .prefill                  = op_prefill,
        .decode_step              = op_decode_step,
        .pin_prefix               = op_pin_prefix,
        .prefill_audio            = op_prefill_audio,
        .prefill_image            = op_prefill_image,
        .peek_logits              = op_peek_logits,
        .peek_next_token          = op_peek_next_token,
        .verify_forward           = op_verify_forward,
        .kv_truncate              = op_kv_truncate,
        .kv_len                   = op_kv_len,
        .session_alloc            = op_session_alloc,
        .session_free             = op_session_free,
        .session_attach           = op_session_attach,
};
