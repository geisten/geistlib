/*
 * src/archs/transformer/arch.c — transformer decoder arch_ops impl.
 *
 * Layer: ARCHITECTURE.
 *
 * Thin thunks between the engine's geist_arch_ops_decoder vtable and the
 * internal entry points in arch_ops.c / forward/. Model-level ops take
 * the opaque arch_state; per-session ops take the opaque session handle
 * (a struct transformer_arch_session) — the engine always passes a real
 * session for this arch since it implements session_alloc.
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

/* Mutable view of the ZO-tuning gains. Fail-closed on two counts:
 *   - built without GEIST_TUNE  → st->gains is null, nothing to hand out;
 *   - backend runs the fused tensor path (GPU) → that path returns before
 *     the dispatcher applies gains, so handing out a writable array would
 *     silently produce ungained output. Refuse instead. */
static enum geist_status op_gains(void *arch_state, float **out, size_t *n) {
    struct transformer_arch_state *st = arch_state;
    if (st == nullptr || out == nullptr || n == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    *n   = 0;
    if (st->gains == nullptr) {
        geist_backend_set_error(
                st->backend, GEIST_E_UNSUPPORTED, "gains: build geist with -DGEIST_TUNE");
        return GEIST_E_UNSUPPORTED;
    }
    if (geist_backend_fused_tbl(st->backend)->linear_t != nullptr) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "gains: backend '%s' uses the fused tensor "
                                "linear path, which bypasses the gain apply",
                                geist_backend_name(st->backend));
        return GEIST_E_UNSUPPORTED;
    }
    *out = st->gains;
    *n   = st->n_gains;
    return GEIST_OK;
}

static void op_state_reset(void *session) {
    if (session != nullptr) {
        transformer_session_reset(session);
    }
}

static enum geist_status op_set_session_opts(void *session, const struct geist_session_opts *opts) {
    if (session == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    return transformer_session_apply_opts(session, opts);
}

/* Append a sequence of tokens to the KV cache via the batched seq>1
 * path. After the call, the session holds logits for the next position;
 * ops->decode_step will return the argmax of those logits on its first
 * invocation. */
static enum geist_status op_prefill(void *session, size_t n, const geist_token_t ids[static n]) {
    struct transformer_arch_session *sess = session;
    if (sess == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n == 0) {
        return GEIST_OK; /* no-op */
    }
    return transformer_prefill_text_batch(sess, n, ids);
}

/* Append `n` audio soft-tokens to the KV cache via the batched seq>1
 * path. Each soft-token is a HIDDEN-dim FP32 vector produced upstream
 * by the audio encoder. */
static enum geist_status op_prefill_audio(void *session, size_t n, const float *soft_tokens) {
    struct transformer_arch_session *sess = session;
    if (sess == nullptr || soft_tokens == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n == 0) {
        return GEIST_OK; /* no-op */
    }
    return transformer_prefill_audio_batch(sess, n, soft_tokens);
}

/* Vision soft-tokens follow the same wire format as audio (1536-dim
 * fp32 per token). The transformer side just memcpys them into the
 * residual stream and runs the layer loop — no embedding lookup, no
 * scale. Delegate to the audio prefill batch path. */
static enum geist_status op_prefill_image(void *session, size_t n, const float *soft_tokens) {
    struct transformer_arch_session *sess = session;
    if (sess == nullptr || soft_tokens == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n == 0) {
        return GEIST_OK; /* no-op */
    }
    return transformer_prefill_audio_batch(sess, n, soft_tokens);
}

/* Pin `n` prefix tokens into the KV cache. Truncates cache, prefills the
 * prefix once, snapshots the resulting kv_len as the reset target.
 * Subsequent state_reset calls truncate kv_len back to the pinned length,
 * keeping the prefix's KV state across conversation turns. */
static enum geist_status op_pin_prefix(void *session, size_t n, const geist_token_t ids[static n]) {
    struct transformer_arch_session *sess = session;
    if (sess == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    return transformer_pin_prefix(sess, n, ids);
}

/* Greedy one-token autoregressive step. Returns the prediction computed
 * by the prior prefill/decode call, then advances the KV cache with that
 * prediction so the next call's pending value is the prediction for the
 * following position. Mirrors lm.c::lm_decode_step's "return-then-advance"
 * cadence. */
static enum geist_status op_decode_step(void *session, geist_token_t *out) {
    struct transformer_arch_session *sess = session;
    if (sess == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (sess->model != nullptr && geist_pooling_is_embedding(sess->model->config.pooling)) {
        /* Distinct from the "prefill first" case below: prefill DID run,
         * there is simply no token to emit. Saying so beats a state error
         * that sends the caller looking for a missing prefill. */
        return GEIST_E_UNSUPPORTED;
    }
    if (!sess->logits_valid) {
        return GEIST_E_INVALID_STATE; /* nothing pending — prefill first */
    }
    const geist_token_t     prev    = sess->next_token_pending;
    geist_token_t           scratch = -1;
    const enum geist_status s       = transformer_decode_step(sess, prev, &scratch);
    if (s != GEIST_OK) {
        return s;
    }
    *out = prev;
    return GEIST_OK;
}

static size_t op_hidden_dim(const void *arch_state) {
    const struct transformer_arch_state *st = arch_state;
    return st != nullptr ? st->d_model : 0;
}

static geist_token_t op_peek_next_token(void *session) {
    const struct transformer_arch_session *sess = session;
    if (sess == nullptr || !sess->logits_valid)
        return -1;
    return sess->next_token_pending;
}

/* Returns a pointer to the pending next-position logits and the vocab
 * size in *n_logits. nullptr (with *n_logits=0) if no logits are pending.
 * CPU-only: pointer aliases the backend buffer and is valid until the
 * next mutating call on the session. */
static const float *op_peek_logits(void *session, size_t *n_logits) {
    struct transformer_arch_session *sess = session;
    if (n_logits == nullptr)
        return nullptr;
    if (sess == nullptr || !sess->logits_valid || sess->scratch_logits == nullptr) {
        *n_logits = 0;
        return nullptr;
    }
    struct transformer_arch_state *st = sess->model;
    /* The spec-head fast path leaves scratch_logits SPARSE (-inf off its
     * top-K) — right for greedy, wrong for value consumers. Recompute the
     * dense head lazily, once, from the hidden still in scratch_h_a. */
    if (sess->logits_sparse && transformer_head_dense_recompute(sess) != GEIST_OK) {
        *n_logits = 0;
        return nullptr;
    }
    *n_logits = (size_t) st->vocab_size;
    float *p  = (float *) st->backend->desc->vtbl->buffer_map(sess->scratch_logits);
    /* The greedy argmax path skips the Gemma final-logit softcap (monotonic,
     * so the argmax is invariant and it saves ~262k tanhf/token on decode).
     * peek_logits exposes the VALUES to scoring/perplexity consumers, which
     * need the model-conformant softcapped logits — apply it lazily, once. */
    if (p != nullptr && st->config.logit_softcap > 0.0f && !sess->logits_softcapped) {
        const float c = st->config.logit_softcap;
        for (size_t i = 0; i < (size_t) st->vocab_size; i++) {
            p[i] = tanhf(p[i] / c) * c;
        }
        sess->logits_softcapped = true;
    }
    return (const float *) p;
}

/* Returns the pooled, L2-normalised sentence embedding and its dimension
 * in *n_dims. nullptr (with *n_dims=0) on a generative model, or before a
 * prefill has produced one. The pointer aliases session scratch and is
 * valid until the next mutating call on the session — same contract as
 * op_peek_logits. */
static const float *op_peek_embedding(void *session, size_t *n_dims) {
    struct transformer_arch_session *sess = session;
    if (n_dims == nullptr) {
        return nullptr;
    }
    *n_dims = 0;
    if (sess == nullptr || !sess->embedding_valid || sess->scratch_h_a == nullptr) {
        return nullptr;
    }
    struct transformer_arch_state *st = sess->model;
    const float *p = (const float *) st->backend->desc->vtbl->buffer_map(sess->scratch_h_a);
    if (p == nullptr) {
        return nullptr;
    }
    *n_dims = (size_t) st->d_model;
    return p;
}

static enum geist_status op_verify_forward(void               *session,
                                           size_t              k,
                                           const geist_token_t ids[static k],
                                           geist_token_t       out_tokens[static k]) {
    return transformer_verify_forward(
            (struct transformer_arch_session *) session, k, ids, out_tokens);
}

static enum geist_status op_draft_tokens(
        void *session, size_t k_max, geist_token_t seed, geist_token_t *out_tokens, size_t *n_out) {
    return transformer_mtp_draft(
            (struct transformer_arch_session *) session, k_max, seed, out_tokens, n_out);
}

static enum geist_status op_kv_truncate(void *session, size_t new_len) {
    return transformer_kv_truncate((struct transformer_arch_session *) session, new_len);
}

static size_t op_kv_len(const void *session) {
    const struct transformer_arch_session *sess = session;
    return sess != nullptr ? sess->kv_len : 0;
}

/* ---- Session lifecycle vtable hooks (P1.2.f) -------------------------- */

static void *op_session_alloc(void *arch_state, const struct geist_session_opts *opts) {
    return transformer_session_alloc(arch_state, opts);
}

static void op_session_free(void *arch_state, void *session) {
    transformer_session_free(arch_state, session);
}

const struct geist_arch_ops_decoder geist_arch_transformer = {
        .name                     = "transformer",
        .state_create             = op_state_create,
        .state_create_from_memory = op_state_create_from_memory,
        .state_destroy            = op_state_destroy,
        .gains                    = op_gains,
        .state_reset              = op_state_reset,
        .set_session_opts         = op_set_session_opts,
        .prefill                  = op_prefill,
        .decode_step              = op_decode_step,
        .pin_prefix               = op_pin_prefix,
        .prefill_audio            = op_prefill_audio,
        .prefill_image            = op_prefill_image,
        .peek_logits              = op_peek_logits,
        .peek_embedding           = op_peek_embedding,
        .hidden_dim               = op_hidden_dim,
        .peek_next_token          = op_peek_next_token,
        .draft_tokens             = op_draft_tokens,
        .verify_forward           = op_verify_forward,
        .kv_truncate              = op_kv_truncate,
        .kv_len                   = op_kv_len,
        .session_alloc            = op_session_alloc,
        .session_free             = op_session_free,
};
