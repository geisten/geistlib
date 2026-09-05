/*
 * src/engine/session.c — geist_session lifecycle + decode dispatch.
 *
 * Layer: ENGINE.
 *
 * Each session holds a reference to the model and dispatches set_prompt,
 * prefill, decode_step, and attach_audio through the model's arch_ops
 * vtables. The arch-specific state is owned by geist_model (1:1 for now;
 * multi-session model sharing comes when the arch_state distinguishes
 * per-session KV from per-model weights).
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "model.h"
#include "error.h"

#include <geist_arch.h> /* arch_ops vtables the engine dispatches through */

#include <stdatomic.h>

#include "checked.h"
#include "heap.h"
#include "image_pipeline.h"
#include "sp_bpe_tokenizer.h"
#include "gguf_tokenizer.h"

#include <geist.h>
#include <geist_util.h> /* tokenize/prefill/attach/peek/speculative/stats moved here in 0.2.0 */

#include <stdalign.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct geist_session_full {
    struct geist_model   *model;
    struct geist_backend *backend;

    /* P1.2.f: per-session arch state. NULL when the architecture doesn't
     * implement session_alloc — engine falls back to using the model's
     * default session (single-session-per-model semantics). */
    void *arch_session;

    /* Sampler defaults; per-call overrides come in Phase B-4b. */
    float    temperature;
    float    top_p;
    int      top_k;
    uint64_t random_seed;

    enum geist_status err_code;
    char              err_msg[512];

    /* Counters for geist_session_get_stats. Monotonic-ns timing uses
     * clock_gettime(CLOCK_MONOTONIC); deltas are accumulated at each
     * top-level entry point (decode_step, prefill_tokens, attach_audio,
     * decode_speculative). */
    uint64_t n_tokens_decoded;
    uint64_t total_decode_ns;
    uint64_t total_prefill_ns;
    uint64_t total_audio_encode_ns;

    /* Streaming audio turn (#256). samples is _Atomic because push may
     * run on a capture thread while end reads it on the session thread.
     * buf/injected are session-thread-only (poll/end). */
    bool           audio_streaming;
    _Atomic size_t audio_stream_samples;
    float         *audio_stream_buf; /* bound-sized scratch, alive begin→end */
    size_t         audio_stream_cap; /* tokens the scratch can hold */
    size_t         audio_stream_injected;
};

static inline uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

/* Engine-internal: cast the public opaque handle to the full struct. */
static inline struct geist_session_full *as_full(struct geist_session *s) {
    return (struct geist_session_full *) s;
}

/* The session handle passed to per-session vtable ops. For architectures
 * with session_alloc this is the arch session this geist_session owns;
 * for single-session archs (no session_alloc) it is the arch_state
 * itself — such an arch's model IS its one session. */
static inline void *arch_sess(const struct geist_session_full *sf) {
    return sf->arch_session != nullptr ? sf->arch_session : sf->model->text_decoder.arch_meta;
}

/* Record a failed per-session dispatch. The op's status return IS the
 * control flow; the backend error slot only contributes detail text for
 * the session's error message. */
static enum geist_status
session_op_failed(struct geist_session_full *sf, enum geist_status s, const char *what) {
    sf->err_code = s;
    /* The backend's message belongs to THIS failure only if its sticky code
     * still matches. Optional probes (a missing untied output.weight at
     * load) leave a stale message behind, and reporting it verbatim points
     * every debugger at the wrong tensor. */
    const bool fresh = geist_backend_errcode(sf->backend) == s;
    snprintf(sf->err_msg,
             sizeof(sf->err_msg),
             "%s failed (status %d)%s%s",
             what,
             (int) s,
             fresh ? ": " : " [backend set no message; stale: ",
             geist_backend_errmsg(sf->backend));
    if (!fresh) {
        const size_t n = strlen(sf->err_msg);
        if (n + 2 <= sizeof(sf->err_msg)) {
            snprintf(sf->err_msg + n, sizeof(sf->err_msg) - n, "]");
        }
    }
    return s;
}

[[nodiscard]] enum geist_status geist_session_create(struct geist_model              *m,
                                                     struct geist_backend            *be,
                                                     const struct geist_session_opts *opts,
                                                     struct geist_session           **out) {
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (m == nullptr || be == nullptr) {
        geist_error_set_create_time(
                GEIST_E_INVALID_ARG, "geist_session_create", "model or backend is null");
        return GEIST_E_INVALID_ARG;
    }

    /* `be` is a STABLE-signature witness: it must be the backend the
     * model was loaded on. A session on a different backend instance
     * would mix buffer ownership and per-backend workspaces. */
    if (be != m->backend) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG,
                                    "geist_session_create",
                                    "backend does not match the one the model was loaded on "
                                    "(model: %s, given: %s)",
                                    m->backend != nullptr ? geist_backend_name(m->backend)
                                                          : "(null)",
                                    geist_backend_name(be));
        return GEIST_E_INVALID_ARG;
    }

    struct geist_session_full *sf =
            heap_alloc_aligned(sizeof(*sf), alignof(struct geist_session_full));
    if (sf == nullptr) {
        geist_error_set_create_time(
                GEIST_E_OOM, "geist_session_create", "failed to allocate session");
        return GEIST_E_OOM;
    }
    *sf = (struct geist_session_full) {
            .model                 = m,
            .backend               = be,
            .arch_session          = nullptr,
            .temperature           = opts != nullptr ? opts->temperature : 0.0f,
            .top_p                 = opts != nullptr ? opts->top_p : 1.0f,
            .top_k                 = opts != nullptr ? opts->top_k : 0,
            .random_seed           = opts != nullptr ? opts->random_seed : 0,
            .err_code              = GEIST_OK,
            .n_tokens_decoded      = 0,
            .total_decode_ns       = 0,
            .total_prefill_ns      = 0,
            .total_audio_encode_ns = 0,
    };
    sf->err_msg[0] = '\0';

    /* P1.2.f: allocate a per-session arch state if the architecture
     * supports it. This gives the session its own KV cache + scratch
     * pool + sampler RNG, separate from any other session on the same
     * model. session_alloc already calls apply_opts inside, so the
     * session-level temperature / top_p / top_k / seed are wired up.
     *
     * Architectures without session_alloc stay single-session-per-model
     * — the engine passes the arch_state as the session handle on every
     * dispatch (such an arch's model IS its one session), and
     * set_session_opts is the only path for sampler config. */
    const struct geist_arch_ops_decoder *ops = m->text_decoder.arch_ops;
    if (ops != nullptr && ops->session_alloc != nullptr) {
        sf->arch_session = ops->session_alloc(m->text_decoder.arch_meta, opts);
        if (sf->arch_session == nullptr) {
            geist_error_set_create_time(
                    GEIST_E_OOM, "geist_session_create", "arch session_alloc returned null");
            void *tmp = sf;
            safe_free(&tmp);
            return GEIST_E_OOM;
        }
    } else if (opts != nullptr && ops != nullptr && ops->set_session_opts != nullptr) {
        /* Legacy single-session path: push opts onto the arch's one
         * session (= its arch_state). When multiple legacy sessions
         * share one model, the last set_session_opts call wins. */
        const enum geist_status os = ops->set_session_opts(m->text_decoder.arch_meta, opts);
        if (os != GEIST_OK) {
            geist_error_set_create_time(
                    GEIST_E_OOM, "geist_session_create", "set_session_opts failed");
            void *tmp = sf;
            safe_free(&tmp);
            return os;
        }
    }

    *out = (struct geist_session *) sf;
    return GEIST_OK;
}

void *geist_session_internal_arch_session(struct geist_session *s) {
    return s != nullptr ? arch_sess(as_full(s)) : nullptr;
}

/* Close an open streaming audio turn and release its scratch, without
 * finishing it. Idempotent: safe when no turn is open and safe to call
 * twice, so destroy can call it unconditionally.
 *
 * Both halves matter. The scratch is a session-lifetime allocation that
 * only audio_end used to free, so a session destroyed mid-turn leaked it.
 * And the encoder stream is state inside the ENCODER, shared across
 * sessions of the same model: leaving it open outlives the session that
 * opened it. */
static void session_audio_stream_release(struct geist_session_full *sf) {
    if (sf->audio_streaming) {
        const struct geist_arch_ops_encoder *enc_ops = sf->model->audio_encoder.arch_ops;
        void                                *enc_st  = sf->model->audio_encoder.arch_meta;
        if (enc_ops != nullptr && enc_st != nullptr && enc_ops->stream_abort != nullptr) {
            enc_ops->stream_abort(enc_st);
        }
        sf->audio_streaming = false;
    }
    safe_free((void **) &sf->audio_stream_buf);
    sf->audio_stream_cap      = 0;
    sf->audio_stream_injected = 0;
}

void geist_session_destroy(struct geist_session *s) {
    if (s == nullptr) {
        return;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    /* A turn still open at destroy is aborted, not abandoned. */
    session_audio_stream_release(sf);
    /* P1.2.f: release this session's per-session arch state if it owns
     * one. */
    if (sf->arch_session != nullptr && ops != nullptr && ops->session_free != nullptr) {
        ops->session_free(sf->model->text_decoder.arch_meta, sf->arch_session);
        sf->arch_session = nullptr;
    }
    /* Weights, RoPE tables, and the model's default session are owned
     * by geist_model — do NOT destroy here. */
    safe_free((void **) &s);
}

const char *geist_session_errmsg(const struct geist_session *s) {
    if (s == nullptr) {
        return nullptr;
    }
    const struct geist_session_full *sf = (const struct geist_session_full *) s;
    return sf->err_msg[0] != '\0' ? sf->err_msg : "(no error)";
}

[[nodiscard]] enum geist_status geist_session_reset(struct geist_session *s) {
    if (s == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->state_reset == nullptr) {
        return GEIST_E_INVALID_STATE;
    }
    ops->state_reset(arch_sess(sf));
    sf->n_tokens_decoded = 0;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status geist_session_set_prompt(struct geist_session *s,
                                                         const char           *prompt) {
    if (s == nullptr || prompt == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf   = as_full(s);
    const struct geist_arch_ops_decoder *ops  = sf->model->text_decoder.arch_ops;
    struct sp_bpe_tokenizer             *tok  = geist_model_internal_tokenizer(sf->model);
    struct gguf_tokenizer               *gtok = geist_model_internal_gguf_tokenizer(sf->model);
    if (ops == nullptr || ops->prefill == nullptr) {
        return GEIST_E_INVALID_STATE;
    }
    if (tok == nullptr && gtok == nullptr) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "set_prompt: no tokenizer found. SentencePiece path needs "
                 "tokenizer.bin alongside the model (set GEIST_TOKENIZER_PATH "
                 "or place it in the model dir); Llama-family path needs the "
                 "tokenizer embedded in the GGUF (general.architecture + "
                 "tokenizer.ggml.* keys). Falling back to "
                 "geist_session_prefill_tokens with pre-tokenized IDs.");
        sf->err_code = GEIST_E_NOT_FOUND;
        return GEIST_E_NOT_FOUND;
    }

    /* P1.6: GGUF-embedded tokenizer path. Prepends BOS automatically;
     * gguf_tokenizer_encode handles byte-level BPE + merges. */
    if (gtok != nullptr) {
        const size_t cap     = strlen(prompt) + 8; /* upper bound */
        int32_t     *enc_ids = heap_alloc_array_aligned(int32_t, cap);
        if (enc_ids == nullptr) {
            snprintf(sf->err_msg, sizeof(sf->err_msg), "set_prompt: alloc fail");
            sf->err_code = GEIST_E_OOM;
            return GEIST_E_OOM;
        }
        size_t n_enc = 0;
        if (gtok->bos_id >= 0 && gtok->add_bos) {
            enc_ids[n_enc++] = gtok->bos_id;
        }
        size_t enc_n = 0;
        if (!gguf_tokenizer_encode(gtok, prompt, enc_ids + n_enc, cap - n_enc, &enc_n)) {
            void *p = enc_ids;
            safe_free(&p);
            snprintf(sf->err_msg, sizeof(sf->err_msg), "set_prompt: gguf_tokenizer_encode failed");
            sf->err_code = GEIST_E_IO;
            return GEIST_E_IO;
        }
        n_enc += enc_n;
        const uint64_t          t0 = monotonic_ns();
        const enum geist_status ps =
                ops->prefill(arch_sess(sf), n_enc, (const geist_token_t *) enc_ids);
        sf->total_prefill_ns += monotonic_ns() - t0;
        void *p = enc_ids;
        safe_free(&p);
        return ps == GEIST_OK ? GEIST_OK : session_op_failed(sf, ps, "prefill");
    }

    uint32_t *ids   = nullptr;
    size_t    n_ids = 0;
    if (!sp_bpe_tokenizer_encode(tok, prompt, &ids, &n_ids)) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "tokenizer encode failed");
        sf->err_code = GEIST_E_IO;
        return GEIST_E_IO;
    }
    /* sp_bpe yields uint32_t; bit-pattern of u32 ≡ i32 for IDs in 21-bit
     * vocab range. arch->prefill takes geist_token_t (int32_t). */
    const uint64_t          t0 = monotonic_ns();
    const enum geist_status ps = ops->prefill(arch_sess(sf), n_ids, (const geist_token_t *) ids);
    sf->total_prefill_ns += monotonic_ns() - t0;
    safe_free((void **) &ids);
    return ps == GEIST_OK ? GEIST_OK : session_op_failed(sf, ps, "prefill");
}

[[nodiscard]] enum geist_status geist_session_tokenize(struct geist_session *s,
                                                       const char           *text,
                                                       size_t                out_capacity,
                                                       geist_token_t out_ids[static out_capacity],
                                                       size_t       *n_out) {
    if (s == nullptr || text == nullptr || n_out == nullptr ||
        (out_capacity > 0 && out_ids == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    *n_out                        = 0;
    struct geist_session_full *sf = as_full(s);

    /* GGUF-embedded tokenizer path (mirrors set_prompt). No BOS prepended —
     * tokenize reports the content tokens; callers add BOS if they need it. */
    struct gguf_tokenizer *gtok = geist_model_internal_gguf_tokenizer(sf->model);
    if (gtok != nullptr) {
        int32_t *enc = heap_alloc_array_aligned(int32_t, out_capacity ? out_capacity : 1);
        if (enc == nullptr) {
            sf->err_code = GEIST_E_OOM;
            return GEIST_E_OOM;
        }
        size_t enc_n = 0;
        if (!gguf_tokenizer_encode(gtok, text, enc, out_capacity, &enc_n)) {
            void *p = enc;
            safe_free(&p);
            snprintf(sf->err_msg, sizeof(sf->err_msg), "tokenize: gguf encode failed");
            sf->err_code = GEIST_E_IO;
            return GEIST_E_IO;
        }
        for (size_t i = 0; i < enc_n; i++)
            out_ids[i] = (geist_token_t) enc[i];
        *n_out  = enc_n;
        void *p = enc;
        safe_free(&p);
        return GEIST_OK;
    }

    struct sp_bpe_tokenizer *tok = geist_model_internal_tokenizer(sf->model);
    if (tok == nullptr) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "tokenize: no tokenizer found (GGUF-embedded or external tokenizer.bin)");
        sf->err_code = GEIST_E_NOT_FOUND;
        return GEIST_E_NOT_FOUND;
    }

    uint32_t *ids   = nullptr;
    size_t    n_ids = 0;
    if (!sp_bpe_tokenizer_encode(tok, text, &ids, &n_ids)) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "tokenize: encode failed");
        sf->err_code = GEIST_E_IO;
        return GEIST_E_IO;
    }
    if (n_ids > out_capacity) {
        safe_free((void **) &ids);
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "tokenize: %zu tokens > out_capacity %zu",
                 n_ids,
                 out_capacity);
        sf->err_code = GEIST_E_INVALID_ARG;
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < n_ids; i++)
        out_ids[i] = (geist_token_t) ids[i];
    *n_out = n_ids;
    safe_free((void **) &ids);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
geist_session_prefill_tokens(struct geist_session *s, size_t n, const geist_token_t ids[static n]) {
    /* `ids` is declared [static n] — the contract guarantees non-null;
     * GCC -Wnonnull-compare rejects an explicit ids==null check here. */
    if (s == nullptr || n == 0) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->prefill == nullptr) {
        return GEIST_E_INVALID_STATE;
    }
    const uint64_t          t0 = monotonic_ns();
    const enum geist_status ps = ops->prefill(arch_sess(sf), n, ids);
    sf->total_prefill_ns += monotonic_ns() - t0;
    return ps == GEIST_OK ? GEIST_OK : session_op_failed(sf, ps, "prefill");
}

[[nodiscard]] enum geist_status geist_session_decode_step(struct geist_session *s,
                                                          geist_token_t        *out_token) {
    if (s == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->decode_step == nullptr) {
        return GEIST_E_INVALID_STATE;
    }
    const uint64_t          t0 = monotonic_ns();
    const enum geist_status ds = ops->decode_step(arch_sess(sf), out_token);
    sf->total_decode_ns += monotonic_ns() - t0;
    if (ds != GEIST_OK) {
        return session_op_failed(sf, ds, "decode_step");
    }
    sf->n_tokens_decoded++;
    return GEIST_OK;
}

const float *geist_session_peek_logits(size_t *n_logits, struct geist_session *s) {
    if (n_logits == nullptr)
        return nullptr;
    *n_logits = 0;
    if (s == nullptr)
        return nullptr;
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->peek_logits == nullptr)
        return nullptr;
    return ops->peek_logits(n_logits, arch_sess(sf));
}

const float *geist_session_peek_embedding(size_t *n_dims, struct geist_session *s) {
    if (n_dims == nullptr)
        return nullptr;
    *n_dims = 0;
    if (s == nullptr)
        return nullptr;
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->peek_embedding == nullptr)
        return nullptr;
    return ops->peek_embedding(n_dims, arch_sess(sf));
}

/* N-gram drafter: find the longest suffix of `history + [seed]` (up to
 * `max_suffix`) that occurs earlier in `history`, and copy the up-to-
 * (k_max-1) tokens that followed it into `out + 1`. `out[0] = seed`
 * (the model's free argmax) is always set. Returns the total draft
 * count (≥ 1). When `out_match_L` is non-null, writes the length of
 * the suffix match that produced the drafts (0 if no match → only seed
 * returned). Pure function; no state. */
static size_t propose_drafts_ngram(const geist_token_t *history,
                                   size_t               history_n,
                                   geist_token_t        seed,
                                   size_t               k_max,
                                   size_t               max_suffix,
                                   geist_token_t       *out,
                                   size_t              *out_match_L) {
    out[0] = seed;
    if (out_match_L != nullptr)
        *out_match_L = 0;
    if (k_max <= 1 || history_n == 0)
        return 1;

    /* Two-stage search: try longest suffix L first (most specific). At each
     * L, scan all candidate positions and pick the one with the MOST
     * following-context (so we can emit as many drafts as possible).
     * Ties broken by most-recent position — locally coherent text reuses
     * recent context more often than distant. */
    if (max_suffix > 8)
        max_suffix = 8;
    for (size_t L = max_suffix; L >= 1; L--) {
        if (L > history_n + 1)
            continue;
        const size_t suffix_hist = L - 1;
        if (history_n < suffix_hist)
            continue;
        const size_t scan_upper = history_n - suffix_hist;

        size_t best_p      = 0;
        size_t best_follow = 0; /* most-following-context tokens */
        bool   found       = false;

        for (size_t pi = scan_upper; pi > 0; pi--) {
            const size_t p     = pi - 1;
            bool         match = true;
            for (size_t i = 0; i + 1 < L; i++) {
                const size_t hi_idx = history_n - suffix_hist + i;
                if (history[p + i] != history[hi_idx]) {
                    match = false;
                    break;
                }
            }
            if (!match)
                continue;
            if (L > 1 && history[p + L - 1] != seed)
                continue;
            if (L == 1 && history[p] != seed)
                continue;

            const size_t follow_start = p + L;
            if (follow_start >= history_n)
                continue; /* match at tail, no drafts */
            const size_t follow = history_n - follow_start;
            const size_t capped = follow < (k_max - 1) ? follow : (k_max - 1);

            if (!found || capped > best_follow) {
                best_p      = p;
                best_follow = capped;
                found       = true;
                /* Early-exit if we already have enough follow-context. */
                if (best_follow >= k_max - 1)
                    break;
            }
        }

        if (found) {
            size_t draft_count = 1;
            for (size_t i = 0; i < best_follow; i++) {
                out[draft_count++] = history[best_p + L + i];
            }
            if (out_match_L != nullptr)
                *out_match_L = L;
            return draft_count;
        }
    }
    return 1; /* no match — just the seed */
}

#define GEIST_SPEC_KMAX_HARDCAP 16

/* Retaining the verified correction avoids one redundant target prefill, but
 * changes speculative call boundaries and can lower n-gram acceptance enough
 * to regress end-to-end throughput. Keep it opt-in until a drafter can carry
 * the pending token across calls without losing useful matches. */
static bool spec_retain_verified_pending_enabled(void) {
    const char *env = getenv("GEIST_SPEC_RETAIN_PENDING");
    return env != nullptr && strcmp(env, "1") == 0;
}

/* Single-token fallback used whenever spec_step can't draft or verify
 * (missing arch primitives, no pending logits, empty drafter result).
 * Sequential decode_step path is contractually identical so the caller
 * gets the same next token either way. */
[[nodiscard]] static enum geist_status
spec_fallback_single(struct geist_session *s, geist_token_t out_tokens[static 1], size_t *n_out) {
    geist_token_t           t;
    const enum geist_status sx = geist_session_decode_step(s, &t);
    if (sx != GEIST_OK)
        return sx;
    out_tokens[0] = t;
    *n_out        = 1;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
geist_session_decode_speculative(struct geist_session *s,
                                 size_t                k_max,
                                 size_t                history_n,
                                 const geist_token_t   history[static history_n],
                                 size_t                out_capacity,
                                 geist_token_t         out_tokens[static out_capacity],
                                 size_t               *n_out) {
    if (s == nullptr || n_out == nullptr || (history_n > 0 && history == nullptr) ||
        (out_capacity > 0 && out_tokens == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    if (k_max == 0 || k_max > GEIST_SPEC_KMAX_HARDCAP) {
        return GEIST_E_INVALID_ARG;
    }
    if (out_capacity < k_max + 1) {
        return GEIST_E_INVALID_ARG;
    }
    *n_out = 0;

    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    void                                *st  = sf->arch_session;
    if (ops == nullptr)
        return GEIST_E_INVALID_STATE;

    const bool can_spec = ops->peek_next_token != nullptr && ops->verify_forward != nullptr &&
                          ops->kv_truncate != nullptr && ops->kv_len != nullptr;
    if (!can_spec)
        return spec_fallback_single(s, out_tokens, n_out);

    const uint64_t      t_start = monotonic_ns();
    const geist_token_t seed    = ops->peek_next_token(st);
    if (seed < 0)
        return spec_fallback_single(s, out_tokens, n_out);

    geist_token_t drafts[GEIST_SPEC_KMAX_HARDCAP];
    size_t        match_L      = 0;
    size_t        k            = 0;
    bool          native_draft = false;
    if (ops->draft_tokens != nullptr) {
        const enum geist_status ds = ops->draft_tokens(st, k_max, seed, drafts, &k);
        if (ds == GEIST_OK) {
            native_draft = k > 0;
        } else if (ds != GEIST_E_UNSUPPORTED) {
            return session_op_failed(sf, ds, "architecture-native draft");
        }
    }
    if (!native_draft) {
        k = propose_drafts_ngram(
                history, history_n, seed, k_max, /*max_suffix=*/4, drafts, &match_L);
    }
    if (k <= 1)
        return spec_fallback_single(s, out_tokens, n_out);

    /* Lazy-draft gate: skip the verify-forward pass when the n-gram
     * match was weak (only seed-token matched, i.e. L=1). On novel-
     * structure prompts (smart-home commands, fresh Python code) most
     * positions only get L=1 matches with low accept rates — paying
     * verify-forward (~1.6-2.3× single decode) to land 1 token there
     * is a net loss. L≥2 means at least one prior token of context
     * matched too, which empirically maps to materially higher accept
     * rates. Override the threshold via GEIST_SPEC_MIN_L=N. */
    /* Relaxed-atomic first-use cache: concurrent sessions reach this on
     * their own threads. */
    static _Atomic int min_L_cached = -1;
    int                min_L        = atomic_load_explicit(&min_L_cached, memory_order_relaxed);
    if (min_L < 0) {
        const char *env = getenv("GEIST_SPEC_MIN_L");
        const long  v   = (env != nullptr) ? atol(env) : 2;
        min_L           = (v <= 0) ? 1 : (v > 8 ? 8 : (int) v);
        atomic_store_explicit(&min_L_cached, min_L, memory_order_relaxed);
    }
    if (!native_draft && (int) match_L < min_L) {
        return spec_fallback_single(s, out_tokens, n_out);
    }

    const size_t            kv_before = ops->kv_len(st);
    geist_token_t           verify_out[GEIST_SPEC_KMAX_HARDCAP];
    const enum geist_status vs = ops->verify_forward(st, k, drafts, verify_out);
    if (vs != GEIST_OK)
        return vs;

    /* Accept the longest prefix where the model's per-position prediction
     * matched the drafter's next-token guess. */
    size_t accepted_extras = 0;
    for (size_t i = 0; i + 1 < k; i++) {
        if (verify_out[i] != drafts[i + 1])
            break;
        accepted_extras++;
    }

    geist_token_t correction;
    size_t        accepted_tokens;
    if (accepted_extras == k - 1) {
        /* All drafts verified. The model's bonus prediction remains pending
         * when the architecture can retain its verified last-row logits. */
        correction      = verify_out[k - 1];
        accepted_tokens = k;
    } else {
        /* Partial accept. Keep KV[..kv_before + accepted_extras + 1) (= the
         * accepted draft positions) and discard the rest. Architectures with
         * recurrent state replay that prefix and retain its prediction. */
        const enum geist_status ts = ops->kv_truncate(st, kv_before + accepted_extras + 1);
        if (ts != GEIST_OK) {
            return session_op_failed(sf, ts, "speculative state rollback");
        }
        correction      = verify_out[accepted_extras];
        accepted_tokens = accepted_extras + 1;
    }

    memcpy(out_tokens, drafts, accepted_tokens * sizeof(geist_token_t));
    size_t emitted = accepted_tokens;

    /* A capable architecture retains the prediction after the committed
     * verify prefix as its pending token. Leave it for the next API call: it
     * has not been emitted yet, so no cache advance is required. Older
     * architectures, and the default mode, keep the established
     * correction-emission fallback. */
    const bool correction_ready =
            spec_retain_verified_pending_enabled() && ops->peek_next_token(st) == correction;
    if (!correction_ready) {
        out_tokens[emitted++]      = correction;
        const enum geist_status cs = ops->prefill(st, 1, &correction);
        if (cs != GEIST_OK) {
            return session_op_failed(sf, cs, "speculative correction prefill");
        }
    }

    *n_out = emitted;
    sf->n_tokens_decoded += emitted;
    sf->total_decode_ns += monotonic_ns() - t_start;
    return GEIST_OK;
}

const char *geist_session_token_to_str(struct geist_session *s, geist_token_t t) {
    if (s == nullptr || t < 0) {
        return nullptr;
    }
    struct geist_session_full *sf = as_full(s);
    /* GGUF-embedded (GPT-2 byte-level BPE) path: Llama-family models (BitNet,
     * Mistral, SmolLM2, …) carry their tokenizer in the GGUF, not as an
     * external SentencePiece tokenizer.bin. The encode path (set_prompt)
     * already uses gtok; mirror it here for decode. Byte-level BPE needs the
     * codepoint→byte reconstruction (gguf_tokenizer_decode), so decode the
     * single token into a thread-local buffer rather than returning a raw
     * vocab pointer. Valid until the next call — matches the streaming
     * decode-step usage in callers. */
    struct gguf_tokenizer *gtok = geist_model_internal_gguf_tokenizer(sf->model);
    if (gtok != nullptr) {
        static _Thread_local char tok_buf[256];
        const int32_t             id = (int32_t) t;
        /* decode returns the would-be total length, which may exceed the cap;
         * clamp before indexing so a long token surface form cannot write past
         * tok_buf. (decode already NUL-terminates internally on truncation.) */
        size_t n = gguf_tokenizer_decode(gtok, &id, 1, tok_buf, sizeof tok_buf - 1);
        if (n >= sizeof tok_buf)
            n = sizeof tok_buf - 1;
        tok_buf[n] = '\0';
        return tok_buf;
    }
    struct sp_bpe_tokenizer *tok = geist_model_internal_tokenizer(sf->model);
    if (tok == nullptr) {
        return nullptr;
    }
    size_t      len  = 0;
    const char *text = sp_bpe_tokenizer_id_to_text(tok, (uint32_t) t, &len);
    return text; /* Pointer into tokenizer's mmap region — valid for tok's lifetime. */
}

/* Audio path: PCM → mel → audio_conformer encode → soft tokens →
 * decoder arch_ops->prefill_audio. */
[[nodiscard]] enum geist_status
geist_session_attach_audio(struct geist_session *s,
                           size_t                n_samples,
                           const int16_t         pcm_samples[static n_samples],
                           int                   sample_rate) {
    if (s == nullptr || n_samples == 0 || pcm_samples == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf      = as_full(s);
    const struct geist_arch_ops_encoder *enc_ops = sf->model->audio_encoder.arch_ops;
    void                                *enc_st  = sf->model->audio_encoder.arch_meta;
    const struct geist_arch_ops_decoder *dec_ops = sf->model->text_decoder.arch_ops;
    if (enc_ops == nullptr || enc_st == nullptr) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_audio: model has no audio encoder loaded "
                 "(missing audio_tower.safetensors or mel_constants.bin)");
        sf->err_code = GEIST_E_NOT_FOUND;
        return GEIST_E_NOT_FOUND;
    }
    if (dec_ops == nullptr || dec_ops->prefill_audio == nullptr) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_audio: decoder arch lacks audio injection path");
        sf->err_code = GEIST_E_UNSUPPORTED;
        return GEIST_E_UNSUPPORTED;
    }
    if (sample_rate != 16000) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_audio: only 16 kHz PCM supported (got %d Hz)",
                 sample_rate);
        sf->err_code = GEIST_E_UNSUPPORTED;
        return GEIST_E_UNSUPPORTED;
    }

    /* Size the soft-token buffer from the audio length (#247) — the old
     * hardcoded 256 silently dropped everything past ~10 s while still
     * paying the full encode. Fallback stays for encoders without the op. */
    size_t max_soft = 256;
    if (enc_ops->max_soft_tokens != nullptr) {
        max_soft = enc_ops->max_soft_tokens(enc_st, n_samples);
    }
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *soft     = heap_alloc_array_aligned(float, max_soft *soft_dim);
    if (soft == nullptr) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "attach_audio: soft-token buffer alloc failed");
        sf->err_code = GEIST_E_OOM;
        return GEIST_E_OOM;
    }
    const uint64_t t_enc0 = monotonic_ns();
    size_t         n_soft = enc_ops->encode_pcm(enc_st, n_samples, max_soft, pcm_samples, soft);
    sf->total_audio_encode_ns += monotonic_ns() - t_enc0;
    if (n_soft == 0) {
        safe_free((void **) &soft);
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_audio: audio encoder produced 0 soft tokens "
                 "(too-short input, audio beyond the encoder's 30 s limit, "
                 "or encoder failure)");
        sf->err_code = GEIST_E_IO;
        return GEIST_E_IO;
    }
    if (n_soft >= max_soft) {
        /* The bound carries headroom, so a full buffer can only mean the
         * encoder hit the cap — truncated audio must be an error, not a
         * silent drop (#247). */
        safe_free((void **) &soft);
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_audio: soft-token bound too small (%zu tokens for %zu samples) — "
                 "audio would be truncated",
                 max_soft,
                 n_samples);
        sf->err_code = GEIST_E_INTERNAL;
        return GEIST_E_INTERNAL;
    }

    const uint64_t          t_pre0 = monotonic_ns();
    const enum geist_status as     = dec_ops->prefill_audio(arch_sess(sf), n_soft, soft);
    sf->total_prefill_ns += monotonic_ns() - t_pre0;
    safe_free((void **) &soft);
    return as == GEIST_OK ? GEIST_OK : session_op_failed(sf, as, "prefill_audio");
}

/* Streaming audio turn (#256): equivalent to attach_audio over the
 * concatenated PCM, but the encoder overlaps its work with the arriving
 * audio — end() pays only the tail. */
[[nodiscard]] enum geist_status geist_session_audio_begin(struct geist_session *s) {
    if (s == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf      = as_full(s);
    const struct geist_arch_ops_encoder *enc_ops = sf->model->audio_encoder.arch_ops;
    void                                *enc_st  = sf->model->audio_encoder.arch_meta;
    const struct geist_arch_ops_decoder *dec_ops = sf->model->text_decoder.arch_ops;
    if (enc_ops == nullptr || enc_st == nullptr || dec_ops == nullptr ||
        dec_ops->prefill_audio == nullptr) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "audio_begin: model cannot hear");
        sf->err_code = GEIST_E_NOT_FOUND;
        return GEIST_E_NOT_FOUND;
    }
    if (enc_ops->stream_begin == nullptr || enc_ops->stream_push == nullptr ||
        enc_ops->stream_end == nullptr) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "audio_begin: encoder has no streaming path");
        sf->err_code = GEIST_E_UNSUPPORTED;
        return GEIST_E_UNSUPPORTED;
    }
    if (sf->audio_streaming) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "audio_begin: streaming turn already open");
        sf->err_code = GEIST_E_INVALID_STATE;
        return GEIST_E_INVALID_STATE;
    }
    /* Scratch for poll/end, sized for the encoder's 30 s worst case so
     * incremental injection never reallocates mid-turn.
     *
     * Allocated BEFORE stream_begin, deliberately. The other order left a
     * window where the encoder stream was running and the allocation had
     * failed: audio_begin returned OOM, audio_streaming stayed false, and
     * nothing afterwards could reach the open stream to close it. Every
     * fallible step now happens while there is still nothing to unwind,
     * so the only thing after the commit point is bookkeeping. */
    size_t cap = 256;
    if (enc_ops->max_soft_tokens != nullptr) {
        cap = enc_ops->max_soft_tokens(enc_st, (size_t) -1); /* clamped internally */
    }
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *scratch  = heap_alloc_n_aligned(cap, soft_dim * sizeof(float), alignof(float));
    if (scratch == nullptr) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "audio_begin: scratch alloc failed");
        sf->err_code = GEIST_E_OOM;
        return GEIST_E_OOM;
    }
    if (!enc_ops->stream_begin(enc_st)) {
        safe_free((void **) &scratch);
        snprintf(sf->err_msg, sizeof(sf->err_msg), "audio_begin: encoder stream_begin failed");
        sf->err_code = GEIST_E_BACKEND;
        return GEIST_E_BACKEND;
    }
    sf->audio_stream_buf      = scratch;
    sf->audio_stream_cap      = cap;
    sf->audio_stream_injected = 0;
    sf->audio_streaming       = true;
    atomic_store_explicit(&sf->audio_stream_samples, 0, memory_order_relaxed);
    return GEIST_OK;
}

/* Inject whatever soft tokens the encoder has ready — phase 2 of #256:
 * called from the session thread between pushes, it overlaps the LM
 * prefill with the still-running capture, shrinking end()'s tail. */
[[nodiscard]] enum geist_status geist_session_audio_poll(struct geist_session *s) {
    if (s == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf      = as_full(s);
    const struct geist_arch_ops_encoder *enc_ops = sf->model->audio_encoder.arch_ops;
    void                                *enc_st  = sf->model->audio_encoder.arch_meta;
    const struct geist_arch_ops_decoder *dec_ops = sf->model->text_decoder.arch_ops;
    if (!sf->audio_streaming) {
        return GEIST_E_INVALID_STATE;
    }
    if (enc_ops->stream_poll == nullptr || sf->audio_stream_injected >= sf->audio_stream_cap) {
        return GEIST_OK; /* nothing to do — end() covers it */
    }
    const uint64_t t0  = monotonic_ns();
    const size_t   got = enc_ops->stream_poll(
            enc_st, sf->audio_stream_cap - sf->audio_stream_injected, sf->audio_stream_buf);
    sf->total_audio_encode_ns += monotonic_ns() - t0;
    if (got == 0) {
        return GEIST_OK;
    }
    const enum geist_status as = dec_ops->prefill_audio(arch_sess(sf), got, sf->audio_stream_buf);
    if (as != GEIST_OK) {
        return session_op_failed(sf, as, "prefill_audio (poll)");
    }
    sf->audio_stream_injected += got;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
geist_session_audio_push(struct geist_session *s, size_t n, const int16_t pcm[static n]) {
    if (s == nullptr || n == 0 || pcm == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf      = as_full(s);
    const struct geist_arch_ops_encoder *enc_ops = sf->model->audio_encoder.arch_ops;
    void                                *enc_st  = sf->model->audio_encoder.arch_meta;
    if (!sf->audio_streaming) {
        return GEIST_E_INVALID_STATE;
    }
    if (!enc_ops->stream_push(enc_st, n, pcm)) {
        /* Buffer overflow (>30 s) or encoder shutdown — surface loudly,
         * never drop audio silently (#247's rule). */
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "audio_push: encoder refused %zu samples (>30 s buffered?)",
                 n);
        sf->err_code = GEIST_E_UNSUPPORTED;
        return GEIST_E_UNSUPPORTED;
    }
    atomic_fetch_add_explicit(&sf->audio_stream_samples, n, memory_order_relaxed);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status geist_session_audio_end(struct geist_session *s) {
    if (s == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf      = as_full(s);
    const struct geist_arch_ops_encoder *enc_ops = sf->model->audio_encoder.arch_ops;
    void                                *enc_st  = sf->model->audio_encoder.arch_meta;
    const struct geist_arch_ops_decoder *dec_ops = sf->model->text_decoder.arch_ops;
    if (!sf->audio_streaming) {
        return GEIST_E_INVALID_STATE;
    }
    sf->audio_streaming = false;

    const size_t n_samples = atomic_load_explicit(&sf->audio_stream_samples, memory_order_relaxed);
    float       *soft      = sf->audio_stream_buf;
    sf->audio_stream_buf   = nullptr;
    const size_t remaining_cap = sf->audio_stream_cap > sf->audio_stream_injected
                                         ? sf->audio_stream_cap - sf->audio_stream_injected
                                         : 0;

    const uint64_t t_enc0 = monotonic_ns();
    const size_t n_soft = remaining_cap > 0 ? enc_ops->stream_end(enc_st, remaining_cap, soft) : 0;
    sf->total_audio_encode_ns += monotonic_ns() - t_enc0;

    const size_t total = sf->audio_stream_injected + n_soft;
    if (total == 0) {
        safe_free((void **) &soft);
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "audio_end: encoder produced 0 soft tokens (no audio pushed, "
                 "or encoder failure)");
        sf->err_code = GEIST_E_IO;
        return GEIST_E_IO;
    }
    if (total >= sf->audio_stream_cap) {
        safe_free((void **) &soft);
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "audio_end: soft-token bound too small (%zu tokens for %zu samples)",
                 sf->audio_stream_cap,
                 n_samples);
        sf->err_code = GEIST_E_INTERNAL;
        return GEIST_E_INTERNAL;
    }
    enum geist_status as = GEIST_OK;
    if (n_soft > 0) {
        as = dec_ops->prefill_audio(arch_sess(sf), n_soft, soft);
    }
    safe_free((void **) &soft);
    return as == GEIST_OK ? GEIST_OK : session_op_failed(sf, as, "prefill_audio");
}

/* Vision path: RGB pixels → image_pipeline → vision_siglip tower →
 * pool → projector → soft tokens → decoder arch_ops->prefill_image. */
[[nodiscard]] enum geist_status
geist_session_attach_image(struct geist_session *s,
                           size_t                height,
                           size_t                width,
                           const uint8_t         rgb[static height * width * 3]) {
    if (s == nullptr || height == 0 || width == 0 || rgb == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    /* Geometry is the caller's and is multiplied out several layers down
     * (height * width * 3 for the pixel extent, width * 3 for a row stride
     * that narrows to int on the way into the resizer). Bound it at the
     * public entry so no product downstream can be formed from a value
     * that was never representable. */
    if (height > IMAGE_PIPELINE_MAX_DIM || width > IMAGE_PIPELINE_MAX_DIM) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf      = as_full(s);
    const struct geist_arch_ops_vision  *enc_ops = sf->model->vision_encoder.arch_ops;
    void                                *enc_st  = sf->model->vision_encoder.arch_meta;
    const struct geist_arch_ops_decoder *dec_ops = sf->model->text_decoder.arch_ops;
    if (enc_ops == nullptr || enc_st == nullptr) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_image: model has no vision encoder loaded "
                 "(missing vision_tower.safetensors)");
        sf->err_code = GEIST_E_NOT_FOUND;
        return GEIST_E_NOT_FOUND;
    }
    if (dec_ops == nullptr || dec_ops->prefill_image == nullptr) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_image: decoder arch lacks vision injection path");
        sf->err_code = GEIST_E_UNSUPPORTED;
        return GEIST_E_UNSUPPORTED;
    }

    /* Gemma 4: max 280 soft tokens per image. soft_dim is 1536 (== text
     * hidden_size). */
    const size_t max_soft = 280;
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *soft = heap_alloc_n_aligned(max_soft, soft_dim * sizeof(float), alignof(float));
    if (soft == nullptr) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "attach_image: soft-token buffer alloc failed");
        sf->err_code = GEIST_E_OOM;
        return GEIST_E_OOM;
    }
    const uint64_t t_enc0 = monotonic_ns();
    size_t         n_soft = enc_ops->encode_image(enc_st, height, width, max_soft, rgb, soft);
    sf->total_audio_encode_ns += monotonic_ns() - t_enc0;
    if (n_soft == 0) {
        safe_free((void **) &soft);
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_image: vision encoder produced 0 soft tokens "
                 "(degenerate image dims or encoder failure)");
        sf->err_code = GEIST_E_IO;
        return GEIST_E_IO;
    }

    const uint64_t          t_pre0 = monotonic_ns();
    const enum geist_status is     = dec_ops->prefill_image(arch_sess(sf), n_soft, soft);
    sf->total_prefill_ns += monotonic_ns() - t_pre0;
    safe_free((void **) &soft);
    return is == GEIST_OK ? GEIST_OK : session_op_failed(sf, is, "prefill_image");
}

/* Video path: per-frame run_image → concat soft tokens →
 * decoder arch_ops->prefill_image. Reuses prefill_image since image
 * and video soft tokens share the 1536-dim wire format. */
[[nodiscard]] enum geist_status
geist_session_attach_video(struct geist_session *s,
                           size_t                n_frames,
                           size_t                height,
                           size_t                width,
                           const uint8_t         frames[static n_frames * height * width * 3]) {
    if (s == nullptr || n_frames == 0 || height == 0 || width == 0 || frames == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    /* Same bound as attach_image, plus the clip extent: the encoder walks
     * `frames` in height*width*3 strides for n_frames of them, and the
     * soft-token buffer below is 70 * n_frames rows. Both products are
     * caller-controlled. */
    size_t frame_px = 0, frame_bytes = 0, clip_bytes = 0;
    if (height > IMAGE_PIPELINE_MAX_DIM || width > IMAGE_PIPELINE_MAX_DIM ||
        ckd_mul(&frame_px, height, width) || ckd_mul(&frame_bytes, frame_px, 3u) ||
        ckd_mul(&clip_bytes, frame_bytes, n_frames)) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf      = as_full(s);
    const struct geist_arch_ops_vision  *enc_ops = sf->model->vision_encoder.arch_ops;
    void                                *enc_st  = sf->model->vision_encoder.arch_meta;
    const struct geist_arch_ops_decoder *dec_ops = sf->model->text_decoder.arch_ops;
    if (enc_ops == nullptr || enc_st == nullptr) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_video: model has no vision encoder loaded "
                 "(missing vision_tower.safetensors)");
        sf->err_code = GEIST_E_NOT_FOUND;
        return GEIST_E_NOT_FOUND;
    }
    if (dec_ops == nullptr || dec_ops->prefill_image == nullptr) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_video: decoder arch lacks vision injection path");
        sf->err_code = GEIST_E_UNSUPPORTED;
        return GEIST_E_UNSUPPORTED;
    }

    /* 70 soft tokens per frame × n_frames; soft_dim = 1536 (matches LM
     * residual stream). */
    size_t max_soft = 0;
    if (ckd_mul(&max_soft, (size_t) 70, n_frames)) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_video: n_frames %zu is too large",
                 n_frames);
        sf->err_code = GEIST_E_INVALID_ARG;
        return GEIST_E_INVALID_ARG;
    }
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *soft = heap_alloc_n_aligned(max_soft, soft_dim * sizeof(float), alignof(float));
    if (soft == nullptr) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "attach_video: soft-token buffer alloc failed");
        sf->err_code = GEIST_E_OOM;
        return GEIST_E_OOM;
    }
    const uint64_t t_enc0 = monotonic_ns();
    size_t n_soft = enc_ops->encode_video(enc_st, n_frames, height, width, max_soft, frames, soft);
    sf->total_audio_encode_ns += monotonic_ns() - t_enc0;
    if (n_soft == 0) {
        safe_free((void **) &soft);
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_video: vision encoder produced 0 soft tokens "
                 "(degenerate frame dims or encoder failure)");
        sf->err_code = GEIST_E_IO;
        return GEIST_E_IO;
    }

    const uint64_t          t_pre0 = monotonic_ns();
    const enum geist_status is     = dec_ops->prefill_image(arch_sess(sf), n_soft, soft);
    sf->total_prefill_ns += monotonic_ns() - t_pre0;
    safe_free((void **) &soft);
    return is == GEIST_OK ? GEIST_OK : session_op_failed(sf, is, "prefill_image");
}

[[nodiscard]] enum geist_status
geist_session_pin_prefix(struct geist_session *s, size_t n, const geist_token_t ids[static n]) {
    if (s == nullptr || (n > 0 && ids == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->pin_prefix == nullptr) {
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "pin_prefix: active architecture does not support prefix pinning");
        sf->err_code = GEIST_E_UNSUPPORTED;
        return GEIST_E_UNSUPPORTED;
    }
    const enum geist_status ps = ops->pin_prefix(arch_sess(sf), n, ids);
    return ps == GEIST_OK ? GEIST_OK : session_op_failed(sf, ps, "pin_prefix");
}

[[nodiscard]] enum geist_status geist_session_get_stats(const struct geist_session *s,
                                                        struct geist_session_stats *out) {
    if (s == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const struct geist_session_full *sf = (const struct geist_session_full *) s;
    *out                                = (struct geist_session_stats) {
            .n_tokens_decoded      = sf->n_tokens_decoded,
            .total_decode_ns       = sf->total_decode_ns,
            .total_prefill_ns      = sf->total_prefill_ns,
            .total_audio_encode_ns = sf->total_audio_encode_ns,
            /* buffer_alloc_* and per_op_* still stubbed at zero — those need
             * backend-side counters / opt-in op profiling. */
    };
    return GEIST_OK;
}

[[nodiscard]] enum geist_status geist_session_reset_stats(struct geist_session *s) {
    if (s == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full *sf = as_full(s);
    sf->n_tokens_decoded          = 0;
    sf->total_decode_ns           = 0;
    sf->total_prefill_ns          = 0;
    sf->total_audio_encode_ns     = 0;
    return GEIST_OK;
}
