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

#include "session.h"
#include "model.h"
#include "error.h"

#include <geist_arch.h> /* arch_ops vtables the engine dispatches through */

#include "heap.h"
#include "sp_bpe_tokenizer.h"
#include "gguf_tokenizer.h"

#include <geist.h>
#include <geist_util.h> /* tokenize/prefill/attach/peek/speculative/stats moved here in 0.2.0 */

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

/* P1.2.f: bind this session's per-session arch state as the active one
 * on the model before any vtable dispatch. No-op if the architecture
 * doesn't support multi-session (arch_session is then nullptr and the
 * model's default session is what dispatch operates on). */
static inline void session_attach(const struct geist_session_full *sf) {
    if (sf == nullptr || sf->arch_session == nullptr) {
        return;
    }
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops != nullptr && ops->session_attach != nullptr) {
        ops->session_attach(sf->model->text_decoder.arch_meta, sf->arch_session);
    }
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
     * Architectures without session_alloc (or without session_attach)
     * stay single-session-per-model — the engine falls back to using
     * the model's default session, and set_session_opts is the only
     * path for sampler config. */
    const struct geist_arch_ops_decoder *ops = m->text_decoder.arch_ops;
    if (ops != nullptr && ops->session_alloc != nullptr && ops->session_attach != nullptr) {
        sf->arch_session = ops->session_alloc(m->text_decoder.arch_meta, opts);
        if (sf->arch_session == nullptr) {
            geist_error_set_create_time(
                    GEIST_E_OOM, "geist_session_create", "arch session_alloc returned null");
            void *tmp = sf;
            safe_free(&tmp);
            return GEIST_E_OOM;
        }
    } else if (opts != nullptr && ops != nullptr && ops->set_session_opts != nullptr) {
        /* Legacy single-session path: push opts onto the model's default
         * session. When multiple legacy sessions share one model, the
         * last set_session_opts call wins. */
        ops->set_session_opts(m->text_decoder.arch_meta, opts);
    }

    *out = (struct geist_session *) sf;
    return GEIST_OK;
}

void geist_session_destroy(struct geist_session *s) {
    if (s == nullptr) {
        return;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    /* P1.2.f: release this session's per-session arch state if it owns
     * one. session_free detaches itself if it was the active session,
     * restoring the model's default. */
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
    session_attach(sf);
    ops->state_reset(sf->model->text_decoder.arch_meta);
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
        if (gtok->bos_id >= 0) {
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
        session_attach(sf);
        const uint64_t    t0 = monotonic_ns();
        enum geist_status ps = ops->prefill(
                sf->model->text_decoder.arch_meta, n_enc, (const geist_token_t *) enc_ids);
        sf->total_prefill_ns += monotonic_ns() - t0;
        void *p = enc_ids;
        safe_free(&p);
        if (ps != GEIST_OK) {
            snprintf(sf->err_msg, sizeof(sf->err_msg), "set_prompt: prefill failed");
            sf->err_code = ps;
            return ps;
        }
        return GEIST_OK;
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
    session_attach(sf);
    const uint64_t    t0 = monotonic_ns();
    enum geist_status ps =
            ops->prefill(sf->model->text_decoder.arch_meta, n_ids, (const geist_token_t *) ids);
    sf->total_prefill_ns += monotonic_ns() - t0;
    safe_free((void **) &ids);
    if (ps != GEIST_OK) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "set_prompt: prefill failed");
        sf->err_code = ps;
        return ps;
    }
    return GEIST_OK;
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
        /* Encode into a scratch buffer one slot larger than out_capacity so
         * an over-length result is detectable: gguf_tokenizer_encode clamps
         * to its cap and reports success either way, so with cap == capacity
         * a truncated result is indistinguishable from an exact fit. The
         * extra slot lets us return GEIST_E_INVALID_ARG on overflow, matching
         * the SentencePiece path's contract (see doc in geist_util.h). */
        const size_t scratch_cap =
                out_capacity + 1; /* out_capacity is a token count, no overflow */
        int32_t *enc = heap_alloc_array_aligned(int32_t, scratch_cap);
        if (enc == nullptr) {
            sf->err_code = GEIST_E_OOM;
            return GEIST_E_OOM;
        }
        size_t enc_n = 0;
        if (!gguf_tokenizer_encode(gtok, text, enc, scratch_cap, &enc_n)) {
            void *p = enc;
            safe_free(&p);
            snprintf(sf->err_msg, sizeof(sf->err_msg), "tokenize: gguf encode failed");
            sf->err_code = GEIST_E_IO;
            return GEIST_E_IO;
        }
        if (enc_n > out_capacity) {
            void *p = enc;
            safe_free(&p);
            snprintf(sf->err_msg,
                     sizeof(sf->err_msg),
                     "tokenize: output buffer too small (need > %zu ids)",
                     out_capacity);
            sf->err_code = GEIST_E_INVALID_ARG;
            return GEIST_E_INVALID_ARG;
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
    session_attach(sf);
    const uint64_t    t0 = monotonic_ns();
    enum geist_status ps = ops->prefill(sf->model->text_decoder.arch_meta, n, ids);
    sf->total_prefill_ns += monotonic_ns() - t0;
    if (ps != GEIST_OK) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "prefill_tokens: prefill failed");
        sf->err_code = ps;
        return ps;
    }
    return GEIST_OK;
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
    session_attach(sf);
    const uint64_t t0 = monotonic_ns();
    *out_token        = ops->decode_step(sf->model->text_decoder.arch_meta);
    sf->total_decode_ns += monotonic_ns() - t0;
    sf->n_tokens_decoded++;
    return GEIST_OK;
}

const float *geist_session_peek_logits(struct geist_session *s, size_t *n_logits) {
    if (n_logits == nullptr)
        return nullptr;
    *n_logits = 0;
    if (s == nullptr)
        return nullptr;
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->peek_logits == nullptr)
        return nullptr;
    session_attach(sf);
    return ops->peek_logits(sf->model->text_decoder.arch_meta, n_logits);
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
    void                                *st  = sf->model->text_decoder.arch_meta;
    if (ops == nullptr)
        return GEIST_E_INVALID_STATE;

    const bool can_spec = ops->peek_next_token != nullptr && ops->verify_forward != nullptr &&
                          ops->kv_truncate != nullptr && ops->kv_len != nullptr;
    if (!can_spec)
        return spec_fallback_single(s, out_tokens, n_out);

    session_attach(sf);
    const uint64_t      t_start = monotonic_ns();
    const geist_token_t seed    = ops->peek_next_token(st);
    if (seed < 0)
        return spec_fallback_single(s, out_tokens, n_out);

    geist_token_t drafts[GEIST_SPEC_KMAX_HARDCAP];
    size_t        match_L = 0;
    const size_t  k       = propose_drafts_ngram(
            history, history_n, seed, k_max, /*max_suffix=*/4, drafts, &match_L);
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
    static int min_L_cached = -1;
    if (min_L_cached < 0) {
        const char *env = getenv("GEIST_SPEC_MIN_L");
        const long  v   = (env != nullptr) ? atol(env) : 2;
        min_L_cached    = (v <= 0) ? 1 : (v > 8 ? 8 : (int) v);
    }
    if ((int) match_L < min_L_cached) {
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
    size_t        emitted;
    if (accepted_extras == k - 1) {
        /* All drafts verified — emit them + the model's bonus prediction. */
        memcpy(out_tokens, drafts, k * sizeof(geist_token_t));
        correction    = verify_out[k - 1];
        out_tokens[k] = correction;
        emitted       = k + 1;
    } else {
        /* Partial accept. Keep KV[..kv_before + accepted_extras + 1) (= the
         * accepted draft positions), discard the rest, then re-push the
         * correction so the cache + logits are ready for the next call. */
        ops->kv_truncate(st, kv_before + accepted_extras + 1);
        memcpy(out_tokens, drafts, (accepted_extras + 1) * sizeof(geist_token_t));
        correction                      = verify_out[accepted_extras];
        out_tokens[accepted_extras + 1] = correction;
        emitted                         = accepted_extras + 2;
    }

    /* The last emit (bonus or correction) is not yet in the cache. The
     * next spec_step / decode_step needs pending logits computed from
     * it, so push it now via a single-token prefill. Numerically the
     * same as ending the previous batched forward one token earlier. */
    enum geist_status ps = ops->prefill(st, 1, &correction);
    if (ps != GEIST_OK) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "decode_speculative: correction prefill failed");
        sf->err_code = ps;
        return ps;
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

    /* struct AudioEncoder caps soft tokens — 188 is the practical max for the
     * Gemma 4 tower (30 s of audio at 50 Hz frame rate, downsampled 4×,
     * then 2× by Conformer). Allocate generously. */
    const size_t max_soft = 256;
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *soft     = heap_alloc_array_aligned(float, max_soft *soft_dim);
    if (soft == nullptr) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "attach_audio: soft-token buffer alloc failed");
        sf->err_code = GEIST_E_OOM;
        return GEIST_E_OOM;
    }
    const uint64_t t_enc0 = monotonic_ns();
    size_t         n_soft = enc_ops->encode_pcm(enc_st, pcm_samples, n_samples, soft, max_soft);
    sf->total_audio_encode_ns += monotonic_ns() - t_enc0;
    if (n_soft == 0) {
        safe_free((void **) &soft);
        snprintf(sf->err_msg,
                 sizeof(sf->err_msg),
                 "attach_audio: audio encoder produced 0 soft tokens "
                 "(too-short input or encoder failure)");
        sf->err_code = GEIST_E_IO;
        return GEIST_E_IO;
    }

    session_attach(sf);
    const uint64_t    t_pre0 = monotonic_ns();
    enum geist_status ps = dec_ops->prefill_audio(sf->model->text_decoder.arch_meta, n_soft, soft);
    sf->total_prefill_ns += monotonic_ns() - t_pre0;
    safe_free((void **) &soft);
    if (ps != GEIST_OK) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "attach_audio: prefill failed");
        sf->err_code = ps;
        return ps;
    }
    return GEIST_OK;
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
    float       *soft     = heap_alloc_array_aligned(float, max_soft *soft_dim);
    if (soft == nullptr) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "attach_image: soft-token buffer alloc failed");
        sf->err_code = GEIST_E_OOM;
        return GEIST_E_OOM;
    }
    const uint64_t t_enc0 = monotonic_ns();
    size_t         n_soft = enc_ops->encode_image(enc_st, rgb, height, width, soft, max_soft);
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

    session_attach(sf);
    const uint64_t    t_pre0 = monotonic_ns();
    enum geist_status ps = dec_ops->prefill_image(sf->model->text_decoder.arch_meta, n_soft, soft);
    sf->total_prefill_ns += monotonic_ns() - t_pre0;
    safe_free((void **) &soft);
    if (ps != GEIST_OK) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "attach_image: prefill failed");
        sf->err_code = ps;
        return ps;
    }
    return GEIST_OK;
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
    const size_t max_soft = (size_t) 70 * n_frames;
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *soft     = heap_alloc_array_aligned(float, max_soft *soft_dim);
    if (soft == nullptr) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "attach_video: soft-token buffer alloc failed");
        sf->err_code = GEIST_E_OOM;
        return GEIST_E_OOM;
    }
    const uint64_t t_enc0 = monotonic_ns();
    size_t n_soft = enc_ops->encode_video(enc_st, frames, n_frames, height, width, soft, max_soft);
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

    session_attach(sf);
    const uint64_t    t_pre0 = monotonic_ns();
    enum geist_status ps = dec_ops->prefill_image(sf->model->text_decoder.arch_meta, n_soft, soft);
    sf->total_prefill_ns += monotonic_ns() - t_pre0;
    safe_free((void **) &soft);
    if (ps != GEIST_OK) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "attach_video: prefill failed");
        sf->err_code = ps;
        return ps;
    }
    return GEIST_OK;
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
    session_attach(sf);
    enum geist_status ps = ops->pin_prefix(sf->model->text_decoder.arch_meta, n, ids);
    if (ps != GEIST_OK) {
        snprintf(sf->err_msg, sizeof(sf->err_msg), "pin_prefix: prefill failed");
        sf->err_code = ps;
        return ps;
    }
    return GEIST_OK;
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
