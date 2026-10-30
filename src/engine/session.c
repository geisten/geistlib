/*
 * src/engine/session.c — geist_session lifecycle + decode dispatch.
 *
 * Layer: ENGINE.
 *
 * Each session holds a reference to the model and dispatches set_prompt,
 * prefill, decode_step, and attach_audio through the model's arch_ops
 * vtables. An arch with session_alloc gives each session its own arch
 * session; otherwise the model's arch_state is its one session.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "model.h"
#include "error.h"

#include <geist_arch.h> /* arch_ops vtables the engine dispatches through */

#include <stdatomic.h>

#include "checked.h"
#include "clock.h"
#include "heap.h"
#include "parse.h"
#include "image_pipeline.h"
#include "dry_breakers.h"
#include "sampler.h"
#include "gguf_tokenizer.h"

#include <geist.h>
#include <geist_util.h> /* tokenize/prefill/attach/peek/speculative/stats */

#include <stdalign.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* geist_session_token_to_str's strings. A token is decoded the first time
 * the session asks for it, NUL-terminated, into blocks that are neither
 * moved nor freed before geist_session_destroy, so the pointer is stable for
 * the session's lifetime as the STABLE contract says. A repeat is one probe
 * of an open-addressing map from id to string. Loading pays nothing, and
 * memory follows the tokens converted rather than the vocabulary. Map and
 * blocks grow geometrically, like a high-water buffer (AGENT.md §3): a
 * session converting tokens it has converted before allocates nothing.
 *
 * Called once per token right after a decode step has evicted the caches,
 * so the free end of the newest block is kept here, next to the map, and a
 * block is only walked to free it. */
struct token_str_block {
    struct token_str_block *next;
    char                    bytes[];
};

struct token_strs {
    geist_token_t          *id;        /* cap slots, -1 empty: the probe reads only these */
    const char            **str;       /* the string of id[i] */
    size_t                  cap;       /* a power of two; 0 before the first call */
    size_t                  n;         /* slots in use, at most cap / 2 */
    char                   *tail;      /* the newest block's first free byte */
    size_t                  room;      /* bytes free from tail */
    size_t                  block_cap; /* the newest block's size */
    struct token_str_block *block;     /* newest first */
};

struct geist_session_full {
    struct geist_model   *model;
    struct geist_backend *backend;

    /* Per-session arch state, from the arch's session_alloc. */
    void *arch_session;

    /* geist_session_set_draft (#192): a small model's session that drafts
     * for decode_speculative; the caller owns it. */
    struct geist_session *draft;

    /* Sampler defaults from geist_session_opts. */
    float    temperature;
    float    top_p;
    int      top_k;
    uint64_t random_seed;

    enum geist_status err_code;
    char              err_msg[512];

    /* Counters for geist_session_get_stats. Timing uses
     * geist_now_ns (clock.h); deltas are accumulated at each
     * top-level entry point (decode_step, prefill_tokens, attach_audio,
     * decode_speculative). */
    uint64_t n_tokens_decoded;
    uint64_t total_decode_ns;
    uint64_t total_prefill_ns;
    uint64_t total_audio_encode_ns;
    uint64_t total_vision_encode_ns; /* attach_image, attach_video */

    /* Streaming audio turn (#256). samples is _Atomic because push may
     * run on a capture thread while end reads it on the session thread.
     * buf/injected are session-thread-only (poll/end). */
    bool           audio_streaming;
    _Atomic size_t audio_stream_samples;
    float         *audio_stream_buf; /* bound-sized scratch, alive begin→end */
    size_t         audio_stream_cap; /* tokens the scratch can hold */
    size_t         audio_stream_injected;

    struct token_strs strs; /* geist_session_token_to_str; empty until then */

    /* geist_session_cancel (#628): set from any thread, consumed by the
     * call that sees it (cancel_requested, through the arch's check). */
    _Atomic bool cancel;
};

/* Engine-internal: cast the public opaque handle to the full struct. */
static inline struct geist_session_full *as_full(struct geist_session *s) {
    return (struct geist_session_full *) s;
}

/* The session handle passed to per-session vtable ops: the arch session
 * this geist_session owns. */
static inline void *arch_sess(const struct geist_session_full *sf) {
    return sf->arch_session;
}

/* ---- struct token_strs ---------------------------------------------------- */

constexpr size_t TOKEN_STRS_SLOTS0    = 256;
constexpr size_t TOKEN_STRS_BLOCK0    = 4096;
constexpr size_t TOKEN_STRS_BLOCK_MAX = 65536;

static size_t token_strs_home(geist_token_t id, size_t cap) {
    const uint32_t h = (uint32_t) id * 0x9E3779B1u;
    return (size_t) (h ^ (h >> 16)) & (cap - 1);
}

static const char *token_strs_find(const struct token_strs *ts, geist_token_t id) {
    if (ts->cap == 0) {
        return nullptr;
    }
    /* At most half full, so an empty slot ends every probe. */
    for (size_t i = token_strs_home(id, ts->cap); ts->id[i] >= 0; i = (i + 1) & (ts->cap - 1)) {
        if (ts->id[i] == id) {
            return ts->str[i];
        }
    }
    return nullptr;
}

static void token_strs_place(size_t        cap,
                             geist_token_t id_slot[static cap],
                             const char   *str_slot[static cap],
                             geist_token_t id,
                             const char   *s) {
    size_t i = token_strs_home(id, cap);
    while (id_slot[i] >= 0) {
        i = (i + 1) & (cap - 1);
    }
    id_slot[i]  = id;
    str_slot[i] = s;
}

/* Room in the map for one more id: it doubles once it would pass half full. */
[[nodiscard]] static bool token_strs_make_room(struct token_strs *ts) {
    if (2 * (ts->n + 1) <= ts->cap) {
        return true;
    }
    const size_t   cap = ts->cap != 0 ? 2 * ts->cap : TOKEN_STRS_SLOTS0;
    geist_token_t *id  = heap_alloc_array_aligned(geist_token_t, cap);
    const char   **str = heap_alloc_array_aligned(const char *, cap);
    if (id == nullptr || str == nullptr) {
        safe_free((void **) &id);
        safe_free((void **) &str);
        return false;
    }
    for (size_t i = 0; i < cap; i++) {
        id[i] = -1;
    }
    for (size_t i = 0; i < ts->cap; i++) {
        if (ts->id[i] >= 0) {
            token_strs_place(cap, id, str, ts->id[i], ts->str[i]);
        }
    }
    safe_free((void **) &ts->id);
    safe_free((void **) &ts->str);
    ts->id  = id;
    ts->str = str;
    ts->cap = cap;
    return true;
}

/* need free bytes at ts->tail, starting a new block when the newest has
 * fewer: twice its size up to 64 KB, or exactly need for a longer token.
 * nullptr when that cannot be allocated. */
[[nodiscard]] static char *token_strs_room(struct token_strs *ts, size_t need) {
    if (ts->room >= need) {
        return ts->tail;
    }
    size_t cap = ts->block == nullptr                        ? TOKEN_STRS_BLOCK0
                 : ts->block_cap >= TOKEN_STRS_BLOCK_MAX / 2 ? TOKEN_STRS_BLOCK_MAX
                                                             : 2 * ts->block_cap;
    if (cap < need) {
        cap = need;
    }
    size_t bytes = 0;
    if (ckd_add(&bytes, sizeof(struct token_str_block), cap)) {
        return nullptr;
    }
    struct token_str_block *b = heap_alloc_aligned(bytes, alignof(struct token_str_block));
    if (b == nullptr) {
        return nullptr;
    }
    b->next       = ts->block;
    ts->block     = b;
    ts->tail      = b->bytes;
    ts->room      = cap;
    ts->block_cap = cap;
    return ts->tail;
}

static void token_strs_free(struct token_strs *ts) {
    safe_free((void **) &ts->id);
    safe_free((void **) &ts->str);
    while (ts->block != nullptr) {
        struct token_str_block *next = ts->block->next;
        safe_free((void **) &ts->block);
        ts->block = next;
    }
    *ts = (struct token_strs) {0};
}

/* Record a session error (code + formatted message) and return the code. */
[[gnu::format(printf, 3, 4)]] static enum geist_status
session_fail(struct geist_session_full *sf, enum geist_status code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(sf->err_msg, sizeof sf->err_msg, fmt, ap);
    va_end(ap);
    sf->err_code = code;
    return code;
}

/* Record a failed per-session dispatch. The op's status return IS the
 * control flow; the backend error slot only contributes detail text for
 * the session's error message. */
static enum geist_status
session_op_failed(struct geist_session_full *sf, enum geist_status s, const char *what) {
    /* The backend's message belongs to THIS failure only if its sticky code
     * still matches. Optional probes (a missing untied output.weight at
     * load) leave a stale message behind, and reporting it verbatim points
     * every debugger at the wrong tensor. */
    const bool fresh = geist_backend_errcode(sf->backend) == s;
    (void) session_fail(sf,
                        s,
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

/* The arch's view of geist_session_cancel: true once per request. */
static bool cancel_requested(void *ctx) {
    struct geist_session_full *sf = ctx;
    return atomic_load_explicit(&sf->cancel, memory_order_relaxed) &&
           atomic_exchange_explicit(&sf->cancel, false, memory_order_acq_rel);
}

void geist_session_cancel(struct geist_session *s) {
    if (s != nullptr) {
        atomic_store_explicit(&as_full(s)->cancel, true, memory_order_release);
    }
}

/* DRY sequence breakers (#695): the opts' strings, or llama.cpp's defaults,
 * matched against the vocabulary here, where the tokenizer is, and handed
 * to the architecture as token sequences. Only when DRY is on. */
[[nodiscard]] static enum geist_status
set_dry_breakers(struct geist_session_full                 *sf,
                 const struct geist_sampler_penalty_params *p,
                 const struct geist_session_opts           *opts) {
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (!geist_sampler_dry_on(p) || ops == nullptr || ops->set_dry_breakers == nullptr) {
        return GEIST_OK;
    }
    const char *const *strs   = opts->dry_sequence_breakers;
    size_t             n_strs = opts->n_dry_sequence_breakers;
    if (strs == nullptr) {
        strs   = geist_dry_default_breakers;
        n_strs = sizeof geist_dry_default_breakers / sizeof geist_dry_default_breakers[0];
    }
    /* A tokenizer.bin has never resolved them. */
    const struct gguf_tokenizer *tok     = geist_model_internal_gguf_tokenizer(sf->model);
    size_t                       n_words = 0;
    geist_token_t               *packed  = nullptr;
    enum geist_status            s       = geist_dry_breakers_resolve(
            &n_words, &packed, tok != nullptr && !tok->tokenizer_bin ? tok : nullptr, n_strs, strs);
    if (s == GEIST_OK && n_words > 0) {
        s = ops->set_dry_breakers(arch_sess(sf), n_words, packed);
    }
    safe_free((void **) &packed);
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
    if (!geist_model_accepts_backend(m, be)) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG,
                                    "geist_session_create",
                                    "backend does not match the one the model was loaded on "
                                    "(model: %s, given: %s)",
                                    m->backend != nullptr ? geist_backend_name(m->backend)
                                                          : "(null)",
                                    geist_backend_name(be));
        return GEIST_E_INVALID_ARG;
    }

    /* Repetition control (#695): refuse invalid values before anything is
     * allocated; the architecture resolves the same parameters again. */
    struct geist_sampler_penalty_params pen;
    if (geist_sampler_penalty_params_from_opts(&pen, opts) != GEIST_OK) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG,
                                    "geist_session_create",
                                    "invalid repetition penalty options (a value is not finite, "
                                    "a window or length is negative, repeat_penalty < 0 or "
                                    "dry_base in (0, 1))");
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
            .model                  = m,
            .backend                = m->backend,
            .arch_session           = nullptr,
            .temperature            = opts != nullptr ? opts->temperature : 0.0f,
            .top_p                  = opts != nullptr ? opts->top_p : 1.0f,
            .top_k                  = opts != nullptr ? opts->top_k : 0,
            .random_seed            = opts != nullptr ? opts->random_seed : 0,
            .err_code               = GEIST_OK,
            .n_tokens_decoded       = 0,
            .total_decode_ns        = 0,
            .total_prefill_ns       = 0,
            .total_audio_encode_ns  = 0,
            .total_vision_encode_ns = 0,
    };
    sf->err_msg[0] = '\0';

    /* Every decoder arch has session_alloc (model_arch_gate refuses one
     * without): the session gets its own KV cache + scratch pool + sampler
     * RNG, separate from any other session on the same model. session_alloc
     * applies the opts, so the session-level temperature / top_p / top_k /
     * seed are wired up. */
    const struct geist_arch_ops_decoder *ops = m->text_decoder.arch_ops;
    sf->arch_session                         = ops->session_alloc(m->text_decoder.arch_meta, opts);
    if (sf->arch_session == nullptr) {
        geist_error_set_create_time(
                GEIST_E_OOM, "geist_session_create", "arch session_alloc returned null");
        void *tmp = sf;
        safe_free(&tmp);
        return GEIST_E_OOM;
    }
    if (ops->bind_cancel != nullptr) {
        ops->bind_cancel(sf->arch_session, cancel_requested, sf);
    }

    if (opts != nullptr) {
        const enum geist_status ds = set_dry_breakers(sf, &pen, opts);
        if (ds != GEIST_OK) {
            geist_error_set_create_time(
                    ds, "geist_session_create", "DRY sequence breakers could not be resolved");
            geist_session_destroy((struct geist_session *) sf);
            return ds;
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
 * Both halves matter. The scratch would leak if a session is destroyed
 * mid-turn. And the encoder stream is state inside the ENCODER, shared across
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
    /* Release this session's arch state if it owns one. */
    if (sf->arch_session != nullptr) {
        ops->session_free(sf->model->text_decoder.arch_meta, sf->arch_session);
        sf->arch_session = nullptr;
    }
    token_strs_free(&sf->strs);
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
    struct gguf_tokenizer               *gtok = geist_model_internal_gguf_tokenizer(sf->model);
    if (ops == nullptr || ops->prefill == nullptr) {
        return GEIST_E_INVALID_STATE;
    }
    if (gtok == nullptr) {
        return session_fail(sf,
                            GEIST_E_NOT_FOUND,
                            "set_prompt: no tokenizer found. SentencePiece path needs "
                            "tokenizer.bin alongside the model (set GEIST_TOKENIZER_PATH "
                            "or place it in the model dir); Llama-family path needs the "
                            "tokenizer embedded in the GGUF (general.architecture + "
                            "tokenizer.ggml.* keys). Falling back to "
                            "geist_session_prefill_tokens with pre-tokenized IDs.");
    }

    /* ≤ 1 id per byte + ▁ prefix + BOS. An SPM vocab without a ▁ piece
     * spends 3 byte ids per space; encode then fails, it never truncates —
     * except for a tokenizer.bin, whose room covers that (it never failed). */
    size_t cap;
    if (ckd_mul(&cap, strlen(prompt), gtok->tokenizer_bin ? (size_t) 3 : (size_t) 1) ||
        ckd_add(&cap, cap, 8)) {
        return session_fail(sf, GEIST_E_OOM, "set_prompt: prompt too long");
    }
    int32_t *enc_ids = heap_alloc_array_aligned(int32_t, cap);
    if (enc_ids == nullptr) {
        return session_fail(sf, GEIST_E_OOM, "set_prompt: alloc fail");
    }
    /* Exactly one BOS when the model expects it (add_bos), on both tokenizer
     * paths (#263): prepended unless the prompt's own tokens begin with it,
     * as a rendered chat template's ("<bos>...") do. The prompt encodes
     * from enc_ids[1]; enc_ids[0] holds the BOS when it is prepended. */
    size_t enc_n = 0;
    if (gguf_tokenizer_encode(gtok, prompt, cap - 1, enc_ids + 1, &enc_n) != GEIST_OK) {
        safe_free((void **) &enc_ids);
        return session_fail(sf, GEIST_E_IO, "set_prompt: gguf_tokenizer_encode failed");
    }
    const bool prepend =
            gtok->bos_id >= 0 && gtok->add_bos && (enc_n == 0 || enc_ids[1] != gtok->bos_id);
    enc_ids[0]                    = gtok->bos_id;
    const int32_t          *ids   = prepend ? enc_ids : enc_ids + 1;
    const size_t            n_enc = prepend ? enc_n + 1 : enc_n;
    const uint64_t          t0    = geist_now_ns();
    const enum geist_status ps    = ops->prefill(arch_sess(sf), n_enc, (const geist_token_t *) ids);
    sf->total_prefill_ns += geist_now_ns() - t0;
    safe_free((void **) &enc_ids);
    return ps == GEIST_OK ? GEIST_OK : session_op_failed(sf, ps, "prefill");
}

[[nodiscard]] enum geist_status geist_session_tokenize(struct geist_session *s,
                                                       const char           *text,
                                                       size_t                out_capacity,
                                                       geist_token_t        *out_ids,
                                                       size_t               *n_out) {
    if (s == nullptr || text == nullptr || n_out == nullptr ||
        (out_capacity > 0 && out_ids == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    *n_out                        = 0;
    struct geist_session_full *sf = as_full(s);

    /* No BOS prepended — tokenize reports the content tokens; callers add BOS
     * if they need it. */
    struct gguf_tokenizer *gtok = geist_model_internal_gguf_tokenizer(sf->model);
    if (gtok == nullptr) {
        return session_fail(
                sf,
                GEIST_E_NOT_FOUND,
                "tokenize: no tokenizer found (GGUF-embedded or external tokenizer.bin)");
    }
    const enum geist_status st = gguf_tokenizer_encode(gtok, text, out_capacity, out_ids, n_out);
    if (st != GEIST_OK) {
        return session_fail(sf,
                            st,
                            "tokenize: %s (out_capacity %zu)",
                            st == GEIST_E_INVALID_ARG ? "more tokens than fit"
                                                      : "gguf encode failed",
                            out_capacity);
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
geist_session_prefill_tokens(struct geist_session *s, size_t n, const geist_token_t *ids) {
    if (s == nullptr || n == 0 || ids == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->prefill == nullptr) {
        return GEIST_E_INVALID_STATE;
    }
    const uint64_t          t0 = geist_now_ns();
    const enum geist_status ps = ops->prefill(arch_sess(sf), n, ids);
    sf->total_prefill_ns += geist_now_ns() - t0;
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
    if (cancel_requested(sf)) {
        return session_fail(sf, GEIST_E_CANCELLED, "decode_step cancelled");
    }
    const uint64_t          t0 = geist_now_ns();
    const enum geist_status ds = ops->decode_step(arch_sess(sf), out_token);
    sf->total_decode_ns += geist_now_ns() - t0;
    if (ds != GEIST_OK) {
        return session_op_failed(sf, ds, "decode_step");
    }
    sf->n_tokens_decoded++;
    return GEIST_OK;
}

size_t geist_model_decode_batch_max(const struct geist_model *m) {
    if (m == nullptr) {
        return 0;
    }
    const struct geist_arch_ops_decoder *ops = m->text_decoder.arch_ops;
    if (ops == nullptr || ops->decode_batch == nullptr || ops->decode_batch_max == nullptr) {
        return 0;
    }
    return ops->decode_batch_max(m->text_decoder.arch_meta);
}

[[nodiscard]] enum geist_status geist_session_decode_batch(size_t                      n,
                                                           struct geist_session *const sessions[],
                                                           geist_token_t out_tokens[]) {
    if (out_tokens != nullptr) {
        for (size_t b = 0; b < n; b++) {
            out_tokens[b] = -1;
        }
    }
    if (n == 0 || sessions == nullptr || out_tokens == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n == 1) {
        return geist_session_decode_step(sessions[0], &out_tokens[0]);
    }
    if (sessions[0] == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_model *model = as_full(sessions[0])->model;
    if (n > geist_model_decode_batch_max(model)) {
        return GEIST_E_UNSUPPORTED;
    }
    /* Bounded by decode_batch_max: no allocation on the decode path. */
    void *arch[64];
    if (n > sizeof arch / sizeof arch[0]) {
        return GEIST_E_UNSUPPORTED;
    }
    for (size_t b = 0; b < n; b++) {
        if (sessions[b] == nullptr || as_full(sessions[b])->model != model) {
            return GEIST_E_INVALID_ARG;
        }
        if (cancel_requested(as_full(sessions[b]))) {
            return GEIST_E_CANCELLED;
        }
        arch[b] = arch_sess(as_full(sessions[b]));
    }
    const uint64_t          t0 = geist_now_ns();
    const enum geist_status ds = model->text_decoder.arch_ops->decode_batch(n, arch, out_tokens);
    const uint64_t          dt = geist_now_ns() - t0;
    for (size_t b = 0; b < n; b++) {
        struct geist_session_full *sf = as_full(sessions[b]);
        sf->total_decode_ns += dt / n;
        if (ds == GEIST_OK) {
            sf->n_tokens_decoded++;
        } else {
            (void) session_op_failed(sf, ds, "decode_batch");
        }
    }
    return ds;
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

/* Verify k candidate tokens (drafts[0] the pending seed, drafts[1..] the
 * guesses after it) in one forward, accept the longest matching prefix, roll
 * back the rest and push the bonus or correction token (shared by
 * decode_speculative and verify_drafts). */
[[nodiscard]] static enum geist_status spec_verify(struct geist_session_full           *sf,
                                                   const struct geist_arch_ops_decoder *ops,
                                                   void                                *st,
                                                   size_t                               k,
                                                   const geist_token_t                 *drafts,
                                                   geist_token_t                       *out_tokens,
                                                   size_t                              *n_out,
                                                   uint64_t                             t_start) {
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
        /* All drafts verified: emit them + the model's bonus prediction. */
        correction      = verify_out[k - 1];
        accepted_tokens = k;
    } else {
        /* Partial accept. Keep KV[..kv_before + accepted_extras + 1) (= the
         * accepted draft positions) and discard the rest; recurrent state
         * replays that prefix. */
        const enum geist_status ts = ops->kv_truncate(st, kv_before + accepted_extras + 1);
        if (ts != GEIST_OK) {
            return session_op_failed(sf, ts, "speculative state rollback");
        }
        correction      = verify_out[accepted_extras];
        accepted_tokens = accepted_extras + 1;
    }

    memcpy(out_tokens, drafts, accepted_tokens * sizeof(geist_token_t));
    out_tokens[accepted_tokens] = correction;
    const size_t emitted        = accepted_tokens + 1;

    /* The last emit (bonus or correction) is not yet in the cache. The next
     * spec_step / decode_step needs pending logits computed from it, so push
     * it now via a single-token prefill. Deferring it to the next call saves
     * this prefill but shifts the call boundaries the n-gram drafter sees and
     * costs more than it saves (benchmark/results/QWEN35.md). */
    const enum geist_status cs = ops->prefill(st, 1, &correction);
    if (cs != GEIST_OK) {
        return session_op_failed(sf, cs, "speculative correction prefill");
    }

    *n_out = emitted;
    sf->n_tokens_decoded += emitted;
    sf->total_decode_ns += geist_now_ns() - t_start;
    return GEIST_OK;
}

/* A draft model's session drafts (#192). It is brought to the target's
 * committed tokens (history; usually the previous call's correction is all
 * that is missing), then the seed and k_max - 1 greedy steps follow, with a
 * checkpoint after the seed and after every step but the last, so that
 * draft_model_rollback can return to seed + accepted drafts. */
[[nodiscard]] static enum geist_status draft_model_propose(struct geist_session *d,
                                                           size_t                history_n,
                                                           const geist_token_t  *history,
                                                           geist_token_t         seed,
                                                           size_t                k_max,
                                                           geist_token_t        *drafts,
                                                           size_t               *k) {
    size_t have = geist_session_length(d);
    if (have > history_n) {
        /* ahead of the target (an unexpected path): back to it, or start over */
        if (geist_session_truncate(d, history_n) == GEIST_OK) {
            have = history_n;
        } else {
            if (geist_session_reset(d) != GEIST_OK)
                return GEIST_E_INVALID_STATE;
            have = 0;
        }
    }
    enum geist_status s = GEIST_OK;
    if (have < history_n)
        s = geist_session_prefill_tokens(d, history_n - have, history + have);
    if (s == GEIST_OK)
        s = geist_session_prefill_tokens(d, 1, &seed);
    /* Without checkpoints (e.g. the KIVI KV cache) a rejected guess makes
     * the next call start the draft over: slower, still exact. */
    bool ckpt = true;
    if (s == GEIST_OK) {
        s = geist_session_checkpoint(d);
        if (s == GEIST_E_UNSUPPORTED) {
            ckpt = false;
            s    = GEIST_OK;
        }
    }
    drafts[0] = seed;
    *k        = 1;
    for (size_t i = 0; s == GEIST_OK && i + 1 < k_max; i++) {
        s = geist_session_decode_step(d, &drafts[*k]);
        if (s == GEIST_OK)
            ++*k;
        if (s == GEIST_OK && ckpt && i + 2 < k_max)
            s = geist_session_checkpoint(d);
    }
    return s;
}

/* After the verify: the draft keeps seed + the accepted guesses. n_out is
 * spec_verify's count, accepted guesses + seed + correction. A failed
 * rollback leaves the draft ahead; draft_model_propose then starts over. */
static void
draft_model_rollback(struct geist_session *d, size_t history_n, size_t guesses, size_t n_out) {
    const size_t accepted = n_out >= 2 ? n_out - 2 : 0;
    if (accepted < guesses)
        (void) geist_session_truncate(d, history_n + 1 + accepted);
}

[[nodiscard]] enum geist_status geist_session_set_draft(struct geist_session *s,
                                                        struct geist_session *draft) {
    if (s == nullptr || draft == s) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full *sf = as_full(s);
    if (draft == nullptr) {
        sf->draft = nullptr;
        return GEIST_OK;
    }
    struct geist_session_full   *df = as_full(draft);
    const struct gguf_tokenizer *a  = geist_model_internal_gguf_tokenizer(sf->model);
    const struct gguf_tokenizer *b  = geist_model_internal_gguf_tokenizer(df->model);
    /* Vocabularies from a tokenizer.bin have never been compared. */
    bool same = sf->model == df->model || (a != nullptr && b != nullptr && !a->tokenizer_bin &&
                                           !b->tokenizer_bin && a->vocab_size == b->vocab_size);
    for (size_t t = 0; same && sf->model != df->model && t < a->vocab_size; t++) {
        same = a->token_len[t] == b->token_len[t] &&
               memcmp(a->token_str[t], b->token_str[t], a->token_len[t]) == 0;
    }
    if (!same) {
        return session_fail(
                sf, GEIST_E_INVALID_ARG, "set_draft: the draft model's vocabulary differs");
    }
    sf->draft = draft;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status geist_session_decode_speculative(struct geist_session *s,
                                                                 size_t                k_max,
                                                                 size_t                history_n,
                                                                 const geist_token_t  *history,
                                                                 size_t                out_capacity,
                                                                 geist_token_t        *out_tokens,
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

    const uint64_t      t_start = geist_now_ns();
    const geist_token_t seed    = ops->peek_next_token(st);
    if (seed < 0)
        return spec_fallback_single(s, out_tokens, n_out);

    geist_token_t drafts[GEIST_SPEC_KMAX_HARDCAP];
    size_t        match_L      = 0;
    size_t        k            = 0;
    bool          native_draft = false;
    if (sf->draft != nullptr) {
        const enum geist_status ds =
                draft_model_propose(sf->draft, history_n, history, seed, k_max, drafts, &k);
        if (ds != GEIST_OK)
            return session_op_failed(sf, ds, "draft model");
        const enum geist_status vs =
                spec_verify(sf, ops, st, k, drafts, out_tokens, n_out, t_start);
        if (vs == GEIST_OK)
            draft_model_rollback(sf->draft, history_n, k - 1, *n_out);
        return vs;
    }
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

    /* Lazy-draft gate: skip verify-forward (~1.6-2.3x a single decode) when
     * the n-gram match is shorter than GEIST_SPEC_MIN_L (default 2). A
     * seed-only match (L=1) is accepted too rarely to pay for it. */
    /* Relaxed-atomic first-use cache: concurrent sessions reach this on
     * their own threads. */
    static _Atomic int min_L_cached = -1;
    int                min_L        = atomic_load_explicit(&min_L_cached, memory_order_relaxed);
    if (min_L < 0) {
        const char *env = getenv("GEIST_SPEC_MIN_L");
        long        v;
        if (!geist_parse_long(env, &v))
            v = 2;
        min_L = (v <= 0) ? 1 : (v > 8 ? 8 : (int) v);
        atomic_store_explicit(&min_L_cached, min_L, memory_order_relaxed);
    }
    if (!native_draft && (int) match_L < min_L) {
        return spec_fallback_single(s, out_tokens, n_out);
    }

    return spec_verify(sf, ops, st, k, drafts, out_tokens, n_out, t_start);
}

[[nodiscard]] enum geist_status geist_session_verify_drafts(struct geist_session *s,
                                                            size_t                n_drafts,
                                                            const geist_token_t  *drafts,
                                                            size_t                out_capacity,
                                                            geist_token_t        *out_tokens,
                                                            size_t               *n_out) {
    if (s == nullptr || n_out == nullptr || (n_drafts > 0 && drafts == nullptr) ||
        out_tokens == nullptr || n_drafts + 1 > GEIST_SPEC_KMAX_HARDCAP ||
        out_capacity < n_drafts + 2) {
        return GEIST_E_INVALID_ARG;
    }
    *n_out                                   = 0;
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    void                                *st  = sf->arch_session;
    if (ops == nullptr)
        return GEIST_E_INVALID_STATE;
    const bool can_spec = ops->peek_next_token != nullptr && ops->verify_forward != nullptr &&
                          ops->kv_truncate != nullptr && ops->kv_len != nullptr;
    if (!can_spec || n_drafts == 0)
        return spec_fallback_single(s, out_tokens, n_out);
    const uint64_t      t_start = geist_now_ns();
    const geist_token_t seed    = ops->peek_next_token(st);
    if (seed < 0)
        return spec_fallback_single(s, out_tokens, n_out);
    geist_token_t rows[GEIST_SPEC_KMAX_HARDCAP];
    rows[0] = seed;
    memcpy(rows + 1, drafts, n_drafts * sizeof(geist_token_t));
    return spec_verify(sf, ops, st, n_drafts + 1, rows, out_tokens, n_out, t_start);
}

/* The surface form comes from the model's tokenizer: the GGUF-embedded one
 * decodes (byte-level BPE maps its codepoints back to bytes, SentencePiece
 * ▁ to a space and <0xXX> to the byte), a tokenizer.bin piece is taken as
 * stored (tokenizer_bin). Either way it is kept in the session (struct
 * token_strs), where later calls cannot overwrite it; like every session
 * call, not from two threads at once. An id past a GGUF vocab reads "<unk>", as the decoder
 * writes it; past a tokenizer.bin vocab there is nothing. */
const char *geist_session_token_to_str(struct geist_session *s, geist_token_t t) {
    if (s == nullptr || t < 0) {
        return nullptr;
    }
    struct geist_session_full *sf  = as_full(s);
    const char                *hit = token_strs_find(&sf->strs, t);
    if (hit != nullptr) {
        return hit;
    }
    struct gguf_tokenizer *gtok = geist_model_internal_gguf_tokenizer(sf->model);
    if (gtok == nullptr) {
        return nullptr;
    }
    if ((size_t) t >= gtok->vocab_size) {
        return gtok->tokenizer_bin ? nullptr : "<unk>";
    }
    /* the tokenizer.bin piece; else decoding, which only ever shortens a token */
    const char *piece = gtok->tokenizer_bin ? gtok->token_str[t] : nullptr;
    size_t      len   = gtok->token_len[t];
    size_t      need  = 0;
    char       *dst   = nullptr;
    if (!ckd_add(&need, len, 1) && token_strs_make_room(&sf->strs)) {
        dst = token_strs_room(&sf->strs, need);
    }
    if (dst == nullptr) {
        (void) session_fail(sf, GEIST_E_OOM, "token_to_str: out of memory for token %d", t);
        return nullptr;
    }
    if (piece != nullptr) {
        memcpy(dst, piece, len);
    } else {
        const int32_t id = (int32_t) t;
        len              = gguf_tokenizer_decode(gtok, &id, 1, dst, need);
        if (len >= need) { /* longer than stored: refuse rather than cut */
            (void) session_fail(sf,
                                GEIST_E_INTERNAL,
                                "token_to_str: token %d decodes longer "
                                "than it is stored",
                                t);
            return nullptr;
        }
    }
    dst[len] = '\0';
    sf->strs.tail += len + 1;
    sf->strs.room -= len + 1;
    token_strs_place(sf->strs.cap, sf->strs.id, sf->strs.str, t, dst);
    sf->strs.n++;
    return dst;
}

/* The tail every attach_* shares: n_soft soft tokens the encoder wrote into
 * `soft` (heap, freed here) go into the decoder through `prefill`, timed as
 * prefill. An encoder that produced none fails with `empty` (GEIST_E_IO). */
[[nodiscard]] static enum geist_status
inject_soft(struct geist_session_full *sf,
            size_t                     n_soft,
            float                     *soft,
            enum geist_status (*prefill)(void *session, size_t n, const float *soft_tokens),
            const char *op,
            const char *empty) {
    if (n_soft == 0) {
        safe_free((void **) &soft);
        return session_fail(sf, GEIST_E_IO, "%s", empty);
    }
    const uint64_t          t0 = geist_now_ns();
    const enum geist_status ps = prefill(arch_sess(sf), n_soft, soft);
    sf->total_prefill_ns += geist_now_ns() - t0;
    safe_free((void **) &soft);
    return ps == GEIST_OK ? GEIST_OK : session_op_failed(sf, ps, op);
}

/* Audio path: PCM → mel → audio_conformer encode → soft tokens →
 * decoder arch_ops->prefill_audio. */
[[nodiscard]] enum geist_status geist_session_attach_audio(struct geist_session *s,
                                                           size_t                n_samples,
                                                           const int16_t        *pcm_samples,
                                                           int                   sample_rate) {
    if (s == nullptr || n_samples == 0 || pcm_samples == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf      = as_full(s);
    const struct geist_arch_ops_encoder *enc_ops = sf->model->audio_encoder.arch_ops;
    void                                *enc_st  = sf->model->audio_encoder.arch_meta;
    const struct geist_arch_ops_decoder *dec_ops = sf->model->text_decoder.arch_ops;
    if (enc_ops == nullptr || enc_st == nullptr) {
        return session_fail(sf,
                            GEIST_E_NOT_FOUND,
                            "attach_audio: model has no audio encoder loaded "
                            "(missing audio_tower.safetensors or mel_constants.bin)");
    }
    if (dec_ops == nullptr || dec_ops->prefill_audio == nullptr) {
        return session_fail(
                sf, GEIST_E_UNSUPPORTED, "attach_audio: decoder arch lacks audio injection path");
    }
    if (sample_rate != 16000) {
        return session_fail(sf,
                            GEIST_E_UNSUPPORTED,
                            "attach_audio: only 16 kHz PCM supported (got %d Hz)",
                            sample_rate);
    }

    /* Size the soft-token buffer from the audio length so long audio is
     * never cut (#247). The fallback is for encoders without the op. */
    size_t max_soft = 256;
    if (enc_ops->max_soft_tokens != nullptr) {
        max_soft = enc_ops->max_soft_tokens(enc_st, n_samples);
    }
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *soft     = heap_alloc_array_aligned(float, max_soft *soft_dim);
    if (soft == nullptr) {
        return session_fail(sf, GEIST_E_OOM, "attach_audio: soft-token buffer alloc failed");
    }
    const uint64_t t_enc0 = geist_now_ns();
    size_t         n_soft = enc_ops->encode_pcm(enc_st, n_samples, max_soft, pcm_samples, soft);
    sf->total_audio_encode_ns += geist_now_ns() - t_enc0;
    if (n_soft != 0 && n_soft >= max_soft) {
        /* The bound carries headroom, so a full buffer can only mean the
         * encoder hit the cap — truncated audio must be an error, not a
         * silent drop (#247). */
        safe_free((void **) &soft);
        return session_fail(
                sf,
                GEIST_E_INTERNAL,
                "attach_audio: soft-token bound too small (%zu tokens for %zu samples) — "
                "audio would be truncated",
                max_soft,
                n_samples);
    }

    return inject_soft(sf,
                       n_soft,
                       soft,
                       dec_ops->prefill_audio,
                       "prefill_audio",
                       "attach_audio: audio encoder produced 0 soft tokens "
                       "(too-short input, audio beyond the encoder's 30 s limit, "
                       "or encoder failure)");
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
        return session_fail(sf, GEIST_E_NOT_FOUND, "audio_begin: model cannot hear");
    }
    if (enc_ops->stream_begin == nullptr || enc_ops->stream_push == nullptr ||
        enc_ops->stream_end == nullptr) {
        return session_fail(sf, GEIST_E_UNSUPPORTED, "audio_begin: encoder has no streaming path");
    }
    if (sf->audio_streaming) {
        return session_fail(sf, GEIST_E_INVALID_STATE, "audio_begin: streaming turn already open");
    }
    /* Scratch for poll/end, sized for the encoder's 30 s worst case so
     * incremental injection never reallocates mid-turn.
     *
     * Allocated BEFORE stream_begin: every fallible step happens while
     * there is nothing to unwind, so a failure never leaves an encoder
     * stream open that nothing can reach to close. */
    size_t cap = 256;
    if (enc_ops->max_soft_tokens != nullptr) {
        cap = enc_ops->max_soft_tokens(enc_st, (size_t) -1); /* clamped internally */
    }
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *scratch  = heap_alloc_n_aligned(cap, soft_dim * sizeof(float), alignof(float));
    if (scratch == nullptr) {
        return session_fail(sf, GEIST_E_OOM, "audio_begin: scratch alloc failed");
    }
    if (!enc_ops->stream_begin(enc_st)) {
        safe_free((void **) &scratch);
        return session_fail(sf, GEIST_E_BACKEND, "audio_begin: encoder stream_begin failed");
    }
    sf->audio_stream_buf      = scratch;
    sf->audio_stream_cap      = cap;
    sf->audio_stream_injected = 0;
    sf->audio_streaming       = true;
    atomic_store_explicit(&sf->audio_stream_samples, 0, memory_order_relaxed);
    return GEIST_OK;
}

/* Inject whatever soft tokens the encoder has ready (#256):
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
    const uint64_t t0  = geist_now_ns();
    const size_t   got = enc_ops->stream_poll(
            enc_st, sf->audio_stream_cap - sf->audio_stream_injected, sf->audio_stream_buf);
    sf->total_audio_encode_ns += geist_now_ns() - t0;
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
geist_session_audio_push(struct geist_session *s, size_t n, const int16_t *pcm) {
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
        return session_fail(sf,
                            GEIST_E_UNSUPPORTED,
                            "audio_push: encoder refused %zu samples (>30 s buffered?)",
                            n);
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

    const uint64_t t_enc0 = geist_now_ns();
    const size_t n_soft = remaining_cap > 0 ? enc_ops->stream_end(enc_st, remaining_cap, soft) : 0;
    sf->total_audio_encode_ns += geist_now_ns() - t_enc0;

    const size_t total = sf->audio_stream_injected + n_soft;
    if (total == 0) {
        safe_free((void **) &soft);
        return session_fail(sf,
                            GEIST_E_IO,
                            "audio_end: encoder produced 0 soft tokens (no audio pushed, "
                            "or encoder failure)");
    }
    if (total >= sf->audio_stream_cap) {
        safe_free((void **) &soft);
        return session_fail(sf,
                            GEIST_E_INTERNAL,
                            "audio_end: soft-token bound too small (%zu tokens for %zu samples)",
                            sf->audio_stream_cap,
                            n_samples);
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
[[nodiscard]] enum geist_status geist_session_attach_image(struct geist_session *s,
                                                           size_t                height,
                                                           size_t                width,
                                                           const uint8_t        *rgb) {
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
        return session_fail(sf,
                            GEIST_E_NOT_FOUND,
                            "attach_image: model has no vision encoder loaded "
                            "(missing vision_tower.safetensors)");
    }
    if (dec_ops == nullptr || dec_ops->prefill_image == nullptr) {
        return session_fail(
                sf, GEIST_E_UNSUPPORTED, "attach_image: decoder arch lacks vision injection path");
    }

    /* Gemma 4: max 280 soft tokens per image; soft_dim == text hidden size. */
    const size_t max_soft = 280;
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *soft = heap_alloc_n_aligned(max_soft, soft_dim * sizeof(float), alignof(float));
    if (soft == nullptr) {
        return session_fail(sf, GEIST_E_OOM, "attach_image: soft-token buffer alloc failed");
    }
    const uint64_t t_enc0 = geist_now_ns();
    size_t         n_soft = enc_ops->encode_image(enc_st, height, width, max_soft, rgb, soft);
    sf->total_vision_encode_ns += geist_now_ns() - t_enc0;
    return inject_soft(sf,
                       n_soft,
                       soft,
                       dec_ops->prefill_image,
                       "prefill_image",
                       "attach_image: vision encoder produced 0 soft tokens "
                       "(degenerate image dims or encoder failure)");
}

/* Video path: per-frame run_image → concat soft tokens →
 * decoder arch_ops->prefill_image. Reuses prefill_image since image
 * and video soft tokens share one wire format. */
[[nodiscard]] enum geist_status geist_session_attach_video(struct geist_session *s,
                                                           size_t                n_frames,
                                                           size_t                height,
                                                           size_t                width,
                                                           const uint8_t        *frames) {
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
        return session_fail(sf,
                            GEIST_E_NOT_FOUND,
                            "attach_video: model has no vision encoder loaded "
                            "(missing vision_tower.safetensors)");
    }
    if (dec_ops == nullptr || dec_ops->prefill_image == nullptr) {
        return session_fail(
                sf, GEIST_E_UNSUPPORTED, "attach_video: decoder arch lacks vision injection path");
    }

    /* 70 soft tokens per frame × n_frames; soft_dim == text hidden size. */
    size_t max_soft = 0;
    if (ckd_mul(&max_soft, (size_t) 70, n_frames)) {
        return session_fail(
                sf, GEIST_E_INVALID_ARG, "attach_video: n_frames %zu is too large", n_frames);
    }
    const size_t soft_dim = enc_ops->soft_token_dim(enc_st);
    float       *soft = heap_alloc_n_aligned(max_soft, soft_dim * sizeof(float), alignof(float));
    if (soft == nullptr) {
        return session_fail(sf, GEIST_E_OOM, "attach_video: soft-token buffer alloc failed");
    }
    const uint64_t t_enc0 = geist_now_ns();
    size_t n_soft = enc_ops->encode_video(enc_st, n_frames, height, width, max_soft, frames, soft);
    sf->total_vision_encode_ns += geist_now_ns() - t_enc0;
    return inject_soft(sf,
                       n_soft,
                       soft,
                       dec_ops->prefill_image,
                       "prefill_image",
                       "attach_video: vision encoder produced 0 soft tokens "
                       "(degenerate frame dims or encoder failure)");
}

[[nodiscard]] enum geist_status
geist_session_pin_prefix(struct geist_session *s, size_t n, const geist_token_t *ids) {
    if (s == nullptr || (n > 0 && ids == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->pin_prefix == nullptr) {
        return session_fail(sf,
                            GEIST_E_UNSUPPORTED,
                            "pin_prefix: active architecture does not support prefix pinning");
    }
    const enum geist_status ps = ops->pin_prefix(arch_sess(sf), n, ids);
    return ps == GEIST_OK ? GEIST_OK : session_op_failed(sf, ps, "pin_prefix");
}

size_t geist_session_length(const struct geist_session *s) {
    if (s == nullptr) {
        return 0;
    }
    const struct geist_session_full     *sf  = (const struct geist_session_full *) s;
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    return ops != nullptr && ops->kv_len != nullptr ? ops->kv_len(arch_sess(sf)) : 0;
}

[[nodiscard]] enum geist_status geist_session_truncate(struct geist_session *s, size_t n) {
    if (s == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->truncate == nullptr) {
        return session_fail(
                sf, GEIST_E_UNSUPPORTED, "truncate: the active architecture cannot truncate");
    }
    const enum geist_status ts = ops->truncate(arch_sess(sf), n);
    return ts == GEIST_OK ? GEIST_OK : session_op_failed(sf, ts, "truncate");
}

[[nodiscard]] enum geist_status geist_session_checkpoint(struct geist_session *s) {
    if (s == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->checkpoint == nullptr) {
        return session_fail(
                sf, GEIST_E_UNSUPPORTED, "checkpoint: the active architecture cannot checkpoint");
    }
    const enum geist_status cs = ops->checkpoint(arch_sess(sf));
    return cs == GEIST_OK ? GEIST_OK : session_op_failed(sf, cs, "checkpoint");
}

[[nodiscard]] enum geist_status geist_session_kv_bytes_per_token(const struct geist_session *s,
                                                                 size_t *out_bytes) {
    if (out_bytes != nullptr) {
        *out_bytes = 0;
    }
    if (s == nullptr || out_bytes == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const struct geist_session_full     *sf  = (const struct geist_session_full *) s;
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->kv_bytes_per_token == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    return ops->kv_bytes_per_token(out_bytes, arch_sess(sf));
}

/* The decoder ops for a snapshot call, or nullptr after recording why
 * there are none (the architecture has no snapshot) or why not now (an
 * audio stream is feeding the session). */
static const struct geist_arch_ops_decoder *snapshot_ops(struct geist_session_full *sf,
                                                         const char                *what) {
    const struct geist_arch_ops_decoder *ops = sf->model->text_decoder.arch_ops;
    if (ops == nullptr || ops->snapshot_size == nullptr || ops->snapshot == nullptr ||
        ops->restore == nullptr) {
        (void) session_fail(sf,
                            GEIST_E_UNSUPPORTED,
                            "%s: active architecture does not support session snapshots",
                            what);
        return nullptr;
    }
    if (sf->audio_streaming) {
        (void) session_fail(sf, GEIST_E_INVALID_STATE, "%s: an audio stream is open", what);
        return nullptr;
    }
    return ops;
}

[[nodiscard]] enum geist_status geist_session_snapshot_size(size_t               *out_bytes,
                                                            struct geist_session *s) {
    if (out_bytes == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_bytes = 0;
    if (s == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = snapshot_ops(sf, "snapshot_size");
    if (ops == nullptr) {
        return sf->err_code;
    }
    const enum geist_status ss = ops->snapshot_size(out_bytes, arch_sess(sf));
    return ss == GEIST_OK ? GEIST_OK : session_op_failed(sf, ss, "snapshot_size");
}

[[nodiscard]] enum geist_status
geist_session_snapshot(size_t *out_bytes, size_t capacity, void *buf, struct geist_session *s) {
    if (out_bytes == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_bytes = 0;
    if (s == nullptr || buf == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = snapshot_ops(sf, "snapshot");
    if (ops == nullptr) {
        return sf->err_code;
    }
    const enum geist_status ss = ops->snapshot(out_bytes, capacity, buf, arch_sess(sf));
    return ss == GEIST_OK ? GEIST_OK : session_op_failed(sf, ss, "snapshot");
}

[[nodiscard]] enum geist_status
geist_session_restore(size_t n_bytes, const void *buf, struct geist_session *s) {
    if (s == nullptr || buf == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_session_full           *sf  = as_full(s);
    const struct geist_arch_ops_decoder *ops = snapshot_ops(sf, "restore");
    if (ops == nullptr) {
        return sf->err_code;
    }
    const enum geist_status rs = ops->restore(n_bytes, buf, arch_sess(sf));
    return rs == GEIST_OK ? GEIST_OK : session_op_failed(sf, rs, "restore");
}

[[nodiscard]] enum geist_status geist_session_get_stats(const struct geist_session *s,
                                                        struct geist_session_stats *out) {
    if (s == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const struct geist_session_full *sf = (const struct geist_session_full *) s;
    *out                                = (struct geist_session_stats) {
            .n_tokens_decoded       = sf->n_tokens_decoded,
            .total_decode_ns        = sf->total_decode_ns,
            .total_prefill_ns       = sf->total_prefill_ns,
            .total_audio_encode_ns  = sf->total_audio_encode_ns,
            .total_vision_encode_ns = sf->total_vision_encode_ns,
            /* buffer_alloc_* and per_op_* stay zero: they need backend-side
             * counters / opt-in op profiling. */
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
    sf->total_vision_encode_ns    = 0;
    return GEIST_OK;
}
