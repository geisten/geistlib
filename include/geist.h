/*
 * geist — Pure C23 multimodal-realtime inference library.
 *
 * Core public API: the minimal surface to load and run a model
 *   backend → model → session → set_prompt → decode_step → token_to_str.
 *
 * Helpers and advanced features (special tokens, tokenization, multimodal
 * attach, speculative decode, raw logits, telemetry, backend-capability
 * queries) live in <geist_util.h>. Low-level tensor / op / dtype types
 * (backend-author territory) live in <geist_types.h>. Everything under
 * src/ is internal and may break between versions without notice.
 *
 * Stability tags per declaration:
 *   @stability STABLE        — won't break in 0.x; deprecation cycle for 1.x.
 *   @stability EXPERIMENTAL  — signature may change in any minor version.
 *
 * See README.md and docs/ARCHITECTURE.md for the design rationale.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/* Version                                                                 */
/* ====================================================================== */

#define GEIST_VERSION_MAJOR 0
#define GEIST_VERSION_MINOR 3
#define GEIST_VERSION_PATCH 0
#define GEIST_VERSION_STRING "0.3.0"

/* @stability STABLE since 0.1.0 */
const char *geist_version_string(void);
void        geist_version_components(int *major, int *minor, int *patch);

/* ====================================================================== */
/* Status / Errors                                                         */
/* ====================================================================== */

enum geist_status {
    GEIST_OK = 0,

    /* Generic */
    GEIST_E_OOM,             /* allocation failed */
    GEIST_E_INVALID_ARG,     /* nullptr where not allowed, bad enum, etc. */
    GEIST_E_INTERNAL,        /* programmer error, shouldn't happen */

    /* I/O */
    GEIST_E_FILE_NOT_FOUND,
    GEIST_E_IO,              /* read/write failed */
    GEIST_E_FORMAT,          /* corrupt file, wrong magic, etc. */

    /* Capability */
    GEIST_E_UNSUPPORTED,     /* backend cannot run this op/dtype/layout */
    GEIST_E_NOT_FOUND,       /* tensor name not in model, etc. */
    GEIST_E_BACKEND,         /* backend-specific failure */

    /* Lifecycle */
    GEIST_E_INVALID_STATE,   /* op called in wrong order */
    GEIST_E_TOO_MANY_TOKENS, /* hit max_seq_len */
};

/* @stability STABLE since 0.1.0
 * Returns a stable identifier string for the status code. Never returns
 * nullptr. Useful for log messages. */
const char *geist_status_to_string(enum geist_status s);

/* @stability STABLE since 0.1.0
 * Thread-local fallback for errors during create-time API (e.g.
 * geist_backend_create) where no handle exists yet. After a handle is
 * obtained, prefer the per-handle errmsg accessors. */
const char *geist_last_create_error(void);

/* ====================================================================== */
/* Logging                                                                 */
/* ====================================================================== */

enum geist_log_level {
    GEIST_LOG_ERROR = 0,
    GEIST_LOG_WARN  = 1,
    GEIST_LOG_INFO  = 2,
    GEIST_LOG_DEBUG = 3,
    GEIST_LOG_TRACE = 4,
};

/* @stability EXPERIMENTAL — categories and call frequency may evolve. */
typedef void (*geist_log_callback_t)(enum geist_log_level level,
                                     const char           *category,
                                     const char           *message,
                                     void                 *user_data);

/* ====================================================================== */
/* Memory / Allocator                                                      */
/* ====================================================================== */

/* @stability STABLE since 0.1.0 */
struct geist_allocator {
    void *(*alloc)(void *ctx, size_t bytes, size_t alignment);
    void  (*free)(void *ctx, void *ptr);
    void  (*free_all)(void *ctx); /* optional, arena-style; nullptr for malloc-based */
    void *ctx;
};

/* libc malloc/free wrapper. Default if user passes nullptr to *_create. */
extern const struct geist_allocator geist_libc_allocator;

/* ====================================================================== */
/* Backend                                                                 */
/* ====================================================================== */

struct geist_backend;

struct geist_backend_opts {
    /* @stability EXPERIMENTAL — additional fields may be added. */
    int max_threads;             /* hint; 0 = backend default */
    int max_concurrent_sessions; /* hint for scratch-pool sizing */

    geist_log_callback_t log_cb;
    void                *log_user_data;
    enum geist_log_level log_level_max; /* WARN by default */

    bool enable_op_profiling; /* opt-in expensive timing */
};

/* @stability STABLE since 0.1.0
 * Create a backend by name (e.g. "cpu_neon", "cpu_scalar", "auto"). The
 * special name "auto" picks the best linked backend for the host. Pass
 * nullptr opts/alloc for defaults. */
enum geist_status geist_backend_create(const char                    *name,
                                       const struct geist_backend_opts *opts,
                                       const struct geist_allocator   *alloc,
                                       struct geist_backend          **out);

void              geist_backend_destroy(struct geist_backend *be);
const char       *geist_backend_name(const struct geist_backend *be);
const char       *geist_backend_errmsg(const struct geist_backend *be);
enum geist_status geist_backend_errcode(const struct geist_backend *be);

/* ====================================================================== */
/* Model                                                                   */
/* ====================================================================== */

struct geist_model;

/* @stability STABLE since 0.1.0
 * Loads a GGUF model file. Architecture is detected from the GGUF
 * `general.architecture` metadata key; returns GEIST_E_UNSUPPORTED if
 * no architecture matching this build's compiled set is registered. */
enum geist_status geist_model_load(const char            *path,
                                   struct geist_backend  *be,
                                   struct geist_model   **out);

/* @stability STABLE since 0.2.1
 * Load a GGUF that is already in memory — e.g. embedded in the executable, so
 * the engine *and* the model ship as a single binary. The bytes are aliased
 * read-only (zero-copy, like the file path's mmap) and are NOT freed by
 * geist_model_destroy: the caller must keep `data` valid for the model's
 * lifetime (for an `.incbin`-embedded blob that is automatic — it lives in
 * .rodata). The GGUF must carry its own tokenizer (no sibling file is searched)
 * and is text-only (no external vision/audio safetensors). See
 * `make EMBED_MODEL=...` for the build-side helper. */
enum geist_status geist_model_load_from_memory(const void           *data,
                                               size_t                size,
                                               struct geist_backend *be,
                                               struct geist_model  **out);

void        geist_model_destroy(struct geist_model *m);
const char *geist_model_errmsg(const struct geist_model *m);

/* @stability EXPERIMENTAL — name may evolve. */
const char *geist_model_arch(const struct geist_model *m);

/* ====================================================================== */
/* Session                                                                 */
/* ====================================================================== */

struct geist_session;

typedef int32_t geist_token_t;

/* @stability EXPERIMENTAL — per-session KV-cache quantization mode.
 * AUTO = env / platform default; FP32/INT8/KIVI = explicit override. */
enum geist_kv_mode {
    GEIST_KV_AUTO = 0,
    GEIST_KV_FP32 = 1,
    GEIST_KV_INT8 = 2,
    GEIST_KV_KIVI = 3,
};

struct geist_session_opts {
    /* Sequence length cap; 0 = use model default. */
    size_t max_seq_len;

    /* Sampler configuration. Applied per-session at session_create time
     * via the architecture's set_session_opts hook; not yet overridable
     * on individual decode_step calls.
     *
     *   temperature  0.0    → greedy argmax (default). >0 → softmax-sample.
     *   top_k        0 or 1 → no top-k filter (or argmax when temp=0).
     *                >1     → keep the top_k largest logits before sampling.
     *   top_p        1.0    → no nucleus filter (default).
     *                0<p<1  → smallest set whose cumulative prob exceeds p.
     *   random_seed  0      → architecture picks a default fixed seed.
     *                != 0   → use this value (deterministic across runs).
     *
     * When both top_k>1 and top_p<1 are set, top_k takes precedence. */
    float    temperature;
    float    top_p;
    int      top_k;
    uint64_t random_seed;

    /* @stability EXPERIMENTAL — AWQ (Activation-aware Weight Quantization)
     * scales file. nullptr = no AWQ. When set, the arch loads scales from
     * the given path and folds attn_norm/ffn_norm gamma at load time plus
     * applies the o_proj/down_proj input scale at runtime. Orthogonal to
     * the weight quantization format. */
    const char *awq_scales_path;

    /* @stability EXPERIMENTAL — per-session KV cache quantization mode.
     * AUTO = take the env-var / platform default (GEIST_KV_KIVI > GEIST_KV_INT8
     *        > Apple FP32 / non-Apple INT8); other values override the env.
     * Different sessions on the same model may use different modes. */
    enum geist_kv_mode kv_mode;

    /* @stability EXPERIMENTAL — verify-forward batch cap. Sizes scratch
     * buffers + KIVI residual ring. 0 = arch default (transformer = 64). */
    size_t m_max;
};

/* @stability STABLE since 0.1.0 */
enum geist_status geist_session_create(struct geist_model              *m,
                                       struct geist_backend            *be,
                                       const struct geist_session_opts *opts,
                                       struct geist_session           **out);

void        geist_session_destroy(struct geist_session *s);
const char *geist_session_errmsg(const struct geist_session *s);

/* @stability STABLE since 0.1.0
 * Reset KV / logits state for a new conversation, keep weights and the
 * session's sampler config. */
enum geist_status geist_session_reset(struct geist_session *s);

/* @stability STABLE since 0.1.0
 * Tokenize `prompt` and prefill it into the session. The generation loop
 * then calls geist_session_decode_step until it emits the model's EOS
 * token (get its id from geist_model_eos_token in <geist_util.h>). */
enum geist_status geist_session_set_prompt(struct geist_session *s, const char *prompt);

/* @stability STABLE since 0.1.0
 * Decode one token autoregressively. Returns GEIST_OK and writes the
 * token id to *out_token. EOS is signalled by token-id, not status —
 * compare against geist_model_eos_token (<geist_util.h>) to stop. */
enum geist_status geist_session_decode_step(struct geist_session *s, geist_token_t *out_token);

/* @stability STABLE since 0.1.0
 * Translate a token id back to its surface form. Returns nullptr for
 * unknown / control tokens. The pointer is stable for the session's
 * lifetime. */
const char *geist_session_token_to_str(struct geist_session *s, geist_token_t t);

#ifdef __cplusplus
} /* extern "C" */
#endif
