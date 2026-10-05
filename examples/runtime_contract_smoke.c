/*
 * runtime_contract_smoke.c — the API geist-runtime builds on, enforced by the
 * compiler and the linker (#622, docs/API_CONTRACT.md).
 *
 * geist-runtime (github.com/geisten/geist-runtime) is the chat layer over
 * libgeist: templates, streaming text, a conversation it can rewind, a
 * context window chosen to fit into memory. Every symbol it calls is bound
 * here to an explicitly typed function pointer, as in agent_contract_smoke.c:
 *   - a CHANGED SIGNATURE fails to compile here,
 *   - a REMOVED SYMBOL fails to link here.
 * Nothing is called, so no model is needed.
 *
 * The STABLE part is a promise. The EXPERIMENTAL part (#622's new calls) may
 * still change, but not unnoticed: a change fails this gate, and the runtime
 * follows in the same release.
 *
 *   make runtime-contract-smoke
 */
#include <geist.h>
#include <geist_util.h>

#include <stdio.h>

/* The EXPERIMENTAL calls land in the release after 0.11.0 (the version is
 * bumped at release, docs/RELEASING.md); an older SDK fails to link. */
#if (GEIST_VERSION_MAJOR * 10000 + GEIST_VERSION_MINOR * 100) < 1100
#error "the geist-runtime contract requires geistlib > 0.11.0"
#endif

/* ---- STABLE ------------------------------------------------------------- */

static enum geist_status (*const c_backend_create)(const char *,
                                                   const struct geist_backend_opts *,
                                                   const struct geist_allocator *,
                                                   struct geist_backend **) = geist_backend_create;
static void (*const c_backend_destroy)(struct geist_backend *)              = geist_backend_destroy;
static const char *(*const c_backend_name)(const struct geist_backend *)    = geist_backend_name;
static const char *(*const c_backend_errmsg)(const struct geist_backend *)  = geist_backend_errmsg;

static enum geist_status (*const c_load_with_opts)(const char *,
                                                   struct geist_backend *,
                                                   const struct geist_session_opts *,
                                                   struct geist_model **) =
        geist_model_load_with_opts;
static enum geist_status (*const c_load_from_memory)(const void *,
                                                     size_t,
                                                     struct geist_backend *,
                                                     struct geist_model **) =
        geist_model_load_from_memory;
static enum geist_status (*const c_load_from_memory_with_opts)(const void *,
                                                               size_t,
                                                               struct geist_backend *,
                                                               const struct geist_session_opts *,
                                                               struct geist_model **) =
        geist_model_load_from_memory_with_opts;
static void (*const c_model_destroy)(struct geist_model *)             = geist_model_destroy;
static const char *(*const c_model_errmsg)(const struct geist_model *) = geist_model_errmsg;
static const char *(*const c_model_arch)(const struct geist_model *)   = geist_model_arch;
static geist_token_t (*const c_bos)(const struct geist_model *)        = geist_model_bos_token;
static geist_token_t (*const c_eos)(const struct geist_model *)        = geist_model_eos_token;
static bool (*const c_add_bos)(const struct geist_model *)             = geist_model_add_bos;
static geist_token_t (*const c_token_by_text)(const struct geist_model *,
                                              const char *)            = geist_model_token_by_text;

static enum geist_status (*const c_session_create)(struct geist_model *,
                                                   struct geist_backend *,
                                                   const struct geist_session_opts *,
                                                   struct geist_session **) = geist_session_create;
static void (*const c_session_destroy)(struct geist_session *)              = geist_session_destroy;
static const char *(*const c_session_errmsg)(const struct geist_session *)  = geist_session_errmsg;
static enum geist_status (*const c_reset)(struct geist_session *)           = geist_session_reset;
static enum geist_status (*const c_tokenize)(struct geist_session *,
                                             const char *,
                                             size_t,
                                             geist_token_t *,
                                             size_t *) = geist_session_tokenize;
static enum geist_status (*const c_prefill_tokens)(
        struct geist_session *, size_t, const geist_token_t *)   = geist_session_prefill_tokens;
static enum geist_status (*const c_decode_step)(struct geist_session *,
                                                geist_token_t *) = geist_session_decode_step;
static const char *(*const c_token_to_str)(struct geist_session *,
                                           geist_token_t)        = geist_session_token_to_str;
static enum geist_status (*const c_pin_prefix)(struct geist_session *,
                                               size_t,
                                               const geist_token_t *) = geist_session_pin_prefix;

/* ---- EXPERIMENTAL (#622) -------------------------------------------------- */

static const char *(*const c_metadata_str)(const struct geist_model *,
                                           const char *,
                                           size_t *)                  = geist_model_metadata_str;
static size_t (*const c_context_length)(const struct geist_model *)   = geist_model_context_length;
static size_t (*const c_session_length)(const struct geist_session *) = geist_session_length;
static enum geist_status (*const c_truncate)(struct geist_session *,
                                             size_t)                  = geist_session_truncate;
static enum geist_status (*const c_kv_bytes)(const struct geist_session *,
                                             size_t *) = geist_session_kv_bytes_per_token;

static enum geist_status (*const c_plan)(const char *,
                                         struct geist_backend *,
                                         const struct geist_session_opts *,
                                         struct geist_model_plan *) = geist_model_plan;
static enum geist_status (*const c_plan_from_memory)(const void *,
                                                     size_t,
                                                     struct geist_backend *,
                                                     const struct geist_session_opts *,
                                                     struct geist_model_plan *) =
        geist_model_plan_from_memory;

static const struct {
    const char *name;
    const void *fn;
} contract[] = {
        {"geist_backend_create", (const void *) &c_backend_create},
        {"geist_backend_destroy", (const void *) &c_backend_destroy},
        {"geist_backend_name", (const void *) &c_backend_name},
        {"geist_backend_errmsg", (const void *) &c_backend_errmsg},
        {"geist_model_load_with_opts", (const void *) &c_load_with_opts},
        {"geist_model_load_from_memory", (const void *) &c_load_from_memory},
        {"geist_model_load_from_memory_with_opts", (const void *) &c_load_from_memory_with_opts},
        {"geist_model_destroy", (const void *) &c_model_destroy},
        {"geist_model_errmsg", (const void *) &c_model_errmsg},
        {"geist_model_arch", (const void *) &c_model_arch},
        {"geist_model_bos_token", (const void *) &c_bos},
        {"geist_model_eos_token", (const void *) &c_eos},
        {"geist_model_add_bos", (const void *) &c_add_bos},
        {"geist_model_token_by_text", (const void *) &c_token_by_text},
        {"geist_session_create", (const void *) &c_session_create},
        {"geist_session_destroy", (const void *) &c_session_destroy},
        {"geist_session_errmsg", (const void *) &c_session_errmsg},
        {"geist_session_reset", (const void *) &c_reset},
        {"geist_session_tokenize", (const void *) &c_tokenize},
        {"geist_session_prefill_tokens", (const void *) &c_prefill_tokens},
        {"geist_session_decode_step", (const void *) &c_decode_step},
        {"geist_session_token_to_str", (const void *) &c_token_to_str},
        {"geist_session_pin_prefix", (const void *) &c_pin_prefix},
        {"geist_model_metadata_str", (const void *) &c_metadata_str},
        {"geist_model_context_length", (const void *) &c_context_length},
        {"geist_session_length", (const void *) &c_session_length},
        {"geist_session_truncate", (const void *) &c_truncate},
        {"geist_session_kv_bytes_per_token", (const void *) &c_kv_bytes},
        {"geist_model_plan", (const void *) &c_plan},
        {"geist_model_plan_from_memory", (const void *) &c_plan_from_memory},
};

int main(void) {
    const size_t n = sizeof contract / sizeof contract[0];
    for (size_t i = 0; i < n; i++) {
        if (contract[i].fn == nullptr) { /* unreachable: addresses of file-scope objects */
            fprintf(stderr, "runtime_contract_smoke: %s is missing\n", contract[i].name);
            return 1;
        }
    }
    printf("runtime_contract_smoke: %zu contract symbols present (geist %s)\n",
           n,
           GEIST_VERSION_STRING);
    return 0;
}
