/*
 * agent_contract_smoke.c — the agent-runtime API contract, enforced by the
 * compiler and the linker (docs/API_CONTRACT.md).
 *
 * An out-of-tree agent runtime implements the tool-use
 * loop — constrained decoding, KV-prefix pinning, chat templating — against
 * libgeist. Those symbols are contractual, so a signature change must be a
 * deliberate major bump, never a surprise at the consumer's build.
 *
 * Each entry below binds a contract symbol to an explicitly typed function
 * pointer:
 *   - a CHANGED SIGNATURE fails to compile here (incompatible pointer types),
 *   - a REMOVED SYMBOL fails to link here.
 * Nothing is called, so no model and no weights are needed at runtime. Unlike
 * embed_smoke.c it does pull the backend/model/session objects (taking a
 * function's address forces the linker to resolve it), so it links with the
 * library's own flags:
 *
 *   make agent-contract-smoke
 */
#include <geist.h>
#include <geist_util.h>

#include <stdio.h>

/* The contract exists from this release on; building against an older SDK is
 * a hard error rather than a confusing link failure. */
#if (GEIST_VERSION_MAJOR * 10000 + GEIST_VERSION_MINOR * 100) < 600
#  error "the agent-runtime contract requires geistlib >= 0.6.0"
#endif

/* ---- include/geist.h ---------------------------------------------------- */

static enum geist_status (*const c_backend_create)(const char *,
                                                   const struct geist_backend_opts *,
                                                   const struct geist_allocator *,
                                                   struct geist_backend **) = geist_backend_create;
static void (*const c_backend_destroy)(struct geist_backend *) = geist_backend_destroy;

static enum geist_status (*const c_model_load)(const char *, struct geist_backend *,
                                               struct geist_model **) = geist_model_load;
static void (*const c_model_destroy)(struct geist_model *) = geist_model_destroy;
/* The chat template lives out of tree; the family it is chosen by does not. */
static const char *(*const c_model_arch)(const struct geist_model *) = geist_model_arch;

static enum geist_status (*const c_session_create)(struct geist_model *,
                                                   struct geist_backend *,
                                                   const struct geist_session_opts *,
                                                   struct geist_session **) = geist_session_create;
static void (*const c_session_destroy)(struct geist_session *) = geist_session_destroy;
static enum geist_status (*const c_set_prompt)(struct geist_session *,
                                               const char *) = geist_session_set_prompt;
static enum geist_status (*const c_decode_step)(struct geist_session *,
                                                geist_token_t *) = geist_session_decode_step;
static const char *(*const c_token_to_str)(struct geist_session *,
                                           geist_token_t) = geist_session_token_to_str;
static enum geist_status (*const c_reset)(struct geist_session *) = geist_session_reset;

/* ---- include/geist_util.h ----------------------------------------------- */

static geist_token_t (*const c_eos)(const struct geist_model *) = geist_model_eos_token;
static geist_token_t (*const c_bos)(const struct geist_model *) = geist_model_bos_token;
static geist_token_t (*const c_token_by_text)(const struct geist_model *,
                                              const char *) = geist_model_token_by_text;

static enum geist_status (*const c_prefill_tokens)(struct geist_session *, size_t,
                                                   const geist_token_t *) = geist_session_prefill_tokens;
static enum geist_status (*const c_tokenize)(struct geist_session *, const char *, size_t,
                                             geist_token_t *, size_t *) = geist_session_tokenize;
static enum geist_status (*const c_pin_prefix)(struct geist_session *, size_t,
                                               const geist_token_t *) = geist_session_pin_prefix;
/* The grammar mask: without logit access there is no constrained decoding, and
 * a model that was never tool-trained cannot be made to emit a valid call. */
static const float *(*const c_peek_logits)(size_t *,
                                           struct geist_session *) = geist_session_peek_logits;

static const struct {
    const char *name;
    const void *fn;
} contract[] = {
        {"geist_backend_create", (const void *) &c_backend_create},
        {"geist_backend_destroy", (const void *) &c_backend_destroy},
        {"geist_model_load", (const void *) &c_model_load},
        {"geist_model_destroy", (const void *) &c_model_destroy},
        {"geist_model_arch", (const void *) &c_model_arch},
        {"geist_session_create", (const void *) &c_session_create},
        {"geist_session_destroy", (const void *) &c_session_destroy},
        {"geist_session_set_prompt", (const void *) &c_set_prompt},
        {"geist_session_decode_step", (const void *) &c_decode_step},
        {"geist_session_token_to_str", (const void *) &c_token_to_str},
        {"geist_session_reset", (const void *) &c_reset},
        {"geist_model_eos_token", (const void *) &c_eos},
        {"geist_model_bos_token", (const void *) &c_bos},
        {"geist_model_token_by_text", (const void *) &c_token_by_text},
        {"geist_session_prefill_tokens", (const void *) &c_prefill_tokens},
        {"geist_session_tokenize", (const void *) &c_tokenize},
        {"geist_session_pin_prefix", (const void *) &c_pin_prefix},
        {"geist_session_peek_logits", (const void *) &c_peek_logits},
};

int main(void) {
    const size_t n = sizeof contract / sizeof contract[0];
    for (size_t i = 0; i < n; i++) {
        if (contract[i].fn == nullptr) { /* unreachable: addresses of file-scope objects */
            fprintf(stderr, "agent_contract_smoke: %s is missing\n", contract[i].name);
            return 1;
        }
    }
    printf("agent_contract_smoke: %zu contract symbols present (geist %s)\n", n,
           GEIST_VERSION_STRING);
    return 0;
}
