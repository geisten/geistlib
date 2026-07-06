/*
 * src/engine/backend.c — geist_backend_create dispatch + lifecycle.
 *
 * Layer: ENGINE.
 *
 * Looks up the requested backend in the compiled-in registry, allocates
 * a struct geist_backend via the user-provided allocator, hooks the
 * descriptor + vtable in, and calls the backend's optional create() hook.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "model.h" /* for geist_arch_registry forward decl pattern */

#include <geist.h>
#include <geist_backend.h>

#include "error.h"

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Defined in src/engine/backend_registry.c. NULL-terminated. */
extern const struct geist_backend_descriptor *const geist_backend_registry[];

/* Auto-pick: first entry in the registry. Backends list themselves in the
 * registry in preference order (cpu_neon before cpu_scalar etc.). */
static const struct geist_backend_descriptor *resolve_descriptor(const char *name) {
    if (name == nullptr || name[0] == '\0' || strcmp(name, "auto") == 0) {
        /* GEIST_BACKEND overrides auto (same convention as the bench
         * harness's GEIST_BENCH_BACKEND) — without it every "auto"
         * caller silently gets registry[0] and GPU builds cannot be
         * exercised end-to-end from the CLI or eval tools. */
        const char *env = getenv("GEIST_BACKEND");
        if (env == nullptr || env[0] == '\0') {
            return geist_backend_registry[0]; /* may be nullptr if no backends linked */
        }
        name = env;
    }
    for (size_t i = 0; geist_backend_registry[i] != nullptr; i++) {
        if (strcmp(geist_backend_registry[i]->name, name) == 0) {
            return geist_backend_registry[i];
        }
    }
    return nullptr;
}

enum geist_status geist_backend_create(const char                      *name,
                                       const struct geist_backend_opts *opts,
                                       const struct geist_allocator    *alloc,
                                       struct geist_backend           **out) {
    if (out == nullptr) {
        geist_error_set_create_time(
                GEIST_E_INVALID_ARG, "geist_backend_create", "out parameter is null");
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;

    const struct geist_backend_descriptor *desc = resolve_descriptor(name);
    if (desc == nullptr) {
        geist_error_set_create_time(GEIST_E_NOT_FOUND,
                                    "geist_backend_create",
                                    "no backend named '%s' is compiled in",
                                    name == nullptr ? "(null)" : name);
        return GEIST_E_NOT_FOUND;
    }
    if (desc->vtbl == nullptr || desc->vtbl->destroy == nullptr) {
        geist_error_set_create_time(GEIST_E_INTERNAL,
                                    "geist_backend_create",
                                    "backend '%s' has incomplete vtable",
                                    desc->name);
        return GEIST_E_INTERNAL;
    }

    const struct geist_allocator *a = alloc != nullptr ? alloc : &geist_libc_allocator;

    struct geist_backend *be = a->alloc(a->ctx, sizeof(*be), alignof(struct geist_backend));
    if (be == nullptr) {
        geist_error_set_create_time(GEIST_E_OOM,
                                    "geist_backend_create",
                                    "failed to allocate %zu bytes for backend handle",
                                    sizeof(*be));
        return GEIST_E_OOM;
    }

    *be = (struct geist_backend) {
            .desc     = desc,
            .alloc    = *a,
            .state    = nullptr,
            .err_code = GEIST_OK,
    };
    be->err_msg[0] = '\0';

    if (desc->vtbl->create != nullptr) {
        enum geist_status s = desc->vtbl->create(be, opts);
        if (s != GEIST_OK) {
            /* Backend failed during init; copy its error to the create-time slot
             * so user can retrieve via geist_last_create_error(). */
            geist_error_set_create_time(s,
                                        "geist_backend_create",
                                        "backend '%s' init failed: %s",
                                        desc->name,
                                        be->err_msg[0] != '\0' ? be->err_msg : "(no detail)");
            a->free(a->ctx, be);
            return s;
        }
    }

    *out = be;
    return GEIST_OK;
}

void geist_backend_destroy(struct geist_backend *be) {
    if (be == nullptr) {
        return;
    }
    if (be->desc != nullptr && be->desc->vtbl != nullptr && be->desc->vtbl->destroy != nullptr) {
        be->desc->vtbl->destroy(be);
    }
    /* Stash allocator before freeing the handle so the free uses correct ctx. */
    struct geist_allocator a = be->alloc;
    a.free(a.ctx, be);
}

const char *geist_backend_name(const struct geist_backend *be) {
    return be != nullptr && be->desc != nullptr ? be->desc->name : nullptr;
}

const char *geist_backend_errmsg(const struct geist_backend *be) {
    if (be == nullptr) {
        return nullptr;
    }
    return be->err_msg[0] != '\0' ? be->err_msg : "(no error)";
}

enum geist_status geist_backend_errcode(const struct geist_backend *be) {
    return be != nullptr ? be->err_code : GEIST_E_INVALID_ARG;
}

enum geist_support geist_backend_supports_op(struct geist_backend                *be,
                                             const struct geist_op_support_query *query) {
    if (be == nullptr || query == nullptr || be->desc == nullptr || be->desc->vtbl == nullptr ||
        be->desc->vtbl->supports_op == nullptr) {
        return GEIST_SUPPORT_NONE;
    }
    return be->desc->vtbl->supports_op(be, query);
}

/* ---------- Backend-side helpers (declared in geist_backend.h) ---------- */

void geist_backend_set_error(struct geist_backend *be,
                             enum geist_status     code,
                             const char           *fmt,
                             ...) {
    if (be == nullptr) {
        return;
    }
    be->err_code = code;

    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(be->err_msg, sizeof(be->err_msg), fmt, ap);
    va_end(ap);

    if (n < 0) {
        be->err_msg[0] = '\0';
    }
}

void *geist_backend_alloc(struct geist_backend *be, size_t bytes, size_t alignment) {
    if (be == nullptr || bytes == 0) {
        return nullptr;
    }
    return be->alloc.alloc(be->alloc.ctx, bytes, alignment);
}

void geist_backend_free(struct geist_backend *be, void *ptr) {
    if (be == nullptr || ptr == nullptr) {
        return;
    }
    be->alloc.free(be->alloc.ctx, ptr);
}
