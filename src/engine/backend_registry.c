/*
 * src/engine/backend_registry.c — compiled-in backend list.
 *
 * Layer: ENGINE.
 *
 * Per Q28: each compiled-in backend appears here, gated by GEIST_BACKEND_*
 * defines that the Makefile sets based on the BACKENDS=... variable.
 * The registry is NULL-terminated and ordered by preference (auto-pick
 * takes the first entry).
 *
 * Adding a new backend:
 *   1. Implement src/backends/<name>/backend.c exporting
 *      `extern const struct geist_backend_descriptor geist_backend_<name>`.
 *   2. Add `mk/backend-<name>.mk` setting BACKEND_SOURCES += ...
 *   3. Add `#if GEIST_BACKEND_<NAME>` block below.
 *   4. Build with `make BACKENDS="cpu_neon <name>"`.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include <geist_backend.h>

#if defined(GEIST_BACKEND_CPU_NEON) && GEIST_BACKEND_CPU_NEON
extern const struct geist_backend_descriptor geist_backend_cpu_neon;
#endif

#if defined(GEIST_BACKEND_CPU_X86) && GEIST_BACKEND_CPU_X86
extern const struct geist_backend_descriptor geist_backend_cpu_x86;
#endif

#if defined(GEIST_BACKEND_CPU_SCALAR) && GEIST_BACKEND_CPU_SCALAR
extern const struct geist_backend_descriptor geist_backend_cpu_scalar;
#endif

const struct geist_backend_descriptor *const geist_backend_registry[] = {
#if defined(GEIST_BACKEND_CPU_NEON) && GEIST_BACKEND_CPU_NEON
        &geist_backend_cpu_neon,
#endif
#if defined(GEIST_BACKEND_CPU_X86) && GEIST_BACKEND_CPU_X86
        &geist_backend_cpu_x86,
#endif
#if defined(GEIST_BACKEND_CPU_SCALAR) && GEIST_BACKEND_CPU_SCALAR
        &geist_backend_cpu_scalar,
#endif
        nullptr,
};
