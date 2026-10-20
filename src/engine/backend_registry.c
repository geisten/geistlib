/*
 * src/engine/backend_registry.c — compiled-in backend list.
 *
 * Layer: ENGINE.
 *
 * Each compiled-in backend appears here, gated by GEIST_BACKEND_*
 * defines that the Makefile sets based on the BACKENDS=... variable.
 * The registry is NULL-terminated and ordered by preference: "auto" takes
 * geist_backend_auto_first when it is set and comes up, else the first
 * entry.
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
#if defined(GEIST_BACKEND_METAL) && GEIST_BACKEND_METAL
extern const struct geist_backend_descriptor geist_backend_metal;
#endif
#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
extern const struct geist_backend_descriptor geist_backend_vulkan;
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
#if defined(GEIST_BACKEND_METAL) && GEIST_BACKEND_METAL
        &geist_backend_metal,
#endif
#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
        &geist_backend_vulkan,
#endif
        nullptr,
};

/* The backend "auto" tries before the registry order: Metal on Apple
 * silicon, which beats cpu_neon on every model measured from 360M to 27B
 * parameters (prefill 5-14x, decode 1.9-2.5x on an M1 Max, PR #120). A
 * model load it refuses retries on the first registry entry (model.c).
 * nullptr elsewhere: Vulkan stays opt-in. */
#if defined(GEIST_BACKEND_METAL) && GEIST_BACKEND_METAL && defined(__APPLE__) && \
        defined(__aarch64__)
const struct geist_backend_descriptor *const geist_backend_auto_first = &geist_backend_metal;
#else
const struct geist_backend_descriptor *const geist_backend_auto_first = nullptr;
#endif
