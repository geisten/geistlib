/*
 * tests/wiring/backend_fixture.h - slots for the backend wiring cases (#345).
 *
 * Each tests/wiring/backend_*.c defines backend tables the way a backend
 * under src/backends/ does, with the cpu_scalar functions as slots.
 * scripts/check_wiring.py compiles it with -DWIRING_OK (must compile) and
 * without (must fail with the message its `// expect:` line names).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "src/backends/common/backend_tables.h"
#include "src/backends/cpu_scalar/internal.h"

[[maybe_unused]] static void fixture_destroy(struct geist_backend *be) {
    (void) be;
}

#define FIXTURE_VTBL_SLOTS                                                               \
    fixture_destroy, geist_host_buffer_create, geist_host_buffer_destroy,               \
            geist_host_buffer_create_aliased, geist_host_buffer_upload,                 \
            geist_host_buffer_download, geist_host_buffer_map, geist_host_buffer_unmap

#define FIXTURE_PRIM_SLOTS                                                               \
    cpu_scalar_rmsnorm, cpu_scalar_add, cpu_scalar_mul, cpu_scalar_gelu_tanh,           \
            cpu_scalar_silu, cpu_scalar_rope_apply, cpu_scalar_attention

/* One more expansion, so the slot lists above split into arguments. */
#define FIXTURE_VTBL(...)  GEIST_BACKEND_VTBL(__VA_ARGS__)
#define FIXTURE_PRIMS(...) GEIST_BACKEND_PRIMS(__VA_ARGS__)
