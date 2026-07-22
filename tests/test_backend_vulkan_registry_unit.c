/*
 * test_backend_vulkan_registry_unit — walking-skeleton smoke test for the
 * vulkan backend (Phase 1 scaffold).
 *
 * Verifies:
 *   - geist_backend_create("vulkan") succeeds when a Vulkan ICD is present
 *   - name reports "vulkan"
 *   - create/destroy cycles twice without leaking device state
 *
 * On machines without a Vulkan loader/device the test SKIPs (exit 0) so CI
 * without a GPU stays green — mirrors the graceful-fail contract of
 * vk_create.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <string.h>

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

int main(void) {
    int fails = 0;

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("vulkan", nullptr, nullptr, &be);
    if (s == GEIST_E_UNSUPPORTED || s == GEIST_E_NOT_FOUND) {
        /* UNSUPPORTED: no Vulkan loader/device at runtime. NOT_FOUND: the
         * backend is not compiled in (BACKENDS without "vulkan" — every
         * default CI build). Both are environment facts, not failures. */
        fprintf(stderr, "SKIP: vulkan backend unavailable (not built or no device)\n");
        return GEIST_TEST_SKIP;
    }
    fails += check(s == GEIST_OK, "geist_backend_create(\"vulkan\") returns OK");
    fails += check(be != nullptr, "backend handle is non-null");
    if (be == nullptr) {
        return 1;
    }
    fails += check(strcmp(geist_backend_name(be), "vulkan") == 0, "name is \"vulkan\"");
    geist_backend_destroy(be);

    /* Second cycle — catches leaked instance/device state. */
    be = nullptr;
    s  = geist_backend_create("vulkan", nullptr, nullptr, &be);
    fails += check(s == GEIST_OK && be != nullptr, "second create/destroy cycle works");
    if (be != nullptr) {
        geist_backend_destroy(be);
    }

    if (fails == 0) {
        printf("test_backend_vulkan_registry_unit: all checks passed\n");
    }
    return fails == 0 ? 0 : 1;
}
