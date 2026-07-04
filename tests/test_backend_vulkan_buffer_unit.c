/*
 * test_backend_vulkan_buffer_unit — buffer-op contract tests for the vulkan
 * backend on a discrete GPU (Phase 1).
 *
 * Covers the three memory paths the backend distinguishes:
 *   1. host-visible (SCRATCH): map is non-null, upload/download round-trip
 *   2. device-local (WEIGHT): map is nullptr, staged upload/download
 *      round-trips bit-exactly through VRAM
 *   3. aliased: map returns the exact host pointer; upload/download copy
 *      through the aliased region
 *
 * SKIPs (exit 0) when no Vulkan runtime/device is present.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
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
    if (s == GEIST_E_UNSUPPORTED) {
        fprintf(stderr, "SKIP: no Vulkan runtime/device on this machine\n");
        return 0;
    }
    if (check(s == GEIST_OK && be != nullptr, "backend create") != 0) {
        return 1;
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;

    enum { N = 4096 };
    uint8_t pattern[N], readback[N];
    for (size_t i = 0; i < N; ++i) {
        pattern[i] = (uint8_t) (i * 131u + 7u);
    }

    /* 1. host-visible scratch */
    struct geist_buffer *scratch = nullptr;
    s = v->buffer_create(be, N, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &scratch);
    fails += check(s == GEIST_OK, "scratch buffer_create");
    if (s == GEIST_OK) {
        void *p = v->buffer_map(scratch);
        fails += check(p != nullptr, "scratch buffer_map is non-null");
        fails += check(v->buffer_upload(scratch, N, pattern) == GEIST_OK, "scratch upload");
        memset(readback, 0, N);
        fails += check(v->buffer_download(N, readback, scratch) == GEIST_OK,
                       "scratch download");
        fails += check(memcmp(pattern, readback, N) == 0, "scratch round-trip bit-exact");
        if (p != nullptr) {
            fails += check(memcmp(p, pattern, N) == 0, "scratch map sees uploaded bytes");
        }
        v->buffer_unmap(scratch);
        v->buffer_destroy(be, scratch);
    }

    /* 2. device-local weight (staged copies through VRAM) — DEVICE_LOCAL
     * needs the explicit flag; role alone stays host-visible because the
     * arch layer maps WEIGHT-role buffers (cos/sin tables). */
    struct geist_buffer *weight = nullptr;
    s = v->buffer_create(be, N, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &weight);
    fails += check(s == GEIST_OK, "weight buffer_create");
    if (s == GEIST_OK) {
        fails += check(v->buffer_map(weight) == nullptr,
                       "weight (device-local) buffer_map is nullptr");
        fails += check(v->buffer_upload(weight, N, pattern) == GEIST_OK,
                       "weight staged upload");
        memset(readback, 0, N);
        fails += check(v->buffer_download(N, readback, weight) == GEIST_OK,
                       "weight staged download");
        fails += check(memcmp(pattern, readback, N) == 0, "weight round-trip bit-exact");
        v->buffer_destroy(be, weight);
    }

    /* 3. aliased host region */
    uint8_t *host = malloc(N);
    memset(host, 0xAB, N);
    struct geist_buffer *aliased = nullptr;
    s = v->buffer_create_aliased(be, host, N, GEIST_BUFFER_SCRATCH, &aliased);
    fails += check(s == GEIST_OK, "aliased buffer create");
    if (s == GEIST_OK) {
        fails += check(v->buffer_map(aliased) == host, "aliased map returns host pointer");
        fails += check(v->buffer_upload(aliased, N, pattern) == GEIST_OK, "aliased upload");
        fails += check(memcmp(host, pattern, N) == 0, "aliased upload lands in host region");
        memset(readback, 0, N);
        fails += check(v->buffer_download(N, readback, aliased) == GEIST_OK,
                       "aliased download");
        fails += check(memcmp(pattern, readback, N) == 0, "aliased round-trip bit-exact");
        v->buffer_destroy(be, aliased);
        /* destroy must not free the aliased bytes — touch them afterwards */
        fails += check(host[0] == pattern[0], "aliased bytes survive buffer_destroy");
    }
    free(host);

    geist_backend_destroy(be);
    if (fails == 0) {
        printf("test_backend_vulkan_buffer_unit: all checks passed\n");
    }
    return fails == 0 ? 0 : 1;
}
