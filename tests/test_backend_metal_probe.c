/*
 * test_backend_metal_probe — does this host expose a usable Metal device?
 *
 * A spike, not a coverage test. The Metal backend has 52 compute kernels,
 * 8 of them hand-written simdgroup GEMMs, and no numerical parity gate at
 * all (vulkan has three). Before writing that gate it is worth knowing
 * whether CI can ever run it: GitHub's macOS runners are virtualized, and
 * a parity test that SKIPs forever is not a gate.
 *
 * So this asks the narrow question and prints the answer: create the
 * backend, then round-trip a buffer through it. Backend create alone is
 * not enough — it proves a device handle, not that the device works.
 *
 * SKIPs (exit 0) when metal is not compiled in or no device is present,
 * which is every default build. Run it with:
 *
 *   make test-unit TARGET=mac-omp BACKENDS="metal cpu_neon cpu_scalar"
 *
 * If this reports PASS on a CI runner, the parity gate is worth building
 * and wiring into a job. If it reports SKIP there, the gate would only
 * ever run on a developer's Mac and should be documented as such.
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
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("metal", nullptr, nullptr, &be);
    if (s == GEIST_E_UNSUPPORTED || s == GEIST_E_NOT_FOUND) {
        /* NOT_FOUND: not compiled in (BACKENDS without "metal").
         * UNSUPPORTED: no Metal device at runtime. Both are environment
         * facts — and telling them apart is the point of the spike. */
        fprintf(stderr,
                "SKIP: metal backend unavailable (%s): %s\n",
                s == GEIST_E_NOT_FOUND ? "not built into this binary" : "no device at runtime",
                geist_last_create_error());
        return GEIST_TEST_SKIP;
    }
    if (check(s == GEIST_OK && be != nullptr, "metal backend create") != 0) {
        return GEIST_TEST_FAIL;
    }

    int                              fails = 0;
    const struct geist_backend_vtbl *v     = be->desc->vtbl;

    /* Round-trip a buffer: proves the device actually executes, not just
     * that a handle was produced. */
    enum { N = 4096 };
    uint8_t pattern[N], readback[N];
    for (size_t i = 0; i < N; i++) {
        pattern[i] = (uint8_t) (i * 131u + 7u);
    }
    memset(readback, 0, sizeof readback);

    struct geist_buffer *buf = nullptr;
    s = v->buffer_create(be, N, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &buf);
    fails += check(s == GEIST_OK && buf != nullptr, "buffer_create");
    if (s == GEIST_OK && buf != nullptr) {
        fails += check(v->buffer_upload(buf, N, pattern) == GEIST_OK, "buffer_upload");
        fails += check(v->buffer_download(N, readback, buf) == GEIST_OK, "buffer_download");
        fails += check(memcmp(pattern, readback, N) == 0, "round-trip is bit-exact");
        v->buffer_destroy(be, buf);
    }

    geist_backend_destroy(be);

    if (fails > 0) {
        fprintf(stderr, "FAIL: %d metal probe assertion(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: metal device present and a buffer round-trips — a parity "
           "gate can run in this environment.\n");
    return GEIST_TEST_PASS;
}
