/*
 * test_host_buffer_unit — the host-memory geist_buffer the CPU backends
 * share (src/backends/common/host_buffer.c), driven directly.
 *
 * The CPU backends only ever reach these entry points with well-formed
 * arguments, so their refusals and both allocation failures (the handle
 * from the backend's allocator, the data from heap.h) went untested when
 * the code moved here from cpu_scalar (#62, coverage ratchet #141). Each
 * case checks the status, that *out is null on failure, and that the
 * backend's error slot names the backend — or "cpu" when it has no name.
 */
#define GEIST_INTERNAL_BACKEND_LAYER
#include "host_buffer.h"
#undef GEIST_INTERNAL_BACKEND_LAYER

#include "heap.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Allocator that counts live handles and can be told to fail. It uses
 * plain malloc, not heap.h, so heap_fail_allocations() below fails only the
 * data allocation; the handles need no more than max_align_t. */
struct counting_alloc {
    int  live;
    bool fail;
};

static void *counting_alloc_fn(void *ctx, size_t bytes, size_t alignment) {
    struct counting_alloc *c = ctx;
    if (c->fail) {
        return nullptr;
    }
    if (alignment > alignof(max_align_t)) {
        return nullptr;
    }
    void *p = malloc(bytes);
    if (p != nullptr) {
        c->live++;
    }
    return p;
}

static void counting_free_fn(void *ctx, void *ptr) {
    struct counting_alloc *c = ctx;
    c->live--;
    free(ptr);
}

static int expect_err(const struct geist_backend *be,
                      enum geist_status           code,
                      const char                 *needle,
                      const char                 *what) {
    if (be->err_code != code || strstr(be->err_msg, needle) == nullptr) {
        fprintf(stderr,
                "FAIL: %s: error slot is (%d, \"%s\"), want (%d, \"...%s...\")\n",
                what,
                (int) be->err_code,
                be->err_msg,
                (int) code,
                needle);
        return 1;
    }
    return 0;
}

static void reset_err(struct geist_backend *be) {
    be->err_code   = GEIST_OK;
    be->err_msg[0] = '\0';
}

int main(void) {
    int fails = 0;

    struct counting_alloc                 counter = {0};
    const struct geist_backend_descriptor named   = {.name = "hb_test"};
    const struct geist_backend_descriptor unnamed = {.name = nullptr};
    struct geist_backend                  be      = {
            .desc  = &named,
            .alloc = {.alloc = counting_alloc_fn, .free = counting_free_fn, .ctx = &counter},
    };
    if (pthread_mutex_init(&be.err_mu, nullptr) != 0) {
        fprintf(stderr, "ERROR: pthread_mutex_init\n");
        return GEIST_TEST_ERROR;
    }

    struct geist_buffer *buf = nullptr;

    /* ---- create: refusals ---------------------------------------------- */
    fails += geist_expect(geist_host_buffer_create(&be, 64, GEIST_BUFFER_ACTIVATION, 0, nullptr) ==
                                  GEIST_E_INVALID_ARG,
                          "create: null out is refused");

    buf = (struct geist_buffer *) &counter; /* any non-null: must be cleared */
    fails += geist_expect(geist_host_buffer_create(&be, 0, GEIST_BUFFER_ACTIVATION, 0, &buf) ==
                                  GEIST_E_INVALID_ARG,
                          "create: zero bytes is refused");
    fails += geist_expect(buf == nullptr, "create: *out is null after a refusal");
    fails += expect_err(&be, GEIST_E_INVALID_ARG, "hb_test: zero-byte", "create zero bytes");

    /* The error names "cpu" when the backend has no descriptor name, and
     * when there is no backend at all nothing is recorded (no crash). */
    reset_err(&be);
    be.desc = &unnamed;
    fails += geist_expect(geist_host_buffer_create(&be, 0, GEIST_BUFFER_ACTIVATION, 0, &buf) ==
                                  GEIST_E_INVALID_ARG,
                          "create: zero bytes is refused (unnamed backend)");
    fails += expect_err(&be, GEIST_E_INVALID_ARG, "cpu: zero-byte", "unnamed backend");
    be.desc = nullptr;
    reset_err(&be);
    fails += geist_expect(geist_host_buffer_create(&be, 0, GEIST_BUFFER_ACTIVATION, 0, &buf) ==
                                  GEIST_E_INVALID_ARG,
                          "create: zero bytes is refused (no descriptor)");
    fails += expect_err(&be, GEIST_E_INVALID_ARG, "cpu: zero-byte", "no descriptor");
    be.desc = &named;
    fails += geist_expect(geist_host_buffer_create(nullptr, 0, GEIST_BUFFER_ACTIVATION, 0, &buf) ==
                                  GEIST_E_INVALID_ARG,
                          "create: zero bytes is refused (no backend)");

    /* ---- create: allocation failures ----------------------------------- */
    reset_err(&be);
    counter.fail = true;
    fails += geist_expect(geist_host_buffer_create(&be, 64, GEIST_BUFFER_ACTIVATION, 0, &buf) ==
                                  GEIST_E_OOM,
                          "create: a failing handle allocation is OOM");
    fails += geist_expect(buf == nullptr, "create: *out is null after handle OOM");
    fails += expect_err(&be, GEIST_E_OOM, "buffer handle", "handle OOM");
    counter.fail = false;

    reset_err(&be);
    heap_fail_allocations(true);
    const enum geist_status s_data =
            geist_host_buffer_create(&be, 64, GEIST_BUFFER_ACTIVATION, 0, &buf);
    heap_fail_allocations(false);
    fails += geist_expect(s_data == GEIST_E_OOM, "create: a failing data allocation is OOM");
    fails += geist_expect(buf == nullptr, "create: *out is null after data OOM");
    fails += expect_err(&be, GEIST_E_OOM, "64-byte host buffer", "data OOM");
    fails += geist_expect(counter.live == 0, "create: data OOM releases the handle");

    /* ---- create, upload, download, map, destroy ------------------------ */
    uint8_t src[48];
    uint8_t dst[48];
    for (size_t i = 0; i < sizeof(src); i++) {
        src[i] = (uint8_t) (i * 7 + 3);
    }
    if (geist_host_buffer_create(&be, 32, GEIST_BUFFER_WEIGHT, 0, &buf) != GEIST_OK ||
        buf == nullptr) {
        fprintf(stderr, "FAIL: create 32 bytes\n");
        return GEIST_TEST_FAIL;
    }
    fails += geist_expect(buf->bytes == 32 && buf->role == GEIST_BUFFER_WEIGHT &&
                                  buf->memory_flags == 0,
                          "create: fields recorded");
    fails += geist_expect(((uintptr_t) buf->host % OPTIMAL_ALIGNMENT) == 0,
                          "create: data at OPTIMAL_ALIGNMENT");
    fails += geist_expect(geist_host_buffer_upload(buf, 33, src) == GEIST_E_INVALID_ARG,
                          "upload: past the end is refused");
    fails += geist_expect(geist_host_buffer_upload(nullptr, 8, src) == GEIST_E_INVALID_ARG,
                          "upload: null buffer is refused");
    fails += geist_expect(geist_host_buffer_upload(buf, 32, src) == GEIST_OK, "upload: 32 bytes");
    memset(dst, 0, sizeof(dst));
    fails += geist_expect(geist_host_buffer_download(33, dst, buf) == GEIST_E_INVALID_ARG,
                          "download: past the end is refused");
    fails += geist_expect(geist_host_buffer_download(8, dst, nullptr) == GEIST_E_INVALID_ARG,
                          "download: null buffer is refused");
    fails += geist_expect(geist_host_buffer_download(32, dst, buf) == GEIST_OK,
                          "download: 32 bytes");
    fails += geist_expect(memcmp(src, dst, 32) == 0, "download returns what upload wrote");
    fails += geist_expect(geist_host_buffer_map(buf) == buf->host, "map: the host pointer");
    fails += geist_expect(geist_host_buffer_map(nullptr) == nullptr, "map: null buffer");
    geist_host_buffer_unmap(buf);
    geist_host_buffer_unmap(nullptr);
    geist_host_buffer_destroy(&be, buf);
    geist_host_buffer_destroy(&be, nullptr);
    fails += geist_expect(counter.live == 0, "destroy: handle released");

    /* ---- aliased ------------------------------------------------------- */
    fails += geist_expect(
            geist_host_buffer_create_aliased(&be, src, sizeof(src), GEIST_BUFFER_WEIGHT, nullptr) ==
                    GEIST_E_INVALID_ARG,
            "aliased: null out is refused");
    reset_err(&be);
    buf = (struct geist_buffer *) &counter;
    fails += geist_expect(geist_host_buffer_create_aliased(
                                  &be, nullptr, sizeof(src), GEIST_BUFFER_WEIGHT, &buf) ==
                                  GEIST_E_INVALID_ARG,
                          "aliased: null host pointer is refused");
    fails += geist_expect(buf == nullptr, "aliased: *out is null after a refusal");
    fails += expect_err(&be, GEIST_E_INVALID_ARG, "needs host_ptr", "aliased null host");
    reset_err(&be);
    fails += geist_expect(geist_host_buffer_create_aliased(
                                  &be, src, 0, GEIST_BUFFER_WEIGHT, &buf) == GEIST_E_INVALID_ARG,
                          "aliased: zero bytes is refused");
    fails += expect_err(&be, GEIST_E_INVALID_ARG, "needs host_ptr", "aliased zero bytes");

    reset_err(&be);
    counter.fail = true;
    fails += geist_expect(geist_host_buffer_create_aliased(
                                  &be, src, sizeof(src), GEIST_BUFFER_WEIGHT, &buf) == GEIST_E_OOM,
                          "aliased: a failing handle allocation is OOM");
    fails += geist_expect(buf == nullptr, "aliased: *out is null after handle OOM");
    fails += expect_err(&be, GEIST_E_OOM, "hb_test: buffer handle", "aliased handle OOM");
    counter.fail = false;

    if (geist_host_buffer_create_aliased(&be, src, sizeof(src), GEIST_BUFFER_WEIGHT, &buf) !=
                GEIST_OK ||
        buf == nullptr) {
        fprintf(stderr, "FAIL: aliased create\n");
        return GEIST_TEST_FAIL;
    }
    fails += geist_expect(buf->host == src && buf->bytes == sizeof(src) &&
                                  (buf->memory_flags & GEIST_MEMORY_ALIASED) != 0,
                          "aliased: wraps the caller's memory");
    fails += geist_expect(geist_host_buffer_download(sizeof(dst), dst, buf) == GEIST_OK &&
                                  memcmp(src, dst, sizeof(dst)) == 0,
                          "aliased: download reads the caller's memory");
    /* Destroy frees only the handle; src is a stack array, so freeing it
     * would be an ASan error here. */
    geist_host_buffer_destroy(&be, buf);
    fails += geist_expect(counter.live == 0, "aliased destroy: handle released");

    /* A buffer created with the ALIASED flag through the owning entry
     * point keeps its data on destroy too; free it here instead. */
    if (geist_host_buffer_create(&be, 16, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_ALIASED, &buf) !=
                GEIST_OK ||
        buf == nullptr) {
        fprintf(stderr, "FAIL: create with GEIST_MEMORY_ALIASED\n");
        return GEIST_TEST_FAIL;
    }
    void *data = buf->host;
    geist_host_buffer_destroy(&be, buf);
    safe_free(&data);
    fails += geist_expect(counter.live == 0, "flagged destroy: handle released");

    pthread_mutex_destroy(&be.err_mu);
    if (fails != 0) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: host buffers refuse bad arguments, survive allocation failures, round-trip "
           "data\n");
    return GEIST_TEST_PASS;
}
