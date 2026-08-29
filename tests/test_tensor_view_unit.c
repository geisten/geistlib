/*
 * test_tensor_view_unit — malformed tensor views must be refused, on
 * every CPU backend (issue #330).
 *
 * The elementwise ops used to unpack a view by folding shape[0..ndim)
 * with an unchecked multiply and adding `offset` to the host pointer.
 * ndim was tested for >= 1 but never against the eight slots shape[]
 * actually has, the product could wrap, the resulting byte range was
 * never compared with the buffer, and alignment was not considered. Each
 * case below is one of those, aimed at whichever CPU backends this build
 * contains — the point is that they agree.
 *
 * The positive control matters as much as the rejections: a validator
 * that refuses everything would pass a suite of only negative cases.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;

static void fail(const char *backend, const char *what) {
    fprintf(stderr, "FAIL [%s]: %s\n", backend, what);
    g_fail = 1;
}

static struct geist_tensor dense_1d(struct geist_buffer *buf, int64_t n) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = 0,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 1,
            .shape  = {n},
            .stride = {1},
    };
}

/* Feed `bad` to add() as the b operand against two well-formed tensors.
 * Any status other than GEIST_OK is acceptable — backends differ on
 * whether a malformed view is UNSUPPORTED or INVALID_ARG, and the
 * contract being tested is "refused", not "refused with this code". */
static void expect_rejected(struct geist_backend      *be,
                            const char                *backend,
                            const struct geist_tensor *good,
                            const struct geist_tensor *bad,
                            struct geist_tensor       *out,
                            const char                *what) {
    const enum geist_status s = be->desc->prims->add(be, good, bad, out);
    if (s == GEIST_OK) {
        fail(backend, what);
    }
}

static void run_backend(const char *backend) {
    const int             fails_before = g_fail;
    struct geist_backend *be           = nullptr;
    if (geist_backend_create(backend, nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("  [%s] not in this build — skipped\n", backend);
        return;
    }

    constexpr int64_t                N  = 64;
    const size_t                     nb = (size_t) N * sizeof(float);
    struct geist_buffer             *ba = nullptr;
    struct geist_buffer             *bb = nullptr;
    struct geist_buffer             *by = nullptr;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    if (v->buffer_create(be, nb, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &ba) != GEIST_OK ||
        v->buffer_create(be, nb, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bb) != GEIST_OK ||
        v->buffer_create(be, nb, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by) != GEIST_OK) {
        fail(backend, "buffer_create");
        geist_backend_destroy(be);
        return;
    }
    float zeros[N] = {0};
    (void) v->buffer_upload(ba, nb, (const uint8_t *) zeros);
    (void) v->buffer_upload(bb, nb, (const uint8_t *) zeros);

    struct geist_tensor good = dense_1d(ba, N);
    struct geist_tensor b    = dense_1d(bb, N);
    struct geist_tensor y    = dense_1d(by, N);

    /* Positive control first: without this the rejections below prove
     * nothing about the validator, only that add() can fail. */
    if (be->desc->prims->add(be, &good, &b, &y) != GEIST_OK) {
        fail(backend, "a well-formed view was rejected");
    }

    struct geist_tensor t;

    t      = b;
    t.ndim = 0;
    expect_rejected(be, backend, &good, &t, &y, "ndim = 0 accepted");

    /* shape[] has eight slots; ndim was only ever checked for >= 1. */
    t      = b;
    t.ndim = 9;
    expect_rejected(be, backend, &good, &t, &y, "ndim = 9 accepted (shape[] holds 8)");

    t          = b;
    t.shape[0] = 0;
    expect_rejected(be, backend, &good, &t, &y, "zero-length dimension accepted");

    t          = b;
    t.shape[0] = -4;
    expect_rejected(be, backend, &good, &t, &y, "negative dimension accepted");

    /* Product overflows size_t: the classic "huge logical size becomes a
     * small allocation" shape. */
    t          = b;
    t.ndim     = 3;
    t.shape[0] = INT64_MAX;
    t.shape[1] = INT64_MAX;
    t.shape[2] = 4;
    expect_rejected(be, backend, &good, &t, &y, "overflowing dimension product accepted");

    /* Internally consistent, but larger than the buffer behind it. */
    t          = b;
    t.shape[0] = N * 16;
    expect_rejected(be, backend, &good, &t, &y, "element count past the buffer accepted");

    /* Offset lands the view's tail past the end. */
    t        = b;
    t.offset = nb / 2;
    expect_rejected(be, backend, &good, &t, &y, "offset running past the buffer accepted");

    t        = b;
    t.offset = nb * 4;
    expect_rejected(be, backend, &good, &t, &y, "offset entirely past the buffer accepted");

    /* Misaligned start: a float load from an odd address. */
    t          = b;
    t.offset   = 1;
    t.shape[0] = N - 1;
    expect_rejected(be, backend, &good, &t, &y, "misaligned offset accepted");

    /* And the output tensor gets the same treatment — it is written, not
     * just read, so a bad view there is worse. */
    t      = y;
    t.ndim = 9;
    if (be->desc->prims->add(be, &good, &b, &t) == GEIST_OK) {
        fail(backend, "malformed OUTPUT view accepted");
    }

    /* Still healthy afterwards: rejecting a view must not have disturbed
     * anything. */
    if (be->desc->prims->add(be, &good, &b, &y) != GEIST_OK) {
        fail(backend, "well-formed view rejected after the malformed ones");
    }

    (void) v->buffer_destroy(be, ba);
    (void) v->buffer_destroy(be, bb);
    (void) v->buffer_destroy(be, by);
    geist_backend_destroy(be);
    printf("  [%s] %s\n", backend, g_fail == fails_before ? "OK" : "FAILED");
}

int main(void) {
    static const char *const BACKENDS[] = {"cpu_scalar", "cpu_neon", "cpu_x86"};
    for (size_t i = 0; i < sizeof(BACKENDS) / sizeof(BACKENDS[0]); i++) {
        run_backend(BACKENDS[i]);
    }
    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: malformed tensor views (ndim, dimension sign, overflowing product, "
           "buffer extent, offset, alignment) are refused by every CPU backend in "
           "this build\n");
    return GEIST_TEST_PASS;
}
