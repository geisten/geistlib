/*
 * test_tensor_view_unit — malformed tensor views must be refused, on
 * every CPU backend (issue #330).
 *
 * The elementwise and transformer ops unpack a view by folding
 * shape[0..ndim) and adding `offset` to the host pointer. Each case below breaks one check that
 * unpacking needs — ndim within the eight slots shape[] has, a product
 * that does not wrap, a byte range inside the buffer, alignment — aimed at
 * whichever CPU backends this build contains; the point is that they agree.
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
    g_fail++;
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

    /* shape[] has eight slots. */
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

    /* The transformer ops unpack their views through the same check (they
     * once folded the shape without comparing it to the buffer). Rows of
     * 4 x 4 fit the 64 floats behind them; 1024 rows do not, and the op
     * would read and write 16 KiB past the end. */
    if (be->desc->prims->rope_apply != nullptr) {
        struct geist_tensor x = good, cs = good;
        x.ndim       = 3;
        x.shape[0]   = 4;
        x.shape[1]   = 1;
        x.shape[2]   = 4;
        x.stride[0]  = 4;
        x.stride[1]  = 4;
        x.stride[2]  = 1;
        cs.ndim      = 2;
        cs.shape[0]  = 4;
        cs.shape[1]  = 4;
        cs.stride[0] = 4;
        cs.stride[1] = 1;
        if (be->desc->prims->rope_apply(be, &x, &cs, &cs) != GEIST_OK) {
            fail(backend, "rope_apply rejected views that fit their buffer");
        }
        x.shape[0]  = 1024;
        cs.shape[0] = 1024;
        if (be->desc->prims->rope_apply(be, &x, &cs, &cs) == GEIST_OK) {
            fail(backend, "rope_apply accepted views past their buffer");
        }
    }

    /* And views that fit their buffers but not x (#67): rope_apply reads
     * seq * n_rot floats of cos and sin and rotates the first n_rot dims
     * of each head. x [2,2,8] in ba; cos and sin [2,8] side by
     * side in bb. */
    if (be->desc->prims->rope_apply != nullptr) {
        struct geist_tensor x = good, cs = b, sn = b;
        x.ndim       = 3;
        x.shape[0]   = 2;
        x.shape[1]   = 2;
        x.shape[2]   = 8;
        x.stride[0]  = 16;
        x.stride[1]  = 8;
        x.stride[2]  = 1;
        cs.ndim      = 2;
        cs.shape[0]  = 2;
        cs.shape[1]  = 8;
        cs.stride[0] = 8;
        cs.stride[1] = 1;
        sn           = cs;
        sn.offset    = 16 * sizeof(float);
        if (be->desc->prims->rope_apply(be, &x, &cs, &sn) != GEIST_OK) {
            fail(backend, "rope_apply rejected well-formed views");
        }
        static const struct {
            int64_t     cos_rows, sin_rows, n_rot;
            const char *what;
        } ROPE_BAD[] = {
                {1, 2, 8, "rope_apply accepted cos with fewer rows than x"},
                {2, 1, 8, "rope_apply accepted sin with fewer rows than x"},
                {2, 2, 10, "rope_apply accepted n_rot > head_dim"},
                {2, 2, 7, "rope_apply accepted an odd n_rot"},
        };
        for (size_t i = 0; i < sizeof(ROPE_BAD) / sizeof(ROPE_BAD[0]); i++) {
            struct geist_tensor c = cs, sb = sn;
            c.shape[0]  = ROPE_BAD[i].cos_rows;
            sb.shape[0] = ROPE_BAD[i].sin_rows;
            c.shape[1] = sb.shape[1] = ROPE_BAD[i].n_rot;
            c.stride[0] = sb.stride[0] = ROPE_BAD[i].n_rot;
            if (be->desc->prims->rope_apply(be, &x, &c, &sb) == GEIST_OK) {
                fail(backend, ROPE_BAD[i].what);
            }
        }
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
