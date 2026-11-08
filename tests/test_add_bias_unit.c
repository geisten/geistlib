/*
 * test_add_bias_unit — prims->add_bias (#492) on every compiled backend:
 * one broadcast dispatch must give the same bits as one prims->add per row
 * (what blk_add_bias ran before) and as the host's x + b, in place and out
 * of place, with the bias at an offset inside its buffer as the loader
 * places it. Backends that are not compiled in, or have no add_bias, skip.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <string.h>

enum { MAX_ROWS = 203, MAX_COLS = 2880, B_PAD = 3 };

static float g_x[MAX_ROWS * MAX_COLS];
static float g_b[B_PAD + MAX_COLS];
static float g_ref[MAX_ROWS * MAX_COLS];
static float g_rows[MAX_ROWS * MAX_COLS];
static float g_got[MAX_ROWS * MAX_COLS];

static struct geist_tensor
rows_view(struct geist_buffer *buf, size_t off_floats, size_t rows, size_t cols) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = (int64_t) (off_floats * sizeof(float)),
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 2,
            .shape  = {(int64_t) rows, (int64_t) cols},
            .stride = {(int64_t) cols, 1},
    };
}

static struct geist_buffer *upload(struct geist_backend *be, size_t n, const float *src) {
    struct geist_buffer *buf = nullptr;
    if (be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &buf) !=
                GEIST_OK ||
        be->desc->vtbl->buffer_upload(buf, n * sizeof(float), (const uint8_t *) src) != GEIST_OK) {
        return nullptr;
    }
    return buf;
}

/* One shape on one backend: add_bias in place and into a separate y, and
 * prims->add row by row; all three bit-identical to the host's x + b. */
static int check_shape(struct geist_backend *be, const char *name, size_t rows, size_t cols) {
    const size_t n = rows * cols;
    for (size_t i = 0; i < n; i++)
        g_x[i] = (float) ((int) (i * 2654435761u % 2001u) - 1000) * 1.37e-3f;
    for (size_t c = 0; c < B_PAD + cols; c++)
        g_b[c] = (float) ((int) (c * 40503u % 977u) - 488) * 3.1e-2f;
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            g_ref[r * cols + c] = g_x[r * cols + c] + g_b[B_PAD + c];

    const struct geist_backend_vtbl *vt  = be->desc->vtbl;
    struct geist_buffer             *bx  = upload(be, n, g_x);
    struct geist_buffer             *bb  = upload(be, B_PAD + cols, g_b);
    struct geist_buffer             *by  = upload(be, n, g_x);
    struct geist_buffer             *bxr = upload(be, n, g_x);
    int bad = bx == nullptr || bb == nullptr || by == nullptr || bxr == nullptr;
    if (!bad) {
        struct geist_tensor t_x = rows_view(bx, 0, rows, cols);
        struct geist_tensor t_y = rows_view(by, 0, rows, cols);
        struct geist_tensor t_b = rows_view(bb, B_PAD, 1, cols);
        /* in place, as blk_add_bias calls it */
        bad |= be->desc->prims->add_bias(be, &t_x, &t_b, &t_x) != GEIST_OK;
        /* out of place, y over a copy of x it must overwrite */
        struct geist_tensor t_x0 = rows_view(bxr, 0, rows, cols);
        bad |= be->desc->prims->add_bias(be, &t_x0, &t_b, &t_y) != GEIST_OK;
        /* the old path: one add per row, on the x copy */
        for (size_t r = 0; r < rows && !bad; r++) {
            struct geist_tensor t_r = rows_view(bxr, r * cols, 1, cols);
            bad |= be->desc->prims->add(be, &t_r, &t_b, &t_r) != GEIST_OK;
        }
    }
    if (!bad) {
        bad |= vt->buffer_download(n * sizeof(float), (uint8_t *) g_got, bx) != GEIST_OK;
        bad |= memcmp(g_got, g_ref, n * sizeof(float)) != 0;
        if (bad)
            fprintf(stderr, "FAIL %s add_bias in place %zux%zu\n", name, rows, cols);
    }
    if (!bad) {
        bad |= vt->buffer_download(n * sizeof(float), (uint8_t *) g_got, by) != GEIST_OK;
        bad |= vt->buffer_download(n * sizeof(float), (uint8_t *) g_rows, bxr) != GEIST_OK;
        bad |= memcmp(g_got, g_ref, n * sizeof(float)) != 0;
        bad |= memcmp(g_rows, g_ref, n * sizeof(float)) != 0;
        if (bad)
            fprintf(stderr, "FAIL %s add_bias into y / per-row add %zux%zu\n", name, rows, cols);
    }
    struct geist_buffer *bufs[] = {bx, bb, by, bxr};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if (bufs[i] != nullptr)
            (void) vt->buffer_destroy(be, bufs[i]);
    return bad;
}

int main(void) {
    /* gpt-oss's q / kv / o widths (4096, 512, 2880) and ragged ones */
    static const size_t shapes[][2] = {
            {1, 4096}, {7, 512}, {64, 2880}, {203, 2880}, {3, 1}, {17, 77}, {128, 4096}};
    static const char *const names[] = {"cpu_scalar", "cpu_neon", "cpu_x86", "metal", "vulkan"};
    int                      fails = 0, ran = 0;
    for (size_t k = 0; k < sizeof names / sizeof names[0]; k++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(names[k], nullptr, nullptr, &be) != GEIST_OK) {
            printf("  %s: not available, skipped\n", names[k]);
            continue;
        }
        if (be->desc->prims->add_bias == nullptr) {
            printf("  %s: no add_bias, skipped\n", names[k]);
            geist_backend_destroy(be);
            continue;
        }
        for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
            if (shapes[s][0] * shapes[s][1] <= (size_t) MAX_ROWS * MAX_COLS)
                fails += check_shape(be, names[k], shapes[s][0], shapes[s][1]);
        printf("  %s: add_bias bit-identical to per-row add\n", names[k]);
        ran++;
        geist_backend_destroy(be);
    }
    if (ran == 0) {
        fprintf(stderr, "FAIL: no backend with add_bias\n");
        return 1;
    }
    if (fails != 0) {
        fprintf(stderr, "test_add_bias_unit: %d failure(s)\n", fails);
        return 1;
    }
    printf("test_add_bias_unit: OK\n");
    return 0;
}
