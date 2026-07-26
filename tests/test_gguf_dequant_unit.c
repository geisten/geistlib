/*
 * test_gguf_dequant — dequantizes a tensor from a GGUF file and writes
 * raw FP32 binary. validate_gguf_dequant.py compares this against gguf-py's
 * dequantization for bit-exact parity.
 *
 * The optional row cap keeps that comparison affordable. Whole-tensor dequant
 * of a large embedding table is measured in tens of GB of FP32 (gemma-4-E2B's
 * per_layer_token_embd alone is 2.3e9 elements), which no Pi and no CI runner
 * can hold. With a cap we walk the same kernels row by row instead.
 */
#include "gguf_dequant.h"
#include "gguf_reader.h"
#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 4, "<model.gguf> <tensor_name> <out.bin> [max_rows]");
    const size_t max_rows = argc > 4 ? strtoull(argv[4], nullptr, 10) : 0;

    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "gguf_open: %s\n", err);
        return 1;
    }

    const struct gguf_tensor_t *t = gguf_get_tensor(ctx, argv[2]);
    if (!t) {
        fprintf(stderr, "missing tensor: %s\n", argv[2]);
        gguf_close(ctx);
        return 1;
    }

    size_t elems = gguf_tensor_elem_count(t);
    fprintf(stderr, "tensor %s: %s, dims=[", t->name, gguf_dtype_name(t->dtype));
    for (int d = 0; d < t->n_dims; d++) {
        if (d > 0)
            fprintf(stderr, ",");
        fprintf(stderr, "%llu", (unsigned long long) t->dims[d]);
    }
    fprintf(stderr, "], %zu elems, %zu bytes\n", elems, t->nbytes);

    /* row_elems is the row length in elements = the fastest-varying dim. */
    const size_t row_elems = (size_t) t->dims[0];
    const size_t rows      = row_elems ? elems / row_elems : 0;
    const size_t want_rows = (max_rows && max_rows < rows) ? max_rows : rows;

    FILE *fo = fopen(argv[3], "wb");
    if (!fo) {
        perror("fopen out");
        gguf_close(ctx);
        return 1;
    }

    int rc = 0;
    if (want_rows == rows) {
        /* Dispatch through the engine's own tensor-aware dequant rather than a
         * switch of our own: the check then covers exactly the dtypes the
         * loader supports, and gains new ones without an edit here. */
        float *out = gguf_dequant_to_fp32(t);
        if (!out) {
            fprintf(stderr, "dequant failed for dtype %s\n", gguf_dtype_name(t->dtype));
            rc = 1;
        } else {
            xfwrite(out, sizeof(float), elems, fo);
            fprintf(stderr, "wrote %zu floats to %s\n", elems, argv[3]);
            free(out);
        }
    } else {
        /* Same kernels, one row at a time — this is the path the PLE loader
         * takes on a memory-constrained board. */
        float *row = (float *) malloc(row_elems * sizeof(float));
        if (!row) {
            rc = 1;
        }
        for (size_t r = 0; rc == 0 && r < want_rows; r++) {
            if (!gguf_dequant_row_to_fp32(t, r, row_elems, row)) {
                fprintf(stderr,
                        "row dequant failed at row %zu (%s)\n",
                        r,
                        gguf_dtype_name(t->dtype));
                rc = 1;
                break;
            }
            xfwrite(row, sizeof(float), row_elems, fo);
        }
        free(row);
        if (rc == 0) {
            fprintf(stderr,
                    "wrote %zu of %zu rows (%zu floats) to %s\n",
                    want_rows,
                    rows,
                    want_rows * row_elems,
                    argv[3]);
        }
    }

    fclose(fo);
    gguf_close(ctx);
    return rc;
}
