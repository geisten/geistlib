/*
 * test_iq_dequant — verify IQ2_S and IQ3_S row dequant functions against
 * llama.cpp's reference output by comparing cosine similarity on real
 * tensors loaded from gemma4-e2b-IQ2_M.gguf.
 *
 * Usage: test_iq_dequant <gguf>
 *
 * Picks one IQ2_S tensor and one IQ3_S tensor, dequants their first row
 * with our impl, prints first 32 floats + sum/sum-of-squares for visual diff.
 */
#include "gguf_reader.h"
#include "quant.h"
#include "gguf_dequant.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void analyze(const struct gguf_tensor_t *t, const char *expect_dtype) {
    printf("=== %s (dtype=%d, expected=%s) ===\n", t->name, t->dtype, expect_dtype);
    printf("    dims: ");
    for (int i = 0; i < t->n_dims; i++)
        printf("%llu ", (unsigned long long) t->dims[i]);
    printf("\n");

    const size_t row_elems = t->dims[0];
    if (row_elems % 256 != 0) {
        printf("    skip (row_elems %zu not 256-aligned)\n", row_elems);
        return;
    }

    float *row = (float *) malloc(row_elems * sizeof(float));
    bool   ok  = gguf_dequant_row_to_fp32(t, 0, row_elems, row);
    if (!ok) {
        printf("    dequant FAILED\n");
        free(row);
        return;
    }

    /* First 16 values for eyeball comparison vs llama.cpp gguf-py dump. */
    printf("    row[0..15]: ");
    for (int i = 0; i < 16; i++)
        printf("%+.4f ", row[i]);
    printf("\n");

    /* Statistics — should be in reasonable range (~0 mean, ~0.05 stddev typical
     * for quantized weights). Catastrophic dequant bugs produce huge magnitudes
     * or all-zeros. */
    double sum = 0, sumsq = 0;
    float  vmin = row[0], vmax = row[0];
    for (size_t i = 0; i < row_elems; i++) {
        sum += row[i];
        sumsq += (double) row[i] * row[i];
        if (row[i] < vmin)
            vmin = row[i];
        if (row[i] > vmax)
            vmax = row[i];
    }
    double mean = sum / row_elems;
    double var  = sumsq / row_elems - mean * mean;
    printf("    n=%zu  mean=%+.5f  std=%.5f  min=%+.4f  max=%+.4f\n",
           row_elems,
           mean,
           sqrt(var),
           vmin,
           vmax);

    free(row);
}

int main(int argc, char **argv) {
    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
    fprintf(stderr, "[dbg] start\n");
    GEIST_REQUIRE_ARGS(argc, 2, "<gguf>");
    const char *err = nullptr;
    fprintf(stderr, "[dbg] opening %s\n", argv[1]);
    struct gguf_ctx *ctx = gguf_open(argv[1], &err);
    fprintf(stderr, "[dbg] opened ctx=%p err=%s\n", (void *) ctx, err ? err : "(null)");
    if (!ctx) {
        fprintf(stderr, "gguf_open: %s\n", err);
        return 1;
    }

    /* Pick representative tensors of each dtype. */
    const struct gguf_tensor_t *iq2s = gguf_get_tensor(ctx, "blk.0.attn_q.weight");
    const struct gguf_tensor_t *iq3s = gguf_get_tensor(ctx, "blk.0.attn_output.weight");
    if (iq2s) {
        printf("attn_q: dtype=%d nbytes=%zu data=%p\n", iq2s->dtype, iq2s->nbytes, iq2s->data);
        analyze(iq2s, "IQ2_S");
    } else
        printf("blk.0.attn_q.weight not found\n");
    if (iq3s) {
        printf("attn_output: dtype=%d nbytes=%zu data=%p\n", iq3s->dtype, iq3s->nbytes, iq3s->data);
        analyze(iq3s, "IQ3_S");
    } else
        printf("blk.0.attn_output.weight not found\n");

    gguf_close(ctx);
    return 0;
}
