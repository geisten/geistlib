#pragma once

#include <stdio.h>
#include <stdlib.h>

static void qwen47_trace(const struct transformer_layer_forward_ctx *ctx,
                         const char                                 *stage,
                         size_t                                      count,
                         const struct geist_buffer                  *buffer) {
    if (!getenv("QWEN47_TRACE") || ctx->seq != 1 || (ctx->q_position != 0 && ctx->q_position != 90))
        return;
    uint8_t *bytes = heap_alloc_array_aligned(uint8_t, count * sizeof(float));
    if (!bytes || ctx->v->buffer_download(count * sizeof(float), bytes, buffer) != GEIST_OK) {
        fputs("qwen47 buffer download failed\n", stderr);
        abort();
    }
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    printf("trace %zu %d %s %zu %016llx\n",
           ctx->q_position,
           ctx->layer_idx,
           stage,
           count,
           (unsigned long long) hash);
    safe_free((void **) &bytes);
}
