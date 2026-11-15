#pragma once

#include "checked.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void qwen47_write_vector(const char   *dir,
                                size_t        position,
                                size_t        count,
                                const char   *stage,
                                const uint8_t bytes[static count]) {
    char      path[1024];
    const int len = snprintf(path, sizeof(path), "%s/%zu-%s.f32", dir, position, stage);
    if (len < 0 || (size_t) len >= sizeof(path))
        abort();
    FILE *file = fopen(path, "wb");
    if (!file)
        abort();
    const size_t written = fwrite(bytes, 1, count, file);
    if (fclose(file) != 0 || written != count)
        abort();
}

static void qwen47_trace(const struct transformer_layer_forward_ctx *ctx,
                         const char                                 *stage,
                         size_t                                      count,
                         const struct geist_buffer                  *buffer) {
    if (!getenv("QWEN47_TRACE") || ctx->seq != 1 || (ctx->q_position != 0 && ctx->q_position != 90))
        return;
    size_t n_bytes;
    if (ckd_mul(&n_bytes, count, sizeof(float)) || n_bytes == 0)
        abort();
    uint8_t *bytes = heap_alloc_array_aligned(uint8_t, n_bytes);
    if (!bytes || ctx->v->buffer_download(n_bytes, bytes, buffer) != GEIST_OK) {
        fputs("qwen47 buffer download failed\n", stderr);
        abort();
    }
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n_bytes; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    printf("trace %zu %d %s %zu %016llx\n",
           ctx->q_position,
           ctx->layer_idx,
           stage,
           count,
           (unsigned long long) hash);
    const char *dir = getenv("QWEN47_RAW_DIR");
    if (dir && ctx->layer_idx == 0 &&
        (!strcmp(stage, "dn_qkv_input") || !strcmp(stage, "dn_qkv") || !strcmp(stage, "dn_z")))
        qwen47_write_vector(dir, ctx->q_position, n_bytes, stage, bytes);
    safe_free((void **) &bytes);
}
