#pragma once

#include "checked.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t qwen47_hash(size_t n_bytes, const uint8_t bytes[static n_bytes]) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n_bytes; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

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
    const uint64_t hash = qwen47_hash(n_bytes, bytes);
    printf("trace %zu %d %s %zu %016llx\n",
           ctx->q_position,
           ctx->layer_idx,
           stage,
           count,
           (unsigned long long) hash);
    const char *dir = getenv("QWEN47_RAW_DIR");
    if (dir && ctx->layer_idx == 0 &&
        (!strcmp(stage, "dn_qkv_input") || !strcmp(stage, "dn_qkv") || !strcmp(stage, "dn_z") ||
         !strcmp(stage, "dn_beta") || !strcmp(stage, "dn_alpha") || !strcmp(stage, "dn_state") ||
         !strcmp(stage, "dn_qkv_after") || !strcmp(stage, "dn_mix") ||
         !strcmp(stage, "dn_out_projection")))
        qwen47_write_vector(dir, ctx->q_position, n_bytes, stage, bytes);
    safe_free((void **) &bytes);
}

/* Exact M1 layer-zero reference for the differing channels in case 47. */
static inline void qwen47_patch_qkv(const struct transformer_layer_forward_ctx *ctx,
                                    size_t                                      count,
                                    struct geist_buffer                        *buffer) {
    if (!getenv("QWEN47_PATCH_QKV") || ctx->q_position != 0 || ctx->layer_idx != 0 || ctx->seq != 1)
        return;
    const struct {
        size_t   channel;
        uint32_t bits;
    } values[] = {
            {397, UINT32_C(0x050233bb)},
            {423, UINT32_C(0x029f6b50)},
            {438, UINT32_C(0x0393c458)},
            {453, UINT32_C(0x81aa9d60)},
            {470, UINT32_C(0x8515711c)},
            {476, UINT32_C(0x0405a6e6)},
            {479, UINT32_C(0x04224c0f)},
            {2445, UINT32_C(0x84183557)},
            {2471, UINT32_C(0x03e2b36f)},
            {2486, UINT32_C(0x050e9f0b)},
            {2501, UINT32_C(0x04c7a17d)},
            {2518, UINT32_C(0x0524e56d)},
            {2524, UINT32_C(0x8422e90b)},
            {2527, UINT32_C(0x04a6780c)},
    };
    size_t n_bytes;
    if (count != 6144 || ckd_mul(&n_bytes, count, sizeof(float)))
        abort();
    uint8_t *bytes = heap_alloc_array_aligned(uint8_t, n_bytes);
    if (!bytes || ctx->v->buffer_download(n_bytes, bytes, buffer) != GEIST_OK ||
        qwen47_hash(n_bytes, bytes) != UINT64_C(0x09b2e7a2c3bcce5c)) {
        fputs("Unexpected M6 source QKV vector\n", stderr);
        abort();
    }
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i)
        memcpy(bytes + values[i].channel * sizeof(float), &values[i].bits, sizeof(float));
    if (qwen47_hash(n_bytes, bytes) != UINT64_C(0x3e4d5851646aad12) ||
        ctx->v->buffer_upload(buffer, n_bytes, bytes) != GEIST_OK) {
        fputs("Could not set the exact M1 QKV vector\n", stderr);
        abort();
    }
    safe_free((void **) &bytes);
}
