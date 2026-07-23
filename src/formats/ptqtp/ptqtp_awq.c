/*
 * ptqtp_awq.c — AWQ scale loader. See ptqtp_awq.h for format.
 */
#include "ptqtp_awq.h"
#include "heap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct awq_entry {
    const char  *name; /* into name_arena */
    size_t       n_in;
    const float *scales; /* into data_arena */
};

struct ptqtp_awq_ctx {
    char             *name_arena;
    size_t            name_arena_used;
    float            *data_arena;
    struct awq_entry *entries;
    size_t            n_entries;
};

static const char ERR_OPEN[]  = "awq: open() failed";
static const char ERR_TRUNC[] = "awq: file truncated / parse error";
static const char ERR_MAGIC[] = "awq: bad magic";
static const char ERR_VER[]   = "awq: unsupported version";
static const char ERR_NOMEM[] = "awq: out of memory";

void ptqtp_awq_close(struct ptqtp_awq_ctx *ctx) {
    if (!ctx)
        return;
    safe_free((void **) &ctx->name_arena);
    safe_free((void **) &ctx->data_arena);
    safe_free((void **) &ctx->entries);
    safe_free((void **) &ctx);
}

struct ptqtp_awq_ctx *ptqtp_awq_open(const char *path, const char **err) {
    if (err)
        *err = nullptr;
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (err)
            *err = ERR_OPEN;
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 12) {
        fclose(f);
        if (err)
            *err = ERR_TRUNC;
        return nullptr;
    }

    uint8_t *raw = heap_alloc_array_aligned(uint8_t, sz);
    if (!raw) {
        fclose(f);
        if (err)
            *err = ERR_NOMEM;
        return nullptr;
    }
    if (fread(raw, 1, (size_t) sz, f) != (size_t) sz) {
        safe_free((void **) &raw);
        fclose(f);
        if (err)
            *err = ERR_TRUNC;
        return nullptr;
    }
    fclose(f);

    if (memcmp(raw, "AWQS", 4) != 0) {
        safe_free((void **) &raw);
        if (err)
            *err = ERR_MAGIC;
        return nullptr;
    }
    uint32_t version = 0, n_norms = 0;
    memcpy(&version, raw + 4, 4);
    memcpy(&n_norms, raw + 8, 4);
    if (version != 1) {
        safe_free((void **) &raw);
        if (err)
            *err = ERR_VER;
        return nullptr;
    }

    struct ptqtp_awq_ctx *ctx = heap_calloc_array_aligned(struct ptqtp_awq_ctx, 1);
    if (!ctx) {
        safe_free((void **) &raw);
        if (err)
            *err = ERR_NOMEM;
        return nullptr;
    }

    /* First pass: count name + data bytes. */
    size_t pos        = 12;
    size_t name_total = 0, data_total = 0;
    for (uint32_t i = 0; i < n_norms; i++) {
        if (pos + 4 > (size_t) sz)
            goto err_trunc;
        uint32_t nlen;
        memcpy(&nlen, raw + pos, 4);
        pos += 4;
        if (pos + nlen + 4 > (size_t) sz)
            goto err_trunc;
        name_total += (size_t) nlen + 1;
        pos += nlen;
        uint32_t n_in;
        memcpy(&n_in, raw + pos, 4);
        pos += 4;
        if (pos + (size_t) n_in * 4 > (size_t) sz)
            goto err_trunc;
        data_total += (size_t) n_in * sizeof(float);
        pos += (size_t) n_in * 4;
    }

    ctx->name_arena = heap_alloc_array_aligned(char, name_total);
    ctx->data_arena = heap_alloc_aligned(data_total, alignof(float));
    ctx->entries    = heap_calloc_array_aligned(struct awq_entry, n_norms);
    if (!ctx->name_arena || !ctx->data_arena || !ctx->entries) {
        safe_free((void **) &raw);
        ptqtp_awq_close(ctx);
        if (err) {
            *err = ERR_NOMEM;
        }
        return nullptr;
    }
    ctx->n_entries = n_norms;

    /* Second pass: populate. data_pos is in float-units, not bytes. */
    pos             = 12;
    size_t data_pos = 0;
    for (uint32_t i = 0; i < n_norms; i++) {
        uint32_t nlen;
        memcpy(&nlen, raw + pos, 4);
        pos += 4;
        char *dst = ctx->name_arena + ctx->name_arena_used;
        memcpy(dst, raw + pos, nlen);
        dst[nlen] = '\0';
        ctx->name_arena_used += (size_t) nlen + 1;
        ctx->entries[i].name = dst;
        pos += nlen;
        uint32_t n_in;
        memcpy(&n_in, raw + pos, 4);
        pos += 4;
        ctx->entries[i].n_in   = n_in;
        ctx->entries[i].scales = ctx->data_arena + data_pos;
        memcpy(ctx->data_arena + data_pos, raw + pos, (size_t) n_in * sizeof(float));
        data_pos += n_in;
        pos += (size_t) n_in * sizeof(float);
    }

    safe_free((void **) &raw);
    return ctx;

err_trunc:
    safe_free((void **) &raw);
    ptqtp_awq_close(ctx);
    if (err)
        *err = ERR_TRUNC;
    return nullptr;
}

const float *ptqtp_awq_get(const struct ptqtp_awq_ctx *ctx, const char *name, size_t *n_out) {
    if (!ctx || !name)
        return nullptr;
    for (size_t i = 0; i < ctx->n_entries; i++) {
        if (strcmp(ctx->entries[i].name, name) == 0) {
            if (n_out)
                *n_out = ctx->entries[i].n_in;
            return ctx->entries[i].scales;
        }
    }
    return nullptr;
}
