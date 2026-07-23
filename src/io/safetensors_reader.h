/*
 * Minimal safetensors reader for Gemma 4 bringup.
 *
 * Format reference:
 *   bytes 0..7   uint64 little-endian: header_size
 *   bytes 8..8+header_size: UTF-8 JSON header
 *   bytes 8+header_size..end: tensor data, contiguous
 *
 * Header JSON layout:
 *   {
 *     "__metadata__": {...},                                    (optional)
 *     "tensor_name_1": {"dtype": "F32", "shape": [..],
 *                       "data_offsets": [start, end]},
 *     ...
 *   }
 *
 * data_offsets are relative to the start of the data region
 * (i.e. byte 8 + header_size).
 */
#ifndef SAFETENSORS_READER_H
#define SAFETENSORS_READER_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    ST_DTYPE_UNKNOWN = 0,
    ST_DTYPE_F64,
    ST_DTYPE_F32,
    ST_DTYPE_F16,
    ST_DTYPE_BF16,
    ST_DTYPE_I64,
    ST_DTYPE_I32,
    ST_DTYPE_I16,
    ST_DTYPE_I8,
    ST_DTYPE_U64,
    ST_DTYPE_U32,
    ST_DTYPE_U16,
    ST_DTYPE_U8,
    ST_DTYPE_BOOL,
} st_dtype_t;

#define ST_MAX_RANK 8

struct st_tensor_t {
    const char *name; // owned by ctx, NUL-terminated
    st_dtype_t  dtype;
    size_t      rank;
    size_t      shape[ST_MAX_RANK];
    size_t      nbytes;
    const void *data; // points into mmap'd region
};

struct st_ctx;

// Open file (mmap), parse header. Returns NULL on error;
// errmsg (if non-NULL) is filled with a static or heap string explaining why.
struct st_ctx *st_open(const char *path, const char **errmsg);

size_t                    st_count(const struct st_ctx *ctx);
const struct st_tensor_t *st_get(const struct st_ctx *ctx, const char *name);

size_t      st_dtype_bytes(st_dtype_t dtype);
const char *st_dtype_name(st_dtype_t dtype);

void st_close(struct st_ctx *ctx);

#endif
