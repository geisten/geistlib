/*
 * Minimal safetensors reader (mmap).
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

#include <stdbool.h>
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

// BF16 weight loaders for the encoder towers. Each prints why it failed,
// prefixed by `who`, and returns failure when the tensor is missing, is
// not BF16 or has another size.
//
// st_load_bf16_f32: n_parts tensors of part_elems elements each, widened
//   to F32 and stacked in `names` order into one fresh array (a single
//   tensor: n_parts 1). Weight-sized, so heap_alloc_large: safe_free
//   returns the pages. nullptr on failure.
// st_load_bf16_scalar: a 0-d BF16 tensor into *out.
// st_load_clip4: <prefix>input_min, input_max, output_min and output_max,
//   the four scalars of a clippable linear.
float *st_load_bf16_f32(const struct st_ctx *ctx,
                        size_t               n_parts,
                        size_t               part_elems,
                        const char *const    names[static n_parts],
                        const char          *who);
bool   st_load_bf16_scalar(const struct st_ctx *ctx, const char *name, const char *who, float *out);
bool   st_load_clip4(const struct st_ctx *ctx,
                     const char          *prefix,
                     const char          *who,
                     float               *in_min,
                     float               *in_max,
                     float               *out_min,
                     float               *out_max);

#endif
