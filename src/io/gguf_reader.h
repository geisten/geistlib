/*
 * Minimal GGUF v3 reader for Gemma 4 Q4_K_M / Q8_0 / F16 / F32 inference.
 *
 * Format ref: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 *
 * File layout:
 *   magic[4] = "GGUF"
 *   version[4] = 3
 *   tensor_count[8]
 *   metadata_kv_count[8]
 *   metadata_kvs (variable)        — KV pairs of typed metadata
 *   tensor_infos (tensor_count)    — name, dims, dtype, offset
 *   <padding to alignment>
 *   tensor_data                     — raw bytes, accessed via offsets
 *
 * This reader skips metadata values (we hardcode model params from config)
 * but walks the KV section to find the tensor_info and data offsets.
 * Typed metadata access via gguf_get_meta_*() (added P1.4.d).
 */
#ifndef GGUF_READER_H
#define GGUF_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    GGUF_TYPE_F32   = 0,
    GGUF_TYPE_F16   = 1,
    GGUF_TYPE_Q4_0  = 2,
    GGUF_TYPE_Q4_1  = 3,
    GGUF_TYPE_Q5_0  = 6,
    GGUF_TYPE_Q5_1  = 7,
    GGUF_TYPE_Q8_0  = 8,
    GGUF_TYPE_Q8_1  = 9,
    GGUF_TYPE_Q2_K  = 10,
    GGUF_TYPE_Q3_K  = 11,
    GGUF_TYPE_Q4_K  = 12,
    GGUF_TYPE_Q5_K  = 13,
    GGUF_TYPE_Q6_K  = 14,
    GGUF_TYPE_Q8_K  = 15,
    GGUF_TYPE_IQ3_S = 21, /* 3.4 bpw, codebook-based */
    GGUF_TYPE_IQ2_S = 22, /* 2.5 bpw, codebook-based */
    GGUF_TYPE_BF16  = 30,
    GGUF_TYPE_TQ1_0 = 34, /* 1.69 bpw ternary; 256-elem block; 5 trits/byte */
    GGUF_TYPE_TQ2_0 = 35, /* 2.06 bpw ternary; 256-elem block; 4 trits/byte */
    /* I2_S is Microsoft bitnet.cpp's custom ggml extension type for BitNet
     * b1.58 weights — NOT in mainline llama.cpp. 2-bit packed signed ternary
     * (4 trits/byte, 2.0 bpw, 256-elem/64-byte blocks) + ONE f32 per-TENSOR
     * scale at the tail (offset n_elems/4). geist reads it as ternary and
     * transcodes the trit field-order to its TQ2_0 SDOT kernel. */
    GGUF_TYPE_I2_S = 36,
} gguf_dtype_t;

/* Metadata value-type codes (mirror of llama.cpp). Exposed so callers
 * of gguf_get_meta_array_info() can interpret the returned elem_vt. */
enum {
    GGUF_META_VT_U8     = 0,
    GGUF_META_VT_I8     = 1,
    GGUF_META_VT_U16    = 2,
    GGUF_META_VT_I16    = 3,
    GGUF_META_VT_U32    = 4,
    GGUF_META_VT_I32    = 5,
    GGUF_META_VT_F32    = 6,
    GGUF_META_VT_BOOL   = 7,
    GGUF_META_VT_STRING = 8,
    GGUF_META_VT_ARRAY  = 9,
    GGUF_META_VT_U64    = 10,
    GGUF_META_VT_I64    = 11,
    GGUF_META_VT_F64    = 12,
};

#define GGUF_MAX_DIMS 4

struct gguf_tensor_t {
    const char  *name; /* NUL-terminated, owned by ctx */
    gguf_dtype_t dtype;
    int          n_dims;
    uint64_t     dims[GGUF_MAX_DIMS];
    uint64_t     offset; /* relative to data section start */
    size_t       nbytes; /* total tensor size */
    const void  *data;   /* points into mmap region */
};

struct gguf_ctx;

/* Open + parse header. errmsg is a static-string-or-heap pointer on failure. */
struct gguf_ctx *gguf_open(const char *path, const char **errmsg);

/* Parse a GGUF already in memory (e.g. embedded in the binary). The buffer is
 * aliased read-only and NOT freed by gguf_close — the caller must keep it alive
 * for the ctx's lifetime. Backs geist_model_load_from_memory. */
struct gguf_ctx *gguf_open_memory(const void *data, size_t size, const char **errmsg);

void gguf_close(struct gguf_ctx *ctx);

size_t                      gguf_tensor_count(const struct gguf_ctx *ctx);
const struct gguf_tensor_t *gguf_tensor_at(const struct gguf_ctx *ctx, size_t idx);
const struct gguf_tensor_t *gguf_get_tensor(const struct gguf_ctx *ctx, const char *name);

const char *gguf_dtype_name(gguf_dtype_t dt);

/* Total element count across all dims of a tensor. */
size_t gguf_tensor_elem_count(const struct gguf_tensor_t *t);

/* ---- Metadata KV access (P1.4.d) -------------------------------------- *
 *
 * Typed getters for `metadata_kv` entries. All return `false`/`nullptr`
 * when the key is missing or the on-disk value type doesn't match the
 * requested type. Keys are NUL-terminated; callers compare against the
 * llama.cpp convention "<arch>.<group>.<name>".
 *
 * String values point into the mmap region — valid for the ctx lifetime.
 * `out_len` excludes any NUL terminator (GGUF strings are length-prefixed
 * and may be empty). */
const char *gguf_get_meta_string(const struct gguf_ctx *ctx, const char *key, size_t *out_len);
bool        gguf_get_meta_u32(const struct gguf_ctx *ctx, const char *key, uint32_t *out);
bool        gguf_get_meta_f32(const struct gguf_ctx *ctx, const char *key, float *out);
bool        gguf_get_meta_bool(const struct gguf_ctx *ctx, const char *key, bool *out);

/* Array access (P1.5.f). Returns false when the key is missing or the
 * value type isn't ARRAY. Caller receives the GGUF element type code
 * (GGUF_VT_*), element count, and a pointer at the first element's
 * payload — i.e. just past the (u32 elem_vt + u64 count) array header.
 * The caller walks the elements based on `elem_vt`: for STRING that
 * means a sequence of (u64 length + bytes) pairs; for the scalar types
 * (U32/F32/etc.) it's a flat packed array of fixed-width values.
 *
 * The payload pointer is valid for the ctx's lifetime (points into
 * the mmap region). */
bool gguf_get_meta_array_info(const struct gguf_ctx *ctx,
                              const char            *key,
                              uint32_t              *out_elem_vt,
                              uint64_t              *out_count,
                              const uint8_t        **out_payload);

#endif
