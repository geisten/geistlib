/*
 * geist_types.h — low-level tensor / op / dtype type definitions.
 *
 * Foundation layer: pure data definitions, no behaviour, no dependency on
 * the rest of the public API. This is **backend-author territory** — a
 * program that only loads and runs a model never needs it (use <geist.h>).
 * It is pulled in by <geist_backend.h> (the backend vtable) and
 * <geist_weight.h> (the weight loader).
 *
 * Split out of <geist.h> in 0.2.0 so the model-running surface stays small.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/* Tensor types — dtype = logical, layout = physical storage               */
/* ====================================================================== */

enum geist_dtype {
    GEIST_DTYPE_F32,
    GEIST_DTYPE_F16,
    GEIST_DTYPE_BF16,

    GEIST_DTYPE_I8,
    GEIST_DTYPE_U8,

    /* GGUF traditional quants (32-element blocks). */
    GEIST_DTYPE_Q4_0,
    GEIST_DTYPE_Q8_0,

    /* GGUF k-quants (256-element super-blocks). Logically signed
     * fixed-point with per-block scale (and optional min for Q4_K/Q5_K).
     * Use layout=GEIST_LAYOUT_BLOCK_QUANTIZED. */
    GEIST_DTYPE_Q3_K,
    GEIST_DTYPE_Q4_K,
    GEIST_DTYPE_Q5_K,
    GEIST_DTYPE_Q6_K,

    /* GGUF IQ-quants (256-element super-blocks). Codebook-based lookups
     * instead of straight quantization grid: ~2.56 bpw (IQ2_S) and
     * ~3.44 bpw (IQ3_S). Both used by IQ2_M / IQ3_S model variants.
     * Use layout=GEIST_LAYOUT_BLOCK_QUANTIZED. */
    GEIST_DTYPE_IQ2_S,
    GEIST_DTYPE_IQ3_S,

    /* Ternary GGUF quants for 1.58-bit models (BitNet, future). 256-elem
     * super-blocks with one fp16 scale; TQ1_0 packs 5 trits per byte
     * (1.6875 bpw), TQ2_0 packs 4 trits per byte (2.0625 bpw). Logical
     * values {-1, 0, +1} × scale. layout=GEIST_LAYOUT_BLOCK_QUANTIZED. */
    GEIST_DTYPE_TQ1_0,
    GEIST_DTYPE_TQ2_0,

    /* BitNet b1.58 official format (GGML_TYPE_I2_S): 2.0 bpw ternary, 256-elem
     * blocks of 64 packed bytes with NO per-block scale — instead ONE f32
     * per-tensor scale stored at the tensor tail (offset n_in*n_out/4). Same
     * {-1,0,+1} trit codes as TQ2_0; only the in-byte 2-bit field order differs
     * (i2_s puts element 32*g+b at shift 6-2g, TQ2_0 at shift 2g). Microsoft's
     * distribution format for BitNet-2B-4T. layout=GEIST_LAYOUT_BLOCK_QUANTIZED. */
    GEIST_DTYPE_I2_S,

    GEIST_DTYPE_BINARY,  /* 1-bit values; storage via layout */
    GEIST_DTYPE_TERNARY, /* {-1, 0, +1}; storage via layout */

    GEIST_DTYPE_CUSTOM,  /* user-extended via geist_quant_desc.flags */
};

enum geist_layout {
    GEIST_LAYOUT_DENSE,
    GEIST_LAYOUT_PACKED_1BIT,
    GEIST_LAYOUT_PACKED_2BIT,
    GEIST_LAYOUT_TERNARY_BITPLANE,
    GEIST_LAYOUT_TERNARY_BASE3,
    GEIST_LAYOUT_BLOCK_QUANTIZED, /* k-quants family — see quant_desc */
    GEIST_LAYOUT_CUSTOM,
};

enum geist_buffer_role {
    GEIST_BUFFER_WEIGHT,     /* read-only, lifetime = model */
    GEIST_BUFFER_ACTIVATION, /* read+write, lifetime = forward pass */
    GEIST_BUFFER_KV_CACHE,   /* read+write, lifetime = session */
    GEIST_BUFFER_SCRATCH,    /* short-lived workspace */
    GEIST_BUFFER_IO,         /* user-provided host buffer */
    GEIST_BUFFER_STAGING,    /* host-mapped, for upload/download */
};

enum geist_memory_flags {
    GEIST_MEMORY_AUTO         = 0,
    GEIST_MEMORY_HOST         = 1 << 0,
    GEIST_MEMORY_DEVICE       = 1 << 1,
    GEIST_MEMORY_HOST_VISIBLE = 1 << 2,
    GEIST_MEMORY_MAPPED       = 1 << 3,
    /* The buffer's host pointer aliases external storage (e.g. an mmap
     * region owned by the GGUF reader). buffer_destroy frees only the
     * buffer-handle metadata; the caller retains ownership of the
     * underlying bytes. Created via buffer_create_aliased. */
    GEIST_MEMORY_ALIASED      = 1 << 4,
};

/* Quantization metadata for block-quantized layouts.
 * bits_per_value as rational num/den so 1.58-bit (158/100) is exact. */
struct geist_quant_desc {
    int bits_per_value_num;
    int bits_per_value_den;

    int block_size;
    int values_per_block;

    enum geist_dtype scale_dtype;
    size_t           scale_offset;

    enum geist_dtype zero_dtype;
    size_t           zero_offset;

    unsigned int flags; /* CUSTOM-dtype subtype tag */
};

/* Buffer = byte-oriented handle. impl is backend-private. */
struct geist_buffer;

/* Tensor = view onto a buffer with dtype, layout, shape, offset.
 * Strides only meaningful for DENSE layouts. */
struct geist_tensor {
    struct geist_buffer *buffer;
    size_t               offset;

    enum geist_dtype  dtype;
    enum geist_layout layout;

    int     ndim;
    int64_t shape[8];
    int64_t stride[8];

    struct geist_quant_desc quant;
};

/* ====================================================================== */
/* Backend Op Vocab + Capability                                           */
/* ====================================================================== */

enum geist_op {
    /* Shared */
    GEIST_OP_LINEAR,           /* y = x @ W */
    GEIST_OP_RMSNORM,
    GEIST_OP_RESIDUAL_ADD,
    GEIST_OP_SILU_GATE,
    GEIST_OP_EMBEDDING_LOOKUP,

    /* Transformer-specific */
    GEIST_OP_ATTENTION,        /* fused QK^T → softmax → V */
    GEIST_OP_ROPE,

    GEIST_OP_COUNT,
};

enum geist_support {
    GEIST_SUPPORT_NONE,     /* backend cannot do this combination */
    GEIST_SUPPORT_EMULATED, /* backend can do it, but slow path */
    GEIST_SUPPORT_NATIVE,   /* backend has native fast implementation */
};

/* Storage description — what kind of tensor (without the data). */
struct geist_tensor_format {
    enum geist_dtype  dtype;
    enum geist_layout layout;

    int storage_bits_per_value_num;
    int storage_bits_per_value_den;

    int block_values;
    int block_bytes;
};

/* Capability query — describes a complete op signature. */
struct geist_op_support_query {
    enum geist_op op;

    int                       input_count;
    struct geist_tensor_format inputs[8];

    int                       output_count;
    struct geist_tensor_format outputs[4];
};

#ifdef __cplusplus
} /* extern "C" */
#endif
