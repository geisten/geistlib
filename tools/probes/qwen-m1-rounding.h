/* Qwen SiLU replay format. Raw words are little-endian on the ARM64 hosts. */
#ifndef QWEN_M1_ROUNDING_H
#define QWEN_M1_ROUNDING_H

#ifdef __METAL_VERSION__
#define Q1_U32 uint
#define Q1_I64 long
#define Q1_READ device const
#else
#include <stdint.h>
#define Q1_U32 uint32_t
#define Q1_I64 int64_t
#define Q1_READ const
#endif

#define Q1_MAGIC 0x514d3154u
#define Q1_VERSION 1u
#define Q1_BLOCK_SIZE 256u
#define Q1_HEADER_WORDS 6u
#define Q1_BLOCK_WORDS 4u
#define Q1_EXP_FIRST 0x33000000u
#define Q1_EXP_LAST 0x43000000u
#define Q1_RCP_FIRST 0x3f800000u
#define Q1_RCP_LAST 0x40000000u
#define Q1_INFINITY 0x7f800000u
#define Q1_NAN 0x7fc00000u

static inline Q1_I64 q1_predict(Q1_U32 count, Q1_U32 index, Q1_U32 first, Q1_U32 last) {
    if (count == 1)
        return first;
    return (Q1_I64) first + ((Q1_I64) last - first) * index / (count - 1);
}

/* A host reader validates the complete table before a kernel can call this. */
static inline Q1_U32 q1_lookup(Q1_U32 key, Q1_READ Q1_U32 *table) {
    const Q1_U32 first = table[2], last = table[3], blocks = table[4];
    if (key < first || key > last)
        return Q1_NAN;
    const Q1_U32    index       = key - first;
    const Q1_U32    entry       = index % Q1_BLOCK_SIZE;
    const Q1_U32    block_index = index / Q1_BLOCK_SIZE;
    Q1_READ Q1_U32 *row         = table + Q1_HEADER_WORDS + block_index * Q1_BLOCK_WORDS;
    const Q1_U32    remaining   = last - first + 1u - block_index * Q1_BLOCK_SIZE;
    const Q1_U32    count       = remaining < Q1_BLOCK_SIZE ? remaining : Q1_BLOCK_SIZE;
    const Q1_U32    width       = row[3];
    Q1_U32          code        = 0;
    if (width) {
        Q1_READ Q1_U32 *payload = table + Q1_HEADER_WORDS + blocks * Q1_BLOCK_WORDS + row[2];
        const Q1_U32    bit     = entry * width;
        const Q1_U32    offset = bit / 32u, shift = bit % 32u;
        code = payload[offset] >> shift;
        if (shift + width > 32u)
            code |= payload[offset + 1] << (32u - shift);
        if (width < 32u)
            code &= (1u << width) - 1u;
    }
    const Q1_I64 residual = (code & 1u) ? -(Q1_I64) (code / 2u) - 1 : (Q1_I64) (code / 2u);
    return (Q1_U32) (q1_predict(count, entry, row[0], row[1]) + residual);
}

static inline Q1_U32 q1_exp_bits(Q1_U32 magnitude, Q1_READ Q1_U32 *table) {
    if (magnitude > Q1_INFINITY)
        return Q1_NAN;
    if (magnitude < Q1_EXP_FIRST)
        return Q1_RCP_FIRST;
    if (magnitude > Q1_EXP_LAST)
        return 0;
    return q1_lookup(magnitude, table);
}

#endif
