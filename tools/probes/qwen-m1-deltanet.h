/* The codec and exponent adjustment are shared by C validation and Metal replay. */
#ifndef QWEN_M1_DELTANET_H
#define QWEN_M1_DELTANET_H
#include "qwen-m1-rounding.h"
#include "qwen-m1-rsqrt-source.h"

#define Q1_RSQRT_FIRST 0x3f800000u
#define Q1_RSQRT_LAST 0x407fffffu

#define Q1_RSQRT_C(...) __VA_ARGS__
Q1_RSQRT_FORMULAS(Q1_RSQRT_C)
#undef Q1_RSQRT_C

static inline Q1_U32 q1_rsqrt_bits(Q1_U32 key, Q1_READ Q1_U32 *table) {
    if (key < 0x00800000u || key > 0x7f7fffffu)
        return Q1_NAN;
    return q1_rsqrt_scale(key, q1_lookup(q1_rsqrt_key(key), table));
}

static inline Q1_U32 q1_beta_neg_bits(Q1_U32 magnitude, Q1_READ Q1_U32 *table) {
    if (magnitude > Q1_INFINITY)
        return Q1_NAN;
    if (magnitude < Q1_EXP_FIRST)
        return 0x3f000000u;
    if (magnitude > Q1_EXP_LAST)
        return 0;
    return q1_lookup(magnitude, table);
}

#endif
