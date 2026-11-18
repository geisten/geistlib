#ifndef QWEN_M1_SOFTPLUS_H
#define QWEN_M1_SOFTPLUS_H

#include "qwen-m1-rounding.h"

static inline Q1_U32
q1_softplus_bits(Q1_U32 key, Q1_READ Q1_U32 *positive, Q1_READ Q1_U32 *negative) {
    const Q1_U32 magnitude = key & 0x7fffffffu;
    if (magnitude > Q1_INFINITY)
        return Q1_NAN;
    if (magnitude < Q1_EXP_FIRST)
        return 0x3f317218u;
    const bool neg = (key >> 31) != 0;
    if (magnitude > Q1_EXP_LAST)
        return neg ? 0u : Q1_INFINITY;
    return q1_lookup(magnitude, neg ? negative : positive);
}

#endif
