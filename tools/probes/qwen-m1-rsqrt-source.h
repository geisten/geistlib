#ifndef QWEN_M1_RSQRT_SOURCE_H
#define QWEN_M1_RSQRT_SOURCE_H

/* Normalize positive normal FP32 keys to [1,4) with an even exponent shift.
 * The sampler verifies that the M1 rsqrt is invariant under these shifts. */
#define Q1_RSQRT_FORMULAS(X)                                                    \
    X(static inline Q1_U32 q1_rsqrt_key(Q1_U32 key) {                           \
        const Q1_U32 exponent  = key >> 23;                                     \
        const Q1_U32 canonical = 128u - (exponent & 1u);                        \
        return (canonical << 23) | (key & 0x007fffffu);                         \
    })                                                                          \
    X(static inline Q1_U32 q1_rsqrt_scale(Q1_U32 key, Q1_U32 canonical_rsqrt) { \
        const Q1_U32 exponent  = key >> 23;                                     \
        const Q1_U32 canonical = 128u - (exponent & 1u);                        \
        const int    shift     = ((int) canonical - (int) exponent) / 2;        \
        return canonical_rsqrt + (Q1_U32) shift * 0x00800000u;                  \
    })

#define Q1_RSQRT_METAL(...) #__VA_ARGS__ "\n"
#define Q1_RSQRT_SHADER "#define Q1_U32 uint\n" Q1_RSQRT_FORMULAS(Q1_RSQRT_METAL) "#undef Q1_U32\n"

#endif
