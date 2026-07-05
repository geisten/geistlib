/*
 * int4_kv.h — symmetric signed 4-bit pack/unpack for the packed-INT4 KV
 * cache (issue #61). Two values per byte: low nibble = even index, high
 * nibble = odd index. Values quantize to [-7,7], stored as 4-bit two's
 * complement. n must be even (head_dim is a power of two).
 */
#ifndef GEIST_INT4_KV_H
#define GEIST_INT4_KV_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Quantize `n` floats at scale `inv` (= 1/scale) into n/2 packed bytes. */
static inline void int4_pack_row(const float *x, float inv, uint8_t *out, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        const int lo = (int) lrintf(x[i] * inv);
        const int hi = (int) lrintf(x[i + 1] * inv);
        out[i >> 1]  = (uint8_t) ((lo & 0x0F) | ((hi & 0x0F) << 4));
    }
}

/* Unpack n/2 bytes into `n` sign-extended int8 values in [-8,7]. */
static inline void int4_unpack_row(const uint8_t *in, int8_t *out, size_t n) {
    for (size_t j = 0; j < n / 2; j++) {
        const uint8_t b = in[j];
        out[2 * j]      = (int8_t) ((int8_t) (b << 4) >> 4);   /* low nibble  */
        out[2 * j + 1]  = (int8_t) ((int8_t) (b & 0xF0) >> 4); /* high nibble */
    }
}

#endif /* GEIST_INT4_KV_H */
