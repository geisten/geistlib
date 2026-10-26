/*
 * src/base/fnv1a.h — 64-bit FNV-1a, the hash of the tokenizers' lookup
 * tables and of the MoE profile cache key (moe_profile_key: its value is
 * on disk, so this function must not change).
 *
 * Layer: ENGINE.
 */
#ifndef GEIST_BASE_FNV1A_H
#define GEIST_BASE_FNV1A_H

#include <stddef.h>
#include <stdint.h>

constexpr uint64_t GEIST_FNV1A_INIT = 0xcbf29ce484222325ull; /* offset basis */

/* h continued over bytes p[0, n); start with GEIST_FNV1A_INIT. */
static inline uint64_t geist_fnv1a(uint64_t h, size_t n, const void *p) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

#endif /* GEIST_BASE_FNV1A_H */
