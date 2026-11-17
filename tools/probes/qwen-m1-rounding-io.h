/* Read and validate all block extents before exposing the packed table. */
#ifndef QWEN_M1_ROUNDING_IO_H
#define QWEN_M1_ROUNDING_IO_H

#include "qwen-m1-rounding.h"
#include <stdbool.h>
#include <stddef.h>

static inline bool q1_table_valid(size_t count, const uint32_t *table) {
    if (count < Q1_HEADER_WORDS || table[0] != Q1_MAGIC || table[1] != Q1_VERSION ||
        table[2] > table[3] || table[3] > Q1_INFINITY)
        return false;
    const uint32_t values = table[3] - table[2] + 1;
    const uint32_t blocks = (values + Q1_BLOCK_SIZE - 1) / Q1_BLOCK_SIZE;
    if (blocks != table[4] || blocks > (count - Q1_HEADER_WORDS) / Q1_BLOCK_WORDS ||
        table[5] != count - Q1_HEADER_WORDS - (size_t) blocks * Q1_BLOCK_WORDS)
        return false;
    const uint32_t *row    = table + Q1_HEADER_WORDS;
    uint32_t        offset = 0;
    uint32_t        left   = values;
    for (uint32_t i = 0; i < blocks; i++, row += Q1_BLOCK_WORDS) {
        const uint32_t used = left < Q1_BLOCK_SIZE ? left : Q1_BLOCK_SIZE;
        if (row[2] != offset || row[3] > 32)
            return false;
        const uint32_t words = (used * row[3] + 31) / 32;
        if (words > table[5] - offset)
            return false;
        offset += words;
        left -= used;
    }
    return offset == table[5];
}

#endif
