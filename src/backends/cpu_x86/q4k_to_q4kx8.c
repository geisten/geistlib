/*
 * src/backends/cpu_x86/q4k_to_q4kx8.c — Q4_K → Q4_Kx8 repacker.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Byte-permutation only; runs once per Q4_K weight at model load. Adapted
 * from llama.cpp's make_block_q4_Kx8 (ggml/src/ggml-cpu/repack.cpp:2836).
 * Original code Copyright (c) 2023-2025 The ggml authors, MIT-licensed.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "q4k_to_q4kx8.h"

#include "quant.h"
#include "quant_blocks.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* blck_size_interleave = 8: each src column of 8 bytes goes to one dst
 * column at a stride of 64 bytes (8 source rows × 8 bytes per stripe). */
constexpr unsigned BLCK_SIZE_INTERLEAVE = 8;

static void make_block_q4_kx8(const struct block_q4_K_t *in, struct block_q4_Kx8 *out) {
    /* Copy fp16 d / dmin per source row. */
    for (int i = 0; i < 8; i++) {
        out->d[i]    = in[i].d;
        out->dmin[i] = in[i].dmin;
    }

    /* Interleave the 256-element (= 128-byte) quant body of each row.
     * end = QK_K * 4 / blck = 256 * 4 / 8 = 128 stripes total
     * (8 rows × 16 stripes per row, but mapped to a row-interleaved order). */
    const int end = Q4_K_BLOCK_ELEMS * 4 / (int) BLCK_SIZE_INTERLEAVE;
    for (int i = 0; i < end; i++) {
        const int src_id     = i % 8;
        const int src_offset = (i / 8) * (int) BLCK_SIZE_INTERLEAVE;
        const int dst_offset = i * (int) BLCK_SIZE_INTERLEAVE;
        uint64_t  elems;
        memcpy(&elems, &in[src_id].qs[src_offset], BLCK_SIZE_INTERLEAVE);
        memcpy(&out->qs[dst_offset], &elems, BLCK_SIZE_INTERLEAVE);
    }

    /* Repack the 12-byte scales descriptor for each row into the 96-byte
     * Q4_Kx8 scales array. The format is laid out so the GEMM kernel can
     * extract sub-block scales + mins via memcpy + bit-masking. */
    uint8_t s[8], m[8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }
        out->scales[i * 12 + 0]  = (uint8_t) ((s[0] & 63) + ((s[4] & 48) << 2));
        out->scales[i * 12 + 1]  = (uint8_t) ((s[1] & 63) + ((s[5] & 48) << 2));
        out->scales[i * 12 + 2]  = (uint8_t) ((s[2] & 63) + ((s[6] & 48) << 2));
        out->scales[i * 12 + 3]  = (uint8_t) ((s[3] & 63) + ((s[7] & 48) << 2));
        out->scales[i * 12 + 4]  = (uint8_t) ((m[0] & 63) + ((m[4] & 48) << 2));
        out->scales[i * 12 + 5]  = (uint8_t) ((m[1] & 63) + ((m[5] & 48) << 2));
        out->scales[i * 12 + 6]  = (uint8_t) ((m[2] & 63) + ((m[6] & 48) << 2));
        out->scales[i * 12 + 7]  = (uint8_t) ((m[3] & 63) + ((m[7] & 48) << 2));
        out->scales[i * 12 + 8]  = (uint8_t) ((s[4] & 15) + ((m[4] & 15) << 4));
        out->scales[i * 12 + 9]  = (uint8_t) ((s[5] & 15) + ((m[5] & 15) << 4));
        out->scales[i * 12 + 10] = (uint8_t) ((s[6] & 15) + ((m[6] & 15) << 4));
        out->scales[i * 12 + 11] = (uint8_t) ((s[7] & 15) + ((m[7] & 15) << 4));
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = (uint8_t) (((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15));
            m[j] = (uint8_t) (((in[j].scales[i + 4] & 192) >> 2) |
                              ((in[j].scales[i + 8] & 240) >> 4));
        }
        out->scales[i * 12 + 48] = (uint8_t) ((s[0] & 63) + ((s[4] & 48) << 2));
        out->scales[i * 12 + 49] = (uint8_t) ((s[1] & 63) + ((s[5] & 48) << 2));
        out->scales[i * 12 + 50] = (uint8_t) ((s[2] & 63) + ((s[6] & 48) << 2));
        out->scales[i * 12 + 51] = (uint8_t) ((s[3] & 63) + ((s[7] & 48) << 2));
        out->scales[i * 12 + 52] = (uint8_t) ((m[0] & 63) + ((m[4] & 48) << 2));
        out->scales[i * 12 + 53] = (uint8_t) ((m[1] & 63) + ((m[5] & 48) << 2));
        out->scales[i * 12 + 54] = (uint8_t) ((m[2] & 63) + ((m[6] & 48) << 2));
        out->scales[i * 12 + 55] = (uint8_t) ((m[3] & 63) + ((m[7] & 48) << 2));
        out->scales[i * 12 + 56] = (uint8_t) ((s[4] & 15) + ((m[4] & 15) << 4));
        out->scales[i * 12 + 57] = (uint8_t) ((s[5] & 15) + ((m[5] & 15) << 4));
        out->scales[i * 12 + 58] = (uint8_t) ((s[6] & 15) + ((m[6] & 15) << 4));
        out->scales[i * 12 + 59] = (uint8_t) ((s[7] & 15) + ((m[7] & 15) << 4));
    }
}

void q4k_to_q4kx8_octet(size_t               n_super,
                        const uint8_t        q4k_rows[static 8 * n_super * Q4_K_BLOCK_BYTES],
                        struct block_q4_Kx8 *q4kx8_out) {
    /* For each super-block index s: gather the s-th super-block from each
     * of the 8 source rows, then call make_block_q4_kx8 to interleave. */
    struct block_q4_K_t tmp[8];
    for (size_t s = 0; s < n_super; s++) {
        for (int r = 0; r < 8; r++) {
            const size_t row_offset = (size_t) r * n_super * Q4_K_BLOCK_BYTES;
            memcpy(&tmp[r], q4k_rows + row_offset + s * Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES);
        }
        make_block_q4_kx8(tmp, &q4kx8_out[s]);
    }
}

void q4k_to_q4kx8_matrix(size_t               n_in,
                         size_t               n_out,
                         const uint8_t       *q4k_data,
                         struct block_q4_Kx8 *q4kx8_out) {
    const size_t n_super        = n_in / Q4_K_BLOCK_ELEMS;
    const size_t row_bytes      = n_super * Q4_K_BLOCK_BYTES;
    const size_t n_octets       = n_out / 8;
    const size_t blocks_per_oct = n_super;
    for (size_t oct = 0; oct < n_octets; oct++) {
        const uint8_t *row_base = q4k_data + oct * 8 * row_bytes;
        q4k_to_q4kx8_octet(n_super, row_base, q4kx8_out + oct * blocks_per_oct);
    }
}
