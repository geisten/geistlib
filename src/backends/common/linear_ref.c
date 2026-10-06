/*
 * src/backends/common/linear_ref.c — see linear_ref.h.
 */
#include "linear_ref.h"

#include "quant.h"

#include <string.h>

/* Elements decoded per step: a multiple of every block size (32, 128 and
 * 256 elements), so a tile is always whole blocks. 4 KB of stack; smaller
 * tiles measured slower than a whole-row decode, 1024 is within noise. */
constexpr size_t REF_TILE = 1024;

/* Rows of x dotted against one decoded tile, each with a double
 * accumulator on the stack; m beyond this decodes the weight again per
 * group of rows. */
constexpr size_t REF_ROWS = 64;

bool geist_linear_ref_decodes(uint16_t dtype) {
    switch ((enum geist_dtype) dtype) {
    case GEIST_DTYPE_F32:
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
    case GEIST_DTYPE_Q4_0:
    case GEIST_DTYPE_Q4_1:
    case GEIST_DTYPE_Q8_0:
    case GEIST_DTYPE_Q3_K:
    case GEIST_DTYPE_Q4_K:
    case GEIST_DTYPE_Q5_K:
    case GEIST_DTYPE_Q6_K:
    case GEIST_DTYPE_IQ2_S:
    case GEIST_DTYPE_IQ3_S:
    case GEIST_DTYPE_IQ4_NL:
    case GEIST_DTYPE_IQ4_XS:
    case GEIST_DTYPE_TQ2_0:
    case GEIST_DTYPE_PQ2_0:
    case GEIST_DTYPE_I2_S:
        return true;
    default:
        return false;
    }
}

/* Elements [k0, k0 + n) of row j, decoded into out. k0 and n are whole
 * blocks: rows are (quant_weight_extent_ok) and REF_TILE is. `e0` below is
 * the element's index in the whole tensor, so e0 / block_elems is its
 * block. */
static void
decode_tile(const struct geist_weight *w, size_t j, size_t k0, size_t n, float out[static n]) {
    const uint8_t *raw  = (const uint8_t *) w->raw;
    const size_t   n_in = (size_t) w->n_in;
    const size_t   e0   = j * n_in + k0;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_F32:
        memcpy(out, raw + e0 * sizeof(float), n * sizeof(float));
        return;
    case GEIST_DTYPE_F16:
        for (size_t i = 0; i < n; i++) {
            const uint8_t *h = raw + (e0 + i) * 2;
            out[i]           = fp16_to_fp32((uint16_t) ((uint16_t) h[0] | ((uint16_t) h[1] << 8)));
        }
        return;
    case GEIST_DTYPE_BF16:
        for (size_t i = 0; i < n; i++) {
            const uint8_t *h = raw + (e0 + i) * 2;
            const uint32_t b = (uint32_t) ((uint16_t) h[0] | ((uint16_t) h[1] << 8)) << 16;
            memcpy(&out[i], &b, sizeof b);
        }
        return;
    case GEIST_DTYPE_Q4_0:
        dequant_q4_0_row(n, raw + e0 / Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_Q4_1:
        dequant_q4_1_row(n, raw + e0 / Q4_1_BLOCK_ELEMS * Q4_1_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_Q8_0:
        dequant_q8_0_row(n, raw + e0 / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_Q3_K:
        dequant_q3_K_row(n, raw + e0 / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_Q4_K:
        dequant_q4_K_row(n, raw + e0 / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_Q5_K:
        dequant_q5_K_row(n, raw + e0 / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_Q6_K:
        dequant_q6_K_row(n, raw + e0 / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_IQ2_S:
        dequant_iq2_s_row(n, raw + e0 / IQ2_S_BLOCK_ELEMS * IQ2_S_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_IQ3_S:
        dequant_iq3_s_row(n, raw + e0 / IQ3_S_BLOCK_ELEMS * IQ3_S_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_IQ4_NL:
        dequant_iq4_nl_row(n, raw + e0 / IQ4_NL_BLOCK_ELEMS * IQ4_NL_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_IQ4_XS:
        dequant_iq4_xs_row(n, raw + e0 / IQ4_XS_BLOCK_ELEMS * IQ4_XS_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_TQ2_0:
        dequant_tq2_0_row(n, raw + e0 / TQ2_0_BLOCK_ELEMS * TQ2_0_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_PQ2_0:
        dequant_pq2_0_row(n, raw + e0 / PQ2_0_BLOCK_ELEMS * PQ2_0_BLOCK_BYTES, out);
        return;
    case GEIST_DTYPE_I2_S: {
        /* BitNet b1.58 official: 256-elem/64-byte ternary blocks, four 2-bit
         * fields per byte in REVERSE order (element 32*g+bb at shift 6-2g),
         * ONE f32 per-TENSOR scale at the tail (offset n_in*n_out/4). */
        float scale;
        memcpy(&scale, raw + i2_s_scale_offset(n_in * (size_t) w->n_out), sizeof scale);
        const uint8_t *row = raw + j * (n_in / 4);
        for (size_t b = k0 / 256; b < (k0 + n) / 256; b++) {
            const uint8_t *qs = row + b * 64;
            float         *o  = out + (b * 256 - k0);
            for (size_t h = 0; h < 2; h++) {
                for (size_t bb = 0; bb < 32; bb++) {
                    const uint8_t byte = qs[h * 32 + bb];
                    for (size_t g = 0; g < 4; g++) {
                        const int trit           = (int) ((byte >> (6 - 2 * g)) & 3) - 1;
                        o[h * 128 + g * 32 + bb] = (float) trit * scale;
                    }
                }
            }
        }
        return;
    }
    default:
        memset(out, 0, n * sizeof *out);
        return;
    }
}

void geist_linear_ref_rows(size_t                     m,
                           size_t                     j0,
                           size_t                     nj,
                           size_t                     ldy,
                           const float               *x,
                           const struct geist_weight *w,
                           float                     *y) {
    const size_t      n_in = (size_t) w->n_in;
    alignas(64) float tile[REF_TILE];
    if (m == 1) {
        /* Decode: the accumulator stays in a register (acc[] below is ~2 %
         * slower here). */
        for (size_t j = 0; j < nj; j++) {
            double a = 0.0;
            for (size_t k0 = 0; k0 < n_in; k0 += REF_TILE) {
                const size_t n = n_in - k0 < REF_TILE ? n_in - k0 : REF_TILE;
                decode_tile(w, j0 + j, k0, n, tile);
                for (size_t k = 0; k < n; k++) {
                    a += (double) x[k0 + k] * (double) tile[k];
                }
            }
            y[j] = (float) a;
        }
        return;
    }
    double acc[REF_ROWS];
    for (size_t i0 = 0; i0 < m; i0 += REF_ROWS) {
        const size_t rows = m - i0 < REF_ROWS ? m - i0 : REF_ROWS;
        for (size_t j = 0; j < nj; j++) {
            for (size_t i = 0; i < rows; i++) {
                acc[i] = 0.0;
            }
            for (size_t k0 = 0; k0 < n_in; k0 += REF_TILE) {
                const size_t n = n_in - k0 < REF_TILE ? n_in - k0 : REF_TILE;
                decode_tile(w, j0 + j, k0, n, tile);
                for (size_t i = 0; i < rows; i++) {
                    const float *xi = x + (i0 + i) * n_in + k0;
                    double       a  = acc[i];
                    for (size_t k = 0; k < n; k++) {
                        a += (double) xi[k] * (double) tile[k];
                    }
                    acc[i] = a;
                }
            }
            for (size_t i = 0; i < rows; i++) {
                y[(i0 + i) * ldy + j] = (float) acc[i];
            }
        }
    }
}

void geist_linear_ref(size_t m, const float *x, const struct geist_weight *w, float *y) {
    geist_linear_ref_rows(m, 0, (size_t) w->n_out, (size_t) w->n_out, x, w, y);
}
