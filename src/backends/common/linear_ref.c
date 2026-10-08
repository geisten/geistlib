/*
 * src/backends/common/linear_ref.c — see linear_ref.h.
 */
#include "linear_ref.h"

#include "quant.h"

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
    case GEIST_DTYPE_Q5_0:
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
 * blocks: rows are (quant_weight_extent_ok) and REF_TILE is. An undecodable
 * dtype (never resolved here: geist_linear_ref_decodes) reads as zeros. */
static void
decode_tile(const struct geist_weight *w, size_t j, size_t k0, size_t n, float out[static n]) {
    const size_t n_in = (size_t) w->n_in;
    (void) quant_dequant_row(
            (enum geist_dtype) w->dtype, n_in * (size_t) w->n_out, j * n_in + k0, n, w->raw, out);
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
