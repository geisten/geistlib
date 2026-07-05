/*
 * src/backends/cpu_neon/tl1.c — W1.58 × A8 LUT-GEMV decode kernel.
 *
 * Layer: BACKEND (cpu_neon). See tl1.h for the algorithmic overview.
 *
 * This file ships the SCALAR REFERENCE (no NEON intrinsics) so the
 * layout, encoding, and accumulation can be unit-tested independently
 * of the SIMD lowering. A subsequent commit replaces the inner loops
 * with vqtbl1q_s8 + int16x8 accumulators.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "tl1.h"
#include "heap.h"

#include "internal.h"
#include "quant.h"

#include <geist_weight.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* Nibble → ternary pair (w0, w1). Encoder writes value (w0+1)*3 + (w1+1)
 * into bits 0..3 of a packed byte (low nibble first, high nibble after).
 * Values 9..15 are unused; the decoder must treat them as (0, 0). */
static const int8_t NIBBLE_W0[16] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
static const int8_t NIBBLE_W1[16] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0};

static inline uint8_t tl1_encode_pair(int8_t w0, int8_t w1) {
    return (uint8_t) ((w0 + 1) * 3 + (w1 + 1));
}

/* Unpack one TQ2_0 256-element block into a contiguous int8 trit array.
 * Mirrors the loop in gguf_quant.c::dequant_tq2_0_row (without the fp32
 * scale multiply): trit at output index `elem_base + 32*l + m` lives in
 * `qs[j_byte + m]` at bit-shift `l*2`. */
static void tl1_tq2_0_unpack_block(const uint8_t *qs, int8_t trits[256]) {
    for (int j = 0; j < 2; j++) {
        const int j_byte    = j * 32;
        const int elem_base = j * 128;
        for (int l = 0; l < 4; l++) {
            const int shift = l * 2;
            const int off   = elem_base + 32 * l;
            for (int m = 0; m < 32; m++) {
                const int trit = (int) ((qs[j_byte + m] >> shift) & 3) - 1;
                trits[off + m] = (int8_t) trit;
            }
        }
    }
}

/* Per-tile storage layout (row-interleaved for vqtbl1q_s8 access):
 *   [0 .. TL1_PACKED_PER_TILE)  : packed nibble bytes
 *   [TL1_PACKED_PER_TILE .. +TL1_SCALES_PER_TILE) : 32 fp32 per-row scales
 *
 * Packed bytes are 16 k_inner blocks of 64 bytes each (16 * 64 = 1024):
 *   k_inner iter c ∈ 0..16 covers 8 K-positions (= 4 K-pairs A,B,C,D):
 *     bytes [0..16) : rows 0..15, K-pairs (A, B)
 *     bytes [16..32): rows 0..15, K-pairs (C, D)
 *     bytes [32..48): rows 16..31, K-pairs (A, B)
 *     bytes [48..64): rows 16..31, K-pairs (C, D)
 *   Byte b in [0..16): row (group_base + b), nibbles
 *     bits 0..3 (low ):  K-pair B (or D) = K-pos (kbase+2, kbase+3)
 *     bits 4..7 (high):  K-pair A (or C) = K-pos (kbase+0, kbase+1)
 *   Where kbase = c*8 (= start of k_inner) and the second offset adds 4. */
#define TL1_PACKED_PER_TILE ((size_t) (TL1_BM * TL1_BBK / 4))   /* 1024 */
#define TL1_SCALES_PER_TILE ((size_t) (TL1_BM * sizeof(float))) /* 128 */
#define TL1_TILE_BYTES (TL1_PACKED_PER_TILE + TL1_SCALES_PER_TILE)
#define TL1_K_INNERS 16        /* K-windows per BBK tile (BBK / 8) */
#define TL1_K_PER_INNER 8      /* K-positions per k_inner iter */
#define TL1_KPAIRS_PER_INNER 4 /* = TL1_K_PER_INNER / 2 */

size_t tl1_pack_size_bytes(size_t n_in, size_t n_out) {
    if (n_in == 0 || n_out == 0)
        return 0;
    if (n_in % TL1_BBK != 0)
        return 0;
    if (n_out % TL1_BM != 0)
        return 0;
    if (n_in % 256 != 0)
        return 0;
    const size_t n_r_tiles = n_out / TL1_BM;
    const size_t n_k_tiles = n_in / TL1_BBK;
    return n_r_tiles * n_k_tiles * TL1_TILE_BYTES;
}

int tl1_pack_from_tq2_0(const void *tq2_0_rows, size_t n_in, size_t n_out, void *out) {
    if (tl1_pack_size_bytes(n_in, n_out) == 0)
        return -1;
    const uint8_t *W              = (const uint8_t *) tq2_0_rows;
    const size_t   blocks_per_row = n_in / 256;
    const size_t   row_bytes_tq2  = blocks_per_row * 66;
    const size_t   n_r_tiles      = n_out / TL1_BM;
    const size_t   n_k_tiles      = n_in / TL1_BBK;
    uint8_t       *out_bytes      = (uint8_t *) out;

    /* Unpack each row's trits once per (rt, kt) and then re-emit into
     * the row-interleaved layout. trits_row[TL1_BBK] holds the BBK
     * trits for one (row, kt) tile. */
    int8_t trits_block[256];

    for (size_t rt = 0; rt < n_r_tiles; rt++) {
        for (size_t kt = 0; kt < n_k_tiles; kt++) {
            const size_t tile_off = (rt * n_k_tiles + kt) * TL1_TILE_BYTES;
            uint8_t     *packed   = out_bytes + tile_off;
            float       *scales   = (float *) (packed + TL1_PACKED_PER_TILE);

            /* Per-row scratch: 32 rows × 128 trits within this K-tile. */
            int8_t row_trits[TL1_BM][TL1_BBK];

            for (size_t r_local = 0; r_local < TL1_BM; r_local++) {
                const size_t   row_global = rt * TL1_BM + r_local;
                const uint8_t *row_ptr    = W + row_global * row_bytes_tq2;

                const size_t k_start        = kt * TL1_BBK;
                const size_t block_idx      = k_start / 256;
                const size_t k_within_block = k_start % 256;

                const uint8_t *blk = row_ptr + block_idx * 66;
                tl1_tq2_0_unpack_block(blk, trits_block);

                const uint16_t d_bits = (uint16_t) blk[64] | ((uint16_t) blk[65] << 8);
                scales[r_local]       = fp16_to_fp32(d_bits);

                for (size_t i = 0; i < TL1_BBK; i++) {
                    row_trits[r_local][i] = trits_block[k_within_block + i];
                }
            }

            /* Emit in row-interleaved layout. */
            for (size_t c = 0; c < TL1_K_INNERS; c++) {
                const size_t kbase  = c * TL1_K_PER_INNER;
                uint8_t     *c_base = packed + c * 64;
                for (size_t g = 0; g < 2; g++) {
                    const size_t group_row_base = g * 16;
                    for (size_t kw = 0; kw < 2; kw++) {
                        const size_t k_off = kw * 4;
                        uint8_t     *pkt   = c_base + g * 32 + kw * 16;
                        for (size_t b = 0; b < 16; b++) {
                            const size_t  row  = group_row_base + b;
                            const uint8_t high = tl1_encode_pair(row_trits[row][kbase + k_off + 0],
                                                                 row_trits[row][kbase + k_off + 1]);
                            const uint8_t low  = tl1_encode_pair(row_trits[row][kbase + k_off + 2],
                                                                 row_trits[row][kbase + k_off + 3]);
                            pkt[b]             = (uint8_t) ((high << 4) | (low & 0xF));
                        }
                    }
                }
            }
        }
    }
    return 0;
}

void cpu_neon_w_tl1_m1(const float               *x,
                       const struct geist_weight *w,
                       struct geist_backend      *be,
                       float                     *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    /* TL1 packed bytes live in aux_fp32 (the M>1 prefill kernel still
     * reads TQ2_0 from raw, so raw must stay intact). */
    const uint8_t *base = (const uint8_t *) w->aux_fp32;

    static _Thread_local int8_t *xq_cache = NULL;
    static _Thread_local size_t  xq_cap   = 0;
    if (xq_cap < n_in) {
        safe_free((void **) &xq_cache);
        xq_cache = heap_alloc_array_aligned(int8_t, n_in);
        if (xq_cache == NULL) {
            xq_cap = 0;
            memset(y, 0, n_out * sizeof *y);
            return;
        }
        xq_cap = n_in;
    }
    int8_t *xq      = xq_cache;
    float   max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > max_abs)
            max_abs = a;
    }
    const float act_scale     = 127.0f / max_abs;
    const float inv_act_scale = max_abs / 127.0f;
    for (size_t i = 0; i < n_in; i++) {
        const float q  = x[i] * act_scale;
        int32_t     qi = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
        if (qi > 127)
            qi = 127;
        if (qi < -128)
            qi = -128;
        xq[i] = (int8_t) qi;
    }

    const size_t n_r_tiles = n_out / TL1_BM;
    const size_t n_k_tiles = n_in / TL1_BBK;

#if defined(__ARM_NEON)
    /* Precompute activation LUTs once per matmul (depends only on x).
     * Layout: lut_scratch[kt][c][p][hi/lo][16]. Total bytes per kt =
     * 16 c * 4 p * 2 (hi/lo) * 16 = 2 KB; fits L1 with margin. */
    const size_t                 lut_bytes_per_inner = TL1_KPAIRS_PER_INNER * 2 * 16;   /* 128 */
    const size_t                 lut_bytes_per_kt = TL1_K_INNERS * lut_bytes_per_inner; /* 2048 */
    const size_t                 lut_total        = n_k_tiles * lut_bytes_per_kt;
    static _Thread_local int8_t *lut_scratch_tl   = NULL;
    static _Thread_local size_t  lut_cap          = 0;
    if (lut_cap < lut_total) {
        safe_free((void **) &lut_scratch_tl);
        lut_scratch_tl = heap_alloc_array_aligned(int8_t, lut_total);
        if (lut_scratch_tl == NULL) {
            lut_cap = 0;
            memset(y, 0, n_out * sizeof *y);
            return;
        }
        lut_cap = lut_total;
    }
    /* Snapshot the thread-local pointer to a plain local so the OpenMP
     * parallel region below shares it across worker threads (workers
     * have their own _Thread_local storage initialized to NULL). */
    int8_t *const lut_scratch = lut_scratch_tl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t kt = 0; kt < n_k_tiles; kt++) {
        int8_t *kt_lut = lut_scratch + kt * lut_bytes_per_kt;
        for (size_t c = 0; c < TL1_K_INNERS; c++) {
            int8_t      *c_lut   = kt_lut + c * lut_bytes_per_inner;
            const size_t kp_base = kt * TL1_BBK + c * TL1_K_PER_INNER;
            for (size_t p = 0; p < TL1_KPAIRS_PER_INNER; p++) {
                int8_t       *p_hi = c_lut + p * 32 + 0;
                int8_t       *p_lo = c_lut + p * 32 + 16;
                const int16_t a    = (int16_t) xq[kp_base + p * 2 + 0];
                const int16_t b    = (int16_t) xq[kp_base + p * 2 + 1];
                for (size_t n = 0; n < 9; n++) {
                    const int16_t v =
                            (int16_t) ((int16_t) NIBBLE_W0[n] * a + (int16_t) NIBBLE_W1[n] * b);
                    p_hi[n] = (int8_t) (v >> 8);
                    p_lo[n] = (int8_t) (v & 0xFF);
                }
                for (size_t n = 9; n < 16; n++) {
                    p_hi[n] = 0;
                    p_lo[n] = 0;
                }
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t rt = 0; rt < n_r_tiles; rt++) {
        float row_sum_fp32[TL1_BM] __attribute__((aligned(16)));
        for (size_t r = 0; r < TL1_BM; r++)
            row_sum_fp32[r] = 0.0f;

        for (size_t kt = 0; kt < n_k_tiles; kt++) {
            int16x8_t vec_c0 = vdupq_n_s16(0);
            int16x8_t vec_c1 = vdupq_n_s16(0);
            int16x8_t vec_c2 = vdupq_n_s16(0);
            int16x8_t vec_c3 = vdupq_n_s16(0);

            const size_t   tile_off = (rt * n_k_tiles + kt) * TL1_TILE_BYTES;
            const uint8_t *packed   = base + tile_off;
            const float   *scales   = (const float *) (packed + TL1_PACKED_PER_TILE);
            const int8_t  *kt_lut   = lut_scratch + kt * lut_bytes_per_kt;

            for (size_t c = 0; c < TL1_K_INNERS; c++) {
                const int8_t   *c_lut     = kt_lut + c * lut_bytes_per_inner;
                const int8x16_t lut_hi_q0 = vld1q_s8(c_lut + 0);
                const int8x16_t lut_lo_q0 = vld1q_s8(c_lut + 16);
                const int8x16_t lut_hi_q1 = vld1q_s8(c_lut + 32);
                const int8x16_t lut_lo_q1 = vld1q_s8(c_lut + 48);
                const int8x16_t lut_hi_q2 = vld1q_s8(c_lut + 64);
                const int8x16_t lut_lo_q2 = vld1q_s8(c_lut + 80);
                const int8x16_t lut_hi_q3 = vld1q_s8(c_lut + 96);
                const int8x16_t lut_lo_q3 = vld1q_s8(c_lut + 112);

                const uint8_t   *c_base = packed + c * 64;
                const uint8x16_t mask   = vdupq_n_u8(0x0F);

#define TL1_ACC_PACKET(PKT_OFF, KP_A_HI, KP_A_LO, KP_B_HI, KP_B_LO, VC_LO, VC_HI)      \
    do {                                                                               \
        const uint8x16_t  _pkt = vld1q_u8((PKT_OFF));                                  \
        const uint8x16_t  _hn  = vshrq_n_u8(_pkt, 4);                                  \
        const uint8x16_t  _ln  = vandq_u8(_pkt, mask);                                 \
        const int8x16_t   _aH  = vqtbl1q_s8((KP_A_HI), _hn);                           \
        const int8x16_t   _aL  = vqtbl1q_s8((KP_A_LO), _hn);                           \
        const int8x16_t   _bH  = vqtbl1q_s8((KP_B_HI), _ln);                           \
        const int8x16_t   _bL  = vqtbl1q_s8((KP_B_LO), _ln);                           \
        const int8x16x2_t _vA  = vzipq_s8(_aL, _aH);                                   \
        const int8x16x2_t _vB  = vzipq_s8(_bL, _bH);                                   \
        (VC_LO)                = vaddq_s16((VC_LO), vreinterpretq_s16_s8(_vA.val[0])); \
        (VC_LO)                = vaddq_s16((VC_LO), vreinterpretq_s16_s8(_vB.val[0])); \
        (VC_HI)                = vaddq_s16((VC_HI), vreinterpretq_s16_s8(_vA.val[1])); \
        (VC_HI)                = vaddq_s16((VC_HI), vreinterpretq_s16_s8(_vB.val[1])); \
    } while (0)

                /* row_group 0, k_window 0: K-pair A (idx 0), B (idx 1) */
                TL1_ACC_PACKET(
                        c_base + 0, lut_hi_q0, lut_lo_q0, lut_hi_q1, lut_lo_q1, vec_c0, vec_c1);
                /* row_group 0, k_window 1: K-pair C (idx 2), D (idx 3) */
                TL1_ACC_PACKET(
                        c_base + 16, lut_hi_q2, lut_lo_q2, lut_hi_q3, lut_lo_q3, vec_c0, vec_c1);
                /* row_group 1, k_window 0 */
                TL1_ACC_PACKET(
                        c_base + 32, lut_hi_q0, lut_lo_q0, lut_hi_q1, lut_lo_q1, vec_c2, vec_c3);
                /* row_group 1, k_window 1 */
                TL1_ACC_PACKET(
                        c_base + 48, lut_hi_q2, lut_lo_q2, lut_hi_q3, lut_lo_q3, vec_c2, vec_c3);
#undef TL1_ACC_PACKET
            }

            /* Widen int16x8 accumulators to int32, fold per-row scale,
             * accumulate into fp32 row sums. */
#define TL1_FOLD(VC, ROW_BASE)                                             \
    do {                                                                   \
        const int32x4_t   _lo  = vmovl_s16(vget_low_s16((VC)));            \
        const int32x4_t   _hi  = vmovl_high_s16((VC));                     \
        const float32x4_t _flo = vcvtq_f32_s32(_lo);                       \
        const float32x4_t _fhi = vcvtq_f32_s32(_hi);                       \
        const float32x4_t _slo = vld1q_f32(scales + (ROW_BASE) + 0);       \
        const float32x4_t _shi = vld1q_f32(scales + (ROW_BASE) + 4);       \
        float32x4_t       _rlo = vld1q_f32(row_sum_fp32 + (ROW_BASE) + 0); \
        float32x4_t       _rhi = vld1q_f32(row_sum_fp32 + (ROW_BASE) + 4); \
        _rlo                   = vfmaq_f32(_rlo, _flo, _slo);              \
        _rhi                   = vfmaq_f32(_rhi, _fhi, _shi);              \
        vst1q_f32(row_sum_fp32 + (ROW_BASE) + 0, _rlo);                    \
        vst1q_f32(row_sum_fp32 + (ROW_BASE) + 4, _rhi);                    \
    } while (0)
            TL1_FOLD(vec_c0, 0);
            TL1_FOLD(vec_c1, 8);
            TL1_FOLD(vec_c2, 16);
            TL1_FOLD(vec_c3, 24);
#undef TL1_FOLD
        }
        const float32x4_t s = vdupq_n_f32(inv_act_scale);
        for (size_t r = 0; r < TL1_BM; r += 4) {
            float32x4_t v = vld1q_f32(row_sum_fp32 + r);
            v             = vmulq_f32(v, s);
            vst1q_f32(y + rt * TL1_BM + r, v);
        }
    }
#else  /* !__ARM_NEON: scalar reference */
    int16_t lut[TL1_KPAIRS_PER_INNER][16];
    for (size_t rt = 0; rt < n_r_tiles; rt++) {
        int32_t tile_acc[TL1_BM];
        float   row_sum_fp32[TL1_BM];
        for (size_t r = 0; r < TL1_BM; r++)
            row_sum_fp32[r] = 0.0f;
        for (size_t kt = 0; kt < n_k_tiles; kt++) {
            for (size_t r = 0; r < TL1_BM; r++)
                tile_acc[r] = 0;
            const size_t   tile_off = (rt * n_k_tiles + kt) * TL1_TILE_BYTES;
            const uint8_t *packed   = base + tile_off;
            const float   *scales   = (const float *) (packed + TL1_PACKED_PER_TILE);
            for (size_t c = 0; c < TL1_K_INNERS; c++) {
                const size_t kp_base = kt * TL1_BBK + c * TL1_K_PER_INNER;
                for (size_t p = 0; p < TL1_KPAIRS_PER_INNER; p++) {
                    const int16_t a = (int16_t) xq[kp_base + p * 2 + 0];
                    const int16_t b = (int16_t) xq[kp_base + p * 2 + 1];
                    for (size_t n = 0; n < 9; n++) {
                        lut[p][n] =
                                (int16_t) ((int16_t) NIBBLE_W0[n] * a + (int16_t) NIBBLE_W1[n] * b);
                    }
                    for (size_t n = 9; n < 16; n++)
                        lut[p][n] = 0;
                }
                const uint8_t *c_base = packed + c * 64;
                for (size_t g = 0; g < 2; g++) {
                    for (size_t kw = 0; kw < 2; kw++) {
                        const uint8_t *pkt = c_base + g * 32 + kw * 16;
                        for (size_t b = 0; b < 16; b++) {
                            const uint8_t byte = pkt[b];
                            tile_acc[g * 16 + b] += (int32_t) lut[kw * 2 + 0][(byte >> 4) & 0xF] +
                                                    (int32_t) lut[kw * 2 + 1][byte & 0xF];
                        }
                    }
                }
            }
            for (size_t r = 0; r < TL1_BM; r++) {
                row_sum_fp32[r] += (float) tile_acc[r] * scales[r];
            }
        }
        for (size_t r = 0; r < TL1_BM; r++) {
            y[rt * TL1_BM + r] = row_sum_fp32[r] * inv_act_scale;
        }
    }
#endif /* __ARM_NEON */
}
