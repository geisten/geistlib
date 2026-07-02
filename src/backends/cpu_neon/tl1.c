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

/* Symmetric int8 activation quant (round half away from zero). Returns
 * the inverse scale (max_abs / 127) to fold back into y. */
static float tl1_quantize_act_row(const float *xt, size_t n_in, int8_t *qt) {
    float max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        const float a = xt[i] < 0.0f ? -xt[i] : xt[i];
        if (a > max_abs)
            max_abs = a;
    }
    const float act_scale = 127.0f / max_abs;
    for (size_t i = 0; i < n_in; i++) {
        const float q  = xt[i] * act_scale;
        int32_t     qi = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
        if (qi > 127)
            qi = 127;
        if (qi < -128)
            qi = -128;
        qt[i] = (int8_t) qi;
    }
    return max_abs / 127.0f;
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
    int8_t     *xq            = xq_cache;
    const float inv_act_scale = tl1_quantize_act_row(x, n_in, xq);

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

/* Prefill LUT-GEMM (m>1). The decode kernel's amortization — build the
 * activation LUT once, let the weights index it — extended to a batch: build m
 * per-token LUTs, then loop output-row tiles OUTER and reuse each weight tile
 * across all m tokens (the weight is the big DRAM stream; the per-token LUT
 * stays L1/L2-resident). Replaces the SDOT prefill's n_out*m*n_in multiplies
 * with m*n_in*9 LUT-build multiplies + cheap tbl lookups — the amortization
 * bitnet.cpp's i2_s GEMM exploits for ternary prefill. y is [m][n_out]. */
void cpu_neon_w_tl1_mN(const float               *x,
                       const struct geist_weight *w,
                       size_t                     m,
                       struct geist_backend      *be,
                       float                     *y) {
    (void) be;
    const size_t   n_in  = (size_t) w->n_in;
    const size_t   n_out = (size_t) w->n_out;
    const uint8_t *base  = (const uint8_t *) w->aux_fp32;
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (n_in % TL1_BBK != 0 || n_out % TL1_BM != 0)
        return;

    const size_t n_r_tiles           = n_out / TL1_BM;
    const size_t n_k_tiles           = n_in / TL1_BBK;
    const size_t lut_bytes_per_inner = TL1_KPAIRS_PER_INNER * 2 * 16;      /* 128 */
    const size_t lut_bytes_per_kt    = TL1_K_INNERS * lut_bytes_per_inner; /* 2048 */
    const size_t lut_per_token       = n_k_tiles * lut_bytes_per_kt;

    static _Thread_local int8_t *xq_tl    = NULL;
    static _Thread_local size_t  xq_cap   = 0;
    static _Thread_local int8_t *lut_tl   = NULL;
    static _Thread_local size_t  lut_cap  = 0;
    static _Thread_local float  *invs_tl  = NULL;
    static _Thread_local size_t  invs_cap = 0;
    if (xq_cap < m * n_in) {
        safe_free((void **) &xq_tl);
        xq_tl = heap_alloc_array_aligned(int8_t, m *n_in);
        if (!xq_tl) {
            xq_cap = 0;
            memset(y, 0, m * n_out * sizeof *y);
            return;
        }
        xq_cap = m * n_in;
    }
    if (lut_cap < m * lut_per_token) {
        safe_free((void **) &lut_tl);
        lut_tl = heap_alloc_array_aligned(int8_t, m *lut_per_token);
        if (!lut_tl) {
            lut_cap = 0;
            memset(y, 0, m * n_out * sizeof *y);
            return;
        }
        lut_cap = m * lut_per_token;
    }
    if (invs_cap < m) {
        safe_free((void **) &invs_tl);
        invs_tl = heap_alloc_array_aligned(float, m);
        if (!invs_tl) {
            invs_cap = 0;
            memset(y, 0, m * n_out * sizeof *y);
            return;
        }
        invs_cap = m;
    }
    int8_t *const xq          = xq_tl;
    int8_t *const lut_scratch = lut_tl;
    float *const  invs        = invs_tl;

    /* Per-token symmetric int8 activation quant. */
    for (size_t t = 0; t < m; t++) {
        invs[t] = tl1_quantize_act_row(x + t * n_in, n_in, xq + t * n_in);
    }

#if defined(__ARM_NEON)
    /* Build m per-token activation LUTs (parallel over tokens). */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t t = 0; t < m; t++) {
        const int8_t *qt   = xq + t * n_in;
        int8_t       *tlut = lut_scratch + t * lut_per_token;
        for (size_t kt = 0; kt < n_k_tiles; kt++) {
            int8_t *kt_lut = tlut + kt * lut_bytes_per_kt;
            for (size_t c = 0; c < TL1_K_INNERS; c++) {
                int8_t      *c_lut   = kt_lut + c * lut_bytes_per_inner;
                const size_t kp_base = kt * TL1_BBK + c * TL1_K_PER_INNER;
                for (size_t p = 0; p < TL1_KPAIRS_PER_INNER; p++) {
                    int8_t       *p_hi = c_lut + p * 32 + 0;
                    int8_t       *p_lo = c_lut + p * 32 + 16;
                    const int16_t a    = (int16_t) qt[kp_base + p * 2 + 0];
                    const int16_t b    = (int16_t) qt[kp_base + p * 2 + 1];
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
    }

    /* Row-tile loop (parallel). Each weight tile is loaded once per (rt,kt) and
     * reused across all m tokens. */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (size_t rt = 0; rt < n_r_tiles; rt++) {
        float row_sum[GEIST_QUANT_M_CAP * TL1_BM] __attribute__((aligned(16)));
        for (size_t i = 0; i < m * TL1_BM; i++)
            row_sum[i] = 0.0f;

        for (size_t kt = 0; kt < n_k_tiles; kt++) {
            const size_t     tile_off = (rt * n_k_tiles + kt) * TL1_TILE_BYTES;
            const uint8_t   *packed   = base + tile_off;
            const float     *scales   = (const float *) (packed + TL1_PACKED_PER_TILE);
            const uint8x16_t mask     = vdupq_n_u8(0x0F);

            /* Token-tiling (MR tokens). The weight nibbles for inner-block c
             * are loaded + extracted ONCE and reused across MR tokens; only the
             * per-token LUT (2 KB/kt) streams in the inner loop, so the working
             * set is MR × 2 KB — L1-resident — instead of m × 2 KB (overflow at
             * m=32). This is the register-blocking bitnet.cpp's i2_s GEMM does:
             * amortize the weight load over the token tile. */
#define TL1MN_MR 4
            for (size_t tt = 0; tt < m; tt += TL1MN_MR) {
                const size_t mr = (m - tt < TL1MN_MR) ? (m - tt) : TL1MN_MR;
                int16x8_t    vc0[TL1MN_MR], vc1[TL1MN_MR], vc2[TL1MN_MR], vc3[TL1MN_MR];
                for (size_t k = 0; k < mr; k++) {
                    vc0[k] = vdupq_n_s16(0);
                    vc1[k] = vdupq_n_s16(0);
                    vc2[k] = vdupq_n_s16(0);
                    vc3[k] = vdupq_n_s16(0);
                }
                for (size_t c = 0; c < TL1_K_INNERS; c++) {
                    const uint8_t *c_base = packed + c * 64;
                    /* Shared weight nibbles for the 4 packets — loaded once. */
                    const uint8x16_t p0  = vld1q_u8(c_base + 0);
                    const uint8x16_t p1  = vld1q_u8(c_base + 16);
                    const uint8x16_t p2  = vld1q_u8(c_base + 32);
                    const uint8x16_t p3  = vld1q_u8(c_base + 48);
                    const uint8x16_t hn0 = vshrq_n_u8(p0, 4), ln0 = vandq_u8(p0, mask);
                    const uint8x16_t hn1 = vshrq_n_u8(p1, 4), ln1 = vandq_u8(p1, mask);
                    const uint8x16_t hn2 = vshrq_n_u8(p2, 4), ln2 = vandq_u8(p2, mask);
                    const uint8x16_t hn3 = vshrq_n_u8(p3, 4), ln3 = vandq_u8(p3, mask);
                    for (size_t k = 0; k < mr; k++) {
                        const int8_t   *c_lut = lut_scratch + (tt + k) * lut_per_token +
                                                kt * lut_bytes_per_kt + c * lut_bytes_per_inner;
                        const int8x16_t h0 = vld1q_s8(c_lut + 0), l0 = vld1q_s8(c_lut + 16);
                        const int8x16_t h1 = vld1q_s8(c_lut + 32), l1 = vld1q_s8(c_lut + 48);
                        const int8x16_t h2 = vld1q_s8(c_lut + 64), l2 = vld1q_s8(c_lut + 80);
                        const int8x16_t h3 = vld1q_s8(c_lut + 96), l3 = vld1q_s8(c_lut + 112);
                        /* tbl with the SHARED weight nibbles + this token's LUT. */
#define TL1MN_ACC(HN, LN, A_HI, A_LO, B_HI, B_LO, VC_LO, VC_HI)                       \
    do {                                                                              \
        const int8x16_t   _aH = vqtbl1q_s8((A_HI), (HN));                             \
        const int8x16_t   _aL = vqtbl1q_s8((A_LO), (HN));                             \
        const int8x16_t   _bH = vqtbl1q_s8((B_HI), (LN));                             \
        const int8x16_t   _bL = vqtbl1q_s8((B_LO), (LN));                             \
        const int8x16x2_t _vA = vzipq_s8(_aL, _aH);                                   \
        const int8x16x2_t _vB = vzipq_s8(_bL, _bH);                                   \
        (VC_LO)               = vaddq_s16((VC_LO), vreinterpretq_s16_s8(_vA.val[0])); \
        (VC_LO)               = vaddq_s16((VC_LO), vreinterpretq_s16_s8(_vB.val[0])); \
        (VC_HI)               = vaddq_s16((VC_HI), vreinterpretq_s16_s8(_vA.val[1])); \
        (VC_HI)               = vaddq_s16((VC_HI), vreinterpretq_s16_s8(_vB.val[1])); \
    } while (0)
                        TL1MN_ACC(hn0, ln0, h0, l0, h1, l1, vc0[k], vc1[k]);
                        TL1MN_ACC(hn1, ln1, h2, l2, h3, l3, vc0[k], vc1[k]);
                        TL1MN_ACC(hn2, ln2, h0, l0, h1, l1, vc2[k], vc3[k]);
                        TL1MN_ACC(hn3, ln3, h2, l2, h3, l3, vc2[k], vc3[k]);
#undef TL1MN_ACC
                    }
                }
                for (size_t k = 0; k < mr; k++) {
                    float *rs = row_sum + (tt + k) * TL1_BM;
#define TL1MN_FOLD(VC, ROW_BASE)                                            \
    do {                                                                    \
        const int32x4_t   _lo  = vmovl_s16(vget_low_s16((VC)));             \
        const int32x4_t   _hi  = vmovl_high_s16((VC));                      \
        const float32x4_t _slo = vld1q_f32(scales + (ROW_BASE) + 0);        \
        const float32x4_t _shi = vld1q_f32(scales + (ROW_BASE) + 4);        \
        float32x4_t       _rlo = vld1q_f32(rs + (ROW_BASE) + 0);            \
        float32x4_t       _rhi = vld1q_f32(rs + (ROW_BASE) + 4);            \
        _rlo                   = vfmaq_f32(_rlo, vcvtq_f32_s32(_lo), _slo); \
        _rhi                   = vfmaq_f32(_rhi, vcvtq_f32_s32(_hi), _shi); \
        vst1q_f32(rs + (ROW_BASE) + 0, _rlo);                               \
        vst1q_f32(rs + (ROW_BASE) + 4, _rhi);                               \
    } while (0)
                    TL1MN_FOLD(vc0[k], 0);
                    TL1MN_FOLD(vc1[k], 8);
                    TL1MN_FOLD(vc2[k], 16);
                    TL1MN_FOLD(vc3[k], 24);
#undef TL1MN_FOLD
                }
            }
#undef TL1MN_MR
        }
        for (size_t t = 0; t < m; t++) {
            const float32x4_t s  = vdupq_n_f32(invs[t]);
            float            *rs = row_sum + t * TL1_BM;
            for (size_t r = 0; r < TL1_BM; r += 4) {
                float32x4_t v = vld1q_f32(rs + r);
                vst1q_f32(y + t * n_out + rt * TL1_BM + r, vmulq_f32(v, s));
            }
        }
    }
#else
    (void) base;
    (void) n_r_tiles;
    (void) n_k_tiles;
    (void) lut_per_token;
    (void) lut_scratch;
    (void) invs;
    (void) xq;
    /* Scalar fallback (non-NEON builds): the batched LUT-GEMM above has no
     * scalar twin, so compute each row via the m1 kernel, which carries its
     * own scalar reference path. Correct, if slower; the SIMD prefill path is
     * NEON-only and this branch is not used on the Pi 5 / ARM targets. */
    for (size_t t = 0; t < m; t++) {
        cpu_neon_w_tl1_m1(x + t * n_in, w, be, y + t * n_out);
    }
#endif /* __ARM_NEON */
}
