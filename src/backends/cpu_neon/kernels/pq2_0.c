/*
 * src/backends/cpu_neon/kernels/pq2_0.c — PQ2_0 W2 x A8 decode GEMV.
 *
 * Layer: BACKEND (cpu_neon). PQ2_0 is PrismML's ternary format for
 * Ternary-Bonsai: 128-element blocks of [fp16 d][32 bytes of 2-bit
 * codes], element j at byte j/4, bits 2*(j%4), value (code - 1) * d.
 * The reference decoder is dequant_pq2_0_row (src/formats/gguf/pq2_0.c).
 *
 * Same recipe as the TQ2_0 q8a kernel: one per-call absmax int8 quant of
 * x, raw codes into vdotq_s32, the -1 bias folded out once per block via
 * the block's activation sum. The one difference is the packing. TQ2_0
 * interleaves its trits so a shifted 16-byte load lines up with 16
 * contiguous activations; PQ2_0 packs four consecutive elements per
 * byte, so shift level l of byte m is element 4m + l. Rather than
 * repacking the weights (7 GB for the 27B), the int8 activation is
 * written in that order once per call:
 *
 *     xq[c*64 + 16*l + m] = q(x[c*64 + 4*m + l])     c = 64-element chunk
 *
 * after which each shift level of a 16-byte weight load is a plain
 * vdotq against 16 contiguous activation bytes.
 *
 * Entry point (declared in internal.h, referenced by the resolver table):
 *   cpu_neon_w_pq2_0_q8a_m1 — M=1 decode, int8 SDOT (dotprod hosts)
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../internal.h"
#include "../parallel.h"
#include "checked.h"
#include "heap.h"

#include "quant.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Σ (code_i) * xq_i over one 128-element block, codes still biased by +1. */
static inline int32_t pq2_0_block_dot_raw(const uint8_t *qs, const int8_t *xb) {
    const uint8x16_t three = vdupq_n_u8(3);
    const uint8x16_t p0    = vld1q_u8(qs);
    const uint8x16_t p1    = vld1q_u8(qs + 16);
    int32x4_t        acc0  = vdupq_n_s32(0);
    int32x4_t        acc1  = vdupq_n_s32(0);
    acc0 = vdotq_s32(acc0, vreinterpretq_s8_u8(vandq_u8(p0, three)), vld1q_s8(xb + 0));
    acc1 = vdotq_s32(acc1, vreinterpretq_s8_u8(vandq_u8(p1, three)), vld1q_s8(xb + 64));
    acc0 = vdotq_s32(
            acc0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p0, 2), three)), vld1q_s8(xb + 16));
    acc1 = vdotq_s32(
            acc1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p1, 2), three)), vld1q_s8(xb + 80));
    acc0 = vdotq_s32(
            acc0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p0, 4), three)), vld1q_s8(xb + 32));
    acc1 = vdotq_s32(
            acc1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p1, 4), three)), vld1q_s8(xb + 96));
    acc0 = vdotq_s32(acc0, vreinterpretq_s8_u8(vshrq_n_u8(p0, 6)), vld1q_s8(xb + 48));
    acc1 = vdotq_s32(acc1, vreinterpretq_s8_u8(vshrq_n_u8(p1, 6)), vld1q_s8(xb + 112));
    return vaddvq_s32(vaddq_s32(acc0, acc1));
}

struct pq2_0_m1_ctx {
    const uint8_t *W;
    const int8_t  *xq;
    const int32_t *bsum;
    float         *y;
    float          inv_act_scale;
    size_t         row_bytes;
    size_t         blocks_per_row;
};

static void pq2_0_m1_row_body(size_t r, void *vctx) {
    const struct pq2_0_m1_ctx *c   = (const struct pq2_0_m1_ctx *) vctx;
    const uint8_t             *Wr  = c->W + r * c->row_bytes;
    float                      sum = 0.0f;
    for (size_t b = 0; b < c->blocks_per_row; b++) {
        const uint8_t *blk = Wr + b * PQ2_0_BLOCK_BYTES;
        const float    d   = fp16_to_fp32((uint16_t) blk[0] | ((uint16_t) blk[1] << 8));
        const int32_t  dot = pq2_0_block_dot_raw(blk + 2, c->xq + b * PQ2_0_BLOCK_ELEMS);
        sum += (float) (dot - c->bsum[b]) * d;
    }
    c->y[r] = sum * c->inv_act_scale;
}

/* Per-call activation prep shared by both kernels: absmax int8 quant in
 * the kernel element order (see file comment) plus the per-block sums
 * that fold the code bias out. Rounding matches tq2_0.c. Thread-local
 * workspace, grown on demand; false (y zeroed) on OOM or a width that
 * is not a whole number of blocks. */
static bool pq2_0_prep(struct cpu_neon_workspace *ws,
                       size_t                     n_in,
                       size_t                     n_out,
                       const float               *x,
                       float                     *y,
                       float                     *inv_act_scale) {
    const size_t nb = n_in / PQ2_0_BLOCK_ELEMS;
    if (n_in % PQ2_0_BLOCK_ELEMS != 0) {
        memset(y, 0, n_out * sizeof *y);
        return false;
    }
    if (ws->m1_xq_cap < n_in) {
        safe_free((void **) &ws->m1_xq);
        ws->m1_xq     = heap_alloc_array_aligned(int8_t, n_in);
        ws->m1_xq_cap = ws->m1_xq != nullptr ? n_in : 0;
    }
    if (ws->m1_bsum_cap < nb) {
        safe_free((void **) &ws->m1_bsum);
        ws->m1_bsum     = heap_alloc_array_aligned(int32_t, nb);
        ws->m1_bsum_cap = ws->m1_bsum != nullptr ? nb : 0;
    }
    if (ws->m1_xq == nullptr || ws->m1_bsum == nullptr) {
        memset(y, 0, n_out * sizeof *y);
        return false;
    }
    float max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > max_abs)
            max_abs = a;
    }
    const float act_scale = 127.0f / max_abs;
    for (size_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (size_t c = 0; c < 2; c++) {
            const size_t base = b * PQ2_0_BLOCK_ELEMS + c * 64;
            for (size_t m = 0; m < 16; m++) {
                for (size_t l = 0; l < 4; l++) {
                    const float q                = x[base + 4 * m + l] * act_scale;
                    int32_t     qi               = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
                    qi                           = qi > 127 ? 127 : (qi < -128 ? -128 : qi);
                    ws->m1_xq[base + 16 * l + m] = (int8_t) qi;
                    s += qi;
                }
            }
        }
        ws->m1_bsum[b] = s;
    }
    *inv_act_scale = max_abs / 127.0f;
    return true;
}

/* Dispatch exactly as cpu_neon_w_tq2_0_q8a_m1. */
static void pq2_0_parallel_for(size_t n, void (*body)(size_t, void *), void *ctx) {
    static _Atomic int pp_enabled = -1;
    if (pp_enabled < 0) {
        const char *e = getenv("GEIST_PP");
        pp_enabled    = (e && e[0] == '1') ? 1 : 0;
    }
    if (pp_enabled) {
        geist_pp_parallel_for(n, body, ctx);
    }
#ifdef _OPENMP
    else if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; i++)
            body(i, ctx);
    }
#endif
    else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < n; i++)
            body(i, ctx);
    }
}

void cpu_neon_w_pq2_0_q8a_m1(const float               *x,
                             const struct geist_weight *w,
                             struct geist_backend      *be,
                             float                     *y) {
    struct cpu_neon_workspace *ws    = cpu_neon_ws((struct cpu_neon_state *) be->state);
    const size_t               n_in  = (size_t) w->n_in;
    const size_t               n_out = (size_t) w->n_out;
    float                      inv   = 0.0f;
    if (!pq2_0_prep(ws, n_in, n_out, x, y, &inv)) {
        return;
    }
    struct pq2_0_m1_ctx ctx = {
            .W              = (const uint8_t *) w->raw,
            .xq             = ws->m1_xq,
            .bsum           = ws->m1_bsum,
            .y              = y,
            .inv_act_scale  = inv,
            .row_bytes      = n_in / PQ2_0_BLOCK_ELEMS * PQ2_0_BLOCK_BYTES,
            .blocks_per_row = n_in / PQ2_0_BLOCK_ELEMS,
    };
    pq2_0_parallel_for(n_out, pq2_0_m1_row_body, &ctx);
}

/* ---- x8: eight rows interleaved (decode) ---------------------------------
 *
 * The row kernel streams one 1.4-17 KB row per output and reaches ~55 GB/s
 * on the 27B; Q4_0's x8 layout (#291) shows what one sequential stream
 * serving eight rows buys. Block = 8 rows x 128 elements, 272 bytes:
 *
 *   [8 x fp16 d][256 bytes of codes]
 *
 * with the codes of row r, 64-element chunk c, source byte m (elements
 * 4m..4m+3 of the chunk) at
 *
 *   qs[c*128 + (r/4)*64 + (m/4)*16 + (r%4)*4 + (m%4)]
 *
 * i.e. every 16-byte vector holds 4 bytes of each of 4 rows. Against the
 * reordered activation (xq[c*64 + 16l + m] = element 4m + l), shift level
 * l of that vector is the lane-SDOT operand for activation vector l, lane
 * m/4: one weight load feeds four vdotq_laneq_s32, and the accumulator
 * lanes are four output rows, so no horizontal add is left. */

static inline size_t pq2_0_x8_idx(size_t r, size_t c, size_t m) {
    return c * 128 + (r / 4) * 64 + (m / 4) * 16 + (r % 4) * 4 + (m % 4);
}

size_t pq2_0_x8_size_bytes(size_t n_in, size_t n_out) {
    if (n_in == 0 || n_out == 0 || n_in % PQ2_0_BLOCK_ELEMS != 0 || n_out % 8 != 0) {
        return 0;
    }
    size_t bytes = 0;
    if (ckd_mul(&bytes, n_out / 8, n_in / PQ2_0_BLOCK_ELEMS) ||
        ckd_mul(&bytes, bytes, PQ2_0_X8_BLOCK_BYTES)) {
        return 0;
    }
    return bytes;
}

int pq2_0_x8_pack(const void *src, size_t n_in, size_t n_out, void *dst) {
    if (pq2_0_x8_size_bytes(n_in, n_out) == 0 || src == nullptr || dst == nullptr) {
        return -1;
    }
    const uint8_t *s  = (const uint8_t *) src;
    uint8_t       *d  = (uint8_t *) dst;
    const size_t   nb = n_in / PQ2_0_BLOCK_ELEMS;
    for (size_t tile = 0; tile < n_out / 8; tile++) {
        for (size_t b = 0; b < nb; b++) {
            uint8_t *ob = d + (tile * nb + b) * PQ2_0_X8_BLOCK_BYTES;
            for (size_t r = 0; r < 8; r++) {
                const uint8_t *sb = s + ((tile * 8 + r) * nb + b) * PQ2_0_BLOCK_BYTES;
                ob[2 * r]         = sb[0];
                ob[2 * r + 1]     = sb[1];
                for (size_t c = 0; c < 2; c++) {
                    for (size_t m = 0; m < 16; m++) {
                        ob[16 + pq2_0_x8_idx(r, c, m)] = sb[2 + c * 16 + m];
                    }
                }
            }
        }
    }
    return 0;
}

struct pq2_0_x8_ctx {
    const uint8_t *W;
    const int8_t  *xq;
    const int32_t *bsum;
    float         *y;
    float          inv_act_scale;
    size_t         blocks_per_row;
};

static void pq2_0_x8_tile_body(size_t tile, void *vctx) {
    const struct pq2_0_x8_ctx *c    = (const struct pq2_0_x8_ctx *) vctx;
    const uint8_t             *row  = c->W + tile * c->blocks_per_row * PQ2_0_X8_BLOCK_BYTES;
    const uint8x16_t           mask = vdupq_n_u8(3);
    float32x4_t                acc0 = vdupq_n_f32(0.0f);
    float32x4_t                acc1 = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < c->blocks_per_row; b++) {
        const uint8_t *blk = row + b * PQ2_0_X8_BLOCK_BYTES;
        __builtin_prefetch(blk + 2 * PQ2_0_X8_BLOCK_BYTES, 0, 0);
        const float16x8_t dh = vreinterpretq_f16_u16(vld1q_u16((const uint16_t *) blk));
        const float32x4_t d0 = vcvt_f32_f16(vget_low_f16(dh));
        const float32x4_t d1 = vcvt_f32_f16(vget_high_f16(dh));
        const uint8_t    *qs = blk + 16;
        const int8_t     *xb = c->xq + b * PQ2_0_BLOCK_ELEMS;
        int32x4_t         a0 = vdupq_n_s32(0);
        int32x4_t         a1 = vdupq_n_s32(0);
        for (size_t ch = 0; ch < 2; ch++) {
            const int8x16_t x0 = vld1q_s8(xb + ch * 64 + 0);
            const int8x16_t x1 = vld1q_s8(xb + ch * 64 + 16);
            const int8x16_t x2 = vld1q_s8(xb + ch * 64 + 32);
            const int8x16_t x3 = vld1q_s8(xb + ch * 64 + 48);
/* lane index must be a literal: unrolled per lane j */
#define PQ2_X8_LANE(j)                                                                 \
    do {                                                                               \
        const uint8x16_t w0_ = vld1q_u8(qs + ch * 128 + (j) * 16);                     \
        const uint8x16_t w1_ = vld1q_u8(qs + ch * 128 + 64 + (j) * 16);                \
        a0 = vdotq_laneq_s32(a0, vreinterpretq_s8_u8(vandq_u8(w0_, mask)), x0, (j));   \
        a1 = vdotq_laneq_s32(a1, vreinterpretq_s8_u8(vandq_u8(w1_, mask)), x0, (j));   \
        a0 = vdotq_laneq_s32(                                                          \
                a0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w0_, 2), mask)), x1, (j)); \
        a1 = vdotq_laneq_s32(                                                          \
                a1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w1_, 2), mask)), x1, (j)); \
        a0 = vdotq_laneq_s32(                                                          \
                a0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w0_, 4), mask)), x2, (j)); \
        a1 = vdotq_laneq_s32(                                                          \
                a1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w1_, 4), mask)), x2, (j)); \
        a0 = vdotq_laneq_s32(a0, vreinterpretq_s8_u8(vshrq_n_u8(w0_, 6)), x3, (j));    \
        a1 = vdotq_laneq_s32(a1, vreinterpretq_s8_u8(vshrq_n_u8(w1_, 6)), x3, (j));    \
    } while (0)
            PQ2_X8_LANE(0);
            PQ2_X8_LANE(1);
            PQ2_X8_LANE(2);
            PQ2_X8_LANE(3);
#undef PQ2_X8_LANE
        }
        const float32x4_t bias = vdupq_n_f32((float) c->bsum[b]);
        acc0                   = vfmaq_f32(acc0, d0, vsubq_f32(vcvtq_f32_s32(a0), bias));
        acc1                   = vfmaq_f32(acc1, d1, vsubq_f32(vcvtq_f32_s32(a1), bias));
    }
    vst1q_f32(c->y + tile * 8 + 0, vmulq_n_f32(acc0, c->inv_act_scale));
    vst1q_f32(c->y + tile * 8 + 4, vmulq_n_f32(acc1, c->inv_act_scale));
}

void cpu_neon_w_pq2_0_x8_m1(const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y) {
    struct cpu_neon_workspace *ws    = cpu_neon_ws((struct cpu_neon_state *) be->state);
    const size_t               n_in  = (size_t) w->n_in;
    const size_t               n_out = (size_t) w->n_out;
    float                      inv   = 0.0f;
    if (w->aux_fp32 == nullptr || !pq2_0_prep(ws, n_in, n_out, x, y, &inv)) {
        return;
    }
    struct pq2_0_x8_ctx ctx = {
            .W              = (const uint8_t *) w->aux_fp32,
            .xq             = ws->m1_xq,
            .bsum           = ws->m1_bsum,
            .y              = y,
            .inv_act_scale  = inv,
            .blocks_per_row = n_in / PQ2_0_BLOCK_ELEMS,
    };
    pq2_0_parallel_for(n_out / 8, pq2_0_x8_tile_body, &ctx);
}

#endif /* __ARM_NEON && __ARM_FEATURE_DOTPROD */
