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
 * Entry points are declared in internal.h and referenced by the resolver
 * table; all need dotprod.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../internal.h"
#include "../parallel.h"
#include "checked.h"
#include "geist_gemm.h"
#include "heap.h"
#include "linear_ref.h"

#include "quant.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>

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

static inline void pq2_0_m1_row(size_t r, const struct pq2_0_m1_ctx *c) {
    const uint8_t *Wr  = c->W + r * c->row_bytes;
    float          sum = 0.0f;
    for (size_t b = 0; b < c->blocks_per_row; b++) {
        const uint8_t *blk = Wr + b * PQ2_0_BLOCK_BYTES;
        const float    d   = fp16_to_fp32((uint16_t) blk[0] | ((uint16_t) blk[1] << 8));
        const int32_t  dot = pq2_0_block_dot_raw(blk + 2, c->xq + b * PQ2_0_BLOCK_ELEMS);
        sum += (float) (dot - c->bsum[b]) * d;
    }
    c->y[r] = sum * c->inv_act_scale;
}

/* Output rows [r0, r1); the ctx is copied so the loop reads no field
 * through the pointer (stores to y may alias it under
 * -fno-strict-aliasing). */
static void pq2_0_m1_rows(void *vctx, size_t r0, size_t r1) {
    const struct pq2_0_m1_ctx c = *(const struct pq2_0_m1_ctx *) vctx;
    for (size_t r = r0; r < r1; r++) {
        pq2_0_m1_row(r, &c);
    }
}

/* Per-call activation prep shared by both kernels: absmax int8 quant in
 * the kernel element order (see file comment) plus the per-block sums
 * that fold the code bias out. Rounding matches tq2_0.c. Thread-local
 * workspace, grown on demand; false on OOM or a width that is not a whole
 * number of blocks, and the caller computes y with geist_linear_ref. */
static bool
pq2_0_prep(struct cpu_neon_workspace *ws, size_t n_in, const float *x, float *inv_act_scale) {
    const size_t nb = n_in / PQ2_0_BLOCK_ELEMS;
    if (ws == nullptr || n_in % PQ2_0_BLOCK_ELEMS != 0) {
        return false;
    }
    if (!cpu_neon_grow_i8(&ws->m1_xq, &ws->m1_xq_cap, n_in) ||
        !cpu_neon_grow_i32(&ws->m1_bsum, &ws->m1_bsum_cap, nb)) {
        return false;
    }
    /* n_in is a whole number of blocks (checked above), so both loops run
     * whole vectors. The quant loop is pq2_0_permute_x plus an int8
     * convert: vld4q's four lanes are the four l values, each stored as 4
     * contiguous bytes at 16*l + 4*q. Rounding is +-0.5 then truncate
     * (vcvtq), bit-identical to the scalar quantizer. */
    float32x4_t mx = vdupq_n_f32(1e-5f);
    for (size_t i = 0; i < n_in; i += 4) {
        mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(x + i)));
    }
    const float       max_abs   = vmaxvq_f32(mx);
    const float       act_scale = 127.0f / max_abs;
    const float32x4_t vscale    = vdupq_n_f32(act_scale);
    const float32x4_t vhalf     = vdupq_n_f32(0.5f);
    const int32x4_t   vmin      = vdupq_n_s32(-128);
    const int32x4_t   vmax      = vdupq_n_s32(127);
    for (size_t b = 0; b < nb; b++) {
        int32x4_t acc = vdupq_n_s32(0);
        for (size_t c = 0; c < 2; c++) {
            const size_t base = b * PQ2_0_BLOCK_ELEMS + c * 64;
            for (size_t q = 0; q < 4; q++) {
                const float32x4x4_t v = vld4q_f32(x + base + 16 * q);
                for (size_t l = 0; l < 4; l++) {
                    const float32x4_t qf  = vmulq_f32(v.val[l], vscale);
                    const float32x4_t adj = vbslq_f32(vcltzq_f32(qf), vnegq_f32(vhalf), vhalf);
                    const int32x4_t   qi =
                            vminq_s32(vmaxq_s32(vcvtq_s32_f32(vaddq_f32(qf, adj)), vmin), vmax);
                    acc                   = vaddq_s32(acc, qi);
                    const int8x8_t packed = vqmovn_s16(vcombine_s16(vqmovn_s32(qi), vdup_n_s16(0)));
                    vst1_lane_s32((int32_t *) (void *) (ws->m1_xq + base + 16 * l + 4 * q),
                                  vreinterpret_s32_s8(packed),
                                  0);
                }
            }
        }
        ws->m1_bsum[b] = vaddvq_s32(acc);
    }
    *inv_act_scale = max_abs / 127.0f;
    return true;
}

void cpu_neon_w_pq2_0_q8a_m1(const float               *x,
                             const struct geist_weight *w,
                             struct geist_backend      *be,
                             float                     *y) {
    struct cpu_neon_workspace *ws    = cpu_neon_ws((struct cpu_neon_state *) be->state);
    const size_t               n_in  = (size_t) w->n_in;
    const size_t               n_out = (size_t) w->n_out;
    float                      inv   = 0.0f;
    if (!pq2_0_prep(ws, n_in, x, &inv)) {
        geist_linear_ref(1, x, w, y);
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
    geist_par_for(n_out, pq2_0_m1_rows, &ctx);
}

/* Two projections over one x (FFN gate/up, attention q/k/v): the int8
 * quantization and the per-block sums depend only on x and n_in, so they
 * are computed once and both row loops read them. The row space is walked
 * as one range, so the thread dispatch is shared too -- the workspace
 * invariant in internal.h asks for exactly this. Items [0, a_n) are
 * body(a, ...), the rest body(b, ...) shifted down by a_n; the m1 and x8
 * pairs both use it. */
struct pq2_0_pair {
    geist_par_fn body;
    void        *a, *b;
    size_t       a_n;
};

static void pq2_0_pair_range(void *vctx, size_t i0, size_t i1) {
    const struct pq2_0_pair *p = (const struct pq2_0_pair *) vctx;
    if (i0 < p->a_n) {
        p->body(p->a, i0, i1 < p->a_n ? i1 : p->a_n);
    }
    if (i1 > p->a_n) {
        p->body(p->b, (i0 > p->a_n ? i0 : p->a_n) - p->a_n, i1 - p->a_n);
    }
}

void cpu_neon_w_pq2_0_q8a_pair_m1(const float               *x,
                                  const struct geist_weight *w0,
                                  const struct geist_weight *w1,
                                  struct geist_backend      *be,
                                  float                     *y0,
                                  float                     *y1) {
    struct cpu_neon_workspace *ws  = cpu_neon_ws((struct cpu_neon_state *) be->state);
    const size_t               nin = (size_t) w0->n_in;
    const size_t               n0 = (size_t) w0->n_out, n1 = (size_t) w1->n_out;
    float                      inv = 0.0f;
    if ((size_t) w1->n_in != nin || !pq2_0_prep(ws, nin, x, &inv)) {
        geist_linear_ref(1, x, w0, y0);
        geist_linear_ref(1, x, w1, y1);
        return;
    }
    const size_t        rb = nin / PQ2_0_BLOCK_ELEMS * PQ2_0_BLOCK_BYTES;
    const size_t        bp = nin / PQ2_0_BLOCK_ELEMS;
    struct pq2_0_m1_ctx a  = {.W              = (const uint8_t *) w0->raw,
                              .xq             = ws->m1_xq,
                              .bsum           = ws->m1_bsum,
                              .y              = y0,
                              .inv_act_scale  = inv,
                              .row_bytes      = rb,
                              .blocks_per_row = bp};
    struct pq2_0_m1_ctx b  = a;
    b.W                    = (const uint8_t *) w1->raw;
    b.y                    = y1;
    struct pq2_0_pair pc   = {.body = pq2_0_m1_rows, .a = &a, .b = &b, .a_n = n0};
    geist_par_for(n0 + n1, pq2_0_pair_range, &pc);
}

/* x8: eight rows interleaved (decode). One sequential stream serves eight
 * output rows, as in Q4_0's x8 layout (#291). Block = 8 rows x 128
 * elements, 272 bytes:
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

/* One pq2_0_x8_pack call. */
struct pq2_0_x8_pack_job {
    const uint8_t *s;
    uint8_t       *d;
    size_t         nb;
};

/* Row tiles [t0, t1) of pq2_0_x8_pack. */
static void pq2_0_x8_pack_tiles(void *ctx, size_t t0, size_t t1) {
    const struct pq2_0_x8_pack_job *job = ctx;
    const uint8_t                  *s   = job->s;
    uint8_t                        *d   = job->d;
    const size_t                    nb  = job->nb;
    for (size_t tile = t0; tile < t1; tile++) {
        for (size_t b = 0; b < nb; b++) {
            uint8_t *ob = d + (tile * nb + b) * PQ2_0_X8_BLOCK_BYTES;
            for (size_t r = 0; r < 8; r++) {
                const uint8_t *sb = s + ((tile * 8 + r) * nb + b) * PQ2_0_BLOCK_BYTES;
                ob[2 * r]         = sb[0];
                ob[2 * r + 1]     = sb[1];
                for (size_t c = 0; c < 2; c++) {
                    for (size_t g = 0; g < 4; g++) {
                        memcpy(ob + 16 + pq2_0_x8_idx(r, c, 4 * g), sb + 2 + c * 16 + 4 * g, 4);
                    }
                }
            }
        }
    }
}

void pq2_0_x8_pack(const void *src, size_t n_in, size_t n_out, void *dst) {
    const uint8_t *s  = (const uint8_t *) src;
    uint8_t       *d  = (uint8_t *) dst;
    const size_t   nb = n_in / PQ2_0_BLOCK_ELEMS;
    /* Load-time repack of the whole tensor (7 GB on a 27B): threaded, 4
     * bytes at a time, since for fixed (r, c) both source and destination
     * are contiguous in m within each group of 4. */
    struct pq2_0_x8_pack_job job = {.s = s, .d = d, .nb = nb};
    geist_par_for(n_out / 8, pq2_0_x8_pack_tiles, &job); /* one tile: on this thread */
}

struct pq2_0_x8_ctx {
    const uint8_t *W;
    const int8_t  *xq;
    const int32_t *bsum;
    float         *y;
    float          inv_act_scale;
    size_t         blocks_per_row;
};

static inline void pq2_0_x8_tile(size_t tile, const struct pq2_0_x8_ctx *c) {
    const uint8_t   *row  = c->W + tile * c->blocks_per_row * PQ2_0_X8_BLOCK_BYTES;
    const uint8x16_t mask = vdupq_n_u8(3);
    float32x4_t      acc0 = vdupq_n_f32(0.0f);
    float32x4_t      acc1 = vdupq_n_f32(0.0f);
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
/* vdotq_laneq_s32 needs a literal lane index, hence the macro. */
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

/* Row tiles [t0, t1), on a copy of the ctx as pq2_0_m1_rows. */
static void pq2_0_x8_tiles(void *vctx, size_t t0, size_t t1) {
    const struct pq2_0_x8_ctx c = *(const struct pq2_0_x8_ctx *) vctx;
    for (size_t tile = t0; tile < t1; tile++) {
        pq2_0_x8_tile(tile, &c);
    }
}

void cpu_neon_w_pq2_0_x8_m1(const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y) {
    struct cpu_neon_workspace *ws    = cpu_neon_ws((struct cpu_neon_state *) be->state);
    const size_t               n_in  = (size_t) w->n_in;
    const size_t               n_out = (size_t) w->n_out;
    float                      inv   = 0.0f;
    if (w->aux_fp32 == nullptr || !pq2_0_prep(ws, n_in, x, &inv)) {
        geist_linear_ref(1, x, w, y);
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
    geist_par_for(n_out / 8, pq2_0_x8_tiles, &ctx);
}

/* The x8 twin of cpu_neon_w_pq2_0_q8a_pair_m1. */
void cpu_neon_w_pq2_0_x8_pair_m1(const float               *x,
                                 const struct geist_weight *w0,
                                 const struct geist_weight *w1,
                                 struct geist_backend      *be,
                                 float                     *y0,
                                 float                     *y1) {
    struct cpu_neon_workspace *ws  = cpu_neon_ws((struct cpu_neon_state *) be->state);
    const size_t               nin = (size_t) w0->n_in;
    const size_t               n0 = (size_t) w0->n_out, n1 = (size_t) w1->n_out;
    float                      inv = 0.0f;
    if ((size_t) w1->n_in != nin || w0->aux_fp32 == nullptr || w1->aux_fp32 == nullptr ||
        !pq2_0_prep(ws, nin, x, &inv)) {
        geist_linear_ref(1, x, w0, y0);
        geist_linear_ref(1, x, w1, y1);
        return;
    }
    struct pq2_0_x8_ctx a = {.W              = (const uint8_t *) w0->aux_fp32,
                             .xq             = ws->m1_xq,
                             .bsum           = ws->m1_bsum,
                             .y              = y0,
                             .inv_act_scale  = inv,
                             .blocks_per_row = nin / PQ2_0_BLOCK_ELEMS};
    struct pq2_0_x8_ctx b = a;
    b.W                   = (const uint8_t *) w1->aux_fp32;
    b.y                   = y1;
    struct pq2_0_pair pc  = {.body = pq2_0_x8_tiles, .a = &a, .b = &b, .a_n = n0 / 8};
    geist_par_for(n0 / 8 + n1 / 8, pq2_0_pair_range, &pc);
}

/* x8 prefill: NEON dequant straight from the x8 copy + SGEMM, so the
 * row-major source mmap is not touched. Tiles come out in the codes'
 * element order (position 16l + m of a 64-element chunk is element
 * 4m + l) and x is permuted into the same order once per call; a dot
 * product only needs both sides to agree on the order. */

/* One pq2_0_permute_x call. */
struct pq2_0_permute_job {
    size_t       n_in;
    const float *x;
    float       *xp;
};

/* Rows [t0, t1) of pq2_0_permute_x. */
static void pq2_0_permute_rows(void *ctx, size_t t0, size_t t1) {
    const struct pq2_0_permute_job *job  = ctx;
    const size_t                    n_in = job->n_in;
    const float                    *x    = job->x;
    float                          *xp   = job->xp;
    for (size_t t = t0; t < t1; t++) {
        const float *xr = x + t * n_in;
        float       *pr = xp + t * n_in;
        for (size_t base = 0; base < n_in; base += 64) {
            for (size_t q = 0; q < 4; q++) {
                const float32x4x4_t v = vld4q_f32(xr + base + 16 * q);
                vst1q_f32(pr + base + 0 + 4 * q, v.val[0]);
                vst1q_f32(pr + base + 16 + 4 * q, v.val[1]);
                vst1q_f32(pr + base + 32 + 4 * q, v.val[2]);
                vst1q_f32(pr + base + 48 + 4 * q, v.val[3]);
            }
        }
    }
}

/* xp[t][b*128 + c*64 + 16l + m] = x[t][b*128 + c*64 + 4m + l]. Threaded:
 * at m = 128 this moves ~22 MB each way per layer; one row runs on this
 * thread. */
static void pq2_0_permute_x(size_t m, size_t n_in, const float *x, float *xp) {
    struct pq2_0_permute_job job = {.n_in = n_in, .x = x, .xp = xp};
    geist_par_for(m, pq2_0_permute_rows, &job);
}

/* 16 codes (one shift level of a row's 16 bytes) -> 16 floats (code-1)*d. */
static inline void pq2_0_expand16(uint8x16_t codes, float d, float *out) {
    const int8x16_t v  = vsubq_s8(vreinterpretq_s8_u8(codes), vdupq_n_s8(1));
    const int16x8_t lo = vmovl_s8(vget_low_s8(v));
    const int16x8_t hi = vmovl_s8(vget_high_s8(v));
    vst1q_f32(out + 0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), d));
    vst1q_f32(out + 4, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), d));
    vst1q_f32(out + 8, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), d));
    vst1q_f32(out + 12, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), d));
}

/* Rows [tile8*8, tile8*8 + 8) of the x8 copy -> tile rows, fp32, permuted
 * element order, row stride n_in. */
static void pq2_0_x8_dequant8(const uint8_t *W, size_t nb, size_t n_in, size_t tile8, float *out) {
    const uint8x16_t mask = vdupq_n_u8(3);
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *blk = W + (tile8 * nb + b) * PQ2_0_X8_BLOCK_BYTES;
        float          d[8];
        vst1q_f32(d, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t *) blk))));
        vst1q_f32(d + 4, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t *) blk + 4))));
        const uint8_t *qs = blk + 16;
        for (size_t g = 0; g < 2; g++) {
            for (size_t c = 0; c < 2; c++) {
                /* V_j = [row0 bytes 4j..4j+3 | row1 | row2 | row3]; a 4x4
                 * u32 transpose turns them into R_i = row i bytes 0..15. */
                const uint32x4_t v0 = vreinterpretq_u32_u8(vld1q_u8(qs + c * 128 + g * 64 + 0));
                const uint32x4_t v1 = vreinterpretq_u32_u8(vld1q_u8(qs + c * 128 + g * 64 + 16));
                const uint32x4_t v2 = vreinterpretq_u32_u8(vld1q_u8(qs + c * 128 + g * 64 + 32));
                const uint32x4_t v3 = vreinterpretq_u32_u8(vld1q_u8(qs + c * 128 + g * 64 + 48));
                const uint32x4_t t0 = vtrn1q_u32(v0, v1), t1 = vtrn2q_u32(v0, v1);
                const uint32x4_t t2 = vtrn1q_u32(v2, v3), t3 = vtrn2q_u32(v2, v3);
                const uint8x16_t R[4] = {
                        vreinterpretq_u8_u64(
                                vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2))),
                        vreinterpretq_u8_u64(
                                vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3))),
                        vreinterpretq_u8_u64(
                                vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2))),
                        vreinterpretq_u8_u64(
                                vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3))),
                };
                for (size_t i = 0; i < 4; i++) {
                    const size_t r   = g * 4 + i;
                    float       *dst = out + r * n_in + b * PQ2_0_BLOCK_ELEMS + c * 64;
                    pq2_0_expand16(vandq_u8(R[i], mask), d[r], dst + 0);
                    pq2_0_expand16(vandq_u8(vshrq_n_u8(R[i], 2), mask), d[r], dst + 16);
                    pq2_0_expand16(vandq_u8(vshrq_n_u8(R[i], 4), mask), d[r], dst + 32);
                    pq2_0_expand16(vshrq_n_u8(R[i], 6), d[r], dst + 48);
                }
            }
        }
    }
}

/* Rows per dequant tile and per SGEMM call. Each thread dequantizes its
 * own tile and runs its own SGEMM, so dequant overlaps the others' AMX
 * work; a shared panel for one big SGEMM serializes the two phases and
 * measured slower. 128 was the best tile size on the 27B (M1 Max). */
constexpr size_t PQ2_0_X8_TILE_ROWS = 128;

/* The tile loop, against an x already permuted into kernel order; the
 * pair path permutes once and runs it twice. */
struct pq2_0_x8_gemm_job {
    struct cpu_neon_state     *st;
    size_t                     m, n_in, n_out;
    const uint8_t             *W;
    const float               *xp, *x;
    const struct geist_weight *w;
    float                     *y;
};

/* Tiles [t0, t1) of pq2_0_x8_gemm_permuted. */
static void pq2_0_x8_gemm_tiles(void *ctx, size_t t0, size_t t1) {
    const struct pq2_0_x8_gemm_job *job   = ctx;
    struct cpu_neon_state          *st    = job->st;
    const size_t                    m     = job->m;
    const size_t                    n_in  = job->n_in;
    const size_t                    n_out = job->n_out;
    const uint8_t                  *W     = job->W;
    const float                    *xp    = job->xp;
    const float                    *x     = job->x;
    const struct geist_weight      *w     = job->w;
    float                          *y     = job->y;
    const size_t                    nb    = n_in / PQ2_0_BLOCK_ELEMS;
    const size_t                    T     = PQ2_0_X8_TILE_ROWS;
    for (size_t ti = t0; ti < t1; ti++) {
        struct cpu_neon_workspace *tws = cpu_neon_ws(st);
        const size_t               r0  = ti * T;
        const size_t               tr  = n_out - r0 < T ? n_out - r0 : T;
        if (tws == nullptr || !cpu_neon_grow_f32(&tws->pq2_tile, &tws->pq2_tile_cap, T * n_in)) {
            /* No tile to stage these rows in: the reference computes them,
             * from the unpermuted x and the source weight. */
            geist_linear_ref_rows(m, r0, tr, n_out, x, w, y + r0);
            continue;
        }
        for (size_t k = 0; k < tr / 8; k++) {
            pq2_0_x8_dequant8(W, nb, n_in, r0 / 8 + k, tws->pq2_tile + k * 8 * n_in);
        }
        geist_sgemm(GEIST_OP_N,
                    GEIST_OP_T,
                    (int) m,
                    (int) tr,
                    (int) n_in,
                    1.0f,
                    xp,
                    (int) n_in,
                    tws->pq2_tile,
                    (int) n_in,
                    0.0f,
                    y + r0,
                    (int) n_out);
    }
}

static void pq2_0_x8_gemm_permuted(struct cpu_neon_state     *st,
                                   size_t                     m,
                                   size_t                     n_in,
                                   size_t                     n_out,
                                   const uint8_t             *W,
                                   const float               *xp,
                                   const float               *x,
                                   const struct geist_weight *w,
                                   float                     *y) {
    const size_t             T   = PQ2_0_X8_TILE_ROWS;
    struct pq2_0_x8_gemm_job job = {.st    = st,
                                    .m     = m,
                                    .n_in  = n_in,
                                    .n_out = n_out,
                                    .W     = W,
                                    .xp    = xp,
                                    .x     = x,
                                    .w     = w,
                                    .y     = y};
    cpu_neon_par_for_dynamic((n_out + T - 1) / T, 1, pq2_0_x8_gemm_tiles, &job);
}

/* Grows the shared permute buffer and fills it. Returns nullptr on
 * refusal, having zeroed nothing -- the caller owns its own y's. */
static const float *
pq2_0_permuted_x(struct cpu_neon_workspace *ws, size_t m, size_t n_in, const float *x) {
    size_t need = 0;
    if (m == 0 || ws == nullptr || ckd_mul(&need, m, n_in) ||
        !cpu_neon_grow_f32(&ws->pq2_xp, &ws->pq2_xp_cap, need)) {
        return nullptr;
    }
    pq2_0_permute_x(m, n_in, x, ws->pq2_xp);
    return ws->pq2_xp;
}

void cpu_neon_w_pq2_0_x8_mN(size_t                     m,
                            const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y) {
    struct cpu_neon_state     *st    = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws    = cpu_neon_ws(st);
    const size_t               n_in  = (size_t) w->n_in;
    const size_t               n_out = (size_t) w->n_out;
    const uint8_t             *W     = (const uint8_t *) w->aux_fp32;
    const float               *xp    = pq2_0_permuted_x(ws, m, n_in, x);
    if (W == nullptr || xp == nullptr) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    pq2_0_x8_gemm_permuted(st, m, n_in, n_out, W, xp, x, w, y);
}

/* Prefill twin of cpu_neon_w_pq2_0_x8_pair_m1: one permute, two GEMMs.
 * The tile loops stay separate parallel regions -- at m = 128 the SGEMM
 * work per region dwarfs the dispatch, and merging them would have both
 * projections contend for the same per-thread dequant tile. */
void cpu_neon_w_pq2_0_x8_pair_mN(size_t                     m,
                                 const float               *x,
                                 const struct geist_weight *w0,
                                 const struct geist_weight *w1,
                                 struct geist_backend      *be,
                                 float                     *y0,
                                 float                     *y1) {
    struct cpu_neon_state     *st  = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws  = cpu_neon_ws(st);
    const size_t               nin = (size_t) w0->n_in;
    const size_t               n0 = (size_t) w0->n_out, n1 = (size_t) w1->n_out;
    const uint8_t             *W0 = (const uint8_t *) w0->aux_fp32;
    const uint8_t             *W1 = (const uint8_t *) w1->aux_fp32;
    const float *xp = (size_t) w1->n_in == nin ? pq2_0_permuted_x(ws, m, nin, x) : nullptr;
    if (W0 == nullptr || W1 == nullptr || xp == nullptr) {
        geist_linear_ref(m, x, w0, y0);
        geist_linear_ref(m, x, w1, y1);
        return;
    }
    pq2_0_x8_gemm_permuted(st, m, nin, n0, W0, xp, x, w0, y0);
    pq2_0_x8_gemm_permuted(st, m, nin, n1, W1, xp, x, w1, y1);
}

#endif /* __ARM_NEON && __ARM_FEATURE_DOTPROD */
