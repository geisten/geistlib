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

void cpu_neon_w_pq2_0_q8a_m1(const float               *x,
                             const struct geist_weight *w,
                             struct geist_backend      *be,
                             float                     *y) {
    struct cpu_neon_workspace *ws             = cpu_neon_ws((struct cpu_neon_state *) be->state);
    const size_t               n_in           = (size_t) w->n_in;
    const size_t               n_out          = (size_t) w->n_out;
    const size_t               blocks_per_row = n_in / PQ2_0_BLOCK_ELEMS;
    if (n_in % PQ2_0_BLOCK_ELEMS != 0) {
        memset(y, 0, n_out * sizeof *y);
        return;
    }

    float max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > max_abs)
            max_abs = a;
    }
    const float act_scale = 127.0f / max_abs;

    if (ws->m1_xq_cap < n_in) {
        safe_free((void **) &ws->m1_xq);
        ws->m1_xq = heap_alloc_array_aligned(int8_t, n_in);
        if (ws->m1_xq == nullptr) {
            ws->m1_xq_cap = 0;
            memset(y, 0, n_out * sizeof *y);
            return;
        }
        ws->m1_xq_cap = n_in;
    }
    if (ws->m1_bsum_cap < blocks_per_row) {
        safe_free((void **) &ws->m1_bsum);
        ws->m1_bsum = heap_alloc_array_aligned(int32_t, blocks_per_row);
        if (ws->m1_bsum == nullptr) {
            ws->m1_bsum_cap = 0;
            memset(y, 0, n_out * sizeof *y);
            return;
        }
        ws->m1_bsum_cap = blocks_per_row;
    }

    /* Quantize in the kernel's element order (see file comment); the
     * block sum does not depend on order. Rounding matches tq2_0.c. */
    int8_t  *xq   = ws->m1_xq;
    int32_t *bsum = ws->m1_bsum;
    for (size_t b = 0; b < blocks_per_row; b++) {
        int32_t s = 0;
        for (size_t c = 0; c < 2; c++) {
            const size_t base = b * PQ2_0_BLOCK_ELEMS + c * 64;
            for (size_t m = 0; m < 16; m++) {
                for (size_t l = 0; l < 4; l++) {
                    const float q         = x[base + 4 * m + l] * act_scale;
                    int32_t     qi        = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
                    qi                    = qi > 127 ? 127 : (qi < -128 ? -128 : qi);
                    xq[base + 16 * l + m] = (int8_t) qi;
                    s += qi;
                }
            }
        }
        bsum[b] = s;
    }

    struct pq2_0_m1_ctx ctx = {
            .W              = (const uint8_t *) w->raw,
            .xq             = xq,
            .bsum           = bsum,
            .y              = y,
            .inv_act_scale  = max_abs / 127.0f,
            .row_bytes      = blocks_per_row * PQ2_0_BLOCK_BYTES,
            .blocks_per_row = blocks_per_row,
    };

    /* Dispatch exactly as cpu_neon_w_tq2_0_q8a_m1. */
    static _Atomic int pp_enabled = -1;
    if (pp_enabled < 0) {
        const char *e = getenv("GEIST_PP");
        pp_enabled    = (e && e[0] == '1') ? 1 : 0;
    }
    if (pp_enabled) {
        geist_pp_parallel_for(n_out, pq2_0_m1_row_body, &ctx);
    }
#ifdef _OPENMP
    else if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t r = 0; r < n_out; r++)
            pq2_0_m1_row_body(r, &ctx);
    }
#endif
    else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t r = 0; r < n_out; r++)
            pq2_0_m1_row_body(r, &ctx);
    }
}

#endif /* __ARM_NEON && __ARM_FEATURE_DOTPROD */
