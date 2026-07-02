/*
 * src/backends/cpu_neon/kernels/q4_K.c — Q4_K W4A8 NEON kernels.
 *
 * Layer: BACKEND (cpu_neon). Extracted from src/formats/gguf/q4_K.c
 * during the kernels-out-of-formats split.
 *
 * Owns the M=1 decode + M>1 prefill paths for Q4_K weights, plus the
 * activation-quant helper (`quantize_x_for_q4k`) shared between them.
 * The pure file-format decoder (`dequant_q4_K_row`) and the block
 * struct stay in src/formats/gguf/.
 *
 * The legacy fp32 reference path (`linear_q4k_decode_fp32`) lives here
 * too — it uses the format struct + helpers but the cblas trampoline
 * in cpu_neon's weight_resolve sees it as a competitor kernel.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "quant.h"
#include "gemma4_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

typedef void (*geist_pp_body_fn)(size_t i, void *ctx);
extern void geist_pp_parallel_for(size_t n, geist_pp_body_fn body_fn, void *ctx)
        __attribute__((weak));

static int q4k_pp_enabled(void) {
    static int enabled = -1;
    if (enabled >= 0)
        return enabled;
    const char *e = getenv("GEIST_PP");
    if (e != NULL && e[0] != '\0') {
        enabled = (e[0] == '1') ? 1 : 0;
        return enabled;
    }
    enabled = 0;
    return enabled;
}

struct q4k_predecode_header {
    uint32_t magic;
    uint32_t n_in;
    uint32_t n_out;
    uint32_t n_blocks_per_row;
    uint32_t block_bytes;
    uint32_t reserved;
};

struct q4k_predecode_block {
    float   d;
    float   dmin;
    uint8_t scales[8];
    uint8_t mins[8];
    int8_t  qs[Q4_K_BLOCK_ELEMS];
};

static constexpr uint32_t Q4K_PREDECODE_MAGIC        = 0x514B3450u; /* "P4KQ" little-endian */
static constexpr uint32_t Q4K_PREDECODE_NTILE4_MAGIC = 0x344B3450u; /* "P4K4" */

static_assert(sizeof(struct q4k_predecode_block) == 280, "q4k_predecode_block layout changed");

static inline const struct q4k_predecode_block *q4k_predecode_blocks(const void *packed) {
    return (const struct q4k_predecode_block *) ((const uint8_t *) packed +
                                                 sizeof(struct q4k_predecode_header));
}

static inline const struct q4k_predecode_block *q4k_predecode_ntile4_blocks(const void *packed) {
    return (const struct q4k_predecode_block *) ((const uint8_t *) packed +
                                                 sizeof(struct q4k_predecode_header));
}

static inline bool q4k_predecode_valid(const void *packed, size_t n_in, size_t n_out) {
    if (packed == NULL)
        return false;
    const struct q4k_predecode_header *h = (const struct q4k_predecode_header *) packed;
    return h->magic == Q4K_PREDECODE_MAGIC && h->n_in == (uint32_t) n_in &&
           h->n_out == (uint32_t) n_out &&
           h->n_blocks_per_row == (uint32_t) (n_in / Q4_K_BLOCK_ELEMS) &&
           h->block_bytes == sizeof(struct q4k_predecode_block);
}

static inline bool q4k_predecode_ntile4_valid(const void *packed, size_t n_in, size_t n_out) {
    if (packed == NULL)
        return false;
    const struct q4k_predecode_header *h = (const struct q4k_predecode_header *) packed;
    return h->magic == Q4K_PREDECODE_NTILE4_MAGIC && h->n_in == (uint32_t) n_in &&
           h->n_out == (uint32_t) n_out &&
           h->n_blocks_per_row == (uint32_t) (n_in / Q4_K_BLOCK_ELEMS) &&
           h->block_bytes == sizeof(struct q4k_predecode_block);
}

size_t q4k_predecode_size_bytes(size_t n_in, size_t n_out) {
    if (n_in == 0 || n_out == 0 || n_in % Q4_K_BLOCK_ELEMS != 0)
        return 0;
    if (n_in / Q4_K_BLOCK_ELEMS > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX) {
        return 0;
    }
    const size_t n_blocks = (n_in / Q4_K_BLOCK_ELEMS) * n_out;
    if (n_blocks >
        (SIZE_MAX - sizeof(struct q4k_predecode_header)) / sizeof(struct q4k_predecode_block)) {
        return 0;
    }
    return sizeof(struct q4k_predecode_header) + n_blocks * sizeof(struct q4k_predecode_block);
}

size_t q4k_predecode_ntile4_size_bytes(size_t n_in, size_t n_out) {
    if (n_in == 0 || n_out == 0 || n_in % Q4_K_BLOCK_ELEMS != 0)
        return 0;
    if (n_in / Q4_K_BLOCK_ELEMS > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX) {
        return 0;
    }
    const size_t n_tiles  = (n_out + 3) / 4;
    const size_t n_blocks = n_tiles * (n_in / Q4_K_BLOCK_ELEMS) * 4;
    if (n_blocks >
        (SIZE_MAX - sizeof(struct q4k_predecode_header)) / sizeof(struct q4k_predecode_block)) {
        return 0;
    }
    return sizeof(struct q4k_predecode_header) + n_blocks * sizeof(struct q4k_predecode_block);
}

int q4k_predecode_pack(const void *w_q4k, size_t n_in, size_t n_out, void *dst) {
    if (w_q4k == NULL || dst == NULL)
        return -1;
    const size_t bytes = q4k_predecode_size_bytes(n_in, n_out);
    if (bytes == 0)
        return -1;

    const size_t                 n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    struct q4k_predecode_header *h                = (struct q4k_predecode_header *) dst;
    *h                                            = (struct q4k_predecode_header) {
            .magic            = Q4K_PREDECODE_MAGIC,
            .n_in             = (uint32_t) n_in,
            .n_out            = (uint32_t) n_out,
            .n_blocks_per_row = (uint32_t) n_blocks_per_row,
            .block_bytes      = (uint32_t) sizeof(struct q4k_predecode_block),
            .reserved         = 0,
    };

    const struct block_q4_K_t  *src = (const struct block_q4_K_t *) w_q4k;
    struct q4k_predecode_block *dstb =
            (struct q4k_predecode_block *) ((uint8_t *) dst + sizeof(*h));

    for (size_t n = 0; n < n_out; n++) {
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_q4_K_t  *s = src + n * n_blocks_per_row + b;
            struct q4k_predecode_block *d = dstb + n * n_blocks_per_row + b;
            d->d                          = fp16_to_fp32(s->d);
            d->dmin                       = fp16_to_fp32(s->dmin);
            for (int is = 0; is < 8; is++) {
                get_scale_min_k4(is, s->scales, &d->scales[is], &d->mins[is]);
            }
            const uint8_t *q = s->qs;
            for (int is = 0; is < 8; is += 2, q += 32) {
                int8_t *qlo = d->qs + is * 32;
                int8_t *qhi = qlo + 32;
                for (int l = 0; l < 32; l++) {
                    qlo[l] = (int8_t) (q[l] & 0x0F);
                    qhi[l] = (int8_t) (q[l] >> 4);
                }
            }
        }
    }
    return 0;
}

int q4k_predecode_ntile4_pack(const void *w_q4k, size_t n_in, size_t n_out, void *dst) {
    if (w_q4k == NULL || dst == NULL)
        return -1;
    const size_t bytes = q4k_predecode_ntile4_size_bytes(n_in, n_out);
    if (bytes == 0)
        return -1;

    const size_t                 n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    struct q4k_predecode_header *h                = (struct q4k_predecode_header *) dst;
    *h                                            = (struct q4k_predecode_header) {
            .magic            = Q4K_PREDECODE_NTILE4_MAGIC,
            .n_in             = (uint32_t) n_in,
            .n_out            = (uint32_t) n_out,
            .n_blocks_per_row = (uint32_t) n_blocks_per_row,
            .block_bytes      = (uint32_t) sizeof(struct q4k_predecode_block),
            .reserved         = 0,
    };

    const struct block_q4_K_t  *src = (const struct block_q4_K_t *) w_q4k;
    struct q4k_predecode_block *dstb =
            (struct q4k_predecode_block *) ((uint8_t *) dst + sizeof(*h));
    memset(dstb, 0, bytes - sizeof(*h));

    const size_t n_tiles = (n_out + 3) / 4;
    for (size_t nt = 0; nt < n_tiles; nt++) {
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            for (size_t nr = 0; nr < 4; nr++) {
                const size_t n = nt * 4 + nr;
                if (n >= n_out)
                    continue;
                const struct block_q4_K_t  *s = src + n * n_blocks_per_row + b;
                struct q4k_predecode_block *d = dstb + (nt * n_blocks_per_row + b) * 4 + nr;
                d->d                          = fp16_to_fp32(s->d);
                d->dmin                       = fp16_to_fp32(s->dmin);
                for (int is = 0; is < 8; is++) {
                    get_scale_min_k4(is, s->scales, &d->scales[is], &d->mins[is]);
                }
                const uint8_t *q = s->qs;
                for (int is = 0; is < 8; is += 2, q += 32) {
                    int8_t *qlo = d->qs + is * 32;
                    int8_t *qhi = qlo + 32;
                    for (int l = 0; l < 32; l++) {
                        qlo[l] = (int8_t) (q[l] & 0x0F);
                        qhi[l] = (int8_t) (q[l] >> 4);
                    }
                }
            }
        }
    }
    return 0;
}

static inline void q4k_subpair_dots(const uint8_t *q,
                                    const float   *x1,
                                    const float   *x2,
                                    float         *a1,
                                    float         *s1,
                                    float         *a2,
                                    float         *s2) {
#if defined(__ARM_NEON)
    float32x4_t a1v = vdupq_n_f32(0.0f), s1v = vdupq_n_f32(0.0f);
    float32x4_t a2v = vdupq_n_f32(0.0f), s2v = vdupq_n_f32(0.0f);
    for (int half = 0; half < 32; half += 16) {
        uint8x16_t qv = vld1q_u8(q + half);
        uint8x16_t lo = vandq_u8(qv, vdupq_n_u8(0x0F));
        uint8x16_t hi = vshrq_n_u8(qv, 4);

        /* Expand uint8x16 → four float32x4 vectors (16 elements). */
        uint16x8_t  lo16_l = vmovl_u8(vget_low_u8(lo));
        uint16x8_t  lo16_h = vmovl_u8(vget_high_u8(lo));
        uint16x8_t  hi16_l = vmovl_u8(vget_low_u8(hi));
        uint16x8_t  hi16_h = vmovl_u8(vget_high_u8(hi));
        float32x4_t lo_f0  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_l)));
        float32x4_t lo_f1  = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16_l)));
        float32x4_t lo_f2  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_h)));
        float32x4_t lo_f3  = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16_h)));
        float32x4_t hi_f0  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_l)));
        float32x4_t hi_f1  = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16_l)));
        float32x4_t hi_f2  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_h)));
        float32x4_t hi_f3  = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16_h)));

        float32x4_t xa0 = vld1q_f32(x1 + half + 0);
        float32x4_t xa1 = vld1q_f32(x1 + half + 4);
        float32x4_t xa2 = vld1q_f32(x1 + half + 8);
        float32x4_t xa3 = vld1q_f32(x1 + half + 12);
        a1v             = vfmaq_f32(a1v, lo_f0, xa0);
        a1v             = vfmaq_f32(a1v, lo_f1, xa1);
        a1v             = vfmaq_f32(a1v, lo_f2, xa2);
        a1v             = vfmaq_f32(a1v, lo_f3, xa3);
        s1v             = vaddq_f32(s1v, vaddq_f32(vaddq_f32(xa0, xa1), vaddq_f32(xa2, xa3)));

        float32x4_t xb0 = vld1q_f32(x2 + half + 0);
        float32x4_t xb1 = vld1q_f32(x2 + half + 4);
        float32x4_t xb2 = vld1q_f32(x2 + half + 8);
        float32x4_t xb3 = vld1q_f32(x2 + half + 12);
        a2v             = vfmaq_f32(a2v, hi_f0, xb0);
        a2v             = vfmaq_f32(a2v, hi_f1, xb1);
        a2v             = vfmaq_f32(a2v, hi_f2, xb2);
        a2v             = vfmaq_f32(a2v, hi_f3, xb3);
        s2v             = vaddq_f32(s2v, vaddq_f32(vaddq_f32(xb0, xb1), vaddq_f32(xb2, xb3)));
    }
    *a1 = vaddvq_f32(a1v);
    *s1 = vaddvq_f32(s1v);
    *a2 = vaddvq_f32(a2v);
    *s2 = vaddvq_f32(s2v);
#else
    float aa1 = 0.0f, ss1 = 0.0f, aa2 = 0.0f, ss2 = 0.0f;
    for (int l = 0; l < 32; l++) {
        aa1 += (float) (q[l] & 0xF) * x1[l];
        aa2 += (float) (q[l] >> 4) * x2[l];
        ss1 += x1[l];
        ss2 += x2[l];
    }
    *a1 = aa1;
    *s1 = ss1;
    *a2 = aa2;
    *s2 = ss2;
#endif
}

float quantize_x_for_q4k(const float *x, size_t n, int8_t *x_q8, int32_t *sum32) {
    float        scale    = quantize_x_int8_sym(x, n, x_q8);
    const size_t n_chunks = n / 32;
    for (size_t s = 0; s < n_chunks; s++) {
#if defined(__ARM_NEON)
        int8x16_t lo = vld1q_s8(x_q8 + s * 32 + 0);
        int8x16_t hi = vld1q_s8(x_q8 + s * 32 + 16);
        sum32[s]     = (int32_t) vaddlvq_s8(lo) + (int32_t) vaddlvq_s8(hi);
#else
        int32_t sum = 0;
        for (int j = 0; j < 32; j++)
            sum += x_q8[s * 32 + j];
        sum32[s] = sum;
#endif
    }
    return scale;
}

void quantize_x_for_q4k_blocks(
        const float *x, size_t n, int8_t *x_q8, int32_t *sum32, float *scale_blocks) {
    if (x == NULL || x_q8 == NULL || sum32 == NULL || scale_blocks == NULL)
        return;
    if (n % Q4_K_BLOCK_ELEMS != 0)
        return;
    const size_t n_blocks = n / Q4_K_BLOCK_ELEMS;
    for (size_t b = 0; b < n_blocks; b++) {
        const size_t elem_off = b * Q4_K_BLOCK_ELEMS;
        scale_blocks[b] = quantize_x_int8_sym(x + elem_off, Q4_K_BLOCK_ELEMS, x_q8 + elem_off);
        for (size_t s = 0; s < Q4_K_BLOCK_ELEMS / 32; s++) {
#if defined(__ARM_NEON)
            const int8_t *p  = x_q8 + elem_off + s * 32;
            int8x16_t     lo = vld1q_s8(p + 0);
            int8x16_t     hi = vld1q_s8(p + 16);
            sum32[b * (Q4_K_BLOCK_ELEMS / 32) + s] =
                    (int32_t) vaddlvq_s8(lo) + (int32_t) vaddlvq_s8(hi);
#else
            int32_t sum = 0;
            for (int j = 0; j < 32; j++) {
                sum += x_q8[elem_off + s * 32 + (size_t) j];
            }
            sum32[b * (Q4_K_BLOCK_ELEMS / 32) + s] = sum;
#endif
        }
    }
}

struct q4k_decode_ctx {
    const struct block_q4_K_t *w;
    const int8_t              *x_q8;
    const int32_t             *sum32;
    float                      scale_x;
    float                     *y;
    size_t                     n_blocks_per_row;
    size_t                     n_out;
};

static inline void q4k_decode_one_row(size_t n, const struct q4k_decode_ctx *c) {
    const struct block_q4_K_t *row = c->w + n * c->n_blocks_per_row;
    float                      acc = 0.0f;
    if (n + 1 < c->n_out)
        __builtin_prefetch(row + c->n_blocks_per_row, 0, 0);
    for (size_t b = 0; b < c->n_blocks_per_row; b++) {
        const struct block_q4_K_t *blk = &row[b];
        if (b + 2 < c->n_blocks_per_row)
            __builtin_prefetch(&row[b + 2], 0, 0);
        const float    d_blk    = fp16_to_fp32(blk->d);
        const float    dmin_blk = fp16_to_fp32(blk->dmin);
        const uint8_t *q        = blk->qs;
        const int8_t  *xb       = c->x_q8 + b * Q4_K_BLOCK_ELEMS;
        const int32_t *sump_blk = c->sum32 + (b * Q4_K_BLOCK_ELEMS) / 32;

        /* P10.1: factor mins-correction OUT of the sub-pair loop.
         * Per super-block:
         *   sumi      = Σ_is scales[is] × dot[is]    (int32)
         *   mins_corr = Σ_is mins[is]   × sump[is]   (int32)
         *   acc += scale_x * (d_blk * sumi - dmin_blk * mins_corr)
         * Phase 1 (acc32): keep sumi as an int32x4_t accumulator across
         * the super-block; single vaddvq_s32 per super-block. */
        uint8_t scales[8], mins[8];
        for (int is = 0; is < 8; is++) {
            get_scale_min_k4(is, blk->scales, &scales[is], &mins[is]);
        }
        int32_t mins_corr = 0;
        for (int is = 0; is < 8; is++) {
            mins_corr += (int32_t) mins[is] * sump_blk[is];
        }

#if defined(__ARM_NEON)
        int32x4_t sumi_acc = vdupq_n_s32(0);
        for (int is = 0; is < 8; is += 2, q += 32, xb += 64) {
            const int32_t s_lo = (int32_t) scales[is + 0];
            const int32_t s_hi = (int32_t) scales[is + 1];
            for (int half = 0; half < 32; half += 16) {
                uint8x16_t qv   = vld1q_u8(q + half);
                int8x16_t  lo   = vreinterpretq_s8_u8(vandq_u8(qv, vdupq_n_u8(0x0F)));
                int8x16_t  hi   = vreinterpretq_s8_u8(vshrq_n_u8(qv, 4));
                int32x4_t  d_lo = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(xb + half));
                int32x4_t  d_hi = vdotq_s32(vdupq_n_s32(0), hi, vld1q_s8(xb + 32 + half));
                sumi_acc        = vmlaq_n_s32(sumi_acc, d_lo, s_lo);
                sumi_acc        = vmlaq_n_s32(sumi_acc, d_hi, s_hi);
            }
        }
        const int32_t sumi = vaddvq_s32(sumi_acc);
#else
        int32_t sumi = 0;
        for (int is = 0; is < 8; is += 2, q += 32, xb += 64) {
            int32_t acc1 = 0, acc2 = 0;
            for (int l = 0; l < 32; l++) {
                acc1 += (int32_t) (q[l] & 0xF) * (int32_t) xb[l];
                acc2 += (int32_t) (q[l] >> 4) * (int32_t) xb[32 + l];
            }
            sumi += (int32_t) scales[is + 0] * acc1 + (int32_t) scales[is + 1] * acc2;
        }
#endif
        acc += c->scale_x * (d_blk * (float) sumi - dmin_blk * (float) mins_corr);
    }
    c->y[n] = acc;
}

static void q4k_pp_row(size_t n, void *vctx) {
    q4k_decode_one_row(n, (const struct q4k_decode_ctx *) vctx);
}

void linear_q4k_decode_w4a8_pre(const int8_t  *x_q8,
                                float          scale_x,
                                const int32_t *sum32,
                                const void    *w_q4k,
                                size_t         n_in,
                                size_t         n_out,
                                float         *y) {
    const struct q4k_decode_ctx ctx = {
            .w                = (const struct block_q4_K_t *) w_q4k,
            .x_q8             = x_q8,
            .sum32            = sum32,
            .scale_x          = scale_x,
            .y                = y,
            .n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS,
            .n_out            = n_out,
    };

    if (q4k_pp_enabled()) {
#if defined(_OPENMP)
        if (!omp_in_parallel())
#endif
        {
            geist_pp_parallel_for(n_out, q4k_pp_row, (void *) &ctx);
            return;
        }
    }

#if defined(_OPENMP)
    if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t n = 0; n < n_out; n++)
            q4k_decode_one_row(n, &ctx);
    } else if (n_out >= 4096) {
#pragma omp parallel for schedule(static)
        for (size_t n = 0; n < n_out; n++)
            q4k_decode_one_row(n, &ctx);
    } else {
#pragma omp parallel for schedule(dynamic, 4)
        for (size_t n = 0; n < n_out; n++)
            q4k_decode_one_row(n, &ctx);
    }
#else
    for (size_t n = 0; n < n_out; n++)
        q4k_decode_one_row(n, &ctx);
#endif
}

struct q4k_pair_ctx {
    const struct q4k_decode_ctx *c0;
    const struct q4k_decode_ctx *c1;
};

static void q4k_pp_pair_row(size_t n, void *vctx) {
    const struct q4k_pair_ctx *c = (const struct q4k_pair_ctx *) vctx;
    if (n < c->c0->n_out)
        q4k_decode_one_row(n, c->c0);
    if (n < c->c1->n_out)
        q4k_decode_one_row(n, c->c1);
}

void linear_q4k_decode_w4a8(
        const float *x, const void *w_q4k, size_t n_in, size_t n_out, float *y) {
    /* G2: per-thread quantize cache. In Gemma 4 / Llama, q/k/v_proj share
     * the same post-attn-norm input x, and gate/up_proj share the post-ffn-
     * norm input x. Caching x_q8 + sum32 + scale across the 3 (or 2) calls
     * saves 2 of 3 (or 1 of 2) quantize_x_for_q4k passes per group. The
     * cache hangs off the master thread (this function runs serially before
     * the inner-loop OMP team is spawned). Key: (x ptr, n_in, fingerprint).
     * Fingerprint guards against same-pointer reuse across rmsnorm cycles. */
    static _Thread_local int8_t      *tl_x_q8         = NULL;
    static _Thread_local int32_t     *tl_sum32        = NULL;
    static _Thread_local size_t       tl_cap_n_in     = 0;
    static _Thread_local const float *tl_last_x       = NULL;
    static _Thread_local size_t       tl_last_n_in    = 0;
    static _Thread_local uint64_t     tl_last_sig     = 0;
    static _Thread_local float        tl_last_scale_x = 0.0f;

    /* Grow scratch caches on demand. */
    if (n_in > tl_cap_n_in) {
        safe_free((void **) &tl_x_q8);
        safe_free((void **) &tl_sum32);
        tl_x_q8  = heap_alloc_array_aligned(int8_t, n_in);
        tl_sum32 = heap_alloc_array_aligned(int32_t, n_in / 32);
        if (tl_x_q8 == NULL || tl_sum32 == NULL) {
            tl_cap_n_in = 0;
            return;
        }
        tl_cap_n_in = n_in;
        tl_last_x   = NULL; /* invalidate */
    }

    /* Cheap fingerprint: 5 fp32 samples across the vector. ~5 ns vs ~1.5 μs
     * for a full re-quantize, so the false-miss cost is irrelevant. */
    union {
        float    f;
        uint32_t u;
    } s0, s1, s2, s3, s4;
    s0.f               = x[0];
    s1.f               = x[n_in / 4];
    s2.f               = x[n_in / 2];
    s3.f               = x[(3 * n_in) / 4];
    s4.f               = x[n_in - 1];
    const uint64_t sig = (uint64_t) s0.u ^ ((uint64_t) s1.u << 7) ^ ((uint64_t) s2.u << 13) ^
                         ((uint64_t) s3.u << 19) ^ ((uint64_t) s4.u << 23);

    float scale_x;
    if (tl_last_x == x && tl_last_n_in == n_in && tl_last_sig == sig) {
        /* Cache hit — reuse pre-quantized data. */
        scale_x = tl_last_scale_x;
    } else {
        scale_x         = quantize_x_for_q4k(x, n_in, tl_x_q8, tl_sum32);
        tl_last_x       = x;
        tl_last_n_in    = n_in;
        tl_last_sig     = sig;
        tl_last_scale_x = scale_x;
    }

    linear_q4k_decode_w4a8_pre(tl_x_q8, scale_x, tl_sum32, w_q4k, n_in, n_out, y);
}

void linear_q4k_decode_w4a8_pair(const float *x,
                                 const void  *w0_q4k,
                                 const void  *w1_q4k,
                                 size_t       n_in,
                                 size_t       n_out0,
                                 size_t       n_out1,
                                 float       *y0,
                                 float       *y1) {
    if (x == NULL || w0_q4k == NULL || w1_q4k == NULL || y0 == NULL || y1 == NULL || n_in == 0 ||
        n_out0 == 0 || n_out1 == 0) {
        return;
    }

    static _Thread_local int8_t  *tl_x_q8     = NULL;
    static _Thread_local int32_t *tl_sum32    = NULL;
    static _Thread_local size_t   tl_cap_n_in = 0;

    if (n_in > tl_cap_n_in) {
        safe_free((void **) &tl_x_q8);
        safe_free((void **) &tl_sum32);
        tl_x_q8  = heap_alloc_array_aligned(int8_t, n_in);
        tl_sum32 = heap_alloc_array_aligned(int32_t, n_in / 32);
        if (tl_x_q8 == NULL || tl_sum32 == NULL) {
            tl_cap_n_in = 0;
            return;
        }
        tl_cap_n_in = n_in;
    }

    const float                 scale_x = quantize_x_for_q4k(x, n_in, tl_x_q8, tl_sum32);
    const struct q4k_decode_ctx c0      = {
            .w                = (const struct block_q4_K_t *) w0_q4k,
            .x_q8             = tl_x_q8,
            .sum32            = tl_sum32,
            .scale_x          = scale_x,
            .y                = y0,
            .n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS,
            .n_out            = n_out0,
    };
    const struct q4k_decode_ctx c1 = {
            .w                = (const struct block_q4_K_t *) w1_q4k,
            .x_q8             = tl_x_q8,
            .sum32            = tl_sum32,
            .scale_x          = scale_x,
            .y                = y1,
            .n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS,
            .n_out            = n_out1,
    };
    const size_t n_out_max = n_out0 > n_out1 ? n_out0 : n_out1;

    if (q4k_pp_enabled()) {
#if defined(_OPENMP)
        if (!omp_in_parallel())
#endif
        {
            const struct q4k_pair_ctx pc = {.c0 = &c0, .c1 = &c1};
            geist_pp_parallel_for(n_out_max, q4k_pp_pair_row, (void *) &pc);
            return;
        }
    }

#if defined(_OPENMP)
    if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t n = 0; n < n_out_max; n++) {
            if (n < n_out0)
                q4k_decode_one_row(n, &c0);
            if (n < n_out1)
                q4k_decode_one_row(n, &c1);
        }
    } else if (n_out_max >= 4096) {
#pragma omp parallel for schedule(static)
        for (size_t n = 0; n < n_out_max; n++) {
            if (n < n_out0)
                q4k_decode_one_row(n, &c0);
            if (n < n_out1)
                q4k_decode_one_row(n, &c1);
        }
    } else {
#pragma omp parallel for schedule(dynamic, 4)
        for (size_t n = 0; n < n_out_max; n++) {
            if (n < n_out0)
                q4k_decode_one_row(n, &c0);
            if (n < n_out1)
                q4k_decode_one_row(n, &c1);
        }
    }
#else
    for (size_t n = 0; n < n_out_max; n++) {
        if (n < n_out0)
            q4k_decode_one_row(n, &c0);
        if (n < n_out1)
            q4k_decode_one_row(n, &c1);
    }
#endif
}

#if defined(__ARM_NEON)
/* Vectorized 6-bit Q4_K scale/min unpack into scales[8]/mins[8]; bit-identical
 * to get_scale_min_k4 ×8 (verified over 2M random inputs). */
static inline void q4k_unpack_scales_mins(const uint8_t *s, uint8_t scales[8], uint8_t mins[8]) {
    uint32_t a32, b32, c32;
    memcpy(&a32, s + 0, 4);
    memcpy(&b32, s + 4, 4);
    memcpy(&c32, s + 8, 4);
    const uint8x8_t va  = vreinterpret_u8_u32(vdup_n_u32(a32));
    const uint8x8_t vb  = vreinterpret_u8_u32(vdup_n_u32(b32));
    const uint8x8_t vc  = vreinterpret_u8_u32(vdup_n_u32(c32));
    const uint8x8_t m3f = vdup_n_u8(0x3F), m0f = vdup_n_u8(0x0F);
    const uint8x8_t dlo = vand_u8(va, m3f);
    const uint8x8_t dhi = vorr_u8(vand_u8(vc, m0f), vshl_n_u8(vshr_n_u8(va, 6), 4));
    const uint8x8_t mlo = vand_u8(vb, m3f);
    const uint8x8_t mhi = vorr_u8(vshr_n_u8(vc, 4), vshl_n_u8(vshr_n_u8(vb, 6), 4));
    const uint8x8_t sel = vreinterpret_u8_u64(vcreate_u64(0xFFFFFFFF00000000ULL));
    vst1_u8(scales, vbsl_u8(sel, dhi, dlo));
    vst1_u8(mins, vbsl_u8(sel, mhi, mlo));
}
#endif

void linear_q4k_w4a8_prefill_pre(const int8_t  *x_q8,
                                 const float   *scale_x,
                                 const int32_t *sum32,
                                 size_t         m,
                                 const void    *w_q4k,
                                 size_t         n_in,
                                 size_t         n_out,
                                 float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    const struct block_q4_K_t *w                = (const struct block_q4_K_t *) w_q4k;
    const size_t               n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t               n_chunks         = n_in / 32;

    const uint8x16_t MASK0F = vdupq_n_u8(0x0F);
#define MT 2

/* Loop-reordered blocked GEMM: tile output rows into NC-row panels; per
 * panel, loop blocks OUTER and rows INNER so each activation block (m×256,
 * ~8KB at m=32) is loaded once and REUSED across all NC rows — it stays
 * L1-resident, eliminating the per-output-row eviction that caused the ~40%
 * backend-idle / 3.2B L1-refills. Partials accumulate in a small L1 ytile
 * (m×NC, ≤32KB); scale_x is applied once at panel end. Bit-identical to the
 * row-major form (scale_x factored out of the block sum). */
#define NC 64
    const size_t n_panels = (n_out + (size_t) NC - 1) / (size_t) NC;

    /* Activation packing (option B, §10.7): default ON; GEIST_Q4K_PACK_ACT=0
     * disables (for A/B). Reorder the activation ONCE into a contiguous
     * block-major panel — packed[((b*4+k)*m + t)*64 + e] — so the inner loop
     * streams sequentially instead of gathering at token-stride n_in. Across
     * the n_panels re-reads this turns strided L2 gathers into prefetchable
     * streams: L1-dcache miss ~2.4% -> ~1% (≈OpenBLAS/llama). Bit-identical.
     * Scratch is a reused thread-local high-water buffer (the kernel runs on
     * the single layer-loop thread; its omp panels only READ the buffer, so no
     * race). Skipped at m<2 where the per-token access is already sequential. */
    static int pack_act = -1;
    if (pack_act < 0) {
        const char *e = getenv("GEIST_Q4K_PACK_ACT");
        pack_act      = (e != NULL && e[0] == '0') ? 0 : 1;
    }
    int8_t *packed = NULL;
    if (pack_act && m >= 2) {
        static _Thread_local int8_t *pack_tl  = NULL;
        static _Thread_local size_t  pack_cap = 0;
        const size_t                 need     = m * n_in;
        if (need > pack_cap) {
            /* Route through heap.h (AGENT.md): the buffer is fully repopulated
             * by the packing loop below before any read, so dropping the old
             * contents on grow is safe. */
            safe_free((void **) &pack_tl);
            pack_tl  = heap_alloc_array_aligned(int8_t, need);
            pack_cap = (pack_tl != NULL) ? need : 0;
        }
        if (pack_cap >= need) {
            packed = pack_tl;
            for (size_t b = 0; b < n_blocks_per_row; b++) {
                for (int k = 0; k < 4; k++) {
                    int8_t       *dst = packed + (size_t) (b * 4 + (size_t) k) * m * 64;
                    const int8_t *src = x_q8 + b * Q4_K_BLOCK_ELEMS + (size_t) k * 64;
                    for (size_t t = 0; t < m; t++)
                        memcpy(dst + t * 64, src + t * n_in, 64);
                }
            }
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (size_t p = 0; p < n_panels; p++) {
        const size_t nc0 = p * (size_t) NC;
        const size_t nc  = (n_out - nc0 < (size_t) NC) ? (n_out - nc0) : (size_t) NC;
        float        ytile[GEIST_QUANT_M_CAP * NC] __attribute__((aligned(16)));
        for (size_t i = 0; i < m; i++)
            for (size_t r = 0; r < nc; r++)
                ytile[i * (size_t) NC + r] = 0.0f;

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            for (size_t r = 0; r < nc; r++) {
                const struct block_q4_K_t *blk = w + (nc0 + r) * n_blocks_per_row + b;
                if (r + 1 < nc)
                    __builtin_prefetch(blk + n_blocks_per_row, 0, 0);
                const float d_blk = fp16_to_fp32(blk->d), dmin_blk = fp16_to_fp32(blk->dmin);
                uint8_t     scales[8], mins[8];
                q4k_unpack_scales_mins(blk->scales, scales, mins);
                /* MT=2 inner with transient per-k weight decode. MT=2 (not 4)
                 * cuts register pressure — measured +3% over MT=4 on Pi 5. Hand-
                 * asm and keeping scales register-resident (vmlaq_laneq) were both
                 * tried and lost: gcc's -O3 scheduling + register allocation here
                 * is near-optimal (the scale stack-reloads it emits are cheap L1
                 * hits hidden in the SDOT latency shadow, not real waste).
                 * Bit-identical (integer SDOT accumulation is order-independent). */
                size_t i = 0;
                for (; i + MT <= m; i += MT) {
                    int32x4_t acc[MT];
                    for (int t = 0; t < MT; t++)
                        acc[t] = vdupq_n_s32(0);
                    for (int k = 0; k < 4; k++) {
                        const uint8x16_t qv0  = vld1q_u8(blk->qs + (size_t) k * 32);
                        const uint8x16_t qv1  = vld1q_u8(blk->qs + (size_t) k * 32 + 16);
                        const int8x16_t  wlo0 = vreinterpretq_s8_u8(vandq_u8(qv0, MASK0F));
                        const int8x16_t  wlo1 = vreinterpretq_s8_u8(vandq_u8(qv1, MASK0F));
                        const int8x16_t  whi0 = vreinterpretq_s8_u8(vshrq_n_u8(qv0, 4));
                        const int8x16_t  whi1 = vreinterpretq_s8_u8(vshrq_n_u8(qv1, 4));
                        const int32_t    s0l = scales[2 * k], s0h = scales[2 * k + 1];
                        const int8_t    *xk_base;
                        size_t           xk_stride;
                        if (packed) {
                            xk_base   = packed + (size_t) (b * 4 + (size_t) k) * m * 64;
                            xk_stride = 64;
                        } else {
                            xk_base   = x_q8 + b * Q4_K_BLOCK_ELEMS + (size_t) k * 64;
                            xk_stride = n_in;
                        }
                        for (int t = 0; t < MT; t++) {
                            const int8_t   *xl   = xk_base + (i + (size_t) t) * xk_stride;
                            const int8x16_t vxl0 = vld1q_s8(xl), vxl1 = vld1q_s8(xl + 16);
                            const int8x16_t vxh0 = vld1q_s8(xl + 32), vxh1 = vld1q_s8(xl + 48);
                            int32x4_t       dlo = vdotq_s32(vdupq_n_s32(0), wlo0, vxl0);
                            dlo                 = vdotq_s32(dlo, wlo1, vxl1);
                            int32x4_t dhi       = vdotq_s32(vdupq_n_s32(0), whi0, vxh0);
                            dhi                 = vdotq_s32(dhi, whi1, vxh1);
                            acc[t]              = vmlaq_n_s32(acc[t], dlo, s0l);
                            acc[t]              = vmlaq_n_s32(acc[t], dhi, s0h);
                        }
                    }
                    for (int t = 0; t < MT; t++) {
                        const size_t   ii = i + (size_t) t;
                        const int32_t *sp = sum32 + ii * n_chunks + b * 8;
                        int32_t        mc = 0;
                        for (int j = 0; j < 8; j++)
                            mc += (int32_t) mins[j] * sp[j];
                        ytile[ii * (size_t) NC + r] +=
                                d_blk * (float) vaddvq_s32(acc[t]) - dmin_blk * (float) mc;
                    }
                }
                for (; i < m; i++) {
                    int32x4_t acc = vdupq_n_s32(0);
                    for (int k = 0; k < 4; k++) {
                        const uint8x16_t qv0  = vld1q_u8(blk->qs + (size_t) k * 32);
                        const uint8x16_t qv1  = vld1q_u8(blk->qs + (size_t) k * 32 + 16);
                        const int8x16_t  wlo0 = vreinterpretq_s8_u8(vandq_u8(qv0, MASK0F));
                        const int8x16_t  wlo1 = vreinterpretq_s8_u8(vandq_u8(qv1, MASK0F));
                        const int8x16_t  whi0 = vreinterpretq_s8_u8(vshrq_n_u8(qv0, 4));
                        const int8x16_t  whi1 = vreinterpretq_s8_u8(vshrq_n_u8(qv1, 4));
                        const int32_t    s0l = scales[2 * k], s0h = scales[2 * k + 1];
                        const int8_t    *xk_base;
                        size_t           xk_stride;
                        if (packed) {
                            xk_base   = packed + (size_t) (b * 4 + (size_t) k) * m * 64;
                            xk_stride = 64;
                        } else {
                            xk_base   = x_q8 + b * Q4_K_BLOCK_ELEMS + (size_t) k * 64;
                            xk_stride = n_in;
                        }
                        const int8_t   *xl   = xk_base + i * xk_stride;
                        const int8x16_t vxl0 = vld1q_s8(xl), vxl1 = vld1q_s8(xl + 16);
                        const int8x16_t vxh0 = vld1q_s8(xl + 32), vxh1 = vld1q_s8(xl + 48);
                        int32x4_t       dlo = vdotq_s32(vdupq_n_s32(0), wlo0, vxl0);
                        dlo                 = vdotq_s32(dlo, wlo1, vxl1);
                        int32x4_t dhi       = vdotq_s32(vdupq_n_s32(0), whi0, vxh0);
                        dhi                 = vdotq_s32(dhi, whi1, vxh1);
                        acc                 = vmlaq_n_s32(acc, dlo, s0l);
                        acc                 = vmlaq_n_s32(acc, dhi, s0h);
                    }
                    const int32_t *sp = sum32 + i * n_chunks + b * 8;
                    int32_t        mc = 0;
                    for (int j = 0; j < 8; j++)
                        mc += (int32_t) mins[j] * sp[j];
                    ytile[i * (size_t) NC + r] +=
                            d_blk * (float) vaddvq_s32(acc) - dmin_blk * (float) mc;
                }
            }
        }
        for (size_t i = 0; i < m; i++)
            for (size_t r = 0; r < nc; r++)
                y[i * n_out + (nc0 + r)] = ytile[i * (size_t) NC + r] * scale_x[i];
    }
#undef NC
#undef MT
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) w_q4k;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q4k_w4a8_prefill_pre: NEON required\n");
#endif
}

void linear_q4k_w4a8_prefill_predecoded(const int8_t  *x_q8,
                                        const float   *scale_x,
                                        const int32_t *sum32,
                                        size_t         m,
                                        const void    *packed,
                                        size_t         n_in,
                                        size_t         n_out,
                                        float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q4k_predecode_valid(packed, n_in, n_out))
        return;
    const struct q4k_predecode_block *w                = q4k_predecode_blocks(packed);
    const size_t                      n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t                      n_chunks         = n_in / 32;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct q4k_predecode_block *row = w + n * n_blocks_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        float   accs[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        int32_t sumi[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        int32_t mins_corr[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        for (size_t i = 0; i < m; i++)
            accs[i] = 0.0f;

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct q4k_predecode_block *blk = &row[b];
            if (b + 2 < n_blocks_per_row)
                __builtin_prefetch(&row[b + 2], 0, 0);

            for (size_t i = 0; i < m; i++) {
                sumi[i]      = 0;
                mins_corr[i] = 0;
            }

            for (int is = 0; is < 8; is++) {
                const int8_t *q       = blk->qs + is * 32;
                const size_t  xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                const size_t  sum_idx = xb_off / 32;
                const int32_t sc      = (int32_t) blk->scales[is];
                const int32_t mn      = (int32_t) blk->mins[is];

                const int8x16_t q0 = vld1q_s8(q + 0);
                const int8x16_t q1 = vld1q_s8(q + 16);
                for (size_t i = 0; i < m; i++) {
                    const int8_t *xb  = x_q8 + i * n_in + xb_off;
                    const int32_t dot = dot16_i8(xb + 0, q0) + dot16_i8(xb + 16, q1);
                    sumi[i] += sc * dot;
                    mins_corr[i] += mn * sum32[i * n_chunks + sum_idx];
                }
            }

            for (size_t i = 0; i < m; i++) {
                accs[i] +=
                        scale_x[i] * (blk->d * (float) sumi[i] - blk->dmin * (float) mins_corr[i]);
            }
        }
        for (size_t i = 0; i < m; i++)
            y[i * n_out + n] = accs[i];
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q4k_w4a8_prefill_predecoded: NEON required\n");
#endif
}

void linear_q4k_w4a8_prefill_predecoded_mtile4(const int8_t  *x_q8,
                                               const float   *scale_x,
                                               const int32_t *sum32,
                                               size_t         m,
                                               const void    *packed,
                                               size_t         n_in,
                                               size_t         n_out,
                                               float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q4k_predecode_valid(packed, n_in, n_out))
        return;
    if (m < 4) {
        linear_q4k_w4a8_prefill_predecoded(x_q8, scale_x, sum32, m, packed, n_in, n_out, y);
        return;
    }

    const struct q4k_predecode_block *w                = q4k_predecode_blocks(packed);
    const size_t                      n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t                      n_chunks         = n_in / 32;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct q4k_predecode_block *row = w + n * n_blocks_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        size_t mt = 0;
        for (; mt + 4 <= m; mt += 4) {
            const float sx0  = scale_x[mt + 0];
            const float sx1  = scale_x[mt + 1];
            const float sx2  = scale_x[mt + 2];
            const float sx3  = scale_x[mt + 3];
            float       acc0 = 0.0f;
            float       acc1 = 0.0f;
            float       acc2 = 0.0f;
            float       acc3 = 0.0f;

            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blk = &row[b];
                if (b + 2 < n_blocks_per_row)
                    __builtin_prefetch(&row[b + 2], 0, 0);

                int32_t sumi0 = 0, sumi1 = 0, sumi2 = 0, sumi3 = 0;
                int32_t min0 = 0, min1 = 0, min2 = 0, min3 = 0;

                for (int is = 0; is < 8; is++) {
                    const int8_t   *q       = blk->qs + is * 32;
                    const int8x16_t q0      = vld1q_s8(q + 0);
                    const int8x16_t q1      = vld1q_s8(q + 16);
                    const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t    sum_idx = xb_off / 32;
                    const int32_t   sc      = (int32_t) blk->scales[is];
                    const int32_t   mn      = (int32_t) blk->mins[is];

                    const int8_t *xb0 = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t *xb1 = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t *xb2 = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t *xb3 = x_q8 + (mt + 3) * n_in + xb_off;

                    const int32_t dot0 = dot16_i8(xb0 + 0, q0) + dot16_i8(xb0 + 16, q1);
                    const int32_t dot1 = dot16_i8(xb1 + 0, q0) + dot16_i8(xb1 + 16, q1);
                    const int32_t dot2 = dot16_i8(xb2 + 0, q0) + dot16_i8(xb2 + 16, q1);
                    const int32_t dot3 = dot16_i8(xb3 + 0, q0) + dot16_i8(xb3 + 16, q1);
                    sumi0 += sc * dot0;
                    sumi1 += sc * dot1;
                    sumi2 += sc * dot2;
                    sumi3 += sc * dot3;

                    min0 += mn * sum32[(mt + 0) * n_chunks + sum_idx];
                    min1 += mn * sum32[(mt + 1) * n_chunks + sum_idx];
                    min2 += mn * sum32[(mt + 2) * n_chunks + sum_idx];
                    min3 += mn * sum32[(mt + 3) * n_chunks + sum_idx];
                }

                const float d    = blk->d;
                const float dmin = blk->dmin;
                acc0 += sx0 * (d * (float) sumi0 - dmin * (float) min0);
                acc1 += sx1 * (d * (float) sumi1 - dmin * (float) min1);
                acc2 += sx2 * (d * (float) sumi2 - dmin * (float) min2);
                acc3 += sx3 * (d * (float) sumi3 - dmin * (float) min3);
            }

            y[(mt + 0) * n_out + n] = acc0;
            y[(mt + 1) * n_out + n] = acc1;
            y[(mt + 2) * n_out + n] = acc2;
            y[(mt + 3) * n_out + n] = acc3;
        }

        for (; mt < m; mt++) {
            const float sx  = scale_x[mt];
            float       acc = 0.0f;
            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blk       = &row[b];
                int32_t                           sumi      = 0;
                int32_t                           mins_corr = 0;
                for (int is = 0; is < 8; is++) {
                    const int8_t   *q       = blk->qs + is * 32;
                    const int8x16_t q0      = vld1q_s8(q + 0);
                    const int8x16_t q1      = vld1q_s8(q + 16);
                    const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t    sum_idx = xb_off / 32;
                    const int8_t   *xb      = x_q8 + mt * n_in + xb_off;
                    const int32_t   dot     = dot16_i8(xb + 0, q0) + dot16_i8(xb + 16, q1);
                    sumi += (int32_t) blk->scales[is] * dot;
                    mins_corr += (int32_t) blk->mins[is] * sum32[mt * n_chunks + sum_idx];
                }
                acc += sx * (blk->d * (float) sumi - blk->dmin * (float) mins_corr);
            }
            y[mt * n_out + n] = acc;
        }
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q4k_w4a8_prefill_predecoded_mtile4: NEON required\n");
#endif
}

/* mtile8 = mtile4 with the inner M-tile widened to 8 rows. Same buffer
 * contracts (x_q8/scale_x/sum32/packed/y), no new workspace. Falls back
 * to mtile4 when m < 8. */
void linear_q4k_w4a8_prefill_predecoded_mtile8(const int8_t  *x_q8,
                                               const float   *scale_x,
                                               const int32_t *sum32,
                                               size_t         m,
                                               const void    *packed,
                                               size_t         n_in,
                                               size_t         n_out,
                                               float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q4k_predecode_valid(packed, n_in, n_out))
        return;
    if (m < 8) {
        linear_q4k_w4a8_prefill_predecoded_mtile4(x_q8, scale_x, sum32, m, packed, n_in, n_out, y);
        return;
    }

    const struct q4k_predecode_block *w                = q4k_predecode_blocks(packed);
    const size_t                      n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t                      n_chunks         = n_in / 32;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct q4k_predecode_block *row = w + n * n_blocks_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        size_t mt = 0;
        for (; mt + 8 <= m; mt += 8) {
            const float sx0  = scale_x[mt + 0];
            const float sx1  = scale_x[mt + 1];
            const float sx2  = scale_x[mt + 2];
            const float sx3  = scale_x[mt + 3];
            const float sx4  = scale_x[mt + 4];
            const float sx5  = scale_x[mt + 5];
            const float sx6  = scale_x[mt + 6];
            const float sx7  = scale_x[mt + 7];
            float       acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float       acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blk = &row[b];
                if (b + 2 < n_blocks_per_row)
                    __builtin_prefetch(&row[b + 2], 0, 0);

                int32_t sumi0 = 0, sumi1 = 0, sumi2 = 0, sumi3 = 0;
                int32_t sumi4 = 0, sumi5 = 0, sumi6 = 0, sumi7 = 0;
                int32_t min0 = 0, min1 = 0, min2 = 0, min3 = 0;
                int32_t min4 = 0, min5 = 0, min6 = 0, min7 = 0;

                for (int is = 0; is < 8; is++) {
                    const int8_t   *q       = blk->qs + is * 32;
                    const int8x16_t q0      = vld1q_s8(q + 0);
                    const int8x16_t q1      = vld1q_s8(q + 16);
                    const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t    sum_idx = xb_off / 32;
                    const int32_t   sc      = (int32_t) blk->scales[is];
                    const int32_t   mn      = (int32_t) blk->mins[is];

                    const int8_t *xb0 = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t *xb1 = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t *xb2 = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t *xb3 = x_q8 + (mt + 3) * n_in + xb_off;
                    const int8_t *xb4 = x_q8 + (mt + 4) * n_in + xb_off;
                    const int8_t *xb5 = x_q8 + (mt + 5) * n_in + xb_off;
                    const int8_t *xb6 = x_q8 + (mt + 6) * n_in + xb_off;
                    const int8_t *xb7 = x_q8 + (mt + 7) * n_in + xb_off;

                    sumi0 += sc * (dot16_i8(xb0 + 0, q0) + dot16_i8(xb0 + 16, q1));
                    sumi1 += sc * (dot16_i8(xb1 + 0, q0) + dot16_i8(xb1 + 16, q1));
                    sumi2 += sc * (dot16_i8(xb2 + 0, q0) + dot16_i8(xb2 + 16, q1));
                    sumi3 += sc * (dot16_i8(xb3 + 0, q0) + dot16_i8(xb3 + 16, q1));
                    sumi4 += sc * (dot16_i8(xb4 + 0, q0) + dot16_i8(xb4 + 16, q1));
                    sumi5 += sc * (dot16_i8(xb5 + 0, q0) + dot16_i8(xb5 + 16, q1));
                    sumi6 += sc * (dot16_i8(xb6 + 0, q0) + dot16_i8(xb6 + 16, q1));
                    sumi7 += sc * (dot16_i8(xb7 + 0, q0) + dot16_i8(xb7 + 16, q1));

                    min0 += mn * sum32[(mt + 0) * n_chunks + sum_idx];
                    min1 += mn * sum32[(mt + 1) * n_chunks + sum_idx];
                    min2 += mn * sum32[(mt + 2) * n_chunks + sum_idx];
                    min3 += mn * sum32[(mt + 3) * n_chunks + sum_idx];
                    min4 += mn * sum32[(mt + 4) * n_chunks + sum_idx];
                    min5 += mn * sum32[(mt + 5) * n_chunks + sum_idx];
                    min6 += mn * sum32[(mt + 6) * n_chunks + sum_idx];
                    min7 += mn * sum32[(mt + 7) * n_chunks + sum_idx];
                }

                const float d    = blk->d;
                const float dmin = blk->dmin;
                acc0 += sx0 * (d * (float) sumi0 - dmin * (float) min0);
                acc1 += sx1 * (d * (float) sumi1 - dmin * (float) min1);
                acc2 += sx2 * (d * (float) sumi2 - dmin * (float) min2);
                acc3 += sx3 * (d * (float) sumi3 - dmin * (float) min3);
                acc4 += sx4 * (d * (float) sumi4 - dmin * (float) min4);
                acc5 += sx5 * (d * (float) sumi5 - dmin * (float) min5);
                acc6 += sx6 * (d * (float) sumi6 - dmin * (float) min6);
                acc7 += sx7 * (d * (float) sumi7 - dmin * (float) min7);
            }

            y[(mt + 0) * n_out + n] = acc0;
            y[(mt + 1) * n_out + n] = acc1;
            y[(mt + 2) * n_out + n] = acc2;
            y[(mt + 3) * n_out + n] = acc3;
            y[(mt + 4) * n_out + n] = acc4;
            y[(mt + 5) * n_out + n] = acc5;
            y[(mt + 6) * n_out + n] = acc6;
            y[(mt + 7) * n_out + n] = acc7;
        }

        /* Residual: 4-row chunk for m % 8 ≥ 4, mirrors mtile4 inner loop. */
        for (; mt + 4 <= m; mt += 4) {
            const float sx0  = scale_x[mt + 0];
            const float sx1  = scale_x[mt + 1];
            const float sx2  = scale_x[mt + 2];
            const float sx3  = scale_x[mt + 3];
            float       acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blk   = &row[b];
                int32_t                           sumi0 = 0, sumi1 = 0, sumi2 = 0, sumi3 = 0;
                int32_t                           min0 = 0, min1 = 0, min2 = 0, min3 = 0;
                for (int is = 0; is < 8; is++) {
                    const int8_t   *q       = blk->qs + is * 32;
                    const int8x16_t q0      = vld1q_s8(q + 0);
                    const int8x16_t q1      = vld1q_s8(q + 16);
                    const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t    sum_idx = xb_off / 32;
                    const int32_t   sc      = (int32_t) blk->scales[is];
                    const int32_t   mn      = (int32_t) blk->mins[is];
                    const int8_t   *xb0     = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t   *xb1     = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t   *xb2     = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t   *xb3     = x_q8 + (mt + 3) * n_in + xb_off;
                    sumi0 += sc * (dot16_i8(xb0 + 0, q0) + dot16_i8(xb0 + 16, q1));
                    sumi1 += sc * (dot16_i8(xb1 + 0, q0) + dot16_i8(xb1 + 16, q1));
                    sumi2 += sc * (dot16_i8(xb2 + 0, q0) + dot16_i8(xb2 + 16, q1));
                    sumi3 += sc * (dot16_i8(xb3 + 0, q0) + dot16_i8(xb3 + 16, q1));
                    min0 += mn * sum32[(mt + 0) * n_chunks + sum_idx];
                    min1 += mn * sum32[(mt + 1) * n_chunks + sum_idx];
                    min2 += mn * sum32[(mt + 2) * n_chunks + sum_idx];
                    min3 += mn * sum32[(mt + 3) * n_chunks + sum_idx];
                }
                const float d    = blk->d;
                const float dmin = blk->dmin;
                acc0 += sx0 * (d * (float) sumi0 - dmin * (float) min0);
                acc1 += sx1 * (d * (float) sumi1 - dmin * (float) min1);
                acc2 += sx2 * (d * (float) sumi2 - dmin * (float) min2);
                acc3 += sx3 * (d * (float) sumi3 - dmin * (float) min3);
            }
            y[(mt + 0) * n_out + n] = acc0;
            y[(mt + 1) * n_out + n] = acc1;
            y[(mt + 2) * n_out + n] = acc2;
            y[(mt + 3) * n_out + n] = acc3;
        }

        /* 1-row residual for m % 4. */
        for (; mt < m; mt++) {
            const float sx  = scale_x[mt];
            float       acc = 0.0f;
            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blk       = &row[b];
                int32_t                           sumi      = 0;
                int32_t                           mins_corr = 0;
                for (int is = 0; is < 8; is++) {
                    const int8_t   *q       = blk->qs + is * 32;
                    const int8x16_t q0      = vld1q_s8(q + 0);
                    const int8x16_t q1      = vld1q_s8(q + 16);
                    const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t    sum_idx = xb_off / 32;
                    const int8_t   *xb      = x_q8 + mt * n_in + xb_off;
                    const int32_t   dot     = dot16_i8(xb + 0, q0) + dot16_i8(xb + 16, q1);
                    sumi += (int32_t) blk->scales[is] * dot;
                    mins_corr += (int32_t) blk->mins[is] * sum32[mt * n_chunks + sum_idx];
                }
                acc += sx * (blk->d * (float) sumi - blk->dmin * (float) mins_corr);
            }
            y[mt * n_out + n] = acc;
        }
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q4k_w4a8_prefill_predecoded_mtile8: NEON required\n");
#endif
}

void linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4(const int8_t  *x_q8,
                                                      const float   *scale_x,
                                                      const int32_t *sum32,
                                                      size_t         m,
                                                      const void    *packed,
                                                      size_t         n_in,
                                                      size_t         n_out,
                                                      float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q4k_predecode_valid(packed, n_in, n_out))
        return;
    if (m < 4 || n_out < 4) {
        linear_q4k_w4a8_prefill_predecoded_mtile4(x_q8, scale_x, sum32, m, packed, n_in, n_out, y);
        return;
    }

    const struct q4k_predecode_block *w                = q4k_predecode_blocks(packed);
    const size_t                      n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t                      n_chunks         = n_in / 32;
    const size_t                      n_tile_end       = n_out & ~(size_t) 3;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t nt = 0; nt < n_tile_end; nt += 4) {
        const struct q4k_predecode_block *row0 = w + (nt + 0) * n_blocks_per_row;
        const struct q4k_predecode_block *row1 = w + (nt + 1) * n_blocks_per_row;
        const struct q4k_predecode_block *row2 = w + (nt + 2) * n_blocks_per_row;
        const struct q4k_predecode_block *row3 = w + (nt + 3) * n_blocks_per_row;

        size_t mt = 0;
        for (; mt + 4 <= m; mt += 4) {
            const float sx0 = scale_x[mt + 0];
            const float sx1 = scale_x[mt + 1];
            const float sx2 = scale_x[mt + 2];
            const float sx3 = scale_x[mt + 3];
            float       a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
            float       a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
            float       a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;
            float       a30 = 0.0f, a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;

            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blks[4] = {
                        &row0[b],
                        &row1[b],
                        &row2[b],
                        &row3[b],
                };
                int32_t sumi[4][4] = {{0}};
                int32_t minc[4][4] = {{0}};

                for (int is = 0; is < 8; is++) {
                    const size_t  xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t  sum_idx = xb_off / 32;
                    const int8_t *xb0     = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t *xb1     = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t *xb2     = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t *xb3     = x_q8 + (mt + 3) * n_in + xb_off;

                    for (int nr = 0; nr < 4; nr++) {
                        const struct q4k_predecode_block *blk = blks[nr];
                        const int8_t                     *q   = blk->qs + is * 32;
                        const int8x16_t                   q0  = vld1q_s8(q + 0);
                        const int8x16_t                   q1  = vld1q_s8(q + 16);
                        const int32_t                     sc  = (int32_t) blk->scales[is];
                        const int32_t                     mn  = (int32_t) blk->mins[is];
                        const int32_t dot0 = dot16_i8(xb0 + 0, q0) + dot16_i8(xb0 + 16, q1);
                        const int32_t dot1 = dot16_i8(xb1 + 0, q0) + dot16_i8(xb1 + 16, q1);
                        const int32_t dot2 = dot16_i8(xb2 + 0, q0) + dot16_i8(xb2 + 16, q1);
                        const int32_t dot3 = dot16_i8(xb3 + 0, q0) + dot16_i8(xb3 + 16, q1);
                        sumi[nr][0] += sc * dot0;
                        sumi[nr][1] += sc * dot1;
                        sumi[nr][2] += sc * dot2;
                        sumi[nr][3] += sc * dot3;
                        minc[nr][0] += mn * sum32[(mt + 0) * n_chunks + sum_idx];
                        minc[nr][1] += mn * sum32[(mt + 1) * n_chunks + sum_idx];
                        minc[nr][2] += mn * sum32[(mt + 2) * n_chunks + sum_idx];
                        minc[nr][3] += mn * sum32[(mt + 3) * n_chunks + sum_idx];
                    }
                }

#define Q4K_ACC_ROW(NR, A0, A1, A2, A3)                                           \
    do {                                                                          \
        const struct q4k_predecode_block *blk  = blks[(NR)];                      \
        const float                       d    = blk->d;                          \
        const float                       dmin = blk->dmin;                       \
        (A0) += sx0 * (d * (float) sumi[(NR)][0] - dmin * (float) minc[(NR)][0]); \
        (A1) += sx1 * (d * (float) sumi[(NR)][1] - dmin * (float) minc[(NR)][1]); \
        (A2) += sx2 * (d * (float) sumi[(NR)][2] - dmin * (float) minc[(NR)][2]); \
        (A3) += sx3 * (d * (float) sumi[(NR)][3] - dmin * (float) minc[(NR)][3]); \
    } while (0)
                Q4K_ACC_ROW(0, a00, a10, a20, a30);
                Q4K_ACC_ROW(1, a01, a11, a21, a31);
                Q4K_ACC_ROW(2, a02, a12, a22, a32);
                Q4K_ACC_ROW(3, a03, a13, a23, a33);
#undef Q4K_ACC_ROW
            }

            float *y0 = y + (mt + 0) * n_out + nt;
            float *y1 = y + (mt + 1) * n_out + nt;
            float *y2 = y + (mt + 2) * n_out + nt;
            float *y3 = y + (mt + 3) * n_out + nt;
            y0[0]     = a00;
            y0[1]     = a01;
            y0[2]     = a02;
            y0[3]     = a03;
            y1[0]     = a10;
            y1[1]     = a11;
            y1[2]     = a12;
            y1[3]     = a13;
            y2[0]     = a20;
            y2[1]     = a21;
            y2[2]     = a22;
            y2[3]     = a23;
            y3[0]     = a30;
            y3[1]     = a31;
            y3[2]     = a32;
            y3[3]     = a33;
        }
    }

    if (n_tile_end < n_out || (m & (size_t) 3) != 0) {
        linear_q4k_w4a8_prefill_predecoded_mtile4(x_q8, scale_x, sum32, m, packed, n_in, n_out, y);
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4: NEON required\n");
#endif
}

void linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed(const int8_t  *x_q8,
                                                             const float   *scale_x,
                                                             const int32_t *sum32,
                                                             size_t         m,
                                                             const void    *packed,
                                                             size_t         n_in,
                                                             size_t         n_out,
                                                             float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q4k_predecode_ntile4_valid(packed, n_in, n_out))
        return;
    const struct q4k_predecode_block *w                = q4k_predecode_ntile4_blocks(packed);
    const size_t                      n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t                      n_chunks         = n_in / 32;
    const size_t                      n_tiles          = (n_out + 3) / 4;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t nt = 0; nt < n_tiles; nt++) {
        const size_t                      valid_nr = (nt * 4 + 4 <= n_out) ? 4 : (n_out - nt * 4);
        const struct q4k_predecode_block *tile     = w + nt * n_blocks_per_row * 4;

        size_t mt = 0;
        for (; mt + 4 <= m; mt += 4) {
            const float sx0 = scale_x[mt + 0];
            const float sx1 = scale_x[mt + 1];
            const float sx2 = scale_x[mt + 2];
            const float sx3 = scale_x[mt + 3];
            float       a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
            float       a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
            float       a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;
            float       a30 = 0.0f, a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;

            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blks       = tile + b * 4;
                int32_t                           sumi[4][4] = {{0}};
                int32_t                           minc[4][4] = {{0}};

                for (int is = 0; is < 8; is++) {
                    const size_t  xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t  sum_idx = xb_off / 32;
                    const int8_t *xb0     = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t *xb1     = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t *xb2     = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t *xb3     = x_q8 + (mt + 3) * n_in + xb_off;

                    for (size_t nr = 0; nr < valid_nr; nr++) {
                        const struct q4k_predecode_block *blk = blks + nr;
                        const int8_t                     *q   = blk->qs + is * 32;
                        const int8x16_t                   q0  = vld1q_s8(q + 0);
                        const int8x16_t                   q1  = vld1q_s8(q + 16);
                        const int32_t                     sc  = (int32_t) blk->scales[is];
                        const int32_t                     mn  = (int32_t) blk->mins[is];
                        const int32_t dot0 = dot16_i8(xb0 + 0, q0) + dot16_i8(xb0 + 16, q1);
                        const int32_t dot1 = dot16_i8(xb1 + 0, q0) + dot16_i8(xb1 + 16, q1);
                        const int32_t dot2 = dot16_i8(xb2 + 0, q0) + dot16_i8(xb2 + 16, q1);
                        const int32_t dot3 = dot16_i8(xb3 + 0, q0) + dot16_i8(xb3 + 16, q1);
                        sumi[nr][0] += sc * dot0;
                        sumi[nr][1] += sc * dot1;
                        sumi[nr][2] += sc * dot2;
                        sumi[nr][3] += sc * dot3;
                        minc[nr][0] += mn * sum32[(mt + 0) * n_chunks + sum_idx];
                        minc[nr][1] += mn * sum32[(mt + 1) * n_chunks + sum_idx];
                        minc[nr][2] += mn * sum32[(mt + 2) * n_chunks + sum_idx];
                        minc[nr][3] += mn * sum32[(mt + 3) * n_chunks + sum_idx];
                    }
                }

#define Q4K_ACC_ROW_PACKED(NR, A0, A1, A2, A3)                                        \
    do {                                                                              \
        if ((NR) < valid_nr) {                                                        \
            const struct q4k_predecode_block *blk  = blks + (NR);                     \
            const float                       d    = blk->d;                          \
            const float                       dmin = blk->dmin;                       \
            (A0) += sx0 * (d * (float) sumi[(NR)][0] - dmin * (float) minc[(NR)][0]); \
            (A1) += sx1 * (d * (float) sumi[(NR)][1] - dmin * (float) minc[(NR)][1]); \
            (A2) += sx2 * (d * (float) sumi[(NR)][2] - dmin * (float) minc[(NR)][2]); \
            (A3) += sx3 * (d * (float) sumi[(NR)][3] - dmin * (float) minc[(NR)][3]); \
        }                                                                             \
    } while (0)
                Q4K_ACC_ROW_PACKED(0, a00, a10, a20, a30);
                Q4K_ACC_ROW_PACKED(1, a01, a11, a21, a31);
                Q4K_ACC_ROW_PACKED(2, a02, a12, a22, a32);
                Q4K_ACC_ROW_PACKED(3, a03, a13, a23, a33);
#undef Q4K_ACC_ROW_PACKED
            }

            float *y0 = y + (mt + 0) * n_out + nt * 4;
            float *y1 = y + (mt + 1) * n_out + nt * 4;
            float *y2 = y + (mt + 2) * n_out + nt * 4;
            float *y3 = y + (mt + 3) * n_out + nt * 4;
            if (valid_nr > 0) {
                y0[0] = a00;
                y1[0] = a10;
                y2[0] = a20;
                y3[0] = a30;
            }
            if (valid_nr > 1) {
                y0[1] = a01;
                y1[1] = a11;
                y2[1] = a21;
                y3[1] = a31;
            }
            if (valid_nr > 2) {
                y0[2] = a02;
                y1[2] = a12;
                y2[2] = a22;
                y3[2] = a32;
            }
            if (valid_nr > 3) {
                y0[3] = a03;
                y1[3] = a13;
                y2[3] = a23;
                y3[3] = a33;
            }
        }

        for (; mt < m; mt++) {
            for (size_t nr = 0; nr < valid_nr; nr++) {
                float acc = 0.0f;
                for (size_t b = 0; b < n_blocks_per_row; b++) {
                    const struct q4k_predecode_block *blk       = tile + b * 4 + nr;
                    int32_t                           sumi      = 0;
                    int32_t                           mins_corr = 0;
                    for (int is = 0; is < 8; is++) {
                        const int8_t   *q       = blk->qs + is * 32;
                        const int8x16_t q0      = vld1q_s8(q + 0);
                        const int8x16_t q1      = vld1q_s8(q + 16);
                        const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                        const size_t    sum_idx = xb_off / 32;
                        const int8_t   *xb      = x_q8 + mt * n_in + xb_off;
                        const int32_t   dot     = dot16_i8(xb + 0, q0) + dot16_i8(xb + 16, q1);
                        sumi += (int32_t) blk->scales[is] * dot;
                        mins_corr += (int32_t) blk->mins[is] * sum32[mt * n_chunks + sum_idx];
                    }
                    acc += scale_x[mt] * (blk->d * (float) sumi - blk->dmin * (float) mins_corr);
                }
                y[mt * n_out + nt * 4 + nr] = acc;
            }
        }
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed: NEON required\n");
#endif
}

/* mtile8_ntile4_packed = mtile4_ntile4_packed with the M-tile widened
 * to 8. Reuses the existing ntile4 packed format (no new pack). Each
 * inner (is, b) iteration now amortizes 4 weight loads across 8 input
 * rows = 32 dot products, vs 16 in the mtile4 variant. Closes Mac CPU
 * prefill gap vs llama.cpp's `ggml_gemm_q4_K_8x8_q8_K` to the extent
 * achievable without i8mm (M1 has no MATMUL_INT8). Falls back to
 * mtile4_ntile4_packed for m < 8. */
void linear_q4k_w4a8_prefill_predecoded_mtile8_ntile4_packed(const int8_t  *x_q8,
                                                             const float   *scale_x,
                                                             const int32_t *sum32,
                                                             size_t         m,
                                                             const void    *packed,
                                                             size_t         n_in,
                                                             size_t         n_out,
                                                             float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q4k_predecode_ntile4_valid(packed, n_in, n_out))
        return;
    if (m < 8) {
        linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed(
                x_q8, scale_x, sum32, m, packed, n_in, n_out, y);
        return;
    }
    const struct q4k_predecode_block *w                = q4k_predecode_ntile4_blocks(packed);
    const size_t                      n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t                      n_chunks         = n_in / 32;
    const size_t                      n_tiles          = (n_out + 3) / 4;

    /* Dynamic schedule with a chunk of 4 tiles (= 16 weight rows): lets
     * E-cores grab smaller slices when they fall behind P-cores at 8
     * threads on M-class. Static schedule blocks the team on E-core
     * completion at 8t (geist regresses 6t→8t while llama scales).
     * At 4-6t the contention is low enough that static was fine; this
     * dynamic chunk is also fine there because (n_tiles / chunk_size)
     * is much larger than nthreads. */
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t nt = 0; nt < n_tiles; nt++) {
        const size_t                      valid_nr = (nt * 4 + 4 <= n_out) ? 4 : (n_out - nt * 4);
        const struct q4k_predecode_block *tile     = w + nt * n_blocks_per_row * 4;

        size_t mt = 0;
        for (; mt + 8 <= m; mt += 8) {
            const float sx0 = scale_x[mt + 0], sx1 = scale_x[mt + 1];
            const float sx2 = scale_x[mt + 2], sx3 = scale_x[mt + 3];
            const float sx4 = scale_x[mt + 4], sx5 = scale_x[mt + 5];
            const float sx6 = scale_x[mt + 6], sx7 = scale_x[mt + 7];
            float       a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
            float       a20 = 0, a21 = 0, a22 = 0, a23 = 0, a30 = 0, a31 = 0, a32 = 0, a33 = 0;
            float       a40 = 0, a41 = 0, a42 = 0, a43 = 0, a50 = 0, a51 = 0, a52 = 0, a53 = 0;
            float       a60 = 0, a61 = 0, a62 = 0, a63 = 0, a70 = 0, a71 = 0, a72 = 0, a73 = 0;

            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blks       = tile + b * 4;
                int32_t                           sumi[4][8] = {{0}};
                int32_t                           minc[4][8] = {{0}};

                for (int is = 0; is < 8; is++) {
                    const size_t  xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t  sum_idx = xb_off / 32;
                    const int8_t *xb0     = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t *xb1     = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t *xb2     = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t *xb3     = x_q8 + (mt + 3) * n_in + xb_off;
                    const int8_t *xb4     = x_q8 + (mt + 4) * n_in + xb_off;
                    const int8_t *xb5     = x_q8 + (mt + 5) * n_in + xb_off;
                    const int8_t *xb6     = x_q8 + (mt + 6) * n_in + xb_off;
                    const int8_t *xb7     = x_q8 + (mt + 7) * n_in + xb_off;

                    for (size_t nr = 0; nr < valid_nr; nr++) {
                        const struct q4k_predecode_block *blk = blks + nr;
                        const int8_t                     *q   = blk->qs + is * 32;
                        const int8x16_t                   q0  = vld1q_s8(q + 0);
                        const int8x16_t                   q1  = vld1q_s8(q + 16);
                        const int32_t                     sc  = (int32_t) blk->scales[is];
                        const int32_t                     mn  = (int32_t) blk->mins[is];
                        const int32_t d0 = dot16_i8(xb0 + 0, q0) + dot16_i8(xb0 + 16, q1);
                        const int32_t d1 = dot16_i8(xb1 + 0, q0) + dot16_i8(xb1 + 16, q1);
                        const int32_t d2 = dot16_i8(xb2 + 0, q0) + dot16_i8(xb2 + 16, q1);
                        const int32_t d3 = dot16_i8(xb3 + 0, q0) + dot16_i8(xb3 + 16, q1);
                        const int32_t d4 = dot16_i8(xb4 + 0, q0) + dot16_i8(xb4 + 16, q1);
                        const int32_t d5 = dot16_i8(xb5 + 0, q0) + dot16_i8(xb5 + 16, q1);
                        const int32_t d6 = dot16_i8(xb6 + 0, q0) + dot16_i8(xb6 + 16, q1);
                        const int32_t d7 = dot16_i8(xb7 + 0, q0) + dot16_i8(xb7 + 16, q1);
                        sumi[nr][0] += sc * d0;
                        sumi[nr][1] += sc * d1;
                        sumi[nr][2] += sc * d2;
                        sumi[nr][3] += sc * d3;
                        sumi[nr][4] += sc * d4;
                        sumi[nr][5] += sc * d5;
                        sumi[nr][6] += sc * d6;
                        sumi[nr][7] += sc * d7;
                        minc[nr][0] += mn * sum32[(mt + 0) * n_chunks + sum_idx];
                        minc[nr][1] += mn * sum32[(mt + 1) * n_chunks + sum_idx];
                        minc[nr][2] += mn * sum32[(mt + 2) * n_chunks + sum_idx];
                        minc[nr][3] += mn * sum32[(mt + 3) * n_chunks + sum_idx];
                        minc[nr][4] += mn * sum32[(mt + 4) * n_chunks + sum_idx];
                        minc[nr][5] += mn * sum32[(mt + 5) * n_chunks + sum_idx];
                        minc[nr][6] += mn * sum32[(mt + 6) * n_chunks + sum_idx];
                        minc[nr][7] += mn * sum32[(mt + 7) * n_chunks + sum_idx];
                    }
                }

#define Q4K_ACC_8(NR, A0, A1, A2, A3, A4, A5, A6, A7)                                 \
    do {                                                                              \
        if ((NR) < valid_nr) {                                                        \
            const struct q4k_predecode_block *blk  = blks + (NR);                     \
            const float                       d    = blk->d;                          \
            const float                       dmin = blk->dmin;                       \
            (A0) += sx0 * (d * (float) sumi[(NR)][0] - dmin * (float) minc[(NR)][0]); \
            (A1) += sx1 * (d * (float) sumi[(NR)][1] - dmin * (float) minc[(NR)][1]); \
            (A2) += sx2 * (d * (float) sumi[(NR)][2] - dmin * (float) minc[(NR)][2]); \
            (A3) += sx3 * (d * (float) sumi[(NR)][3] - dmin * (float) minc[(NR)][3]); \
            (A4) += sx4 * (d * (float) sumi[(NR)][4] - dmin * (float) minc[(NR)][4]); \
            (A5) += sx5 * (d * (float) sumi[(NR)][5] - dmin * (float) minc[(NR)][5]); \
            (A6) += sx6 * (d * (float) sumi[(NR)][6] - dmin * (float) minc[(NR)][6]); \
            (A7) += sx7 * (d * (float) sumi[(NR)][7] - dmin * (float) minc[(NR)][7]); \
        }                                                                             \
    } while (0)
                Q4K_ACC_8(0, a00, a10, a20, a30, a40, a50, a60, a70);
                Q4K_ACC_8(1, a01, a11, a21, a31, a41, a51, a61, a71);
                Q4K_ACC_8(2, a02, a12, a22, a32, a42, a52, a62, a72);
                Q4K_ACC_8(3, a03, a13, a23, a33, a43, a53, a63, a73);
#undef Q4K_ACC_8
            }

            float *y0 = y + (mt + 0) * n_out + nt * 4;
            float *y1 = y + (mt + 1) * n_out + nt * 4;
            float *y2 = y + (mt + 2) * n_out + nt * 4;
            float *y3 = y + (mt + 3) * n_out + nt * 4;
            float *y4 = y + (mt + 4) * n_out + nt * 4;
            float *y5 = y + (mt + 5) * n_out + nt * 4;
            float *y6 = y + (mt + 6) * n_out + nt * 4;
            float *y7 = y + (mt + 7) * n_out + nt * 4;
            if (valid_nr > 0) {
                y0[0] = a00;
                y1[0] = a10;
                y2[0] = a20;
                y3[0] = a30;
                y4[0] = a40;
                y5[0] = a50;
                y6[0] = a60;
                y7[0] = a70;
            }
            if (valid_nr > 1) {
                y0[1] = a01;
                y1[1] = a11;
                y2[1] = a21;
                y3[1] = a31;
                y4[1] = a41;
                y5[1] = a51;
                y6[1] = a61;
                y7[1] = a71;
            }
            if (valid_nr > 2) {
                y0[2] = a02;
                y1[2] = a12;
                y2[2] = a22;
                y3[2] = a32;
                y4[2] = a42;
                y5[2] = a52;
                y6[2] = a62;
                y7[2] = a72;
            }
            if (valid_nr > 3) {
                y0[3] = a03;
                y1[3] = a13;
                y2[3] = a23;
                y3[3] = a33;
                y4[3] = a43;
                y5[3] = a53;
                y6[3] = a63;
                y7[3] = a73;
            }
        }

        /* 4-row residual: delegate to mtile4_ntile4_packed inline body. */
        for (; mt + 4 <= m; mt += 4) {
            const float sx0 = scale_x[mt + 0], sx1 = scale_x[mt + 1];
            const float sx2 = scale_x[mt + 2], sx3 = scale_x[mt + 3];
            float       a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
            float       a20 = 0, a21 = 0, a22 = 0, a23 = 0, a30 = 0, a31 = 0, a32 = 0, a33 = 0;
            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blks       = tile + b * 4;
                int32_t                           sumi[4][4] = {{0}};
                int32_t                           minc[4][4] = {{0}};
                for (int is = 0; is < 8; is++) {
                    const size_t  xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t  sum_idx = xb_off / 32;
                    const int8_t *xb0     = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t *xb1     = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t *xb2     = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t *xb3     = x_q8 + (mt + 3) * n_in + xb_off;
                    for (size_t nr = 0; nr < valid_nr; nr++) {
                        const struct q4k_predecode_block *blk = blks + nr;
                        const int8_t                     *q   = blk->qs + is * 32;
                        const int8x16_t                   q0  = vld1q_s8(q + 0);
                        const int8x16_t                   q1  = vld1q_s8(q + 16);
                        const int32_t                     sc  = (int32_t) blk->scales[is];
                        const int32_t                     mn  = (int32_t) blk->mins[is];
                        sumi[nr][0] += sc * (dot16_i8(xb0 + 0, q0) + dot16_i8(xb0 + 16, q1));
                        sumi[nr][1] += sc * (dot16_i8(xb1 + 0, q0) + dot16_i8(xb1 + 16, q1));
                        sumi[nr][2] += sc * (dot16_i8(xb2 + 0, q0) + dot16_i8(xb2 + 16, q1));
                        sumi[nr][3] += sc * (dot16_i8(xb3 + 0, q0) + dot16_i8(xb3 + 16, q1));
                        minc[nr][0] += mn * sum32[(mt + 0) * n_chunks + sum_idx];
                        minc[nr][1] += mn * sum32[(mt + 1) * n_chunks + sum_idx];
                        minc[nr][2] += mn * sum32[(mt + 2) * n_chunks + sum_idx];
                        minc[nr][3] += mn * sum32[(mt + 3) * n_chunks + sum_idx];
                    }
                }
#define Q4K_ACC_4(NR, A0, A1, A2, A3)                                                 \
    do {                                                                              \
        if ((NR) < valid_nr) {                                                        \
            const struct q4k_predecode_block *blk  = blks + (NR);                     \
            const float                       d    = blk->d;                          \
            const float                       dmin = blk->dmin;                       \
            (A0) += sx0 * (d * (float) sumi[(NR)][0] - dmin * (float) minc[(NR)][0]); \
            (A1) += sx1 * (d * (float) sumi[(NR)][1] - dmin * (float) minc[(NR)][1]); \
            (A2) += sx2 * (d * (float) sumi[(NR)][2] - dmin * (float) minc[(NR)][2]); \
            (A3) += sx3 * (d * (float) sumi[(NR)][3] - dmin * (float) minc[(NR)][3]); \
        }                                                                             \
    } while (0)
                Q4K_ACC_4(0, a00, a10, a20, a30);
                Q4K_ACC_4(1, a01, a11, a21, a31);
                Q4K_ACC_4(2, a02, a12, a22, a32);
                Q4K_ACC_4(3, a03, a13, a23, a33);
#undef Q4K_ACC_4
            }
            float *y0 = y + (mt + 0) * n_out + nt * 4;
            float *y1 = y + (mt + 1) * n_out + nt * 4;
            float *y2 = y + (mt + 2) * n_out + nt * 4;
            float *y3 = y + (mt + 3) * n_out + nt * 4;
            if (valid_nr > 0) {
                y0[0] = a00;
                y1[0] = a10;
                y2[0] = a20;
                y3[0] = a30;
            }
            if (valid_nr > 1) {
                y0[1] = a01;
                y1[1] = a11;
                y2[1] = a21;
                y3[1] = a31;
            }
            if (valid_nr > 2) {
                y0[2] = a02;
                y1[2] = a12;
                y2[2] = a22;
                y3[2] = a32;
            }
            if (valid_nr > 3) {
                y0[3] = a03;
                y1[3] = a13;
                y2[3] = a23;
                y3[3] = a33;
            }
        }

        /* 1-row residual */
        for (; mt < m; mt++) {
            for (size_t nr = 0; nr < valid_nr; nr++) {
                float acc = 0.0f;
                for (size_t b = 0; b < n_blocks_per_row; b++) {
                    const struct q4k_predecode_block *blk  = tile + b * 4 + nr;
                    int32_t                           sumi = 0, mins_corr = 0;
                    for (int is = 0; is < 8; is++) {
                        const int8_t   *q       = blk->qs + is * 32;
                        const int8x16_t q0      = vld1q_s8(q + 0);
                        const int8x16_t q1      = vld1q_s8(q + 16);
                        const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                        const size_t    sum_idx = xb_off / 32;
                        const int8_t   *xb      = x_q8 + mt * n_in + xb_off;
                        const int32_t   dot     = dot16_i8(xb + 0, q0) + dot16_i8(xb + 16, q1);
                        sumi += (int32_t) blk->scales[is] * dot;
                        mins_corr += (int32_t) blk->mins[is] * sum32[mt * n_chunks + sum_idx];
                    }
                    acc += scale_x[mt] * (blk->d * (float) sumi - blk->dmin * (float) mins_corr);
                }
                y[mt * n_out + nt * 4 + nr] = acc;
            }
        }
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q4k_w4a8_prefill_predecoded_mtile8_ntile4_packed: NEON required\n");
#endif
}

void linear_q4k_w4a8_prefill_pair_predecoded_mtile4_ntile4_packed(const int8_t  *x_q8,
                                                                  const float   *scale_x,
                                                                  const int32_t *sum32,
                                                                  size_t         m,
                                                                  const void    *packed0,
                                                                  const void    *packed1,
                                                                  size_t         n_in,
                                                                  size_t         n_out,
                                                                  float         *y0,
                                                                  float         *y1) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q4k_predecode_ntile4_valid(packed0, n_in, n_out) ||
        !q4k_predecode_ntile4_valid(packed1, n_in, n_out)) {
        return;
    }
    const size_t n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t n_chunks         = n_in / 32;
    const size_t n_tiles          = (n_out + 3) / 4;
    const size_t total_tiles      = n_tiles * 2;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t tt = 0; tt < total_tiles; tt++) {
        const bool                        second   = tt >= n_tiles;
        const size_t                      nt       = second ? (tt - n_tiles) : tt;
        const void                       *packed   = second ? packed1 : packed0;
        float                            *y        = second ? y1 : y0;
        const struct q4k_predecode_block *w        = q4k_predecode_ntile4_blocks(packed);
        const size_t                      valid_nr = (nt * 4 + 4 <= n_out) ? 4 : (n_out - nt * 4);
        const struct q4k_predecode_block *tile     = w + nt * n_blocks_per_row * 4;

        size_t mt = 0;
        for (; mt + 4 <= m; mt += 4) {
            const float sx0 = scale_x[mt + 0];
            const float sx1 = scale_x[mt + 1];
            const float sx2 = scale_x[mt + 2];
            const float sx3 = scale_x[mt + 3];
            float       a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
            float       a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
            float       a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;
            float       a30 = 0.0f, a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;

            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blks       = tile + b * 4;
                int32_t                           sumi[4][4] = {{0}};
                int32_t                           minc[4][4] = {{0}};

                for (int is = 0; is < 8; is++) {
                    const size_t  xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t  sum_idx = xb_off / 32;
                    const int8_t *xb0     = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t *xb1     = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t *xb2     = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t *xb3     = x_q8 + (mt + 3) * n_in + xb_off;

                    for (size_t nr = 0; nr < valid_nr; nr++) {
                        const struct q4k_predecode_block *blk = blks + nr;
                        const int8_t                     *q   = blk->qs + is * 32;
                        const int8x16_t                   q0  = vld1q_s8(q + 0);
                        const int8x16_t                   q1  = vld1q_s8(q + 16);
                        const int32_t                     sc  = (int32_t) blk->scales[is];
                        const int32_t                     mn  = (int32_t) blk->mins[is];
                        const int32_t dot0 = dot16_i8(xb0 + 0, q0) + dot16_i8(xb0 + 16, q1);
                        const int32_t dot1 = dot16_i8(xb1 + 0, q0) + dot16_i8(xb1 + 16, q1);
                        const int32_t dot2 = dot16_i8(xb2 + 0, q0) + dot16_i8(xb2 + 16, q1);
                        const int32_t dot3 = dot16_i8(xb3 + 0, q0) + dot16_i8(xb3 + 16, q1);
                        sumi[nr][0] += sc * dot0;
                        sumi[nr][1] += sc * dot1;
                        sumi[nr][2] += sc * dot2;
                        sumi[nr][3] += sc * dot3;
                        minc[nr][0] += mn * sum32[(mt + 0) * n_chunks + sum_idx];
                        minc[nr][1] += mn * sum32[(mt + 1) * n_chunks + sum_idx];
                        minc[nr][2] += mn * sum32[(mt + 2) * n_chunks + sum_idx];
                        minc[nr][3] += mn * sum32[(mt + 3) * n_chunks + sum_idx];
                    }
                }

#define Q4K_PAIR_ACC_ROW(NR, A0, A1, A2, A3)                                          \
    do {                                                                              \
        if ((NR) < valid_nr) {                                                        \
            const struct q4k_predecode_block *blk  = blks + (NR);                     \
            const float                       d    = blk->d;                          \
            const float                       dmin = blk->dmin;                       \
            (A0) += sx0 * (d * (float) sumi[(NR)][0] - dmin * (float) minc[(NR)][0]); \
            (A1) += sx1 * (d * (float) sumi[(NR)][1] - dmin * (float) minc[(NR)][1]); \
            (A2) += sx2 * (d * (float) sumi[(NR)][2] - dmin * (float) minc[(NR)][2]); \
            (A3) += sx3 * (d * (float) sumi[(NR)][3] - dmin * (float) minc[(NR)][3]); \
        }                                                                             \
    } while (0)
                Q4K_PAIR_ACC_ROW(0, a00, a10, a20, a30);
                Q4K_PAIR_ACC_ROW(1, a01, a11, a21, a31);
                Q4K_PAIR_ACC_ROW(2, a02, a12, a22, a32);
                Q4K_PAIR_ACC_ROW(3, a03, a13, a23, a33);
#undef Q4K_PAIR_ACC_ROW
            }

            float *yy0 = y + (mt + 0) * n_out + nt * 4;
            float *yy1 = y + (mt + 1) * n_out + nt * 4;
            float *yy2 = y + (mt + 2) * n_out + nt * 4;
            float *yy3 = y + (mt + 3) * n_out + nt * 4;
            if (valid_nr > 0) {
                yy0[0] = a00;
                yy1[0] = a10;
                yy2[0] = a20;
                yy3[0] = a30;
            }
            if (valid_nr > 1) {
                yy0[1] = a01;
                yy1[1] = a11;
                yy2[1] = a21;
                yy3[1] = a31;
            }
            if (valid_nr > 2) {
                yy0[2] = a02;
                yy1[2] = a12;
                yy2[2] = a22;
                yy3[2] = a32;
            }
            if (valid_nr > 3) {
                yy0[3] = a03;
                yy1[3] = a13;
                yy2[3] = a23;
                yy3[3] = a33;
            }
        }

        for (; mt < m; mt++) {
            for (size_t nr = 0; nr < valid_nr; nr++) {
                float acc = 0.0f;
                for (size_t b = 0; b < n_blocks_per_row; b++) {
                    const struct q4k_predecode_block *blk       = tile + b * 4 + nr;
                    int32_t                           sumi      = 0;
                    int32_t                           mins_corr = 0;
                    for (int is = 0; is < 8; is++) {
                        const int8_t   *q       = blk->qs + is * 32;
                        const int8x16_t q0      = vld1q_s8(q + 0);
                        const int8x16_t q1      = vld1q_s8(q + 16);
                        const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                        const size_t    sum_idx = xb_off / 32;
                        const int8_t   *xb      = x_q8 + mt * n_in + xb_off;
                        const int32_t   dot     = dot16_i8(xb + 0, q0) + dot16_i8(xb + 16, q1);
                        sumi += (int32_t) blk->scales[is] * dot;
                        mins_corr += (int32_t) blk->mins[is] * sum32[mt * n_chunks + sum_idx];
                    }
                    acc += scale_x[mt] * (blk->d * (float) sumi - blk->dmin * (float) mins_corr);
                }
                y[mt * n_out + nt * 4 + nr] = acc;
            }
        }
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) packed0;
    (void) packed1;
    (void) n_in;
    (void) n_out;
    (void) y0;
    (void) y1;
    fprintf(stderr,
            "linear_q4k_w4a8_prefill_pair_predecoded_mtile4_ntile4_packed: NEON required\n");
#endif
}

static inline void q4k_raw_tile4_mtile4_compute(const int8_t              *x_q8,
                                                const float               *scale_x,
                                                const int32_t             *sum32,
                                                size_t                     m,
                                                const struct block_q4_K_t *w,
                                                size_t                     n_in,
                                                size_t                     n_out,
                                                size_t                     nt,
                                                float                     *y) {
#if defined(__ARM_NEON)
    const size_t n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t n_chunks         = n_in / 32;
    const size_t row0             = nt * 4;
    const size_t valid_nr         = (row0 + 4 <= n_out) ? 4 : (n_out - row0);

    float acc[4][GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
    for (size_t nr = 0; nr < valid_nr; nr++) {
        for (size_t mi = 0; mi < m; mi++)
            acc[nr][mi] = 0.0f;
    }

    for (size_t b = 0; b < n_blocks_per_row; b++) {
        int32_t sumi[4][GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        int32_t minc[4][GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        for (size_t nr = 0; nr < valid_nr; nr++) {
            for (size_t mi = 0; mi < m; mi++) {
                sumi[nr][mi] = 0;
                minc[nr][mi] = 0;
            }
        }

        for (size_t nr = 0; nr < valid_nr; nr++) {
            const struct block_q4_K_t *blk = w + (row0 + nr) * n_blocks_per_row + b;
            uint8_t                    scales[8], mins[8];
            for (int is = 0; is < 8; is++) {
                get_scale_min_k4(is, blk->scales, &scales[is], &mins[is]);
            }

            const uint8_t *q = blk->qs;
            for (int ip = 0; ip < 4; ip++, q += 32) {
                const int     is0     = ip * 2;
                const int     is1     = is0 + 1;
                const size_t  xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) ip * 64;
                const size_t  sum_idx = xb_off / 32;
                const int32_t sc0     = (int32_t) scales[is0];
                const int32_t sc1     = (int32_t) scales[is1];
                const int32_t mn0     = (int32_t) mins[is0];
                const int32_t mn1     = (int32_t) mins[is1];

                for (size_t mi = 0; mi < m; mi++) {
                    minc[nr][mi] += mn0 * sum32[mi * n_chunks + sum_idx] +
                                    mn1 * sum32[mi * n_chunks + sum_idx + 1];
                }

                for (int half = 0; half < 32; half += 16) {
                    const uint8x16_t qv = vld1q_u8(q + half);
                    const int8x16_t  lo = vreinterpretq_s8_u8(vandq_u8(qv, vdupq_n_u8(0x0F)));
                    const int8x16_t  hi = vreinterpretq_s8_u8(vshrq_n_u8(qv, 4));
                    for (size_t mi = 0; mi < m; mi++) {
                        const int8_t *xb = x_q8 + mi * n_in + xb_off + (size_t) half;
                        sumi[nr][mi] += sc0 * dot16_i8(xb, lo) + sc1 * dot16_i8(xb + 32, hi);
                    }
                }
            }
        }

        for (size_t nr = 0; nr < valid_nr; nr++) {
            const struct block_q4_K_t *blk  = w + (row0 + nr) * n_blocks_per_row + b;
            const float                d    = fp16_to_fp32(blk->d);
            const float                dmin = fp16_to_fp32(blk->dmin);
            for (size_t mi = 0; mi < m; mi++) {
                acc[nr][mi] +=
                        scale_x[mi] * (d * (float) sumi[nr][mi] - dmin * (float) minc[nr][mi]);
            }
        }
    }

    for (size_t mi = 0; mi < m; mi++) {
        float *yy = y + mi * n_out + row0;
        for (size_t nr = 0; nr < valid_nr; nr++) {
            yy[nr] = acc[nr][mi];
        }
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) w;
    (void) n_in;
    (void) n_out;
    (void) nt;
    (void) y;
#endif
}

void linear_q4k_w4a8_prefill_pair_raw_mtile4_ntile4(const int8_t  *x_q8,
                                                    const float   *scale_x,
                                                    const int32_t *sum32,
                                                    size_t         m,
                                                    const void    *w0_q4k,
                                                    const void    *w1_q4k,
                                                    size_t         n_in,
                                                    size_t         n_out,
                                                    float         *y0,
                                                    float         *y1) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP || n_out == 0 || x_q8 == NULL || scale_x == NULL ||
        sum32 == NULL || w0_q4k == NULL || w1_q4k == NULL || y0 == NULL || y1 == NULL) {
        return;
    }
    const struct block_q4_K_t *w0      = (const struct block_q4_K_t *) w0_q4k;
    const struct block_q4_K_t *w1      = (const struct block_q4_K_t *) w1_q4k;
    const size_t               n_tiles = (n_out + 3) / 4;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t nt = 0; nt < n_tiles; nt++) {
        q4k_raw_tile4_mtile4_compute(x_q8, scale_x, sum32, m, w0, n_in, n_out, nt, y0);
        q4k_raw_tile4_mtile4_compute(x_q8, scale_x, sum32, m, w1, n_in, n_out, nt, y1);
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) sum32;
    (void) m;
    (void) w0_q4k;
    (void) w1_q4k;
    (void) n_in;
    (void) n_out;
    (void) y0;
    (void) y1;
    fprintf(stderr, "linear_q4k_w4a8_prefill_pair_raw_mtile4_ntile4: NEON required\n");
#endif
}

void linear_q4k_w4a8_prefill_predecoded_mtile4_bscale(const int8_t  *x_q8,
                                                      const float   *scale_blocks,
                                                      const int32_t *sum32,
                                                      size_t         m,
                                                      const void    *packed,
                                                      size_t         n_in,
                                                      size_t         n_out,
                                                      float         *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q4k_predecode_valid(packed, n_in, n_out))
        return;

    const struct q4k_predecode_block *w                = q4k_predecode_blocks(packed);
    const size_t                      n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    const size_t                      n_chunks         = n_in / 32;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct q4k_predecode_block *row = w + n * n_blocks_per_row;
        if (n + 1 < n_out)
            __builtin_prefetch(row + n_blocks_per_row, 0, 0);

        size_t mt = 0;
        for (; mt + 4 <= m; mt += 4) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;

            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blk = &row[b];
                if (b + 2 < n_blocks_per_row)
                    __builtin_prefetch(&row[b + 2], 0, 0);

                int32_t sumi0 = 0, sumi1 = 0, sumi2 = 0, sumi3 = 0;
                int32_t min0 = 0, min1 = 0, min2 = 0, min3 = 0;

                for (int is = 0; is < 8; is++) {
                    const int8_t   *q       = blk->qs + is * 32;
                    const int8x16_t q0      = vld1q_s8(q + 0);
                    const int8x16_t q1      = vld1q_s8(q + 16);
                    const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t    sum_idx = xb_off / 32;
                    const int32_t   sc      = (int32_t) blk->scales[is];
                    const int32_t   mn      = (int32_t) blk->mins[is];

                    const int8_t *xb0 = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t *xb1 = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t *xb2 = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t *xb3 = x_q8 + (mt + 3) * n_in + xb_off;

                    const int32_t dot0 = dot16_i8(xb0 + 0, q0) + dot16_i8(xb0 + 16, q1);
                    const int32_t dot1 = dot16_i8(xb1 + 0, q0) + dot16_i8(xb1 + 16, q1);
                    const int32_t dot2 = dot16_i8(xb2 + 0, q0) + dot16_i8(xb2 + 16, q1);
                    const int32_t dot3 = dot16_i8(xb3 + 0, q0) + dot16_i8(xb3 + 16, q1);
                    sumi0 += sc * dot0;
                    sumi1 += sc * dot1;
                    sumi2 += sc * dot2;
                    sumi3 += sc * dot3;

                    min0 += mn * sum32[(mt + 0) * n_chunks + sum_idx];
                    min1 += mn * sum32[(mt + 1) * n_chunks + sum_idx];
                    min2 += mn * sum32[(mt + 2) * n_chunks + sum_idx];
                    min3 += mn * sum32[(mt + 3) * n_chunks + sum_idx];
                }

                const float d    = blk->d;
                const float dmin = blk->dmin;
                const float sx0  = scale_blocks[(mt + 0) * n_blocks_per_row + b];
                const float sx1  = scale_blocks[(mt + 1) * n_blocks_per_row + b];
                const float sx2  = scale_blocks[(mt + 2) * n_blocks_per_row + b];
                const float sx3  = scale_blocks[(mt + 3) * n_blocks_per_row + b];
                acc0 += sx0 * (d * (float) sumi0 - dmin * (float) min0);
                acc1 += sx1 * (d * (float) sumi1 - dmin * (float) min1);
                acc2 += sx2 * (d * (float) sumi2 - dmin * (float) min2);
                acc3 += sx3 * (d * (float) sumi3 - dmin * (float) min3);
            }

            y[(mt + 0) * n_out + n] = acc0;
            y[(mt + 1) * n_out + n] = acc1;
            y[(mt + 2) * n_out + n] = acc2;
            y[(mt + 3) * n_out + n] = acc3;
        }

        for (; mt < m; mt++) {
            float acc = 0.0f;
            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q4k_predecode_block *blk       = &row[b];
                int32_t                           sumi      = 0;
                int32_t                           mins_corr = 0;
                for (int is = 0; is < 8; is++) {
                    const int8_t   *q       = blk->qs + is * 32;
                    const int8x16_t q0      = vld1q_s8(q + 0);
                    const int8x16_t q1      = vld1q_s8(q + 16);
                    const size_t    xb_off  = b * Q4_K_BLOCK_ELEMS + (size_t) is * 32;
                    const size_t    sum_idx = xb_off / 32;
                    const int8_t   *xb      = x_q8 + mt * n_in + xb_off;
                    const int32_t   dot     = dot16_i8(xb + 0, q0) + dot16_i8(xb + 16, q1);
                    sumi += (int32_t) blk->scales[is] * dot;
                    mins_corr += (int32_t) blk->mins[is] * sum32[mt * n_chunks + sum_idx];
                }
                const float sx = scale_blocks[mt * n_blocks_per_row + b];
                acc += sx * (blk->d * (float) sumi - blk->dmin * (float) mins_corr);
            }
            y[mt * n_out + n] = acc;
        }
    }
#else
    (void) x_q8;
    (void) scale_blocks;
    (void) sum32;
    (void) m;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q4k_w4a8_prefill_predecoded_mtile4_bscale: NEON required\n");
#endif
}

void linear_q4k_decode_w4a8_predecoded(
        const float *x, const void *packed, size_t n_in, size_t n_out, float *y) {
    if (!q4k_predecode_valid(packed, n_in, n_out))
        return;
    int8_t  *x_q8  = heap_alloc_array_aligned(int8_t, n_in);
    int32_t *sum32 = heap_alloc_array_aligned(int32_t, n_in / 32);
    if (x_q8 == NULL || sum32 == NULL) {
        safe_free((void **) &x_q8);
        safe_free((void **) &sum32);
        return;
    }
    const float scale_x = quantize_x_for_q4k(x, n_in, x_q8, sum32);
    linear_q4k_w4a8_prefill_predecoded(x_q8, &scale_x, sum32, 1, packed, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &sum32);
}

void linear_q4k_decode_fp32(
        const float *x, const void *w_q4k, size_t n_in, size_t n_out, float *y) {
    const struct block_q4_K_t *w                = (const struct block_q4_K_t *) w_q4k;
    const size_t               n_blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;

    for (size_t n = 0; n < n_out; n++) {
        const struct block_q4_K_t *row = w + n * n_blocks_per_row;
        float                      acc = 0.0f;
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_q4_K_t *blk  = &row[b];
            const float                d    = fp16_to_fp32(blk->d);
            const float                dmin = fp16_to_fp32(blk->dmin);
            const uint8_t             *q    = blk->qs;
            const float               *xb   = x + b * Q4_K_BLOCK_ELEMS;
            int                        is   = 0;
            uint8_t                    sc, m;
            for (size_t sb = 0; sb < Q4_K_BLOCK_ELEMS; sb += 64) {
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                const float d1 = d * (float) sc;
                const float m1 = dmin * (float) m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                const float d2 = d * (float) sc;
                const float m2 = dmin * (float) m;
                float       a1, s1, a2, s2;
                q4k_subpair_dots(q, xb + sb, xb + sb + 32, &a1, &s1, &a2, &s2);
                acc += d1 * a1 - m1 * s1;
                acc += d2 * a2 - m2 * s2;
                q += 32;
                is += 2;
            }
        }
        y[n] = acc;
    }
}
