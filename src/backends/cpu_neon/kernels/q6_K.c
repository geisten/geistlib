/*
 * src/backends/cpu_neon/kernels/q6_K.c — Q6_K W6A8 NEON kernels.
 *
 * Pure compute. Block layout from src/quant/quant_blocks.h; the
 * file-format decoder dequant_q6_K_row stays in src/formats/gguf/q6_K.c.
 *
 * Owns the M=1 decode (with deferred -32 bias correction), the M>1
 * prefill (per-row workspace allocation), and the FP32 reference.
 */
#include "quant_blocks.h"
#include "heap.h"
#include "quant.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
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

static int q6k_pp_enabled(void) {
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

struct q6k_predecode_header {
    uint32_t magic;
    uint32_t n_in;
    uint32_t n_out;
    uint32_t n_blocks_per_row;
    uint32_t block_bytes;
    uint32_t reserved;
};

struct q6k_predecode_block {
    float  d;
    int8_t scales[16];
    int8_t qs[Q6_K_BLOCK_ELEMS];
};

struct q6k_predecode_stream4 {
    float  ds[16][4];
    int8_t qs[16][4][16];
};

struct q6k_x8_header {
    uint32_t magic;
    uint32_t n_in;
    uint32_t n_out;
    uint32_t n_blocks_per_row;
    uint32_t block_bytes;
    uint32_t reserved;
};

struct q6k_x8_block {
    uint16_t d[8];
    int8_t   scales[16 * 8];
    uint8_t  ql[Q6_K_BLOCK_ELEMS / 2 * 8];
    uint8_t  qh[Q6_K_BLOCK_ELEMS / 4 * 8];
};

struct q8k_activation_block {
    float   d;
    int8_t  qs[Q6_K_BLOCK_ELEMS];
    int16_t bsums[16];
};

static constexpr uint32_t Q6K_PREDECODE_NTILE4_MAGIC        = 0x344B3650u; /* "P6K4" */
static constexpr uint32_t Q6K_PREDECODE_NTILE4_STREAM_MAGIC = 0x34533650u; /* "P6S4" */
static constexpr uint32_t Q6K_X8_GEMV_MAGIC                 = 0x38583650u; /* "P6X8" */

static_assert(sizeof(struct q6k_predecode_block) == 276, "q6k_predecode_block layout changed");
static_assert(sizeof(struct q6k_predecode_stream4) == 1280, "q6k_predecode_stream4 layout changed");
static_assert(sizeof(struct q6k_x8_block) == Q6_K_BLOCK_BYTES * 8, "q6k_x8_block layout changed");

static inline const struct q6k_predecode_block *q6k_predecode_ntile4_blocks(const void *packed) {
    return (const struct q6k_predecode_block *) ((const uint8_t *) packed +
                                                 sizeof(struct q6k_predecode_header));
}

static inline const struct q6k_predecode_stream4 *
q6k_predecode_ntile4_stream_blocks(const void *packed) {
    return (const struct q6k_predecode_stream4 *) ((const uint8_t *) packed +
                                                   sizeof(struct q6k_predecode_header));
}

static inline const struct q6k_x8_block *q6k_x8_blocks(const void *packed) {
    return (const struct q6k_x8_block *) ((const uint8_t *) packed + sizeof(struct q6k_x8_header));
}

static inline bool q6k_predecode_ntile4_valid(const void *packed, size_t n_in, size_t n_out) {
    if (packed == NULL)
        return false;
    const struct q6k_predecode_header *h = (const struct q6k_predecode_header *) packed;
    return h->magic == Q6K_PREDECODE_NTILE4_MAGIC && h->n_in == (uint32_t) n_in &&
           h->n_out == (uint32_t) n_out &&
           h->n_blocks_per_row == (uint32_t) (n_in / Q6_K_BLOCK_ELEMS) &&
           h->block_bytes == sizeof(struct q6k_predecode_block);
}

static inline bool
q6k_predecode_ntile4_stream_valid(const void *packed, size_t n_in, size_t n_out) {
    if (packed == NULL)
        return false;
    const struct q6k_predecode_header *h = (const struct q6k_predecode_header *) packed;
    return h->magic == Q6K_PREDECODE_NTILE4_STREAM_MAGIC && h->n_in == (uint32_t) n_in &&
           h->n_out == (uint32_t) n_out &&
           h->n_blocks_per_row == (uint32_t) (n_in / Q6_K_BLOCK_ELEMS) &&
           h->block_bytes == sizeof(struct q6k_predecode_stream4);
}

static inline bool q6k_x8_valid(const void *packed, size_t n_in, size_t n_out) {
    if (packed == NULL || n_out % 8 != 0)
        return false;
    const struct q6k_x8_header *h = (const struct q6k_x8_header *) packed;
    return h->magic == Q6K_X8_GEMV_MAGIC && h->n_in == (uint32_t) n_in &&
           h->n_out == (uint32_t) n_out &&
           h->n_blocks_per_row == (uint32_t) (n_in / Q6_K_BLOCK_ELEMS) &&
           h->block_bytes == sizeof(struct q6k_x8_block);
}

struct q6k_pp_ctx {
    const void    *w_q6k;
    const int8_t  *x_q8;
    const int16_t *bsums; /* 16 int16 per super-block */
    size_t         n_blocks_per_row;
    size_t         n_out;
    float          scale_x;
    float         *y;
};
static void q6k_decode_one_row(size_t n, const struct q6k_pp_ctx *c);
static void q6k_pp_row(size_t n, void *vctx) {
    q6k_decode_one_row(n, (const struct q6k_pp_ctx *) vctx);
}

#if defined(__ARM_NEON)
static inline float q6k_dot4_scaled(const int8_t *xb,
                                    int8x16_t     q0,
                                    int8x16_t     q1,
                                    int8x16_t     q2,
                                    int8x16_t     q3,
                                    float         scale_x,
                                    float         ds0,
                                    float         ds1,
                                    float         ds2,
                                    float         ds3) {
    const int32_t dot0 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q0, vld1q_s8(xb + 0)));
    const int32_t dot1 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q1, vld1q_s8(xb + 32)));
    const int32_t dot2 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q2, vld1q_s8(xb + 64)));
    const int32_t dot3 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q3, vld1q_s8(xb + 96)));
    return scale_x *
           (ds0 * (float) dot0 + ds1 * (float) dot1 + ds2 * (float) dot2 + ds3 * (float) dot3);
}
#endif

size_t q6k_predecode_ntile4_size_bytes(size_t n_in, size_t n_out) {
    if (n_in == 0 || n_out == 0 || n_in % Q6_K_BLOCK_ELEMS != 0)
        return 0;
    if (n_in / Q6_K_BLOCK_ELEMS > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX) {
        return 0;
    }
    const size_t n_tiles  = (n_out + 3) / 4;
    const size_t n_blocks = n_tiles * (n_in / Q6_K_BLOCK_ELEMS) * 4;
    if (n_blocks >
        (SIZE_MAX - sizeof(struct q6k_predecode_header)) / sizeof(struct q6k_predecode_block)) {
        return 0;
    }
    return sizeof(struct q6k_predecode_header) + n_blocks * sizeof(struct q6k_predecode_block);
}

size_t q6k_predecode_ntile4_stream_size_bytes(size_t n_in, size_t n_out) {
    if (n_in == 0 || n_out == 0 || n_in % Q6_K_BLOCK_ELEMS != 0)
        return 0;
    if (n_in / Q6_K_BLOCK_ELEMS > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX) {
        return 0;
    }
    const size_t n_tiles  = (n_out + 3) / 4;
    const size_t n_blocks = n_tiles * (n_in / Q6_K_BLOCK_ELEMS);
    if (n_blocks >
        (SIZE_MAX - sizeof(struct q6k_predecode_header)) / sizeof(struct q6k_predecode_stream4)) {
        return 0;
    }
    return sizeof(struct q6k_predecode_header) + n_blocks * sizeof(struct q6k_predecode_stream4);
}

size_t q6k_x8_gemv_size_bytes(size_t n_in, size_t n_out) {
    if (n_in == 0 || n_out == 0 || n_in % Q6_K_BLOCK_ELEMS != 0 || n_out % 8 != 0) {
        return 0;
    }
    if (n_in / Q6_K_BLOCK_ELEMS > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX) {
        return 0;
    }
    const size_t n_blocks = (n_out / 8) * (n_in / Q6_K_BLOCK_ELEMS);
    if (n_blocks > (SIZE_MAX - sizeof(struct q6k_x8_header)) / sizeof(struct q6k_x8_block)) {
        return 0;
    }
    return sizeof(struct q6k_x8_header) + n_blocks * sizeof(struct q6k_x8_block);
}

static void q6k_predecode_one_block(const struct block_q6_K_t *s, struct q6k_predecode_block *d) {
    d->d = fp16_to_fp32(s->d);
    memcpy(d->scales, s->scales, sizeof(d->scales));
    for (int half = 0; half < 2; half++) {
        const uint8_t *ql = s->ql + half * 64;
        const uint8_t *qh = s->qh + half * 32;
        int8_t        *q  = d->qs + half * 128;
        for (int l = 0; l < 32; l++) {
            q[l + 0]  = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            q[l + 32] = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            q[l + 64] = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            q[l + 96] = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        }
    }
}

int q6k_predecode_ntile4_pack(const void *w_q6k, size_t n_in, size_t n_out, void *dst) {
    if (w_q6k == NULL || dst == NULL)
        return -1;
    const size_t bytes = q6k_predecode_ntile4_size_bytes(n_in, n_out);
    if (bytes == 0)
        return -1;

    const size_t                 n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;
    struct q6k_predecode_header *h                = (struct q6k_predecode_header *) dst;
    *h                                            = (struct q6k_predecode_header) {
            .magic            = Q6K_PREDECODE_NTILE4_MAGIC,
            .n_in             = (uint32_t) n_in,
            .n_out            = (uint32_t) n_out,
            .n_blocks_per_row = (uint32_t) n_blocks_per_row,
            .block_bytes      = (uint32_t) sizeof(struct q6k_predecode_block),
            .reserved         = 0,
    };

    const struct block_q6_K_t  *src = (const struct block_q6_K_t *) w_q6k;
    struct q6k_predecode_block *dstb =
            (struct q6k_predecode_block *) ((uint8_t *) dst + sizeof(*h));
    memset(dstb, 0, bytes - sizeof(*h));

    const size_t n_tiles = (n_out + 3) / 4;
    for (size_t nt = 0; nt < n_tiles; nt++) {
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            for (size_t nr = 0; nr < 4; nr++) {
                const size_t n = nt * 4 + nr;
                if (n >= n_out)
                    continue;
                const struct block_q6_K_t  *s = src + n * n_blocks_per_row + b;
                struct q6k_predecode_block *d = dstb + (nt * n_blocks_per_row + b) * 4 + nr;
                q6k_predecode_one_block(s, d);
            }
        }
    }
    return 0;
}

int q6k_predecode_ntile4_stream_pack(const void *w_q6k, size_t n_in, size_t n_out, void *dst) {
    if (w_q6k == NULL || dst == NULL)
        return -1;
    const size_t bytes = q6k_predecode_ntile4_stream_size_bytes(n_in, n_out);
    if (bytes == 0)
        return -1;

    const size_t                 n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;
    struct q6k_predecode_header *h                = (struct q6k_predecode_header *) dst;
    *h                                            = (struct q6k_predecode_header) {
            .magic            = Q6K_PREDECODE_NTILE4_STREAM_MAGIC,
            .n_in             = (uint32_t) n_in,
            .n_out            = (uint32_t) n_out,
            .n_blocks_per_row = (uint32_t) n_blocks_per_row,
            .block_bytes      = (uint32_t) sizeof(struct q6k_predecode_stream4),
            .reserved         = 0,
    };

    const struct block_q6_K_t    *src = (const struct block_q6_K_t *) w_q6k;
    struct q6k_predecode_stream4 *dstb =
            (struct q6k_predecode_stream4 *) ((uint8_t *) dst + sizeof(*h));
    memset(dstb, 0, bytes - sizeof(*h));

    const size_t n_tiles = (n_out + 3) / 4;
    for (size_t nt = 0; nt < n_tiles; nt++) {
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            struct q6k_predecode_stream4 *d = dstb + nt * n_blocks_per_row + b;
            for (size_t nr = 0; nr < 4; nr++) {
                const size_t n = nt * 4 + nr;
                if (n >= n_out)
                    continue;
                const struct block_q6_K_t *s = src + n * n_blocks_per_row + b;
                struct q6k_predecode_block tmp;
                q6k_predecode_one_block(s, &tmp);
                for (int is = 0; is < 16; is++) {
                    d->ds[is][nr] = tmp.d * (float) tmp.scales[is];
                    memcpy(d->qs[is][nr], tmp.qs + is * 16, 16);
                }
            }
        }
    }
    return 0;
}

static struct q6k_x8_block q6k_x8_make_block(const struct block_q6_K_t in[8]) {
    struct q6k_x8_block out;
    for (int r = 0; r < 8; r++) {
        out.d[r] = in[r].d;
    }

    const int chunk    = 4;
    const int ql_iters = (Q6_K_BLOCK_ELEMS / 2 * 8) / chunk;
    for (int i = 0; i < ql_iters; i++) {
        const int src_id  = i % 8;
        const int src_off = (i / 8) * chunk;
        const int dst_off = i * chunk;
        memcpy(out.ql + dst_off, in[src_id].ql + src_off, chunk);
    }

    const int qh_iters = (Q6_K_BLOCK_ELEMS / 4 * 8) / chunk;
    for (int i = 0; i < qh_iters; i++) {
        const int src_id  = i % 8;
        const int src_off = (i / 8) * chunk;
        const int dst_off = i * chunk;
        memcpy(out.qh + dst_off, in[src_id].qh + src_off, chunk);
    }

    for (int s = 0; s < 16; s++) {
        for (int r = 0; r < 8; r++) {
            out.scales[s * 8 + r] = in[r].scales[s];
        }
    }
    return out;
}

int q6k_x8_gemv_pack(const void *w_q6k, size_t n_in, size_t n_out, void *dst) {
    if (w_q6k == NULL || dst == NULL)
        return -1;
    const size_t bytes = q6k_x8_gemv_size_bytes(n_in, n_out);
    if (bytes == 0)
        return -1;

    const size_t          n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;
    struct q6k_x8_header *h                = (struct q6k_x8_header *) dst;
    *h                                     = (struct q6k_x8_header) {
            .magic            = Q6K_X8_GEMV_MAGIC,
            .n_in             = (uint32_t) n_in,
            .n_out            = (uint32_t) n_out,
            .n_blocks_per_row = (uint32_t) n_blocks_per_row,
            .block_bytes      = (uint32_t) sizeof(struct q6k_x8_block),
            .reserved         = 0,
    };

    const struct block_q6_K_t *src = (const struct block_q6_K_t *) w_q6k;
    struct q6k_x8_block       *out = (struct q6k_x8_block *) ((uint8_t *) dst + sizeof(*h));
    for (size_t row = 0; row < n_out; row += 8) {
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            struct block_q6_K_t tmp[8];
            for (int r = 0; r < 8; r++) {
                tmp[r] = src[(row + (size_t) r) * n_blocks_per_row + b];
            }
            *out++ = q6k_x8_make_block(tmp);
        }
    }
    return 0;
}

void linear_q6k_decode_w6a8_pre(
        const int8_t *x_q8, float scale_x, const void *w_q6k, size_t n_in, size_t n_out, float *y) {
#if defined(__ARM_NEON)
    const size_t n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;

    /* Precompute per-(super-block, sub-block) activation sums for the
     * deferred -32 bias. The Q6_K weight is encoded as q ∈ [0, 63];
     * true value is (q - 32). The "obvious" path subtracts 32 from each
     * element before vdotq_s32 (16 vsubq_s8 per super-block). Instead
     * we use raw q in vdotq and correct once per super-block via
     *
     *   isum_mins = Σ_j scales[j] × bsums[j]
     *   row_acc  += d * scale_x * (isum_dots - 32 * isum_mins)
     *
     * where bsums[j] is the sum of 16 contiguous int8 activations for
     * sub-block j. Saves 16 vsubq_s8 per super-block (~25% of inner
     * NEON ops). Mirrors ggml_vec_dot_q6_K_q8_K's deferred-bias trick. */
    static _Thread_local int16_t *bsums_tl  = NULL;
    static _Thread_local size_t   bsums_cap = 0;
    /* 16 int16 sub-block bsums per super-block. */
    const size_t bsums_need = n_blocks_per_row * 16;
    if (bsums_cap < bsums_need) {
        safe_free((void **) &bsums_tl);
        bsums_tl = heap_alloc_array_aligned(int16_t, bsums_need);
        if (bsums_tl == NULL) {
            bsums_cap = 0;
            return;
        }
        bsums_cap = bsums_need;
    }
    for (size_t b = 0; b < n_blocks_per_row; b++) {
        const int8_t *xb = x_q8 + b * Q6_K_BLOCK_ELEMS;
        for (int j = 0; j < 16; j++) {
            const int8x16_t v    = vld1q_s8(xb + j * 16);
            bsums_tl[b * 16 + j] = (int16_t) vaddlvq_s8(v);
        }
    }
    int16_t *const bsums = bsums_tl; /* shared with workers */

    const struct q6k_pp_ctx ctx = {
            .w_q6k            = w_q6k,
            .x_q8             = x_q8,
            .bsums            = bsums,
            .n_blocks_per_row = n_blocks_per_row,
            .n_out            = n_out,
            .scale_x          = scale_x,
            .y                = y,
    };
    if (q6k_pp_enabled()) {
#if defined(_OPENMP)
        if (!omp_in_parallel())
#endif
        {
            geist_pp_parallel_for(n_out, q6k_pp_row, (void *) &ctx);
            return;
        }
    }

#if defined(_OPENMP)
    if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t n = 0; n < n_out; n++) {
            q6k_decode_one_row(n, &ctx);
        }
    } else if (n_out >= 4096) {
#pragma omp parallel for schedule(static)
        for (size_t n = 0; n < n_out; n++) {
            q6k_decode_one_row(n, &ctx);
        }
    } else {
#pragma omp parallel for schedule(dynamic, 4)
        for (size_t n = 0; n < n_out; n++) {
            q6k_decode_one_row(n, &ctx);
        }
    }
#else
    for (size_t n = 0; n < n_out; n++) {
        q6k_decode_one_row(n, &ctx);
    }
#endif
#else
    /* Scalar fallback for non-NEON: reuse FP32 reference by dequant-on-fly. */
    (void) x_q8;
    (void) scale_x;
    (void) w_q6k;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q6k_decode_w6a8_pre: NEON required\n");
#endif
}

void linear_q6k_decode_w6a8_x8_pre(const int8_t *x_q8,
                                   float         scale_x,
                                   const void   *packed,
                                   size_t        n_in,
                                   size_t        n_out,
                                   float        *y) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (!q6k_x8_valid(packed, n_in, n_out))
        return;
    const size_t                                      n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;
    static _Thread_local struct q8k_activation_block *q8_tl            = NULL;
    static _Thread_local size_t                       q8_cap           = 0;
    if (q8_cap < n_blocks_per_row) {
        safe_free((void **) &q8_tl);
        q8_tl = heap_alloc_array_aligned(struct q8k_activation_block, n_blocks_per_row);
        if (q8_tl == NULL) {
            q8_cap = 0;
            return;
        }
        q8_cap = n_blocks_per_row;
    }
    for (size_t b = 0; b < n_blocks_per_row; b++) {
        struct q8k_activation_block *qb = q8_tl + b;
        qb->d                           = scale_x;
        memcpy(qb->qs, x_q8 + b * Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_ELEMS);
        for (int s = 0; s < 16; s++) {
            qb->bsums[s] = (int16_t) vaddlvq_s8(vld1q_s8(qb->qs + s * 16));
        }
    }
    struct q8k_activation_block *const q8_blocks = q8_tl;

    const struct q6k_x8_block *w                = q6k_x8_blocks(packed);
    const uint8x16_t           mask_lo4         = vdupq_n_u8(0x0F);
    const uint8x16_t           mask_lo2         = vdupq_n_u8(0x03);
    const uint8x16_t           mask_hi2_shifted = vdupq_n_u8(0x30);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t tile = 0; tile < n_out / 8; tile++) {
        float32x4_t                acc_f32_0 = vdupq_n_f32(0.0f);
        float32x4_t                acc_f32_1 = vdupq_n_f32(0.0f);
        const struct q6k_x8_block *row       = w + tile * n_blocks_per_row;

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct q6k_x8_block         *blk = row + b;
            const struct q8k_activation_block *q8  = q8_blocks + b;

            float d_arr[8];
            for (int i = 0; i < 8; i++)
                d_arr[i] = fp16_to_fp32(blk->d[i]);
            const float32x4_t sb_scale_0 = vmulq_n_f32(vld1q_f32(d_arr + 0), q8->d);
            const float32x4_t sb_scale_1 = vmulq_n_f32(vld1q_f32(d_arr + 4), q8->d);

            int16_t q6_scales[16 * 8];
            for (int s = 0; s < 16; s++) {
                vst1q_s16(q6_scales + s * 8, vmovl_s8(vld1_s8(blk->scales + s * 8)));
            }

            int32x4_t bias_0 = vdupq_n_s32(0);
            int32x4_t bias_1 = vdupq_n_s32(0);
            for (int s = 0; s < 16; s += 4) {
                const int16x4_t bs   = vld1_s16(q8->bsums + s);
                const int16x4_t sc00 = vld1_s16(q6_scales + (s + 0) * 8 + 0);
                const int16x4_t sc01 = vld1_s16(q6_scales + (s + 0) * 8 + 4);
                const int16x4_t sc10 = vld1_s16(q6_scales + (s + 1) * 8 + 0);
                const int16x4_t sc11 = vld1_s16(q6_scales + (s + 1) * 8 + 4);
                const int16x4_t sc20 = vld1_s16(q6_scales + (s + 2) * 8 + 0);
                const int16x4_t sc21 = vld1_s16(q6_scales + (s + 2) * 8 + 4);
                const int16x4_t sc30 = vld1_s16(q6_scales + (s + 3) * 8 + 0);
                const int16x4_t sc31 = vld1_s16(q6_scales + (s + 3) * 8 + 4);
                bias_0               = vmlal_lane_s16(bias_0, sc00, bs, 0);
                bias_1               = vmlal_lane_s16(bias_1, sc01, bs, 0);
                bias_0               = vmlal_lane_s16(bias_0, sc10, bs, 1);
                bias_1               = vmlal_lane_s16(bias_1, sc11, bs, 1);
                bias_0               = vmlal_lane_s16(bias_0, sc20, bs, 2);
                bias_1               = vmlal_lane_s16(bias_1, sc21, bs, 2);
                bias_0               = vmlal_lane_s16(bias_0, sc30, bs, 3);
                bias_1               = vmlal_lane_s16(bias_1, sc31, bs, 3);
            }
            bias_0 = vshlq_n_s32(bias_0, 5);
            bias_1 = vshlq_n_s32(bias_1, 5);

            int32x4_t acc_0 = vdupq_n_s32(0);
            int32x4_t acc_1 = vdupq_n_s32(0);
            for (int half = 0; half < 2; half++) {
                const uint8_t *ql_base = blk->ql + half * 512;
                const uint8_t *qh_base = blk->qh + half * 256;

                for (int sb = 0; sb < 4; sb++) {
                    const int8_t *q8_l_base = q8->qs + half * 128 + sb * 16;
                    const int8_t *q8_h_base = q8_l_base + 64;
                    int8x16_t     q8_l[4];
                    int8x16_t     q8_h[4];
                    for (int i = 0; i < 4; i++) {
                        q8_l[i] = (int8x16_t) vld1q_dup_s32((const int32_t *) (q8_l_base + i * 4));
                        q8_h[i] = (int8x16_t) vld1q_dup_s32((const int32_t *) (q8_h_base + i * 4));
                    }

                    const int    ql_off = sb * 128;
                    const int    qh_off = ql_off & 255;
                    uint8x16x4_t ql_0   = vld1q_u8_x4(ql_base + ql_off);
                    uint8x16x4_t ql_1   = vld1q_u8_x4(ql_base + ql_off + 64);
                    uint8x16x4_t qh_0   = vld1q_u8_x4(qh_base + qh_off);
                    uint8x16x4_t qh_1   = vld1q_u8_x4(qh_base + qh_off + 64);
                    if (sb > 1) {
                        for (int i = 0; i < 4; i++) {
                            qh_0.val[i] = vshrq_n_u8(qh_0.val[i], 2);
                            qh_1.val[i] = vshrq_n_u8(qh_1.val[i], 2);
                        }
                    }

                    const uint8x16_t ql[8] = {
                            ql_0.val[0],
                            ql_0.val[1],
                            ql_0.val[2],
                            ql_0.val[3],
                            ql_1.val[0],
                            ql_1.val[1],
                            ql_1.val[2],
                            ql_1.val[3],
                    };
                    const uint8x16_t qh[8] = {
                            qh_0.val[0],
                            qh_0.val[1],
                            qh_0.val[2],
                            qh_0.val[3],
                            qh_1.val[0],
                            qh_1.val[1],
                            qh_1.val[2],
                            qh_1.val[3],
                    };

                    for (int group = 0; group < 2; group++) {
                        int32x4_t sb_acc_l = vdupq_n_s32(0);
                        int32x4_t sb_acc_h = vdupq_n_s32(0);
                        for (int chunk = 0; chunk < 4; chunk++) {
                            const int       idx = chunk * 2 + group;
                            const int8x16_t q_l = vreinterpretq_s8_u8(vsliq_n_u8(
                                    vandq_u8(ql[idx], mask_lo4), vandq_u8(qh[idx], mask_lo2), 4));
                            const int8x16_t q_h = vreinterpretq_s8_u8(vorrq_u8(
                                    vshrq_n_u8(ql[idx], 4), vandq_u8(qh[idx], mask_hi2_shifted)));
                            sb_acc_l            = vdotq_s32(sb_acc_l, q_l, q8_l[chunk]);
                            sb_acc_h            = vdotq_s32(sb_acc_h, q_h, q8_h[chunk]);
                        }

                        const int       scale_idx_l = half * 8 + sb;
                        const int       scale_idx_h = half * 8 + sb + 4;
                        const int32x4_t sc_l =
                                vmovl_s16(vld1_s16(q6_scales + scale_idx_l * 8 + group * 4));
                        const int32x4_t sc_h =
                                vmovl_s16(vld1_s16(q6_scales + scale_idx_h * 8 + group * 4));
                        if (group == 0) {
                            acc_0 = vmlaq_s32(acc_0, sb_acc_l, sc_l);
                            acc_0 = vmlaq_s32(acc_0, sb_acc_h, sc_h);
                        } else {
                            acc_1 = vmlaq_s32(acc_1, sb_acc_l, sc_l);
                            acc_1 = vmlaq_s32(acc_1, sb_acc_h, sc_h);
                        }
                    }
                }
            }

            acc_0     = vsubq_s32(acc_0, bias_0);
            acc_1     = vsubq_s32(acc_1, bias_1);
            acc_f32_0 = vaddq_f32(acc_f32_0, vmulq_f32(vcvtq_f32_s32(acc_0), sb_scale_0));
            acc_f32_1 = vaddq_f32(acc_f32_1, vmulq_f32(vcvtq_f32_s32(acc_1), sb_scale_1));
        }

        float *dst = y + tile * 8;
        vst1q_f32(dst + 0, acc_f32_0);
        vst1q_f32(dst + 4, acc_f32_1);
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q6k_decode_w6a8_x8_pre: NEON dotprod required\n");
#endif
}

#if defined(__ARM_NEON)
static void q6k_decode_one_row(size_t n, const struct q6k_pp_ctx *c) {
    const struct block_q6_K_t *w                = (const struct block_q6_K_t *) c->w_q6k;
    const size_t               n_blocks_per_row = c->n_blocks_per_row;
    const size_t               n_out            = c->n_out;
    const int8_t              *x_q8             = c->x_q8;
    const int16_t             *bsums            = c->bsums;
    const float                scale_x          = c->scale_x;
    float                     *y                = c->y;

    const struct block_q6_K_t *row = w + n * n_blocks_per_row;
    float                      acc = 0.0f;
#if !defined(GEIST_TARGET_PI5)
    if (n + 1 < n_out)
        __builtin_prefetch(row + n_blocks_per_row, 0, 0);
#else
    (void) n_out;
#endif

    const uint8x16_t mask_lo4 = vdupq_n_u8(0x0F);
    const uint8x16_t mask_lo2 = vdupq_n_u8(0x03);

    for (size_t b = 0; b < n_blocks_per_row; b++) {
        const struct block_q6_K_t *blk = &row[b];
#if !defined(GEIST_TARGET_PI5)
        if (b + 2 < n_blocks_per_row)
            __builtin_prefetch(&row[b + 2], 0, 0);
#endif

        const float    d  = fp16_to_fp32(blk->d);
        const int8_t  *sc = blk->scales;
        const int8_t  *xb = x_q8 + b * Q6_K_BLOCK_ELEMS;
        const int16_t *bs = bsums + b * 16;

        /* Deferred -32 bias correction: isum_mins = Σ scales[j] × bsums[j].
         * Done once per super-block via int16 vmull + horizontal sum. */
        const int16x8x2_t q8sums   = {{vld1q_s16(bs + 0), vld1q_s16(bs + 8)}};
        const int8x16_t   scales_b = vld1q_s8(sc);
        const int16x8x2_t q6scales = {{
                vmovl_s8(vget_low_s8(scales_b)),
                vmovl_s8(vget_high_s8(scales_b)),
        }};
        const int32x4_t   prod     = vaddq_s32(
                vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[0]), vget_low_s16(q6scales.val[0])),
                          vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[1]), vget_low_s16(q6scales.val[1])),
                          vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
        const int32_t isum_mins = vaddvq_s32(prod);

        /* Inner dot accumulation as raw int32 (sum of stream contributions
         * weighted by per-stream scales), without -32 subtract.
         *
         * Phase 1 NOTE: we tried the acc32 pattern here (sustained
         * int32x4_t accumulator + vmlaq_n_s32 + single vaddvq_s32) and
         * measured a -6 % regression on Mac M1. Q6_K already has 4
         * independent vdotq_s32 + scalar fold pattern that keeps the
         * NEON and integer pipelines busy in parallel; the acc32 change
         * collapses both into the NEON pipeline and loses ILP. Leaving
         * the explicit scalar isum here is the right shape for Q6_K. */
#if defined(GEIST_TARGET_PI5)
        int32x4_t isumv = vdupq_n_s32(0);
#else
        int32_t isum = 0;
#endif

        for (int half = 0; half < 2; half++) {
            const uint8_t *ql = blk->ql + half * 64;
            const uint8_t *qh = blk->qh + half * 32;
            for (int sub_off = 0; sub_off < 32; sub_off += 16) {
                const uint8x16_t ql_b0 = vld1q_u8(ql + sub_off + 0);
                const uint8x16_t ql_b1 = vld1q_u8(ql + sub_off + 32);
                const uint8x16_t qhv   = vld1q_u8(qh + sub_off);

                const uint8x16_t qh0 = vandq_u8(qhv, mask_lo2);
                const uint8x16_t qh1 = vandq_u8(vshrq_n_u8(qhv, 2), mask_lo2);
                const uint8x16_t qh2 = vandq_u8(vshrq_n_u8(qhv, 4), mask_lo2);
                const uint8x16_t qh3 = vshrq_n_u8(qhv, 6);

                /* Skip -32 subtract: keep q ∈ [0, 63]. */
                const int8x16_t q0 = vreinterpretq_s8_u8(
                        vorrq_u8(vandq_u8(ql_b0, mask_lo4), vshlq_n_u8(qh0, 4)));
                const int8x16_t q1 = vreinterpretq_s8_u8(
                        vorrq_u8(vandq_u8(ql_b1, mask_lo4), vshlq_n_u8(qh1, 4)));
                const int8x16_t q2 =
                        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_b0, 4), vshlq_n_u8(qh2, 4)));
                const int8x16_t q3 =
                        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_b1, 4), vshlq_n_u8(qh3, 4)));

                const int8x16_t x0v = vld1q_s8(xb + sub_off + 0);
                const int8x16_t x1v = vld1q_s8(xb + sub_off + 32);
                const int8x16_t x2v = vld1q_s8(xb + sub_off + 64);
                const int8x16_t x3v = vld1q_s8(xb + sub_off + 96);

#if defined(GEIST_TARGET_PI5)
                const int32x4_t dot0 = vdotq_s32(vdupq_n_s32(0), q0, x0v);
                const int32x4_t dot1 = vdotq_s32(vdupq_n_s32(0), q1, x1v);
                const int32x4_t dot2 = vdotq_s32(vdupq_n_s32(0), q2, x2v);
                const int32x4_t dot3 = vdotq_s32(vdupq_n_s32(0), q3, x3v);

                const int sub_idx = sub_off / 16;
                isumv             = vmlaq_n_s32(isumv, dot0, (int32_t) sc[0 + sub_idx]);
                isumv             = vmlaq_n_s32(isumv, dot1, (int32_t) sc[2 + sub_idx]);
                isumv             = vmlaq_n_s32(isumv, dot2, (int32_t) sc[4 + sub_idx]);
                isumv             = vmlaq_n_s32(isumv, dot3, (int32_t) sc[6 + sub_idx]);
#else
                const int32_t dot0 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q0, x0v));
                const int32_t dot1 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q1, x1v));
                const int32_t dot2 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q2, x2v));
                const int32_t dot3 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q3, x3v));

                const int sub_idx = sub_off / 16;
                isum += (int32_t) sc[0 + sub_idx] * dot0 + (int32_t) sc[2 + sub_idx] * dot1 +
                        (int32_t) sc[4 + sub_idx] * dot2 + (int32_t) sc[6 + sub_idx] * dot3;
#endif
            }
            xb += 128;
            sc += 8;
        }
#if defined(GEIST_TARGET_PI5)
        const int32_t isum = vaddvq_s32(isumv);
#endif
        acc += d * scale_x * (float) (isum - 32 * isum_mins);
    }
    y[n] = acc;
}
#else
static void q6k_decode_one_row(size_t n, const struct q6k_pp_ctx *c) {
    (void) n;
    (void) c;
}
#endif

void linear_q6k_w6a8_prefill_pre(const int8_t *x_q8,
                                 const float  *scale_x,
                                 size_t        m,
                                 const void   *w_q6k,
                                 size_t        n_in,
                                 size_t        n_out,
                                 float        *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    const struct block_q6_K_t *w                = (const struct block_q6_K_t *) w_q6k;
    const size_t               n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;

    /* Activation packing (§10.11): pack x ONCE into block-major layout
     * packed[(b*m + t)*256 + e] so per-token reads are sequential instead of
     * strided by n_in (=6144 for down) — the same lever that won gate_up
     * (+4-6%, L1-miss 2.4%->0.95%). down is the #1 prefill gap vs llama-BLAS:
     * strided access made it slower AND thrash at large m. GEIST_Q6K_PACK_ACT=0
     * disables. Reused thread-local high-water buffer (kernel runs on the single
     * layer-loop thread; its omp panels only READ the buffer). m<2 skipped. */
    static int q6k_pack_act = -1;
    if (q6k_pack_act < 0) {
        const char *e = getenv("GEIST_Q6K_PACK_ACT");
        q6k_pack_act  = (e != NULL && e[0] == '0') ? 0 : 1;
    }
    const int8_t *packed = NULL;
    if (q6k_pack_act && m >= 2) {
        static _Thread_local int8_t *q6kpack_tl  = NULL;
        static _Thread_local size_t  q6kpack_cap = 0;
        const size_t                 need        = m * n_in;
        if (need > q6kpack_cap) {
            /* Route through heap.h (AGENT.md): the buffer is fully repopulated
             * by the packing loop below before any read, so dropping the old
             * contents on grow is safe. */
            safe_free((void **) &q6kpack_tl);
            q6kpack_tl  = heap_alloc_array_aligned(int8_t, need);
            q6kpack_cap = (q6kpack_tl != NULL) ? need : 0;
        }
        if (q6kpack_cap >= need) {
            int8_t *pk = q6kpack_tl;
            for (size_t b = 0; b < n_blocks_per_row; b++)
                for (size_t t = 0; t < m; t++)
                    memcpy(pk + (b * m + t) * Q6_K_BLOCK_ELEMS,
                           x_q8 + t * n_in + b * Q6_K_BLOCK_ELEMS,
                           Q6_K_BLOCK_ELEMS);
            packed = pk;
        }
    }

    /* Activation offset (within the 256-elem block) for each of the 16
     * reconstructed q-vectors, in (half, sub_off, chunk) iteration order.
     * q-vector g maps to x[blk_off + xoff_tab[g] .. +15]. */
    static const int xoff_tab[16] = {
            0,
            32,
            64,
            96,
            16,
            48,
            80,
            112, /* half 0: sub0, sub16 */
            128,
            160,
            192,
            224,
            144,
            176,
            208,
            240, /* half 1: sub0, sub16 */
    };

/* Recon-once-per-block (Plan A §10.3): unpack a Q6_K block ONCE into a
 * 16×int8x16 L1 scratch (QREG) + per-vector int32 scales (SREG); all m
 * tokens then SDOT against it. Bit-identical (same integer ops + deferred
 * fp32 scale). Uses mask_lo4/mask_lo2/bias_32 + sub_off from the caller. */
#define Q6K_RECON_BLOCK(BLK, QREG, SREG)                                                           \
    do {                                                                                           \
        int g_ = 0;                                                                                \
        for (int half_ = 0; half_ < 2; half_++) {                                                  \
            const uint8_t *ql_ = (BLK)->ql + half_ * 64;                                           \
            const uint8_t *qh_ = (BLK)->qh + half_ * 32;                                           \
            const int8_t  *sc_ = (BLK)->scales + half_ * 8;                                        \
            for (int so_ = 0; so_ < 32; so_ += 16) {                                               \
                uint8x16_t la_ = vandq_u8(vld1q_u8(ql_ + so_ + 0), mask_lo4);                      \
                uint8x16_t lb_ = vshrq_n_u8(vld1q_u8(ql_ + so_ + 0), 4);                           \
                uint8x16_t lc_ = vandq_u8(vld1q_u8(ql_ + so_ + 32), mask_lo4);                     \
                uint8x16_t le_ = vshrq_n_u8(vld1q_u8(ql_ + so_ + 32), 4);                          \
                uint8x16_t hv_ = vld1q_u8(qh_ + so_);                                              \
                uint8x16_t h0_ = vandq_u8(hv_, mask_lo2);                                          \
                uint8x16_t h1_ = vandq_u8(vshrq_n_u8(hv_, 2), mask_lo2);                           \
                uint8x16_t h2_ = vandq_u8(vshrq_n_u8(hv_, 4), mask_lo2);                           \
                uint8x16_t h3_ = vshrq_n_u8(hv_, 6);                                               \
                (QREG)[g_ + 0] =                                                                   \
                        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(la_, vshlq_n_u8(h0_, 4))), bias_32); \
                (QREG)[g_ + 1] =                                                                   \
                        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lc_, vshlq_n_u8(h1_, 4))), bias_32); \
                (QREG)[g_ + 2] =                                                                   \
                        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lb_, vshlq_n_u8(h2_, 4))), bias_32); \
                (QREG)[g_ + 3] =                                                                   \
                        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(le_, vshlq_n_u8(h3_, 4))), bias_32); \
                const int si_  = so_ / 16;                                                         \
                (SREG)[g_ + 0] = (int32_t) sc_[0 + si_];                                           \
                (SREG)[g_ + 1] = (int32_t) sc_[2 + si_];                                           \
                (SREG)[g_ + 2] = (int32_t) sc_[4 + si_];                                           \
                (SREG)[g_ + 3] = (int32_t) sc_[6 + si_];                                           \
                g_ += 4;                                                                           \
            }                                                                                      \
        }                                                                                          \
    } while (0)

    const size_t n_pairs = n_out / 2;

    /* NR=2 microkernel (Plan A + NR-tiling §10.3): compute TWO output rows per
     * activation read so each x[token,block] L2 load feeds both rows' SDOTs —
     * halves activation L2->L1 traffic, which is the bound once recon is
     * amortized. qreg0/qreg1 (1 KB total) live in L1 scratch. */
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 2)
#endif
    for (size_t np = 0; np < n_pairs; np++) {
        const size_t               n    = np * 2;
        const struct block_q6_K_t *row0 = w + n * n_blocks_per_row;
        const struct block_q6_K_t *row1 = w + (n + 1) * n_blocks_per_row;

        float accs0[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        float accs1[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        for (size_t i = 0; i < m; i++) {
            accs0[i] = 0.0f;
            accs1[i] = 0.0f;
        }

        const uint8x16_t mask_lo4 = vdupq_n_u8(0x0F);
        const uint8x16_t mask_lo2 = vdupq_n_u8(0x03);
        const int8x16_t  bias_32  = vdupq_n_s8(32);

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_q6_K_t *blk0 = &row0[b];
            const struct block_q6_K_t *blk1 = &row1[b];
            if (b + 2 < n_blocks_per_row) {
                __builtin_prefetch(&row0[b + 2], 0, 0);
                __builtin_prefetch(&row1[b + 2], 0, 0);
            }
            const float d0 = fp16_to_fp32(blk0->d);
            const float d1 = fp16_to_fp32(blk1->d);

            int8x16_t qreg0[16], qreg1[16];
            int32_t   sreg0[16] __attribute__((aligned(16)));
            int32_t   sreg1[16] __attribute__((aligned(16)));
            Q6K_RECON_BLOCK(blk0, qreg0, sreg0);
            Q6K_RECON_BLOCK(blk1, qreg1, sreg1);

            const size_t  blk_off = b * Q6_K_BLOCK_ELEMS;
            const int8_t *xblk;
            size_t        xstride;
            if (packed) {
                xblk    = packed + (size_t) b * m * Q6_K_BLOCK_ELEMS;
                xstride = Q6_K_BLOCK_ELEMS;
            } else {
                xblk    = x_q8 + blk_off;
                xstride = n_in;
            }
            size_t i = 0;
            for (; i + 4 <= m; i += 4) { /* MR=4 tokens × NR=2 rows */
                int32x4_t a00 = vdupq_n_s32(0), a01 = vdupq_n_s32(0);
                int32x4_t a02 = vdupq_n_s32(0), a03 = vdupq_n_s32(0);
                int32x4_t a10 = vdupq_n_s32(0), a11 = vdupq_n_s32(0);
                int32x4_t a12 = vdupq_n_s32(0), a13 = vdupq_n_s32(0);
                for (int g = 0; g < 16; g++) {
                    const size_t    xg = (size_t) xoff_tab[g];
                    const int8x16_t v0 = vld1q_s8(xblk + (i + 0) * xstride + xg);
                    const int8x16_t v1 = vld1q_s8(xblk + (i + 1) * xstride + xg);
                    const int8x16_t v2 = vld1q_s8(xblk + (i + 2) * xstride + xg);
                    const int8x16_t v3 = vld1q_s8(xblk + (i + 3) * xstride + xg);
                    const int8x16_t q0 = qreg0[g];
                    const int32_t   s0 = sreg0[g];
                    const int8x16_t q1 = qreg1[g];
                    const int32_t   s1 = sreg1[g];
                    a00                = vmlaq_n_s32(a00, vdotq_s32(vdupq_n_s32(0), q0, v0), s0);
                    a01                = vmlaq_n_s32(a01, vdotq_s32(vdupq_n_s32(0), q0, v1), s0);
                    a02                = vmlaq_n_s32(a02, vdotq_s32(vdupq_n_s32(0), q0, v2), s0);
                    a03                = vmlaq_n_s32(a03, vdotq_s32(vdupq_n_s32(0), q0, v3), s0);
                    a10                = vmlaq_n_s32(a10, vdotq_s32(vdupq_n_s32(0), q1, v0), s1);
                    a11                = vmlaq_n_s32(a11, vdotq_s32(vdupq_n_s32(0), q1, v1), s1);
                    a12                = vmlaq_n_s32(a12, vdotq_s32(vdupq_n_s32(0), q1, v2), s1);
                    a13                = vmlaq_n_s32(a13, vdotq_s32(vdupq_n_s32(0), q1, v3), s1);
                }
                accs0[i + 0] += d0 * scale_x[i + 0] * (float) vaddvq_s32(a00);
                accs0[i + 1] += d0 * scale_x[i + 1] * (float) vaddvq_s32(a01);
                accs0[i + 2] += d0 * scale_x[i + 2] * (float) vaddvq_s32(a02);
                accs0[i + 3] += d0 * scale_x[i + 3] * (float) vaddvq_s32(a03);
                accs1[i + 0] += d1 * scale_x[i + 0] * (float) vaddvq_s32(a10);
                accs1[i + 1] += d1 * scale_x[i + 1] * (float) vaddvq_s32(a11);
                accs1[i + 2] += d1 * scale_x[i + 2] * (float) vaddvq_s32(a12);
                accs1[i + 3] += d1 * scale_x[i + 3] * (float) vaddvq_s32(a13);
            }
            for (; i < m; i++) { /* token remainder */
                int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
                for (int g = 0; g < 16; g++) {
                    const int8x16_t v = vld1q_s8(xblk + i * xstride + (size_t) xoff_tab[g]);
                    acc0 = vmlaq_n_s32(acc0, vdotq_s32(vdupq_n_s32(0), qreg0[g], v), sreg0[g]);
                    acc1 = vmlaq_n_s32(acc1, vdotq_s32(vdupq_n_s32(0), qreg1[g], v), sreg1[g]);
                }
                accs0[i] += d0 * scale_x[i] * (float) vaddvq_s32(acc0);
                accs1[i] += d1 * scale_x[i] * (float) vaddvq_s32(acc1);
            }
        }
        for (size_t i = 0; i < m; i++) {
            y[i * n_out + n + 0] = accs0[i];
            y[i * n_out + n + 1] = accs1[i];
        }
    }

    /* NR=1 tail for odd n_out (n_out=1536 is even for Gemma 4 down, so this is
     * a correctness fallback). */
    for (size_t n = n_pairs * 2; n < n_out; n++) {
        const struct block_q6_K_t *row = w + n * n_blocks_per_row;
        float                      accs[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        for (size_t i = 0; i < m; i++)
            accs[i] = 0.0f;

        const uint8x16_t mask_lo4 = vdupq_n_u8(0x0F);
        const uint8x16_t mask_lo2 = vdupq_n_u8(0x03);
        const int8x16_t  bias_32  = vdupq_n_s8(32);

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_q6_K_t *blk = &row[b];
            const float                d   = fp16_to_fp32(blk->d);
            int8x16_t                  qreg[16];
            int32_t                    sreg[16] __attribute__((aligned(16)));
            Q6K_RECON_BLOCK(blk, qreg, sreg);

            const size_t  blk_off = b * Q6_K_BLOCK_ELEMS;
            const int8_t *xblk;
            size_t        xstride;
            if (packed) {
                xblk    = packed + (size_t) b * m * Q6_K_BLOCK_ELEMS;
                xstride = Q6_K_BLOCK_ELEMS;
            } else {
                xblk    = x_q8 + blk_off;
                xstride = n_in;
            }
            for (size_t i = 0; i < m; i++) {
                int32x4_t acc = vdupq_n_s32(0);
                for (int g = 0; g < 16; g++) {
                    const int8_t *xb = xblk + i * xstride + (size_t) xoff_tab[g];
                    acc              = vmlaq_n_s32(
                            acc, vdotq_s32(vdupq_n_s32(0), qreg[g], vld1q_s8(xb)), sreg[g]);
                }
                accs[i] += d * scale_x[i] * (float) vaddvq_s32(acc);
            }
        }
        for (size_t i = 0; i < m; i++)
            y[i * n_out + n] = accs[i];
    }
#undef Q6K_RECON_BLOCK
#else
    (void) x_q8;
    (void) scale_x;
    (void) m;
    (void) w_q6k;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q6k_w6a8_prefill_pre: NEON required\n");
#endif
}

void linear_q6k_w6a8_prefill_pre_accum_blocks(const int8_t *x_q8,
                                              const float  *scale_x,
                                              size_t        m,
                                              const void   *w_q6k,
                                              size_t        n_in_total,
                                              size_t        n_out,
                                              size_t        block_start,
                                              size_t        n_blocks,
                                              float        *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP || n_blocks == 0 || x_q8 == NULL || scale_x == NULL ||
        w_q6k == NULL || y == NULL) {
        return;
    }
    const struct block_q6_K_t *w                = (const struct block_q6_K_t *) w_q6k;
    const size_t               n_blocks_per_row = n_in_total / Q6_K_BLOCK_ELEMS;
    const size_t               n_in_tile        = n_blocks * Q6_K_BLOCK_ELEMS;
    if (block_start + n_blocks > n_blocks_per_row)
        return;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q6_K_t *row = w + n * n_blocks_per_row + block_start;

        float accs[GEIST_QUANT_M_CAP] __attribute__((aligned(16)));
        for (size_t i = 0; i < m; i++)
            accs[i] = 0.0f;

        for (size_t b = 0; b < n_blocks; b++) {
            const struct block_q6_K_t *blk = &row[b];
            const float                d   = fp16_to_fp32(blk->d);
            const int8_t              *sc  = blk->scales;

            const uint8x16_t mask_lo4 = vdupq_n_u8(0x0F);
            const uint8x16_t mask_lo2 = vdupq_n_u8(0x03);
            const int8x16_t  bias_32  = vdupq_n_s8(32);

            for (int half = 0; half < 2; half++) {
                const uint8_t *ql          = blk->ql + half * 64;
                const uint8_t *qh          = blk->qh + half * 32;
                const size_t   xb_half_off = b * Q6_K_BLOCK_ELEMS + (size_t) half * 128;

                for (int sub_off = 0; sub_off < 32; sub_off += 16) {
                    uint8x16_t ql_lo  = vandq_u8(vld1q_u8(ql + sub_off + 0), mask_lo4);
                    uint8x16_t ql_hi  = vshrq_n_u8(vld1q_u8(ql + sub_off + 0), 4);
                    uint8x16_t ql_lo2 = vandq_u8(vld1q_u8(ql + sub_off + 32), mask_lo4);
                    uint8x16_t ql_hi2 = vshrq_n_u8(vld1q_u8(ql + sub_off + 32), 4);
                    uint8x16_t qhv    = vld1q_u8(qh + sub_off);
                    uint8x16_t qh0    = vandq_u8(qhv, mask_lo2);
                    uint8x16_t qh1    = vandq_u8(vshrq_n_u8(qhv, 2), mask_lo2);
                    uint8x16_t qh2    = vandq_u8(vshrq_n_u8(qhv, 4), mask_lo2);
                    uint8x16_t qh3    = vshrq_n_u8(qhv, 6);

                    int8x16_t q0 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(ql_lo, vshlq_n_u8(qh0, 4))), bias_32);
                    int8x16_t q1 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(ql_lo2, vshlq_n_u8(qh1, 4))), bias_32);
                    int8x16_t q2 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(ql_hi, vshlq_n_u8(qh2, 4))), bias_32);
                    int8x16_t q3 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(ql_hi2, vshlq_n_u8(qh3, 4))), bias_32);

                    const int   sub_idx = sub_off / 16;
                    const float ds0     = d * (float) sc[0 + sub_idx];
                    const float ds1     = d * (float) sc[2 + sub_idx];
                    const float ds2     = d * (float) sc[4 + sub_idx];
                    const float ds3     = d * (float) sc[6 + sub_idx];

                    for (size_t i = 0; i < m; i++) {
                        const int8_t *xb = x_q8 + i * n_in_tile + xb_half_off + (size_t) sub_off;
                        accs[i] +=
                                q6k_dot4_scaled(xb, q0, q1, q2, q3, scale_x[i], ds0, ds1, ds2, ds3);
                    }
                }
                sc += 8;
            }
        }
        for (size_t i = 0; i < m; i++)
            y[i * n_out + n] += accs[i];
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) m;
    (void) w_q6k;
    (void) n_in_total;
    (void) n_out;
    (void) block_start;
    (void) n_blocks;
    (void) y;
    fprintf(stderr, "linear_q6k_w6a8_prefill_pre_accum_blocks: NEON required\n");
#endif
}

void linear_q6k_w6a8_prefill_predecoded_ntile4(const int8_t *x_q8,
                                               const float  *scale_x,
                                               size_t        m,
                                               const void   *packed,
                                               size_t        n_in,
                                               size_t        n_out,
                                               float        *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q6k_predecode_ntile4_valid(packed, n_in, n_out))
        return;

    const struct q6k_predecode_block *w                = q6k_predecode_ntile4_blocks(packed);
    const size_t                      n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;
    const size_t                      n_tiles          = (n_out + 3) / 4;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t nt = 0; nt < n_tiles; nt++) {
        const size_t                      valid_nr = (nt * 4 + 4 <= n_out) ? 4 : (n_out - nt * 4);
        const struct q6k_predecode_block *tile     = w + nt * n_blocks_per_row * 4;

        /* Recon-free deferred-int32 microkernel (Plan B §10.4): qs is already the
         * dequantized int8 weight in element order, so the dot is pure SDOT — no
         * 6-bit unpack. NR=4 rows (contiguous in the ntile4 pack) × MR=2 tokens:
         * each x[token,sub] load feeds all 4 rows; per-block int32 accumulation
         * with int scales[is] and ONE vaddvq per (row,token,block) — the deferred
         * scale that the original per-sub-fp32 kernel lacked. Padded rows (n>=
         * n_out) are zero in the pack, so all 4 rows compute branch-free; only
         * valid_nr outputs are written. */
        size_t mt = 0;
        for (; mt + 2 <= m; mt += 2) {
            const float sx0   = scale_x[mt + 0];
            const float sx1   = scale_x[mt + 1];
            float       r0[4] = {0}, r1[4] = {0};

            for (size_t b = 0; b < n_blocks_per_row; b++) {
                const struct q6k_predecode_block *blks = tile + b * 4;
                if (b + 1 < n_blocks_per_row)
                    __builtin_prefetch((const char *) (blks + 4), 0, 0);
                const size_t xr0 = (mt + 0) * n_in + b * Q6_K_BLOCK_ELEMS;
                const size_t xr1 = (mt + 1) * n_in + b * Q6_K_BLOCK_ELEMS;

                int32x4_t a00 = vdupq_n_s32(0), a01 = vdupq_n_s32(0);
                int32x4_t a10 = vdupq_n_s32(0), a11 = vdupq_n_s32(0);
                int32x4_t a20 = vdupq_n_s32(0), a21 = vdupq_n_s32(0);
                int32x4_t a30 = vdupq_n_s32(0), a31 = vdupq_n_s32(0);
                for (int is = 0; is < 16; is++) {
                    const int8x16_t x0 = vld1q_s8(x_q8 + xr0 + (size_t) is * 16);
                    const int8x16_t x1 = vld1q_s8(x_q8 + xr1 + (size_t) is * 16);
                    const int8x16_t q0 = vld1q_s8(blks[0].qs + is * 16);
                    const int8x16_t q1 = vld1q_s8(blks[1].qs + is * 16);
                    const int8x16_t q2 = vld1q_s8(blks[2].qs + is * 16);
                    const int8x16_t q3 = vld1q_s8(blks[3].qs + is * 16);
                    const int32_t   s0 = blks[0].scales[is], s1 = blks[1].scales[is];
                    const int32_t   s2 = blks[2].scales[is], s3 = blks[3].scales[is];
                    a00 = vmlaq_n_s32(a00, vdotq_s32(vdupq_n_s32(0), q0, x0), s0);
                    a01 = vmlaq_n_s32(a01, vdotq_s32(vdupq_n_s32(0), q0, x1), s0);
                    a10 = vmlaq_n_s32(a10, vdotq_s32(vdupq_n_s32(0), q1, x0), s1);
                    a11 = vmlaq_n_s32(a11, vdotq_s32(vdupq_n_s32(0), q1, x1), s1);
                    a20 = vmlaq_n_s32(a20, vdotq_s32(vdupq_n_s32(0), q2, x0), s2);
                    a21 = vmlaq_n_s32(a21, vdotq_s32(vdupq_n_s32(0), q2, x1), s2);
                    a30 = vmlaq_n_s32(a30, vdotq_s32(vdupq_n_s32(0), q3, x0), s3);
                    a31 = vmlaq_n_s32(a31, vdotq_s32(vdupq_n_s32(0), q3, x1), s3);
                }
                r0[0] += blks[0].d * sx0 * (float) vaddvq_s32(a00);
                r1[0] += blks[0].d * sx1 * (float) vaddvq_s32(a01);
                r0[1] += blks[1].d * sx0 * (float) vaddvq_s32(a10);
                r1[1] += blks[1].d * sx1 * (float) vaddvq_s32(a11);
                r0[2] += blks[2].d * sx0 * (float) vaddvq_s32(a20);
                r1[2] += blks[2].d * sx1 * (float) vaddvq_s32(a21);
                r0[3] += blks[3].d * sx0 * (float) vaddvq_s32(a30);
                r1[3] += blks[3].d * sx1 * (float) vaddvq_s32(a31);
            }

            float *y0 = y + (mt + 0) * n_out + nt * 4;
            float *y1 = y + (mt + 1) * n_out + nt * 4;
            for (size_t nr = 0; nr < valid_nr; nr++) {
                y0[nr] = r0[nr];
                y1[nr] = r1[nr];
            }
        }

        for (; mt < m; mt++) { /* token remainder: single token, deferred int32 */
            for (size_t nr = 0; nr < valid_nr; nr++) {
                float acc = 0.0f;
                for (size_t b = 0; b < n_blocks_per_row; b++) {
                    const struct q6k_predecode_block *blk = tile + b * 4 + nr;
                    int32x4_t                         a   = vdupq_n_s32(0);
                    for (int is = 0; is < 16; is++) {
                        const int8x16_t xb = vld1q_s8(x_q8 + mt * n_in + b * Q6_K_BLOCK_ELEMS +
                                                      (size_t) is * 16);
                        const int8x16_t qv = vld1q_s8(blk->qs + is * 16);
                        a                  = vmlaq_n_s32(
                                a, vdotq_s32(vdupq_n_s32(0), qv, xb), (int32_t) blk->scales[is]);
                    }
                    acc += scale_x[mt] * blk->d * (float) vaddvq_s32(a);
                }
                y[mt * n_out + nt * 4 + nr] = acc;
            }
        }
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) m;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q6k_w6a8_prefill_predecoded_ntile4: NEON required\n");
#endif
}

void linear_q6k_w6a8_prefill_predecoded_ntile4_stream(const int8_t *x_q8,
                                                      const float  *scale_x,
                                                      size_t        m,
                                                      const void   *packed,
                                                      size_t        n_in,
                                                      size_t        n_out,
                                                      float        *y) {
#if defined(__ARM_NEON)
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    if (!q6k_predecode_ntile4_stream_valid(packed, n_in, n_out))
        return;

    const struct q6k_predecode_stream4 *w = q6k_predecode_ntile4_stream_blocks(packed);
    const size_t                        n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;
    const size_t                        n_tiles          = (n_out + 3) / 4;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t nt = 0; nt < n_tiles; nt++) {
        const size_t                        valid_nr = (nt * 4 + 4 <= n_out) ? 4 : (n_out - nt * 4);
        const struct q6k_predecode_stream4 *tile     = w + nt * n_blocks_per_row;

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
                const struct q6k_predecode_stream4 *blk = tile + b;
                for (int is = 0; is < 16; is++) {
                    const size_t  xb_off = b * Q6_K_BLOCK_ELEMS + (size_t) is * 16;
                    const int8_t *xb0    = x_q8 + (mt + 0) * n_in + xb_off;
                    const int8_t *xb1    = x_q8 + (mt + 1) * n_in + xb_off;
                    const int8_t *xb2    = x_q8 + (mt + 2) * n_in + xb_off;
                    const int8_t *xb3    = x_q8 + (mt + 3) * n_in + xb_off;

#define Q6K_STREAM4_ACC(NR, A0, A1, A2, A3)                 \
    do {                                                    \
        if ((NR) < valid_nr) {                              \
            const int8x16_t qv = vld1q_s8(blk->qs[is][NR]); \
            const float     ds = blk->ds[is][NR];           \
            (A0) += sx0 * ds * (float) dot16_i8(xb0, qv);   \
            (A1) += sx1 * ds * (float) dot16_i8(xb1, qv);   \
            (A2) += sx2 * ds * (float) dot16_i8(xb2, qv);   \
            (A3) += sx3 * ds * (float) dot16_i8(xb3, qv);   \
        }                                                   \
    } while (0)
                    Q6K_STREAM4_ACC(0, a00, a10, a20, a30);
                    Q6K_STREAM4_ACC(1, a01, a11, a21, a31);
                    Q6K_STREAM4_ACC(2, a02, a12, a22, a32);
                    Q6K_STREAM4_ACC(3, a03, a13, a23, a33);
#undef Q6K_STREAM4_ACC
                }
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
                    const struct q6k_predecode_stream4 *blk = tile + b;
                    for (int is = 0; is < 16; is++) {
                        const size_t    xb_off = b * Q6_K_BLOCK_ELEMS + (size_t) is * 16;
                        const int8_t   *xb     = x_q8 + mt * n_in + xb_off;
                        const int8x16_t qv     = vld1q_s8(blk->qs[is][nr]);
                        const int32_t   dot    = dot16_i8(xb, qv);
                        acc += scale_x[mt] * blk->ds[is][nr] * (float) dot;
                    }
                }
                y[mt * n_out + nt * 4 + nr] = acc;
            }
        }
    }
#else
    (void) x_q8;
    (void) scale_x;
    (void) m;
    (void) packed;
    (void) n_in;
    (void) n_out;
    (void) y;
    fprintf(stderr, "linear_q6k_w6a8_prefill_predecoded_ntile4_stream: NEON required\n");
#endif
}

void linear_q6k_w6a8_prefill(
        const float *x, const void *w_q6k, size_t m, size_t n_in, size_t n_out, float *y) {
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;
    int8_t *x_q8    = heap_alloc_array_aligned(int8_t, m *n_in);
    float  *scale_x = heap_alloc_array_aligned(float, m);
    if (x_q8 == NULL || scale_x == NULL) {
        safe_free((void **) &x_q8);
        safe_free((void **) &scale_x);
        return;
    }
    for (size_t i = 0; i < m; i++) {
        scale_x[i] = quantize_x_int8_sym(x + i * n_in, n_in, x_q8 + i * n_in);
    }
    linear_q6k_w6a8_prefill_pre(x_q8, scale_x, m, w_q6k, n_in, n_out, y);
    safe_free((void **) &x_q8);
    safe_free((void **) &scale_x);
}

void linear_q6k_decode_w6a8(
        const float *x, const void *w_q6k, size_t n_in, size_t n_out, float *y) {
    static _Thread_local int8_t *tl_x_q8     = NULL;
    static _Thread_local size_t  tl_cap_n_in = 0;
    if (n_in > tl_cap_n_in) {
        safe_free((void **) &tl_x_q8);
        tl_x_q8 = heap_alloc_array_aligned(int8_t, n_in);
        if (tl_x_q8 == NULL) {
            tl_cap_n_in = 0;
            return;
        }
        tl_cap_n_in = n_in;
    }
    float scale_x = quantize_x_int8_sym(x, n_in, tl_x_q8);
    linear_q6k_decode_w6a8_pre(tl_x_q8, scale_x, w_q6k, n_in, n_out, y);
}

void linear_q6k_decode_w6a8_x8(
        const float *x, const void *packed, size_t n_in, size_t n_out, float *y) {
    static _Thread_local int8_t *tl_x_q8     = NULL;
    static _Thread_local size_t  tl_cap_n_in = 0;
    if (n_in > tl_cap_n_in) {
        safe_free((void **) &tl_x_q8);
        tl_x_q8 = heap_alloc_array_aligned(int8_t, n_in);
        if (tl_x_q8 == NULL) {
            tl_cap_n_in = 0;
            return;
        }
        tl_cap_n_in = n_in;
    }
    float scale_x = quantize_x_int8_sym(x, n_in, tl_x_q8);
    linear_q6k_decode_w6a8_x8_pre(tl_x_q8, scale_x, packed, n_in, n_out, y);
}

void linear_q6k_decode_fp32(
        const float *x, const void *w_q6k, size_t n_in, size_t n_out, float *y) {
    const struct block_q6_K_t *w                = (const struct block_q6_K_t *) w_q6k;
    const size_t               n_blocks_per_row = n_in / Q6_K_BLOCK_ELEMS;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const struct block_q6_K_t *row = w + n * n_blocks_per_row;
        float                      acc = 0.0f;
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            const struct block_q6_K_t *blk = &row[b];
            const float                d   = fp16_to_fp32(blk->d);
            const uint8_t             *ql  = blk->ql;
            const uint8_t             *qh  = blk->qh;
            const int8_t              *sc  = blk->scales;
            const float               *xb  = x + b * Q6_K_BLOCK_ELEMS;
            for (size_t half = 0; half < Q6_K_BLOCK_ELEMS; half += 128) {
                /* 4 streams of 32 elements each within this 128-element half. */
                float a[4] = {0, 0, 0, 0};
                for (int l = 0; l < 32; l++) {
                    int    is = l / 16;
                    int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    a[0] += (float) sc[is + 0] * (float) q1 * xb[l + 0];
                    a[1] += (float) sc[is + 2] * (float) q2 * xb[l + 32];
                    a[2] += (float) sc[is + 4] * (float) q3 * xb[l + 64];
                    a[3] += (float) sc[is + 6] * (float) q4 * xb[l + 96];
                }
                acc += d * (a[0] + a[1] + a[2] + a[3]);
                ql += 64;
                qh += 32;
                sc += 8;
                xb += 128;
            }
        }
        y[n] = acc;
    }
}
