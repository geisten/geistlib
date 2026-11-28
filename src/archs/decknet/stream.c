/*
 * src/archs/decknet/stream.c — decknet on a camera stream (#533).
 *
 * Layer: ARCHITECTURE. The same pass as run.c (decknet_gains, decknet_plane),
 * fed differently. A still image pair pays, per call, for the homography
 * arithmetic of every crop pixel and for resampling the empty deck, and a
 * camera that delivers YUV pays for a colour conversion of the whole frame
 * before that. A stream has one camera pose and one empty deck, so:
 *
 *   sample tables   per homography, where each crop pixel reads the frame
 *                   and with which weights (decknet_sample_at and
 *                   decknet_weights, once);
 *   region          only for the crop pixels the pass reads
 *                   (decknet_plane_pixels);
 *   tiles           the frame is squares of DECKNET_TILE pixels, and only
 *                   the tiles a sample reads are touched. RGB8 is sampled in
 *                   place; of a YUV frame those tiles are
 *                   converted, 16 pixels at a time, and sampled then: slot
 *                   by slot, by the thread that samples the slot, into a
 *                   packed image of what that slot reads (pixels_box);
 *   the empty deck  sampled once, when it is set;
 *   gating          optional (decknet_stream_set_gate): a tile whose bytes
 *                   differ from the kept ones by no more than a threshold
 *                   (sum of absolute differences) keeps the kept ones, and
 *                   what the pass computed from kept tiles is kept as well.
 *
 * None of it changes a number: a stream without gating gives what
 * decknet_run gives for the same two images, bit for bit, and a gated one
 * what decknet_run gives for the frame in which each tile within the
 * threshold is the kept one (tests/test_decknet_stream_unit.c).
 *
 *   spans           a plane whose kernels have few weights is sampled as
 *                   prefix sums along its runs of crop pixels, and a kernel
 *                   takes the sum of a row of equal weights from two of
 *                   them (decknet_work.prefix; not where gating keeps the
 *                   crop).
 *
 *   two threads     optional (decknet_stream_set_threads): the slots are
 *                   independent of each other, so one half of them is
 *                   converted and computed on a second thread, in a second
 *                   workspace.
 *
 *   a queue of one  optional (decknet_stream_offer, _push_offered): the
 *                   thread that captures leaves a pointer to its newest
 *                   frame, the thread that computes takes it. One atomic
 *                   pointer; a frame that is replaced before it is taken is
 *                   handed back to the caller and counted.
 *
 *   a GPU           optional (decknet_stream_set_gpu): the integer part of
 *                   a frame that is not gated (conversion, sampling, the
 *                   kernels' pixel sums) is a backend's, in two stages, and
 *                   this file computes what needs F64 from the first stage
 *                   while the second runs.
 *
 * Everything is allocated in decknet_stream_create and, for gating, in the
 * first decknet_stream_set_gate that turns it on, and for a GPU in
 * decknet_stream_set_gpu; the reference and the frames allocate nothing.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#include "decknet.h"

#include "checked.h"
#include "cleanup.h"
#include "heap.h"
#include "par.h"

#include <stdlib.h>
#include <string.h>

static_assert(__FINITE_MATH_ONLY__ == 0,
              "decknet needs -fno-finite-math-only (mk/common.mk): the sample tables clamp NaN and "
              "Inf by comparison");

/* ---- frames ----------------------------------------------------------- */

/* The bytes of one row of the frame's first plane and how many rows of
 * `stride` it has in all. False for a format that is not one of the four
 * or that cannot hold the frame's size. */
static bool frame_rows(const struct geist_decknet_frame *f, size_t *row, size_t *rows) {
    switch (f->format) {
    case GEIST_DECKNET_FORMAT_RGB8:
        *rows = f->height;
        return !ckd_mul(row, f->width, 3);
    case GEIST_DECKNET_FORMAT_NV12:
        *row = f->width;
        return f->width % 2 == 0 && f->height % 2 == 0 && !ckd_add(rows, f->height, f->height / 2);
    case GEIST_DECKNET_FORMAT_YUYV:
    case GEIST_DECKNET_FORMAT_UYVY:
        *rows = f->height;
        return f->width % 2 == 0 && !ckd_mul(row, f->width, 2);
    }
    return false; /* a value the enum does not name */
}

/* Whether every pixel of the frame lies inside its n_bytes. The frame is
 * the caller's: its size, stride and extent are checked here, once, and the
 * code below indexes it without a further check. */
static bool frame_ok(const struct geist_decknet *d, const struct geist_decknet_frame *f) {
    size_t row = 0, rows = 0, last = 0, need = 0;
    if (f->data == nullptr || f->width != d->img_w || f->height != d->img_h ||
        f->colour > GEIST_DECKNET_COLOUR_BT709_FULL || !frame_rows(f, &row, &rows) ||
        f->stride < row) {
        return false;
    }
    /* The last row starts at (rows - 1) * stride and is `row` bytes long. */
    return !ckd_mul(&last, rows - 1, f->stride) && !ckd_add(&need, last, row) && need <= f->n_bytes;
}

/* Sixteen bytes, eight 16-bit lanes and four 32-bit ones (a crop pixel,
 * run.c's q4). The compilers' vector types, without a platform's intrinsics:
 * a shuffle with constant indices is one table lookup on NEON and one PSHUFB
 * (or two and an OR) on x86-64. */
typedef uint8_t  b16 [[gnu::vector_size(16)]];
typedef int8_t   s16 [[gnu::vector_size(16)]];
typedef uint16_t h8 [[gnu::vector_size(16)]];
typedef uint32_t q4 [[gnu::vector_size(16)]];
static_assert(sizeof(q4) == DECKNET_CROP_CH * sizeof(uint32_t));
static_assert(sizeof(b16) == DECKNET_SAMPLE_BYTES);
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "rgb_lanes reads a byte and three zeros as a 32-bit lane, and convert two bytes as "
              "a 16-bit one");

static inline uint8_t clamp8(int v) {
    return (uint8_t) (v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* A colour's conversion, the formula of docs/DECKNET.md ("Stream"):
 *   c = Ky (Y - Yo) + 128
 *   R = clamp((c + Krv (V - 128)) >> 8)
 *   G = clamp((c - Kgu (U - 128) - Kgv (V - 128)) >> 8)
 *   B = clamp((c + Kbu (U - 128)) >> 8)
 * with the multiples of 256 taken out of the shift, so that 16 bits hold
 * what is left: Ky = 256 + my, Krv = 256 qrv + arv, -Kgu = 256 qgu + agu,
 * -Kgv = 256 qgv + agv, Kbu = 256 qbu + abu. The parts are chosen so that
 * the sums under the shift, the 128 included, stay inside 16 signed bits:
 * for BT.601 limited range between -20128 and 29597, and the widest, BT.709
 * limited range's green, between -22889 and 32446. */
struct yuv_k {
    int yo, my, qrv, arv, qgu, agu, qgv, agv, qbu, abu;
};
static const struct yuv_k yuv_ks[4] = {
        [GEIST_DECKNET_COLOUR_BT601_LIMITED] = {16, 42, 1, 153, 0, -100, -1, 48, 2, 4},
        [GEIST_DECKNET_COLOUR_BT601_FULL]    = {0, 0, 1, 103, 0, -88, -1, 73, 2, -58},
        [GEIST_DECKNET_COLOUR_BT709_LIMITED] = {16, 42, 2, -53, 0, -55, -1, 120, 2, 29},
        [GEIST_DECKNET_COLOUR_BT709_FULL]    = {0, 0, 2, -109, 0, -48, 0, -120, 2, -37},
};

/* Two pixels of a row, which share their chroma, as RGB8: 32768 = 128 x 256
 * is added before the shift and 128 taken off behind it, so that the shift
 * is of an unsigned value. All 2^24 (Y, U, V) give the bytes of the formula
 * above, for each of the four. */
static inline void yuv_pair(uint8_t      luma0,
                            uint8_t      luma1,
                            uint8_t      cb,
                            uint8_t      cr,
                            struct yuv_k k,
                            uint8_t      rgb[static 6]) {
    const int luma[2] = {luma0 - k.yo, luma1 - k.yo};
    const int u = cb - 128, v = cr - 128;
    for (size_t i = 0; i < 2; i++) {
        const int      y = luma[i];
        const unsigned r = (unsigned) (k.my * y + k.arv * v + 128 + 32768);
        const unsigned g = (unsigned) (k.my * y + k.agu * u + k.agv * v + 128 + 32768);
        const unsigned b = (unsigned) (k.my * y + k.abu * u + 128 + 32768);
        rgb[3 * i]       = clamp8(y + k.qrv * v + (int) (r >> 8U) - 128);
        rgb[3 * i + 1]   = clamp8(y + k.qgu * u + k.qgv * v + (int) (g >> 8U) - 128);
        rgb[3 * i + 2]   = clamp8(y + k.qbu * u + (int) (b >> 8U) - 128);
    }
}

/* One colour of 16 pixels as bytes. `even` and `odd` have, in 16 bits that
 * wrap, the values of the even and of the odd pixels before the clamp,
 * which lie between -277 and 534 for BT.601 limited range and inside 16
 * signed bits for the others: the low byte is the colour where the high
 * byte is 0, and the high byte's sign says which way it is out. */
[[gnu::always_inline]] static inline b16 clamp16(h8 even, h8 odd) {
    const b16 e = (b16) even, o = (b16) odd;
    const b16 lo = __builtin_shufflevector(
            e, o, 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
    const s16 hi = (s16) __builtin_shufflevector(
            e, o, 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31);
    return (lo | (b16) (hi > 0)) & ~(b16) (hi < 0);
}

/* 16 pixels of a row as RGB8, yuv_pair for eight pairs at once: the luma
 * of each pair's first pixel in `even`, of its second in `odd`, and its
 * chroma in u and v, all as the bytes are (0 to 255).
 *
 * With the Yo, the 128s and the 32768 of yuv_pair gathered into one
 * constant per term, in 16 bits that wrap; for BT.601 limited range:
 *   r = 42 Y + 153 V + 12640           R = Y + V - 272 + (r >> 8)
 *   g = 42 Y - 100 U + 48 V + 38880    G = Y - V - 16 + (g >> 8)
 *   b = 42 Y + 4 U + 31712             B = Y + 2 U - 400 + (b >> 8)
 * r, g and b are yuv_pair's, below 65536. What depends on the chroma alone
 * is computed once for the pair. With a `k` the compiler knows, the
 * products by 0 and by 1 are gone. */
/* A term's constant, in 16 bits that wrap. */
static inline uint16_t k16(int x) {
    return (uint16_t) x;
}

[[gnu::always_inline]] static inline void
yuv16(h8 even, h8 odd, h8 u, h8 v, struct yuv_k k, uint8_t rgb[static 48]) {
    const int c0 = 32896 - k.my * k.yo, d0 = -k.yo - 128;
    const h8  r_c = k16(k.arv) * v + k16(c0 - 128 * k.arv);
    const h8  g_c = k16(k.agv) * v + k16(k.agu) * u + k16(c0 - 128 * (k.agu + k.agv));
    const h8  b_c = k16(k.abu) * u + k16(c0 - 128 * k.abu);
    const h8  r_d = k16(k.qrv) * v + k16(d0 - 128 * k.qrv);
    const h8  g_d = k16(k.qgv) * v + k16(k.qgu) * u + k16(d0 - 128 * (k.qgu + k.qgv));
    const h8  b_d = k16(k.qbu) * u + k16(d0 - 128 * k.qbu);
    const h8  m_e = k16(k.my) * even, m_o = k16(k.my) * odd;
    const b16 r = clamp16(even + r_d + ((m_e + r_c) >> 8), odd + r_d + ((m_o + r_c) >> 8));
    const b16 g = clamp16(even + g_d + ((m_e + g_c) >> 8), odd + g_d + ((m_o + g_c) >> 8));
    const b16 b = clamp16(even + b_d + ((m_e + b_c) >> 8), odd + b_d + ((m_o + b_c) >> 8));
    /* R G B R G B ...: r and g side by side with a gap, then b into it. */
    constexpr int x = -1; /* any byte */
    const b16     rg0 =
            __builtin_shufflevector(r, g, 0, 16, x, 1, 17, x, 2, 18, x, 3, 19, x, 4, 20, x, 5);
    const b16 rg1 =
            __builtin_shufflevector(r, g, 21, x, 6, 22, x, 7, 23, x, 8, 24, x, 9, 25, x, 10, 26);
    const b16 rg2 =
            __builtin_shufflevector(r, g, x, 11, 27, x, 12, 28, x, 13, 29, x, 14, 30, x, 15, 31, x);
    const b16 out0 = __builtin_shufflevector(
            rg0, b, 0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15);
    const b16 out1 = __builtin_shufflevector(
            rg1, b, 0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15);
    const b16 out2 = __builtin_shufflevector(
            rg2, b, 26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31);
    memcpy(rgb, &out0, sizeof out0);
    memcpy(rgb + 16, &out1, sizeof out1);
    memcpy(rgb + 32, &out2, sizeof out2);
}

/* yuv16 for the colour `k` points at, with BT.601 limited range, which is
 * what most cameras deliver, as constants: through the table alone it took
 * longer (docs/DECKNET.md, "Measured"). */
[[gnu::always_inline]] static inline void
yuv16_of(h8 even, h8 odd, h8 u, h8 v, const struct yuv_k *k, uint8_t rgb[static 48]) {
    if (k == &yuv_ks[GEIST_DECKNET_COLOUR_BT601_LIMITED]) {
        yuv16(even, odd, u, v, yuv_ks[GEIST_DECKNET_COLOUR_BT601_LIMITED], rgb);
    } else {
        yuv16(even, odd, u, v, *k, rgb);
    }
}

/* ---- tiles ------------------------------------------------------------ */

/* An RGB8 image to sample: the bytes of pixel (x, y) start at
 * rgb[y x stride + x x 3 - bias], and `wide` is the first such offset from
 * which DECKNET_SAMPLE_BYTES bytes of the row below are not all there to
 * read (0: nowhere are they). `bias` is 0 for an image that starts with its
 * pixel (0, 0); one that holds a rectangle of a frame has the offset its
 * first pixel would have had. */
struct pixels {
    const uint8_t *rgb;
    size_t         stride, bias, wide;
};

/* A frame frame_ok accepted, RGB8, as it lies in the caller's memory. */
static struct pixels pixels_of(const struct geist_decknet_frame *f) {
    const size_t need = f->stride + DECKNET_SAMPLE_BYTES;
    return (struct pixels) {.rgb    = f->data,
                            .stride = f->stride,
                            .wide   = f->n_bytes < need ? 0 : f->n_bytes - need + 1};
}

/* s->rgb as an image of the frame's size, which has DECKNET_SAMPLE_BYTES
 * behind its last pixel. */
static struct pixels pixels_own(const struct geist_decknet_stream *s) {
    return (struct pixels) {.rgb = s->rgb, .stride = s->d->img_w * 3, .wide = SIZE_MAX};
}

/* The tiles of `box` as an image in `rgb` (s->rgb or s->rgb2), packed: its
 * rows are as long as the rectangle is wide, so that what a slot reads of a
 * frame lies in a few hundred KB whatever the frame's size, and in the
 * same ones for every slot a thread computes. It is no larger than the
 * frame, and has DECKNET_SAMPLE_BYTES behind its last pixel as well. */
static struct pixels
pixels_box(const struct geist_decknet_stream *s, struct decknet_box box, const uint8_t *rgb) {
    const size_t x0     = box.x0 * DECKNET_TILE;
    const size_t x1     = ((size_t) box.x1 + 1) * DECKNET_TILE;
    const size_t stride = ((x1 < s->d->img_w ? x1 : s->d->img_w) - x0) * 3;
    return (struct pixels) {.rgb    = rgb,
                            .stride = stride,
                            .bias   = box.y0 * DECKNET_TILE * stride + x0 * 3,
                            .wide   = SIZE_MAX};
}

/* The pixels [x0, x1) x [y0, y1) of a frame frame_ok accepted, as RGB8 into
 * the image `to`, whose bytes are `rgb`. x0 and x1 are even for the YUV
 * formats: tiles start at multiples of DECKNET_TILE and those frames have an
 * even width. */
static void convert(const struct geist_decknet_frame *f,
                    size_t                            x0,
                    size_t                            x1,
                    size_t                            y0,
                    size_t                            y1,
                    const struct pixels              *to,
                    uint8_t                          *rgb) {
    const struct yuv_k *k = &yuv_ks[f->colour];
    for (size_t y = y0; y < y1; y++) {
        const uint8_t *row = f->data + y * f->stride;
        uint8_t       *out = rgb + (y * to->stride + x0 * 3 - to->bias);
        switch (f->format) {
        case GEIST_DECKNET_FORMAT_RGB8:
            memcpy(out, row + x0 * 3, (x1 - x0) * 3);
            break;
        case GEIST_DECKNET_FORMAT_NV12: {
            const uint8_t *uv = f->data + (f->height + y / 2) * f->stride;
            size_t         x  = x0;
            for (; x + 16 <= x1; x += 16, out += 48) {
                h8 luma, chroma; /* a lane is two Y, and a U with its V */
                memcpy(&luma, row + x, sizeof luma);
                memcpy(&chroma, uv + x, sizeof chroma);
                yuv16_of(luma & 0xFF, luma >> 8, chroma & 0xFF, chroma >> 8, k, out);
            }
            for (; x < x1; x += 2, out += 6) {
                yuv_pair(row[x], row[x + 1], uv[x], uv[x + 1], *k, out);
            }
            break;
        }
        case GEIST_DECKNET_FORMAT_YUYV:
        case GEIST_DECKNET_FORMAT_UYVY: {
            /* Y0 U Y1 V, or U Y0 V Y1. */
            const bool     uy = f->format == GEIST_DECKNET_FORMAT_UYVY;
            const uint8_t *p  = row + x0 * 2;
            size_t         x  = x0;
            for (; x + 16 <= x1; x += 16, p += 32, out += 48) {
                h8 lo, hi;
                memcpy(&lo, p, sizeof lo);
                memcpy(&hi, p + 16, sizeof hi);
                /* The lanes with Y0 and U, and those with Y1 and V, the Y
                 * in the low byte of a lane or in the high one. A shift by
                 * a variable in place of the branch cost 0.05 ms of a
                 * 1920 x 1080 frame (M1 Max). */
                const h8 first  = __builtin_shufflevector(lo, hi, 0, 2, 4, 6, 8, 10, 12, 14);
                const h8 second = __builtin_shufflevector(lo, hi, 1, 3, 5, 7, 9, 11, 13, 15);
                if (uy) {
                    yuv16_of(first >> 8, second >> 8, first & 0xFF, second & 0xFF, k, out);
                } else {
                    yuv16_of(first & 0xFF, second & 0xFF, first >> 8, second >> 8, k, out);
                }
            }
            for (; x < x1; x += 2, p += 4, out += 6) {
                yuv_pair(p[uy], p[2 + uy], p[1 - uy], p[3 - uy], *k, out);
            }
            break;
        }
        }
    }
}

/* The bytes of a frame that belong to the pixels [x0, x1) x [y0, y1), as up
 * to two blocks of rows: rows [row0, row1) of the frame's `stride`, bytes
 * [col0, col1) of each. NV12 has the second one, in its chroma plane. */
struct block {
    size_t row0, row1, col0, col1;
};

static size_t blocks(const struct geist_decknet_frame *f,
                     size_t                            x0,
                     size_t                            x1,
                     size_t                            y0,
                     size_t                            y1,
                     struct block                      out[static 2]) {
    switch (f->format) {
    case GEIST_DECKNET_FORMAT_RGB8:
        out[0] = (struct block) {y0, y1, x0 * 3, x1 * 3};
        return 1;
    case GEIST_DECKNET_FORMAT_NV12:
        out[0] = (struct block) {y0, y1, x0, x1};
        out[1] = (struct block) {f->height + y0 / 2, f->height + (y1 + 1) / 2, x0, x1};
        return 2;
    case GEIST_DECKNET_FORMAT_YUYV:
    case GEIST_DECKNET_FORMAT_UYVY:
        out[0] = (struct block) {y0, y1, x0 * 2, x1 * 2};
        return 1;
    }
    return 0;
}

/* Whether the bytes of a tile (its blocks) differ from the kept ones, whose
 * rows are `row` bytes apart, by more than `threshold` as the sum of
 * absolute differences. Most tiles of a camera that stands still are the
 * same bytes, which memcmp finds fastest. */
static bool tile_differs(const struct geist_decknet_frame *f,
                         size_t                            n_blocks,
                         const struct block                b[static n_blocks],
                         size_t                            row,
                         uint32_t                          threshold,
                         const uint8_t                    *kept) {
    bool same = true;
    for (size_t i = 0; same && i < n_blocks; i++) {
        for (size_t r = b[i].row0; same && r < b[i].row1; r++) {
            same = memcmp(f->data + r * f->stride + b[i].col0,
                          kept + r * row + b[i].col0,
                          b[i].col1 - b[i].col0) == 0;
        }
    }
    if (same || threshold == 0) {
        return !same;
    }
    uint32_t diff = 0;
    for (size_t i = 0; i < n_blocks; i++) {
        for (size_t r = b[i].row0; r < b[i].row1; r++) {
            const uint8_t *x = f->data + r * f->stride, *y = kept + r * row;
            for (size_t c = b[i].col0; c < b[i].col1; c++) {
                diff += (uint32_t) abs((int) x[c] - (int) y[c]);
            }
        }
    }
    return diff > threshold;
}

/* Marks the tiles the 2 x 2 camera pixels of each sample lie in, for reader
 * `who`: a slot, or d->n_slots for the worktable strip. */
static void mark_tiles(struct geist_decknet_stream *s,
                       size_t                       n,
                       size_t                       who,
                       const struct decknet_sample  table[static n]) {
    const uint64_t      bit = who < s->d->n_slots ? (uint64_t) 1 << who : 0;
    struct decknet_box *box = &s->read_box[who];
    for (size_t i = 0; i < n; i++) {
        const size_t x = table[i].x, y = table[i].y;
        for (size_t k = 0; k < 4; k++) {
            const size_t t = (y + k / 2) / DECKNET_TILE * s->tiles_w + (x + k % 2) / DECKNET_TILE;
            s->used[t]     = 1;
            s->slot_tiles[t] |= bit;
        }
        /* The first and the last of the four, which lie in tiles below
         * DECKNET_MAX_IMAGE_DIM / DECKNET_TILE. */
        const uint16_t tx0 = (uint16_t) (x / DECKNET_TILE), ty0 = (uint16_t) (y / DECKNET_TILE);
        const uint16_t tx1 = (uint16_t) ((x + 1) / DECKNET_TILE);
        const uint16_t ty1 = (uint16_t) ((y + 1) / DECKNET_TILE);
        box->x0            = tx0 < box->x0 ? tx0 : box->x0;
        box->x1            = tx1 > box->x1 ? tx1 : box->x1;
        box->y0            = ty0 < box->y0 ? ty0 : box->y0;
        box->y1            = ty1 > box->y1 ? ty1 : box->y1;
    }
}

/* fn over [0, n) on the stream's threads: on this one, or split in two with
 * the second half on a thread of the pool, whatever the caller's team is. */
static void on_threads(const struct geist_decknet_stream *s, size_t n, geist_par_fn fn, void *ctx) {
    if (s->n_threads == 1) {
        fn(ctx, 0, n);
        return;
    }
    const size_t team = geist_par_max_threads();
    geist_par_set_max_threads(s->n_threads);
    geist_par_for(n, fn, ctx);
    geist_par_set_max_threads(team);
}

/* Whether reader `who` reads tile t: a slot, or with d->n_slots anything
 * (the worktable strip is given every tile a sample reads in its rectangle,
 * which is its own and whatever slot lies there too). */
static inline bool tile_read(const struct geist_decknet_stream *s, size_t who, size_t t) {
    return who < s->d->n_slots ? (s->slot_tiles[t] >> who & 1U) != 0 : s->used[t] != 0;
}

/* Without gating, a YUV frame: the tiles reader `who` reads as
 * RGB8 into `rgb`, packed (pixels_box), which is the image returned.
 * Neighbouring tiles of a row go in one piece. Every reader has a sample
 * (the strip a pixel, a slot a pooling row), so its rectangle has a tile. */
static struct pixels tiles_convert(const struct geist_decknet_stream *s,
                                   const struct geist_decknet_frame  *f,
                                   size_t                             who,
                                   uint8_t                           *rgb) {
    const struct decknet_box box = s->read_box[who];
    const struct pixels      to  = pixels_box(s, box, rgb);
    for (size_t ty = box.y0; ty <= box.y1; ty++) {
        const size_t row = ty * s->tiles_w;
        const size_t y0  = ty * DECKNET_TILE;
        const size_t y1  = y0 + DECKNET_TILE < f->height ? y0 + DECKNET_TILE : f->height;
        for (size_t tx = box.x0; tx <= box.x1; tx++) {
            if (!tile_read(s, who, row + tx)) {
                continue;
            }
            const size_t x0 = tx * DECKNET_TILE;
            while (tx < box.x1 && tile_read(s, who, row + tx + 1)) {
                tx++;
            }
            const size_t x1 = (tx + 1) * DECKNET_TILE;
            convert(f, x0, x1 < f->width ? x1 : f->width, y0, y1, &to, rgb);
        }
    }
    return to;
}

/* With gating: each tile the network reads is compared with the kept one,
 * and where they differ by more than the threshold the frame's replaces it,
 * in the kept bytes and in s->rgb. s->changed then says, per tile, whether
 * a sample whose first pixel lies in it reads a replaced tile: the tile
 * itself or the one right of it, below it or below right. */
static void tiles_gate(struct geist_decknet_stream *s, const struct geist_decknet_frame *f) {
    const bool          yuv      = f->format != GEIST_DECKNET_FORMAT_RGB8;
    const struct pixels own      = pixels_own(s);
    uint8_t            *kept     = yuv ? s->kept : s->rgb;
    struct block        b[2]     = {};
    const size_t        n_blocks = blocks(f, 0, f->width, 0, 1, b);
    const size_t        row      = b[0].col1; /* the bytes of a kept row: no padding */
    const size_t        n        = s->tiles_w * s->tiles_h;
    const bool          all =
            !s->have_kept || s->kept_format != f->format || (yuv && s->kept_colour != f->colour);
    s->n_tiles_changed = 0;
    for (size_t ty = 0; ty < s->tiles_h; ty++) {
        const uint8_t *used    = s->used + ty * s->tiles_w;
        uint8_t       *changed = s->changed + ty * s->tiles_w;
        const size_t   y0      = ty * DECKNET_TILE;
        const size_t   y1      = y0 + DECKNET_TILE < f->height ? y0 + DECKNET_TILE : f->height;
        for (size_t tx = 0; tx < s->tiles_w; tx++) {
            const size_t x0 = tx * DECKNET_TILE;
            const size_t x1 = x0 + DECKNET_TILE < f->width ? x0 + DECKNET_TILE : f->width;
            (void) blocks(f, x0, x1, y0, y1, b);
            changed[tx] =
                    used[tx] != 0 && (all || tile_differs(f, n_blocks, b, row, s->threshold, kept));
            s->n_tiles_changed += changed[tx];
        }
        /* Taken: neighbouring tiles of a row in one piece, which is what
         * makes a frame that changed all over cost a copy and no more. */
        for (size_t tx = 0; tx < s->tiles_w; tx++) {
            if (changed[tx] == 0) {
                continue;
            }
            const size_t x0 = tx * DECKNET_TILE;
            while (tx + 1 < s->tiles_w && changed[tx + 1] != 0) {
                tx++;
            }
            const size_t x1 =
                    (tx + 1) * DECKNET_TILE < f->width ? (tx + 1) * DECKNET_TILE : f->width;
            (void) blocks(f, x0, x1, y0, y1, b);
            for (size_t i = 0; i < n_blocks; i++) {
                for (size_t r = b[i].row0; r < b[i].row1; r++) {
                    memcpy(kept + r * row + b[i].col0,
                           f->data + r * f->stride + b[i].col0,
                           b[i].col1 - b[i].col0);
                }
            }
            if (yuv) {
                convert(f, x0, x1, y0, y1, &own, s->rgb);
            }
        }
    }
    /* In place: a tile reads the flags right of and below it, set before.
     * And the sums over the flags, row by row. */
    const size_t sum_w = s->tiles_w + 1;
    for (size_t t = 0; t < n; t++) {
        const bool right = t % s->tiles_w + 1 < s->tiles_w, below = t + s->tiles_w < n;
        s->changed[t] |= (uint8_t) ((right ? s->changed[t + 1] : 0) |
                                    (below ? s->changed[t + s->tiles_w] : 0) |
                                    (right && below ? s->changed[t + s->tiles_w + 1] : 0));
        uint32_t *sum = s->changed_sum + (t / s->tiles_w + 1) * sum_w + t % s->tiles_w + 1;
        *sum = s->changed[t] + sum[-1] + sum[-(ptrdiff_t) sum_w] - sum[-(ptrdiff_t) sum_w - 1];
    }
    s->have_kept   = true;
    s->kept_format = f->format;
    s->kept_colour = f->colour;
}

/* ---- sampling --------------------------------------------------------- */

/* table[i] for the i-th crop pixel of the runs, of a crop that is out_w
 * wide, through homography h. */
static void build_table(const struct geist_decknet *d,
                        size_t                      out_w,
                        size_t                      n_runs,
                        const float                 h[static 9],
                        const struct decknet_run    run[static n_runs],
                        struct decknet_sample      *table) {
    for (size_t r = 0; r < n_runs; r++) {
        for (size_t p = run[r].px; p < (size_t) run[r].px + run[r].n; p++, table++) {
            double u, v;
            decknet_sample_at(d->img_w, d->img_h, p % out_w, p / out_w, h, &u, &v);
            *table = (struct decknet_sample) {.x = (uint16_t) u, .y = (uint16_t) v};
            decknet_weights(u, v, table->w);
        }
    }
}

/* The R, G and B bytes that start at byte `first` of v as three lanes, and
 * a zero. */
[[gnu::always_inline]] static inline q4 rgb_lanes(b16 v, bool second) {
    constexpr int z    = 16; /* a byte of the zero vector */
    const b16     zero = {0};
    return second ? (q4) __builtin_shufflevector(
                            v, zero, 3, z, z, z, 4, z, z, z, 5, z, z, z, z, z, z, z)
                  : (q4) __builtin_shufflevector(
                            v, zero, 0, z, z, z, 1, z, z, z, 2, z, z, z, z, z, z, z);
}

/* One sample of an image, a crop pixel: run.c's resample() with the
 * coordinates and the weights from the table, and decknet_blend for the
 * three colours at once. With w00 = 2^24 - w0 - w1 - w2 its sum is
 *   2^24 b00 + w0 (b01 - b00) + w1 (b10 - b00) + w2 (b11 - b00)
 * which in 32 bits, where the differences wrap, is the same number: the sum
 * itself is below 2^32. */
[[gnu::always_inline]] static inline q4 sample_one(const struct decknet_sample *t,
                                                   const struct pixels         *img) {
    const size_t   off = t->y * img->stride + (size_t) t->x * 3 - img->bias;
    const uint8_t *r0  = img->rgb + off;
    const uint8_t *r1  = r0 + img->stride;
    if (off >= img->wide) {
        /* At the end of a frame that has nothing behind its last pixels. */
        return (q4) {decknet_blend(t->w, r0[0], r0[3], r1[0], r1[3]),
                     decknet_blend(t->w, r0[1], r0[4], r1[1], r1[4]),
                     decknet_blend(t->w, r0[2], r0[5], r1[2], r1[5]),
                     0};
    }
    b16 row0, row1;
    q4  w;
    memcpy(&row0, r0, sizeof row0);
    memcpy(&row1, r1, sizeof row1);
    memcpy(&w, t, sizeof w); /* the fourth lane is x and y, and not used */
    const q4 b00 = rgb_lanes(row0, false);
    q4       sum = (b00 << 24) + 128;
    sum += (rgb_lanes(row0, true) - b00) * (q4) {w[0], w[0], w[0], w[0]};
    sum += (rgb_lanes(row1, false) - b00) * (q4) {w[1], w[1], w[1], w[1]};
    sum += (rgb_lanes(row1, true) - b00) * (q4) {w[2], w[2], w[2], w[2]};
    return sum >> 8;
}

/* out[px] for every sample of the table, whose crop pixels are the runs.
 * With `sums` not the samples but their prefix sums along each run
 * (decknet_work.prefix), which takes a pixel more behind a run. */
static void sample(size_t                       n_runs,
                   bool                         sums,
                   const struct decknet_run     run[static n_runs],
                   const struct decknet_sample *table,
                   const struct pixels         *img,
                   uint32_t                    *out) {
    for (size_t r = 0; r < n_runs; r++) {
        uint32_t *o   = out + (size_t) run[r].px * DECKNET_CROP_CH;
        q4        sum = {0, 0, 0, 0};
        for (size_t i = 0; i < run[r].n; i++) {
            const q4 v = sample_one(table++, img);
            memcpy(o + i * DECKNET_CROP_CH, sums ? &sum : &v, sizeof v);
            sum += v;
        }
        if (sums) {
            memcpy(o + (size_t) run[r].n * DECKNET_CROP_CH, &sum, sizeof sum);
        }
    }
}

/* s->box: per chunk of s->tab, the tiles its samples' first pixels lie in. */
static void build_boxes(struct geist_decknet_stream *s, size_t n_tab) {
    for (size_t first = 0; first < n_tab; first += DECKNET_CHUNK) {
        struct decknet_box box = {.x0 = UINT16_MAX, .y0 = UINT16_MAX};
        for (size_t i = first; i < n_tab && i < first + DECKNET_CHUNK; i++) {
            const uint16_t x = (uint16_t) (s->tab[i].x / DECKNET_TILE);
            const uint16_t y = (uint16_t) (s->tab[i].y / DECKNET_TILE);
            box.x0           = x < box.x0 ? x : box.x0;
            box.x1           = x > box.x1 ? x : box.x1;
            box.y0           = y < box.y0 ? y : box.y0;
            box.y1           = y > box.y1 ? y : box.y1;
        }
        s->box[first / DECKNET_CHUNK] = box;
    }
}

/* out[px] for the samples of s->tab from `first` on, whose crop pixels are
 * the runs, that read a replaced tile (s->changed); the others keep what
 * `out` has. Marks in `stale` what it wrote, as decknet_work takes it: per
 * crop pixel on the floor plane, per cell on the others. Returns how many
 * samples it took. */
static size_t sample_changed(const struct geist_decknet_stream *s,
                             enum decknet_plane                 plane,
                             size_t                             first,
                             size_t                             n_runs,
                             const struct decknet_run           run[static n_runs],
                             uint8_t                           *stale,
                             uint32_t                          *out) {
    const struct geist_decknet *d       = s->d;
    const struct pixels         img     = pixels_own(s);
    const size_t                cells_w = (d->crop_w + DECKNET_CELL - 1) / DECKNET_CELL;
    const size_t                sum_w   = s->tiles_w + 1;
    size_t                      taken   = 0;
    size_t                      i       = first;
    for (size_t r = 0; r < n_runs; r++) {
        size_t       px      = run[r].px;
        const size_t run_end = i + run[r].n;
        while (i < run_end) {
            const struct decknet_box *box = &s->box[i / DECKNET_CHUNK];
            size_t                    end = (i / DECKNET_CHUNK + 1) * DECKNET_CHUNK;
            end                           = end < run_end ? end : run_end;
            const uint32_t *lo            = s->changed_sum + box->y0 * sum_w;
            const uint32_t *hi            = s->changed_sum + ((size_t) box->y1 + 1) * sum_w;
            if (hi[box->x1 + 1] - hi[box->x0] - lo[box->x1 + 1] + lo[box->x0] == 0) {
                px += end - i; /* no changed tile in the chunk's rectangle */
                i = end;
                continue;
            }
            for (; i < end; i++, px++) {
                const struct decknet_sample *t = &s->tab[i];
                if (s->changed[t->y / DECKNET_TILE * s->tiles_w + t->x / DECKNET_TILE] == 0) {
                    continue;
                }
                const q4 v = sample_one(t, &img);
                memcpy(out + px * DECKNET_CROP_CH, &v, sizeof v);
                taken++;
                if (plane == DECKNET_PLANE_FLOOR) {
                    stale[px] = 1;
                } else {
                    stale[px / d->crop_w / DECKNET_CELL * cells_w + px % d->crop_w / DECKNET_CELL] =
                            1;
                }
            }
        }
    }
    return taken;
}

/* ---- a GPU ------------------------------------------------------------- */

/* Where the GPU writes: the crops slot by slot and plane by plane, then the
 * worktable strip (<geist_decknet_gpu.h>, "pixels"). */
static size_t gpu_crop(size_t slot, size_t plane) {
    return slot * DECKNET_PLANES + plane;
}

/* The samples in the order the GPU takes them, in pieces of one table and
 * one crop each: the strip and the floor plane of every slot, which is
 * stage 0, then the other planes slot by slot. */
struct gpu_piece {
    size_t                       n, n_runs;
    const struct decknet_run    *run; /* the crop pixels of the samples; nullptr: 0 .. n - 1 */
    size_t                       crop;
    const struct decknet_sample *table;
};

static size_t gpu_n_pieces(const struct geist_decknet_stream *s) {
    return 1 + s->d->n_slots * DECKNET_PLANES;
}

static struct gpu_piece gpu_piece_at(const struct geist_decknet_stream *s, size_t i) {
    const struct geist_decknet *d = s->d;
    if (i == 0) {
        return (struct gpu_piece) {.n = d->bg_px, .crop = gpu_crop(d->n_slots, 0), .table = s->bg};
    }
    /* Pieces 1 .. n_slots are the floors; behind them a slot has the rest
     * of its planes side by side. */
    constexpr size_t rest     = DECKNET_PLANES - 1;
    const bool       on_floor = i <= d->n_slots;
    const size_t     slot     = on_floor ? i - 1 : (i - 1 - d->n_slots) / rest;
    const size_t     plane    = on_floor ? DECKNET_PLANE_FLOOR : 1 + (i - 1 - d->n_slots) % rest;
    const struct decknet_sample *t = s->tab + slot * s->per_slot;
    for (size_t p = 0; p < plane; p++) {
        t += s->n_px[p];
    }
    return (struct gpu_piece) {.n      = s->n_px[plane],
                               .n_runs = s->n_runs[plane],
                               .run    = s->run[plane],
                               .crop   = gpu_crop(slot, plane),
                               .table  = t};
}

/* dst[i]: which pixel of the pixel memory the GPU's sample i writes. */
static void gpu_dst(const struct geist_decknet_stream *s, uint32_t *dst) {
    for (size_t i = 0; i < gpu_n_pieces(s); i++) {
        const struct gpu_piece piece = gpu_piece_at(s, i);
        const size_t           base  = piece.crop * s->d->crop_px;
        for (size_t k = 0; piece.run == nullptr && k < piece.n; k++) {
            *dst++ = (uint32_t) (base + k);
        }
        for (size_t r = 0; r < piece.n_runs; r++) {
            for (size_t k = 0; k < piece.run[r].n; k++) {
                *dst++ = (uint32_t) (base + piece.run[r].px + k);
            }
        }
    }
}

/* The tables of the current homographies (decknet_stream_set_reference) to
 * the GPU, and the rows of a frame they read. */
[[nodiscard]] static enum geist_status gpu_tables(struct geist_decknet_stream *s) {
    const struct geist_decknet      *d  = s->d;
    struct geist_decknet_gpu_sample *to = s->gpu_mem.samples;
    for (size_t i = 0; i < gpu_n_pieces(s); i++) {
        const struct gpu_piece piece = gpu_piece_at(s, i);
        memcpy(to, piece.table, piece.n * sizeof piece.table[0]);
        to += piece.n;
    }
    size_t y0 = SIZE_MAX, y1 = 0;
    for (size_t who = 0; who <= d->n_slots; who++) {
        y0 = s->read_box[who].y0 < y0 ? s->read_box[who].y0 : y0;
        y1 = s->read_box[who].y1 > y1 ? s->read_box[who].y1 : y1;
    }
    s->gpu_y0 = y0 * DECKNET_TILE;
    s->gpu_y1 = (y1 + 1) * DECKNET_TILE < d->img_h ? (y1 + 1) * DECKNET_TILE : d->img_h;
    return s->gpu_ops->samples_written(s->gpu);
}

/* A kernel as the GPU takes it: the taps of each part, then dy dx per tap. */
static int32_t *gpu_taps(const struct decknet_kernel *k) {
    const size_t n_taps = k->k * k->k; /* at most DECKNET_MAX_KERNEL squared */
    int32_t     *taps   = heap_alloc_array_aligned(int32_t, k->n_parts + 2 * n_taps);
    if (taps == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < k->n_parts; i++) {
        taps[i] = (int32_t) k->part[i].n_taps;
    }
    for (size_t i = 0; i < n_taps; i++) {
        taps[k->n_parts + 2 * i]     = k->tap[i].dy;
        taps[k->n_parts + 2 * i + 1] = k->tap[i].dx;
    }
    return taps;
}

/* [*lo, *hi) widened to the crop pixels `p` reads. */
static void pool_span(const struct decknet_pool *p, size_t *lo, size_t *hi) {
    for (size_t r = 0; r < p->n_rows; r++) {
        *lo = p->px[r] < *lo ? p->px[r] : *lo;
        *hi = p->px[r] >= *hi ? (size_t) p->px[r] + 1 : *hi;
    }
}

/* What the CPU reads of the GPU's pixels (push_gpu), for a backend that has
 * to copy it back: after stage 0 the strip and, of each floor crop, the
 * pixels of its three pools (read[0 .. n_slots]); after stage 1, of each tip
 * crop, those of the white unit's pool (the n_slots spans behind). The
 * other planes are read as their kernels' sums. */
static void gpu_read_spans(const struct geist_decknet_stream *s,
                           struct geist_decknet_gpu_span     *read) {
    const struct geist_decknet *d = s->d;
    size_t floor_lo = d->crop_px, floor_hi = 0, tip_lo = d->crop_px, tip_hi = 0;
    pool_span(&d->occ_front, &floor_lo, &floor_hi);
    pool_span(&d->occ_rear, &floor_lo, &floor_hi);
    pool_span(&d->wall_pool, &floor_lo, &floor_hi);
    pool_span(&d->on, &tip_lo, &tip_hi);
    read[0] = (struct geist_decknet_gpu_span) {
            .first = (uint32_t) (gpu_crop(d->n_slots, 0) * d->crop_px), .n = (uint32_t) d->bg_px};
    for (size_t slot = 0; slot < d->n_slots; slot++) {
        const size_t floor = gpu_crop(slot, DECKNET_PLANE_FLOOR) * d->crop_px;
        const size_t tip   = gpu_crop(slot, DECKNET_PLANE_TIP) * d->crop_px;
        read[1 + slot]     = (struct geist_decknet_gpu_span) {
                .first = (uint32_t) (floor + (floor_lo < floor_hi ? floor_lo : 0)),
                .n     = (uint32_t) (floor_lo < floor_hi ? floor_hi - floor_lo : 0)};
        read[1 + d->n_slots + slot] = (struct geist_decknet_gpu_span) {
                .first = (uint32_t) (tip + (tip_lo < tip_hi ? tip_lo : 0)),
                .n     = (uint32_t) (tip_lo < tip_hi ? tip_hi - tip_lo : 0)};
    }
}

enum geist_status decknet_stream_set_gpu(struct geist_decknet_stream        *s,
                                         const struct geist_decknet_gpu_ops *ops,
                                         struct geist_backend               *be) {
    const struct geist_decknet *d = s->d;
    if (ops == nullptr) {
        if (s->gpu_ops != nullptr) {
            s->gpu_ops->destroy(s->gpu);
        }
        s->gpu_ops = nullptr;
        s->gpu     = nullptr;
        return GEIST_OK;
    }
    /* What the GPU indexes with 32 bits: the pixels, the samples and the
     * bytes of a frame. A model whose image or crops are larger stays on
     * the CPU. */
    const size_t n_crops  = d->n_slots * DECKNET_PLANES; /* at most 5 x DECKNET_MAX_SLOTS */
    size_t       n_pixels = 0, n_samples = 0, n_frame = 0;
    if (ckd_mul(&n_pixels, n_crops, d->crop_px) || ckd_add(&n_pixels, n_pixels, d->bg_px) ||
        ckd_mul(&n_samples, d->n_slots, s->per_slot) || ckd_add(&n_samples, n_samples, d->bg_px) ||
        ckd_mul(&n_frame, d->img_w * d->img_h, 3) || n_pixels > UINT32_MAX ||
        n_samples > UINT32_MAX || n_frame > UINT32_MAX - 3) {
        return GEIST_E_UNSUPPORTED;
    }
    constexpr size_t                    n_kernel_planes = DECKNET_PLANES - 1;
    GEIST_CLEANUP(heap_freep) uint32_t *dst = heap_alloc_array_aligned(uint32_t, n_samples);
    GEIST_CLEANUP(heap_freep) uint32_t *crops =
            heap_alloc_array_aligned(uint32_t, d->n_slots *(n_kernel_planes + 1));
    GEIST_CLEANUP(heap_freep) int32_t                       *blob = gpu_taps(&d->blob);
    GEIST_CLEANUP(heap_freep) int32_t                       *tube = gpu_taps(&d->tube_kernel);
    GEIST_CLEANUP(heap_freep) struct geist_decknet_gpu_span *read =
            heap_alloc_array_aligned(struct geist_decknet_gpu_span, 1 + 2 * d->n_slots);
    if (dst == nullptr || crops == nullptr || blob == nullptr || tube == nullptr ||
        read == nullptr) {
        return GEIST_E_OOM;
    }
    gpu_dst(s, dst);
    gpu_read_spans(s, read);
    /* The blob kernel on every plane but the floor, the tube kernel on the
     * hole plane; the sums come back in this order (push_gpu). */
    uint32_t *blob_crops = crops, *tube_crops = crops + d->n_slots * n_kernel_planes;
    for (size_t slot = 0; slot < d->n_slots; slot++) {
        for (size_t p = DECKNET_PLANE_FLOOR + 1; p < DECKNET_PLANES; p++) {
            blob_crops[slot * n_kernel_planes + p - 1] = (uint32_t) gpu_crop(slot, p);
        }
        tube_crops[slot] = (uint32_t) gpu_crop(slot, DECKNET_PLANE_HOLE);
    }
    const struct geist_decknet_gpu_plan plan = {
            .frame_bytes = n_frame,
            .crop_w      = d->crop_w,
            .crop_h      = d->crop_h,
            .n_pixels    = n_pixels,
            .n_samples   = n_samples,
            .n_first     = d->bg_px + d->n_slots * s->n_px[DECKNET_PLANE_FLOOR],
            .dst         = dst,
            .kernel      = {{.n_parts = d->blob.n_parts,
                             .n_taps  = d->blob.k * d->blob.k,
                             .taps    = blob,
                             .n_px    = d->n_blob_px,
                             .px      = d->blob_px,
                             .n_crops = d->n_slots * n_kernel_planes,
                             .crop    = blob_crops},
                            {.n_parts = d->tube_kernel.n_parts,
                             .n_taps  = d->tube_kernel.k * d->tube_kernel.k,
                             .taps    = tube,
                             .n_px    = d->n_tube_px,
                             .px      = d->tube_px,
                             .n_crops = d->n_slots,
                             .crop    = tube_crops}},
            .n_read      = {1 + d->n_slots, d->n_slots},
            .read        = {read, read + 1 + d->n_slots},
    };
    struct geist_decknet_gpu       *gpu = nullptr;
    struct geist_decknet_gpu_memory mem = {0};
    const enum geist_status         st  = ops->create(be, &plan, &gpu, &mem);
    if (st != GEIST_OK) {
        return st;
    }
    if (s->gpu_ops != nullptr) {
        s->gpu_ops->destroy(s->gpu);
    }
    s->gpu_ops = ops;
    s->gpu     = gpu;
    s->gpu_mem = mem;
    /* A reference that is set already has its tables; a stale one's are
     * replaced by the next decknet_stream_set_reference, before any frame. */
    return s->have_ref ? gpu_tables(s) : GEIST_OK;
}

/* The rows of the frame a sample reads into the GPU's frame memory, without
 * their padding: a row is `row` bytes there. */
static void
gpu_upload(const struct geist_decknet_stream *s, const struct geist_decknet_frame *f, size_t row) {
    struct block b[2]     = {};
    const size_t n_blocks = blocks(f, 0, f->width, s->gpu_y0, s->gpu_y1, b);
    for (size_t i = 0; i < n_blocks; i++) {
        if (f->stride == row) {
            memcpy(s->gpu_mem.frame + b[i].row0 * row,
                   f->data + b[i].row0 * row,
                   (b[i].row1 - b[i].row0) * row);
            continue;
        }
        for (size_t r = b[i].row0; r < b[i].row1; r++) {
            memcpy(s->gpu_mem.frame + r * row, f->data + r * f->stride, row);
        }
    }
}

struct gpu_job {
    struct geist_decknet_stream *s;
    const float                 *ga;
};

/* The slots of part `part` of the stream's n_threads parts, as slots(). */
static inline size_t gpu_slot0(const struct geist_decknet_stream *s, size_t part) {
    return part * s->d->n_slots / s->n_threads;
}

/* Stage 0: the floor plane of each slot from the pixels the GPU sampled. */
static void gpu_floors(void *ctx, size_t begin, size_t end) {
    const struct gpu_job        *job = ctx;
    struct geist_decknet_stream *s   = job->s;
    struct geist_decknet        *d   = s->d;
    for (size_t part = begin; part < end; part++) {
        struct decknet_ws *ws = part == 0 ? &d->ws : &s->ws2;
        for (size_t slot = gpu_slot0(s, part); slot < gpu_slot0(s, part + 1); slot++) {
            const struct decknet_work work = {
                    .crop     = s->gpu_mem.pixels +
                                gpu_crop(slot, DECKNET_PLANE_FLOOR) * d->crop_px * DECKNET_CROP_CH,
                    .crop_ref = s->floor_ref + slot * d->crop_px * 3,
                    .mix      = ws->mix,
            };
            decknet_plane(d, ws, slot, DECKNET_PLANE_FLOOR, job->ga, &work);
        }
    }
}

/* Stage 1: the other planes, whose kernels the GPU summed part by part. */
static void gpu_planes(void *ctx, size_t begin, size_t end) {
    const struct gpu_job        *job    = ctx;
    struct geist_decknet_stream *s      = job->s;
    struct geist_decknet        *d      = s->d;
    constexpr size_t             planes = DECKNET_PLANES - 1;
    const size_t                 n_blob = d->n_blob_px * d->blob.n_parts * DECKNET_CROP_CH;
    const size_t                 n_tube = d->n_tube_px * d->tube_kernel.n_parts * DECKNET_CROP_CH;
    const uint32_t              *tube   = s->gpu_mem.sums + d->n_slots * planes * n_blob;
    for (size_t part = begin; part < end; part++) {
        struct decknet_ws *ws = part == 0 ? &d->ws : &s->ws2;
        for (size_t slot = gpu_slot0(s, part); slot < gpu_slot0(s, part + 1); slot++) {
            for (size_t p = DECKNET_PLANE_FLOOR + 1; p < DECKNET_PLANES; p++) {
                /* `kept` without `stale`: the sums are there, no kernel is
                 * computed. The crop is read for the white unit. */
                const struct decknet_work work = {
                        .crop = s->gpu_mem.pixels +
                                gpu_crop(slot, p) * d->crop_px * DECKNET_CROP_CH,
                        .sums = ws->sums,
                        .tube = ws->tube_sums,
                        .kept = true,
                };
                decknet_kernel_parts(&d->blob,
                                     d->n_blob_px,
                                     s->gpu_mem.sums + (slot * planes + p - 1) * n_blob,
                                     ws->sums);
                if (p == DECKNET_PLANE_HOLE) {
                    decknet_kernel_parts(
                            &d->tube_kernel, d->n_tube_px, tube + slot * n_tube, ws->tube_sums);
                }
                decknet_plane(d, ws, slot, (enum decknet_plane) p, job->ga, &work);
            }
        }
    }
}

/* A frame that is not gated, with its integer part on the GPU. The frame's
 * rows are copied to the GPU's memory and both stages submitted; then this
 * thread waits for the first, which is short, and computes the gains and
 * the floor planes from it while the GPU samples the other planes and sums
 * their kernels. */
[[nodiscard]] static enum geist_status push_gpu(struct geist_decknet_stream      *s,
                                                const struct geist_decknet_frame *frame) {
    struct geist_decknet *d   = s->d;
    size_t                row = 0, rows = 0;
    (void) frame_rows(frame, &row, &rows); /* frame_ok accepted it */
    gpu_upload(s, frame, row);
    enum geist_status st =
            s->gpu_ops->submit(s->gpu, frame->format, frame->colour, row, frame->height);
    if (st != GEIST_OK) {
        return st; /* nothing was computed: the last result stays */
    }
    float          ga[3], gb[3];
    struct gpu_job job = {.s = s, .ga = ga};
    st                 = s->gpu_ops->wait(s->gpu, 0);
    if (st == GEIST_OK) {
        const size_t strip = gpu_crop(d->n_slots, 0) * d->crop_px * DECKNET_CROP_CH;
        decknet_strip(d, s->gpu_mem.pixels + strip, d->bg_a);
        decknet_gains(d, d->bg_a, s->bg_ref, ga, gb);
        if (d->result.checkable) {
            on_threads(s, s->n_threads, gpu_floors, &job);
        }
    }
    const enum geist_status st1 = s->gpu_ops->wait(s->gpu, 1);
    st                          = st != GEIST_OK ? st : st1;
    if (st != GEIST_OK) {
        /* Half a result is none. */
        s->have_result = false;
        return st;
    }
    if (d->result.checkable) {
        on_threads(s, s->n_threads, gpu_planes, &job);
    }
    s->n_tiles_changed = s->n_tiles;
    s->n_samples       = d->bg_px + d->n_slots * s->per_slot;
    s->timestamp       = frame->timestamp;
    s->have_result     = true;
    s->n_frames++;
    return GEIST_OK;
}

/* ---- entry points ----------------------------------------------------- */

void decknet_stream_destroy(struct geist_decknet_stream *s) {
    if (s == nullptr) {
        return;
    }
    if (s->gpu_ops != nullptr) {
        s->gpu_ops->destroy(s->gpu);
    }
    for (size_t p = 0; p < DECKNET_PLANES; p++) {
        heap_freep(&s->run[p]);
    }
    heap_freep(&s->bg);
    heap_freep(&s->tab);
    heap_freep(&s->bg_ref);
    heap_freep(&s->floor_ref);
    heap_freep(&s->used);
    heap_freep(&s->slot_tiles);
    heap_freep(&s->read_box);
    heap_freep(&s->rgb2);
    heap_freep(&s->changed);
    heap_freep(&s->changed_sum);
    heap_freep(&s->box);
    heap_freep(&s->rgb);
    heap_freep(&s->kept);
    heap_freep(&s->crop);
    heap_freep(&s->sums);
    heap_freep(&s->tube);
    heap_freep(&s->mix);
    heap_freep(&s->stale);
    decknet_ws_free(&s->ws2);
    heap_freep(&s);
}

/* s->run[plane]: the crop pixels that plane reads. */
static bool plane_pixels(struct geist_decknet_stream *s, size_t plane, uint8_t *mark) {
    const struct geist_decknet *d = s->d;
    decknet_plane_pixels(d, (enum decknet_plane) plane, mark);
    size_t n = 0, n_runs = 0;
    for (size_t i = 0; i < d->crop_px; i++) {
        n += mark[i];
        n_runs += mark[i] != 0 && (i == 0 || mark[i - 1] == 0);
    }
    s->run[plane] =
            heap_alloc_array_aligned(struct decknet_run, n_runs); /* >= 1: a pool has a row */
    if (s->run[plane] == nullptr) {
        return false;
    }
    struct decknet_run *run = s->run[plane] - 1;
    for (size_t i = 0; i < d->crop_px; i++) {
        if (mark[i] == 0) {
            continue;
        }
        if (i == 0 || mark[i - 1] == 0) {
            *++run = (struct decknet_run) {.px = (uint32_t) i};
        }
        run->n++;
    }
    s->n_px[plane]   = n;
    s->n_runs[plane] = n_runs;
    s->per_slot += n; /* at most DECKNET_PLANES * DECKNET_MAX_PIXELS */
    return true;
}

enum geist_status decknet_stream_create(struct geist_decknet         *d,
                                        struct geist_decknet_stream **out) {
    *out                           = nullptr;
    struct geist_decknet_stream *s = heap_calloc_array_aligned(struct geist_decknet_stream, 1);
    if (s == nullptr) {
        return GEIST_E_OOM;
    }
    s->d         = d;
    s->n_threads = 1;
    s->spans     = d->blob.span != nullptr && d->tube_kernel.span != nullptr;
    atomic_init(&s->offered, nullptr);
    atomic_init(&s->n_dropped, 0);
    s->tiles_w                              = (d->img_w + DECKNET_TILE - 1) / DECKNET_TILE;
    s->tiles_h                              = (d->img_h + DECKNET_TILE - 1) / DECKNET_TILE;
    GEIST_CLEANUP(heap_freep) uint8_t *mark = heap_alloc_array_aligned(uint8_t, d->crop_px);
    bool                               ok   = mark != nullptr;
    for (size_t p = 0; ok && p < DECKNET_PLANES; p++) {
        ok = plane_pixels(s, p, mark);
    }
    /* img_w and img_h are at most DECKNET_MAX_IMAGE_DIM each. */
    size_t n_tab = 0, n_floor = 0, n_bg = 0, n_rgb = 0;
    ok = ok && !ckd_mul(&n_tab, d->n_slots, s->per_slot) && !ckd_mul(&n_floor, d->crop_px, 3) &&
         !ckd_mul(&n_floor, n_floor, d->n_slots) && !ckd_mul(&n_bg, d->bg_px, 3) &&
         !ckd_mul(&n_rgb, d->img_w * d->img_h, 3) && !ckd_add(&n_rgb, n_rgb, DECKNET_SAMPLE_BYTES);
    if (ok) {
        s->bg     = heap_alloc_array_aligned(struct decknet_sample, d->bg_px);
        s->tab    = heap_alloc_array_aligned(struct decknet_sample, n_tab);
        s->bg_ref = heap_alloc_array_aligned(float, n_bg);
        /* Zeroed: decknet_plane reads it at the floor pixels only, which
         * the reference writes, but nothing here is ever uninitialised. */
        s->floor_ref  = heap_calloc_array_aligned(float, n_floor);
        s->used       = heap_calloc_array_aligned(uint8_t, s->tiles_w * s->tiles_h);
        s->slot_tiles = heap_calloc_array_aligned(uint64_t, s->tiles_w * s->tiles_h);
        s->read_box   = heap_alloc_array_aligned(struct decknet_box, d->n_slots + 1);
        s->changed    = heap_calloc_array_aligned(uint8_t, s->tiles_w * s->tiles_h);
        /* Zeroed: its first row and column stay 0. */
        s->changed_sum = heap_calloc_array_aligned(uint32_t, (s->tiles_w + 1) * (s->tiles_h + 1));
        s->box         = heap_alloc_array_aligned(struct decknet_box, n_tab / DECKNET_CHUNK + 1);
        s->rgb         = heap_calloc_array_aligned(uint8_t, n_rgb);
        s->rgb2        = heap_calloc_array_aligned(uint8_t, n_rgb);
        ok = s->bg != nullptr && s->tab != nullptr && s->bg_ref != nullptr &&
             s->floor_ref != nullptr && s->used != nullptr && s->slot_tiles != nullptr &&
             s->read_box != nullptr && s->rgb2 != nullptr && s->changed != nullptr &&
             s->changed_sum != nullptr && s->box != nullptr && s->rgb != nullptr &&
             decknet_ws_alloc(d, &s->ws2);
    }
    if (!ok) {
        decknet_stream_destroy(s);
        return GEIST_E_OOM;
    }
    *out = s;
    return GEIST_OK;
}

enum geist_status
decknet_stream_set_gate(struct geist_decknet_stream *s, bool on, uint32_t threshold) {
    const struct geist_decknet *d = s->d;
    if (on && s->crop == nullptr) {
        /* What a gated stream keeps from frame to frame: the bytes of an
         * YUV frame (two per pixel at most; RGB8 is kept in
         * s->rgb), and per slot and plane the crop and what decknet_plane
         * computes from it alone. */
        size_t n_kept = 0, n_crop = 0, n_sums = 0, n_tube = 0, n_mix = 0, n_stale = 0;
        if (ckd_mul(&n_kept, d->img_w * d->img_h, 2) ||
            ckd_mul(&n_stale, d->crop_px, DECKNET_STREAM_THREADS) ||
            ckd_mul(&n_crop, d->crop_px, DECKNET_CROP_CH * DECKNET_PLANES) ||
            ckd_mul(&n_crop, n_crop, d->n_slots) ||
            ckd_mul(&n_sums, d->n_blob_px, 3 * (DECKNET_PLANES - 1)) ||
            ckd_mul(&n_sums, n_sums, d->n_slots) || ckd_mul(&n_tube, d->n_tube_px, 3) ||
            ckd_mul(&n_tube, n_tube, d->n_slots) || ckd_mul(&n_mix, d->n_mix, d->n_slots)) {
            return GEIST_E_OOM;
        }
        s->kept  = heap_calloc_array_aligned(uint8_t, n_kept);
        s->crop  = heap_calloc_array_aligned(uint32_t, n_crop);
        s->sums  = heap_calloc_array_aligned(double, n_sums);
        s->tube  = heap_calloc_array_aligned(double, n_tube);
        s->mix   = heap_calloc_array_aligned(float, n_mix);
        s->stale = heap_calloc_array_aligned(uint8_t, n_stale);
        if (s->kept == nullptr || s->crop == nullptr || s->sums == nullptr || s->tube == nullptr ||
            s->mix == nullptr || s->stale == nullptr) {
            heap_freep(&s->kept);
            heap_freep(&s->crop);
            heap_freep(&s->sums);
            heap_freep(&s->tube);
            heap_freep(&s->mix);
            heap_freep(&s->stale);
            return GEIST_E_OOM;
        }
    }
    s->gate      = on;
    s->threshold = threshold;
    s->have_kept = false; /* the next frame is taken whole */
    s->cached    = false;
    return GEIST_OK;
}

enum geist_status decknet_stream_set_threads(struct geist_decknet_stream *s, size_t n) {
    if (n < 1 || n > DECKNET_STREAM_THREADS) {
        return GEIST_E_INVALID_ARG;
    }
    s->n_threads = n;
    return GEIST_OK;
}

enum geist_status decknet_stream_set_reference(struct geist_decknet_stream      *s,
                                               const struct geist_decknet_frame *ref) {
    const struct geist_decknet *d = s->d;
    if (!frame_ok(d, ref)) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t              n_floor = s->n_runs[DECKNET_PLANE_FLOOR];
    const struct decknet_run *floor   = s->run[DECKNET_PLANE_FLOOR];
    const struct decknet_run  strip   = {.px = 0, .n = (uint32_t) d->bg_px};
    struct pixels             img     = pixels_own(s);
    if (ref->format == GEIST_DECKNET_FORMAT_RGB8) {
        img = pixels_of(ref);
    } else {
        /* Once, so all of it: which tiles its own homographies read is not
         * worth working out. */
        convert(ref, 0, ref->width, 0, ref->height, &img, s->rgb);
    }
    float gb[3];
    /* The empty deck through its own homographies; the tables are the
     * current image's again below. */
    build_table(d, d->bg_w, 1, d->h_ref, &strip, s->bg);
    sample(1, false, &strip, s->bg, &img, d->bg_q);
    decknet_strip(d, d->bg_q, s->bg_ref);
    decknet_gain(d, s->bg_ref, gb);
    const double scale[3] = {decknet_scale(gb[0]), decknet_scale(gb[1]), decknet_scale(gb[2])};
    for (size_t slot = 0; slot < d->n_slots; slot++) {
        const float *h   = d->h_ref + (1 + DECKNET_PLANES * slot) * 9;
        float       *out = s->floor_ref + slot * d->crop_px * 3;
        build_table(d, d->crop_w, n_floor, h, floor, s->tab);
        sample(n_floor, false, floor, s->tab, &img, d->ws.crop);
        for (size_t r = 0; r < n_floor; r++) {
            for (size_t px = floor[r].px; px < (size_t) floor[r].px + floor[r].n; px++) {
                for (size_t c = 0; c < 3; c++) {
                    out[px * 3 + c] = decknet_unit(d->ws.crop[px * DECKNET_CROP_CH + c], scale[c]);
                }
            }
        }
    }
    memset(s->used, 0, s->tiles_w * s->tiles_h);
    memset(s->slot_tiles, 0, s->tiles_w * s->tiles_h * sizeof s->slot_tiles[0]);
    for (size_t who = 0; who <= d->n_slots; who++) {
        s->read_box[who] = (struct decknet_box) {.x0 = UINT16_MAX, .y0 = UINT16_MAX};
    }
    build_table(d, d->bg_w, 1, d->h_cur, &strip, s->bg);
    mark_tiles(s, d->bg_px, d->n_slots, s->bg);
    struct decknet_sample *t = s->tab;
    for (size_t slot = 0; slot < d->n_slots; slot++) {
        for (size_t p = 0; p < DECKNET_PLANES; p++) {
            const float *h = d->h_cur + (1 + DECKNET_PLANES * slot + p) * 9;
            build_table(d, d->crop_w, s->n_runs[p], h, s->run[p], t);
            mark_tiles(s, s->n_px[p], slot, t);
            t += s->n_px[p];
        }
    }
    build_boxes(s, d->n_slots * s->per_slot);
    s->n_tiles = 0;
    for (size_t i = 0; i < s->tiles_w * s->tiles_h; i++) {
        s->n_tiles += s->used[i];
    }
    s->h_gen     = d->h_gen;
    s->have_ref  = true;
    s->have_kept = false; /* s->rgb may hold the empty deck, and the tiles are other ones */
    s->cached    = false;
    if (s->gpu_ops != nullptr) {
        const enum geist_status st = gpu_tables(s);
        if (st != GEIST_OK) {
            s->have_ref = false; /* the GPU's tables are neither the old ones nor these */
            return st;
        }
    }
    return GEIST_OK;
}

/* What the slots of one frame are computed from. */
struct slots_job {
    struct geist_decknet_stream *s;
    struct pixels                img; /* the frame, or s->rgb */
    /* nullptr, or the YUV frame of which each slot converts what
     * it reads, in place of `img`. */
    const struct geist_decknet_frame *convert;
    const float                      *ga;
    /* A gated stream keeps what this frame computes; the caches hold the
     * crops of the frame before; and the 1x1 values of the floor were
     * computed with this frame's gains. */
    bool   keep, cached, same_gain;
    size_t n_samples[DECKNET_STREAM_THREADS];
};

/* Parts [begin, end) of the stream's n_threads parts: part i is the slots
 * from i x n_slots / n_threads on, computed in workspace i. A slot writes
 * its own outputs and its own caches, so two parts may run at once. */
static void slots(void *ctx, size_t begin, size_t end) {
    struct slots_job            *job     = ctx;
    struct geist_decknet_stream *s       = job->s;
    struct geist_decknet        *d       = s->d;
    const size_t                 n_cells = (d->crop_w + DECKNET_CELL - 1) / DECKNET_CELL *
                                           ((d->crop_h + DECKNET_CELL - 1) / DECKNET_CELL);
    for (size_t part = begin; part < end; part++) {
        struct decknet_ws *ws    = part == 0 ? &d->ws : &s->ws2;
        uint8_t           *stale = job->keep ? s->stale + part * d->crop_px : nullptr;
        size_t             n     = 0;
        for (size_t slot = part * d->n_slots / s->n_threads;
             slot < (part + 1) * d->n_slots / s->n_threads;
             slot++) {
            /* What the slot reads, converted by the thread that samples it
             * and into the memory it converted the slot before into. */
            const struct pixels img =
                    job->convert == nullptr
                            ? job->img
                            : tiles_convert(s, job->convert, slot, part == 0 ? s->rgb : s->rgb2);
            const struct decknet_sample *t    = s->tab + slot * s->per_slot;
            struct decknet_work          work = {.crop     = ws->crop,
                                                 .crop_ref = s->floor_ref + slot * d->crop_px * 3,
                                                 .mix      = ws->mix,
                                                 .sums     = ws->sums,
                                                 .tube     = ws->tube_sums};
            if (job->keep) {
                work.mix  = s->mix + slot * d->n_mix;
                work.tube = s->tube + slot * d->n_tube_px * 3;
            }
            for (size_t p = 0; p < DECKNET_PLANES; p++) {
                const enum decknet_plane plane = (enum decknet_plane) p;
                size_t                   taken = s->n_px[p];
                /* A plane with kernels, and both have spans: its crop as
                 * prefix sums, unless the stream keeps the crop. */
                const bool sums = !job->keep && plane != DECKNET_PLANE_FLOOR && s->spans;
                work.crop       = sums ? nullptr : ws->crop;
                work.prefix     = sums ? ws->crop : nullptr;
                if (!job->keep) {
                    sample(s->n_runs[p], sums, s->run[p], t, &img, ws->crop);
                } else {
                    uint32_t *crop =
                            s->crop + (slot * DECKNET_PLANES + p) * d->crop_px * DECKNET_CROP_CH;
                    work.crop = crop;
                    if (plane != DECKNET_PLANE_FLOOR) {
                        work.sums =
                                s->sums + (slot * (DECKNET_PLANES - 1) + p - 1) * d->n_blob_px * 3;
                    }
                    work.kept = job->cached && (plane != DECKNET_PLANE_FLOOR || job->same_gain);
                    if (job->cached) {
                        taken      = sample_changed(s,
                                                    plane,
                                                    (size_t) (t - s->tab),
                                                    s->n_runs[p],
                                                    s->run[p],
                                                    stale,
                                                    crop);
                        work.stale = taken > 0 ? stale : nullptr;
                    } else {
                        sample(s->n_runs[p], false, s->run[p], t, &img, crop);
                    }
                }
                decknet_plane(d, ws, slot, plane, job->ga, &work);
                if (job->cached && taken > 0) {
                    memset(stale, 0, plane == DECKNET_PLANE_FLOOR ? d->crop_px : n_cells);
                }
                n += taken;
                t += s->n_px[p];
            }
        }
        job->n_samples[part] = n;
    }
}

enum geist_status decknet_stream_push_frame(struct geist_decknet_stream      *s,
                                            const struct geist_decknet_frame *frame) {
    struct geist_decknet *d = s->d;
    if (!frame_ok(d, frame)) {
        return GEIST_E_INVALID_ARG;
    }
    if (!s->have_ref || s->h_gen != d->h_gen) {
        return GEIST_E_INVALID_STATE;
    }
    /* A gated frame is the CPU's also where a GPU is set: most of it is
     * kept, and what is left is less than a round trip to the GPU. */
    if (s->gpu_ops != nullptr && !s->gate) {
        return push_gpu(s, frame);
    }
    const struct decknet_run strip = {.px = 0, .n = (uint32_t) d->bg_px};
    struct pixels            img   = pixels_own(s);
    s->n_tiles_changed             = s->n_tiles;
    /* A YUV frame that is not gated is not converted as a frame:
     * the worktable strip's tiles here, and each slot's by the thread that
     * samples the slot (slots()), packed and always into the same memory:
     * 0.8 MB that the core has just written, where an image of the frame
     * is 6 MB that a region before wrote (docs/DECKNET.md, "Where the
     * conversion's 2 ms went"). A tile two slots read is converted twice. */
    const bool by_slot = !s->gate && frame->format != GEIST_DECKNET_FORMAT_RGB8;
    if (s->gate) {
        tiles_gate(s, frame);
    } else if (frame->format == GEIST_DECKNET_FORMAT_RGB8) {
        img = pixels_of(frame);
    } else {
        img = tiles_convert(s, frame, d->n_slots, s->rgb);
    }
    float ga[3], gb[3];
    sample(1, false, &strip, s->bg, &img, d->bg_q);
    decknet_strip(d, d->bg_q, d->bg_a);
    decknet_gains(d, d->bg_a, s->bg_ref, ga, gb);
    s->n_samples = d->bg_px;
    /* A frame that changed nearly all over is computed as one that is not
     * gated, from the kept tiles (which are the frame's now) and in the
     * workspace: keeping every crop of it costs more than computing it, and
     * the next frame that mostly stands still fills the caches again. */
    const bool whole = s->gate && s->n_tiles_changed * DECKNET_GATE_WHOLE_DEN >
                                          s->n_tiles * DECKNET_GATE_WHOLE_NUM;
    if (d->result.checkable) {
        const bool       cached = s->gate && s->cached && !whole;
        struct slots_job job    = {.s         = s,
                                   .img       = img,
                                   .convert   = by_slot ? frame : nullptr,
                                   .ga        = ga,
                                   .keep      = s->gate && !whole,
                                   .cached    = cached,
                                   .same_gain = cached && memcmp(ga, s->mix_ga, sizeof ga) == 0};
        on_threads(s, s->n_threads, slots, &job);
        for (size_t part = 0; part < s->n_threads; part++) {
            s->n_samples += job.n_samples[part];
        }
    }
    /* A pair that is not checkable had no slot sampled: the caches are of
     * an earlier frame, but the kept tiles moved on. */
    s->cached = s->gate && d->result.checkable && !whole;
    memcpy(s->mix_ga, ga, sizeof ga);
    s->timestamp   = frame->timestamp;
    s->have_result = true;
    s->n_frames++;
    return GEIST_OK;
}

/* ---- a queue of one --------------------------------------------------- */

enum geist_status decknet_stream_offer(struct geist_decknet_stream       *s,
                                       const struct geist_decknet_frame  *frame,
                                       const struct geist_decknet_frame **dropped) {
    *dropped = nullptr;
    /* Of the stream this reads the model's image size and nothing else,
     * which no call changes: the thread that computes may be anywhere. */
    if (!frame_ok(s->d, frame)) {
        return GEIST_E_INVALID_ARG;
    }
    *dropped = atomic_exchange(&s->offered, frame);
    if (*dropped != nullptr) {
        atomic_fetch_add(&s->n_dropped, 1);
    }
    return GEIST_OK;
}

enum geist_status decknet_stream_push_offered(struct geist_decknet_stream       *s,
                                              const struct geist_decknet_frame **done) {
    *done = atomic_exchange(&s->offered, nullptr);
    return *done == nullptr ? GEIST_OK : decknet_stream_push_frame(s, *done);
}

uint64_t decknet_stream_dropped(const struct geist_decknet_stream *s) {
    return atomic_load(&s->n_dropped);
}
