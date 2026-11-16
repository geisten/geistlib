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
 *                   place; of an NV12 or YUYV frame those tiles are
 *                   converted, each pixel once, and sampled then;
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
 *   two threads     optional (decknet_stream_set_threads): the slots are
 *                   independent of each other, so one half of them is
 *                   computed on a second thread, in a second workspace.
 *
 * Everything is allocated in decknet_stream_create and, for gating, in the
 * first decknet_stream_set_gate that turns it on; the reference and the
 * frames allocate nothing.
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
 * `stride` it has in all. False for a format that is not one of the three
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
        !frame_rows(f, &row, &rows) || f->stride < row) {
        return false;
    }
    /* The last row starts at (rows - 1) * stride and is `row` bytes long. */
    return !ckd_mul(&last, rows - 1, f->stride) && !ckd_add(&need, last, row) && need <= f->n_bytes;
}

constexpr size_t CONVERT_CHUNK = 64;

static inline uint8_t clamp8(int16_t v) {
    return (uint8_t) (v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* n pixels of one row as RGB8, n even and at most CONVERT_CHUNK: pixel i has
 * the luma luma[i x luma_step], and the pair i / 2 the chroma
 * cb[(i / 2) x chroma_step] and cr[(i / 2) x chroma_step].
 *
 * BT.601 limited range, the formula of docs/DECKNET.md ("Stream"):
 *   c = 298 (Y - 16) + 128
 *   R = clamp((c + 409 (V - 128)) >> 8)
 *   G = clamp((c - 100 (U - 128) - 208 (V - 128)) >> 8)
 *   B = clamp((c + 516 (U - 128)) >> 8)
 * computed in 16 bits, and exactly: the multiples of 256 are taken out of
 * the shift (298 = 256 + 42, 409 = 256 + 153, 208 = 256 - 48, 516 = 512 + 4),
 * which leaves sums between -20128 and 29597, and 32768 = 128 x 256 is added
 * before the shift and 128 taken off behind it, so that the shift is of an
 * unsigned value. All 2^24 (Y, U, V) give the bytes of the formula above.
 * Plain loops over small arrays, for the compiler to make vectors of. */
[[gnu::always_inline]] static inline void yuv_row(size_t         n,
                                                  size_t         luma_step,
                                                  size_t         chroma_step,
                                                  const uint8_t *luma,
                                                  const uint8_t *cb,
                                                  const uint8_t *cr,
                                                  uint8_t       *rgb) {
    int16_t y[CONVERT_CHUNK], u[CONVERT_CHUNK], v[CONVERT_CHUNK];
    for (size_t i = 0; i < n; i++) {
        y[i] = (int16_t) (luma[i * luma_step] - 16);
    }
    for (size_t j = 0; j < n / 2; j++) {
        u[2 * j] = u[2 * j + 1] = (int16_t) (cb[j * chroma_step] - 128);
        v[2 * j] = v[2 * j + 1] = (int16_t) (cr[j * chroma_step] - 128);
    }
    for (size_t i = 0; i < n; i++) {
        const uint16_t r = (uint16_t) (42 * y[i] + 153 * v[i] + 128 + 32768);
        const uint16_t g = (uint16_t) (42 * y[i] - 100 * u[i] + 48 * v[i] + 128 + 32768);
        const uint16_t b = (uint16_t) (42 * y[i] + 4 * u[i] + 128 + 32768);
        rgb[3 * i]       = clamp8((int16_t) (y[i] + v[i] + (r >> 8U) - 128));
        rgb[3 * i + 1]   = clamp8((int16_t) (y[i] - v[i] + (g >> 8U) - 128));
        rgb[3 * i + 2]   = clamp8((int16_t) (y[i] + 2 * u[i] + (b >> 8U) - 128));
    }
}

/* ---- tiles ------------------------------------------------------------ */

/* The pixels [x0, x1) x [y0, y1) of a frame frame_ok accepted, as RGB8 into
 * `rgb`, an image of the frame's size without padding. x0 and x1 are even
 * for NV12 and YUYV: tiles start at multiples of DECKNET_TILE and those
 * frames have an even width. */
static void convert(const struct geist_decknet_frame *f,
                    size_t                            x0,
                    size_t                            x1,
                    size_t                            y0,
                    size_t                            y1,
                    uint8_t                          *rgb) {
    for (size_t y = y0; y < y1; y++) {
        const uint8_t *row = f->data + y * f->stride;
        uint8_t       *out = rgb + (y * f->width + x0) * 3;
        switch (f->format) {
        case GEIST_DECKNET_FORMAT_RGB8:
            memcpy(out, row + x0 * 3, (x1 - x0) * 3);
            break;
        case GEIST_DECKNET_FORMAT_NV12: {
            const uint8_t *uv = f->data + (f->height + y / 2) * f->stride;
            for (size_t x = x0; x < x1; x += CONVERT_CHUNK, out += CONVERT_CHUNK * 3) {
                const size_t n = x1 - x < CONVERT_CHUNK ? x1 - x : CONVERT_CHUNK;
                yuv_row(n, 1, 2, row + x, uv + x, uv + x + 1, out);
            }
            break;
        }
        case GEIST_DECKNET_FORMAT_YUYV:
            for (size_t x = x0; x < x1; x += CONVERT_CHUNK, out += CONVERT_CHUNK * 3) {
                const size_t   n = x1 - x < CONVERT_CHUNK ? x1 - x : CONVERT_CHUNK;
                const uint8_t *p = row + x * 2; /* Y0 U Y1 V */
                yuv_row(n, 2, 4, p, p + 1, p + 3, out);
            }
            break;
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

/* Marks the tiles the 2 x 2 camera pixels of each sample lie in. */
static void
mark_tiles(struct geist_decknet_stream *s, size_t n, const struct decknet_sample table[static n]) {
    for (size_t i = 0; i < n; i++) {
        const size_t x = table[i].x, y = table[i].y;
        for (size_t k = 0; k < 4; k++) {
            s->used[(y + k / 2) / DECKNET_TILE * s->tiles_w + (x + k % 2) / DECKNET_TILE] = 1;
        }
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

struct convert_job {
    const struct geist_decknet_stream *s;
    const struct geist_decknet_frame  *f;
};

/* Without gating, an NV12 or YUYV frame: of the tile rows [begin, end) the
 * tiles the network reads, as RGB8 into s->rgb. Neighbouring tiles of a row
 * go in one piece. */
static void tiles_convert(void *ctx, size_t begin, size_t end) {
    const struct convert_job          *job = ctx;
    const struct geist_decknet_stream *s   = job->s;
    const struct geist_decknet_frame  *f   = job->f;
    for (size_t ty = begin; ty < end; ty++) {
        const uint8_t *used = s->used + ty * s->tiles_w;
        const size_t   y0   = ty * DECKNET_TILE;
        const size_t   y1   = y0 + DECKNET_TILE < f->height ? y0 + DECKNET_TILE : f->height;
        for (size_t tx = 0; tx < s->tiles_w; tx++) {
            if (used[tx] == 0) {
                continue;
            }
            const size_t x0 = tx * DECKNET_TILE;
            while (tx + 1 < s->tiles_w && used[tx + 1] != 0) {
                tx++;
            }
            const size_t x1 = (tx + 1) * DECKNET_TILE;
            convert(f, x0, x1 < f->width ? x1 : f->width, y0, y1, s->rgb);
        }
    }
}

/* With gating: each tile the network reads is compared with the kept one,
 * and where they differ by more than the threshold the frame's replaces it,
 * in the kept bytes and in s->rgb. s->changed then says, per tile, whether
 * a sample whose first pixel lies in it reads a replaced tile: the tile
 * itself or the one right of it, below it or below right. */
static void tiles_gate(struct geist_decknet_stream *s, const struct geist_decknet_frame *f) {
    const bool   yuv      = f->format != GEIST_DECKNET_FORMAT_RGB8;
    uint8_t     *kept     = yuv ? s->kept : s->rgb;
    struct block b[2]     = {};
    const size_t n_blocks = blocks(f, 0, f->width, 0, 1, b);
    const size_t row      = b[0].col1; /* the bytes of a kept row: no padding */
    const size_t n        = s->tiles_w * s->tiles_h;
    const bool   all      = !s->have_kept || s->kept_format != f->format;
    s->n_tiles_changed    = 0;
    for (size_t t = 0; t < n; t++) {
        s->changed[t] = 0;
        if (s->used[t] == 0) {
            continue;
        }
        const size_t x0 = t % s->tiles_w * DECKNET_TILE, y0 = t / s->tiles_w * DECKNET_TILE;
        const size_t x1 = x0 + DECKNET_TILE < f->width ? x0 + DECKNET_TILE : f->width;
        const size_t y1 = y0 + DECKNET_TILE < f->height ? y0 + DECKNET_TILE : f->height;
        (void) blocks(f, x0, x1, y0, y1, b);
        if (!all && !tile_differs(f, n_blocks, b, row, s->threshold, kept)) {
            continue;
        }
        for (size_t i = 0; i < n_blocks; i++) {
            for (size_t r = b[i].row0; r < b[i].row1; r++) {
                memcpy(kept + r * row + b[i].col0,
                       f->data + r * f->stride + b[i].col0,
                       b[i].col1 - b[i].col0);
            }
        }
        if (yuv) {
            convert(f, x0, x1, y0, y1, s->rgb);
        }
        s->changed[t] = 1;
        s->n_tiles_changed++;
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

/* An RGB8 image to sample: rows `stride` bytes apart, and `wide`, the first
 * byte offset from which DECKNET_SAMPLE_BYTES bytes of the row below are not
 * all there to read (0: nowhere are they). */
struct pixels {
    const uint8_t *rgb;
    size_t         stride, wide;
};

/* A frame frame_ok accepted, RGB8, as it lies in the caller's memory. */
static struct pixels pixels_of(const struct geist_decknet_frame *f) {
    const size_t need = f->stride + DECKNET_SAMPLE_BYTES;
    return (struct pixels) {.rgb    = f->data,
                            .stride = f->stride,
                            .wide   = f->n_bytes < need ? 0 : f->n_bytes - need + 1};
}

/* s->rgb, which has DECKNET_SAMPLE_BYTES behind its last pixel. */
static struct pixels pixels_own(const struct geist_decknet_stream *s) {
    return (struct pixels) {.rgb = s->rgb, .stride = s->d->img_w * 3, .wide = SIZE_MAX};
}

/* Sixteen bytes, and four 32-bit lanes: a crop pixel (run.c's q4). The
 * compilers' vector types, without a platform's intrinsics: a shuffle with
 * constant indices is one table lookup on NEON and one PSHUFB on x86-64. */
typedef uint8_t  b16 [[gnu::vector_size(16)]];
typedef uint32_t q4 [[gnu::vector_size(16)]];
static_assert(sizeof(q4) == DECKNET_CROP_CH * sizeof(uint32_t));
static_assert(sizeof(b16) == DECKNET_SAMPLE_BYTES);
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "rgb_lanes reads a byte and three zeros as a 32-bit lane");

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

/* One sample of an image into the crop pixel o: run.c's resample() with the
 * coordinates and the weights from the table, and decknet_blend for the
 * three colours at once. With w00 = 2^24 - w0 - w1 - w2 its sum is
 *   2^24 b00 + w0 (b01 - b00) + w1 (b10 - b00) + w2 (b11 - b00)
 * which in 32 bits, where the differences wrap, is the same number: the sum
 * itself is below 2^32. */
[[gnu::always_inline]] static inline void
sample_one(const struct decknet_sample *t, const struct pixels *img, uint32_t *o) {
    const size_t   off = t->y * img->stride + (size_t) t->x * 3;
    const uint8_t *r0  = img->rgb + off;
    const uint8_t *r1  = r0 + img->stride;
    if (off >= img->wide) {
        /* At the end of a frame that has nothing behind its last pixels. */
        for (size_t c = 0; c < 3; c++) {
            o[c] = decknet_blend(t->w, r0[c], r0[3 + c], r1[c], r1[3 + c]);
        }
        o[3] = 0;
        return;
    }
    b16 row0, row1;
    q4  w;
    memcpy(&row0, r0, sizeof row0);
    memcpy(&row1, r1, sizeof row1);
    memcpy(&w, t, sizeof w); /* the fourth lane is x and y, and not used */
    const q4 b00 = rgb_lanes(row0, false);
    q4       sum = (b00 << 24) + 128;
    sum += (rgb_lanes(row0, true) - b00) * __builtin_shufflevector(w, w, 0, 0, 0, 0);
    sum += (rgb_lanes(row1, false) - b00) * __builtin_shufflevector(w, w, 1, 1, 1, 1);
    sum += (rgb_lanes(row1, true) - b00) * __builtin_shufflevector(w, w, 2, 2, 2, 2);
    sum >>= 8;
    memcpy(o, &sum, sizeof sum);
}

/* out[px] for every sample of the table, whose crop pixels are the runs. */
static void sample(size_t                       n_runs,
                   const struct decknet_run     run[static n_runs],
                   const struct decknet_sample *table,
                   const struct pixels         *img,
                   uint32_t                    *out) {
    for (size_t r = 0; r < n_runs; r++) {
        uint32_t *o = out + (size_t) run[r].px * DECKNET_CROP_CH;
        for (size_t i = 0; i < run[r].n; i++) {
            sample_one(table++, img, o + i * DECKNET_CROP_CH);
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
                sample_one(t, &img, out + px * DECKNET_CROP_CH);
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

/* ---- entry points ----------------------------------------------------- */

void decknet_stream_destroy(struct geist_decknet_stream *s) {
    if (s == nullptr) {
        return;
    }
    for (size_t p = 0; p < DECKNET_PLANES; p++) {
        heap_freep(&s->run[p]);
    }
    heap_freep(&s->bg);
    heap_freep(&s->tab);
    heap_freep(&s->bg_ref);
    heap_freep(&s->floor_ref);
    heap_freep(&s->used);
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
    s->d                                    = d;
    s->n_threads                            = 1;
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
        s->floor_ref = heap_calloc_array_aligned(float, n_floor);
        s->used      = heap_calloc_array_aligned(uint8_t, s->tiles_w * s->tiles_h);
        s->changed   = heap_calloc_array_aligned(uint8_t, s->tiles_w * s->tiles_h);
        /* Zeroed: its first row and column stay 0. */
        s->changed_sum = heap_calloc_array_aligned(uint32_t, (s->tiles_w + 1) * (s->tiles_h + 1));
        s->box         = heap_alloc_array_aligned(struct decknet_box, n_tab / DECKNET_CHUNK + 1);
        s->rgb         = heap_calloc_array_aligned(uint8_t, n_rgb);
        ok             = s->bg != nullptr && s->tab != nullptr && s->bg_ref != nullptr &&
                         s->floor_ref != nullptr && s->used != nullptr && s->changed != nullptr &&
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
         * NV12 or YUYV frame (two per pixel at most; RGB8 is kept in
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
        convert(ref, 0, ref->width, 0, ref->height, s->rgb);
    }
    float gb[3];
    /* The empty deck through its own homographies; the tables are the
     * current image's again below. */
    build_table(d, d->bg_w, 1, d->h_ref, &strip, s->bg);
    sample(1, &strip, s->bg, &img, d->bg_q);
    decknet_strip(d, d->bg_q, s->bg_ref);
    decknet_gain(d, s->bg_ref, gb);
    const double scale[3] = {decknet_scale(gb[0]), decknet_scale(gb[1]), decknet_scale(gb[2])};
    for (size_t slot = 0; slot < d->n_slots; slot++) {
        const float *h   = d->h_ref + (1 + DECKNET_PLANES * slot) * 9;
        float       *out = s->floor_ref + slot * d->crop_px * 3;
        build_table(d, d->crop_w, n_floor, h, floor, s->tab);
        sample(n_floor, floor, s->tab, &img, d->ws.crop);
        for (size_t r = 0; r < n_floor; r++) {
            for (size_t px = floor[r].px; px < (size_t) floor[r].px + floor[r].n; px++) {
                for (size_t c = 0; c < 3; c++) {
                    out[px * 3 + c] = decknet_unit(d->ws.crop[px * DECKNET_CROP_CH + c], scale[c]);
                }
            }
        }
    }
    memset(s->used, 0, s->tiles_w * s->tiles_h);
    build_table(d, d->bg_w, 1, d->h_cur, &strip, s->bg);
    mark_tiles(s, d->bg_px, s->bg);
    struct decknet_sample *t = s->tab;
    for (size_t slot = 0; slot < d->n_slots; slot++) {
        for (size_t p = 0; p < DECKNET_PLANES; p++) {
            const float *h = d->h_cur + (1 + DECKNET_PLANES * slot + p) * 9;
            build_table(d, d->crop_w, s->n_runs[p], h, s->run[p], t);
            mark_tiles(s, s->n_px[p], t);
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
    return GEIST_OK;
}

/* What the slots of one frame are computed from. */
struct slots_job {
    struct geist_decknet_stream *s;
    struct pixels                img; /* the frame, or s->rgb */
    const float                 *ga;
    /* The caches hold the crops of the frame before; and the 1x1 values of
     * the floor were computed with this frame's gains. */
    bool   cached, same_gain;
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
        uint8_t           *stale = s->gate ? s->stale + part * d->crop_px : nullptr;
        size_t             n     = 0;
        for (size_t slot = part * d->n_slots / s->n_threads;
             slot < (part + 1) * d->n_slots / s->n_threads;
             slot++) {
            const struct decknet_sample *t    = s->tab + slot * s->per_slot;
            struct decknet_work          work = {.crop     = ws->crop,
                                                 .crop_ref = s->floor_ref + slot * d->crop_px * 3,
                                                 .mix      = ws->mix,
                                                 .sums     = ws->sums,
                                                 .tube     = ws->tube_sums};
            if (s->gate) {
                work.mix  = s->mix + slot * d->n_mix;
                work.tube = s->tube + slot * d->n_tube_px * 3;
            }
            for (size_t p = 0; p < DECKNET_PLANES; p++) {
                const enum decknet_plane plane = (enum decknet_plane) p;
                size_t                   taken = s->n_px[p];
                if (!s->gate) {
                    sample(s->n_runs[p], s->run[p], t, &job->img, ws->crop);
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
                        sample(s->n_runs[p], s->run[p], t, &job->img, crop);
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
    const struct decknet_run strip = {.px = 0, .n = (uint32_t) d->bg_px};
    struct pixels            img   = pixels_own(s);
    s->n_tiles_changed             = s->n_tiles;
    if (s->gate) {
        tiles_gate(s, frame);
    } else if (frame->format == GEIST_DECKNET_FORMAT_RGB8) {
        img = pixels_of(frame);
    } else {
        struct convert_job job = {.s = s, .f = frame};
        on_threads(s, s->tiles_h, tiles_convert, &job);
    }
    float ga[3], gb[3];
    sample(1, &strip, s->bg, &img, d->bg_q);
    decknet_strip(d, d->bg_q, d->bg_a);
    decknet_gains(d, d->bg_a, s->bg_ref, ga, gb);
    s->n_samples = d->bg_px;
    if (d->result.checkable) {
        const bool       cached = s->gate && s->cached;
        struct slots_job job    = {.s         = s,
                                   .img       = img,
                                   .ga        = ga,
                                   .cached    = cached,
                                   .same_gain = cached && memcmp(ga, s->mix_ga, sizeof ga) == 0};
        on_threads(s, s->n_threads, slots, &job);
        for (size_t part = 0; part < s->n_threads; part++) {
            s->n_samples += job.n_samples[part];
        }
    }
    /* A pair that is not checkable had no slot sampled: the caches are of
     * an earlier frame, but the kept tiles moved on. */
    s->cached = s->gate && d->result.checkable;
    memcpy(s->mix_ga, ga, sizeof ga);
    s->timestamp   = frame->timestamp;
    s->have_result = true;
    s->n_frames++;
    return GEIST_OK;
}
