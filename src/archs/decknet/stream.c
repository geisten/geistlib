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
 *                   and with which fractions (decknet_sample_at, once);
 *   region          only for the crop pixels the pass reads
 *                   (decknet_plane_pixels), so nothing else of a frame is
 *                   touched;
 *   in place        the 2 x 2 camera pixels of a sample are read in the
 *                   frame's own format (RGB8, NV12, YUYV);
 *   the empty deck  sampled once, when it is set.
 *
 * None of it changes a number: every float the pass reads is computed by the
 * expression run.c uses, so a stream gives what decknet_run gives for the
 * same two images, bit for bit (tests/test_decknet_stream_unit.c).
 *
 * Everything is allocated in decknet_stream_create; the reference and the
 * frames allocate nothing.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#include "decknet.h"

#include "checked.h"
#include "cleanup.h"
#include "heap.h"

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
 * samplers below index it without a further check. */
static bool frame_ok(const struct geist_decknet *d, const struct geist_decknet_frame *f) {
    size_t row = 0, rows = 0, last = 0, need = 0;
    if (f->data == nullptr || f->width != d->img_w || f->height != d->img_h ||
        !frame_rows(f, &row, &rows) || f->stride < row) {
        return false;
    }
    /* The last row starts at (rows - 1) * stride and is `row` bytes long. */
    return !ckd_mul(&last, rows - 1, f->stride) && !ckd_add(&need, last, row) && need <= f->n_bytes;
}

/* v / 256 clamped to a byte, v being a colour in 1/256. */
static inline uint8_t clip8(int32_t v) {
    return v < 0 ? 0 : (v > 255 * 256 ? 255 : (uint8_t) (v / 256));
}

/* BT.601 limited range to RGB8 (docs/DECKNET.md, "Stream"). */
static inline void yuv_rgb(uint8_t luma, uint8_t cb, uint8_t cr, uint8_t rgb[static 3]) {
    const int32_t c = 298 * ((int32_t) luma - 16) + 128;
    const int32_t u = (int32_t) cb - 128;
    const int32_t v = (int32_t) cr - 128;
    rgb[0]          = clip8(c + 409 * v);
    rgb[1]          = clip8(c - 100 * u - 208 * v);
    rgb[2]          = clip8(c + 516 * u);
}

/* The RGB8 value of pixel (x, y) of a frame frame_ok accepted. `format` is
 * a constant at each call, so the switch folds away. */
[[gnu::always_inline]] static inline void pixel(enum geist_decknet_format         format,
                                                const struct geist_decknet_frame *f,
                                                size_t                            x,
                                                size_t                            y,
                                                uint8_t                           rgb[static 3]) {
    const uint8_t *row = f->data + y * f->stride;
    switch (format) {
    case GEIST_DECKNET_FORMAT_RGB8:
        rgb[0] = row[x * 3];
        rgb[1] = row[x * 3 + 1];
        rgb[2] = row[x * 3 + 2];
        return;
    case GEIST_DECKNET_FORMAT_NV12: {
        const uint8_t *uv = f->data + (f->height + y / 2) * f->stride + (x & ~(size_t) 1);
        yuv_rgb(row[x], uv[0], uv[1], rgb);
        return;
    }
    case GEIST_DECKNET_FORMAT_YUYV: {
        const uint8_t *pair = row + (x & ~(size_t) 1) * 2; /* Y0 U Y1 V */
        yuv_rgb(row[x * 2], pair[1], pair[3], rgb);
        return;
    }
    }
}

/* ---- sampling --------------------------------------------------------- */

/* table[i] for crop pixel px[i] (or i, when px is nullptr) of a crop that is
 * out_w wide, through homography h. */
static void build_table(const struct geist_decknet *d,
                        size_t                      out_w,
                        size_t                      n,
                        const float                 h[static 9],
                        const uint32_t             *px,
                        struct decknet_sample       table[static n]) {
    for (size_t i = 0; i < n; i++) {
        const size_t p = px != nullptr ? px[i] : i;
        double       u, v;
        decknet_sample_at(d->img_w, d->img_h, p % out_w, p / out_w, h, &u, &v);
        const size_t u0 = (size_t) u;
        const size_t v0 = (size_t) v;
        table[i]        = (struct decknet_sample) {
                .fu = u - (double) u0,
                .fv = v - (double) v0,
                .px = (uint32_t) p,
                .x  = (uint16_t) u0,
                .y  = (uint16_t) v0,
        };
    }
}

/* out[px][c] for each sample: run.c's resample() with the coordinates from
 * the table. lut[c][b] is what resample() computes per read, the byte b
 * scaled to 0..1 and by the gain of colour c, in F32, then widened. */
[[gnu::always_inline]] static inline void sample_as(enum geist_decknet_format   format,
                                                    size_t                      n,
                                                    const struct decknet_sample table[static n],
                                                    const struct geist_decknet_frame *f,
                                                    const double lut[static 3][256],
                                                    float       *out) {
    for (size_t i = 0; i < n; i++) {
        const struct decknet_sample *t = &table[i];
        uint8_t                      p[4][3];
        pixel(format, f, t->x, t->y, p[0]);
        pixel(format, f, (size_t) t->x + 1, t->y, p[1]);
        pixel(format, f, t->x, (size_t) t->y + 1, p[2]);
        pixel(format, f, (size_t) t->x + 1, (size_t) t->y + 1, p[3]);
        const double fu = t->fu;
        const double fv = t->fv;
        float       *o  = out + (size_t) t->px * 3;
        for (size_t c = 0; c < 3; c++) {
            const double a = lut[c][p[0][c]];
            const double b = lut[c][p[1][c]];
            const double e = lut[c][p[2][c]];
            const double g = lut[c][p[3][c]];
            o[c]           = (float) ((a * (1.0 - fu) + b * fu) * (1.0 - fv) +
                                      (e * (1.0 - fu) + g * fu) * fv);
        }
    }
}

static void sample(const struct geist_decknet_stream *s,
                   size_t                             n,
                   const float                        gain[static 3],
                   const struct decknet_sample        table[static n],
                   const struct geist_decknet_frame  *f,
                   float                             *out) {
    double lut[3][256];
    for (size_t c = 0; c < 3; c++) {
        for (size_t b = 0; b < 256; b++) {
            lut[c][b] = (double) (s->scale[b] * gain[c]);
        }
    }
    switch (f->format) {
    case GEIST_DECKNET_FORMAT_RGB8:
        sample_as(GEIST_DECKNET_FORMAT_RGB8, n, table, f, lut, out);
        return;
    case GEIST_DECKNET_FORMAT_NV12:
        sample_as(GEIST_DECKNET_FORMAT_NV12, n, table, f, lut, out);
        return;
    case GEIST_DECKNET_FORMAT_YUYV:
        sample_as(GEIST_DECKNET_FORMAT_YUYV, n, table, f, lut, out);
        return;
    }
}

/* ---- entry points ----------------------------------------------------- */

void decknet_stream_destroy(struct geist_decknet_stream *s) {
    if (s == nullptr) {
        return;
    }
    for (size_t p = 0; p < DECKNET_PLANES; p++) {
        heap_freep(&s->px[p]);
    }
    heap_freep(&s->bg);
    heap_freep(&s->tab);
    heap_freep(&s->bg_ref);
    heap_freep(&s->floor_ref);
    heap_freep(&s);
}

/* s->px[plane]: the crop pixels that plane reads. */
static bool plane_pixels(struct geist_decknet_stream *s, size_t plane, uint8_t *mark) {
    const struct geist_decknet *d = s->d;
    decknet_plane_pixels(d, (enum decknet_plane) plane, mark);
    size_t n = 0;
    for (size_t i = 0; i < d->crop_px; i++) {
        n += mark[i];
    }
    s->px[plane] = heap_alloc_array_aligned(uint32_t, n); /* n >= 1: a pool has a row */
    if (s->px[plane] == nullptr) {
        return false;
    }
    size_t k = 0;
    for (size_t i = 0; i < d->crop_px; i++) {
        if (mark[i] != 0) {
            s->px[plane][k++] = (uint32_t) i;
        }
    }
    s->n_px[plane] = n;
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
    s->d = d;
    for (size_t b = 0; b < 256; b++) {
        s->scale[b] = (float) b / 255.0f;
    }
    GEIST_CLEANUP(heap_freep) uint8_t *mark = heap_alloc_array_aligned(uint8_t, d->crop_px);
    bool                               ok   = mark != nullptr;
    for (size_t p = 0; ok && p < DECKNET_PLANES; p++) {
        ok = plane_pixels(s, p, mark);
    }
    size_t n_tab = 0, n_floor = 0, n_bg = 0;
    ok = ok && !ckd_mul(&n_tab, d->n_slots, s->per_slot) && !ckd_mul(&n_floor, d->crop_px, 3) &&
         !ckd_mul(&n_floor, n_floor, d->n_slots) && !ckd_mul(&n_bg, d->bg_px, 3);
    if (ok) {
        s->bg     = heap_alloc_array_aligned(struct decknet_sample, d->bg_px);
        s->tab    = heap_alloc_array_aligned(struct decknet_sample, n_tab);
        s->bg_ref = heap_alloc_array_aligned(float, n_bg);
        /* Zeroed: decknet_plane reads it at the floor pixels only, which
         * the reference writes, but nothing here is ever uninitialised. */
        s->floor_ref = heap_calloc_array_aligned(float, n_floor);
        ok           = s->bg != nullptr && s->tab != nullptr && s->bg_ref != nullptr &&
                       s->floor_ref != nullptr;
    }
    if (!ok) {
        decknet_stream_destroy(s);
        return GEIST_E_OOM;
    }
    *out = s;
    return GEIST_OK;
}

enum geist_status decknet_stream_set_reference(struct geist_decknet_stream      *s,
                                               const struct geist_decknet_frame *ref) {
    static const float          one[3] = {1.0f, 1.0f, 1.0f};
    const struct geist_decknet *d      = s->d;
    if (!frame_ok(d, ref)) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t          n_floor = s->n_px[DECKNET_PLANE_FLOOR];
    const uint32_t *const floor   = s->px[DECKNET_PLANE_FLOOR];
    float                 gb[3];
    /* The empty deck through its own homographies; the tables are the
     * current image's again below. */
    build_table(d, d->bg_w, d->bg_px, d->h_ref, nullptr, s->bg);
    sample(s, d->bg_px, one, s->bg, ref, s->bg_ref);
    decknet_gain(d, s->bg_ref, gb);
    for (size_t slot = 0; slot < d->n_slots; slot++) {
        const float *h = d->h_ref + (1 + DECKNET_PLANES * slot) * 9;
        build_table(d, d->crop_w, n_floor, h, floor, s->tab);
        sample(s, n_floor, gb, s->tab, ref, s->floor_ref + slot * d->crop_px * 3);
    }
    build_table(d, d->bg_w, d->bg_px, d->h_cur, nullptr, s->bg);
    struct decknet_sample *t = s->tab;
    for (size_t slot = 0; slot < d->n_slots; slot++) {
        for (size_t p = 0; p < DECKNET_PLANES; p++) {
            const float *h = d->h_cur + (1 + DECKNET_PLANES * slot + p) * 9;
            build_table(d, d->crop_w, s->n_px[p], h, s->px[p], t);
            t += s->n_px[p];
        }
    }
    s->h_gen    = d->h_gen;
    s->have_ref = true;
    return GEIST_OK;
}

enum geist_status decknet_stream_push_frame(struct geist_decknet_stream      *s,
                                            const struct geist_decknet_frame *frame) {
    static const float    one[3] = {1.0f, 1.0f, 1.0f};
    struct geist_decknet *d      = s->d;
    if (!frame_ok(d, frame)) {
        return GEIST_E_INVALID_ARG;
    }
    if (!s->have_ref || s->h_gen != d->h_gen) {
        return GEIST_E_INVALID_STATE;
    }
    float ga[3], gb[3];
    sample(s, d->bg_px, one, s->bg, frame, d->bg_a);
    decknet_gains(d, d->bg_a, s->bg_ref, ga, gb);
    const struct decknet_sample *t = s->tab;
    for (size_t slot = 0; d->result.checkable && slot < d->n_slots; slot++) {
        for (size_t p = 0; p < DECKNET_PLANES; p++) {
            sample(s, s->n_px[p], ga, t, frame, d->crop_a);
            decknet_plane(d,
                          slot,
                          (enum decknet_plane) p,
                          d->crop_a,
                          s->floor_ref + slot * d->crop_px * 3);
            t += s->n_px[p];
        }
    }
    s->timestamp   = frame->timestamp;
    s->have_result = true;
    s->n_frames++;
    return GEIST_OK;
}
