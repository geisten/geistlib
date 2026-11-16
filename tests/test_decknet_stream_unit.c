#define _POSIX_C_SOURCE 200809L /* sched_yield */
/* tests/test_decknet_stream_unit.c — a decknet stream against the still-image
 * path (#533).
 *
 * The still path (geist_decknet_run, checked against the numpy reference by
 * test_decknet_unit) is the oracle. A stream is the same pass fed from
 * sample tables, a region of interest, a cached empty deck and the frame's
 * own format; none of that may change a bit. Everything runs on the public
 * invented network of tests/data/decknet_synth/ and on decknet_tiny.h.
 *
 *   parity, RGB8    every output of every slot, bit for bit, for a sequence
 *                   of frames, with padded rows, after replaced homographies
 *                   and across a frame that is not checkable;
 *   parity, NV12    bit for bit against the still path on the frame
 *   and YUYV        converted to RGB8 by the formula of docs/DECKNET.md,
 *                   which this file implements a second time (and checks
 *                   against the colours BT.601 names, within one code);
 *   gating          a gated stream against the still path on the image a
 *                   threshold defines: this file keeps its own copy of each
 *                   tile, replaces it when the frame's differs by more than
 *                   the threshold (its own sum of absolute differences) and
 *                   runs the still path on that image; bit for bit, for
 *                   threshold 0 (which is the frame itself) and above, in
 *                   each format, across a frame that is not checkable and a
 *                   new reference; and where a kept value could be stale: a
 *                   kernel window across a cell of the crop, a tile exactly
 *                   at the threshold, each tile changed on its own, a
 *                   sample at the seam of two stretches of the tables;
 *   conversion      NV12 and YUYV frames of arbitrary bytes, the whole range;
 *   no allocation   after geist_decknet_stream_create and the first
 *                   geist_decknet_stream_set_gate: not for the reference,
 *                   not for the first frame, not for any later one, in any
 *                   format, gated or not (heap.h counts). The count is
 *                   compared with zero, so one allocation fails the test;
 *   geometry        each way a frame can lie about itself is refused with
 *                   GEIST_E_INVALID_ARG and leaves the last result; frames
 *                   that fit their buffer to the byte are accepted (the
 *                   buffers are exact-size, so ASan sees a read behind one);
 *   the sample      the fixed-point bilinear sample a crop pixel is: its
 *                   weights never exceed the whole (at fractions where three
 *                   roundings up would), a flat image samples to itself, and
 *                   any four bytes to within two units of 1/65536 of the
 *                   exact value;
 *   two threads     the same bits, with and without gating;
 *   the frame's end a sample at the last pixels of a frame is computed from
 *                   those pixels alone when the buffer ends with them, and
 *                   from 16 bytes of each row when it does not: the same
 *                   bits, for every length of what is behind the frame;
 *   a queue of one  geist_decknet_stream_offer and _push_offered: step by
 *                   step on one thread (what is handed back, when, and
 *                   counted), with a consumer that takes one frame for
 *                   every three a second thread offers (it computes the
 *                   third, two are dropped, in every round), and with the
 *                   second thread offering as fast as it can: every frame
 *                   comes back exactly once, dropped or computed, the
 *                   computed ones in order and with the bits of a
 *                   push_frame on the same frame;
 *   state           no frame before a reference or after new homographies;
 *                   no result before a frame; each allocation of a create
 *                   failing once.
 */
#include "decknet_tiny.h"
#include "heap.h"
#include "test_helpers.h"

/* The fixed-point sample (decknet_weights, decknet_blend), tested as it is. */
#define GEIST_INTERNAL_ARCH_LAYER
#include "../src/archs/decknet/decknet.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <geist.h>
#include <geist_decknet.h>

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <time.h>

static int g_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail = 1;                                                     \
        }                                                                   \
    } while (0)

[[noreturn]] static void broken(const char *what, const char *path) {
    fprintf(stderr, "ERROR: %s: %s\n", what, path);
    exit(GEIST_TEST_ERROR);
}

static const char GGUF[] = "tests/data/decknet_synth/synth.gguf";

/* Binary PPM (P6, maxval 255) of exactly w x h. */
static uint8_t *read_ppm(const char *path, size_t w, size_t h) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
        broken("cannot open", path);
    }
    unsigned pw = 0, ph = 0, max = 0;
    if (fscanf(f, "P6 %u %u %u", &pw, &ph, &max) != 3 || fgetc(f) != '\n' || pw != w || ph != h ||
        max != 255) {
        broken("not a P6 image of the network's size", path);
    }
    uint8_t *img = xmalloc(w * h * 3);
    xfread(img, 1, w * h * 3, f);
    fclose(f);
    return img;
}

static struct geist_decknet *load_synth(void) {
    struct geist_decknet *d = nullptr;
    if (geist_decknet_load(GGUF, &d) != GEIST_OK) {
        broken(geist_last_create_error(), GGUF);
    }
    return d;
}

/* ---- frames ----------------------------------------------------------- */

/* The conversion of docs/DECKNET.md, written out a second time. */
static uint8_t clip(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t) v);
}
static void yuv_to_rgb(int y, int u, int v, uint8_t rgb[static 3]) {
    /* floor((x + 128) / 256) for a negative x as well */
    const int c = 298 * (y - 16) + 128 + (1 << 20);
    rgb[0]      = clip((c + 409 * (v - 128)) / 256 - (1 << 12));
    rgb[1]      = clip((c - 100 * (u - 128) - 208 * (v - 128)) / 256 - (1 << 12));
    rgb[2]      = clip((c + 516 * (u - 128)) / 256 - (1 << 12));
}

static void rgb_to_yuv(const uint8_t *p, uint8_t yuv[static 3]) {
    const int r = p[0], g = p[1], b = p[2];
    yuv[0] = clip(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
    yuv[1] = clip(((-38 * r - 74 * g + 112 * b + 128 + (128 << 8)) >> 8));
    yuv[2] = clip(((112 * r - 94 * g - 18 * b + 128 + (128 << 8)) >> 8));
}

static size_t row_bytes(enum geist_decknet_format format, size_t w) {
    return format == GEIST_DECKNET_FORMAT_RGB8   ? w * 3
           : format == GEIST_DECKNET_FORMAT_NV12 ? w
                                                 : w * 2;
}

/* The smallest buffer that holds a w x h frame with rows `stride` apart:
 * the last row is not padded. */
static size_t exact_bytes(enum geist_decknet_format format, size_t w, size_t h, size_t stride) {
    const size_t rows = format == GEIST_DECKNET_FORMAT_NV12 ? h + h / 2 : h;
    return (rows - 1) * stride + row_bytes(format, w);
}

/* A packed RGB8 image as a frame of `format` in an exact-size buffer, the
 * padding filled with 0xA5. The caller frees frame.data. */
static struct geist_decknet_frame
make_frame(enum geist_decknet_format format, size_t w, size_t h, size_t pad, const uint8_t *rgb) {
    const size_t stride = row_bytes(format, w) + pad;
    const size_t n      = exact_bytes(format, w, h, stride);
    uint8_t     *out    = xmalloc(n);
    memset(out, 0xA5, n);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            const uint8_t *p = rgb + (y * w + x) * 3;
            uint8_t        yuv[3];
            rgb_to_yuv(p, yuv);
            if (format == GEIST_DECKNET_FORMAT_RGB8) {
                memcpy(out + y * stride + x * 3, p, 3);
            } else if (format == GEIST_DECKNET_FORMAT_NV12) {
                out[y * stride + x] = yuv[0];
                if (x % 2 == 0 && y % 2 == 0) {
                    out[(h + y / 2) * stride + x]     = yuv[1];
                    out[(h + y / 2) * stride + x + 1] = yuv[2];
                }
            } else {
                out[y * stride + x * 2] = yuv[0];
                if (x % 2 == 0) {
                    out[y * stride + x * 2 + 1] = yuv[1];
                    out[y * stride + x * 2 + 3] = yuv[2];
                }
            }
        }
    }
    return (struct geist_decknet_frame) {
            .format = format, .width = w, .height = h, .stride = stride, .n_bytes = n, .data = out};
}

/* What the still path is given for a frame: packed RGB8. */
static uint8_t *frame_to_rgb8(const struct geist_decknet_frame *f) {
    const size_t w = f->width, h = f->height;
    uint8_t     *rgb = xmalloc(w * h * 3);
    for (size_t y = 0; y < h; y++) {
        const uint8_t *row = f->data + y * f->stride;
        for (size_t x = 0; x < w; x++) {
            uint8_t     *o    = rgb + (y * w + x) * 3;
            const size_t even = x - x % 2;
            if (f->format == GEIST_DECKNET_FORMAT_RGB8) {
                memcpy(o, row + x * 3, 3);
            } else if (f->format == GEIST_DECKNET_FORMAT_NV12) {
                const uint8_t *uv = f->data + (h + y / 2) * f->stride + even;
                yuv_to_rgb(row[x], uv[0], uv[1], o);
            } else {
                yuv_to_rgb(row[x * 2], row[even * 2 + 1], row[even * 2 + 3], o);
            }
        }
    }
    return rgb;
}

/* Frame k of a sequence: the image with a block somewhere else each time. */
static uint8_t *moved(size_t w, size_t h, size_t k, const uint8_t *img) {
    uint8_t *out = xmalloc(w * h * 3);
    memcpy(out, img, w * h * 3);
    if (k == 0) {
        return out;
    }
    const size_t x0 = 30 + (k * 23) % (w - 60), y0 = 10 + (k * 17) % (h - 40);
    for (size_t y = y0; y < y0 + 24; y++) {
        for (size_t x = x0; x < x0 + 24; x++) {
            out[(y * w + x) * 3 + k % 3] = (uint8_t) (((x + y) % 2) * 255);
        }
    }
    return out;
}

/* ---- parity ----------------------------------------------------------- */

static bool same_bits(size_t n, const float *a, const float *b) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

/* Whether the last frame of the stream gave, bit for bit, what the last run
 * of `still` gave. Counts the compared outputs. */
static bool same_outputs(const struct geist_decknet        *still,
                         const struct geist_decknet_result *want,
                         const struct geist_decknet_stream *s,
                         const struct geist_decknet_info   *info,
                         size_t                            *n_outputs) {
    struct geist_decknet_stream_result got;
    if (geist_decknet_stream_result(s, &got) != GEIST_OK) {
        return false;
    }
    bool same = got.image.checkable == want->checkable &&
                same_bits(3, got.image.gain, want->gain) &&
                same_bits(1, &got.image.bg_changed, &want->bg_changed);
    *n_outputs += 4;
    for (size_t i = 0; i < info->n_slots; i++) {
        struct geist_decknet_slot a, b;
        const enum geist_status   sa = geist_decknet_slot(still, i, &a);
        const enum geist_status   sb = geist_decknet_stream_slot(s, i, &b);
        if (!want->checkable) {
            same = same && sa == GEIST_E_INVALID_STATE && sb == GEIST_E_INVALID_STATE;
            continue;
        }
        if (sa != GEIST_OK || sb != GEIST_OK) {
            return false;
        }
        const float sca[8] = {
                a.occ_front, a.occ_rear, a.wall_front, a.rack, a.plate, a.deep, a.tube_n, a.tube_r};
        const float scb[8] = {
                b.occ_front, b.occ_rear, b.wall_front, b.rack, b.plate, b.deep, b.tube_n, b.tube_r};
        same = same && same_bits(8, sca, scb) && same_bits(info->n_tips, a.tips, b.tips) &&
               same_bits(info->n_tubes, a.tubes, b.tubes) &&
               same_bits(info->n_tubes, a.tubes_rot, b.tubes_rot);
        *n_outputs += 8 + info->n_tips + 2 * info->n_tubes;
    }
    return same;
}

/* A sequence of frames in one format through a stream and, converted,
 * through the still path on a second handle. `moved_h` replaces the
 * homographies of both images on both handles first. */
static void test_parity(enum geist_decknet_format format,
                        size_t                    pad,
                        bool                      moved_h,
                        size_t                    threads,
                        const char               *what) {
    struct geist_decknet     *still = load_synth();
    struct geist_decknet     *d     = load_synth();
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    const size_t w = info.image_width, h = info.image_height, n_rgb = w * h * 3;
    uint8_t     *cur = read_ppm("tests/data/decknet_synth/cur.ppm", w, h);
    uint8_t     *ref = read_ppm("tests/data/decknet_synth/ref.ppm", w, h);

    struct geist_decknet_stream *s = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK && s != nullptr);
    CHECK(geist_decknet_stream_set_threads(s, threads) == GEIST_OK);
    struct geist_decknet_frame fref = make_frame(format, w, h, pad, ref);
    uint8_t                   *ref8 = frame_to_rgb8(&fref);
    CHECK(geist_decknet_stream_set_reference(s, &fref) == GEIST_OK);

    if (moved_h) {
        /* Other homographies for both images: a little off the file's. */
        float *hm = xmalloc(info.n_homographies * 9 * sizeof(float));
        for (int image = 0; image < 2; image++) {
            for (size_t i = 0; i < info.n_homographies; i++) {
                const float sx   = image == 0 ? 0.97f : 1.02f;
                const float m[9] = {sx,
                                    0.01f,
                                    3.0f + (float) i,
                                    -0.02f,
                                    1.0f,
                                    2.5f + 2.0f * (float) (i % 7),
                                    1e-4f,
                                    -1e-4f,
                                    1.0f};
                memcpy(hm + i * 9, m, sizeof m);
            }
            const enum geist_decknet_image which =
                    image == 0 ? GEIST_DECKNET_IMAGE_CURRENT : GEIST_DECKNET_IMAGE_EMPTY;
            CHECK(geist_decknet_set_homographies(d, info.n_homographies, which, hm) == GEIST_OK);
            CHECK(geist_decknet_set_homographies(still, info.n_homographies, which, hm) ==
                  GEIST_OK);
        }
        free(hm);
        /* The stream's tables are the old ones: it says so, and the
         * reference takes the new ones over. */
        CHECK(geist_decknet_stream_push_frame(s, &fref) == GEIST_E_INVALID_STATE);
        CHECK(geist_decknet_stream_set_reference(s, &fref) == GEIST_OK);
    }

    size_t           n_outputs = 0, n_checkable = 0;
    constexpr size_t n_frames = 7;
    for (size_t k = 0; k < n_frames; k++) {
        uint8_t *img = moved(w, h, k, cur);
        if (k == 4) { /* half brightness: not checkable, and checkable again after it */
            for (size_t i = 0; i < n_rgb; i++) {
                img[i] >>= 1;
            }
        }
        struct geist_decknet_frame f = make_frame(format, w, h, pad, img);
        f.timestamp                  = 1000 + k;
        uint8_t *img8                = frame_to_rgb8(&f);

        struct geist_decknet_result want;
        CHECK(geist_decknet_run(still, n_rgb, img8, ref8, &want) == GEIST_OK);
        CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
        struct geist_decknet_stream_result got;
        CHECK(geist_decknet_stream_result(s, &got) == GEIST_OK && got.timestamp == 1000 + k &&
              got.n_frames == k + 1);
        if (!same_outputs(still, &want, s, &info, &n_outputs)) {
            fprintf(stderr, "FAIL %s: frame %zu differs from the still path\n", what, k);
            g_fail = 1;
        }
        CHECK(want.checkable == (k != 4));
        n_checkable += want.checkable;
        free(img8);
        free((void *) (uintptr_t) f.data);
        free(img);
    }
    printf("%-34s %zu frames (%zu checkable), %zu outputs bit-identical to the still path\n",
           what,
           (size_t) n_frames,
           n_checkable,
           n_outputs);

    geist_decknet_stream_destroy(s);
    free(ref8);
    free((void *) (uintptr_t) fref.data);
    free(cur);
    free(ref);
    geist_decknet_destroy(d);
    geist_decknet_destroy(still);
}

/* ---- gating ----------------------------------------------------------- */

/* Where the bytes of the pixels [x0, x1) x [y0, y1) of a frame are, pixel by
 * pixel: the header's "bytes of a tile", read a second time. Returns how
 * many. */
static size_t tile_bytes(const struct geist_decknet_frame *f,
                         size_t                            x0,
                         size_t                            x1,
                         size_t                            y0,
                         size_t                            y1,
                         size_t out[static GEIST_DECKNET_TILE * GEIST_DECKNET_TILE * 3]) {
    size_t n = 0;
    for (size_t y = y0; y < y1; y++) {
        for (size_t x = x0; x < x1; x++) {
            if (f->format == GEIST_DECKNET_FORMAT_RGB8) {
                for (size_t c = 0; c < 3; c++) {
                    out[n++] = y * f->stride + x * 3 + c;
                }
            } else if (f->format == GEIST_DECKNET_FORMAT_YUYV) {
                out[n++] = y * f->stride + x * 2;     /* its Y */
                out[n++] = y * f->stride + x * 2 + 1; /* the U or the V of its pair */
            } else {
                out[n++] = y * f->stride + x;
                if (x % 2 == 0 && y % 2 == 0) { /* the U V pair of its 2 x 2 block */
                    out[n++] = (f->height + y / 2) * f->stride + x;
                    out[n++] = (f->height + y / 2) * f->stride + x + 1;
                }
            }
        }
    }
    return n;
}

/* Frames through a gated stream and, as the image the threshold defines,
 * through the still path. */
static void test_gate(enum geist_decknet_format format,
                      size_t                    pad,
                      uint32_t                  threshold,
                      size_t                    threads,
                      const char               *what) {
    struct geist_decknet     *still = load_synth();
    struct geist_decknet     *d     = load_synth();
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    const size_t w = info.image_width, h = info.image_height, n_rgb = w * h * 3;
    uint8_t     *cur = read_ppm("tests/data/decknet_synth/cur.ppm", w, h);
    uint8_t     *ref = read_ppm("tests/data/decknet_synth/ref.ppm", w, h);

    struct geist_decknet_stream *s = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK && s != nullptr);
    CHECK(geist_decknet_stream_set_gate(s, true, threshold) == GEIST_OK);
    CHECK(geist_decknet_stream_set_threads(s, threads) == GEIST_OK);
    struct geist_decknet_frame fref = make_frame(format, w, h, pad, ref);
    uint8_t                   *ref8 = frame_to_rgb8(&fref);
    CHECK(geist_decknet_stream_set_reference(s, &fref) == GEIST_OK);

    /* This file's kept tiles: a frame of the same layout. */
    uint8_t *kept      = xmalloc(fref.n_bytes);
    bool     have_kept = false;
    memset(kept, 0xA5, fref.n_bytes);

    /* The first frame, the one after a frame that is not checkable and the
     * one after a new reference are taken whole and computed as a frame
     * that is not gated; the frame after each of them fills the caches; the
     * others are computed from them. */
    constexpr size_t n_frames  = 14;
    size_t           n_outputs = 0, n_held = 0, n_skipped = 0, n_differ = 0, n_cached = 0;
    size_t           n_whole = 0; /* the samples of a frame computed whole */
    for (size_t k = 0; k < n_frames; k++) {
        uint8_t *img = moved(w, h, k % 4, cur); /* the block comes back to where it was */
        /* Above threshold 0 a little noise, on other pixels each frame;
         * at 0 it would change every tile and nothing would be kept. */
        for (size_t y = 0; threshold > 0 && y < h; y++) {
            for (size_t x = 0; x < w; x++) {
                if ((x * 7 + y * 13 + k * 5) % 97 == 0) {
                    img[(y * w + x) * 3 + 1] ^= 3;
                }
            }
        }
        if (k == 5) { /* half brightness: not checkable, and every tile changes */
            for (size_t i = 0; i < n_rgb; i++) {
                img[i] >>= 1;
            }
        }
        if (k == 10) { /* a new reference: the next frame is taken whole */
            CHECK(geist_decknet_stream_set_reference(s, &fref) == GEIST_OK);
            have_kept = false;
        }
        struct geist_decknet_frame f = make_frame(format, w, h, pad, img);
        f.timestamp                  = k;

        size_t n_taken = 0;
        for (size_t y0 = 0; y0 < h; y0 += GEIST_DECKNET_TILE) {
            for (size_t x0 = 0; x0 < w; x0 += GEIST_DECKNET_TILE) {
                size_t       at[GEIST_DECKNET_TILE * GEIST_DECKNET_TILE * 3];
                const size_t x1   = x0 + GEIST_DECKNET_TILE < w ? x0 + GEIST_DECKNET_TILE : w;
                const size_t y1   = y0 + GEIST_DECKNET_TILE < h ? y0 + GEIST_DECKNET_TILE : h;
                const size_t n    = tile_bytes(&f, x0, x1, y0, y1, at);
                uint32_t     diff = 0;
                for (size_t i = 0; i < n; i++) {
                    diff += (uint32_t) abs((int) f.data[at[i]] - (int) kept[at[i]]);
                }
                if (have_kept && diff <= threshold) {
                    n_held += diff > 0; /* differs, and stays */
                    continue;
                }
                for (size_t i = 0; i < n; i++) {
                    kept[at[i]] = f.data[at[i]];
                }
                n_taken++;
            }
        }
        have_kept                        = true;
        struct geist_decknet_frame image = f;
        image.data                       = kept;
        uint8_t *img8                    = frame_to_rgb8(&image);
        uint8_t *cur8                    = frame_to_rgb8(&f);
        n_differ += memcmp(img8, cur8, n_rgb) != 0;

        struct geist_decknet_result want;
        CHECK(geist_decknet_run(still, n_rgb, img8, ref8, &want) == GEIST_OK);
        CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
        struct geist_decknet_stream_result got;
        CHECK(geist_decknet_stream_result(s, &got) == GEIST_OK && got.timestamp == k);
        if (!same_outputs(still, &want, s, &info, &n_outputs)) {
            fprintf(stderr,
                    "FAIL %s: frame %zu differs from the still path on the kept tiles\n",
                    what,
                    k);
            g_fail = 1;
        }
        /* The stream keeps only the tiles the network reads. */
        CHECK(got.n_tiles > 0 && got.n_tiles <= ((w + 15) / 16) * ((h + 15) / 16));
        CHECK(got.n_tiles_changed <= got.n_tiles && got.n_tiles_changed <= n_taken);
        CHECK(want.checkable == (k != 5));
        n_skipped += got.n_tiles_changed < got.n_tiles;
        if (k == 0) {
            n_whole = got.n_samples;
            CHECK(got.n_tiles_changed == got.n_tiles);
        }
        n_cached += want.checkable && got.n_samples < n_whole;
        free(cur8);
        free(img8);
        free((void *) (uintptr_t) f.data);
        free(img);
    }
    /* The sequence has to reach what it claims to test: frames in which
     * tiles were kept, and above threshold 0 tiles that differed and were
     * kept all the same, so that the image is not the frame. */
    CHECK(n_skipped >= 8);
    CHECK(n_cached >= 6); /* frames that sampled less than all: computed from what was kept */
    CHECK(threshold == 0 ? (n_held == 0 && n_differ == 0) : (n_held > 20 && n_differ >= 5));
    printf("%-34s %zu frames, %zu with tiles kept, %zu computed from the caches, %zu outputs "
           "bit-identical to the still path on the kept tiles (%zu tiles kept although they "
           "differed)\n",
           what,
           (size_t) n_frames,
           n_skipped,
           n_cached,
           n_outputs,
           n_held);

    geist_decknet_stream_destroy(s);
    free(kept);
    free(ref8);
    free((void *) (uintptr_t) fref.data);
    free(cur);
    free(ref);
    geist_decknet_destroy(d);
    geist_decknet_destroy(still);
}

/* What a gated stream keeps is wrong when it is stale, and the places where
 * that hides are edges: a kernel window that reaches into the next cell of
 * the crop, and a frame that is not checkable (its tiles are kept, but no
 * slot is computed from them). A network on which one camera pixel is one
 * crop pixel and a tile away from the next: crop pixel (x, y) reads image
 * pixel (40 x, y), or (x, 40 y) when `vertical`; the units read the crop
 * pixels 15 and 16 along that axis, either side of a cell border, with 3 x 3
 * kernels. Each frame is compared with the still path on the same frame. */
constexpr size_t EDGE_LONG       = 960;
constexpr size_t EDGE_SHORT      = 16;
constexpr size_t EDGE_CROP_LONG  = 24;
constexpr size_t EDGE_CROP_SHORT = 6;

static struct geist_decknet *load_edges(bool vertical) {
    const size_t crop_w = vertical ? EDGE_CROP_SHORT : EDGE_CROP_LONG;
    struct tiny  m;
    tiny_default(&m);
    m.img_w  = (uint32_t) (vertical ? EDGE_SHORT : EDGE_LONG);
    m.img_h  = (uint32_t) (vertical ? EDGE_LONG : EDGE_SHORT);
    m.crop_w = (uint32_t) crop_w;
    m.crop_h = (uint32_t) (vertical ? EDGE_CROP_LONG : EDGE_CROP_SHORT);
    float hm[11 * 9];
    for (size_t i = 0; i < 11; i++) {
        const float along[9]  = {40, 0, 0.5f, 0, 1, 0.5f, 0, 0, 1};
        const float across[9] = {1, 0, 0.5f, 0, 40, 0.5f, 0, 0, 1};
        memcpy(hm + i * 9, vertical ? across : along, sizeof along);
    }
    tiny_set(&m.t[TINY_T_HOMOGRAPHY], "decknet.homography", 3, 11, 3, 3, 99, hm);
    /* Crop pixel k along the long axis, two in from the edge on the other. */
#define AT(k) ((float) (vertical ? (k) * crop_w + 2 : 2 * crop_w + (k)))
    const float on[6]   = {0, AT(15), 1, 1, AT(16), 1};
    const float off[6]  = {0, AT(14), 1, 1, AT(17), 1};
    const float tube[6] = {0, AT(15), 1, 1, AT(16), 1};
    tiny_pool(&m.t[TINY_T_POOL_ON], "decknet.pool.on", 2, on);
    tiny_pool(&m.t[TINY_T_POOL_OFF], "decknet.pool.off", 2, off);
    tiny_pool(&m.t[TINY_T_POOL_TUBE], "decknet.pool.tube", 2, tube);
#undef AT
    static struct tiny_buf buf;
    tiny_write(&m, &buf);
    struct geist_decknet *d = nullptr;
    if (geist_decknet_load_from_memory(buf.n, buf.b, &d) != GEIST_OK) {
        broken(geist_last_create_error(), "decknet_tiny, edges");
    }
    return d;
}

static void test_gate_edges(bool vertical) {
    const size_t w = vertical ? EDGE_SHORT : EDGE_LONG, h = vertical ? EDGE_LONG : EDGE_SHORT;
    struct geist_decknet     *still = load_edges(vertical), *d = load_edges(vertical);
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);

    const size_t n_rgb = w * h * 3;
    uint8_t     *ref   = xmalloc(n_rgb);
    uint8_t     *img   = xmalloc(n_rgb);
    memset(ref, 120, n_rgb);
    struct geist_decknet_frame   f = {.format  = GEIST_DECKNET_FORMAT_RGB8,
                                      .width   = w,
                                      .height  = h,
                                      .stride  = w * 3,
                                      .n_bytes = n_rgb,
                                      .data    = ref};
    struct geist_decknet_stream *s = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    CHECK(geist_decknet_stream_set_gate(s, true, 0) == GEIST_OK);
    CHECK(geist_decknet_stream_set_reference(s, &f) == GEIST_OK);
    f.data = img;

    /* Per frame: which crop pixels along the axis are brighter, and whether
     * the first pixel of the worktable strip is white (not checkable). */
    const struct {
        bool   bright15, bright16, strip;
        size_t tiles_changed; /* SIZE_MAX: not asserted */
    } frames[] = {
            {false, false, false, SIZE_MAX}, /* the first: taken whole, computed whole */
            {false, false, false, 0},        /* the same: fills the caches */
            {false, true, false, 1},         /* 16 changes: the unit at 15 reads it */
            {true, false, false, 2},         /* 15 changes, 16 back: the unit at 16 reads 15 */
            {true, false, false, 0},         /* nothing */
            {true, true, true, 2},           /* not checkable, and 16 changes */
            {true, true, false, 1},  /* only the strip changes back: 16 is of the frame before */
            {true, false, false, 1}, /* and 16 back, from the caches that frame filled */
    };
    size_t n_outputs = 0, n_whole = 0;
    for (size_t k = 0; k < sizeof frames / sizeof frames[0]; k++) {
        memset(img, 120, n_rgb);
        const size_t along[2] = {15 * 40, 16 * 40};
        const bool   on_[2]   = {frames[k].bright15, frames[k].bright16};
        for (size_t i = 0; i < 2; i++) {
            const size_t x = vertical ? 2 : along[i], y = vertical ? along[i] : 2;
            memset(img + (y * w + x) * 3, on_[i] ? 160 : 120, 3);
        }
        if (frames[k].strip) {
            memset(img, 255, 3);
        }
        struct geist_decknet_result        want;
        struct geist_decknet_stream_result got;
        CHECK(geist_decknet_run(still, n_rgb, img, ref, &want) == GEIST_OK);
        CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
        CHECK(geist_decknet_stream_result(s, &got) == GEIST_OK);
        CHECK(want.checkable == !frames[k].strip);
        CHECK(frames[k].tiles_changed == SIZE_MAX ||
              got.n_tiles_changed == frames[k].tiles_changed);
        /* Frames 2 to 4 and the last are computed from the caches: they
         * sample less than a whole frame (as frame 5 does, of which only
         * the strip is sampled). */
        n_whole = k == 0 ? got.n_samples : n_whole;
        CHECK((k >= 2 && k != 6) == (got.n_samples < n_whole));
        if (!same_outputs(still, &want, s, &info, &n_outputs)) {
            fprintf(stderr,
                    "FAIL gate edges (%s): frame %zu differs from the still path\n",
                    vertical ? "vertical" : "horizontal",
                    k);
            g_fail = 1;
        }
        if (want.checkable && k > 1) { /* a bright pixel shows in the units at all */
            struct geist_decknet_slot slot;
            CHECK(geist_decknet_stream_slot(s, 0, &slot) == GEIST_OK &&
                  (slot.plate > 0.0f || slot.rack > 0.0f));
        }
    }
    printf("gated, a window across a cell border, %-10s %zu outputs bit-identical to the still "
           "path\n",
           vertical ? "vertical:" : "horizontal:",
           n_outputs);
    geist_decknet_stream_destroy(s);
    free(img);
    free(ref);
    geist_decknet_destroy(d);
    geist_decknet_destroy(still);
}

/* The threshold itself: a tile that differs from the kept one by exactly the
 * threshold is kept, by one more it is taken. On the network above, crop
 * pixel 16 of the long axis moves by 30 and then by 31 in all against the
 * empty deck's grey, at threshold 30. */
static void test_gate_threshold(void) {
    constexpr size_t          w     = EDGE_LONG;
    constexpr size_t          h     = EDGE_SHORT;
    constexpr size_t          n_rgb = w * h * 3;
    struct geist_decknet     *still = load_edges(false), *d = load_edges(false);
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    uint8_t *ref = xmalloc(n_rgb), *img = xmalloc(n_rgb);
    memset(ref, 120, n_rgb);
    struct geist_decknet_frame   f = {.format  = GEIST_DECKNET_FORMAT_RGB8,
                                      .width   = w,
                                      .height  = h,
                                      .stride  = w * 3,
                                      .n_bytes = n_rgb,
                                      .data    = ref};
    struct geist_decknet_stream *s = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    CHECK(geist_decknet_stream_set_gate(s, true, 30) == GEIST_OK);
    CHECK(geist_decknet_stream_set_reference(s, &f) == GEIST_OK);
    CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK); /* the grey frame is the kept one */
    CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK); /* and what the caches are of */
    f.data = img;

    const uint8_t by[2][3]  = {{130, 130, 130}, {131, 130, 130}}; /* 30 in all, then 31 */
    size_t        n_outputs = 0;
    for (size_t k = 0; k < 2; k++) {
        memset(img, 120, n_rgb);
        memcpy(img + (2 * w + 16 * 40) * 3, by[k], 3);
        struct geist_decknet_result        want;
        struct geist_decknet_stream_result got;
        /* What the still path gives for the image of the kept tiles: the
         * grey one while the tile is kept, the frame once it is taken. */
        CHECK(geist_decknet_run(still, n_rgb, k == 0 ? ref : img, ref, &want) == GEIST_OK);
        CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
        CHECK(geist_decknet_stream_result(s, &got) == GEIST_OK);
        CHECK(got.n_tiles_changed == k);
        if (!same_outputs(still, &want, s, &info, &n_outputs)) {
            fprintf(stderr, "FAIL gate threshold: a difference of %zu\n", 30 + k);
            g_fail = 1;
        }
    }
    printf("gated, a tile at the threshold is kept and one above it taken: %zu outputs "
           "bit-identical\n",
           n_outputs);
    geist_decknet_stream_destroy(s);
    free(img);
    free(ref);
    geist_decknet_destroy(d);
    geist_decknet_destroy(still);
}

/* Every tile on its own. A gated stream skips whole stretches of its sample
 * tables when no tile they read was taken; a stretch skipped wrongly is a
 * stale result. On the invented network, frame k differs from the image in
 * two pixels at a corner of tile k alone (and so from the frame before in
 * tile k - 1 as well); each is compared with the still path on that frame. */
static void test_gate_each_tile(void) {
    struct geist_decknet     *still = load_synth();
    struct geist_decknet     *d     = load_synth();
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    const size_t                 w = info.image_width, h = info.image_height, n_rgb = w * h * 3;
    uint8_t                     *cur = read_ppm("tests/data/decknet_synth/cur.ppm", w, h);
    uint8_t                     *ref = read_ppm("tests/data/decknet_synth/ref.ppm", w, h);
    uint8_t                     *img = xmalloc(n_rgb);
    struct geist_decknet_frame   f   = {.format  = GEIST_DECKNET_FORMAT_RGB8,
                                        .width   = w,
                                        .height  = h,
                                        .stride  = w * 3,
                                        .n_bytes = n_rgb,
                                        .data    = ref};
    struct geist_decknet_stream *s   = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    CHECK(geist_decknet_stream_set_gate(s, true, 0) == GEIST_OK);
    CHECK(geist_decknet_stream_set_reference(s, &f) == GEIST_OK);
    f.data = img;

    const size_t tiles_w   = (w + GEIST_DECKNET_TILE - 1) / GEIST_DECKNET_TILE;
    const size_t tiles_h   = (h + GEIST_DECKNET_TILE - 1) / GEIST_DECKNET_TILE;
    size_t       n_outputs = 0, n_read = 0, n_checkable = 0;
    for (size_t k = 0; k < tiles_w * tiles_h * 4; k++) {
        const size_t t = k / 4, corner = k % 4;
        const size_t x0 = t % tiles_w * GEIST_DECKNET_TILE, y0 = t / tiles_w * GEIST_DECKNET_TILE;
        const size_t x1 = x0 + GEIST_DECKNET_TILE < w ? x0 + GEIST_DECKNET_TILE : w;
        const size_t y1 = y0 + GEIST_DECKNET_TILE < h ? y0 + GEIST_DECKNET_TILE : h;
        const size_t x = corner % 2 == 0 ? x0 : x1 - 2, y = corner / 2 == 0 ? y0 : y1 - 1;
        memcpy(img, cur, n_rgb);
        for (size_t i = 0; i < 6; i++) { /* two pixels, well off what they were */
            img[(y * w + x) * 3 + i] = (uint8_t) (img[(y * w + x) * 3 + i] + 90);
        }
        struct geist_decknet_result        want;
        struct geist_decknet_stream_result got;
        CHECK(geist_decknet_run(still, n_rgb, img, ref, &want) == GEIST_OK);
        CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
        CHECK(geist_decknet_stream_result(s, &got) == GEIST_OK);
        CHECK(k == 0 || got.n_tiles_changed <= 2);
        n_read += k > 0 && got.n_tiles_changed > 0;
        n_checkable += want.checkable;
        if (!same_outputs(still, &want, s, &info, &n_outputs)) {
            fprintf(stderr,
                    "FAIL gate, tile %zu, corner %zu: differs from the still path\n",
                    t,
                    corner);
            g_fail = 1;
        }
    }
    /* Most tiles are read by the network, and most frames are checkable:
     * otherwise this compares little. */
    CHECK(n_read > tiles_w * tiles_h * 2 && n_checkable > tiles_w * tiles_h * 2);
    printf("gated, each of %zu tiles changed on its own at each corner: %zu outputs bit-identical "
           "to the still path (%zu frames checkable)\n",
           tiles_w * tiles_h,
           n_outputs,
           n_checkable);
    geist_decknet_stream_destroy(s);
    free(img);
    free(cur);
    free(ref);
    geist_decknet_destroy(d);
    geist_decknet_destroy(still);
}

/* The same at the seams of the sample tables. They are skipped in stretches
 * of 256 samples, by the rectangle of tiles a stretch reads, and the sample
 * that ends a stretch is the one a rectangle is most easily short of. A
 * network whose samples walk down the image, three to a crop row and each
 * crop row in tiles of its own (crop pixel (x, y) reads image pixel
 * (x, 40 y), kernels at every row), so that the sample that ends the first
 * stretch is alone in its tile. Around that row, one image pixel changes at
 * a time. */
static void test_gate_seams(void) {
    constexpr size_t w      = 16;
    constexpr size_t h      = 3840;
    constexpr size_t crop_w = 6;
    constexpr size_t n_rgb  = w * h * 3;
    struct tiny      m;
    tiny_default(&m);
    m.img_w  = (uint32_t) w;
    m.img_h  = (uint32_t) h;
    m.crop_w = (uint32_t) crop_w;
    m.crop_h = 96;
    float hm[11 * 9];
    for (size_t i = 0; i < 11; i++) {
        const float down[9] = {1, 0, 0.5f, 0, 40, 0.5f, 0, 0, 1};
        memcpy(hm + i * 9, down, sizeof down);
    }
    tiny_set(&m.t[TINY_T_HOMOGRAPHY], "decknet.homography", 3, 11, 3, 3, 99, hm);
    /* The units read crop pixel (2, y) for y = 1 .. 94: pool.off the rows
     * 1 to 48 and pool.on the rows 49 to 94, two units each. A brighter pixel
     * next to a row of pool.on shows in its unit; next to one of pool.off it
     * would be cut off at 0. */
    float on[46 * 3], off[48 * 3];
    for (size_t y = 1; y <= 94; y++) {
        float *row = y <= 48 ? off + (y - 1) * 3 : on + (y - 49) * 3;
        row[0]     = (float) ((y - 1) % 48 / 24);
        row[1]     = (float) (y * crop_w + 2);
        row[2]     = 1.0f / 24.0f;
    }
    const float tube[6] = {0, 15 * crop_w + 2, 1, 1, 16 * crop_w + 2, 1};
    tiny_pool(&m.t[TINY_T_POOL_ON], "decknet.pool.on", 46, on);
    tiny_pool(&m.t[TINY_T_POOL_OFF], "decknet.pool.off", 48, off);
    tiny_pool(&m.t[TINY_T_POOL_TUBE], "decknet.pool.tube", 2, tube);
    static struct tiny_buf buf;
    tiny_write(&m, &buf);
    struct geist_decknet *still = nullptr, *d = nullptr;
    if (geist_decknet_load_from_memory(buf.n, buf.b, &still) != GEIST_OK ||
        geist_decknet_load_from_memory(buf.n, buf.b, &d) != GEIST_OK) {
        broken(geist_last_create_error(), "decknet_tiny, seams");
    }
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);

    uint8_t *ref = xmalloc(n_rgb), *img = xmalloc(n_rgb);
    memset(ref, 120, n_rgb);
    struct geist_decknet_frame   f = {.format  = GEIST_DECKNET_FORMAT_RGB8,
                                      .width   = w,
                                      .height  = h,
                                      .stride  = w * 3,
                                      .n_bytes = n_rgb,
                                      .data    = ref};
    struct geist_decknet_stream *s = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    CHECK(geist_decknet_stream_set_gate(s, true, 0) == GEIST_OK);
    CHECK(geist_decknet_stream_set_reference(s, &f) == GEIST_OK);
    CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
    struct geist_decknet_stream_result got;
    CHECK(geist_decknet_stream_result(s, &got) == GEIST_OK);
    /* 6 floor samples and 96 rows of 3 on each other plane, per slot: the
     * first stretch of 256 ends on the first sample of a crop row. */
    CHECK(got.n_samples == 20 + 2 * (6 + 4 * 288));
    CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK); /* fills the caches */
    f.data = img;

    size_t n_outputs = 0;
    for (size_t y = 78; y <= 92; y++) {
        for (size_t x = 1; x <= 3; x++) {
            memset(img, 120, n_rgb);
            memset(img + (y * 40 * w + x) * 3, 170, 3);
            struct geist_decknet_result want;
            CHECK(geist_decknet_run(still, n_rgb, img, ref, &want) == GEIST_OK);
            CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
            CHECK(geist_decknet_stream_result(s, &got) == GEIST_OK);
            CHECK(want.checkable && got.n_tiles_changed >= 1 && got.n_tiles_changed <= 2);
            CHECK(got.n_samples < 20 + 2 * (6 + 4 * 288)); /* from the caches */
            struct geist_decknet_slot slot;
            CHECK(geist_decknet_slot(still, 0, &slot) == GEIST_OK && slot.plate > 0.0f);
            if (!same_outputs(still, &want, s, &info, &n_outputs)) {
                fprintf(stderr, "FAIL gate seams: crop pixel (%zu, %zu)\n", x, y);
                g_fail = 1;
            }
        }
    }
    printf("gated, one sample at a time across the seam of two stretches of the tables: %zu "
           "outputs bit-identical to the still path\n",
           n_outputs);
    geist_decknet_stream_destroy(s);
    free(img);
    free(ref);
    geist_decknet_destroy(d);
    geist_decknet_destroy(still);
}

/* ---- the sample ------------------------------------------------------- */

/* decknet_weights and decknet_blend, which every crop pixel of the still
 * path and of a stream goes through. */
static void test_sample(void) {
    constexpr size_t n_f   = 12;
    const double     whole = (double) DECKNET_WEIGHT;
    /* Fractions of a pixel: the ends, the middle, and pairs just below 1 at
     * which the three rounded weights add up to more than the whole. */
    const double  f[n_f]      = {0.0,
                                 0x1p-30,
                                 1.0 / 3.0,
                                 0.5,
                                 0.999,
                                 1.0 - 1831.8 / whole,
                                 1.0 - 1831.77 / whole,
                                 1.0 - 0x1p-13,
                                 1.0 - 0x1p-24,
                                 1.0 - 0x1p-25,
                                 1.0 - 0x1p-40,
                                 1.0 - 0x1p-50};
    const uint8_t bytes[6][4] = {{0, 0, 0, 0},
                                 {255, 255, 255, 255},
                                 {1, 1, 1, 1},
                                 {0, 255, 0, 255},
                                 {255, 0, 0, 255},
                                 {17, 200, 254, 3}};
    size_t        n_over = 0, n_checked = 0;
    for (size_t i = 0; i < n_f; i++) {
        for (size_t j = 0; j < n_f; j++) {
            /* The fraction a coordinate between two pixels has, as the
             * function takes it. */
            const double u = 3.0 + f[i], v = 5.0 + f[j];
            const double fu = u - 3.0, fv = v - 5.0;
            uint32_t     w[3];
            CHECK(u < 4.0 && v < 6.0);
            decknet_weights(u, v, w);
            /* Rounded on their own they may exceed the whole ... */
            const double sum = floor(fu * (1.0 - fv) * whole + 0.5) +
                               floor((1.0 - fu) * fv * whole + 0.5) + floor(fu * fv * whole + 0.5);
            n_over += sum > whole;
            /* ... the weights never do. */
            CHECK((uint64_t) w[0] + w[1] + w[2] <= DECKNET_WEIGHT);
            for (size_t k = 0; k < 6; k++) {
                const uint8_t *b = bytes[k];
                const uint32_t q = decknet_blend(w, b[0], b[1], b[2], b[3]);
                const double   exact =
                        ((double) b[0] * (1.0 - fu) + (double) b[1] * fu) * (1.0 - fv) +
                        ((double) b[2] * (1.0 - fu) + (double) b[3] * fu) * fv;
                CHECK(q <= DECKNET_Q_ONE);
                CHECK(fabs((double) q - exact * 65536.0) <= 2.0);
                if (b[0] == b[1] && b[1] == b[2] && b[2] == b[3]) {
                    CHECK(q == (uint32_t) b[0] << 16U); /* a flat image samples to itself */
                }
                n_checked++;
            }
        }
    }
    CHECK(n_over > 0); /* the fractions reach the case the clamp is for */
    printf("the fixed-point sample: %zu samples within 2 / 65536 of a byte value of the exact one, "
           "%zu pairs of fractions whose rounded weights exceed the whole\n",
           n_checked,
           n_over);
}

/* A kernel whose pixels do not sum in 32 bits: 17 x 17 taps of one weight on
 * a white image are 289 x 255 x 65536, more than 2^32. The loader splits a
 * weight with more than 256 taps, and the sum goes on in 64 bits. The mean
 * of a flat image is the image, so every tube unit is sat(3 x 0.5) = 1, at
 * the crop's edge (taps clamped) and inside it. */
static void test_kernel_sum(void) {
    constexpr size_t side = 17;
    constexpr size_t size = 48;
    struct tiny      m;
    tiny_default(&m);
    m.img_w = m.img_h = (uint32_t) size;
    m.crop_w = m.crop_h = 40;
    float flat[side * side];
    for (size_t i = 0; i < side * side; i++) {
        flat[i] = 1.0f / (float) (side * side);
    }
    tiny_set(&m.t[TINY_T_TUBE], "decknet.conv.tube", 2, side, side, 0, side * side, flat);
    const float tube[6] = {0, 19, 1, 1, 20 * 40 + 20, 1}; /* in the first row, and in the middle */
    tiny_pool(&m.t[TINY_T_POOL_TUBE], "decknet.pool.tube", 2, tube);
    static struct tiny_buf buf;
    tiny_write(&m, &buf);
    struct geist_decknet *d = nullptr;
    if (geist_decknet_load_from_memory(buf.n, buf.b, &d) != GEIST_OK) {
        broken(geist_last_create_error(), "decknet_tiny, 17 x 17 kernel");
    }
    uint8_t *white = xmalloc(size * size * 3);
    memset(white, 255, size * size * 3);
    struct geist_decknet_result res;
    struct geist_decknet_slot   slot;
    CHECK(geist_decknet_run(d, size * size * 3, white, white, &res) == GEIST_OK && res.checkable);
    CHECK(geist_decknet_slot(d, 0, &slot) == GEIST_OK);
    CHECK(slot.tubes[0] == 1.0f && slot.tubes_rot[0] == 1.0f);
    printf("a kernel of 289 taps of one weight on a white image: summed past 32 bits\n");
    free(white);
    geist_decknet_destroy(d);
}

/* The conversion over the whole range of its bytes. The frames of the parity
 * tests come from RGB images, so their Y, U and V stay inside what a colour
 * gives; a camera's need not. On the network above (its worktable strip
 * reads image columns 0 to 120, its units columns 520 to 720) every byte of
 * the columns from 500 on is drawn anew for each frame, and the stream has
 * to give the bits of the still path on the frame as this file converts
 * it. */
static void test_conversion_range(enum geist_decknet_format format) {
    constexpr size_t          w     = EDGE_LONG;
    constexpr size_t          h     = EDGE_SHORT;
    struct geist_decknet     *still = load_edges(false), *d = load_edges(false);
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    uint8_t *grey = xmalloc(w * h * 3);
    memset(grey, 120, w * h * 3);
    struct geist_decknet_frame   f    = make_frame(format, w, h, 5, grey);
    uint8_t                     *ref8 = frame_to_rgb8(&f);
    struct geist_decknet_stream *s    = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    CHECK(geist_decknet_stream_set_reference(s, &f) == GEIST_OK);
    uint8_t *bytes = (uint8_t *) (uintptr_t) f.data;

    uint64_t         seed      = 0x533;
    size_t           n_outputs = 0, n_extreme = 0;
    constexpr size_t n_frames = 200;
    for (size_t k = 0; k < n_frames; k++) {
        for (size_t y = 0; y < h; y++) {
            for (size_t x = 500; x < w; x++) {
                /* a third of the bytes at an end of the range */
                seed            = seed * 6364136223846793005ull + 1442695040888963407ull;
                const uint8_t v = (uint8_t) (seed >> 33U);
                const uint8_t b = (seed >> 41U) % 3 == 0 ? (v & 1 ? 255 : 0) : v;
                n_extreme += b == 0 || b == 255;
                if (format == GEIST_DECKNET_FORMAT_NV12) {
                    bytes[y * f.stride + x] = b;
                    if (y % 2 == 0) {
                        bytes[(h + y / 2) * f.stride + x] = (uint8_t) (b * 7 + v);
                    }
                } else {
                    bytes[y * f.stride + x * 2]     = b;
                    bytes[y * f.stride + x * 2 + 1] = (uint8_t) (b * 7 + v);
                }
            }
        }
        uint8_t                    *img8 = frame_to_rgb8(&f);
        struct geist_decknet_result want;
        CHECK(geist_decknet_run(still, w * h * 3, img8, ref8, &want) == GEIST_OK);
        CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
        CHECK(want.checkable);
        if (!same_outputs(still, &want, s, &info, &n_outputs)) {
            fprintf(stderr, "FAIL conversion range, format %u: frame %zu\n", (unsigned) format, k);
            g_fail = 1;
        }
        free(img8);
    }
    printf("%-34s %zu frames of arbitrary bytes (%zu of them 0 or 255), %zu outputs bit-identical "
           "to the still path\n",
           format == GEIST_DECKNET_FORMAT_NV12 ? "NV12, the whole byte range"
                                               : "YUYV, the whole byte range",
           (size_t) n_frames,
           n_extreme,
           n_outputs);
    geist_decknet_stream_destroy(s);
    free(bytes);
    free(ref8);
    free(grey);
    geist_decknet_destroy(d);
    geist_decknet_destroy(still);
}

/* The second implementation of the conversion gives the colours BT.601
 * names for its limited-range code points, and clamps outside them. */
static void test_conversion_reference(void) {
    const struct {
        int     y, u, v;
        uint8_t rgb[3];
    } known[] = {
            {16, 128, 128, {0, 0, 0}},
            {235, 128, 128, {255, 255, 255}},
            {81, 90, 240, {255, 0, 0}},
            {145, 54, 34, {0, 255, 0}},
            {41, 240, 110, {0, 0, 255}},
            {126, 128, 128, {128, 128, 128}},
            {0, 128, 128, {0, 0, 0}},
            {255, 128, 128, {255, 255, 255}},
            {180, 128, 128, {191, 191, 191}},
    };
    for (size_t i = 0; i < sizeof known / sizeof known[0]; i++) {
        uint8_t rgb[3];
        yuv_to_rgb(known[i].y, known[i].u, known[i].v, rgb);
        /* The standard's colours are rounded to code points: within one. */
        if (abs(rgb[0] - known[i].rgb[0]) > 1 || abs(rgb[1] - known[i].rgb[1]) > 1 ||
            abs(rgb[2] - known[i].rgb[2]) > 1) {
            fprintf(stderr,
                    "FAIL: YUV (%d, %d, %d) gives RGB (%u, %u, %u)\n",
                    known[i].y,
                    known[i].u,
                    known[i].v,
                    rgb[0],
                    rgb[1],
                    rgb[2]);
            g_fail = 1;
        }
    }
}

/* ---- the end of a frame ----------------------------------------------- */

/* A sample is computed from 16 bytes of each of its rows where the frame has
 * them, and from its six pixels' bytes alone where the buffer ends before.
 * Crops that lie on the last pixels of the image (and beyond it, clamped),
 * in buffers with 0 to 20 bytes behind the last pixel: the same bits as the
 * still path each time, and ASan sees a read behind an exact buffer. */
static void test_frame_end(void) {
    constexpr size_t w     = TINY_IMG_W;
    constexpr size_t h     = TINY_IMG_H;
    constexpr size_t n_rgb = w * h * 3;
    struct tiny      m;
    tiny_default(&m);
    float hm[11 * 9];
    for (size_t i = 0; i < 11; i++) {
        /* crop pixel (x, y) reads the image at (w - 4 + 0.7 x, h - 3 + 0.6 y) */
        const float corner[9] = {0.7f, 0, (float) w - 3.6f, 0, 0.6f, (float) h - 2.7f, 0, 0, 1};
        memcpy(hm + i * 9, corner, sizeof corner);
    }
    tiny_set(&m.t[TINY_T_HOMOGRAPHY], "decknet.homography", 3, 11, 3, 3, 99, hm);
    static struct tiny_buf buf;
    tiny_write(&m, &buf);
    struct geist_decknet *still = nullptr, *d = nullptr;
    if (geist_decknet_load_from_memory(buf.n, buf.b, &still) != GEIST_OK ||
        geist_decknet_load_from_memory(buf.n, buf.b, &d) != GEIST_OK) {
        broken(geist_last_create_error(), "decknet_tiny, the end of a frame");
    }
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    uint8_t ref[n_rgb], img[n_rgb];
    for (size_t i = 0; i < n_rgb; i++) {
        ref[i] = (uint8_t) (110 + (i * 7) % 21);
        img[i] = (uint8_t) (ref[i] + (i * 5) % 7 - 3); /* near it: the pair stays checkable */
    }
    struct geist_decknet_stream *s = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    size_t n_outputs = 0, n_checkable = 0;
    for (size_t behind = 0; behind <= 20; behind++) {
        struct geist_decknet_frame f    = {.format  = GEIST_DECKNET_FORMAT_RGB8,
                                           .width   = w,
                                           .height  = h,
                                           .stride  = w * 3,
                                           .n_bytes = n_rgb + behind};
        uint8_t                   *fref = xmalloc(f.n_bytes), *fimg = xmalloc(f.n_bytes);
        memset(fref, 0xA5, f.n_bytes);
        memset(fimg, 0x5A, f.n_bytes);
        memcpy(fref, ref, n_rgb);
        memcpy(fimg, img, n_rgb);
        struct geist_decknet_result want;
        CHECK(geist_decknet_run(still, n_rgb, img, ref, &want) == GEIST_OK);
        f.data = fref;
        CHECK(geist_decknet_stream_set_reference(s, &f) == GEIST_OK);
        f.data = fimg;
        CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
        if (!same_outputs(still, &want, s, &info, &n_outputs)) {
            fprintf(stderr, "FAIL the end of a frame: %zu bytes behind it\n", behind);
            g_fail = 1;
        }
        n_checkable += want.checkable;
        free(fimg);
        free(fref);
    }
    CHECK(n_checkable == 21); /* otherwise no slot was sampled */
    printf("samples at the last pixels of a frame with 0 to 20 bytes behind it: %zu outputs "
           "bit-identical to the still path\n",
           n_outputs);
    geist_decknet_stream_destroy(s);
    geist_decknet_destroy(d);
    geist_decknet_destroy(still);
}

/* ---- a queue of one --------------------------------------------------- */

/* Every output of the last frame, as bits. Returns how many. */
constexpr size_t QUEUE_OUTPUTS = 4096;
static size_t    outputs_of(const struct geist_decknet_stream *s,
                            const struct geist_decknet_info   *info,
                            float                              out[static QUEUE_OUTPUTS]) {
    struct geist_decknet_stream_result res;
    size_t                             n = 0;
    if (geist_decknet_stream_result(s, &res) != GEIST_OK || !res.image.checkable) {
        return 0;
    }
    memcpy(out, res.image.gain, sizeof res.image.gain);
    out[3] = res.image.bg_changed;
    n      = 4;
    for (size_t i = 0; i < info->n_slots; i++) {
        struct geist_decknet_slot slot;
        if (geist_decknet_stream_slot(s, i, &slot) != GEIST_OK ||
            n + 8 + info->n_tips + 2 * info->n_tubes > QUEUE_OUTPUTS) {
            return 0;
        }
        const float sc[8] = {slot.occ_front,
                             slot.occ_rear,
                             slot.wall_front,
                             slot.rack,
                             slot.plate,
                             slot.deep,
                             slot.tube_n,
                             slot.tube_r};
        memcpy(out + n, sc, sizeof sc);
        memcpy(out + n + 8, slot.tips, info->n_tips * sizeof(float));
        memcpy(out + n + 8 + info->n_tips, slot.tubes, info->n_tubes * sizeof(float));
        memcpy(out + n + 8 + info->n_tips + info->n_tubes,
               slot.tubes_rot,
               info->n_tubes * sizeof(float));
        n += 8 + info->n_tips + 2 * info->n_tubes;
    }
    return n;
}

constexpr size_t QUEUE_IMAGES = 4;    /* distinct frames */
constexpr size_t QUEUE_ROUNDS = 12;   /* the slow consumer: three offers, one pass */
constexpr size_t QUEUE_FREE   = 3000; /* offers of the thread that does not wait */

/* What the two threads of the queue test share. A frame is offered once;
 * `dropped` and `done` say how it came back, each written by one thread. */
struct queue_test {
    struct geist_decknet_stream *s;
    size_t                       n_frames;
    struct geist_decknet_frame  *frames;
    uint8_t                     *dropped;     /* by the thread that offers */
    uint8_t                     *done;        /* by the thread that computes */
    atomic_size_t                go, offered; /* rounds asked for, rounds offered */
    atomic_bool                  finished;
    size_t                       n_wrong; /* by the thread that offers */
};

static void *queue_producer(void *arg) {
    struct queue_test *q = arg;
    /* The slow consumer: a round is three frames, and the consumer takes
     * one pass when all three are offered. The first finds nothing waiting;
     * each of the others gets the one before it back. */
    for (size_t r = 0; r < QUEUE_ROUNDS; r++) {
        while (atomic_load(&q->go) <= r) {
            sched_yield();
        }
        for (size_t j = 0; j < 3; j++) {
            const struct geist_decknet_frame *f    = &q->frames[r * 3 + j];
            const struct geist_decknet_frame *back = f;
            q->n_wrong += geist_decknet_stream_offer(q->s, f, &back) != GEIST_OK;
            q->n_wrong += back != (j == 0 ? nullptr : f - 1);
            if (back != nullptr) {
                q->dropped[back - q->frames]++;
            }
        }
        atomic_store(&q->offered, r + 1);
    }
    /* And without waiting for anyone: as fast as this thread can. */
    while (atomic_load(&q->go) <= QUEUE_ROUNDS) {
        sched_yield();
    }
    for (size_t i = QUEUE_ROUNDS * 3; i < q->n_frames; i++) {
        const struct geist_decknet_frame *back = nullptr;
        q->n_wrong += geist_decknet_stream_offer(q->s, &q->frames[i], &back) != GEIST_OK;
        if (back != nullptr) {
            q->n_wrong += back < q->frames || back >= &q->frames[i];
            q->dropped[back - q->frames]++;
        }
        if (i % 4 == 0) { /* so that the other thread gets passes in between */
            nanosleep(&(struct timespec) {.tv_nsec = 50000}, nullptr);
        }
    }
    atomic_store(&q->finished, true);
    return nullptr;
}

static void test_queue(size_t threads, bool gated) {
    struct geist_decknet     *d = load_synth();
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    const size_t w = info.image_width, h = info.image_height;
    uint8_t     *cur = read_ppm("tests/data/decknet_synth/cur.ppm", w, h);
    uint8_t     *ref = read_ppm("tests/data/decknet_synth/ref.ppm", w, h);

    struct geist_decknet_stream *s = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    CHECK(geist_decknet_stream_set_threads(s, threads) == GEIST_OK);
    struct geist_decknet_frame fref = make_frame(GEIST_DECKNET_FORMAT_NV12, w, h, 3, ref);
    CHECK(geist_decknet_stream_set_reference(s, &fref) == GEIST_OK);

    /* The frames, and what geist_decknet_stream_push_frame gives for each. */
    struct geist_decknet_frame image[QUEUE_IMAGES];
    static float               want[QUEUE_IMAGES][QUEUE_OUTPUTS];
    size_t                     n_want = 0;
    for (size_t k = 0; k < QUEUE_IMAGES; k++) {
        uint8_t *img = moved(w, h, k, cur);
        image[k]     = make_frame(GEIST_DECKNET_FORMAT_NV12, w, h, 3, img);
        free(img);
        CHECK(geist_decknet_stream_push_frame(s, &image[k]) == GEIST_OK);
        n_want = outputs_of(s, &info, want[k]);
        CHECK(n_want > 4);
    }
    CHECK(geist_decknet_stream_set_gate(s, gated, 0) == GEIST_OK);
    struct geist_decknet_stream_result res;
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.n_dropped == 0);
    const uint64_t n_before = res.n_frames;

    /* One thread, step by step. */
    struct geist_decknet_frame        a = image[0], b = image[1], c = image[2];
    const struct geist_decknet_frame *back = &a;
    static float                      got[QUEUE_OUTPUTS];
    a.timestamp = 11;
    b.timestamp = 12;
    c.timestamp = 13;
    CHECK(geist_decknet_stream_push_offered(s, &back) == GEIST_OK && back == nullptr);
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.n_frames == n_before);
    CHECK(geist_decknet_stream_offer(s, &a, &back) == GEIST_OK && back == nullptr);
    CHECK(geist_decknet_stream_offer(s, &b, &back) == GEIST_OK && back == &a);
    CHECK(geist_decknet_stream_offer(s, &c, &back) == GEIST_OK && back == &b);
    /* Offered frames are not computed on, and not counted as frames. */
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.n_frames == n_before &&
          res.n_dropped == 2);
    CHECK(geist_decknet_stream_push_offered(s, &back) == GEIST_OK && back == &c);
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.n_frames == n_before + 1 &&
          res.timestamp == 13 && res.n_dropped == 2);
    CHECK(outputs_of(s, &info, got) == n_want && memcmp(got, want[2], n_want * sizeof(float)) == 0);
    CHECK(geist_decknet_stream_push_offered(s, &back) == GEIST_OK && back == nullptr);
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.n_frames == n_before + 1 &&
          res.timestamp == 13);
    /* A frame that would be refused is not queued, and replaces nothing. */
    struct geist_decknet_frame bad = image[3];
    bad.n_bytes--;
    CHECK(geist_decknet_stream_offer(s, &a, &back) == GEIST_OK && back == nullptr);
    back = &a;
    CHECK(geist_decknet_stream_offer(s, &bad, &back) == GEIST_E_INVALID_ARG && back == nullptr);
    CHECK(geist_decknet_stream_push_offered(s, &back) == GEIST_OK && back == &a);
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.timestamp == 11 &&
          res.n_dropped == 2);
    /* Null arguments. */
    back = &a;
    CHECK(geist_decknet_stream_offer(nullptr, &a, &back) == GEIST_E_INVALID_ARG && back == nullptr);
    back = &a;
    CHECK(geist_decknet_stream_offer(s, nullptr, &back) == GEIST_E_INVALID_ARG && back == nullptr);
    CHECK(geist_decknet_stream_offer(s, &a, nullptr) == GEIST_E_INVALID_ARG);
    back = &a;
    CHECK(geist_decknet_stream_push_offered(nullptr, &back) == GEIST_E_INVALID_ARG &&
          back == nullptr);
    CHECK(geist_decknet_stream_push_offered(s, nullptr) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_push_offered(s, &back) == GEIST_OK && back == nullptr);

    /* Two threads. Frame i is image i % QUEUE_IMAGES with timestamp 100 + i. */
    struct queue_test q = {.s = s, .n_frames = QUEUE_ROUNDS * 3 + QUEUE_FREE};
    q.frames            = xmalloc(q.n_frames * sizeof q.frames[0]);
    q.dropped           = xmalloc(q.n_frames);
    q.done              = xmalloc(q.n_frames);
    memset(q.dropped, 0, q.n_frames);
    memset(q.done, 0, q.n_frames);
    for (size_t i = 0; i < q.n_frames; i++) {
        q.frames[i]           = image[i % QUEUE_IMAGES];
        q.frames[i].timestamp = 100 + i;
    }
    pthread_t producer;
    if (pthread_create(&producer, nullptr, queue_producer, &q) != 0) {
        broken("pthread_create", "the queue test");
    }
    size_t n_same = 0;
    /* The consumer is slow by construction: it asks for a round of three
     * offers, waits until they are made, and takes one pass. */
    for (size_t r = 0; r < QUEUE_ROUNDS; r++) {
        atomic_store(&q.go, r + 1);
        while (atomic_load(&q.offered) <= r) {
            sched_yield();
        }
        CHECK(geist_decknet_stream_push_offered(s, &back) == GEIST_OK &&
              back == &q.frames[r * 3 + 2]);
        CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK &&
              res.timestamp == 100 + r * 3 + 2 && res.n_dropped == 2 + 2 * (r + 1));
        const size_t n = outputs_of(s, &info, got);
        n_same += n == n_want &&
                  memcmp(got, want[(r * 3 + 2) % QUEUE_IMAGES], n * sizeof(float)) == 0;
        q.done[r * 3 + 2]++;
    }
    CHECK(n_same == QUEUE_ROUNDS);
    /* And a consumer that takes what is there while the other thread offers
     * without waiting, until that one is done and nothing waits. */
    atomic_store(&q.go, QUEUE_ROUNDS + 1);
    size_t   n_done = 0, n_order = 0;
    uint64_t last = 0;
    for (;;) {
        const bool finished = atomic_load(&q.finished);
        CHECK(geist_decknet_stream_push_offered(s, &back) == GEIST_OK);
        if (back == nullptr) {
            if (finished) {
                break;
            }
            sched_yield();
            continue;
        }
        const size_t i = (size_t) (back - q.frames);
        CHECK(i >= QUEUE_ROUNDS * 3 && i < q.n_frames);
        q.done[i]++;
        CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.timestamp == 100 + i);
        n_order += res.timestamp > last;
        last           = res.timestamp;
        const size_t n = outputs_of(s, &info, got);
        n_same += n == n_want && memcmp(got, want[i % QUEUE_IMAGES], n * sizeof(float)) == 0;
        n_done++;
    }
    pthread_join(producer, nullptr);
    CHECK(q.n_wrong == 0);
    CHECK(n_done >= 1 && n_order == n_done && n_same == QUEUE_ROUNDS + n_done);
    CHECK(last == 100 + q.n_frames - 1); /* the newest frame is never the one dropped */
    size_t n_dropped = 0, n_once = 0;
    for (size_t i = 0; i < q.n_frames; i++) {
        n_dropped += q.dropped[i];
        n_once += q.dropped[i] + q.done[i] == 1;
    }
    CHECK(n_once == q.n_frames); /* every frame came back, and once */
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.n_dropped == 2 + n_dropped &&
          n_dropped + QUEUE_ROUNDS + n_done == q.n_frames);
    printf("a queue of one, %zu thread%s, %s: a consumer that takes one of three computes the "
           "third and %zu are dropped; of %zu frames offered without waiting %zu were computed, "
           "in order, and %zu dropped; each came back once\n",
           threads,
           threads == 1 ? "" : "s",
           gated ? "gated" : "not gated",
           (size_t) (2 * QUEUE_ROUNDS),
           (size_t) QUEUE_FREE,
           n_done,
           n_dropped - 2 * QUEUE_ROUNDS);

    /* A stream without a reference hands the frame back with its status. */
    struct geist_decknet_stream *s2 = nullptr;
    CHECK(geist_decknet_stream_create(d, &s2) == GEIST_E_INVALID_ARG || s2 != nullptr);
    if (s2 != nullptr) {
        CHECK(geist_decknet_stream_offer(s2, &a, &back) == GEIST_OK && back == nullptr);
        CHECK(geist_decknet_stream_push_offered(s2, &back) == GEIST_E_INVALID_STATE && back == &a);
        /* A frame that still waits when the stream goes is the caller's. */
        CHECK(geist_decknet_stream_offer(s2, &a, &back) == GEIST_OK && back == nullptr);
        geist_decknet_stream_destroy(s2);
    }

    geist_decknet_stream_destroy(s);
    free(q.done);
    free(q.dropped);
    free(q.frames);
    for (size_t k = 0; k < QUEUE_IMAGES; k++) {
        free((void *) (uintptr_t) image[k].data);
    }
    free((void *) (uintptr_t) fref.data);
    free(cur);
    free(ref);
    geist_decknet_destroy(d);
}

/* ---- no allocation ---------------------------------------------------- */

static void test_no_alloc(void) {
    struct geist_decknet     *d = load_synth();
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    const size_t w = info.image_width, h = info.image_height;
    uint8_t     *cur  = read_ppm("tests/data/decknet_synth/cur.ppm", w, h);
    uint8_t     *ref  = read_ppm("tests/data/decknet_synth/ref.ppm", w, h);
    uint8_t     *half = xmalloc(w * h * 3); /* malloc, not heap.h: not counted */
    for (size_t i = 0; i < w * h * 3; i++) {
        half[i] = cur[i] >> 1;
    }
    struct geist_decknet_frame frames[3], halves[3], refs[3];
    for (uint32_t f = 0; f < 3; f++) {
        frames[f] = make_frame((enum geist_decknet_format) f, w, h, 5, cur);
        halves[f] = make_frame((enum geist_decknet_format) f, w, h, 5, half);
        refs[f]   = make_frame((enum geist_decknet_format) f, w, h, 5, ref);
    }

    const uint64_t               before_create = heap_alloc_count();
    struct geist_decknet_stream *s             = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    const uint64_t before_gate = heap_alloc_count();
    CHECK(before_gate > before_create); /* the counter sees the stream's allocations */
    /* Gating allocates once, when it is first turned on, and not when it is
     * turned off, on again or given another threshold. */
    CHECK(geist_decknet_stream_set_gate(s, false, 0) == GEIST_OK);
    CHECK(heap_alloc_count() == before_gate);
    CHECK(geist_decknet_stream_set_gate(s, true, 0) == GEIST_OK);
    const uint64_t before = heap_alloc_count();
    CHECK(before > before_gate);
    CHECK(geist_decknet_stream_set_gate(s, false, 0) == GEIST_OK);

    size_t n_calls = 0;
    for (int round = 0; round < 4; round++) {
        for (uint32_t f = 0; f < 3; f++) {
            struct geist_decknet_stream_result res;
            struct geist_decknet_slot          slot;
            /* rounds 0 and 3 without gating, 1 and 2 with; 2 and 3 on two threads */
            CHECK(geist_decknet_stream_set_gate(s, round == 1 || round == 2, (uint32_t) round) ==
                  GEIST_OK);
            CHECK(geist_decknet_stream_set_threads(s, round >= 2 ? 2 : 1) == GEIST_OK);
            CHECK(geist_decknet_stream_set_reference(s, &refs[f]) == GEIST_OK);
            CHECK(geist_decknet_stream_push_frame(s, &frames[f]) == GEIST_OK);
            CHECK(geist_decknet_stream_push_frame(s, &frames[f]) == GEIST_OK); /* nothing changed */
            CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.image.checkable);
            CHECK(geist_decknet_stream_slot(s, 0, &slot) == GEIST_OK);
            /* the other exit of a pass, and a refused frame */
            CHECK(geist_decknet_stream_push_frame(s, &halves[f]) == GEIST_OK);
            CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && !res.image.checkable);
            struct geist_decknet_frame bad = frames[f];
            bad.n_bytes--;
            CHECK(geist_decknet_stream_push_frame(s, &bad) == GEIST_E_INVALID_ARG);
            /* the queue of one: an offer, one that drops it, a pass */
            const struct geist_decknet_frame *back = nullptr;
            CHECK(geist_decknet_stream_offer(s, &halves[f], &back) == GEIST_OK);
            CHECK(geist_decknet_stream_offer(s, &frames[f], &back) == GEIST_OK &&
                  back == &halves[f]);
            CHECK(geist_decknet_stream_push_offered(s, &back) == GEIST_OK && back == &frames[f]);
            n_calls += 13;
            const uint64_t now = heap_alloc_count();
            if (now != before) {
                fprintf(stderr,
                        "FAIL: round %d, format %u: %llu allocations after the create\n",
                        round,
                        f,
                        (unsigned long long) (now - before));
                g_fail = 1;
            }
        }
    }
    printf("no allocation in %zu calls after geist_decknet_stream_create and the first "
           "geist_decknet_stream_set_gate, the first frame included\n",
           n_calls);
    geist_decknet_stream_destroy(s);
    for (uint32_t f = 0; f < 3; f++) {
        free((void *) (uintptr_t) frames[f].data);
        free((void *) (uintptr_t) halves[f].data);
        free((void *) (uintptr_t) refs[f].data);
    }
    free(half);
    free(cur);
    free(ref);
    geist_decknet_destroy(d);
}

/* ---- geometry --------------------------------------------------------- */

static struct geist_decknet *load_tiny(uint32_t img_w, uint32_t img_h) {
    struct tiny m;
    tiny_default(&m);
    m.img_w = img_w;
    m.img_h = img_h;
    static struct tiny_buf buf;
    tiny_write(&m, &buf);
    struct geist_decknet *d = nullptr;
    if (geist_decknet_load_from_memory(buf.n, buf.b, &d) != GEIST_OK) {
        broken(geist_last_create_error(), "decknet_tiny");
    }
    return d;
}

/* Pushes a copy of `f` whose buffer is exactly f->n_bytes long. */
static enum geist_status push_exact(struct geist_decknet_stream *s, struct geist_decknet_frame f) {
    uint8_t *copy = xmalloc(f.n_bytes ? f.n_bytes : 1);
    if (f.data != nullptr) {
        memcpy(copy, f.data, f.n_bytes);
        f.data = copy;
    }
    const enum geist_status st = geist_decknet_stream_push_frame(s, &f);
    free(copy);
    return st;
}

static void test_geometry(void) {
    constexpr size_t      w = TINY_IMG_W;
    constexpr size_t      h = TINY_IMG_H;
    struct geist_decknet *d = load_tiny((uint32_t) w, (uint32_t) h);
    uint8_t               rgb[TINY_IMG_W * TINY_IMG_H * 3];
    for (size_t i = 0; i < sizeof rgb; i++) {
        rgb[i] = (uint8_t) (100 + i % 50);
    }
    struct geist_decknet_stream *s = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    struct geist_decknet_frame ok = make_frame(GEIST_DECKNET_FORMAT_RGB8, w, h, 0, rgb);
    ok.timestamp                  = 7;
    CHECK(geist_decknet_stream_set_reference(s, &ok) == GEIST_OK);
    CHECK(geist_decknet_stream_push_frame(s, &ok) == GEIST_OK);

    size_t n_refused = 0, n_accepted = 0;
    for (uint32_t fmt = 0; fmt < 3; fmt++) {
        const enum geist_decknet_format format = (enum geist_decknet_format) fmt;
        for (size_t pad = 0; pad < 40; pad += 13) {
            struct geist_decknet_frame f = make_frame(format, w, h, pad, rgb);
            f.timestamp                  = 99;
            /* exact fit, and with room behind it */
            CHECK(push_exact(s, f) == GEIST_OK);
            n_accepted++;
            struct geist_decknet_frame bad;
#define REFUSED(change)                                                            \
    do {                                                                           \
        bad = f;                                                                   \
        change;                                                                    \
        CHECK(push_exact(s, bad) == GEIST_E_INVALID_ARG);                          \
        CHECK(geist_decknet_stream_set_reference(s, &bad) == GEIST_E_INVALID_ARG); \
        n_refused++;                                                               \
    } while (0)
            REFUSED(bad.n_bytes -= 1);
            REFUSED(bad.n_bytes = 0);
            REFUSED(bad.stride += 1); /* the same buffer cannot hold longer rows */
            REFUSED(bad.stride  = row_bytes(format, w) - 1;
                    bad.n_bytes = exact_bytes(format, w, h, bad.stride));
            REFUSED(bad.stride = 0);
            REFUSED(bad.stride = SIZE_MAX);
            REFUSED(bad.stride = SIZE_MAX / 2 + 1);
            REFUSED(bad.width = w - 2);
            REFUSED(bad.width = w + 2);
            REFUSED(bad.width = 0);
            REFUSED(bad.width = SIZE_MAX);
            REFUSED(bad.height = h - 2);
            REFUSED(bad.height = h + 2);
            REFUSED(bad.height = SIZE_MAX);
            REFUSED(bad.format = (enum geist_decknet_format) 3);
            REFUSED(bad.format = (enum geist_decknet_format) UINT32_MAX);
            REFUSED(bad.data = nullptr);
            if (format == GEIST_DECKNET_FORMAT_NV12) {
                REFUSED(bad.n_bytes = f.stride * h); /* the Y plane alone */
            }
            /* A frame of another format in a buffer sized for this one. */
            for (uint32_t other = 0; other < 3; other++) {
                bad             = f;
                bad.format      = (enum geist_decknet_format) other;
                const bool fits = bad.stride >= row_bytes(bad.format, w) &&
                                  exact_bytes(bad.format, w, h, bad.stride) <= bad.n_bytes;
                CHECK(push_exact(s, bad) == (fits ? GEIST_OK : GEIST_E_INVALID_ARG));
                n_refused += !fits;
            }
#undef REFUSED
            free((void *) (uintptr_t) f.data);
        }
    }
    /* A refused frame leaves the last result. */
    CHECK(geist_decknet_stream_push_frame(s, &ok) == GEIST_OK);
    struct geist_decknet_stream_result res, after;
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.timestamp == 7);
    struct geist_decknet_frame bad = ok;
    bad.n_bytes--;
    bad.timestamp = 8;
    CHECK(geist_decknet_stream_push_frame(s, &bad) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_result(s, &after) == GEIST_OK && after.timestamp == 7 &&
          after.n_frames == res.n_frames);
    free((void *) (uintptr_t) ok.data);
    geist_decknet_stream_destroy(s);
    geist_decknet_destroy(d);

    /* A model whose image is odd each way takes RGB8 only. */
    constexpr size_t ow = TINY_IMG_W - 1;
    constexpr size_t oh = TINY_IMG_H - 1;
    d                   = load_tiny((uint32_t) ow, (uint32_t) oh);
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK);
    struct geist_decknet_frame odd = make_frame(GEIST_DECKNET_FORMAT_RGB8, ow, oh, 0, rgb);
    CHECK(geist_decknet_stream_set_reference(s, &odd) == GEIST_OK);
    CHECK(geist_decknet_stream_push_frame(s, &odd) == GEIST_OK);
    free((void *) (uintptr_t) odd.data);
    uint8_t big[TINY_IMG_W * TINY_IMG_H * 4] = {0};
    for (uint32_t fmt = 1; fmt < 3; fmt++) {
        struct geist_decknet_frame f = {.format  = (enum geist_decknet_format) fmt,
                                        .width   = ow,
                                        .height  = oh,
                                        .stride  = TINY_IMG_W * 2,
                                        .n_bytes = sizeof big,
                                        .data    = big};
        CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_E_INVALID_ARG);
        n_refused++;
    }
    geist_decknet_stream_destroy(s);
    geist_decknet_destroy(d);
    printf("frame geometry: %zu frames refused with GEIST_E_INVALID_ARG, %zu exact-fit frames "
           "accepted\n",
           n_refused,
           n_accepted);
}

/* ---- state and arguments ---------------------------------------------- */

static void test_state(void) {
    struct geist_decknet *d = load_tiny((uint32_t) TINY_IMG_W, (uint32_t) TINY_IMG_H);
    uint8_t               rgb[TINY_IMG_W * TINY_IMG_H * 3];
    memset(rgb, 120, sizeof rgb);
    const struct geist_decknet_frame f = {.format  = GEIST_DECKNET_FORMAT_RGB8,
                                          .width   = TINY_IMG_W,
                                          .height  = TINY_IMG_H,
                                          .stride  = TINY_IMG_W * 3,
                                          .n_bytes = sizeof rgb,
                                          .data    = rgb};

    struct geist_decknet_stream *s = (struct geist_decknet_stream *) (uintptr_t) 1;
    CHECK(geist_decknet_stream_create(nullptr, &s) == GEIST_E_INVALID_ARG && s == nullptr);
    CHECK(geist_decknet_stream_create(d, nullptr) == GEIST_E_INVALID_ARG);
    geist_decknet_stream_destroy(nullptr);

    /* Each allocation of a create failing once. */
    int64_t n = 0;
    for (;; n++) {
        heap_fail_after(n);
        const enum geist_status st = geist_decknet_stream_create(d, &s);
        heap_fail_after(-1);
        if (st == GEIST_OK) {
            break;
        }
        CHECK(st == GEIST_E_OOM && s == nullptr);
        if (n > 100) {
            CHECK(!"a create never succeeded");
            break;
        }
    }
    CHECK(n >= 9 && s != nullptr); /* the handle, a mark, five pixel lists, tables, the reference */

    /* The same for the first set_gate that turns gating on; a failed one
     * leaves a stream that works, and off needs no memory. */
    CHECK(geist_decknet_stream_set_gate(nullptr, true, 0) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_set_threads(nullptr, 1) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_set_threads(s, 0) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_set_threads(s, 3) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_set_threads(s, 2) == GEIST_OK);
    CHECK(geist_decknet_stream_set_threads(s, 1) == GEIST_OK);
    int64_t n_gate = 0;
    for (;; n_gate++) {
        heap_fail_after(n_gate);
        const enum geist_status off = geist_decknet_stream_set_gate(s, false, 0);
        const enum geist_status st  = geist_decknet_stream_set_gate(s, true, 3);
        heap_fail_after(-1);
        CHECK(off == GEIST_OK);
        if (st == GEIST_OK) {
            break;
        }
        CHECK(st == GEIST_E_OOM);
        if (n_gate > 100) {
            CHECK(!"a set_gate never succeeded");
            break;
        }
    }
    CHECK(n_gate >= 5); /* the kept tiles, the crops, the kernel sums twice, the 1x1 values */

    struct geist_decknet_stream_result res = {.timestamp = 1};
    struct geist_decknet_slot          slot;
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_E_INVALID_STATE && res.timestamp == 0);
    CHECK(geist_decknet_stream_slot(s, 0, &slot) == GEIST_E_INVALID_STATE);
    CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_E_INVALID_STATE); /* no reference */
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_E_INVALID_STATE);

    CHECK(geist_decknet_stream_set_reference(nullptr, &f) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_set_reference(s, nullptr) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_push_frame(nullptr, &f) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_push_frame(s, nullptr) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_result(nullptr, &res) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_result(s, nullptr) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_slot(nullptr, 0, &slot) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_stream_slot(s, 0, nullptr) == GEIST_E_INVALID_ARG);

    CHECK(geist_decknet_stream_set_reference(s, &f) == GEIST_OK);
    CHECK(geist_decknet_stream_push_frame(s, &f) == GEIST_OK);
    CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.n_frames == 1 &&
          res.image.checkable);
    CHECK(geist_decknet_stream_slot(s, 0, &slot) == GEIST_OK);
    CHECK(geist_decknet_stream_slot(s, 2, &slot) == GEIST_E_INVALID_ARG); /* two slots */
    geist_decknet_stream_destroy(s);
    geist_decknet_destroy(d);
    printf("a create survives each of its %lld allocations failing, a set_gate each of its %lld; "
           "state and argument checks\n",
           (long long) n,
           (long long) n_gate);
}

int main(void) {
    test_conversion_reference();
    test_parity(GEIST_DECKNET_FORMAT_RGB8, 0, false, 1, "RGB8");
    test_parity(GEIST_DECKNET_FORMAT_RGB8, 37, false, 1, "RGB8, padded rows");
    test_parity(GEIST_DECKNET_FORMAT_RGB8, 0, true, 1, "RGB8, replaced homographies");
    test_parity(GEIST_DECKNET_FORMAT_NV12, 0, false, 1, "NV12");
    test_parity(GEIST_DECKNET_FORMAT_NV12, 37, true, 1, "NV12, padded, replaced homographies");
    test_parity(GEIST_DECKNET_FORMAT_YUYV, 0, false, 1, "YUYV");
    test_parity(GEIST_DECKNET_FORMAT_YUYV, 37, true, 1, "YUYV, padded, replaced homographies");
    test_parity(GEIST_DECKNET_FORMAT_RGB8, 37, true, 2, "RGB8, two threads");
    test_parity(GEIST_DECKNET_FORMAT_NV12, 0, false, 2, "NV12, two threads");
    test_gate(GEIST_DECKNET_FORMAT_RGB8, 0, 0, 1, "gated, threshold 0, RGB8");
    test_gate(GEIST_DECKNET_FORMAT_NV12, 37, 0, 1, "gated, threshold 0, NV12, padded");
    test_gate(GEIST_DECKNET_FORMAT_YUYV, 0, 0, 1, "gated, threshold 0, YUYV");
    test_gate(GEIST_DECKNET_FORMAT_RGB8, 37, 40, 1, "gated, threshold 40, RGB8, padded");
    test_gate(GEIST_DECKNET_FORMAT_NV12, 0, 40, 1, "gated, threshold 40, NV12");
    test_gate(GEIST_DECKNET_FORMAT_YUYV, 37, 40, 1, "gated, threshold 40, YUYV, padded");
    test_gate(GEIST_DECKNET_FORMAT_YUYV, 0, 0, 2, "gated, 0, YUYV, two threads");
    test_gate(GEIST_DECKNET_FORMAT_RGB8, 0, 40, 2, "gated, 40, RGB8, two threads");
    test_conversion_range(GEIST_DECKNET_FORMAT_NV12);
    test_conversion_range(GEIST_DECKNET_FORMAT_YUYV);
    test_sample();
    test_kernel_sum();
    test_gate_edges(false);
    test_gate_edges(true);
    test_gate_threshold();
    test_gate_each_tile();
    test_gate_seams();
    test_frame_end();
    test_queue(1, false);
    test_queue(2, true);
    test_no_alloc();
    test_geometry();
    test_state();
    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("a decknet stream gives the still path's outputs bit for bit: RGB8 as it is, NV12 and "
           "YUYV as converted by the formula of docs/DECKNET.md; a gated one on the image of its "
           "kept tiles\n");
    return GEIST_TEST_PASS;
}
