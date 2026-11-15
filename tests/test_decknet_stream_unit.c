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
 *   no allocation   after geist_decknet_stream_create: not for the
 *                   reference, not for the first frame, not for any later
 *                   one, in any format (heap.h counts). The count is
 *                   compared with zero, so one allocation fails the test;
 *   geometry        each way a frame can lie about itself is refused with
 *                   GEIST_E_INVALID_ARG and leaves the last result; frames
 *                   that fit their buffer to the byte are accepted (the
 *                   buffers are exact-size, so ASan sees a read behind one);
 *   state           no frame before a reference or after new homographies;
 *                   no result before a frame; each allocation of a create
 *                   failing once.
 */
#include "decknet_tiny.h"
#include "heap.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_decknet.h>

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
static void
test_parity(enum geist_decknet_format format, size_t pad, bool moved_h, const char *what) {
    struct geist_decknet     *still = load_synth();
    struct geist_decknet     *d     = load_synth();
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    const size_t w = info.image_width, h = info.image_height, n_rgb = w * h * 3;
    uint8_t     *cur = read_ppm("tests/data/decknet_synth/cur.ppm", w, h);
    uint8_t     *ref = read_ppm("tests/data/decknet_synth/ref.ppm", w, h);

    struct geist_decknet_stream *s = nullptr;
    CHECK(geist_decknet_stream_create(d, &s) == GEIST_OK && s != nullptr);
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
    const uint64_t before = heap_alloc_count();
    CHECK(before > before_create); /* the counter sees the stream's allocations */

    size_t n_calls = 0;
    for (int round = 0; round < 3; round++) {
        for (uint32_t f = 0; f < 3; f++) {
            struct geist_decknet_stream_result res;
            struct geist_decknet_slot          slot;
            CHECK(geist_decknet_stream_set_reference(s, &refs[f]) == GEIST_OK);
            CHECK(geist_decknet_stream_push_frame(s, &frames[f]) == GEIST_OK);
            CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && res.image.checkable);
            CHECK(geist_decknet_stream_slot(s, 0, &slot) == GEIST_OK);
            /* the other exit of a pass, and a refused frame */
            CHECK(geist_decknet_stream_push_frame(s, &halves[f]) == GEIST_OK);
            CHECK(geist_decknet_stream_result(s, &res) == GEIST_OK && !res.image.checkable);
            struct geist_decknet_frame bad = frames[f];
            bad.n_bytes--;
            CHECK(geist_decknet_stream_push_frame(s, &bad) == GEIST_E_INVALID_ARG);
            n_calls += 7;
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
    printf("no allocation in %zu calls after geist_decknet_stream_create, the first frame "
           "included\n",
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
    printf("a create survives each of its %lld allocations failing; state and argument checks\n",
           (long long) n);
}

int main(void) {
    test_conversion_reference();
    test_parity(GEIST_DECKNET_FORMAT_RGB8, 0, false, "RGB8");
    test_parity(GEIST_DECKNET_FORMAT_RGB8, 37, false, "RGB8, padded rows");
    test_parity(GEIST_DECKNET_FORMAT_RGB8, 0, true, "RGB8, replaced homographies");
    test_parity(GEIST_DECKNET_FORMAT_NV12, 0, false, "NV12");
    test_parity(GEIST_DECKNET_FORMAT_NV12, 37, true, "NV12, padded, replaced homographies");
    test_parity(GEIST_DECKNET_FORMAT_YUYV, 0, false, "YUYV");
    test_parity(GEIST_DECKNET_FORMAT_YUYV, 37, true, "YUYV, padded, replaced homographies");
    test_no_alloc();
    test_geometry();
    test_state();
    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("a decknet stream gives the still path's outputs bit for bit: RGB8 as it is, NV12 and "
           "YUYV "
           "as converted by the formula of docs/DECKNET.md\n");
    return GEIST_TEST_PASS;
}
