/*
 * tools/bench_stream.c — decknet on a camera stream against the still-image
 * path called once per frame (#533).
 *
 * Needs no camera, no file and no model: the frames come from memory (a
 * static scene with a block moving over it; with --scene all every pixel
 * changes as well, which is gating at its worst; with --scene half the
 * upper half of them; with --scene burst half a second of the one and half
 * a second of the other, in turn) and the network is invented
 * here, in the shape of a deck check (six slots, 31 homographies, a
 * 256 x 192 crop, a 12 x 8 raster of units with disc and ring pooling, 9 x 9
 * and 13 x 13 kernels, two region pools over most of the floor crop). It
 * describes no device. How much of a crop a network reads decides what the
 * region of interest saves, so the numbers are this network's; the line
 * "samples per frame" says what it is.
 *
 * Three ways through the same frames, for each size and format:
 *
 *   still   geist_decknet_run per frame. It takes RGB8, so a YUV
 *           frame is first converted whole (the formula of docs/DECKNET.md);
 *           the empty deck is converted once, outside the timing.
 *   stream  geist_decknet_stream_push_frame on the frame as it is.
 *   gated   the same with geist_decknet_stream_set_gate(s, true, 0): tiles
 *           whose bytes did not change keep what was computed from them.
 *
 * Reported per row: frames per second, and the median and 99th percentile
 * of the time one frame takes; for the two streams also what a frame
 * touched: the tiles the network reads, how many of them were taken from
 * the frame, the samples computed, and the frame bytes that stands for.
 * One thread, or with --threads 2 the two streams on two. Every output of
 * every frame is hashed, and the three ways
 * must give the same hash: the bench fails if a stream is not bit-identical
 * to the still path. With --backend metal or vulkan the streams are given
 * that backend (geist_decknet_stream_set_backend), which takes the integer
 * part of a frame that is not gated; the hashes are compared all the same.
 * Which device a Vulkan backend is on is the backend's to say
 * (GEIST_VK_DEVICE, GEIST_VK_VERBOSE=1; docs/BACKENDS.md).
 *
 * Each row also says how much processor time the process used for its
 * timed frames, all threads together (getrusage): what a frame costs in
 * energy as far as a program can tell without privileges. It does not see
 * what a GPU draws.
 *
 * With --camera FPS the frames come at that rate instead, from a second
 * thread, as a camera delivers them: it hands each to
 * geist_decknet_stream_offer, and this thread takes what waits with
 * geist_decknet_stream_push_offered (it looks every 50 microseconds; a
 * program would be woken by its capture loop). Reported per row: how many
 * frames were computed and how many dropped because the pass was not done
 * when the next one came, and the time from the offer of a frame to its
 * result. The still path is not run, and nothing is hashed: which frames
 * are computed depends on the clock. The processor time of such a row
 * includes this thread's looking, which is given next to it (the looks that
 * found nothing, times what one costs, measured once before the rows). With --trace also each pass
 * that took more than a tenth longer than the median one, with what the system did since the pass
 * before: the minor page faults and the involuntary context switches of this process (getrusage),
 * and on Linux the pages the kernel migrated (pgmigrate_success in /proc/vmstat; memory compaction
 * moves the pages of a running process, and a pass that touches one waits for it). That is how
 * docs/DECKNET.md ("The tail behind a camera") tells a slow pass of the library from one the board
 * made slow.
 *
 * With --window each row is followed by a line with the monotonic clock at
 * the first and behind the last of its timed frames and how many they were:
 * what tools/stream_energy.py lays over the power it samples from outside
 * meanwhile, so that no sampling is in this process (docs/DECKNET.md,
 * "Energy").
 *
 * Run it under the host lock like every timing (CONTRIBUTING.md):
 *   make DECKNET=1 bin/<target>/release/tools/bench_stream
 *   bench_stream [--frames N] [--threads 1|2] [--scene block|all|half|burst]
 *                [--only still|stream|gated] [--size 720|1080] [--camera FPS]
 *                [--trace] [--backend cpu|metal|vulkan] [--window]
 *                [--format rgb8|nv12|yuyv|uyvy]
 *                [--colour bt601|bt601-full|bt709|bt709-full]
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1 /* macOS hides the counters of struct rusage under strict POSIX */
#include <geist.h>
#include <geist_decknet.h>

#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

constexpr size_t N_SLOTS  = 6;
constexpr size_t CROP_W   = 256;
constexpr size_t CROP_H   = 192;
constexpr size_t BG_W     = 32;
constexpr size_t BG_H     = 160;
constexpr size_t RASTER_W = 12;
constexpr size_t RASTER_H = 8;
constexpr size_t N_TUBES  = 8;
constexpr size_t RING     = 8;  /* distinct frames, pushed round robin */
constexpr size_t BURST    = 30; /* --scene burst: frames of one kind in a row */
constexpr size_t WARMUP   = 20;

[[noreturn]] static void die(const char *what) {
    fprintf(stderr, "bench_stream: %s\n", what);
    exit(1);
}

static void *must_alloc(size_t n) {
    void *p = malloc(n);
    if (p == nullptr) {
        die("out of memory");
    }
    return p;
}

/* ---- the invented network, as a GGUF in memory ------------------------- */

struct buf {
    uint8_t *b;
    size_t   n, cap;
};

static void put(struct buf *o, const void *p, size_t n) {
    if (o->n + n > o->cap) {
        o->cap = (o->n + n) * 2;
        o->b   = realloc(o->b, o->cap);
        if (o->b == nullptr) {
            die("out of memory");
        }
    }
    memcpy(o->b + o->n, p, n);
    o->n += n;
}
static void put_u32(struct buf *o, uint32_t v) {
    put(o, &v, 4);
}
static void put_u64(struct buf *o, uint64_t v) {
    put(o, &v, 8);
}
static void put_str(struct buf *o, const char *s) {
    put_u64(o, strlen(s));
    put(o, s, strlen(s));
}
static void put_pad(struct buf *o) {
    while (o->n % 32 != 0) {
        put(o, "", 1);
    }
}

struct tensor {
    const char *name;
    uint32_t    n_dims;
    uint64_t    dims[3]; /* GGUF order: innermost first */
    struct buf  v;       /* F32 */
};

static void push_f32(struct tensor *t, float v) {
    put(&t->v, &v, 4);
}

/* Pooling rows for the crop pixels within [r_lo, r_hi) of (cx, cy), in
 * ascending pixel order, equal weights that sum to 1. */
static void
pool_ring(struct tensor *t, size_t unit, double cx, double cy, double r_lo, double r_hi) {
    size_t n = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (size_t y = 0; y < CROP_H; y++) {
            for (size_t x = 0; x < CROP_W; x++) {
                const double r = hypot((double) x - cx, (double) y - cy);
                if (r < r_lo || r >= r_hi) {
                    continue;
                }
                if (pass == 0) {
                    n++;
                } else {
                    push_f32(t, (float) unit);
                    push_f32(t, (float) (y * CROP_W + x));
                    push_f32(t, 1.0f / (float) n);
                }
            }
        }
    }
    t->dims[1] += n;
}

/* One unit over the rectangle [x0, x1) x [y0, y1). */
static void pool_rect(struct tensor *t, size_t x0, size_t y0, size_t x1, size_t y1) {
    const size_t n = (x1 - x0) * (y1 - y0);
    for (size_t y = y0; y < y1; y++) {
        for (size_t x = x0; x < x1; x++) {
            push_f32(t, 0.0f);
            push_f32(t, (float) (y * CROP_W + x));
            push_f32(t, 1.0f / (float) n);
        }
    }
    t->dims[1] = n;
}

/* A point-symmetric side x side kernel with sum 0: +1 inside r_in, a
 * negative rim that balances it. */
static void kernel(struct tensor *t, size_t side, double r_in) {
    const double mid   = (double) (side / 2);
    size_t       n_in  = 0;
    size_t       n_out = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (size_t a = 0; a < side; a++) {
            for (size_t b = 0; b < side; b++) {
                const bool in = hypot((double) a - mid, (double) b - mid) <= r_in;
                if (pass == 0) {
                    n_in += in;
                    n_out += !in;
                } else {
                    push_f32(t, in ? 1.0f / (float) n_in : -1.0f / (float) n_out);
                }
            }
        }
    }
    t->dims[0] = t->dims[1] = side;
}

/* The file for an img_w x img_h camera image. The crops are a fixed number
 * of pixels; the homographies spread them over whatever the image is. */
static struct buf network(size_t img_w, size_t img_h) {
    const double w = (double) img_w, h = (double) img_h;
    enum {
        T_H,
        T_PLANES,
        T_NOMINAL,
        T_BLOB,
        T_TUBE,
        T_CHANGE,
        T_WALL,
        T_ON,
        T_OFF,
        T_PTUBE,
        T_PWALL,
        T_FRONT,
        T_REAR,
        N_T
    };
    struct tensor t[N_T] = {
            [T_H]       = {"decknet.homography", 3, {3, 3, 1 + 5 * N_SLOTS}, {0}},
            [T_PLANES]  = {"decknet.planes_mm", 1, {5}, {0}},
            [T_NOMINAL] = {"decknet.bg_nominal", 1, {3}, {0}},
            [T_BLOB]    = {"decknet.conv.blob", 2, {0}, {0}},
            [T_TUBE]    = {"decknet.conv.tube", 2, {0}, {0}},
            [T_CHANGE]  = {"decknet.change.w", 2, {6, 4}, {0}},
            [T_WALL]    = {"decknet.wall.w", 2, {6, 3}, {0}},
            [T_ON]      = {"decknet.pool.on", 2, {3, 0}, {0}},
            [T_OFF]     = {"decknet.pool.off", 2, {3, 0}, {0}},
            [T_PTUBE]   = {"decknet.pool.tube", 2, {3, 0}, {0}},
            [T_PWALL]   = {"decknet.pool.wall", 2, {3, 0}, {0}},
            [T_FRONT]   = {"decknet.pool.occ_front", 2, {3, 0}, {0}},
            [T_REAR]    = {"decknet.pool.occ_rear", 2, {3, 0}, {0}},
    };

    /* [0]: the worktable strip, down the left edge. Then per slot (three
     * columns, two rows) five planes, each a little off the one below, with
     * some shear and perspective so that no sample falls on a pixel centre. */
    const double strip[9] = {h / 720.0, 0.0, 0.015 * w, 0.0, h / 720.0, 0.2 * h, 0.0, 0.0, 1.0};
    for (size_t i = 0; i < 9; i++) {
        push_f32(&t[T_H], (float) strip[i]);
    }
    for (size_t s = 0; s < N_SLOTS; s++) {
        const double col = (double) (s % 3), row = (double) (s / 3);
        for (size_t p = 0; p < 5; p++) {
            const double grow = 1.0 + 0.012 * (double) p;
            const double m[9] = {
                    grow * 0.27 * w / (double) CROP_W,
                    0.013 * (col - 1.0),
                    (0.10 + 0.295 * col) * w - 0.004 * w * (double) p,
                    0.011 * (row - 0.5),
                    grow * 0.42 * h / (double) CROP_H,
                    (0.04 + 0.48 * row) * h - 0.006 * h * (double) p,
                    2.0e-5 * (col - 1.0),
                    3.0e-5 * (0.5 - row),
                    1.0,
            };
            for (size_t i = 0; i < 9; i++) {
                push_f32(&t[T_H], (float) m[i]);
            }
        }
    }
    const float planes[5] = {0.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    put(&t[T_PLANES].v, planes, sizeof planes);
    const float nominal[3] = {0.5f, 0.5f, 0.5f};
    put(&t[T_NOMINAL].v, nominal, sizeof nominal);
    kernel(&t[T_BLOB], 9, 2.0);
    kernel(&t[T_TUBE], 13, 3.0);
    const float change[24] = {0.9f,  1.1f,    0.7f,   -0.9f, -1.1f, -0.7f,  -0.9f,  -1.1f,
                              -0.7f, 0.9f,    1.1f,   0.7f,  0.02f, -0.01f, 0.015f, -0.02f,
                              0.01f, -0.015f, -0.01f, 0.02f, 0.01f, 0.01f,  -0.02f, -0.01f};
    put(&t[T_CHANGE].v, change, sizeof change);
    const float wall[18] = {-1.0f,
                            0.2f,
                            0.8f,
                            0.5f,
                            -0.1f,
                            -0.4f,
                            0.3f,
                            -1.0f,
                            0.7f,
                            -0.2f,
                            0.5f,
                            -0.3f,
                            0.6f,
                            0.4f,
                            -1.0f,
                            -0.3f,
                            -0.2f,
                            0.5f};
    put(&t[T_WALL].v, wall, sizeof wall);

    for (size_t u = 0; u < RASTER_W * RASTER_H; u++) {
        const double cx = ((double) (u % RASTER_W) + 0.5) * (double) CROP_W / (double) RASTER_W;
        const double cy = ((double) (u / RASTER_W) + 0.5) * (double) CROP_H / (double) RASTER_H;
        pool_ring(&t[T_ON], u, cx, cy, 0.0, 2.1);
        pool_ring(&t[T_OFF], u, cx, cy, 3.6, 4.5);
    }
    for (size_t u = 0; u < 2 * N_TUBES; u++) {
        const double along = ((double) (u % N_TUBES) + 0.5) * (double) CROP_W / (double) N_TUBES;
        const double cx    = u < N_TUBES ? along : (double) CROP_W - along + 5.0;
        pool_ring(&t[T_PTUBE], u, cx, u < N_TUBES ? 70.0 : 122.0, 0.0, 3.1);
    }
    pool_rect(&t[T_PWALL], 0, 4, CROP_W, 10);
    pool_rect(&t[T_FRONT], 16, 100, 240, 180);
    pool_rect(&t[T_REAR], 16, 12, 240, 92);

    const struct {
        const char *key;
        float       v;
    } scalars[] = {
            {"decknet.t_change", 0.2f},
            {"decknet.g_change", 3.0f},
            {"decknet.t_wall", 0.3f},
            {"decknet.g_wall", 4.0f},
            {"decknet.t_tip", 0.02f},
            {"decknet.t_dark", 0.02f},
            {"decknet.t_tube", 0.02f},
            {"decknet.g_unit", 8.0f},
            {"decknet.t_white", 0.3f},
            {"decknet.g_white", 6.0f},
            {"decknet.gain_min", 0.7f},
            {"decknet.gain_max", 1.4f},
            {"decknet.bg_rest_max", 0.05f},
            {"decknet.occ", 0.4f},
            {"decknet.smooth", 0.45f},
    };
    const struct {
        const char *key;
        uint32_t    v;
    } sizes[] = {
            {"decknet.image.width", (uint32_t) img_w},
            {"decknet.image.height", (uint32_t) img_h},
            {"decknet.crop.width", (uint32_t) CROP_W},
            {"decknet.crop.height", (uint32_t) CROP_H},
            {"decknet.bg.width", (uint32_t) BG_W},
            {"decknet.bg.height", (uint32_t) BG_H},
    };
    constexpr size_t n_scalars = sizeof scalars / sizeof scalars[0];
    constexpr size_t n_sizes   = sizeof sizes / sizeof sizes[0];

    struct buf o = {0};
    put(&o, "GGUF", 4);
    put_u32(&o, 3);
    put_u64(&o, N_T);
    put_u64(&o, 2 + n_sizes + n_scalars);
    put_str(&o, "general.architecture");
    put_u32(&o, 8);
    put_str(&o, "decknet");
    put_str(&o, "decknet.slots");
    put_u32(&o, 8);
    put_str(&o, "ABCDEF");
    for (size_t i = 0; i < n_sizes; i++) {
        put_str(&o, sizes[i].key);
        put_u32(&o, 4);
        put_u32(&o, sizes[i].v);
    }
    for (size_t i = 0; i < n_scalars; i++) {
        put_str(&o, scalars[i].key);
        put_u32(&o, 6);
        put(&o, &scalars[i].v, 4);
    }
    uint64_t offset = 0;
    for (size_t i = 0; i < N_T; i++) {
        put_str(&o, t[i].name);
        put_u32(&o, t[i].n_dims);
        for (uint32_t k = 0; k < t[i].n_dims; k++) {
            put_u64(&o, t[i].dims[k]);
        }
        put_u32(&o, 0); /* F32 */
        put_u64(&o, offset);
        offset += (t[i].v.n + 31) / 32 * 32;
    }
    put_pad(&o);
    for (size_t i = 0; i < N_T; i++) {
        put(&o, t[i].v.b, t[i].v.n);
        put_pad(&o);
        free(t[i].v.b);
    }
    return o;
}

/* ---- frames ----------------------------------------------------------- */

static uint8_t byte_of(double v) {
    return v < 0.0 ? 0 : (v > 255.0 ? 255 : (uint8_t) v);
}

/* The static scene, RGB8 without padding: a gradient, dots on a grid and
 * noise, around mid grey so that the brightness check passes. */
static void still_scene(size_t w, size_t h, uint8_t *rgb) {
    uint32_t lcg = 533;
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            lcg                = lcg * 1664525u + 1013904223u;
            const double noise = (double) (lcg >> 24) / 255.0 * 24.0 - 12.0;
            const bool   dot   = (x / 6 + y / 6) % 5 == 0 && x % 6 < 4 && y % 6 < 4;
            const double base =
                    128.0 + 30.0 * (double) x / (double) w - 20.0 * (double) y / (double) h;
            uint8_t *p = rgb + (y * w + x) * 3;
            p[0]       = byte_of(base + noise + (dot ? 70.0 : 0.0));
            p[1]       = byte_of(base - noise * 0.5 + (dot ? -60.0 : 0.0));
            p[2]       = byte_of(base + noise * 0.25 + (dot ? 40.0 : -6.0));
        }
    }
}

/* What moves in the frames. */
enum scene : uint32_t {
    SCENE_BLOCK = 0, /* a block over a scene that stands still */
    SCENE_ALL   = 1, /* and every pixel */
    SCENE_HALF  = 2, /* and every pixel of the upper half */
    SCENE_BURST = 3, /* BURST frames of SCENE_ALL, BURST of SCENE_BLOCK, in turn */
};
static_assert(sizeof(enum scene) == sizeof(uint32_t));

/* Frame k of the ring: the scene with a checkered block somewhere on the
 * slots, never on the worktable strip. With SCENE_ALL every byte of the
 * scene is also k brighter, so that no tile of a frame is a tile of the one
 * before: what gating costs when it saves nothing. With SCENE_HALF the
 * bytes of the upper half are. */
static void
moving(size_t w, size_t h, size_t k, enum scene scene, const uint8_t *still, uint8_t *rgb) {
    memcpy(rgb, still, w * h * 3);
    const size_t n_brighter =
            scene == SCENE_ALL ? w * h * 3 : (scene == SCENE_HALF ? w * (h / 2) * 3 : 0);
    for (size_t i = 0; i < n_brighter; i++) {
        rgb[i] = byte_of((double) rgb[i] + (double) k);
    }
    const size_t side = h / 6;
    const size_t x0   = w / 8 + k * (w - w / 8 - side) / RING;
    const size_t y0   = h / 10 + k * (h - h / 10 - side) / RING;
    for (size_t y = y0; y < y0 + side; y++) {
        for (size_t x = x0; x < x0 + side; x++) {
            const uint8_t v = ((x - x0) / 8 + (y - y0) / 8) % 2 == 0 ? 235 : 20;
            uint8_t      *p = rgb + (y * w + x) * 3;
            p[0]            = v;
            p[1]            = (uint8_t) (255 - v);
            p[2]            = v;
        }
    }
}

static void rgb_yuv(const uint8_t *p, uint8_t yuv[static 3]) {
    const double r = p[0], g = p[1], b = p[2];
    yuv[0] = byte_of(16.5 + 0.257 * r + 0.504 * g + 0.098 * b);
    yuv[1] = byte_of(128.5 - 0.148 * r - 0.291 * g + 0.439 * b);
    yuv[2] = byte_of(128.5 + 0.439 * r - 0.368 * g - 0.291 * b);
}

/* The bytes a w x h frame of `format` takes with rows of `stride`. */
static size_t frame_bytes(enum geist_decknet_format format, size_t h, size_t stride) {
    return format == GEIST_DECKNET_FORMAT_NV12 ? stride * (h + h / 2) : stride * h;
}

/* An RGB8 image as a camera would deliver it in `format`: the chroma of a
 * block is that of its first pixel. */
static void to_format(enum geist_decknet_format format,
                      size_t                    w,
                      size_t                    h,
                      size_t                    stride,
                      const uint8_t            *rgb,
                      uint8_t                  *out) {
    memset(out, 0, frame_bytes(format, h, stride));
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            const uint8_t *p = rgb + (y * w + x) * 3;
            uint8_t        yuv[3];
            rgb_yuv(p, yuv);
            switch (format) {
            case GEIST_DECKNET_FORMAT_RGB8:
                memcpy(out + y * stride + x * 3, p, 3);
                break;
            case GEIST_DECKNET_FORMAT_NV12:
                out[y * stride + x] = yuv[0];
                if (x % 2 == 0 && y % 2 == 0) {
                    out[(h + y / 2) * stride + x]     = yuv[1];
                    out[(h + y / 2) * stride + x + 1] = yuv[2];
                }
                break;
            case GEIST_DECKNET_FORMAT_YUYV:
                out[y * stride + x * 2] = yuv[0];
                if (x % 2 == 0) {
                    out[y * stride + x * 2 + 1] = yuv[1];
                    out[y * stride + x * 2 + 3] = yuv[2];
                }
                break;
            case GEIST_DECKNET_FORMAT_UYVY:
                out[y * stride + x * 2 + 1] = yuv[0];
                if (x % 2 == 0) {
                    out[y * stride + x * 2]     = yuv[1];
                    out[y * stride + x * 2 + 2] = yuv[2];
                }
                break;
            }
        }
    }
}

/* --colour: what the YUV frames are said to be, and are converted as for
 * the still path. Ky, Yo, Krv, Kgu, Kgv, Kbu of docs/DECKNET.md. */
static enum geist_decknet_colour g_colour;
static const char *const         colour_names[4] = {"bt601", "bt601-full", "bt709", "bt709-full"};
static const int32_t             colour_k[4][6]  = {{298, 16, 409, 100, 208, 516},
                                                    {256, 0, 359, 88, 183, 454},
                                                    {298, 16, 459, 55, 136, 541},
                                                    {256, 0, 403, 48, 120, 475}};

static uint8_t clip8(int32_t v) {
    return (uint8_t) (v < 0 ? 0 : (v > 255 * 256 ? 255 : v / 256));
}

/* What the still path needs first: the whole frame as RGB8 without padding,
 * by the formula of docs/DECKNET.md. */
static void to_rgb8(enum geist_decknet_format format,
                    size_t                    w,
                    size_t                    h,
                    size_t                    stride,
                    const uint8_t            *in,
                    uint8_t                  *rgb) {
    for (size_t y = 0; y < h; y++) {
        const uint8_t *row = in + y * stride;
        const uint8_t *uv  = in + (h + y / 2) * stride; /* NV12 */
        uint8_t       *o   = rgb + y * w * 3;
        for (size_t x = 0; x < w; x++, o += 3) {
            const size_t even = x & ~(size_t) 1;
            int32_t      luma, u, v;
            if (format == GEIST_DECKNET_FORMAT_NV12) {
                luma = row[x];
                u    = uv[even];
                v    = uv[even + 1];
            } else {
                const size_t c = format == GEIST_DECKNET_FORMAT_UYVY ? 0 : 1; /* the U */
                luma           = row[x * 2 + 1 - c];
                u              = row[even * 2 + c];
                v              = row[even * 2 + c + 2];
            }
            const int32_t *k = colour_k[g_colour];
            const int32_t  c = k[0] * (luma - k[1]) + 128;
            o[0]             = clip8(c + k[2] * (v - 128));
            o[1]             = clip8(c - k[3] * (u - 128) - k[4] * (v - 128));
            o[2]             = clip8(c + k[5] * (u - 128));
        }
    }
}

/* ---- measuring -------------------------------------------------------- */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec * 1e-6;
}

static int by_value(const void *a, const void *b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

/* FNV-1a over the bits of every output of the last pass. */
static uint64_t hash_bytes(uint64_t hash, size_t n, const void *p) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) {
        hash = (hash ^ b[i]) * 0x100000001B3ull;
    }
    return hash;
}
static uint64_t hash_outputs(uint64_t                           hash,
                             const struct geist_decknet_info   *info,
                             const struct geist_decknet_result *res,
                             const struct geist_decknet        *d,
                             const struct geist_decknet_stream *s) {
    if (!res->checkable) {
        die("the bench frames are not checkable: no slot was evaluated");
    }
    hash = hash_bytes(hash, sizeof res->gain, res->gain);
    hash = hash_bytes(hash, sizeof res->bg_changed, &res->bg_changed);
    for (size_t i = 0; i < info->n_slots; i++) {
        struct geist_decknet_slot slot;
        const enum geist_status   st = s != nullptr ? geist_decknet_stream_slot(s, i, &slot)
                                                    : geist_decknet_slot(d, i, &slot);
        if (st != GEIST_OK) {
            die("a slot could not be read");
        }
        const float scalars[8] = {slot.occ_front,
                                  slot.occ_rear,
                                  slot.wall_front,
                                  slot.rack,
                                  slot.plate,
                                  slot.deep,
                                  slot.tube_n,
                                  slot.tube_r};
        hash                   = hash_bytes(hash, sizeof scalars, scalars);
        hash                   = hash_bytes(hash, info->n_tips * sizeof(float), slot.tips);
        hash                   = hash_bytes(hash, info->n_tubes * sizeof(float), slot.tubes);
        hash                   = hash_bytes(hash, info->n_tubes * sizeof(float), slot.tubes_rot);
    }
    return hash;
}

struct timing {
    double   fps, p50, p99;
    uint64_t hash;
};

/* The processor time this process has used, all threads, in seconds. */
static double cpu_seconds(void) {
    struct rusage ru = {0};
    getrusage(RUSAGE_SELF, &ru);
    return (double) ru.ru_utime.tv_sec + (double) ru.ru_stime.tv_sec +
           ((double) ru.ru_utime.tv_usec + (double) ru.ru_stime.tv_usec) * 1e-6;
}

/* --window: when a row's timed frames ran, on the clock a sampler outside
 * this process reads as well (CLOCK_MONOTONIC, in seconds). */
static void report_window(const char *way,
                          size_t      w,
                          size_t      h,
                          const char *format,
                          size_t      n,
                          double      t0_ms,
                          double      t1_ms) {
    printf("%4zux%-4zu  %-4s  %-6s  window %.6f to %.6f s on the monotonic clock, %zu frames\n",
           w,
           h,
           format,
           way,
           t0_ms * 1e-3,
           t1_ms * 1e-3,
           n);
    fflush(stdout);
}

static void report(const char    *way,
                   size_t         w,
                   size_t         h,
                   const char    *format,
                   size_t         n,
                   double         cpu_s,
                   double        *ms,
                   struct timing *t) {
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        total += ms[i];
    }
    qsort(ms, n, sizeof ms[0], by_value);
    t->fps = (double) n * 1e3 / total;
    t->p50 = ms[n / 2];
    t->p99 = ms[n * 99 / 100];
    printf("%4zux%-4zu  %-4s  %-6s  %8.1f frames/s   p50 %7.3f ms   p99 %7.3f ms   cpu %6.3f s\n",
           w,
           h,
           format,
           way,
           t->fps,
           t->p50,
           t->p99,
           cpu_s);
    fflush(stdout);
}

/* ---- a camera --------------------------------------------------------- */

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000u + (uint64_t) ts.tv_nsec;
}

/* How long the thread that computes sleeps when no frame waits. */
constexpr long LOOK_NS = 50000;

/* The processor time one such look costs, in seconds: a sleep and a call
 * that finds nothing, measured over a few thousand of them. */
static double look_seconds(void) {
    constexpr size_t n  = 4000;
    const double     t0 = cpu_seconds();
    for (size_t i = 0; i < n; i++) {
        nanosleep(&(struct timespec) {.tv_nsec = LOOK_NS}, nullptr);
    }
    return (cpu_seconds() - t0) / (double) n;
}

static void sleep_until(uint64_t t_ns) {
    const uint64_t now = now_ns();
    if (t_ns > now) {
        nanosleep(&(struct timespec) {.tv_sec  = (time_t) ((t_ns - now) / 1000000000u),
                                      .tv_nsec = (long) ((t_ns - now) % 1000000000u)},
                  nullptr);
    }
}

/* The thread that captures: frame i is offered at i periods from the start,
 * with the time of the offer as its timestamp. It never waits for the pass;
 * a frame it gets back was dropped, and is the camera's to fill again. */
struct camera {
    struct geist_decknet_stream *s;
    size_t                       n;
    struct geist_decknet_frame  *frames; /* n: a frame is lent until it comes back */
    double                       period_ns;
    atomic_bool                  done;
};

static void *camera_run(void *arg) {
    struct camera *c  = arg;
    const uint64_t t0 = now_ns();
    for (size_t i = 0; i < c->n; i++) {
        sleep_until(t0 + (uint64_t) ((double) (i + 1) * c->period_ns));
        const struct geist_decknet_frame *dropped = nullptr;
        c->frames[i].timestamp                    = now_ns();
        if (geist_decknet_stream_offer(c->s, &c->frames[i], &dropped) != GEIST_OK) {
            die("geist_decknet_stream_offer");
        }
    }
    atomic_store(&c->done, true);
    return nullptr;
}

/* --trace: one pass behind the camera, and the counters when it was done. */
struct pass {
    size_t   frame;
    double   at_s, ms, waited_ms; /* offered at; the pass; offer to the start of the pass */
    long     minor_faults, switches;
    uint64_t migrated; /* UINT64_MAX: the system does not say */
};

/* The pages the kernel has migrated since boot, where /proc/vmstat says. */
static uint64_t pages_migrated(void) {
    static const char key[] = "\npgmigrate_success ";
    static int        fd    = -2;
    static char       text[16384];
    if (fd == -2) {
        fd = open("/proc/vmstat", O_RDONLY);
    }
    const ssize_t n = fd < 0 ? -1 : pread(fd, text, sizeof text - 1, 0);
    if (n <= 0) {
        return UINT64_MAX;
    }
    text[n]        = '\0';
    const char *at = strstr(text, key);
    return at == nullptr ? UINT64_MAX : strtoull(at + sizeof key - 1, nullptr, 10);
}

static void pass_counters(struct pass *p) {
    struct rusage ru = {0};
    getrusage(RUSAGE_SELF, &ru);
    p->minor_faults = ru.ru_minflt;
    p->switches     = ru.ru_nivcsw;
    p->migrated     = pages_migrated();
}

/* The passes that took more than a tenth longer than the median one, each
 * with what happened since the pass before it. */
static void report_slow(size_t n, const struct pass *pass, double *scratch) {
    constexpr size_t max_lines = 24;
    for (size_t i = 0; i < n; i++) {
        scratch[i] = pass[i].ms;
    }
    qsort(scratch, n, sizeof scratch[0], by_value);
    const double median = scratch[n / 2];
    size_t       n_slow = 0, n_migrated = 0, n_faults = 0;
    for (size_t i = 1; i < n; i++) {
        const struct pass *p = &pass[i], *q = &pass[i - 1];
        if (p->ms <= median * 1.1) {
            continue;
        }
        const bool     known    = p->migrated != UINT64_MAX && q->migrated != UINT64_MAX;
        const uint64_t migrated = known ? p->migrated - q->migrated : 0;
        n_slow++;
        n_migrated += migrated > 0;
        n_faults += p->minor_faults > q->minor_faults;
        if (n_slow > max_lines) {
            continue;
        }
        printf("    frame %4zu at %6.3f s: pass %6.2f ms after %5.2f ms in the queue, %ld minor "
               "faults, %ld involuntary switches",
               p->frame,
               p->at_s,
               p->ms,
               p->waited_ms,
               p->minor_faults - q->minor_faults,
               p->switches - q->switches);
        if (known) {
            printf(", %llu pages migrated", (unsigned long long) migrated);
        }
        printf("\n");
    }
    printf("    %zu of %zu passes took more than 1.1 x the median %.2f ms%s: %zu while the "
           "kernel migrated pages, %zu with minor faults of this process\n",
           n_slow,
           n,
           median,
           n_slow > max_lines ? " (the first 24 above)" : "",
           n_migrated,
           n_faults);
    fflush(stdout);
}

static const char *const format_names[4] = {"rgb8", "nv12", "yuyv", "uyvy"};

/* Which frame of the two rings is pushed i-th. */
static size_t frame_of(enum scene scene, size_t i) {
    return i % RING + (scene == SCENE_BURST && i / BURST % 2 == 1 ? RING : 0);
}

/* One size and format, both ways. False when the two disagree. */
static bool measure(size_t                    w,
                    size_t                    h,
                    enum geist_decknet_format format,
                    size_t                    n_frames,
                    size_t                    threads,
                    enum scene                scene,
                    double                    camera_fps,
                    bool                      trace,
                    bool                      window,
                    const char               *only,
                    struct geist_backend     *be,
                    double                    look_s) {
    struct buf            file = network(w, h);
    struct geist_decknet *d    = nullptr;
    if (geist_decknet_load_from_memory(file.n, file.b, &d) != GEIST_OK) {
        die(geist_last_create_error());
    }
    free(file.b);
    struct geist_decknet_info info;
    if (geist_decknet_info(d, &info) != GEIST_OK) {
        die("geist_decknet_info");
    }

    /* Rows padded to 64 bytes, as capture buffers are. */
    const size_t row    = format == GEIST_DECKNET_FORMAT_RGB8   ? w * 3
                          : format == GEIST_DECKNET_FORMAT_NV12 ? w
                                                                : w * 2;
    const size_t stride = (row + 63) / 64 * 64;
    const size_t n_cam  = frame_bytes(format, h, stride);
    const size_t n_rgb  = w * h * 3;
    uint8_t     *still  = must_alloc(n_rgb);
    uint8_t     *tmp    = must_alloc(n_rgb);
    uint8_t     *ref    = must_alloc(n_cam);
    uint8_t     *ref8   = must_alloc(n_rgb); /* the empty deck as the still path takes it */
    uint8_t     *cur8   = must_alloc(n_rgb);
    /* Two rings for SCENE_BURST, which pushes from one and then the other. */
    uint8_t *ring[2 * RING];
    still_scene(w, h, still);
    to_format(format, w, h, stride, still, ref);
    for (size_t k = 0; k < 2 * RING; k++) {
        const enum scene of_ring =
                scene != SCENE_BURST ? scene : (k < RING ? SCENE_ALL : SCENE_BLOCK);
        ring[k] = must_alloc(n_cam);
        moving(w, h, k % RING, of_ring, still, tmp);
        to_format(format, w, h, stride, tmp, ring[k]);
    }
    if (format == GEIST_DECKNET_FORMAT_RGB8) {
        memcpy(ref8, still, n_rgb);
    } else {
        to_rgb8(format, w, h, stride, ref, ref8);
    }

    double       *ms      = must_alloc(n_frames * sizeof(double));
    struct timing t_still = {0};
    const char   *name    = format_names[format];

    if (camera_fps <= 0.0 && (only == nullptr || strcmp(only, "still") == 0)) {
        double cpu0 = 0.0, first = 0.0;
        for (size_t i = 0; i < WARMUP + n_frames; i++) {
            cpu0                            = i == WARMUP ? cpu_seconds() : cpu0;
            const uint8_t              *cam = ring[frame_of(scene, i)];
            struct geist_decknet_result res;
            const double                t0 = now_ms();
            first                          = i == WARMUP ? t0 : first;
            const uint8_t *in              = cam;
            if (format != GEIST_DECKNET_FORMAT_RGB8) {
                to_rgb8(format, w, h, stride, cam, cur8);
                in = cur8;
            } else if (stride != w * 3) {
                for (size_t y = 0; y < h; y++) { /* the still path takes no padding */
                    memcpy(cur8 + y * w * 3, cam + y * stride, w * 3);
                }
                in = cur8;
            }
            if (geist_decknet_run(d, n_rgb, in, ref8, &res) != GEIST_OK) {
                die("geist_decknet_run");
            }
            const double t1 = now_ms();
            if (i >= WARMUP) {
                ms[i - WARMUP] = t1 - t0;
                t_still.hash   = hash_outputs(t_still.hash, &info, &res, d, nullptr);
            }
        }
        const double last = now_ms();
        report("still", w, h, name, n_frames, cpu_seconds() - cpu0, ms, &t_still);
        if (window) {
            report_window("still", w, h, name, n_frames, first, last);
        }
    }
    /* Twice: without gating, then gated at threshold 0. */
    struct timing t_way[2] = {};
    for (size_t gated = 0; gated < 2; gated++) {
        const char *way = gated ? "gated" : "stream";
        if (only != nullptr && strcmp(only, way) != 0) {
            continue;
        }
        struct geist_decknet_stream *s = nullptr;
        if (geist_decknet_stream_create(d, &s) != GEIST_OK ||
            geist_decknet_stream_set_gate(s, gated != 0, 0) != GEIST_OK ||
            geist_decknet_stream_set_threads(s, threads) != GEIST_OK) {
            die("geist_decknet_stream_create");
        }
        if (be != nullptr && geist_decknet_stream_set_backend(s, be) != GEIST_OK) {
            die(geist_backend_errmsg(be));
        }
        struct geist_decknet_frame f = {.format  = format,
                                        .colour  = g_colour,
                                        .width   = w,
                                        .height  = h,
                                        .stride  = stride,
                                        .n_bytes = n_cam,
                                        .data    = ref};
        if (geist_decknet_stream_set_reference(s, &f) != GEIST_OK) {
            die("geist_decknet_stream_set_reference");
        }
        if (camera_fps > 0.0) {
            struct camera c = {.s = s, .n = WARMUP + n_frames, .period_ns = 1e9 / camera_fps};
            c.frames        = must_alloc(c.n * sizeof c.frames[0]);
            for (size_t i = 0; i < c.n; i++) {
                c.frames[i]      = f;
                c.frames[i].data = ring[frame_of(scene, i)];
            }
            pthread_t thread;
            if (pthread_create(&thread, nullptr, camera_run, &c) != 0) {
                die("pthread_create");
            }
            size_t       n_computed = 0, n_late = 0, n_looks = 0;
            double       worst = 0.0, cpu0 = 0.0;
            struct pass *pass    = trace ? must_alloc(c.n * sizeof pass[0]) : nullptr;
            size_t       n_pass  = 0;
            uint64_t     t_first = 0, t_from = 0, t_to = 0;
            for (;;) {
                const bool                        done = atomic_load(&c.done);
                const struct geist_decknet_frame *got  = nullptr;
                const uint64_t                    t0   = now_ns();
                if (geist_decknet_stream_push_offered(s, &got) != GEIST_OK) {
                    die("geist_decknet_stream_push_offered");
                }
                if (got == nullptr) {
                    if (done) {
                        break;
                    }
                    nanosleep(&(struct timespec) {.tv_nsec = LOOK_NS}, nullptr);
                    n_looks += n_computed > 0;
                    continue;
                }
                const uint64_t t1      = now_ns();
                const double   latency = (double) (t1 - got->timestamp) * 1e-6;
                if (got >= c.frames + WARMUP) {
                    /* The first timed frame's own pass is not in the
                     * processor time; the ones after it are. */
                    cpu0             = n_computed == 0 ? cpu_seconds() : cpu0;
                    t_from           = n_computed == 0 ? got->timestamp : t_from;
                    t_to             = t1;
                    ms[n_computed++] = latency;
                    n_late += latency > 1e3 / camera_fps;
                    worst = latency > worst ? latency : worst;
                    if (trace) {
                        t_first = n_pass == 0 ? got->timestamp : t_first;
                        pass[n_pass] =
                                (struct pass) {.frame = (size_t) (got - c.frames),
                                               .at_s  = (double) (got->timestamp - t_first) * 1e-9,
                                               .ms    = (double) (t1 - t0) * 1e-6,
                                               .waited_ms = (double) (t0 - got->timestamp) * 1e-6};
                        pass_counters(&pass[n_pass++]);
                    }
                }
            }
            const double cpu_s = cpu_seconds() - cpu0;
            pthread_join(thread, nullptr);
            struct geist_decknet_stream_result res;
            if (geist_decknet_stream_result(s, &res) != GEIST_OK || n_computed == 0) {
                die("geist_decknet_stream_result");
            }
            qsort(ms, n_computed, sizeof ms[0], by_value);
            printf("%4zux%-4zu  %-4s  %-6s  camera at %.1f frames/s: %zu frames, %llu dropped "
                   "(warm-up included), offer to result p50 %7.3f ms   p99 %7.3f ms   worst "
                   "%7.3f ms, %zu later than a frame period; cpu %.3f s for %zu passes, about "
                   "%.3f s of it the bench looking for a frame\n",
                   w,
                   h,
                   name,
                   way,
                   camera_fps,
                   c.n,
                   (unsigned long long) res.n_dropped,
                   ms[n_computed / 2],
                   ms[n_computed * 99 / 100],
                   worst,
                   n_late,
                   cpu_s,
                   n_computed - 1,
                   (double) n_looks * look_s);
            fflush(stdout);
            if (window) {
                report_window(
                        way, w, h, name, n_computed, (double) t_from * 1e-6, (double) t_to * 1e-6);
            }
            if (trace) {
                report_slow(n_pass, pass, ms);
                free(pass);
            }
            free(c.frames);
            geist_decknet_stream_destroy(s);
            continue;
        }
        double tiles = 0.0, changed = 0.0, samples = 0.0, cpu0 = 0.0, cpu_s = 0.0, first = 0.0;
        for (size_t i = 0; i < WARMUP + n_frames; i++) {
            /* The hashing below is the bench's, and not in the time. */
            cpu0            = cpu_seconds();
            f.data          = ring[frame_of(scene, i)];
            f.timestamp     = i;
            const double t0 = now_ms();
            first           = i == WARMUP ? t0 : first;
            if (geist_decknet_stream_push_frame(s, &f) != GEIST_OK) {
                die("geist_decknet_stream_push_frame");
            }
            const double t1 = now_ms();
            if (i >= WARMUP) {
                cpu_s += cpu_seconds() - cpu0;
                struct geist_decknet_stream_result res;
                if (geist_decknet_stream_result(s, &res) != GEIST_OK || res.timestamp != i) {
                    die("geist_decknet_stream_result");
                }
                ms[i - WARMUP]    = t1 - t0;
                t_way[gated].hash = hash_outputs(t_way[gated].hash, &info, &res.image, d, s);
                tiles             = (double) res.n_tiles;
                changed += (double) res.n_tiles_changed;
                samples += (double) res.n_samples;
            }
        }
        const double last = now_ms();
        report(way, w, h, name, n_frames, cpu_s, ms, &t_way[gated]);
        if (window) {
            /* The bench's hashing between the frames is in this window. */
            report_window(way, w, h, name, n_frames, first, last);
        }
        /* A tile is GEIST_DECKNET_TILE squared pixels (less at the frame's
         * edge, which is not counted here). Without gating the samples read
         * a part of each tile; gated, every tile is compared. */
        const double tile_bytes =
                (double) (GEIST_DECKNET_TILE * GEIST_DECKNET_TILE) * (double) row / (double) w;
        changed /= (double) n_frames;
        printf("%4zux%-4zu  %-4s  %-6s  per frame: %.0f of %.0f tiles taken (%.1f %%), %.0f "
               "samples, "
               "%s %.2f MB of the frame's %.2f MB\n",
               w,
               h,
               name,
               way,
               changed,
               tiles,
               100.0 * changed / tiles,
               samples / (double) n_frames,
               gated ? "compared" : "read at most",
               tiles * tile_bytes / 1e6,
               (double) frame_bytes(format, h, row) / 1e6);
        geist_decknet_stream_destroy(s);
    }
    bool same = true;
    for (size_t gated = 0; camera_fps <= 0.0 && only == nullptr && gated < 2; gated++) {
        const bool ok = t_way[gated].hash == t_still.hash;
        same          = same && ok;
        printf("%4zux%-4zu  %-4s  %s / still %.2fx frames/s, outputs %s (hash %016llx)\n",
               w,
               h,
               name,
               gated ? "gated" : "stream",
               t_way[gated].fps / t_still.fps,
               ok ? "bit-identical" : "DIFFER",
               (unsigned long long) t_way[gated].hash);
    }
    for (size_t k = 0; k < 2 * RING; k++) {
        free(ring[k]);
    }
    free(ms);
    free(cur8);
    free(ref8);
    free(ref);
    free(tmp);
    free(still);
    geist_decknet_destroy(d);
    return same;
}

static const char *const scene_names[4] = {"block", "all", "half", "burst"};
static const char *const scene_text[4]  = {
        "a block moves over a still scene",
        "every pixel changes from frame to frame",
        "every pixel of the upper half changes from frame to frame",
        "30 frames in which every pixel changes, 30 with a block over a still scene, in turn",
};

static bool scene_of(const char *name, enum scene *out) {
    for (uint32_t i = 0; i < 4; i++) {
        if (strcmp(name, scene_names[i]) == 0) {
            *out = (enum scene) i;
            return true;
        }
    }
    return false;
}

int main(int argc, char **argv) {
    size_t      n_frames = 300;
    size_t      threads  = 1;
    enum scene  scene    = SCENE_BLOCK;
    const char *only     = nullptr;
    const char *backend  = "cpu";
    size_t      height   = 0;   /* both sizes */
    double      camera   = 0.0; /* frames per second; 0: as fast as the pass */
    bool        trace    = false;
    bool        window   = false;
    const char *format   = nullptr; /* all three */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            n_frames = (size_t) strtoul(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = (size_t) strtoul(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--scene") == 0 && i + 1 < argc &&
                   scene_of(argv[i + 1], &scene)) {
            i++;
        } else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc &&
                   (strcmp(argv[i + 1], "still") == 0 || strcmp(argv[i + 1], "stream") == 0 ||
                    strcmp(argv[i + 1], "gated") == 0)) {
            only = argv[++i];
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc &&
                   (strcmp(argv[i + 1], "720") == 0 || strcmp(argv[i + 1], "1080") == 0)) {
            height = (size_t) strtoul(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--camera") == 0 && i + 1 < argc) {
            camera = strtod(argv[++i], nullptr);
            if (!(camera >= 1.0 && camera <= 1000.0)) {
                die("--camera: 1 to 1000 frames per second");
            }
        } else if (strcmp(argv[i], "--trace") == 0) {
            trace = true;
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
        } else if (strcmp(argv[i], "--window") == 0) {
            window = true;
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc &&
                   (strcmp(argv[i + 1], format_names[0]) == 0 ||
                    strcmp(argv[i + 1], format_names[1]) == 0 ||
                    strcmp(argv[i + 1], format_names[2]) == 0 ||
                    strcmp(argv[i + 1], format_names[3]) == 0)) {
            format = argv[++i];
        } else if (strcmp(argv[i], "--colour") == 0 && i + 1 < argc) {
            uint32_t c = 0;
            while (c < 4 && strcmp(argv[i + 1], colour_names[c]) != 0) {
                c++;
            }
            if (c == 4) {
                die("--colour bt601|bt601-full|bt709|bt709-full");
            }
            g_colour = (enum geist_decknet_colour) c;
            i++;
        } else {
            fprintf(stderr,
                    "usage: bench_stream [--frames N] [--threads 1|2] [--scene "
                    "block|all|half|burst] "
                    "[--only still|stream|gated] [--size 720|1080] [--camera FPS] [--trace] "
                    "[--backend cpu|metal|vulkan] [--window] [--format rgb8|nv12|yuyv|uyvy] "
                    "[--colour bt601|bt601-full|bt709|bt709-full]\n");
            return 2;
        }
    }
    if (n_frames < 1 || n_frames > 1000000) {
        die("--frames: 1 to 1000000");
    }
    if (threads < 1 || threads > 2) {
        die("--threads: 1 or 2");
    }
    if (trace && camera <= 0.0) {
        die("--trace: with --camera");
    }
    /* A backend for the streams: any name geist_backend_create takes that
     * has a decknet path. "cpu" is none: the pass is the library's own C. */
    struct geist_backend *be = nullptr;
    if (strcmp(backend, "cpu") != 0 &&
        geist_backend_create(backend, nullptr, nullptr, &be) != GEIST_OK) {
        die("--backend: that backend is not built in, or has no device here");
    }
    const double look_s = camera > 0.0 ? look_seconds() : 0.0;
    printf("bench_stream: invented network, %zu slots, %zu homographies, crop %zu x %zu; "
           "%zu frames per row after %zu warm-up; %s; the streams on %zu thread%s, the still "
           "path on one; backend %s%s\n",
           N_SLOTS,
           1 + 5 * N_SLOTS,
           CROP_W,
           CROP_H,
           n_frames,
           WARMUP,
           scene_text[scene],
           threads,
           threads == 1 ? "" : "s",
           backend,
           be != nullptr ? " for the frames of a stream that are not gated" : "");
    const size_t sizes[2][2] = {{1280, 720}, {1920, 1080}};
    bool         same        = true;
    for (size_t i = 0; i < 2; i++) {
        if (height != 0 && sizes[i][1] != height) {
            continue;
        }
        for (uint32_t f = 0; f < 4; f++) {
            if (format != nullptr && strcmp(format, format_names[f]) != 0) {
                continue;
            }
            same = measure(sizes[i][0],
                           sizes[i][1],
                           (enum geist_decknet_format) f,
                           n_frames,
                           threads,
                           scene,
                           camera,
                           trace,
                           window,
                           only,
                           be,
                           look_s) &&
                   same;
        }
    }
    geist_backend_destroy(be);
    if (!same) {
        fprintf(stderr, "bench_stream: a stream is not bit-identical to the still path\n");
        return 1;
    }
    return 0;
}
