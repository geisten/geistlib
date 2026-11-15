/* tests/decknet_tiny.h — a decknet GGUF small enough to build in memory.
 *
 * Shared by test_decknet_malformed_unit (which breaks one thing at a time)
 * and fuzz_decknet (which uses it as the seed). Two slots, a 16 x 12 image,
 * an 8 x 6 crop, a 4 x 5 worktable strip, 3 x 3 kernels and a handful of
 * pooling rows: the whole file contract of docs/DECKNET.md in about 2 kB.
 *
 * Every homography is [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], which maps crop
 * pixel (x, y) onto image pixel (x, y) exactly, so a test can compute the
 * expected units by hand.
 *
 * A caller fills `struct tiny` with tiny_default(), changes what its case is
 * about, and serializes with tiny_write().
 */
#ifndef GEIST_TEST_DECKNET_TINY_H
#define GEIST_TEST_DECKNET_TINY_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t TINY_IMG_W   = 16;
constexpr size_t TINY_IMG_H   = 12;
constexpr size_t TINY_CROP_W  = 8;
constexpr size_t TINY_CROP_H  = 6;
constexpr size_t TINY_MAX_H   = 16;
constexpr size_t TINY_SCALARS = 15;

struct tiny_buf {
    uint8_t b[4096];
    size_t  n;
    size_t  n_data; /* tiny_write: bytes up to the end of the last tensor's values */
};

/* One tensor as it is written: numpy shape, F32 payload. */
struct tiny_tensor {
    const char *name;
    uint32_t    n_dims;
    uint64_t    shape[3]; /* numpy order; written reversed */
    uint32_t    dtype;    /* 0 = GGUF F32 */
    size_t      n;        /* payload values written */
    float       v[TINY_MAX_H * 9];
};

enum {
    TINY_T_HOMOGRAPHY,
    TINY_T_PLANES,
    TINY_T_BG_NOMINAL,
    TINY_T_BLOB,
    TINY_T_TUBE,
    TINY_T_CHANGE,
    TINY_T_WALL,
    TINY_T_POOL_ON,
    TINY_T_POOL_OFF,
    TINY_T_POOL_TUBE,
    TINY_T_POOL_WALL,
    TINY_T_POOL_OCC_FRONT,
    TINY_T_POOL_OCC_REAR,
    TINY_N_TENSORS,
};

struct tiny {
    const char        *arch;  /* nullptr leaves general.architecture out */
    const char        *slots; /* nullptr leaves decknet.slots out */
    uint32_t           img_w, img_h, crop_w, crop_h, bg_w, bg_h;
    const char        *drop_key;              /* a metadata key to leave out, or nullptr */
    const char        *string_key;            /* a metadata key to write as a string, or nullptr */
    float              scalars[TINY_SCALARS]; /* in tiny_scalar_keys order */
    struct tiny_tensor t[TINY_N_TENSORS + 1];
    size_t             n_tensors; /* TINY_N_TENSORS; + 1 writes t[TINY_N_TENSORS] too */
};

static const char *const tiny_scalar_keys[TINY_SCALARS] = {
        "decknet.t_change",
        "decknet.g_change",
        "decknet.t_wall",
        "decknet.g_wall",
        "decknet.t_tip",
        "decknet.t_dark",
        "decknet.t_tube",
        "decknet.g_unit",
        "decknet.t_white",
        "decknet.g_white",
        "decknet.gain_min",
        "decknet.gain_max",
        "decknet.bg_rest_max",
        "decknet.occ",
        "decknet.smooth",
};
enum {
    TINY_S_T_CHANGE,
    TINY_S_G_CHANGE,
    TINY_S_T_WALL,
    TINY_S_G_WALL,
    TINY_S_T_TIP,
    TINY_S_T_DARK,
    TINY_S_T_TUBE,
    TINY_S_G_UNIT,
    TINY_S_T_WHITE,
    TINY_S_G_WHITE,
    TINY_S_GAIN_MIN,
    TINY_S_GAIN_MAX,
    TINY_S_BG_REST_MAX,
    TINY_S_OCC,
    TINY_S_SMOOTH,
};

static void tiny_put(struct tiny_buf *o, const void *p, size_t n) {
    if (o->n + n > sizeof o->b) {
        fprintf(stderr, "decknet_tiny: buffer overflow\n");
        exit(99);
    }
    memcpy(o->b + o->n, p, n);
    o->n += n;
}
static void tiny_u32(struct tiny_buf *o, uint32_t v) {
    tiny_put(o, &v, 4);
}
static void tiny_u64(struct tiny_buf *o, uint64_t v) {
    tiny_put(o, &v, 8);
}
static void tiny_f32(struct tiny_buf *o, float v) {
    tiny_put(o, &v, 4);
}
static void tiny_str(struct tiny_buf *o, const char *s) {
    tiny_u64(o, strlen(s));
    tiny_put(o, s, strlen(s));
}

static void tiny_set(struct tiny_tensor *t,
                     const char         *name,
                     uint32_t            n_dims,
                     uint64_t            d0,
                     uint64_t            d1,
                     uint64_t            d2,
                     size_t              n,
                     const float        *v) {
    *t = (struct tiny_tensor) {.name = name, .n_dims = n_dims, .shape = {d0, d1, d2}, .n = n};
    memcpy(t->v, v, n * sizeof(float));
}

/* Pooling rows (unit, pixel, weight). */
static void tiny_pool(struct tiny_tensor *t, const char *name, size_t rows, const float *v) {
    tiny_set(t, name, 2, rows, 3, 0, rows * 3, v);
}

static void tiny_default(struct tiny *m) {
    *m = (struct tiny) {
            .arch      = "decknet",
            .slots     = "AB",
            .img_w     = (uint32_t) TINY_IMG_W,
            .img_h     = (uint32_t) TINY_IMG_H,
            .crop_w    = (uint32_t) TINY_CROP_W,
            .crop_h    = (uint32_t) TINY_CROP_H,
            .bg_w      = 4,
            .bg_h      = 5,
            .n_tensors = TINY_N_TENSORS,
            /* thresholds 0 and gains 1 unless a test says otherwise */
            .scalars = {[TINY_S_G_CHANGE]    = 1.0f,
                        [TINY_S_G_WALL]      = 1.0f,
                        [TINY_S_G_UNIT]      = 1.0f,
                        [TINY_S_G_WHITE]     = 1.0f,
                        [TINY_S_GAIN_MIN]    = 0.8f,
                        [TINY_S_GAIN_MAX]    = 1.25f,
                        [TINY_S_BG_REST_MAX] = 0.02f,
                        [TINY_S_OCC]         = 0.25f,
                        [TINY_S_SMOOTH]      = 0.3f},
    };
    float h[11 * 9];
    for (size_t i = 0; i < 11; i++) {
        const float one[9] = {1, 0, 0.5f, 0, 1, 0.5f, 0, 0, 1};
        memcpy(h + i * 9, one, sizeof one);
    }
    tiny_set(&m->t[TINY_T_HOMOGRAPHY], "decknet.homography", 3, 11, 3, 3, 99, h);
    const float planes[5] = {0, 14.4f, 41.4f, 50, 60};
    tiny_set(&m->t[TINY_T_PLANES], "decknet.planes_mm", 1, 5, 0, 0, 5, planes);
    const float nominal[3] = {0.5f, 0.5f, 0.5f};
    tiny_set(&m->t[TINY_T_BG_NOMINAL], "decknet.bg_nominal", 1, 3, 0, 0, 3, nominal);
    const float blob[9] = {
            -0.125f, -0.125f, -0.125f, -0.125f, 1, -0.125f, -0.125f, -0.125f, -0.125f};
    tiny_set(&m->t[TINY_T_BLOB], "decknet.conv.blob", 2, 3, 3, 0, 9, blob);
    tiny_set(&m->t[TINY_T_TUBE], "decknet.conv.tube", 2, 3, 3, 0, 9, blob);
    const float change[36] = {1,  0, 0, -1, 0, 0, 0, 1,  0, 0, -1, 0, 0, 0, 1,  0, 0, -1,
                              -1, 0, 0, 1,  0, 0, 0, -1, 0, 0, 1,  0, 0, 0, -1, 0, 0, 1};
    tiny_set(&m->t[TINY_T_CHANGE], "decknet.change.w", 2, 6, 6, 0, 36, change);
    const float wall[12] = {-1, 0, 1, 0.5f, 0, -0.5f, 0, -1, 1, 0, 0.5f, -0.5f};
    tiny_set(&m->t[TINY_T_WALL], "decknet.wall.w", 2, 2, 6, 0, 12, wall);
    const float on[9] = {0, 9, 0.5f, 0, 10, 0.5f, 1, 20, 1};
    tiny_pool(&m->t[TINY_T_POOL_ON], "decknet.pool.on", 3, on);
    const float off[6] = {0, 12, 1, 1, 30, 1};
    tiny_pool(&m->t[TINY_T_POOL_OFF], "decknet.pool.off", 2, off);
    const float tube[6] = {0, 19, 1, 1, 40, 1};
    tiny_pool(&m->t[TINY_T_POOL_TUBE], "decknet.pool.tube", 2, tube);
    const float wallp[6] = {0, 1, 0.5f, 0, 2, 0.5f};
    tiny_pool(&m->t[TINY_T_POOL_WALL], "decknet.pool.wall", 2, wallp);
    const float front[6] = {0, 3, 0.5f, 0, 4, 0.5f};
    tiny_pool(&m->t[TINY_T_POOL_OCC_FRONT], "decknet.pool.occ_front", 2, front);
    const float rear[6] = {0, 43, 0.5f, 0, 44, 0.5f};
    tiny_pool(&m->t[TINY_T_POOL_OCC_REAR], "decknet.pool.occ_rear", 2, rear);
    const float extra[1] = {0};
    tiny_set(&m->t[TINY_N_TENSORS], "decknet.conv.next", 1, 1, 0, 0, 1, extra);
}

static bool tiny_is(const char *a, const char *b) {
    return a != nullptr && strcmp(a, b) == 0;
}

static void tiny_meta_u32(struct tiny_buf *o, const struct tiny *m, const char *key, uint32_t v) {
    if (tiny_is(m->drop_key, key)) {
        return;
    }
    tiny_str(o, key);
    if (tiny_is(m->string_key, key)) {
        tiny_u32(o, 8);
        tiny_str(o, "7");
        return;
    }
    tiny_u32(o, 4);
    tiny_u32(o, v);
}

static void tiny_write(const struct tiny *m, struct tiny_buf *o) {
    o->n = 0;
    tiny_put(o, "GGUF", 4);
    tiny_u32(o, 3);
    tiny_u64(o, m->n_tensors);
    const size_t n_meta_at = o->n;
    tiny_u64(o, 0); /* patched below */
    uint64_t n_meta = 0;
    if (m->arch != nullptr) {
        tiny_str(o, "general.architecture");
        tiny_u32(o, 8);
        tiny_str(o, m->arch);
        n_meta++;
    }
    if (m->slots != nullptr) {
        tiny_str(o, "decknet.slots");
        tiny_u32(o, 8);
        tiny_str(o, m->slots);
        n_meta++;
    }
    const struct {
        const char *key;
        uint32_t    v;
    } dims[6] = {
            {"decknet.image.width", m->img_w},
            {"decknet.image.height", m->img_h},
            {"decknet.crop.width", m->crop_w},
            {"decknet.crop.height", m->crop_h},
            {"decknet.bg.width", m->bg_w},
            {"decknet.bg.height", m->bg_h},
    };
    for (size_t i = 0; i < 6; i++) {
        const size_t before = o->n;
        tiny_meta_u32(o, m, dims[i].key, dims[i].v);
        n_meta += o->n != before;
    }
    for (size_t i = 0; i < TINY_SCALARS; i++) {
        if (tiny_is(m->drop_key, tiny_scalar_keys[i])) {
            continue;
        }
        tiny_str(o, tiny_scalar_keys[i]);
        if (tiny_is(m->string_key, tiny_scalar_keys[i])) {
            tiny_u32(o, 8);
            tiny_str(o, "7");
        } else {
            tiny_u32(o, 6);
            tiny_f32(o, m->scalars[i]);
        }
        n_meta++;
    }
    memcpy(o->b + n_meta_at, &n_meta, 8);

    uint64_t offset = 0;
    for (size_t i = 0; i < m->n_tensors; i++) {
        const struct tiny_tensor *t = &m->t[i];
        tiny_str(o, t->name);
        tiny_u32(o, t->n_dims);
        for (uint32_t k = 0; k < t->n_dims; k++) {
            tiny_u64(o, t->shape[t->n_dims - 1 - k]);
        }
        tiny_u32(o, t->dtype);
        tiny_u64(o, offset);
        offset += (t->n * 4 + 31) / 32 * 32;
    }
    while (o->n % 32 != 0) {
        tiny_put(o, "", 1);
    }
    for (size_t i = 0; i < m->n_tensors; i++) {
        const struct tiny_tensor *t = &m->t[i];
        tiny_put(o, t->v, t->n * 4);
        o->n_data = o->n;
        while (o->n % 32 != 0) {
            tiny_put(o, "", 1);
        }
    }
}

#endif /* GEIST_TEST_DECKNET_TINY_H */
