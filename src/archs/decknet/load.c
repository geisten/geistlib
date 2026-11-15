/*
 * src/archs/decknet/load.c — the decknet file contract (#532).
 *
 * Layer: ARCHITECTURE. Everything in the file is untrusted: each metadata
 * value and each tensor is checked against the contract in docs/DECKNET.md
 * and copied into the handle, and a file that differs in any way is refused
 * with GEIST_E_FORMAT and a message naming the key, the tensor and the row.
 * Nothing is skipped or defaulted, and a file with a tensor this loader does
 * not know is refused as well: a newer network read by an older loader would
 * otherwise give an answer with a layer missing.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#include "decknet.h"

#include "checked.h"
#include "cleanup.h"
#include "heap.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* The loader's NaN and Inf checks are comparisons the compiler may fold
 * away under -ffinite-math-only (part of -ffast-math). */
static_assert(__FINITE_MATH_ONLY__ == 0,
              "decknet needs -fno-finite-math-only (mk/common.mk): the loader rejects NaN and Inf "
              "by comparison");

constexpr size_t N_TENSORS = 13;
/* F32 holds every integer below 2^24, so an index stored as F32 is exact. */
constexpr size_t F32_EXACT = (size_t) 1 << 24U;
static_assert(DECKNET_MAX_PIXELS <= F32_EXACT);

struct loader {
    const struct gguf_ctx *g;
    size_t                 cap;
    char                  *err;
    enum geist_status      status;
};

[[gnu::format(printf, 3, 4)]] static bool
fail(struct loader *l, enum geist_status status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(l->err, l->cap, fmt, ap);
    va_end(ap);
    l->status = status;
    return false;
}

static bool oom(struct loader *l) {
    return fail(l, GEIST_E_OOM, "out of memory");
}

/* ---- metadata --------------------------------------------------------- */

static bool meta_size(struct loader *l, const char *key, size_t lo, size_t hi, size_t *out) {
    uint32_t v = 0;
    if (!gguf_get_meta_u32(l->g, key, &v)) {
        return fail(l, GEIST_E_FORMAT, "%s: missing or not an unsigned integer", key);
    }
    if (v < lo || v > hi) {
        return fail(l, GEIST_E_FORMAT, "%s = %u: outside [%zu, %zu]", key, v, lo, hi);
    }
    *out = v;
    return true;
}

static bool meta_float(struct loader *l, const char *key, float *out) {
    float v = 0.0f;
    if (!gguf_get_meta_f32(l->g, key, &v)) {
        return fail(l, GEIST_E_FORMAT, "%s: missing or not an F32", key);
    }
    if (!isfinite(v)) {
        return fail(l, GEIST_E_FORMAT, "%s is not finite", key);
    }
    *out = v;
    return true;
}

static bool load_meta(struct loader *l, struct geist_decknet *d) {
    size_t      len   = 0;
    const char *slots = gguf_get_meta_string(l->g, "decknet.slots", &len);
    if (slots == nullptr || len < 1 || len > DECKNET_MAX_SLOTS) {
        return fail(l,
                    GEIST_E_FORMAT,
                    "decknet.slots: missing, or not a string of 1 to %zu letters",
                    DECKNET_MAX_SLOTS);
    }
    for (size_t i = 0; i < len; i++) {
        if (slots[i] < 0x21 || slots[i] > 0x7e) {
            return fail(l, GEIST_E_FORMAT, "decknet.slots[%zu] is not a printable letter", i);
        }
    }
    memcpy(d->slots, slots, len);
    d->slots[len] = '\0';
    d->n_slots    = len;
    d->n_h        = 1 + DECKNET_PLANES * len;

    /* Bilinear sampling reads a 2 x 2 neighbourhood, so an image has at
     * least two pixels each way. */
    if (!meta_size(l, "decknet.image.width", 2, DECKNET_MAX_IMAGE_DIM, &d->img_w) ||
        !meta_size(l, "decknet.image.height", 2, DECKNET_MAX_IMAGE_DIM, &d->img_h) ||
        !meta_size(l, "decknet.crop.width", 1, DECKNET_MAX_PIXELS, &d->crop_w) ||
        !meta_size(l, "decknet.crop.height", 1, DECKNET_MAX_PIXELS, &d->crop_h) ||
        !meta_size(l, "decknet.bg.width", 1, DECKNET_MAX_PIXELS, &d->bg_w) ||
        !meta_size(l, "decknet.bg.height", 1, DECKNET_MAX_PIXELS, &d->bg_h)) {
        return false;
    }
    if (ckd_mul(&d->img_bytes, d->img_w, d->img_h) || ckd_mul(&d->img_bytes, d->img_bytes, 3)) {
        return fail(l, GEIST_E_FORMAT, "decknet.image: size overflows");
    }
    if (ckd_mul(&d->crop_px, d->crop_w, d->crop_h) || d->crop_px > DECKNET_MAX_PIXELS) {
        return fail(l, GEIST_E_FORMAT, "decknet.crop: more than %zu pixels", DECKNET_MAX_PIXELS);
    }
    if (ckd_mul(&d->bg_px, d->bg_w, d->bg_h) || d->bg_px > DECKNET_MAX_PIXELS) {
        return fail(l, GEIST_E_FORMAT, "decknet.bg: more than %zu pixels", DECKNET_MAX_PIXELS);
    }

    const struct {
        const char *key;
        float      *out;
    } scalars[] = {
            {"decknet.t_change", &d->t_change},
            {"decknet.g_change", &d->g_change},
            {"decknet.t_wall", &d->t_wall},
            {"decknet.g_wall", &d->g_wall},
            {"decknet.t_tip", &d->t_tip},
            {"decknet.t_dark", &d->t_dark},
            {"decknet.t_tube", &d->t_tube},
            {"decknet.g_unit", &d->g_unit},
            {"decknet.t_white", &d->t_white},
            {"decknet.g_white", &d->g_white},
            {"decknet.gain_min", &d->gain_min},
            {"decknet.gain_max", &d->gain_max},
            {"decknet.bg_rest_max", &d->bg_rest_max},
            {"decknet.occ", &d->occ},
            {"decknet.smooth", &d->smooth},
    };
    for (size_t i = 0; i < sizeof scalars / sizeof scalars[0]; i++) {
        if (!meta_float(l, scalars[i].key, scalars[i].out)) {
            return false;
        }
    }
    return true;
}

/* ---- tensors ---------------------------------------------------------- */

/* The F32 tensor `name` with exactly n_dims dimensions, or nullptr. */
static const struct gguf_tensor_t *tensor(struct loader *l, const char *name, int n_dims) {
    const struct gguf_tensor_t *t = gguf_get_tensor(l->g, name);
    if (t == nullptr) {
        fail(l, GEIST_E_FORMAT, "%s: tensor missing", name);
        return nullptr;
    }
    if (t->dtype != GGUF_TYPE_F32 || t->n_dims != n_dims) {
        fail(l, GEIST_E_FORMAT, "%s: not an F32 tensor of %d dimensions", name, n_dims);
        return nullptr;
    }
    return t;
}

/* Copies the n F32 values of t to dst. False when the payload is not
 * exactly n values or one of them is NaN or Inf. */
static bool
read_floats(struct loader *l, const struct gguf_tensor_t *t, size_t n, float dst[static n]) {
    size_t need = 0;
    if (ckd_mul(&need, n, sizeof(float)) || t->data == nullptr || t->nbytes != need) {
        return fail(l, GEIST_E_FORMAT, "%s: payload is not %zu F32 values", t->name, n);
    }
    memcpy(dst, t->data, need);
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(dst[i])) {
            return fail(l, GEIST_E_FORMAT, "%s[%zu] is not finite", t->name, i);
        }
    }
    return true;
}

static bool load_vector(struct loader *l, const char *name, size_t n, float dst[static n]) {
    const struct gguf_tensor_t *t = tensor(l, name, 1);
    if (t == nullptr) {
        return false;
    }
    if (t->dims[0] != n) {
        return fail(l, GEIST_E_FORMAT, "%s: not %zu values", name, n);
    }
    return read_floats(l, t, n, dst);
}

/* numpy (n_h, 3, 3); GGUF lists the innermost axis first. */
static bool load_homographies(struct loader *l, struct geist_decknet *d) {
    const struct gguf_tensor_t *t = tensor(l, "decknet.homography", 3);
    if (t == nullptr) {
        return false;
    }
    if (t->dims[0] != 3 || t->dims[1] != 3 || t->dims[2] != d->n_h) {
        return fail(l,
                    GEIST_E_FORMAT,
                    "decknet.homography: not %zu x 3 x 3 (1 + %zu planes for each of %zu slots)",
                    d->n_h,
                    DECKNET_PLANES,
                    d->n_slots);
    }
    const size_t n = d->n_h * 9; /* n_h <= 1 + 5 * 64 */
    d->h_cur       = heap_alloc_array_aligned(float, n);
    d->h_ref       = heap_alloc_array_aligned(float, n);
    if (d->h_cur == nullptr || d->h_ref == nullptr) {
        return oom(l);
    }
    if (!read_floats(l, t, n, d->h_cur)) {
        return false;
    }
    memcpy(d->h_ref, d->h_cur, n * sizeof(float));
    return true;
}

static bool load_kernel(struct loader *l, const char *name, struct decknet_kernel *k) {
    const struct gguf_tensor_t *t = tensor(l, name, 2);
    if (t == nullptr) {
        return false;
    }
    const uint64_t side = t->dims[0];
    if (side != t->dims[1] || side > DECKNET_MAX_KERNEL || side % 2 == 0) {
        return fail(l,
                    GEIST_E_FORMAT,
                    "%s: not square with an odd side of at most %zu",
                    name,
                    DECKNET_MAX_KERNEL);
    }
    k->k = (size_t) side;
    k->w = heap_alloc_array_aligned(float, k->k * k->k);
    if (k->w == nullptr) {
        return oom(l);
    }
    return read_floats(l, t, k->k * k->k, k->w);
}

/* numpy (rows, 6). */
static bool load_mix(struct loader *l, const char *name, struct decknet_mix *m) {
    const struct gguf_tensor_t *t = tensor(l, name, 2);
    if (t == nullptr) {
        return false;
    }
    if (t->dims[0] != DECKNET_MIX_IN || t->dims[1] < 1 || t->dims[1] > DECKNET_MAX_MIX_ROWS) {
        return fail(l,
                    GEIST_E_FORMAT,
                    "%s: not 1 to %zu rows of %zu weights",
                    name,
                    DECKNET_MAX_MIX_ROWS,
                    DECKNET_MIX_IN);
    }
    m->rows = (size_t) t->dims[1];
    m->w    = heap_alloc_array_aligned(float, m->rows *DECKNET_MIX_IN);
    if (m->w == nullptr) {
        return oom(l);
    }
    return read_floats(l, t, m->rows * DECKNET_MIX_IN, m->w);
}

/* v is an integer in [0, limit), limit <= 2^24. False for NaN, Inf and a
 * fraction: every comparison with NaN is false. */
static bool is_index(float v, size_t limit) {
    return v >= 0.0f && v < (float) limit && v == floorf(v);
}

/* numpy (rows, 3): unit, pixel index in the crop, weight. The rows are
 * ascending by (unit, pixel) with no unit left out, as the exporter writes
 * them: that excludes a pixel listed twice for a unit and a unit that pools
 * nothing, and bounds the unit count by the row count. */
static bool load_pool(struct loader *l, const char *name, size_t n_px, struct decknet_pool *p) {
    const struct gguf_tensor_t *t = tensor(l, name, 2);
    if (t == nullptr) {
        return false;
    }
    constexpr size_t row_bytes = 3 * sizeof(float);
    if (t->dims[0] != 3 || t->dims[1] < 1 || t->dims[1] > SIZE_MAX / row_bytes ||
        t->data == nullptr || t->nbytes != (size_t) t->dims[1] * row_bytes) {
        return fail(l, GEIST_E_FORMAT, "%s: not rows of (unit, pixel, weight)", name);
    }
    p->n_rows = (size_t) t->dims[1];
    p->unit   = heap_alloc_array_aligned(uint32_t, p->n_rows);
    p->px     = heap_alloc_array_aligned(uint32_t, p->n_rows);
    p->w      = heap_alloc_array_aligned(float, p->n_rows);
    if (p->unit == nullptr || p->px == nullptr || p->w == nullptr) {
        return oom(l);
    }
    const uint8_t *rows = t->data;
    for (size_t r = 0; r < p->n_rows; r++) {
        float row[3];
        memcpy(row, rows + r * sizeof row, sizeof row);
        if (!is_index(row[0], F32_EXACT)) {
            return fail(l, GEIST_E_FORMAT, "%s row %zu: unit is not an index", name, r);
        }
        if (!is_index(row[1], n_px)) {
            return fail(l,
                        GEIST_E_FORMAT,
                        "%s row %zu: pixel is not an index into the %zu crop pixels",
                        name,
                        r,
                        n_px);
        }
        if (!isfinite(row[2])) {
            return fail(l, GEIST_E_FORMAT, "%s row %zu: weight is not finite", name, r);
        }
        const uint32_t unit = (uint32_t) row[0];
        const uint32_t px   = (uint32_t) row[1];
        const bool     next = r == 0 ? unit == 0 : unit == p->unit[r - 1] + 1;
        const bool     same = r > 0 && unit == p->unit[r - 1] && px > p->px[r - 1];
        if (!next && !same) {
            return fail(l,
                        GEIST_E_FORMAT,
                        "%s row %zu: rows are not ascending by (unit, pixel) from unit 0",
                        name,
                        r);
        }
        p->unit[r] = unit;
        p->px[r]   = px;
        p->w[r]    = row[2];
    }
    p->n_units = (size_t) p->unit[p->n_rows - 1] + 1;
    return true;
}

/* The crop pixels the rows of `a` and (when not nullptr) `b` name,
 * ascending and each once. */
static bool pixel_list(struct loader             *l,
                       size_t                     n_px,
                       const struct decknet_pool *a,
                       const struct decknet_pool *b,
                       size_t                    *out_n,
                       uint32_t                 **out) {
    GEIST_CLEANUP(heap_freep) uint8_t *mark = heap_calloc_array_aligned(uint8_t, n_px);
    if (mark == nullptr) {
        return oom(l);
    }
    for (size_t r = 0; r < a->n_rows; r++) {
        mark[a->px[r]] = 1;
    }
    for (size_t r = 0; b != nullptr && r < b->n_rows; r++) {
        mark[b->px[r]] = 1;
    }
    size_t n = 0;
    for (size_t i = 0; i < n_px; i++) {
        n += mark[i];
    }
    uint32_t *list = heap_alloc_array_aligned(uint32_t, n); /* n >= 1: a pool has a row */
    if (list == nullptr) {
        return oom(l);
    }
    size_t k = 0;
    for (size_t i = 0; i < n_px; i++) {
        if (mark[i] != 0) {
            list[k++] = (uint32_t) i;
        }
    }
    *out_n = n;
    *out   = list;
    return true;
}

static bool load_tensors(struct loader *l, struct geist_decknet *d) {
    const size_t n_tensors = gguf_tensor_count(l->g);
    if (n_tensors != N_TENSORS) {
        return fail(l,
                    GEIST_E_FORMAT,
                    "%zu tensors where the decknet contract has %zu",
                    n_tensors,
                    N_TENSORS);
    }
    if (!load_homographies(l, d) ||
        !load_vector(l, "decknet.planes_mm", DECKNET_PLANES, d->planes_mm) ||
        !load_vector(l, "decknet.bg_nominal", 3, d->bg_nominal) ||
        !load_kernel(l, "decknet.conv.blob", &d->blob) ||
        !load_kernel(l, "decknet.conv.tube", &d->tube_kernel) ||
        !load_mix(l, "decknet.change.w", &d->change) || !load_mix(l, "decknet.wall.w", &d->wall) ||
        !load_pool(l, "decknet.pool.on", d->crop_px, &d->on) ||
        !load_pool(l, "decknet.pool.off", d->crop_px, &d->off) ||
        !load_pool(l, "decknet.pool.tube", d->crop_px, &d->tube) ||
        !load_pool(l, "decknet.pool.wall", d->crop_px, &d->wall_pool) ||
        !load_pool(l, "decknet.pool.occ_front", d->crop_px, &d->occ_front) ||
        !load_pool(l, "decknet.pool.occ_rear", d->crop_px, &d->occ_rear)) {
        return false;
    }
    if (d->on.n_units != d->off.n_units) {
        return fail(l,
                    GEIST_E_FORMAT,
                    "decknet.pool.on has %zu units, decknet.pool.off %zu",
                    d->on.n_units,
                    d->off.n_units);
    }
    if (d->tube.n_units % 2 != 0) {
        return fail(l,
                    GEIST_E_FORMAT,
                    "decknet.pool.tube has %zu units, not two orientations of equal size",
                    d->tube.n_units);
    }
    if (d->wall_pool.n_units != 1 || d->occ_front.n_units != 1 || d->occ_rear.n_units != 1) {
        return fail(
                l, GEIST_E_FORMAT, "decknet.pool.wall, occ_front and occ_rear have one unit each");
    }
    d->n_tips  = d->on.n_units;
    d->n_tubes = d->tube.n_units / 2;
    return pixel_list(l, d->crop_px, &d->on, &d->off, &d->n_blob_px, &d->blob_px) &&
           pixel_list(l, d->crop_px, &d->tube, nullptr, &d->n_tube_px, &d->tube_px);
}

/* ---- workspace -------------------------------------------------------- */

static bool alloc_workspace(struct loader *l, struct geist_decknet *d) {
    size_t crop_floats = 0, bg_floats = 0, tip_floats = 0, tube_floats = 0;
    if (ckd_mul(&crop_floats, d->crop_px, 3) || ckd_mul(&bg_floats, d->bg_px, 3) ||
        ckd_mul(&tip_floats, d->n_slots, d->n_tips) ||
        ckd_mul(&tube_floats, d->n_slots, d->tube.n_units)) {
        return fail(l, GEIST_E_FORMAT, "workspace size overflows");
    }
    d->crop_a   = heap_alloc_array_aligned(float, crop_floats);
    d->crop_b   = heap_alloc_array_aligned(float, crop_floats);
    d->bg_a     = heap_alloc_array_aligned(float, bg_floats);
    d->bg_b     = heap_alloc_array_aligned(float, bg_floats);
    d->e_bright = heap_alloc_array_aligned(double, d->crop_px);
    d->e_dark   = heap_alloc_array_aligned(double, d->crop_px);
    d->e_tube   = heap_alloc_array_aligned(double, d->crop_px);
    d->acc_on   = heap_alloc_array_aligned(double, d->n_tips);
    d->acc_off  = heap_alloc_array_aligned(double, d->n_tips);
    d->unit     = heap_alloc_array_aligned(double, d->n_tips);
    d->hole     = heap_alloc_array_aligned(double, d->n_tips);
    d->acc_tube = heap_alloc_array_aligned(double, d->tube.n_units);
    d->slot_out = heap_calloc_array_aligned(struct geist_decknet_slot, d->n_slots);
    d->tips     = heap_calloc_array_aligned(float, tip_floats);
    d->tubes    = heap_calloc_array_aligned(float, tube_floats);
    if (d->crop_a == nullptr || d->crop_b == nullptr || d->bg_a == nullptr || d->bg_b == nullptr ||
        d->e_bright == nullptr || d->e_dark == nullptr || d->e_tube == nullptr ||
        d->acc_on == nullptr || d->acc_off == nullptr || d->unit == nullptr || d->hole == nullptr ||
        d->acc_tube == nullptr || d->slot_out == nullptr || d->tips == nullptr ||
        d->tubes == nullptr) {
        return oom(l);
    }
    for (size_t s = 0; s < d->n_slots; s++) {
        d->slot_out[s].tips      = d->tips + s * d->n_tips;
        d->slot_out[s].tubes     = d->tubes + s * d->tube.n_units;
        d->slot_out[s].tubes_rot = d->slot_out[s].tubes + d->n_tubes;
    }
    return true;
}

/* ---- entry points ----------------------------------------------------- */

static void pool_free(struct decknet_pool *p) {
    heap_freep(&p->unit);
    heap_freep(&p->px);
    heap_freep(&p->w);
}

void decknet_destroy(struct geist_decknet *d) {
    if (d == nullptr) {
        return;
    }
    heap_freep(&d->h_cur);
    heap_freep(&d->h_ref);
    heap_freep(&d->change.w);
    heap_freep(&d->wall.w);
    heap_freep(&d->blob.w);
    heap_freep(&d->tube_kernel.w);
    pool_free(&d->on);
    pool_free(&d->off);
    pool_free(&d->tube);
    pool_free(&d->wall_pool);
    pool_free(&d->occ_front);
    pool_free(&d->occ_rear);
    heap_freep(&d->blob_px);
    heap_freep(&d->tube_px);
    heap_freep(&d->crop_a);
    heap_freep(&d->crop_b);
    heap_freep(&d->bg_a);
    heap_freep(&d->bg_b);
    heap_freep(&d->e_bright);
    heap_freep(&d->e_dark);
    heap_freep(&d->e_tube);
    heap_freep(&d->acc_on);
    heap_freep(&d->acc_off);
    heap_freep(&d->unit);
    heap_freep(&d->hole);
    heap_freep(&d->acc_tube);
    heap_freep(&d->slot_out);
    heap_freep(&d->tips);
    heap_freep(&d->tubes);
    heap_freep(&d);
}

enum geist_status decknet_create(size_t                 err_cap,
                                 const struct gguf_ctx *g,
                                 char                   err[static err_cap],
                                 struct geist_decknet **out) {
    *out               = nullptr;
    struct loader l    = {.g = g, .cap = err_cap, .err = err, .status = GEIST_OK};
    size_t        n    = 0;
    const char   *arch = gguf_get_meta_string(g, "general.architecture", &n);
    if (arch == nullptr || n != strlen("decknet") || memcmp(arch, "decknet", n) != 0) {
        fail(&l, GEIST_E_UNSUPPORTED, "general.architecture is not \"decknet\"");
        return l.status;
    }
    struct geist_decknet *d = heap_calloc_array_aligned(struct geist_decknet, 1);
    if (d == nullptr) {
        oom(&l);
        return l.status;
    }
    if (!load_meta(&l, d) || !load_tensors(&l, d) || !alloc_workspace(&l, d)) {
        decknet_destroy(d);
        return l.status;
    }
    *out = d;
    return GEIST_OK;
}
