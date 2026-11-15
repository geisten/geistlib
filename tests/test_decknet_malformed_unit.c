/* tests/test_decknet_malformed_unit.c — the decknet loader fails closed (#532).
 *
 * No fixture: tests/decknet_tiny.h builds a complete two-slot decknet file
 * in memory, and each case breaks one thing in it. A rejected file must give
 * the expected status, a null handle and a create-time message that names
 * what was wrong; "it did not crash" is not the assertion.
 *
 *   - the valid file loads, and its units match values computed by hand
 *     (the kernel is applied as a convolution, the pooling rows weigh the
 *     pixels they name);
 *   - metadata: architecture, slots, each size and each scalar missing, of
 *     the wrong type, out of range or not finite;
 *   - tensors: one missing, one too many, wrong type, rank, shape, NaN;
 *   - sparse pooling rows, the untrusted indices of the contract: pixel out
 *     of range, negative, fractional, NaN; unit fractional, NaN, not from 0,
 *     with a gap; rows out of order or repeated; weight NaN or Inf;
 *   - every truncation of the file, byte by byte;
 *   - every allocation of a load failing once (heap_fail_after);
 *   - the entry points' argument checks and the file-path loader.
 */
#define _POSIX_C_SOURCE 200809L
#include "decknet_tiny.h"
#include "heap.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_decknet.h>

#include <math.h>
#include <unistd.h>

static int g_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail = 1;                                                     \
        }                                                                   \
    } while (0)

static enum geist_status load(const struct tiny *m, struct geist_decknet **out) {
    static struct tiny_buf buf;
    tiny_write(m, &buf);
    /* An exact-size copy: a read past the file lands in an ASan redzone. */
    uint8_t *copy = xmalloc(buf.n);
    memcpy(copy, buf.b, buf.n);
    const enum geist_status st = geist_decknet_load_from_memory(buf.n, copy, out);
    free(copy); /* the loader keeps nothing of the file */
    return st;
}

/* ---- cases: each breaks one thing ------------------------------------- */

static float *row(struct tiny *m, int tensor, size_t r) {
    return m->t[tensor].v + r * 3;
}
static void reshape(struct tiny *m, int tensor, uint32_t n_dims, uint64_t d0, uint64_t d1) {
    struct tiny_tensor *t = &m->t[tensor];
    t->n_dims             = n_dims;
    t->shape[0]           = d0;
    t->shape[1]           = d1;
    t->n                  = (size_t) (n_dims == 1 ? d0 : d0 * d1);
}

static void arch_llama(struct tiny *m) {
    m->arch = "llama";
}
static void arch_missing(struct tiny *m) {
    m->arch = nullptr;
}
static void arch_prefix(struct tiny *m) {
    m->arch = "decknet2";
}
static void slots_missing(struct tiny *m) {
    m->slots = nullptr;
}
static void slots_empty(struct tiny *m) {
    m->slots = "";
}
static void slots_blank(struct tiny *m) {
    m->slots = "A B";
}
static void slots_three(struct tiny *m) {
    m->slots = "ABC"; /* 16 homographies needed, 11 in the file */
}
static void image_one_wide(struct tiny *m) {
    m->img_w = 1;
}
static void image_huge(struct tiny *m) {
    m->img_h = 40000;
}
static void crop_zero(struct tiny *m) {
    m->crop_w = 0;
}
static void crop_huge(struct tiny *m) {
    m->crop_w = 4096;
    m->crop_h = 4096;
}
static void bg_zero(struct tiny *m) {
    m->bg_h = 0;
}
static void width_missing(struct tiny *m) {
    m->drop_key = "decknet.image.width";
}
static void crop_height_string(struct tiny *m) {
    m->string_key = "decknet.crop.height";
}
static void scalar_missing(struct tiny *m) {
    m->drop_key = "decknet.t_white";
}
static void scalar_string(struct tiny *m) {
    m->string_key = "decknet.g_unit";
}
static void scalar_nan(struct tiny *m) {
    m->scalars[TINY_S_T_TUBE] = NAN;
}
static void scalar_inf(struct tiny *m) {
    m->scalars[TINY_S_GAIN_MAX] = INFINITY;
}
static void tensor_dropped(struct tiny *m) {
    m->n_tensors = TINY_N_TENSORS - 1;
}
static void tensor_extra(struct tiny *m) {
    m->n_tensors = TINY_N_TENSORS + 1;
}
static void tensor_renamed(struct tiny *m) {
    m->t[TINY_T_WALL].name = "decknet.wall.v";
}
static void tensor_twice(struct tiny *m) {
    m->t[TINY_T_TUBE].name = "decknet.conv.blob";
}
static void tensor_f16(struct tiny *m) {
    m->t[TINY_T_BLOB].dtype = 1; /* GGUF F16: 18 of the 36 bytes */
}
static void homography_rank(struct tiny *m) {
    reshape(m, TINY_T_HOMOGRAPHY, 2, 11, 9);
}
static void homography_short(struct tiny *m) {
    m->t[TINY_T_HOMOGRAPHY].shape[0] = 10;
    m->t[TINY_T_HOMOGRAPHY].n        = 90;
}
static void homography_nan(struct tiny *m) {
    m->t[TINY_T_HOMOGRAPHY].v[50] = NAN;
}
static void planes_four(struct tiny *m) {
    reshape(m, TINY_T_PLANES, 1, 4, 0);
}
static void nominal_inf(struct tiny *m) {
    m->t[TINY_T_BG_NOMINAL].v[1] = -INFINITY;
}
static void kernel_even(struct tiny *m) {
    reshape(m, TINY_T_BLOB, 2, 2, 2);
}
static void kernel_oblong(struct tiny *m) {
    reshape(m, TINY_T_TUBE, 2, 3, 1);
}
static void kernel_nan(struct tiny *m) {
    m->t[TINY_T_TUBE].v[4] = NAN;
}
static void mix_five_wide(struct tiny *m) {
    reshape(m, TINY_T_CHANGE, 2, 6, 5);
}
static void mix_nan(struct tiny *m) {
    m->t[TINY_T_WALL].v[7] = NAN;
}
static void pool_pixel_past_end(struct tiny *m) {
    row(m, TINY_T_POOL_OFF, 1)[1] = (float) (TINY_CROP_W * TINY_CROP_H);
}
static void pool_pixel_far(struct tiny *m) {
    row(m, TINY_T_POOL_OCC_REAR, 1)[1] = 3e9f;
}
static void pool_pixel_negative(struct tiny *m) {
    row(m, TINY_T_POOL_ON, 0)[1] = -1.0f;
}
static void pool_pixel_fraction(struct tiny *m) {
    row(m, TINY_T_POOL_ON, 1)[1] = 10.5f;
}
static void pool_pixel_nan(struct tiny *m) {
    row(m, TINY_T_POOL_TUBE, 0)[1] = NAN;
}
static void pool_pixel_inf(struct tiny *m) {
    row(m, TINY_T_POOL_WALL, 1)[1] = INFINITY;
}
static void pool_weight_nan(struct tiny *m) {
    row(m, TINY_T_POOL_OCC_FRONT, 0)[2] = NAN;
}
static void pool_weight_inf(struct tiny *m) {
    row(m, TINY_T_POOL_ON, 2)[2] = INFINITY;
}
static void pool_unit_fraction(struct tiny *m) {
    row(m, TINY_T_POOL_ON, 2)[0] = 0.5f;
}
static void pool_unit_nan(struct tiny *m) {
    row(m, TINY_T_POOL_OFF, 0)[0] = NAN;
}
static void pool_unit_negative(struct tiny *m) {
    row(m, TINY_T_POOL_OFF, 0)[0] = -1.0f;
}
static void pool_unit_huge(struct tiny *m) {
    row(m, TINY_T_POOL_OFF, 1)[0] = 16777216.0f; /* 2^24 */
}
static void pool_unit_from_one(struct tiny *m) {
    row(m, TINY_T_POOL_WALL, 0)[0] = 1.0f;
    row(m, TINY_T_POOL_WALL, 1)[0] = 1.0f;
}
static void pool_unit_gap(struct tiny *m) {
    row(m, TINY_T_POOL_ON, 2)[0] = 2.0f;
}
static void pool_rows_descending(struct tiny *m) {
    row(m, TINY_T_POOL_ON, 0)[1] = 10.0f;
    row(m, TINY_T_POOL_ON, 1)[1] = 9.0f;
}
static void pool_pixel_twice(struct tiny *m) {
    row(m, TINY_T_POOL_ON, 1)[1] = 9.0f;
}
static void pool_units_differ(struct tiny *m) {
    reshape(m, TINY_T_POOL_OFF, 2, 1, 3); /* off: one unit, on: two */
}
static void pool_tube_odd(struct tiny *m) {
    reshape(m, TINY_T_POOL_TUBE, 2, 1, 3);
}
static void pool_region_two_units(struct tiny *m) {
    row(m, TINY_T_POOL_OCC_FRONT, 1)[0] = 1.0f;
}
static void pool_two_columns(struct tiny *m) {
    reshape(m, TINY_T_POOL_WALL, 2, 3, 2);
}
static void pool_rank_one(struct tiny *m) {
    reshape(m, TINY_T_POOL_WALL, 1, 6, 0);
}

static const struct {
    const char *name;
    void (*break_it)(struct tiny *);
    enum geist_status want;
    const char       *says; /* part of the message */
} CASES[] = {
        {"architecture llama", arch_llama, GEIST_E_UNSUPPORTED, "general.architecture"},
        {"architecture missing", arch_missing, GEIST_E_UNSUPPORTED, "general.architecture"},
        {"architecture decknet2", arch_prefix, GEIST_E_UNSUPPORTED, "general.architecture"},
        {"slots missing", slots_missing, GEIST_E_FORMAT, "decknet.slots"},
        {"slots empty", slots_empty, GEIST_E_FORMAT, "decknet.slots"},
        {"slots with a blank", slots_blank, GEIST_E_FORMAT, "decknet.slots[1]"},
        {"three slots, homographies for two", slots_three, GEIST_E_FORMAT, "decknet.homography"},
        {"image one pixel wide", image_one_wide, GEIST_E_FORMAT, "decknet.image.width"},
        {"image 40000 high", image_huge, GEIST_E_FORMAT, "decknet.image.height"},
        {"crop width 0", crop_zero, GEIST_E_FORMAT, "decknet.crop.width"},
        {"crop of 2^24 pixels", crop_huge, GEIST_E_FORMAT, "decknet.crop"},
        {"strip height 0", bg_zero, GEIST_E_FORMAT, "decknet.bg.height"},
        {"image.width missing", width_missing, GEIST_E_FORMAT, "decknet.image.width"},
        {"crop.height a string", crop_height_string, GEIST_E_FORMAT, "decknet.crop.height"},
        {"t_white missing", scalar_missing, GEIST_E_FORMAT, "decknet.t_white"},
        {"g_unit a string", scalar_string, GEIST_E_FORMAT, "decknet.g_unit"},
        {"t_tube NaN", scalar_nan, GEIST_E_FORMAT, "decknet.t_tube"},
        {"gain_max Inf", scalar_inf, GEIST_E_FORMAT, "decknet.gain_max"},
        {"12 tensors", tensor_dropped, GEIST_E_FORMAT, "12 tensors"},
        {"14 tensors", tensor_extra, GEIST_E_FORMAT, "14 tensors"},
        {"wall.w renamed", tensor_renamed, GEIST_E_FORMAT, "decknet.wall.w"},
        {"conv.blob twice, conv.tube missing", tensor_twice, GEIST_E_FORMAT, "decknet.conv.tube"},
        {"conv.blob F16", tensor_f16, GEIST_E_FORMAT, "decknet.conv.blob"},
        {"homography of rank 2", homography_rank, GEIST_E_FORMAT, "decknet.homography"},
        {"10 homographies", homography_short, GEIST_E_FORMAT, "decknet.homography"},
        {"homography NaN", homography_nan, GEIST_E_FORMAT, "decknet.homography[50]"},
        {"4 planes", planes_four, GEIST_E_FORMAT, "decknet.planes_mm"},
        {"bg_nominal -Inf", nominal_inf, GEIST_E_FORMAT, "decknet.bg_nominal[1]"},
        {"2 x 2 kernel", kernel_even, GEIST_E_FORMAT, "decknet.conv.blob"},
        {"3 x 1 kernel", kernel_oblong, GEIST_E_FORMAT, "decknet.conv.tube"},
        {"kernel NaN", kernel_nan, GEIST_E_FORMAT, "decknet.conv.tube[4]"},
        {"change.w five wide", mix_five_wide, GEIST_E_FORMAT, "decknet.change.w"},
        {"wall.w NaN", mix_nan, GEIST_E_FORMAT, "decknet.wall.w[7]"},
        {"pixel one past the crop", pool_pixel_past_end, GEIST_E_FORMAT, "pool.off row 1: pixel"},
        {"pixel 3e9", pool_pixel_far, GEIST_E_FORMAT, "pool.occ_rear row 1: pixel"},
        {"pixel -1", pool_pixel_negative, GEIST_E_FORMAT, "pool.on row 0: pixel"},
        {"pixel 10.5", pool_pixel_fraction, GEIST_E_FORMAT, "pool.on row 1: pixel"},
        {"pixel NaN", pool_pixel_nan, GEIST_E_FORMAT, "pool.tube row 0: pixel"},
        {"pixel Inf", pool_pixel_inf, GEIST_E_FORMAT, "pool.wall row 1: pixel"},
        {"weight NaN", pool_weight_nan, GEIST_E_FORMAT, "pool.occ_front row 0: weight"},
        {"weight Inf", pool_weight_inf, GEIST_E_FORMAT, "pool.on row 2: weight"},
        {"unit 0.5", pool_unit_fraction, GEIST_E_FORMAT, "pool.on row 2: unit"},
        {"unit NaN", pool_unit_nan, GEIST_E_FORMAT, "pool.off row 0: unit"},
        {"unit -1", pool_unit_negative, GEIST_E_FORMAT, "pool.off row 0: unit"},
        {"unit 2^24", pool_unit_huge, GEIST_E_FORMAT, "pool.off row 1: unit"},
        {"units from 1", pool_unit_from_one, GEIST_E_FORMAT, "pool.wall row 0: rows"},
        {"unit 1 left out", pool_unit_gap, GEIST_E_FORMAT, "pool.on row 2: rows"},
        {"pixels descending", pool_rows_descending, GEIST_E_FORMAT, "pool.on row 1: rows"},
        {"pixel twice in a unit", pool_pixel_twice, GEIST_E_FORMAT, "pool.on row 1: rows"},
        {"on and off differ in units", pool_units_differ, GEIST_E_FORMAT, "decknet.pool.off"},
        {"tube pool of one unit", pool_tube_odd, GEIST_E_FORMAT, "decknet.pool.tube"},
        {"region pool of two units", pool_region_two_units, GEIST_E_FORMAT, "one unit each"},
        {"pool rows of two values", pool_two_columns, GEIST_E_FORMAT, "decknet.pool.wall"},
        {"pool of rank 1", pool_rank_one, GEIST_E_FORMAT, "decknet.pool.wall"},
};

static void test_cases(void) {
    for (size_t i = 0; i < sizeof CASES / sizeof CASES[0]; i++) {
        struct tiny m;
        tiny_default(&m);
        CASES[i].break_it(&m);
        struct geist_decknet   *d   = (struct geist_decknet *) (uintptr_t) 1; /* must be cleared */
        const enum geist_status st  = load(&m, &d);
        const char             *why = geist_last_create_error();
        if (st != CASES[i].want || d != nullptr || why == nullptr ||
            strstr(why, CASES[i].says) == nullptr) {
            fprintf(stderr,
                    "FAIL case \"%s\": status %s (want %s), handle %s, message \"%s\" (want "
                    "\"%s\" in it)\n",
                    CASES[i].name,
                    geist_status_to_string(st),
                    geist_status_to_string(CASES[i].want),
                    d == nullptr ? "null" : "NOT null",
                    why != nullptr ? why : "(none)",
                    CASES[i].says);
            g_fail = 1;
            if (st == GEIST_OK) {
                geist_decknet_destroy(d);
            }
        }
    }
    printf("%zu malformed files rejected with the expected status and message\n",
           sizeof CASES / sizeof CASES[0]);
}

/* ---- the valid file ---------------------------------------------------- */

static void fill(uint8_t *img, uint8_t v) {
    memset(img, v, TINY_IMG_W * TINY_IMG_H * 3);
}
static void set_px(uint8_t *img, size_t y, size_t x, uint8_t v) {
    memset(img + (y * TINY_IMG_W + x) * 3, v, 3);
}
static bool close_to(float got, double want) {
    return fabs((double) got - want) <= 1e-6;
}

/* Units computed by hand. Every homography maps the crop onto the top-left
 * 8 x 6 pixels of the image; bg_nominal is the worktable grey of both
 * images, so both gains are exactly 1. */
static void test_by_hand(void) {
    struct tiny m;
    tiny_default(&m);
    const float grey = 128.0f / 255.0f;
    for (int c = 0; c < 3; c++) {
        m.t[TINY_T_BG_NOMINAL].v[c] = grey;
    }
    /* The tube kernel is a single 1 at tap (0, 0): as a convolution it reads
     * the pixel one down and one right; as a correlation, one up and one
     * left. */
    memset(m.t[TINY_T_TUBE].v, 0, 9 * sizeof(float));
    m.t[TINY_T_TUBE].v[0]    = 1.0f;
    m.scalars[TINY_S_G_UNIT] = 0.25f;

    struct geist_decknet *d = nullptr;
    CHECK(load(&m, &d) == GEIST_OK);
    if (d == nullptr) {
        fprintf(stderr, "  %s\n", geist_last_create_error());
        return;
    }
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    CHECK(strcmp(info.slots, "AB") == 0 && info.n_slots == 2);
    CHECK(info.image_width == TINY_IMG_W && info.image_height == TINY_IMG_H);
    CHECK(info.n_planes == 5 && info.n_homographies == 11);
    CHECK(info.n_tips == 2 && info.n_tubes == 1);
    CHECK(info.planes_mm[1] == 14.4f && info.planes_mm[4] == 60.0f);
    CHECK(info.occ == 0.25f && info.smooth == 0.3f);

    static uint8_t cur[TINY_IMG_W * TINY_IMG_H * 3], ref[TINY_IMG_W * TINY_IMG_H * 3];
    fill(cur, 128);
    fill(ref, 128);
    /* All outside the strip (columns 0..3 of rows 0..4), so it stays grey. */
    set_px(cur, 3, 4, 204); /* 0.8: what tube unit 0 (pixel 19 = row 2, column 3) reads */
    set_px(cur, 5, 1, 51);  /* 0.2: tube unit 1 (pixel 40 = row 5, column 0), row clamped */
    set_px(cur, 0, 4, 179); /* +0.2 per channel on occ_front's second pixel */

    struct geist_decknet_slot slot;
    CHECK(geist_decknet_slot(d, 0, &slot) == GEIST_E_INVALID_STATE); /* nothing ran yet */
    CHECK(slot.tips == nullptr);

    struct geist_decknet_result res;
    CHECK(geist_decknet_run(d, sizeof cur, cur, ref, &res) == GEIST_OK);
    CHECK(res.checkable);
    CHECK(res.gain[0] == 1.0f && res.gain[1] == 1.0f && res.gain[2] == 1.0f);
    CHECK(res.bg_changed == 0.0f);
    for (size_t s = 0; s < 2; s++) { /* both slots see the same crop */
        CHECK(geist_decknet_slot(d, s, &slot) == GEIST_OK);
        CHECK(close_to(slot.tubes[0], 0.25 * 3 * 0.8));     /* 0.6 */
        CHECK(close_to(slot.tubes_rot[0], 0.25 * 3 * 0.2)); /* 0.15 */
        CHECK(close_to(slot.tube_n, 0.6) && close_to(slot.tube_r, 0.15));
        CHECK(close_to(slot.occ_front, 0.5 * 3 * 0.2)); /* weight 0.5, |delta| summed over RGB */
        CHECK(slot.occ_rear == 0.0f);
        if (g_fail) {
            fprintf(stderr,
                    "  slot %zu: tubes %.7f rot %.7f occ_front %.7f\n",
                    s,
                    (double) slot.tubes[0],
                    (double) slot.tubes_rot[0],
                    (double) slot.occ_front);
        }
    }
    CHECK(geist_decknet_slot(d, 2, &slot) == GEIST_E_INVALID_ARG);

    /* Homographies that divide by zero everywhere sample pixel (0, 0): a
     * defined result, not a crash (run under UBSan in the asan mode). */
    float zero[11 * 9] = {0};
    CHECK(geist_decknet_set_homographies(d, 11, GEIST_DECKNET_IMAGE_CURRENT, zero) == GEIST_OK);
    CHECK(geist_decknet_run(d, sizeof cur, cur, ref, &res) == GEIST_OK);
    CHECK(res.checkable && res.gain[0] == 1.0f);
    float huge[11 * 9];
    for (size_t i = 0; i < 99; i++) {
        huge[i] = (i % 2 != 0) ? 3e38f : -3e38f;
    }
    CHECK(geist_decknet_set_homographies(d, 11, GEIST_DECKNET_IMAGE_EMPTY, huge) == GEIST_OK);
    CHECK(geist_decknet_run(d, sizeof cur, cur, ref, &res) == GEIST_OK);

    /* A current image at half the brightness is not checkable, and then
     * there is no slot state to read. */
    float       one[11 * 9];
    const float id[9] = {1, 0, 0.5f, 0, 1, 0.5f, 0, 0, 1};
    for (size_t i = 0; i < 11; i++) {
        memcpy(one + i * 9, id, sizeof id);
    }
    CHECK(geist_decknet_set_homographies(d, 11, GEIST_DECKNET_IMAGE_CURRENT, one) == GEIST_OK);
    CHECK(geist_decknet_set_homographies(d, 11, GEIST_DECKNET_IMAGE_EMPTY, one) == GEIST_OK);
    fill(cur, 64);
    CHECK(geist_decknet_run(d, sizeof cur, cur, ref, &res) == GEIST_OK);
    CHECK(!res.checkable && close_to(res.gain[0], 2.0));
    CHECK(geist_decknet_slot(d, 0, &slot) == GEIST_E_INVALID_STATE);
    CHECK(slot.tubes == nullptr && slot.tube_n == 0.0f);

    /* Argument checks. */
    CHECK(geist_decknet_run(d, sizeof cur - 1, cur, ref, &res) == GEIST_E_INVALID_ARG);
    CHECK(!res.checkable && res.gain[0] == 0.0f);
    CHECK(geist_decknet_run(d, sizeof cur, nullptr, ref, &res) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_run(d, sizeof cur, cur, nullptr, &res) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_run(d, sizeof cur, cur, ref, nullptr) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_run(nullptr, sizeof cur, cur, ref, &res) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_info(nullptr, &info) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_info(d, nullptr) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_slot(nullptr, 0, &slot) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_slot(d, 0, nullptr) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_set_homographies(d, 10, GEIST_DECKNET_IMAGE_CURRENT, one) ==
          GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_set_homographies(d, 11, (enum geist_decknet_image) 2, one) ==
          GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_set_homographies(d, 11, GEIST_DECKNET_IMAGE_CURRENT, nullptr) ==
          GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_set_homographies(nullptr, 11, GEIST_DECKNET_IMAGE_CURRENT, one) ==
          GEIST_E_INVALID_ARG);
    /* A NaN replaces nothing: the pair is as uncheckable as before, not
     * differently broken. */
    one[98] = NAN;
    CHECK(geist_decknet_set_homographies(d, 11, GEIST_DECKNET_IMAGE_CURRENT, one) ==
          GEIST_E_INVALID_ARG);
    fill(cur, 128);
    CHECK(geist_decknet_run(d, sizeof cur, cur, ref, &res) == GEIST_OK);
    CHECK(res.checkable);

    geist_decknet_destroy(d);
    geist_decknet_destroy(nullptr);
}

/* ---- truncation -------------------------------------------------------- */

static void test_truncated(void) {
    struct tiny m;
    tiny_default(&m);
    static struct tiny_buf buf;
    tiny_write(&m, &buf);
    /* Up to the last tensor's last value every byte is needed; the padding
     * after it is not. */
    size_t rejected = 0;
    for (size_t n = 1; n <= buf.n; n++) {
        uint8_t *copy = xmalloc(n);
        memcpy(copy, buf.b, n);
        struct geist_decknet   *d  = nullptr;
        const enum geist_status st = geist_decknet_load_from_memory(n, copy, &d);
        free(copy);
        const bool whole = n >= buf.n_data;
        if ((st == GEIST_OK) != whole || (d != nullptr) != whole) {
            fprintf(stderr,
                    "FAIL: the first %zu of %zu bytes: %s\n",
                    n,
                    buf.n_data,
                    geist_status_to_string(st));
            g_fail = 1;
        }
        rejected += st != GEIST_OK;
        geist_decknet_destroy(d);
    }
    CHECK(rejected == buf.n_data - 1);
    printf("%zu truncations of the %zu-byte file rejected\n", rejected, buf.n_data);
}

/* ---- allocation failure ------------------------------------------------ */

static void test_oom(void) {
    struct tiny m;
    tiny_default(&m);
    int64_t n = 0;
    for (;; n++) {
        heap_fail_after(n);
        struct geist_decknet   *d  = nullptr;
        const enum geist_status st = load(&m, &d);
        heap_fail_after(-1);
        if (st == GEIST_OK) {
            CHECK(d != nullptr);
            geist_decknet_destroy(d);
            break;
        }
        /* The GGUF reader's own allocation failing is a parse failure. */
        CHECK(d == nullptr && (st == GEIST_E_OOM || st == GEIST_E_FORMAT));
        if (n > 1000) {
            CHECK(!"a load never succeeded");
            break;
        }
    }
    CHECK(n > 20); /* the handle, its tensors and its workspace */
    printf("a load survives each of its %lld allocations failing\n", (long long) n);
}

/* ---- entry points ------------------------------------------------------ */

static void test_entry_points(void) {
    struct geist_decknet *d = (struct geist_decknet *) (uintptr_t) 1;
    CHECK(geist_decknet_load(nullptr, &d) == GEIST_E_INVALID_ARG && d == nullptr);
    CHECK(geist_decknet_load("", &d) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_load("x.gguf", nullptr) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_load("/nonexistent/decknet.gguf", &d) == GEIST_E_IO && d == nullptr);
    const uint8_t junk[8] = "notgguf";
    CHECK(geist_decknet_load_from_memory(sizeof junk, junk, &d) == GEIST_E_FORMAT && d == nullptr);
    CHECK(geist_decknet_load_from_memory(0, junk, &d) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_load_from_memory(8, nullptr, &d) == GEIST_E_INVALID_ARG);
    CHECK(geist_decknet_load_from_memory(8, junk, nullptr) == GEIST_E_INVALID_ARG);

    /* The same file through the path loader (mmap). */
    struct tiny m;
    tiny_default(&m);
    static struct tiny_buf buf;
    tiny_write(&m, &buf);
    char path[] = "/tmp/geist_decknet_XXXXXX";
    int  fd     = mkstemp(path);
    if (fd < 0) {
        fprintf(stderr, "mkstemp failed\n");
        exit(GEIST_TEST_ERROR);
    }
    CHECK(write(fd, buf.b, buf.n) == (ssize_t) buf.n);
    close(fd);
    CHECK(geist_decknet_load(path, &d) == GEIST_OK && d != nullptr);
    geist_decknet_destroy(d);
    /* Cut short on disk: the GGUF reader refuses it, which the path loader
     * reports as GEIST_E_IO like geist_model_load. */
    CHECK(truncate(path, (off_t) (buf.n - 40)) == 0);
    CHECK(geist_decknet_load(path, &d) == GEIST_E_IO && d == nullptr);
    unlink(path);
}

int main(void) {
    test_by_hand();
    test_cases();
    test_truncated();
    test_oom();
    test_entry_points();
    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("decknet loader: fails closed\n");
    return GEIST_TEST_PASS;
}
