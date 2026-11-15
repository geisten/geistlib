/* tests/test_decknet_unit.c — decknet against the numpy reference (#532).
 *
 * Two fixtures of the same four files: a network, an image pair (cur.ppm,
 * ref.ppm) and what geistbox's decknet.py computes for them (expected.txt).
 *
 *   tests/data/decknet_synth/   always. A network whose every number is
 *       invented (tests/scripts/decknet_synth_fixture.py: three slots, a
 *       160 x 120 image, seeded random kernels, weights, pooling rows and
 *       homographies) on an image pair of noise and shapes. It describes no
 *       device; it is the arithmetic of the loader and the pass against the
 *       reference, on every platform.
 *   $GEIST_DECKNET_FIXTURE/     where it exists. The exported network of
 *       the device, in the same four-file layout. It holds the deck
 *       geometry, which is confidential, so it is not in the repository;
 *       this part is left out on a host without it.
 *
 * Each case of expected.txt runs through the public API on a fresh handle
 * and every output is compared:
 *
 *   file    the file's homographies for both images
 *   moved   other homographies for the current image
 *           (geist_decknet_set_homographies)
 *   dark    the current image at half brightness: not checkable
 *
 * The tolerance is the issue's 1e-4; the measured deviation is printed and
 * is around 1e-6. decknet has one implementation, portable C that no
 * backend dispatches, so this binary is the parity check for "cpu_scalar
 * and the default CPU backend" of whatever platform builds it.
 *
 * Also here, per fixture: no call allocates, the first included
 * (heap_alloc_count), and geist_model_load still refuses the file.
 */
#include "heap.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_decknet.h>

#include <math.h>

constexpr double TOLERANCE = 1e-4;
constexpr size_t N_SCALARS = 8;
constexpr size_t MAX_CASES = 8;

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

struct expect {
    char    name[32];
    float  *homographies; /* for the current image, or nullptr */
    bool    cur_half;
    bool    checkable;
    double  gain[3];
    double  bg_changed;
    double *slots; /* n_slots x per_slot, or nullptr */
};

static double read_number(FILE *f, const char *path) {
    double v = 0.0;
    if (fscanf(f, "%lf", &v) != 1) {
        broken("number expected", path);
    }
    return v;
}

/* expected.txt: `case <name>` opens a case; then `homographies <n * 9>`,
 * `cur_half 1`, `checkable <0|1>`, `gain <3>`, `bg_changed <1>` and one
 * `slot <letter> <per_slot numbers>` per slot. `#` starts a comment line. */
static size_t read_expected(const char                      *path,
                            const struct geist_decknet_info *info,
                            struct expect                    cases[static MAX_CASES]) {
    FILE *f = fopen(path, "r");
    if (f == nullptr) {
        broken("cannot open", path);
    }
    const size_t   per_slot = N_SCALARS + info->n_tips + 2 * info->n_tubes;
    size_t         n        = 0;
    struct expect *e        = nullptr;
    size_t         slot     = 0;
    char           word[64];
    while (fscanf(f, "%63s", word) == 1) {
        if (word[0] == '#') {
            int c;
            while ((c = fgetc(f)) != EOF && c != '\n') {
            }
        } else if (strcmp(word, "case") == 0) {
            if (n == MAX_CASES) {
                broken("too many cases", path);
            }
            e    = &cases[n++];
            *e   = (struct expect) {0};
            slot = 0;
            if (fscanf(f, "%31s", e->name) != 1) {
                broken("case name expected", path);
            }
        } else if (e == nullptr) {
            broken("a line before the first case", path);
        } else if (strcmp(word, "homographies") == 0) {
            e->homographies = xmalloc(info->n_homographies * 9 * sizeof(float));
            for (size_t i = 0; i < info->n_homographies * 9; i++) {
                e->homographies[i] = (float) read_number(f, path);
            }
        } else if (strcmp(word, "cur_half") == 0) {
            e->cur_half = read_number(f, path) != 0.0;
        } else if (strcmp(word, "checkable") == 0) {
            e->checkable = read_number(f, path) != 0.0;
        } else if (strcmp(word, "gain") == 0) {
            for (int c = 0; c < 3; c++) {
                e->gain[c] = read_number(f, path);
            }
        } else if (strcmp(word, "bg_changed") == 0) {
            e->bg_changed = read_number(f, path);
        } else if (strcmp(word, "slot") == 0) {
            char letter[8];
            if (fscanf(f, "%7s", letter) != 1 || slot >= info->n_slots ||
                letter[0] != info->slots[slot] || letter[1] != '\0') {
                broken("slots out of the file's order", path);
            }
            if (e->slots == nullptr) {
                e->slots = xmalloc(info->n_slots * per_slot * sizeof(double));
            }
            for (size_t i = 0; i < per_slot; i++) {
                e->slots[slot * per_slot + i] = read_number(f, path);
            }
            slot++;
        } else {
            broken("unknown line", path);
        }
    }
    fclose(f);
    return n;
}

/* The largest |got - want| of one case; counts what was compared. */
static double compare_case(const char                      *gguf,
                           const struct geist_decknet_info *info,
                           const struct expect             *e,
                           const uint8_t                   *cur,
                           const uint8_t                   *ref,
                           size_t                          *n_compared) {
    const size_t          n_bytes = info->image_width * info->image_height * 3;
    struct geist_decknet *d       = nullptr;
    if (geist_decknet_load(gguf, &d) != GEIST_OK) {
        broken(geist_last_create_error(), gguf);
    }
    if (e->homographies != nullptr) {
        CHECK(geist_decknet_set_homographies(
                      d, info->n_homographies, GEIST_DECKNET_IMAGE_CURRENT, e->homographies) ==
              GEIST_OK);
    }
    uint8_t *half = nullptr;
    if (e->cur_half) {
        half = xmalloc(n_bytes);
        for (size_t i = 0; i < n_bytes; i++) {
            half[i] = cur[i] >> 1;
        }
        cur = half;
    }
    /* Twice: the second pass over the same pair must not see the first. */
    double worst = 0.0;
    for (int pass = 0; pass < 2; pass++) {
        struct geist_decknet_result res;
        CHECK(geist_decknet_run(d, n_bytes, cur, ref, &res) == GEIST_OK);
        CHECK(res.checkable == e->checkable);
        worst = fmax(worst, fabs((double) res.bg_changed - e->bg_changed));
        for (int c = 0; c < 3; c++) {
            worst = fmax(worst, fabs((double) res.gain[c] - e->gain[c]));
        }
        *n_compared = 4;
        for (size_t s = 0; s < info->n_slots; s++) {
            struct geist_decknet_slot slot;
            const enum geist_status   st = geist_decknet_slot(d, s, &slot);
            if (!e->checkable) {
                CHECK(st == GEIST_E_INVALID_STATE);
                continue;
            }
            CHECK(st == GEIST_OK && e->slots != nullptr);
            if (st != GEIST_OK || e->slots == nullptr) {
                continue;
            }
            const size_t  per_slot           = N_SCALARS + info->n_tips + 2 * info->n_tubes;
            const double *want               = e->slots + s * per_slot;
            const float   scalars[N_SCALARS] = {slot.occ_front,
                                                slot.occ_rear,
                                                slot.wall_front,
                                                slot.rack,
                                                slot.plate,
                                                slot.deep,
                                                slot.tube_n,
                                                slot.tube_r};
            for (size_t i = 0; i < per_slot; i++) {
                const size_t t   = i - N_SCALARS;
                const float  got = i < N_SCALARS      ? scalars[i]
                                   : t < info->n_tips ? slot.tips[t]
                                   : t < info->n_tips + info->n_tubes
                                           ? slot.tubes[t - info->n_tips]
                                           : slot.tubes_rot[t - info->n_tips - info->n_tubes];
                const double dev = fabs((double) got - want[i]);
                if (!(dev <= TOLERANCE)) {
                    fprintf(stderr,
                            "FAIL case %s slot %c output %zu: got %.9g, reference %.9g\n",
                            e->name,
                            info->slots[s],
                            i,
                            (double) got,
                            want[i]);
                    g_fail = 1;
                }
                worst = fmax(worst, dev);
            }
            *n_compared += per_slot;
        }
    }
    CHECK(worst <= TOLERANCE);
    free(half);
    geist_decknet_destroy(d);
    return worst;
}

/* "No allocation per call after the first" (#532), and none in the first:
 * heap.h counts every allocation the library makes. */
static void test_no_alloc(const char *gguf, const uint8_t *cur, const uint8_t *ref, size_t n) {
    struct geist_decknet *d = nullptr;
    if (geist_decknet_load(gguf, &d) != GEIST_OK) {
        broken(geist_last_create_error(), gguf);
    }
    const uint64_t before = heap_alloc_count();
    for (int i = 0; i < 3; i++) {
        struct geist_decknet_result res;
        struct geist_decknet_slot   slot;
        CHECK(geist_decknet_run(d, n, cur, ref, &res) == GEIST_OK);
        CHECK(geist_decknet_slot(d, 0, &slot) ==
              (res.checkable ? GEIST_OK : GEIST_E_INVALID_STATE));
        const uint64_t now = heap_alloc_count();
        if (now != before) {
            fprintf(stderr,
                    "FAIL: call %d made %llu allocations\n",
                    i + 1,
                    (unsigned long long) (now - before));
            g_fail = 1;
        }
    }
    /* A pair that is not checkable takes the other exit. */
    uint8_t *half = xmalloc(n); /* malloc, not heap.h: not counted */
    for (size_t i = 0; i < n; i++) {
        half[i] = cur[i] >> 1;
    }
    struct geist_decknet_result res;
    CHECK(geist_decknet_run(d, n, half, ref, &res) == GEIST_OK && !res.checkable);
    CHECK(heap_alloc_count() == before);
    free(half);
    geist_decknet_destroy(d);
    printf("no allocation in 4 calls, the first included\n");
}

/* The token API has no use for the file and says so. */
static void test_model_load_refuses(const char *gguf) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK) {
        printf("geist_model_load not tried: no cpu_scalar backend in this build\n");
        return;
    }
    struct geist_model *m = nullptr;
    CHECK(geist_model_load(gguf, be, &m) == GEIST_E_UNSUPPORTED && m == nullptr);
    geist_backend_destroy(be);
}

/* One fixture directory: `gguf_name`, cur.ppm, ref.ppm and expected.txt.
 * Returns the largest deviation from the reference over all its cases. */
static double run_fixture(const char *dir, const char *gguf_name) {
    char gguf[1024], cur_path[1024], ref_path[1024], expected[1024];
    snprintf(gguf, sizeof gguf, "%s/%s", dir, gguf_name);
    snprintf(cur_path, sizeof cur_path, "%s/cur.ppm", dir);
    snprintf(ref_path, sizeof ref_path, "%s/ref.ppm", dir);
    snprintf(expected, sizeof expected, "%s/expected.txt", dir);

    struct geist_decknet *d = nullptr;
    if (geist_decknet_load(gguf, &d) != GEIST_OK) {
        broken(geist_last_create_error(), gguf);
    }
    struct geist_decknet_info info;
    CHECK(geist_decknet_info(d, &info) == GEIST_OK);
    printf("%s: slots %s, image %zu x %zu, %zu homographies, %zu tips and 2 x %zu tubes per slot\n",
           gguf,
           info.slots,
           info.image_width,
           info.image_height,
           info.n_homographies,
           info.n_tips,
           info.n_tubes);

    uint8_t             *cur     = read_ppm(cur_path, info.image_width, info.image_height);
    uint8_t             *ref     = read_ppm(ref_path, info.image_width, info.image_height);
    const size_t         n_bytes = info.image_width * info.image_height * 3;
    static struct expect cases[MAX_CASES];
    const size_t         n_cases = read_expected(expected, &info, cases);
    CHECK(n_cases > 0);

    double worst = 0.0;
    for (size_t i = 0; i < n_cases; i++) {
        size_t       n   = 0;
        const double dev = compare_case(gguf, &info, &cases[i], cur, ref, &n);
        printf("case %-6s %s, %4zu outputs, max |C - numpy| = %.3g\n",
               cases[i].name,
               cases[i].checkable ? "checkable    " : "not checkable",
               n,
               dev);
        worst = fmax(worst, dev);
        free(cases[i].homographies);
        free(cases[i].slots);
    }
    test_no_alloc(gguf, cur, ref, n_bytes);
    test_model_load_refuses(gguf);

    geist_decknet_destroy(d);
    free(cur);
    free(ref);
    return worst;
}

int main(void) {
    /* In the repository: its absence is a failure, not a skip. */
    const double synth = run_fixture("tests/data/decknet_synth", "synth.gguf");

    /* On the hosts that have it. A directory that is named and has no
     * network in it is a host without the fixture; one that has a network
     * that does not load or match is a failure. */
    const char *dir      = getenv("GEIST_DECKNET_FIXTURE");
    bool        have     = false;
    double      geometry = 0.0;
    if (dir != nullptr && dir[0] != '\0') {
        char gguf[1024];
        snprintf(gguf, sizeof gguf, "%s/decknet.gguf", dir);
        FILE *probe = fopen(gguf, "rb");
        if (probe != nullptr) {
            fclose(probe);
            have     = true;
            geometry = run_fixture(dir, "decknet.gguf");
        }
    }
    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("decknet matches the numpy reference (tolerance %.0e): invented network %.3g",
           TOLERANCE,
           synth);
    if (have) {
        printf(", GEIST_DECKNET_FIXTURE %.3g\n", geometry);
    } else {
        printf("; GEIST_DECKNET_FIXTURE not set or not on this host, the device network was not "
               "run\n");
    }
    return GEIST_TEST_PASS;
}
