/*
 * test_hadamard_unit — fused->hadamard_rotate, the activation transform
 * for GGUFs whose weights were folded into a rotated basis
 * (prism.hadamard.*, e.g. Ternary-Bonsai-2-27B).
 *
 * Every check runs through the backend slot on each CPU backend compiled
 * in, not through the host helper, so a wiring mistake in a backend's
 * tensor glue fails here too:
 *
 *   1. naive reference: blockwise product with an explicit Sylvester
 *      matrix, accumulated in double;
 *   2. orthonormality: forward twice is the identity, norms are kept;
 *   3. shapes: widths 1024 / 5120 / 6144 / 17408, rows 1 / 3 / 33 / 256,
 *      in place bit-identical to out of place; malformed geometry is
 *      rejected and leaves y untouched;
 *   4. signs and permutation: explicit signs against the formula, the
 *      grouped-value permutation against a hand-written index map, and
 *      the order permutation -> signs -> H;
 *   5. fold equivalence: W x == rotate(W) rotate(x), the identity that
 *      makes a folded checkpoint compute the original function;
 *   6. inverse: inverse(forward(x)) == x, the embedding-lookup path;
 *   7. the CPU backends agree bit for bit.
 *
 * Metal joins as a third fixture when compiled in (BACKENDS=metal ...).
 *
 * Deterministic, no model needed.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_seed = 0x2545F491u;

static float frand(void) {
    g_seed = g_seed * 1103515245u + 12345u;
    return ((float) ((g_seed >> 8) & 0xffffff) / (float) 0x1000000) * 2.0f - 1.0f;
}

static void fill_random(size_t n, float a[static n]) {
    for (size_t i = 0; i < n; i++) {
        a[i] = frand();
    }
}

static void fill_signs(size_t n, float a[static n]) {
    for (size_t i = 0; i < n; i++) {
        a[i] = frand() < 0.0f ? -1.0f : 1.0f;
    }
}

/* max |a - b| / max(max |b|, 1e-30). */
static double rel_err(size_t n, const float a[static n], const float b[static n]) {
    double d = 0.0, m = 0.0;
    for (size_t i = 0; i < n; i++) {
        d = fmax(d, fabs((double) a[i] - (double) b[i]));
        m = fmax(m, fabs((double) b[i]));
    }
    return d / fmax(m, 1e-30);
}

static double norm2(size_t n, const float a[static n]) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        s += (double) a[i] * (double) a[i];
    }
    return sqrt(s);
}

/* ---- References --------------------------------------------------------- */

/* P(x)[d + hd*(r + rep*k)] = x[d + hd*(k + nk*r)], written as the loop over
 * the OUTPUT index so it is independent of the implementation's loop. */
static void ref_permute(size_t hd, size_t nk, size_t rep, const float *x, float *y) {
    const size_t width = hd * nk * rep;
    for (size_t j = 0; j < width; j++) {
        const size_t d = j % hd;
        const size_t g = j / hd; /* g = r + rep*k */
        const size_t r = g % rep;
        const size_t k = g / rep;
        y[j]           = x[d + hd * (k + nk * r)];
    }
}

/* y = H_block(v) with an explicit Sylvester matrix, in double. */
static void ref_hadamard(size_t width, size_t block, const float *v, float *y) {
    const double s = 1.0 / sqrt((double) block);
    for (size_t b = 0; b < width; b += block) {
        for (size_t i = 0; i < block; i++) {
            double acc = 0.0;
            for (size_t j = 0; j < block; j++) {
                const int sign = __builtin_popcountll((unsigned long long) (i & j)) & 1;
                acc += sign ? -(double) v[b + j] : (double) v[b + j];
            }
            y[b + i] = (float) (acc * s);
        }
    }
}

/* Forward reference: y = H(signs * P(x)). */
static void ref_forward(size_t       width,
                        size_t       block,
                        size_t       hd,
                        size_t       nk,
                        size_t       rep,
                        const float *x,
                        const float *signs,
                        float       *y) {
    float *t = xmalloc(width * sizeof(float));
    if (rep > 1) {
        ref_permute(hd, nk, rep, x, t);
    } else {
        memcpy(t, x, width * sizeof(float));
    }
    if (signs != nullptr) {
        for (size_t i = 0; i < width; i++) {
            t[i] *= signs[i];
        }
    }
    ref_hadamard(width, block, t, y);
    free(t);
}

/* ---- Backend plumbing --------------------------------------------------- */

struct fixture {
    struct geist_backend *be;
    const char           *name;
};

static struct geist_buffer *make_buf(struct geist_backend *be, size_t n, const float *init) {
    struct geist_buffer *b = nullptr;
    if (be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &b) !=
        GEIST_OK) {
        return nullptr;
    }
    if (init != nullptr) {
        memcpy(be->desc->vtbl->buffer_map(b), init, n * sizeof(float));
        be->desc->vtbl->buffer_unmap(b);
    }
    return b;
}

static struct geist_tensor view(struct geist_buffer *b, size_t rows, size_t width) {
    return (struct geist_tensor) {
            .buffer = b,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 2,
            .shape  = {(int64_t) rows, (int64_t) width},
            .stride = {(int64_t) width, 1},
    };
}

/* Run the slot. x is copied in; the result lands in y (n = rows*width).
 * in_place runs with y aliasing x. signs_len lets a test pass a
 * wrong-length sign vector. */
static enum geist_status run(const struct fixture *f,
                             size_t                rows,
                             size_t                width,
                             size_t                block,
                             size_t                hd,
                             size_t                nk,
                             size_t                rep,
                             bool                  inverse,
                             bool                  in_place,
                             size_t                signs_len,
                             const float          *x,
                             const float          *signs,
                             float                *y) {
    struct geist_backend *be = f->be;
    const size_t          n  = rows * width;
    struct geist_buffer  *bx = make_buf(be, n, x);
    struct geist_buffer  *by = in_place ? bx : make_buf(be, n, y);
    struct geist_buffer  *bs = signs != nullptr ? make_buf(be, signs_len, signs) : nullptr;
    if (bx == nullptr || by == nullptr || (signs != nullptr && bs == nullptr)) {
        return GEIST_E_OOM;
    }
    struct geist_tensor tx = view(bx, rows, width);
    struct geist_tensor ty = view(by, rows, width);
    struct geist_tensor ts = {
            .buffer = bs,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 1,
            .shape  = {(int64_t) signs_len},
            .stride = {1},
    };
    const struct geist_hadamard_args args = {
            .x        = &tx,
            .signs    = signs != nullptr ? &ts : nullptr,
            .y        = &ty,
            .block    = block,
            .perm_hd  = hd,
            .perm_nk  = nk,
            .perm_rep = rep,
            .inverse  = inverse,
    };
    const enum geist_status s = geist_backend_fused_tbl(be)->hadamard_rotate(be, &args);
    memcpy(y, be->desc->vtbl->buffer_map(by), n * sizeof(float));
    be->desc->vtbl->buffer_unmap(by);
    be->desc->vtbl->buffer_destroy(be, bx);
    if (!in_place) {
        be->desc->vtbl->buffer_destroy(be, by);
    }
    if (bs != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bs);
    }
    return s;
}

static enum geist_status fwd(const struct fixture *f,
                             size_t                rows,
                             size_t                width,
                             const float          *x,
                             const float          *signs,
                             float                *y) {
    return run(f, rows, width, 1024, 0, 0, 0, false, false, width, x, signs, y);
}

/* ---- Checks ------------------------------------------------------------- */

static int check_naive(const struct fixture *f) {
    int          fails = 0;
    const size_t rows = 3, width = 2048;
    float       *x   = xmalloc(rows * width * sizeof(float));
    float       *y   = xmalloc(rows * width * sizeof(float));
    float       *ref = xmalloc(width * sizeof(float));
    fill_random(rows * width, x);
    fails += geist_expect(fwd(f, rows, width, x, nullptr, y) == GEIST_OK, "naive: status");
    for (size_t r = 0; r < rows; r++) {
        ref_forward(width, 1024, 0, 0, 0, x + r * width, nullptr, ref);
        fails += geist_expect(rel_err(width, y + r * width, ref) < 1e-5, "naive: matches H*x");
    }
    free(x);
    free(y);
    free(ref);
    return fails;
}

static int check_orthonormal(const struct fixture *f) {
    int          fails = 0;
    const size_t width = 5120;
    float       *x     = xmalloc(width * sizeof(float));
    float       *y     = xmalloc(width * sizeof(float));
    float       *z     = xmalloc(width * sizeof(float));
    fill_random(width, x);
    fails += geist_expect(fwd(f, 1, width, x, nullptr, y) == GEIST_OK, "ortho: status 1");
    fails += geist_expect(fwd(f, 1, width, y, nullptr, z) == GEIST_OK, "ortho: status 2");
    fails += geist_expect(rel_err(width, z, x) < 1e-5, "ortho: H(H(x)) == x");
    fails += geist_expect(fabs(norm2(width, y) / norm2(width, x) - 1.0) < 1e-5,
                          "ortho: norm preserved");
    free(x);
    free(y);
    free(z);
    return fails;
}

static int check_shapes(const struct fixture *f) {
    int          fails     = 0;
    const size_t widths[]  = {1024, 5120, 6144, 17408};
    const size_t rowset[]  = {1, 3, 33, 256};
    float       *x         = xmalloc(256 * 17408 * sizeof(float));
    float       *y_out     = xmalloc(256 * 17408 * sizeof(float));
    float       *y_in      = xmalloc(256 * 17408 * sizeof(float));
    float       *signs     = xmalloc(17408 * sizeof(float));
    float       *ref       = xmalloc(17408 * sizeof(float));
    char         what[128] = {0};
    for (size_t wi = 0; wi < sizeof widths / sizeof widths[0]; wi++) {
        const size_t width = widths[wi];
        fill_signs(width, signs);
        for (size_t ri = 0; ri < sizeof rowset / sizeof rowset[0]; ri++) {
            const size_t rows = rowset[ri];
            fill_random(rows * width, x);
            snprintf(what, sizeof what, "shapes: w=%zu m=%zu", width, rows);
            const enum geist_status so =
                    run(f, rows, width, 1024, 0, 0, 0, false, false, width, x, signs, y_out);
            const enum geist_status si =
                    run(f, rows, width, 1024, 0, 0, 0, false, true, width, x, signs, y_in);
            fails += geist_expect(so == GEIST_OK && si == GEIST_OK, what);
            fails += geist_expect(memcmp(y_out, y_in, rows * width * sizeof(float)) == 0,
                                  "shapes: in place bit-identical to out of place");
            /* Naive check on the first and last row: the Sylvester product
             * is O(width * block), too slow for every row of a 256-row run. */
            const size_t probe[] = {0, rows - 1};
            for (size_t p = 0; p < 2; p++) {
                ref_forward(width, 1024, 0, 0, 0, x + probe[p] * width, signs, ref);
                fails += geist_expect(rel_err(width, y_out + probe[p] * width, ref) < 1e-5, what);
            }
        }
    }
    free(x);
    free(y_out);
    free(y_in);
    free(signs);
    free(ref);
    return fails;
}

/* Every malformed request fails with INVALID_ARG and leaves y as it was. */
static int check_rejects(const struct fixture *f) {
    int          fails = 0;
    const size_t width = 6144;
    float       *x     = xmalloc(width * sizeof(float));
    float       *y     = xmalloc(width * sizeof(float));
    float       *s     = xmalloc(width * sizeof(float));
    fill_random(width, x);
    fill_signs(width, s);
    const struct {
        const char *what;
        size_t      width, block, hd, nk, rep, signs_len;
        bool        inverse, in_place;
    } cases[] = {
            {"reject: width not a block multiple", 1000, 1024, 0, 0, 0, 1000, false, false},
            {"reject: block not a power of two", 6144, 768, 0, 0, 0, 6144, false, false},
            {"reject: block zero", 6144, 0, 0, 0, 0, 6144, false, false},
            {"reject: permutation on inverse", 6144, 1024, 128, 16, 3, 6144, true, false},
            {"reject: permutation in place", 6144, 1024, 128, 16, 3, 6144, false, true},
            {"reject: permutation geometry", 6144, 1024, 128, 16, 2, 6144, false, false},
            {"reject: signs length", 6144, 1024, 0, 0, 0, 5120, false, false},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        for (size_t j = 0; j < width; j++) {
            y[j] = 42.0f;
        }
        const enum geist_status st = run(f,
                                         1,
                                         cases[i].width,
                                         cases[i].block,
                                         cases[i].hd,
                                         cases[i].nk,
                                         cases[i].rep,
                                         cases[i].inverse,
                                         cases[i].in_place,
                                         cases[i].signs_len,
                                         x,
                                         s,
                                         y);
        fails += geist_expect(st == GEIST_E_INVALID_ARG, cases[i].what);
        if (!cases[i].in_place) {
            bool untouched = true;
            for (size_t j = 0; j < cases[i].width; j++) {
                untouched = untouched && y[j] == 42.0f;
            }
            fails += geist_expect(untouched, "reject: y untouched on failure");
        }
    }
    free(x);
    free(y);
    free(s);
    return fails;
}

static int check_signs_and_perm(const struct fixture *f) {
    int          fails = 0;
    const size_t hd = 128, nk = 16, rep = 3, width = hd * nk * rep; /* 27B ssm_out */
    float       *x   = xmalloc(width * sizeof(float));
    float       *s   = xmalloc(width * sizeof(float));
    float       *y   = xmalloc(width * sizeof(float));
    float       *y1  = xmalloc(width * sizeof(float));
    float       *ref = xmalloc(width * sizeof(float));
    float       *alt = xmalloc(width * sizeof(float));
    fill_random(width, x);
    fill_signs(width, s);

    /* All-plus signs are the identity sign vector, bit for bit. */
    for (size_t i = 0; i < width; i++) {
        y1[i] = 1.0f;
    }
    float *ones = xmalloc(width * sizeof(float));
    memcpy(ones, y1, width * sizeof(float));
    fails += geist_expect(fwd(f, 1, width, x, ones, y1) == GEIST_OK, "signs: +1 status");
    fails += geist_expect(fwd(f, 1, width, x, nullptr, y) == GEIST_OK, "signs: none status");
    fails += geist_expect(memcmp(y, y1, width * sizeof(float)) == 0,
                          "signs: all +1 bit-identical to no signs");

    /* Explicit signs. */
    fails += geist_expect(fwd(f, 1, width, x, s, y) == GEIST_OK, "signs: status");
    ref_forward(width, 1024, 0, 0, 0, x, s, ref);
    fails += geist_expect(rel_err(width, y, ref) < 1e-5, "signs: H(s*x)");

    /* Permutation alone, recovered exactly: undo H with a second forward
     * pass and compare to the hand-written index map. Inputs are distinct
     * integers, so every element's source is identifiable. */
    for (size_t i = 0; i < width; i++) {
        x[i] = (float) i;
    }
    fails += geist_expect(run(f, 1, width, 1024, hd, nk, rep, false, false, width, x, nullptr, y) ==
                                  GEIST_OK,
                          "perm: status");
    fails += geist_expect(fwd(f, 1, width, y, nullptr, y1) == GEIST_OK, "perm: undo status");
    ref_permute(hd, nk, rep, x, ref);
    bool exact = true;
    for (size_t i = 0; i < width; i++) {
        exact = exact && lrintf(y1[i]) == (long) ref[i];
    }
    fails += geist_expect(exact, "perm: tiled -> grouped index map");
    /* Spot values from the formula itself, not from ref_permute. */
    fails += geist_expect(lrintf(y1[0]) == 0, "perm: y[0] = x[0]");
    fails += geist_expect(lrintf(y1[hd]) == (long) (hd * nk), "perm: y[hd] = x[hd*nk] (r=1,k=0)");
    fails += geist_expect(lrintf(y1[hd * rep]) == (long) hd, "perm: y[hd*rep] = x[hd] (r=0,k=1)");

    /* Order: permutation, then signs, then H. Signs applied before the
     * permutation would give a different vector. */
    fill_random(width, x);
    fails += geist_expect(run(f, 1, width, 1024, hd, nk, rep, false, false, width, x, s, y) ==
                                  GEIST_OK,
                          "order: status");
    ref_forward(width, 1024, hd, nk, rep, x, s, ref);
    fails += geist_expect(rel_err(width, y, ref) < 1e-5, "order: H(s * P(x))");
    float *sx = xmalloc(width * sizeof(float));
    for (size_t i = 0; i < width; i++) {
        sx[i] = s[i] * x[i];
    }
    ref_forward(width, 1024, hd, nk, rep, sx, nullptr, alt);
    fails += geist_expect(rel_err(width, y, alt) > 1e-2, "order: differs from H(P(s * x))");

    free(x);
    free(s);
    free(y);
    free(y1);
    free(ref);
    free(alt);
    free(ones);
    free(sx);
    return fails;
}

/* A checkpoint folded for the transform stores W' = W P^T S H, whose rows
 * are rotate(row of W). Then W' rotate(x) = W P^T S H H S P x = W x. */
static int check_fold(const struct fixture *f) {
    int          fails = 0;
    const size_t hd = 128, nk = 16, rep = 3, width = hd * nk * rep, n_out = 8;
    float       *W  = xmalloc(n_out * width * sizeof(float));
    float       *Wf = xmalloc(n_out * width * sizeof(float));
    float       *x  = xmalloc(width * sizeof(float));
    float       *xr = xmalloc(width * sizeof(float));
    float       *s  = xmalloc(width * sizeof(float));
    fill_random(n_out * width, W);
    fill_random(width, x);
    fill_signs(width, s);
    fails += geist_expect(run(f, n_out, width, 1024, hd, nk, rep, false, false, width, W, s, Wf) ==
                                  GEIST_OK,
                          "fold: rotate W");
    fails += geist_expect(run(f, 1, width, 1024, hd, nk, rep, false, false, width, x, s, xr) ==
                                  GEIST_OK,
                          "fold: rotate x");
    double max_rel = 0.0;
    for (size_t o = 0; o < n_out; o++) {
        double y = 0.0, yf = 0.0, mag = 0.0;
        for (size_t i = 0; i < width; i++) {
            y += (double) W[o * width + i] * (double) x[i];
            yf += (double) Wf[o * width + i] * (double) xr[i];
            mag += fabs((double) W[o * width + i] * (double) x[i]);
        }
        max_rel = fmax(max_rel, fabs(y - yf) / mag);
    }
    fails += geist_expect(max_rel < 1e-5, "fold: W x == rotate(W) rotate(x)");
    free(W);
    free(Wf);
    free(x);
    free(xr);
    free(s);
    return fails;
}

static int check_inverse(const struct fixture *f) {
    int          fails = 0;
    const size_t rows = 3, width = 5120;
    float       *x = xmalloc(rows * width * sizeof(float));
    float       *z = xmalloc(rows * width * sizeof(float));
    float       *e = xmalloc(rows * width * sizeof(float));
    float       *s = xmalloc(width * sizeof(float));
    fill_random(rows * width, x);
    fill_signs(width, s);
    /* A latent embedding row z = forward(e); the lookup path restores e. */
    fails += geist_expect(fwd(f, rows, width, x, s, z) == GEIST_OK, "inverse: forward");
    fails += geist_expect(run(f, rows, width, 1024, 0, 0, 0, true, false, width, z, s, e) ==
                                  GEIST_OK,
                          "inverse: status");
    fails += geist_expect(rel_err(rows * width, e, x) < 1e-5, "inverse: inv(fwd(x)) == x");
    fails +=
            geist_expect(run(f, rows, width, 1024, 0, 0, 0, true, true, width, z, s, e) == GEIST_OK,
                         "inverse: in place status");
    fails += geist_expect(rel_err(rows * width, e, x) < 1e-5, "inverse: in place");
    free(x);
    free(z);
    free(e);
    free(s);
    return fails;
}

int main(void) {
    const char    *names[] = {"cpu_scalar", "cpu_neon", "metal"};
    struct fixture fx[3]   = {0};
    size_t         n_fx    = 0;
    int            fails   = 0;
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(names[i], nullptr, nullptr, &be) != GEIST_OK) {
            continue;
        }
        if (geist_backend_fused_tbl(be)->hadamard_rotate == nullptr) {
            fprintf(stderr, "FAIL: %s has no hadamard_rotate slot\n", names[i]);
            fails++;
            geist_backend_destroy(be);
            continue;
        }
        fx[n_fx++] = (struct fixture) {.be = be, .name = names[i]};
    }
    GEIST_SKIP_IF(n_fx == 0 && fails == 0, "no CPU backend compiled in");

    for (size_t i = 0; i < n_fx; i++) {
        g_seed = 0x2545F491u;
        fails += check_naive(&fx[i]);
        fails += check_orthonormal(&fx[i]);
        fails += check_shapes(&fx[i]);
        fails += check_rejects(&fx[i]);
        fails += check_signs_and_perm(&fx[i]);
        fails += check_fold(&fx[i]);
        fails += check_inverse(&fx[i]);
        printf("%s: done\n", fx[i].name);
    }

    /* 7. The CPU backends agree bit for bit on the full 27B ssm_out
     * transform (one shared host implementation; metal normalizes with
     * rsqrt and is held to the tolerance checks above instead). */
    if (n_fx >= 2 && strcmp(fx[0].name, "cpu_scalar") == 0 && strcmp(fx[1].name, "cpu_neon") == 0) {
        const size_t width = 6144, rows = 33;
        float       *x  = xmalloc(rows * width * sizeof(float));
        float       *s  = xmalloc(width * sizeof(float));
        float       *y0 = xmalloc(rows * width * sizeof(float));
        float       *y1 = xmalloc(rows * width * sizeof(float));
        fill_random(rows * width, x);
        fill_signs(width, s);
        const enum geist_status s0 =
                run(&fx[0], rows, width, 1024, 128, 16, 3, false, false, width, x, s, y0);
        const enum geist_status s1 =
                run(&fx[1], rows, width, 1024, 128, 16, 3, false, false, width, x, s, y1);
        fails += geist_expect(s0 == GEIST_OK && s1 == GEIST_OK, "backends: status");
        fails += geist_expect(memcmp(y0, y1, rows * width * sizeof(float)) == 0,
                              "backends: bit-identical");
        free(x);
        free(s);
        free(y0);
        free(y1);
    }

    for (size_t i = 0; i < n_fx; i++) {
        geist_backend_destroy(fx[i].be);
    }
    if (fails == 0) {
        printf("PASS test_hadamard_unit\n");
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
