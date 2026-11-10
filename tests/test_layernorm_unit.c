/*
 * test_layernorm_unit - prims->layernorm (#523) on every compiled backend
 * against a double-precision reference of torch.nn.LayerNorm:
 *
 *   y = (x - mean(x)) / sqrt(var(x) + eps) * w + b,  var over n (not n - 1)
 *
 * at Phi-MoE's width (4096; one row as a decode step has it, a prefill
 * chunk's 257 rows) and ragged ones, in place and into a separate y, with w
 * and b at an offset inside their buffers as the loader places them. Rows
 * alternate between a mean near zero and one several standard deviations
 * away, which is what tells a LayerNorm from an RMSNorm: the same reference
 * without the mean subtracted must miss by more than 1e-2.
 *
 * Every row within relative L2 1e-5 of the reference (the GPU kernels
 * reduce in float). A backend that is not compiled in is skipped; one that
 * is and has no layernorm fails, since the transformer arch has no host
 * path for it.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>

enum { MAX_ROWS = 257, MAX_COLS = 4096, MAX_N = MAX_ROWS * MAX_COLS, PAD = 5 };

static constexpr float  EPS = 1e-5f;
static constexpr double TOL = 1e-5;

static float  g_x[MAX_N], g_got[MAX_N];
static float  g_w[PAD + MAX_COLS], g_b[PAD + MAX_COLS];
static double g_ref[MAX_N];

static struct geist_tensor
rows_view(struct geist_buffer *buf, size_t off_floats, size_t rows, size_t cols) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = (int64_t) (off_floats * sizeof(float)),
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 2,
            .shape  = {(int64_t) rows, (int64_t) cols},
            .stride = {(int64_t) cols, 1},
    };
}

static struct geist_tensor vec_view(struct geist_buffer *buf, size_t off_floats, size_t n) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = (int64_t) (off_floats * sizeof(float)),
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 1,
            .shape  = {(int64_t) n},
            .stride = {1},
    };
}

static struct geist_buffer *upload(struct geist_backend *be, size_t n, const float *src) {
    struct geist_buffer *buf = nullptr;
    if (be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &buf) !=
                GEIST_OK ||
        be->desc->vtbl->buffer_upload(buf, n * sizeof(float), (const uint8_t *) src) != GEIST_OK) {
        return nullptr;
    }
    return buf;
}

static float noise(uint64_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return (float) (*s >> 40) / (float) (1u << 24) - 0.5f;
}

/* The reference of one row; `centre` false leaves the mean in (an RMSNorm
 * with a bias). */
static void
ref_row(size_t n, bool centre, const float *x, const float *w, const float *b, double *y) {
    double sum = 0.0, ss = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += (double) x[i];
    const double mean = centre ? sum / (double) n : 0.0;
    for (size_t i = 0; i < n; i++)
        ss += ((double) x[i] - mean) * ((double) x[i] - mean);
    const double inv = 1.0 / sqrt(ss / (double) n + (double) EPS);
    for (size_t i = 0; i < n; i++)
        y[i] = ((double) x[i] - mean) * inv * (double) w[i] + (double) b[i];
}

/* The largest relative L2 of a row of got against ref. */
static double worst_row(size_t rows, size_t cols, const float *got, const double *ref) {
    double worst = 0.0;
    for (size_t r = 0; r < rows; r++) {
        double num = 0.0, den = 0.0;
        for (size_t c = 0; c < cols; c++) {
            const double d = (double) got[r * cols + c] - ref[r * cols + c];
            num += d * d;
            den += ref[r * cols + c] * ref[r * cols + c];
        }
        const double rel = sqrt(num / (den > 0.0 ? den : 1.0));
        worst            = rel > worst ? rel : worst;
    }
    return worst;
}

static int check_shape(struct geist_backend *be, const char *name, size_t rows, size_t cols) {
    const size_t n = rows * cols;
    uint64_t     s = 0x9e3779b97f4a7c15ull ^ (rows * 131 + cols);
    for (size_t r = 0; r < rows; r++) {
        /* even rows: a mean 3.5 standard deviations off; all: a row scale */
        const float mean = r % 2 == 0 ? 3.5f : -0.02f, scale = 0.25f * (float) (1 + r % 5);
        for (size_t c = 0; c < cols; c++)
            g_x[r * cols + c] = (mean + 3.4641f * noise(&s)) * scale;
    }
    for (size_t c = 0; c < PAD + cols; c++) {
        g_w[c] = 1.0f + noise(&s);
        g_b[c] = 0.6f * noise(&s);
    }
    for (size_t r = 0; r < rows; r++)
        ref_row(cols, true, g_x + r * cols, g_w + PAD, g_b + PAD, g_ref + r * cols);

    const struct geist_backend_vtbl *vt = be->desc->vtbl;
    struct geist_buffer             *bx = upload(be, n, g_x);
    struct geist_buffer             *bi = upload(be, n, g_x);
    struct geist_buffer             *by = upload(be, n, g_x);
    struct geist_buffer             *bw = upload(be, PAD + cols, g_w);
    struct geist_buffer             *bb = upload(be, PAD + cols, g_b);
    int fails = geist_expect(bx != nullptr && bi != nullptr && by != nullptr && bw != nullptr &&
                                     bb != nullptr,
                             "the buffers are created");
    if (fails == 0) {
        struct geist_tensor t_x = rows_view(bx, 0, rows, cols);
        struct geist_tensor t_i = rows_view(bi, 0, rows, cols);
        struct geist_tensor t_y = rows_view(by, 0, rows, cols);
        struct geist_tensor t_w = vec_view(bw, PAD, cols);
        struct geist_tensor t_b = vec_view(bb, PAD, cols);
        const bool ran = be->desc->prims->layernorm(be, &t_x, &t_w, &t_b, EPS, &t_y) == GEIST_OK &&
                         be->desc->prims->layernorm(be, &t_i, &t_w, &t_b, EPS, &t_i) == GEIST_OK;
        if (!ran)
            fprintf(stderr, "  %s %zux%zu: %s\n", name, rows, cols, geist_backend_errmsg(be));
        fails += geist_expect(ran, "layernorm runs, into y and in place");
        if (ran && vt->buffer_download(n * sizeof(float), (uint8_t *) g_got, by) == GEIST_OK) {
            const double into = worst_row(rows, cols, g_got, g_ref);
            fails += geist_expect(into <= TOL, "into y: every row within 1e-5 of the reference");
            /* The reference with the mean left in: not this op (a single
             * column has no mean to tell them apart). */
            for (size_t r = 0; r < rows; r++)
                ref_row(cols, false, g_x + r * cols, g_w + PAD, g_b + PAD, g_ref + r * cols);
            const double rms = worst_row(rows, cols, g_got, g_ref);
            fails += geist_expect(cols < 16 || rms > 1e-2, "an RMSNorm with a bias is not it");
            for (size_t r = 0; r < rows; r++)
                ref_row(cols, true, g_x + r * cols, g_w + PAD, g_b + PAD, g_ref + r * cols);
            double inplace = INFINITY;
            if (vt->buffer_download(n * sizeof(float), (uint8_t *) g_got, bi) == GEIST_OK)
                inplace = worst_row(rows, cols, g_got, g_ref);
            fails += geist_expect(inplace <= TOL, "in place: every row within 1e-5");
            /* x is an input only */
            bool kept = vt->buffer_download(n * sizeof(float), (uint8_t *) g_got, bx) == GEIST_OK;
            for (size_t i = 0; i < n && kept; i++)
                kept = g_got[i] == g_x[i];
            fails += geist_expect(kept, "x is left as it was");
            printf("  %s %3zu x %4zu: relative L2 %.2e into y, %.2e in place (without the mean "
                   "%.2e)\n",
                   name,
                   rows,
                   cols,
                   into,
                   inplace,
                   rms);
        } else if (ran) {
            fails += geist_expect(false, "the result downloads");
        }
    }
    struct geist_buffer *bufs[] = {bx, bi, by, bw, bb};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if (bufs[i] != nullptr)
            (void) vt->buffer_destroy(be, bufs[i]);
    return fails;
}

int main(void) {
    /* Phi-MoE: d_model 4096; a decode step, a short chunk, a 257-row
     * prefill chunk. Then widths around the kernels' 256 threads. */
    static const size_t shapes[][2] = {
            {1, 4096}, {7, 4096}, {257, 4096}, {3, 1}, {17, 77}, {4, 255}, {5, 1000}};
    static const char *const names[] = {"cpu_scalar", "cpu_neon", "cpu_x86", "metal", "vulkan"};
    int                      fails = 0, ran = 0;
    for (size_t k = 0; k < sizeof names / sizeof names[0]; k++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(names[k], nullptr, nullptr, &be) != GEIST_OK) {
            printf("  %s: not available, skipped\n", names[k]);
            continue;
        }
        if (be->desc->prims->layernorm == nullptr) {
            fails += geist_expect(false, "a compiled backend has prims->layernorm");
        } else {
            for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
                fails += check_shape(be, names[k], shapes[s][0], shapes[s][1]);
            ran++;
        }
        geist_backend_destroy(be);
    }
    fails += geist_expect(ran > 0, "at least one backend ran");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
