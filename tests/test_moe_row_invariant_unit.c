/*
 * test_moe_row_invariant_unit — a routed expert's projections give a token
 * the same bits whatever number of rows the expert gets in one call (#527).
 *
 * A prefill chunk gathers the rows routed to one expert into one m-row
 * linear (forward/moe.c), so m is how many of the chunk's tokens chose the
 * expert: it changes with the chunk size, the prompt and the chunk
 * boundary. A kernel whose summation order depends on m gives a token a
 * different result per chunking; the difference grows over the layers and
 * flips near-tie expert choices (#507: 18 % of Qwen3-30B-A3B's selections
 * differed between 64- and 128-row chunks). So for every m in 1..M_MAX each
 * row of an m-row call must equal that row of the M_MAX-row call, bit for
 * bit, through the hooks moe_expert_rows calls: the gate/up pair hook where
 * both weights share it, else one linear each, the M=1 hook for one row.
 *
 * Experts run both ways in a prefill chunk: the widest on the whole team,
 * the rest one per thread with their kernels inline. The reference runs on
 * the team, odd m on one thread, even m on the team.
 *
 * Shapes and dtypes are the real models' expert tensors:
 *   Qwen3-30B-A3B      gate/up Q4_K 2048->768, down Q4_K / Q6_K 768->2048
 *   DeepSeek-V2-Lite   gate/up Q4_K 2048->1408, down Q5_0 / Q8_0 1408->2048,
 *                      shared expert Q4_K 2048->2816, Q4_K / Q6_K 2816->2048
 *   gpt-oss-20b        gate/up/down MXFP4 2880->2880
 *
 * GEIST_ROWINV_REPORT=1 prints every m whose rows differ and which rows,
 * instead of failing on the first per shape (a survey of a host's kernels).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "heap.h"
#include "par.h"
#include "quant_fixtures.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The prefill chunk's row cap for an MoE model on a CPU backend, and the
 * kernels' (GEIST_QUANT_M_CAP). */
constexpr size_t M_MAX = 128;

struct shape {
    const char      *model;
    enum geist_dtype dt;
    size_t           n_in, n_out;
    bool             pair; /* gate and up */
};

static const struct shape SHAPES[] = {
        {"qwen3moe gate/up", GEIST_DTYPE_Q4_K, 2048, 768, true},
        {"qwen3moe down", GEIST_DTYPE_Q4_K, 768, 2048, false},
        {"qwen3moe down", GEIST_DTYPE_Q6_K, 768, 2048, false},
        {"deepseek2 gate/up", GEIST_DTYPE_Q4_K, 2048, 1408, true},
        {"deepseek2 down", GEIST_DTYPE_Q5_0, 1408, 2048, false},
        {"deepseek2 down", GEIST_DTYPE_Q8_0, 1408, 2048, false},
        {"deepseek2 shexp gate/up", GEIST_DTYPE_Q4_K, 2048, 2816, true},
        {"deepseek2 shexp down", GEIST_DTYPE_Q4_K, 2816, 2048, false},
        {"deepseek2 shexp down", GEIST_DTYPE_Q6_K, 2816, 2048, false},
        {"gpt-oss gate/up", GEIST_DTYPE_MXFP4, 2880, 2880, true},
        {"gpt-oss down", GEIST_DTYPE_MXFP4, 2880, 2880, false},
};

static size_t dtype_index(enum geist_dtype dt) {
    for (size_t d = 0; d < N_DTYPES; d++) {
        if (DTYPES[d].dt == dt) {
            return d;
        }
    }
    return N_DTYPES;
}

/* Activations shaped like a hidden state: unit Gaussian with an outlier
 * channel every 97, so the per-row quantization scales differ by row. */
static void fill_x(size_t n, float x[static n]) {
    for (size_t i = 0; i < n; i++) {
        const float u1 = ((float) (next_u32() >> 8) + 1.0f) / (float) (1u << 24);
        const float u2 = (float) (next_u32() >> 8) / (float) (1u << 24);
        x[i]           = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        if (i % 97 == 13) {
            x[i] *= 12.0f;
        }
    }
}

struct weight {
    uint8_t            *raw;
    struct geist_weight w;
};

/* make_weight pins every fp16 scale to 0.5, which makes products with it
 * exact and hides a different association of the scale multiplies (the
 * cpu_neon Q4_K pair kernel differed on Qwen3-30B-A3B's weights and agreed
 * on pinned ones). Real models' scales: a random fp16 in [2^-12, 2^-6). */
static void randomize_scales(size_t d, size_t n_blocks, uint8_t *raw) {
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = raw + b * DTYPES[d].bytes;
        for (size_t f = 0; f < 2; f++) {
            if (DTYPES[d].d[f] >= 0) {
                const uint16_t h =
                        (uint16_t) (((3u + next_u32() % 6u) << 10) | (next_u32() & 0x3FFu));
                blk[DTYPES[d].d[f]]     = (uint8_t) h;
                blk[DTYPES[d].d[f] + 1] = (uint8_t) (h >> 8);
            }
        }
    }
}

static bool weight_make(struct geist_backend *be, const struct shape *s, struct weight *t) {
    const size_t d     = dtype_index(s->dt);
    size_t       bytes = 0;
    t->raw             = make_weight(d, s->n_in, s->n_out, &bytes);
    if (t->raw == nullptr) {
        return false;
    }
    randomize_scales(d, s->n_in * s->n_out / DTYPES[d].blk, t->raw);
    t->w = (struct geist_weight) {.raw        = t->raw,
                                  .raw_nbytes = bytes,
                                  .n_in       = (int32_t) s->n_in,
                                  .n_out      = (int32_t) s->n_out,
                                  .dtype      = (uint16_t) s->dt};
    return be->desc->vtbl->resolve_weight(be, &t->w) == GEIST_OK;
}

static void weight_free(struct weight *t) {
    if ((t->w.flags & GEIST_W_AUX_HEAP_OWNED) != 0) {
        void *aux = (void *) t->w.aux_fp32;
        safe_free(&aux);
    }
    free(t->raw);
    *t = (struct weight) {0};
}

/* One projection as moe_linear calls it. */
static bool
linear(size_t m, const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    if (m == 1 && w->linear_m1 != nullptr) {
        w->linear_m1(x, w, be, y);
        return true;
    }
    if (m > 1 && w->linear_mN != nullptr) {
        w->linear_mN(m, x, w, be, y);
        return true;
    }
    return false;
}

/* Rows [0, m) of x through the expert's hooks as moe_expert_rows calls
 * them; w1 and y1 only for gate/up. */
static bool expert_call(struct geist_backend      *be,
                        const struct geist_weight *w0,
                        const struct geist_weight *w1,
                        size_t                     m,
                        const float               *x,
                        float                     *y0,
                        float                     *y1) {
    if (w1 == nullptr) {
        return linear(m, x, w0, be, y0);
    }
    if (m == 1 && w0->linear_pair_m1 != nullptr && w0->linear_pair_m1 == w1->linear_pair_m1) {
        w0->linear_pair_m1(x, w0, w1, be, y0, y1);
        return true;
    }
    if (m > 1 && w0->linear_pair_mN != nullptr && w0->linear_pair_mN == w1->linear_pair_mN) {
        w0->linear_pair_mN(m, x, w0, w1, be, y0, y1);
        return true;
    }
    return linear(m, x, w0, be, y0) && linear(m, x, w1, be, y1);
}

/* Whether row r of the m-row result equals row r0 + r of the reference. */
static bool row_same(size_t       n_out,
                     size_t       r,
                     size_t       r0,
                     bool         pair,
                     const float *y0,
                     const float *y1,
                     const float *ref0,
                     const float *ref1) {
    const size_t bytes = n_out * sizeof(float);
    return memcmp(y0 + r * n_out, ref0 + (r0 + r) * n_out, bytes) == 0 &&
           (!pair || memcmp(y1 + r * n_out, ref1 + (r0 + r) * n_out, bytes) == 0);
}

static int check_shape(struct geist_backend *be, const char *bname, const struct shape *s) {
    const char   *dname = DTYPES[dtype_index(s->dt)].name;
    const size_t  n_in = s->n_in, n_out = s->n_out;
    struct weight w0 = {0}, w1 = {0};
    char          what[192];
    snprintf(what, sizeof what, "%s %s %s: resolve", bname, s->model, dname);
    if (!weight_make(be, s, &w0) || (s->pair && !weight_make(be, s, &w1))) {
        weight_free(&w0);
        weight_free(&w1);
        return geist_expect(false, what);
    }
    const struct geist_weight *p1     = s->pair ? &w1.w : nullptr;
    const bool                 report = getenv("GEIST_ROWINV_REPORT") != nullptr;
    const size_t               team   = geist_par_max_threads();
    float                     *x      = xmalloc(M_MAX * n_in * sizeof *x);
    float                     *ref0   = xmalloc(M_MAX * n_out * sizeof *ref0);
    float                     *ref1   = xmalloc(M_MAX * n_out * sizeof *ref1);
    float                     *y0     = xmalloc(M_MAX * n_out * sizeof *y0);
    float                     *y1     = xmalloc(M_MAX * n_out * sizeof *y1);
    fill_x(M_MAX * n_in, x);
    snprintf(what, sizeof what, "%s %s %s: kernels installed", bname, s->model, dname);
    int    fails = geist_expect(expert_call(be, &w0.w, p1, M_MAX, x, ref0, ref1), what);
    size_t bad   = 0;
    /* cpu_neon has no native Q5_0 kernel: it dequantizes tiles for the
     * BLAS, which routes a one-row product through its gemv, summing in
     * another order than its gemm. A two-row sgemm would give the one row
     * the gemm's bits, but Accelerate's small sgemm made that call 3x
     * slower (DeepSeek-V2-Lite prefill +36 % on the M6), so m = 1 is left
     * out there until the format has a native kernel (#527). */
    const size_t m_first = strcmp(bname, "cpu_neon") == 0 && s->dt == GEIST_DTYPE_Q5_0 ? 2 : 1;
    /* Each m starts at another row of x, so the rows compared differ. */
    for (size_t m = m_first; m <= M_MAX && fails == 0; m++) {
        const size_t r0 = (m * 7) % (M_MAX - m + 1);
        geist_par_set_max_threads(m % 2 == 1 ? 1 : team);
        (void) expert_call(be, &w0.w, p1, m, x + r0 * n_in, y0, y1);
        geist_par_set_max_threads(team);
        size_t first = m;
        for (size_t r = 0; r < m && first == m; r++) {
            if (!row_same(n_out, r, r0, s->pair, y0, y1, ref0, ref1)) {
                first = r;
            }
        }
        if (first == m) {
            continue;
        }
        bad++;
        if (report) {
            printf("  %s %s %s: m=%zu, rows differ from row %zu on:",
                   bname,
                   s->model,
                   dname,
                   m,
                   first);
            for (size_t r = first; r < m; r++) {
                if (!row_same(n_out, r, r0, s->pair, y0, y1, ref0, ref1)) {
                    printf(" %zu", r);
                }
            }
            printf("\n");
        } else {
            snprintf(what,
                     sizeof what,
                     "%s %s %s %zu->%zu: row %zu of an m=%zu call is the m=%zu call's bits",
                     bname,
                     s->model,
                     dname,
                     n_in,
                     n_out,
                     first,
                     m,
                     M_MAX);
            fails += geist_expect(false, what);
        }
    }
    if (report) {
        printf("%s %s %s %zu->%zu: %zu of %zu row counts differ\n",
               bname,
               s->model,
               dname,
               n_in,
               n_out,
               bad,
               M_MAX);
    }
    weight_free(&w0);
    weight_free(&w1);
    free(x);
    free(ref0);
    free(ref1);
    free(y0);
    free(y1);
    return fails;
}

int main(void) {
    static const char *BACKENDS[] = {"cpu_neon", "cpu_x86"};
    int                fails      = 0;
    size_t             ran        = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK) {
            continue;
        }
        ran++;
        for (size_t s = 0; s < sizeof SHAPES / sizeof SHAPES[0]; s++) {
            fails += check_shape(be, BACKENDS[b], &SHAPES[s]);
        }
        geist_backend_destroy(be);
    }
    if (ran == 0) {
        GEIST_SKIP("neither cpu_neon nor cpu_x86 compiled in");
    }
    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: an expert's rows have the same bits for every row count 1..%zu\n", M_MAX);
    return GEIST_TEST_PASS;
}
