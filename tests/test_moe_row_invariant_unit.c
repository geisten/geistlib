/*
 * test_moe_row_invariant_unit — a routed expert's projections give a token
 * the same bits whatever number of rows the expert gets in one call.
 *
 * A prefill chunk gathers the rows routed to one expert into one m-row
 * linear (forward/moe.c), so m is how many of the chunk's tokens chose the
 * expert: it changes with the chunk size, the prompt and the chunk
 * boundary. A kernel whose summation order depends on m gives a token a
 * different result per chunking; the difference grows over the layers and
 * flips near-tie expert choices (#507: 18 % of Qwen3-30B-A3B's selections
 * differ between 64- and 128-row chunks). So for every m in 1..M_MAX each
 * row of an m-row call must equal that row of the M_MAX-row call, bit for
 * bit, through the hooks forward/moe.c calls (moe_expert_linear,
 * moe_expert_pair).
 *
 * Shapes and dtypes are the real models' expert tensors:
 *   Qwen3-30B-A3B      gate/up Q4_K 2048->768, down Q4_K / Q6_K 768->2048
 *   DeepSeek-V2-Lite   gate/up Q4_K 2048->1408, down Q5_0 / Q8_0 1408->2048,
 *                      shared expert Q4_K 2048->2816, Q4_K / Q6_K 2816->2048
 *   gpt-oss-20b        gate/up/down MXFP4 2880->2880
 *
 * GEIST_ROWINV_REPORT=1 prints every mismatching m instead of stopping at
 * the first per case (a survey of a backend's kernels).
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


constexpr size_t M_MAX = 128;

struct shape {
    const char      *model;
    enum geist_dtype dt;
    size_t           n_in, n_out;
    bool             pair; /* gate+up through the pair hook */
    uint16_t         flags;
};

static const struct shape SHAPES[] = {
        {"qwen3moe gate/up", GEIST_DTYPE_Q4_K, 2048, 768, true, 0},
        {"qwen3moe down", GEIST_DTYPE_Q4_K, 768, 2048, false, 0},
        {"qwen3moe down", GEIST_DTYPE_Q6_K, 768, 2048, false, 0},
        {"deepseek2 gate/up", GEIST_DTYPE_Q4_K, 2048, 1408, true, 0},
        {"deepseek2 down", GEIST_DTYPE_Q5_0, 1408, 2048, false, 0},
        {"deepseek2 down", GEIST_DTYPE_Q8_0, 1408, 2048, false, 0},
        {"deepseek2 shexp gate/up", GEIST_DTYPE_Q4_K, 2048, 2816, true, 0},
        {"deepseek2 shexp down", GEIST_DTYPE_Q4_K, 2816, 2048, false, 0},
        {"deepseek2 shexp down", GEIST_DTYPE_Q6_K, 2816, 2048, false, 0},
        {"gpt-oss gate/up", GEIST_DTYPE_MXFP4, 2880, 2880, true, 0},
        {"gpt-oss down", GEIST_DTYPE_MXFP4, 2880, 2880, false, 0},
        {"qwen3moe router", GEIST_DTYPE_F32, 2048, 128, false, GEIST_W_EXACT},
        {"deepseek2 router", GEIST_DTYPE_F32, 2048, 64, false, GEIST_W_EXACT},
        {"gpt-oss router", GEIST_DTYPE_F32, 2880, 32, false, GEIST_W_EXACT},
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
 * channel every 97, so the per-row quantization scale varies by row. */
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

static bool weight_make(struct geist_backend *be, const struct shape *s, struct weight *t) {
    size_t bytes = 0;
    t->raw       = make_weight(dtype_index(s->dt), s->n_in, s->n_out, &bytes);
    if (t->raw == nullptr) {
        return false;
    }
    t->w = (struct geist_weight) {.raw        = t->raw,
                                  .raw_nbytes = bytes,
                                  .n_in       = (int32_t) s->n_in,
                                  .n_out      = (int32_t) s->n_out,
                                  .dtype      = (uint16_t) s->dt,
                                  .flags      = s->flags};
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

/* Rows [0, m) of x through the expert's hook(s), as forward/moe.c calls
 * them in a prefill chunk; y1 only for a pair. */
static bool run(struct geist_backend *be,
                const struct weight  *w0,
                const struct weight  *w1,
                size_t                m,
                const float          *x,
                float                *y0,
                float                *y1) {
    const bool mn1 = getenv("SURVEY_MN1") != nullptr;
    if (w1 == nullptr) {
        if (m == 1 && !mn1) w0->w.linear_m1(x, &w0->w, be, y0);
        else w0->w.linear_mN(m, x, &w0->w, be, y0);
        return true;
    }
    if (w0->w.linear_pair_mN == nullptr || getenv("SURVEY_NOPAIR")) {
        bool a = run(be, w0, nullptr, m, x, y0, nullptr);
        return a && run(be, w1, nullptr, m, x, y1, nullptr);
    }
    if (m == 1 && !mn1 && w0->w.linear_pair_m1) w0->w.linear_pair_m1(x, &w0->w, &w1->w, be, y0, y1);
    else if (m == 1 && !mn1) { run(be, w0, nullptr, m, x, y0, nullptr); run(be, w1, nullptr, m, x, y1, nullptr);}
    else w0->w.linear_pair_mN(m, x, &w0->w, &w1->w, be, y0, y1);
    return true;
}

static int check_shape(struct geist_backend *be, const char *bname, const struct shape *s) {
    struct weight w0 = {0}, w1 = {0};
    const size_t  n_in = s->n_in, n_out = s->n_out;
    char          what[192];
    snprintf(what, sizeof what, "%s %s %s: resolve", bname, s->model, DTYPES[dtype_index(s->dt)].name);
    if (!weight_make(be, s, &w0) || (s->pair && !weight_make(be, s, &w1))) {
        weight_free(&w0);
        weight_free(&w1);
        return geist_expect(false, what);
    }
    float *x    = xmalloc(M_MAX * n_in * sizeof *x);
    float *ref0 = xmalloc(M_MAX * n_out * sizeof *ref0);
    float *ref1 = xmalloc(M_MAX * n_out * sizeof *ref1);
    float *y0   = xmalloc(M_MAX * n_out * sizeof *y0);
    float *y1   = xmalloc(M_MAX * n_out * sizeof *y1);
    fill_x(M_MAX * n_in, x);
    const struct weight *p1     = s->pair ? &w1 : nullptr;
    int                  fails  = 0;
    const bool           report = getenv("GEIST_ROWINV_REPORT") != nullptr;
    if (!run(be, &w0, p1, M_MAX, x, ref0, ref1)) {
        fails = geist_expect(false, "hooks installed");
    }
    size_t bad = 0;
    const size_t team = geist_par_max_threads();
    if (getenv("SURVEY_INLINE")) geist_par_set_max_threads(1);
    /* Every m from 1 to M_MAX, each starting at a different row of x, so
     * the rows compared differ between calls. */
    for (size_t m = 1; m <= M_MAX && fails == 0; m++) {
        const size_t r0 = (m * 7) % (M_MAX - m + 1);
        (void) run(be, &w0, p1, m, x + r0 * n_in, y0, y1);
        bool same = memcmp(y0, ref0 + r0 * n_out, m * n_out * sizeof *y0) == 0;
        if (s->pair) {
            same = same && memcmp(y1, ref1 + r0 * n_out, m * n_out * sizeof *y1) == 0;
        }
        if (!same) {
            bad++;
            if (report) {
                char rows[256] = "";
                size_t nb = 0;
                for (size_t r = 0; r < m && nb < 200; r++) {
                    bool eq = memcmp(y0 + r * n_out, ref0 + (r0 + r) * n_out, n_out * sizeof *y0) == 0;
                    if (s->pair) eq = eq && memcmp(y1 + r * n_out, ref1 + (r0 + r) * n_out, n_out * sizeof *y1) == 0;
                    if (!eq) nb += (size_t) snprintf(rows + nb, sizeof rows - nb, "%zu,", r);
                }
                printf("  %s %s %s: m=%zu differs from m=%zu, rows %s\n",
                       bname,
                       s->model,
                       DTYPES[dtype_index(s->dt)].name,
                       m,
                       M_MAX, rows);
            } else {
                snprintf(what,
                         sizeof what,
                         "%s %s %s %zu->%zu: rows of an m=%zu call are the m=%zu call's bits",
                         bname,
                         s->model,
                         DTYPES[dtype_index(s->dt)].name,
                         n_in,
                         n_out,
                         m,
                         M_MAX);
                fails += geist_expect(false, what);
            }
        }
    }
    geist_par_set_max_threads(team);
    if (report) {
        printf("%s %s %s %zu->%zu: %zu of %zu m differ\n",
               bname,
               s->model,
               DTYPES[dtype_index(s->dt)].name,
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
    printf("PASS: expert GEMM rows are bit-identical for every m in 1..%zu\n", M_MAX);
    return GEIST_TEST_PASS;
}
