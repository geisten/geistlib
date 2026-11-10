/*
 * test_moe_grouped_unit - a prefill chunk's mixture of experts on the device
 * (#482): fused->moe_route over a whole chunk, and fused->linear_t_ids at
 * chunk sizes, where the backend buckets the routed slots by expert and runs
 * each expert's rows as one GEMM (Metal: moe_map + moe_mm_<T>).
 *
 * Router: 512 rows of 64 logits (random, ties, all zero) for every router
 * the device runs -- softmax with and without renormalisation, a router
 * bias, softmax over the selected (GPT-OSS), and (#376) sigmoid with a
 * selection bias (GigaChat3), a weight scale and groups (DeepSeek-V3) --
 * against the host's moe_route row by row: ids exact, weights within 1e-6
 * (times the scale).
 *
 * Grouped GEMM, for Q4_K, Q6_K, Q8_0, MXFP4 and Q5_0 (#376): 16 experts back
 * to back in one buffer, 136 outputs (not a multiple of the 64-wide tile),
 * 512 inputs; m = 9 (the first chunk size it takes), 37 and 300 tokens of
 * k = 1, 4 and 6 slots; ids random (k distinct experts per token), skewed
 * (expert 0 in every token, the last expert in none) and uniform (every
 * token the same k experts). x one row per token (gate/up) and one per slot
 * (down). Every slot's row of y is written (y starts as NaN), and against
 *   - a double-precision product of the reference decoder's weights
 *     (dequant_*_row): within 1e-3 of the output scale (the GEMM stages x
 *     and the weights as half, as the per-expert prefill GEMMs do);
 *   - fused->linear_t over each expert's rows alone (the per-expert path
 *     a prefill chunk took before): within 1e-3 of the scale;
 *   - its own second run: bit-identical (the bucketing keeps slot order).
 * supported(GEIST_FUSED_MOE_DEVICE) says yes for these dtypes at m = 512
 * (Metal; #499: Vulkan where it has tensor cores, elsewhere linear_t_ids
 * runs a matvec per slot, to the same bounds) and no for F32 there (no
 * grouped GEMM: the chunk stays on the host).
 *
 * The experts' FFN in one call (#486, fused->moe_ffn_ids), per dtype: gate
 * and up of 256 (K-quants) or 160 outputs (a ragged 64-wide tile), down of
 * 136; m / k / ids from 1 x 8 (a decode step) to 300 x 6 (random, skewed,
 * uniform), with the plain SwiGLU and with swiglu_oai and both biases, on
 * the rows each backend fuses (Metal a chunk's, from 9; Vulkan a decode
 * step's, Q4_K gate and up, #505); g and y against
 *   - the four calls it fuses (linear_t_ids gate and up, silu_mul or glu,
 *     linear_t_ids down): within 1e-5 of the scale (the same tiles; the
 *     count of bit-identical cases is printed);
 *   - a double-precision FFN of the dequantized weights: within 2e-3 of
 *     the scale (x and g are staged as half);
 *   - its own second run: bit-identical.
 * The other rows, and gate / up of different dtypes, return
 * GEIST_E_UNSUPPORTED.
 *
 * SKIPs when no backend in the build has the ops.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../src/archs/transformer/forward/blocks.h"
#include "test_helpers.h"

#include "quant.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_EXP = 16, N_IN = 512, N_OUT = 136, M_MAX = 300, K_MAX = 6 };

static uint32_t rng = 0x2468ace1u;
static uint32_t rng_u32(void) {
    rng = rng * 1664525u + 1013904223u;
    return rng;
}
static uint8_t rng_u8(void) {
    return (uint8_t) (rng_u32() >> 24);
}

static struct geist_tensor mat(struct geist_buffer *b, size_t off, size_t rows, size_t cols) {
    return (struct geist_tensor) {.buffer = b,
                                  .offset = off,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

static struct geist_tensor vec(struct geist_buffer *b, size_t off, size_t n) {
    return (struct geist_tensor) {.buffer = b,
                                  .offset = off,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 1,
                                  .shape  = {(int64_t) n},
                                  .stride = {1}};
}

/* ---- the router over a chunk ------------------------------------------- */

enum { R_ROWS = 512, R_N = 64 };

/* A router: gating, renormalisation, the router bias and (#376) the
 * selection bias (nullptr: none), the weight scale and the groups. */
struct route_opts {
    enum geist_moe_route_gating gating;
    bool                        norm;
    const float                *bias, *sel;
    float                       scale;
    size_t                      n_group, topk_group;
};

static enum geist_moe_gating host_gating(enum geist_moe_route_gating g) {
    switch (g) {
    case GEIST_MOE_ROUTE_SOFTMAX:
        return GEIST_MOE_GATING_SOFTMAX;
    case GEIST_MOE_ROUTE_SOFTMAX_WEIGHT:
        return GEIST_MOE_GATING_SOFTMAX_WEIGHT;
    case GEIST_MOE_ROUTE_SIGMOID:
        return GEIST_MOE_GATING_SIGMOID;
    }
    return GEIST_MOE_GATING_SOFTMAX;
}

/* R_ROWS rows of R_N logits routed with k on the device, each row against
 * the host's moe_route with the same router. */
static int check_route_chunk(struct geist_backend    *be,
                             const char              *what,
                             size_t                   k,
                             const struct route_opts *o,
                             const float             *logits) {
    const struct geist_backend_vtbl  *v = be->desc->vtbl;
    const struct geist_backend_fused *f = be->desc->fused;
    struct geist_buffer *bl = nullptr, *bi = nullptr, *bw = nullptr, *bb = nullptr, *bs = nullptr;
    const size_t         nk    = R_ROWS * k;
    int                  fails = geist_expect(
            v->buffer_create(be, R_ROWS * R_N * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bl) ==
                            GEIST_OK &&
                    v->buffer_create(be, (nk + 1) * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) ==
                            GEIST_OK &&
                    v->buffer_create(be, nk * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bw) ==
                            GEIST_OK &&
                    v->buffer_create(be, R_N * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bb) ==
                            GEIST_OK &&
                    v->buffer_create(be, R_N * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bs) ==
                            GEIST_OK,
            "route buffers");
    uint32_t *ids = calloc(nk + 1, sizeof *ids);
    float    *w   = calloc(nk, sizeof *w);
    if (fails > 0 || ids == nullptr || w == nullptr) {
        fails += geist_expect(false, "route host memory");
        goto out;
    }
    fails += geist_expect(
            v->buffer_upload(bl, R_ROWS * R_N * 4, (const uint8_t *) logits) == GEIST_OK &&
                    v->buffer_upload(bi, (nk + 1) * 4, (const uint8_t *) ids) == GEIST_OK &&
                    (o->bias == nullptr ||
                     v->buffer_upload(bb, R_N * 4, (const uint8_t *) o->bias) == GEIST_OK) &&
                    (o->sel == nullptr ||
                     v->buffer_upload(bs, R_N * 4, (const uint8_t *) o->sel) == GEIST_OK),
            "route upload");
    struct geist_tensor               tl = mat(bl, 0, R_ROWS, R_N), tw = vec(bw, 0, nk);
    struct geist_tensor               tb = vec(bb, 0, R_N), ts = vec(bs, 0, R_N);
    const struct geist_moe_route_args ra = {.m          = R_ROWS,
                                            .n          = R_N,
                                            .k          = k,
                                            .gating     = o->gating,
                                            .norm       = o->norm,
                                            .bias       = o->bias != nullptr ? &tb : nullptr,
                                            .sel_bias   = o->sel != nullptr ? &ts : nullptr,
                                            .w_scale    = o->scale,
                                            .n_group    = o->n_group,
                                            .topk_group = o->topk_group};
    const struct moe_router           hr = {.gating     = host_gating(o->gating),
                                            .norm       = o->norm,
                                            .bias       = o->bias,
                                            .sel_bias   = o->sel,
                                            .w_scale    = o->scale,
                                            .n_group    = o->n_group,
                                            .topk_group = o->topk_group};
    fails += geist_expect(f->moe_route(be, &ra, &tl, bi, nk * 4, &tw) == GEIST_OK, what);
    fails += geist_expect(v->buffer_download((nk + 1) * 4, (uint8_t *) ids, bi) == GEIST_OK &&
                                  v->buffer_download(nk * 4, (uint8_t *) w, bw) == GEIST_OK,
                          "route download");
    fails += geist_expect(ids[nk] == 0, "no flag");
    size_t bad = 0;
    double dw  = 0.0;
    for (size_t r = 0; r < R_ROWS && fails == 0; r++) {
        uint32_t ref_ids[R_N];
        float    ref_w[R_N];
        fails += geist_expect(moe_route(R_N, k, &hr, logits + r * R_N, ref_ids, ref_w),
                              "host reference");
        for (size_t j = 0; j < k; j++) {
            bad += ids[r * k + j] != ref_ids[j];
            dw = fmax(dw, fabs((double) w[r * k + j] - (double) ref_w[j]));
        }
    }
    printf("  chunk route, %s: %zu of %zu ids differ, max |dw| %.3g\n", what, bad, nk, dw);
    fails += geist_expect(bad == 0 && dw <= 1e-6 * (o->scale > 1.0f ? o->scale : 1.0f), what);
out:
    free(ids);
    free(w);
    struct geist_buffer *bufs[] = {bl, bi, bw, bb, bs};
    for (size_t i = 0; i < 5; i++)
        if (bufs[i] != nullptr)
            v->buffer_destroy(be, bufs[i]);
    return fails;
}

static int test_route_chunk(struct geist_backend *be) {
    float *lg = malloc(R_ROWS * R_N * sizeof *lg);
    float  bias[R_N], sel[R_N];
    if (lg == nullptr)
        return geist_expect(false, "host memory");
    for (size_t r = 0; r < R_ROWS; r++)
        for (size_t i = 0; i < R_N; i++)
            lg[r * R_N + i] = r % 8 == 1   ? (float) (i % 5) /* ties */
                              : r % 8 == 2 ? 0.0f            /* all zero */
                                           : (float) rng_u8() / 16.0f - 8.0f;
    for (size_t i = 0; i < R_N; i++) {
        bias[i] = (float) rng_u8() / 64.0f - 2.0f;
        sel[i]  = (float) rng_u8() / 512.0f - 0.25f;
    }
    const struct {
        const char       *what;
        size_t            k;
        struct route_opts o;
    } cases[] = {
            {"softmax, renormalised (Qwen3, OLMoE)",
             8,
             {.gating = GEIST_MOE_ROUTE_SOFTMAX, .norm = true}},
            {"softmax (DeepSeek-V2-Lite)", 6, {.gating = GEIST_MOE_ROUTE_SOFTMAX}},
            {"softmax, k = 1", 1, {.gating = GEIST_MOE_ROUTE_SOFTMAX}},
            {"softmax with a router bias", 4, {.gating = GEIST_MOE_ROUTE_SOFTMAX, .bias = bias}},
            {"softmax over the selected", 4, {.gating = GEIST_MOE_ROUTE_SOFTMAX_WEIGHT}},
            {"GPT-OSS: router bias, softmax over the selected",
             4,
             {.gating = GEIST_MOE_ROUTE_SOFTMAX_WEIGHT, .bias = bias}},
            {"GigaChat3: sigmoid, selection bias, renormalised",
             4,
             {.gating = GEIST_MOE_ROUTE_SIGMOID, .norm = true, .sel = sel}},
            {"sigmoid, renormalised, scale 2.5",
             8,
             {.gating = GEIST_MOE_ROUTE_SIGMOID, .norm = true, .scale = 2.5f}},
            {"softmax, selection bias, scale 16",
             6,
             {.gating = GEIST_MOE_ROUTE_SOFTMAX, .sel = sel, .scale = 16.0f}},
            {"V3: sigmoid, selection bias, 8 groups keep 4, renormalised, scale 2.5",
             8,
             {.gating     = GEIST_MOE_ROUTE_SIGMOID,
              .norm       = true,
              .sel        = sel,
              .scale      = 2.5f,
              .n_group    = 8,
              .topk_group = 4}},
            {"softmax, 4 groups keep 2",
             8,
             {.gating = GEIST_MOE_ROUTE_SOFTMAX, .n_group = 4, .topk_group = 2}},
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++)
        fails += check_route_chunk(be, cases[i].what, cases[i].k, &cases[i].o, lg);
    free(lg);
    return fails;
}

/* ---- the grouped expert GEMM -------------------------------------------- */

struct qfmt {
    int         dtype;
    size_t      elems, bytes;
    bool        trailing_d; /* Q6_K: d is the trailing f16 */
    const char *name;
};

static void fill(uint8_t *dst, const struct qfmt *q, size_t rows) {
    for (size_t b = 0; b < rows * (N_IN / q->elems); b++) {
        uint8_t *blk = dst + b * q->bytes;
        for (size_t i = 0; i < q->bytes; i++)
            blk[i] = rng_u8();
        if (q->dtype == GEIST_DTYPE_MXFP4) {
            blk[0] = (uint8_t) (120 + rng_u8() % 12); /* E8M0: 2^-8 .. 2^3 */
            continue;
        }
        uint8_t *d = q->trailing_d ? blk + q->bytes - 2 : blk;
        d[0]       = 0x00; /* fp16(1.0) */
        d[1]       = 0x3C;
        if (q->dtype == GEIST_DTYPE_Q4_K) {
            blk[2] = 0x00; /* dmin = fp16(0.5) */
            blk[3] = 0x38;
        }
    }
}

static void dequant_row(const struct qfmt *q, const uint8_t *row, float out[static N_IN]) {
    if (q->dtype == GEIST_DTYPE_Q4_K)
        dequant_q4_K_row(N_IN, row, out);
    else if (q->dtype == GEIST_DTYPE_Q6_K)
        dequant_q6_K_row(N_IN, row, out);
    else if (q->dtype == GEIST_DTYPE_Q8_0)
        dequant_q8_0_row(N_IN, row, out);
    else if (q->dtype == GEIST_DTYPE_Q5_0)
        dequant_q5_0_row(N_IN, row, out);
    else
        dequant_mxfp4_row(N_IN, row, out);
}

/* ids pattern 0 random, 1 skewed, 2 uniform: k distinct experts a token. */
static void make_ids(int pattern, size_t m, size_t k, uint32_t ids[static m * k]) {
    for (size_t t = 0; t < m; t++) {
        bool used[N_EXP] = {false};
        for (size_t j = 0; j < k; j++) {
            uint32_t e;
            if (pattern == 2)
                e = (uint32_t) (3 + 2 * j) % N_EXP;
            else if (pattern == 1 && j == 0)
                e = 0;
            else {
                const uint32_t span = pattern == 1 ? N_EXP - 1 : N_EXP; /* skewed: never N_EXP-1 */
                do
                    e = rng_u32() % span;
                while (used[e]);
            }
            used[e]        = true;
            ids[t * k + j] = e;
        }
    }
}

static int test_grouped(struct geist_backend *be, const struct qfmt *q) {
    const struct geist_backend_vtbl  *v    = be->desc->vtbl;
    const struct geist_backend_fused *f    = be->desc->fused;
    const size_t                      rb   = (N_IN / q->elems) * q->bytes; /* one output row */
    const size_t                      wb   = N_OUT * rb;
    uint8_t                          *blob = malloc(N_EXP * wb + 16384);
    uint8_t *al = blob != nullptr ? blob + (16384 - ((uintptr_t) blob % 16384)) % 16384 : nullptr;
    struct geist_weight  w[N_EXP] = {0};
    struct geist_buffer *bw = nullptr, *bx = nullptr, *by = nullptr, *bi = nullptr;
    const size_t         nkmax = M_MAX * K_MAX;
    float               *xs    = malloc(nkmax * N_IN * sizeof *xs);
    float               *yg    = malloc(nkmax * N_OUT * sizeof *yg);
    float               *yg2   = malloc(nkmax * N_OUT * sizeof *yg2);
    float               *y1    = malloc(nkmax * N_OUT * sizeof *y1);
    float               *wd    = malloc(N_EXP * N_OUT * N_IN * sizeof *wd); /* dequantized */
    uint32_t            *ids   = malloc(nkmax * sizeof *ids);
    int                  fails = 0;
    if (blob == nullptr || xs == nullptr || yg == nullptr || yg2 == nullptr || y1 == nullptr ||
        wd == nullptr || ids == nullptr) {
        fails += geist_expect(false, "host memory");
        goto out;
    }
    fill(al, q, N_EXP * N_OUT);
    for (size_t r = 0; r < N_EXP * N_OUT; r++)
        dequant_row(q, al + r * rb, wd + r * N_IN);
    for (size_t e = 0; e < N_EXP; e++) {
        w[e] = (struct geist_weight) {.raw        = al + e * wb,
                                      .raw_nbytes = wb,
                                      .n_in       = N_IN,
                                      .n_out      = N_OUT,
                                      .dtype      = (uint16_t) q->dtype};
        fails += geist_expect(v->resolve_weight(be, &w[e]) == GEIST_OK, "resolve");
    }
    const struct geist_fusion_query qq = {.op     = GEIST_FUSED_MOE_DEVICE,
                                          .m      = 512,
                                          .gate_w = &w[0],
                                          .up_w   = &w[0],
                                          .down_w = &w[0]};
    /* #499: Vulkan's grouped GEMM needs tensor cores; without them (lavapipe)
     * the probe says no and linear_t_ids still runs a chunk (a matvec per
     * slot), which the checks below hold to the same bounds. */
    const bool chunk = f->supported(be, &qq);
    if (strcmp(be->desc->name, "metal") == 0)
        fails += geist_expect(chunk, "supported(GEIST_FUSED_MOE_DEVICE) at m = 512");
    else if (!chunk)
        printf("  %s: no grouped GEMM here (the probe says no at m = 512)\n", q->name);
    /* Not exact in half: the GEMM's staging rounds them. */
    for (size_t i = 0; i < nkmax * N_IN; i++)
        xs[i] = (float) rng_u8() / 64.0f - 2.0f + (float) rng_u8() / 65536.0f;
    fails += geist_expect(
            v->buffer_create_aliased(be, al, N_EXP * wb, GEIST_BUFFER_WEIGHT, &bw) == GEIST_OK &&
                    v->buffer_create(
                            be, nkmax * N_IN * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bx) ==
                            GEIST_OK &&
                    v->buffer_create(
                            be, nkmax * N_OUT * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &by) ==
                            GEIST_OK &&
                    v->buffer_create(be, nkmax * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) ==
                            GEIST_OK &&
                    v->buffer_upload(bx, nkmax * N_IN * 4, (const uint8_t *) xs) == GEIST_OK,
            "buffers");
    if (fails > 0)
        goto out;
    const struct geist_tensor ex   = {.buffer = bw,
                                      .offset = 0,
                                      .dtype  = (enum geist_dtype) q->dtype,
                                      .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
                                      .ndim   = 2,
                                      .shape  = {N_EXP * N_OUT, N_IN},
                                      .stride = {N_IN, 1}};
    static const size_t       ms[] = {9, 37, M_MAX}, ks[] = {1, 4, K_MAX};
    double                    worst_d = 0.0, worst_1 = 0.0;
    for (size_t mi = 0; mi < 3 && fails == 0; mi++)
        for (size_t ki = 0; ki < 3 && fails == 0; ki++)
            for (int pat = 0; pat < 3 && fails == 0; pat++)
                for (int per_slot = 0; per_slot < 2 && fails == 0; per_slot++) {
                    const size_t m = ms[mi], k = ks[ki], nk = m * k;
                    make_ids(pat, m, k, ids);
                    for (size_t i = 0; i < nk * N_OUT; i++)
                        yg[i] = NAN;
                    fails += geist_expect(
                            v->buffer_upload(bi, nk * 4, (const uint8_t *) ids) == GEIST_OK &&
                                    v->buffer_upload(by, nk * N_OUT * 4, (const uint8_t *) yg) ==
                                            GEIST_OK,
                            "upload");
                    struct geist_tensor tx = mat(bx, 0, per_slot ? nk : m, N_IN);
                    struct geist_tensor ty = mat(by, 0, nk, N_OUT);
                    for (int run = 0; run < 2; run++) {
                        fails += geist_expect(
                                f->linear_t_ids(
                                        be, N_EXP, m, k, per_slot != 0, &w[0], &ex, bi, &tx, &ty) ==
                                        GEIST_OK,
                                "linear_t_ids");
                        fails += geist_expect(v->buffer_download(nk * N_OUT * 4,
                                                                 (uint8_t *) (run ? yg2 : yg),
                                                                 by) == GEIST_OK,
                                              "download");
                    }
                    fails += geist_expect(memcmp(yg, yg2, nk * N_OUT * 4) == 0,
                                          "a second run is bit-identical");
                    /* The per-expert path: each expert's rows, gathered, as
                     * one linear_t; outputs back to their slots. */
                    float *xe = malloc(nk * N_IN * sizeof *xe);
                    float *ye = malloc(nk * N_OUT * sizeof *ye);
                    for (size_t e = 0; xe != nullptr && ye != nullptr && e < N_EXP; e++) {
                        size_t me = 0;
                        for (size_t s = 0; s < nk; s++)
                            if (ids[s] == e)
                                memcpy(xe + me++ * N_IN,
                                       xs + (per_slot ? s : s / k) * N_IN,
                                       N_IN * sizeof *xe);
                        if (me == 0)
                            continue;
                        struct geist_buffer *bxe = nullptr, *bye = nullptr;
                        fails += geist_expect(v->buffer_create(be,
                                                               me * N_IN * 4,
                                                               GEIST_BUFFER_IO,
                                                               GEIST_MEMORY_AUTO,
                                                               &bxe) == GEIST_OK &&
                                                      v->buffer_create(be,
                                                                       me * N_OUT * 4,
                                                                       GEIST_BUFFER_IO,
                                                                       GEIST_MEMORY_AUTO,
                                                                       &bye) == GEIST_OK &&
                                                      v->buffer_upload(bxe,
                                                                       me * N_IN * 4,
                                                                       (const uint8_t *) xe) ==
                                                              GEIST_OK,
                                              "per-expert buffers");
                        struct geist_tensor tx1 = mat(bxe, 0, me, N_IN);
                        struct geist_tensor ty1 = mat(bye, 0, me, N_OUT);
                        struct geist_tensor w1  = ex;
                        w1.offset               = e * wb;
                        w1.shape[0]             = N_OUT;
                        fails += geist_expect(
                                fails == 0 &&
                                        f->linear_t(be, &tx1, &w[e], &w1, me, &ty1) == GEIST_OK &&
                                        v->buffer_download(me * N_OUT * 4, (uint8_t *) ye, bye) ==
                                                GEIST_OK,
                                "per-expert linear_t");
                        size_t r = 0;
                        for (size_t s = 0; s < nk; s++)
                            if (ids[s] == e)
                                memcpy(y1 + s * N_OUT, ye + r++ * N_OUT, N_OUT * sizeof *y1);
                        if (bxe != nullptr)
                            v->buffer_destroy(be, bxe);
                        if (bye != nullptr)
                            v->buffer_destroy(be, bye);
                    }
                    fails += geist_expect(xe != nullptr && ye != nullptr, "host memory");
                    free(xe);
                    free(ye);
                    double ed = 0.0, e1 = 0.0, scale = 0.0;
                    size_t nan = 0;
                    for (size_t s = 0; s < nk; s++) {
                        const float *xr = xs + (per_slot ? s : s / k) * N_IN;
                        for (size_t o = 0; o < N_OUT; o++) {
                            const float *wr  = wd + (ids[s] * N_OUT + o) * N_IN;
                            double       ref = 0.0;
                            for (size_t i = 0; i < N_IN; i++)
                                ref += (double) wr[i] * (double) xr[i];
                            const double g = yg[s * N_OUT + o];
                            nan += isnan(g);
                            ed    = fmax(ed, fabs(ref - g));
                            e1    = fmax(e1, fabs((double) y1[s * N_OUT + o] - g));
                            scale = fmax(scale, fabs(ref));
                        }
                    }
                    worst_d = fmax(worst_d, scale > 0.0 ? ed / scale : 1.0);
                    worst_1 = fmax(worst_1, scale > 0.0 ? e1 / scale : 1.0);
                    if (nan > 0 || !(ed <= 1e-3 * scale) || !(e1 <= 1e-3 * scale))
                        fprintf(stderr,
                                "  %s m=%zu k=%zu ids %d x per %s: %zu unwritten, vs double %.3g, "
                                "vs per-expert %.3g of scale %.3g\n",
                                q->name,
                                m,
                                k,
                                pat,
                                per_slot ? "slot" : "token",
                                nan,
                                ed,
                                e1,
                                scale);
                    fails += geist_expect(nan == 0, "every slot's row is written");
                    fails += geist_expect(ed <= 1e-3 * scale, "grouped GEMM matches double");
                    fails += geist_expect(e1 <= 1e-3 * scale, "grouped GEMM matches per-expert");
                }
    printf("  %-5s grouped linear_t_ids: max |d|/scale %.3g vs double, %.3g vs per-expert "
           "linear_t\n",
           q->name,
           worst_d,
           worst_1);
out:
    struct geist_buffer *bs[] = {bw, bx, by, bi};
    for (size_t i = 0; i < 4; i++)
        if (bs[i] != nullptr)
            v->buffer_destroy(be, bs[i]);
    free(blob);
    free(xs);
    free(yg);
    free(yg2);
    free(y1);
    free(wd);
    free(ids);
    return fails;
}

/* ---- the experts' FFN in one call (#486) --------------------------------- */

enum { F_OUT = 136, F_CASES = 11 };

/* The rows a backend's moe_ffn_ids takes: Metal a chunk's (#486, from 9),
 * Vulkan a decode step's with Q4_K gate and up (#505, up to 8); every other
 * case returns GEIST_E_UNSUPPORTED (the four calls run instead). */
static bool ffn_takes(const struct geist_backend *be, const struct qfmt *q, size_t m) {
    if (strcmp(be->desc->name, "vulkan") == 0)
        return m <= 8 && q->dtype == GEIST_DTYPE_Q4_K;
    return m >= 9;
}

/* Uploads n_exp experts of rows x cols of q (random blocks) as one weight
 * buffer; their dequantized rows into wd. */
static int make_experts(struct geist_backend *be,
                        const struct qfmt    *q,
                        size_t                rows,
                        size_t                cols,
                        uint8_t             **blob,
                        struct geist_weight   w[static N_EXP],
                        struct geist_buffer **buf,
                        struct geist_tensor  *ex,
                        float                *wd) {
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const size_t                     rb = (cols / q->elems) * q->bytes, wb = rows * rb;
    *blob = malloc(N_EXP * wb + 16384);
    if (*blob == nullptr)
        return geist_expect(false, "host memory");
    uint8_t *al = *blob + (16384 - ((uintptr_t) *blob % 16384)) % 16384;
    for (size_t b = 0; b < N_EXP * rows * (cols / q->elems); b++) {
        uint8_t *blk = al + b * q->bytes;
        for (size_t i = 0; i < q->bytes; i++)
            blk[i] = rng_u8();
        if (q->dtype == GEIST_DTYPE_MXFP4) {
            blk[0] = (uint8_t) (120 + rng_u8() % 4); /* E8M0: 2^-7 .. 2^-4 */
            continue;
        }
        /* d (and Q4_K's dmin) such that a weight is ~0.1 in size: the
         * activations stay where a model's are, well inside half's range
         * (x and g are staged as half). */
        const uint8_t dh = q->dtype == GEIST_DTYPE_Q4_K   ? 0x14  /* 2^-10 */
                           : q->dtype == GEIST_DTYPE_Q6_K ? 0x0C  /* 2^-12 */
                           : q->dtype == GEIST_DTYPE_Q8_0 ? 0x1C  /* 2^-8 */
                                                          : 0x24; /* 2^-6 */
        uint8_t      *d  = q->trailing_d ? blk + q->bytes - 2 : blk;
        d[0]             = 0x00;
        d[1]             = dh;
        if (q->dtype == GEIST_DTYPE_Q4_K) {
            blk[2] = 0x00;
            blk[3] = dh;
        }
    }
    int fails = 0;
    for (size_t r = 0; r < N_EXP * rows; r++) {
        if (q->dtype == GEIST_DTYPE_Q4_K)
            dequant_q4_K_row(cols, al + r * rb, wd + r * cols);
        else if (q->dtype == GEIST_DTYPE_Q6_K)
            dequant_q6_K_row(cols, al + r * rb, wd + r * cols);
        else if (q->dtype == GEIST_DTYPE_Q8_0)
            dequant_q8_0_row(cols, al + r * rb, wd + r * cols);
        else if (q->dtype == GEIST_DTYPE_Q5_0)
            dequant_q5_0_row(cols, al + r * rb, wd + r * cols);
        else
            dequant_mxfp4_row(cols, al + r * rb, wd + r * cols);
    }
    for (size_t e = 0; e < N_EXP; e++) {
        w[e] = (struct geist_weight) {.raw        = al + e * wb,
                                      .raw_nbytes = wb,
                                      .n_in       = (int64_t) cols,
                                      .n_out      = (int64_t) rows,
                                      .dtype      = (uint16_t) q->dtype};
        fails += geist_expect(be->desc->vtbl->resolve_weight(be, &w[e]) == GEIST_OK, "resolve");
    }
    fails += geist_expect(v->buffer_create_aliased(be, al, N_EXP * wb, GEIST_BUFFER_WEIGHT, buf) ==
                                  GEIST_OK,
                          "expert buffer");
    *ex = (struct geist_tensor) {.buffer = *buf,
                                 .offset = 0,
                                 .dtype  = (enum geist_dtype) q->dtype,
                                 .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
                                 .ndim   = 2,
                                 .shape  = {(int64_t) (N_EXP * rows), (int64_t) cols},
                                 .stride = {(int64_t) cols, 1}};
    return fails;
}

static double ref_act(bool oai, double a, double b) {
    if (!oai)
        return a / (1.0 + exp(-a)) * b;
    a = fmin(a, (double) GEIST_SWIGLU_OAI_LIMIT);
    b = fmax(fmin(b, (double) GEIST_SWIGLU_OAI_LIMIT), -(double) GEIST_SWIGLU_OAI_LIMIT);
    return a / (1.0 + exp(-(double) GEIST_SWIGLU_OAI_ALPHA * a)) * (b + 1.0);
}

/* fused->moe_ffn_ids for gate / up of `inter` outputs and down of F_OUT,
 * all of dtype q, with the plain SwiGLU (no biases) and with swiglu_oai
 * and both biases: against the four calls it fuses (linear_t_ids gate and
 * up, glu or silu_mul, linear_t_ids down), against a double-precision FFN
 * of the dequantized weights, and against its own second run. */
static int test_ffn(struct geist_backend *be, const struct qfmt *q, size_t inter) {
    const struct geist_backend_vtbl  *v                 = be->desc->vtbl;
    const struct geist_backend_fused *f                 = be->desc->fused;
    static const size_t               cases[F_CASES][3] = /* m, k, ids pattern */
            {{1, 8, 0},
             {1, 6, 1},
             {4, 8, 2},
             {8, 6, 0},
             {9, 1, 0},
             {9, 6, 1},
             {37, 4, 0},
             {37, 4, 2},
             {37, 6, 1},
             {300, 6, 0},
             {300, 4, 1}};
    const size_t         nkmax   = M_MAX * K_MAX;
    uint8_t             *blob[3] = {nullptr, nullptr, nullptr};
    struct geist_weight  w[3][N_EXP];
    struct geist_buffer *bw[3] = {nullptr, nullptr, nullptr};
    struct geist_tensor  ex[3];
    struct geist_buffer *bx = nullptr, *bg = nullptr, *bu = nullptr, *by = nullptr, *bi = nullptr,
                        *bb = nullptr;
    float    *wd[3]         = {malloc(N_EXP * inter * N_IN * sizeof(float)),
                               malloc(N_EXP * inter * N_IN * sizeof(float)),
                               malloc(N_EXP * F_OUT * inter * sizeof(float))};
    float    *xs            = malloc(M_MAX * N_IN * sizeof *xs);
    float    *bias          = malloc(2 * N_EXP * inter * sizeof *bias);
    float    *gf = malloc(nkmax * inter * sizeof *gf), *g1 = malloc(nkmax * inter * sizeof *g1);
    float    *yf = malloc(nkmax * F_OUT * sizeof *yf), *y1 = malloc(nkmax * F_OUT * sizeof *y1);
    float    *yf2   = malloc(nkmax * F_OUT * sizeof *yf2);
    double   *gr    = malloc(inter * sizeof *gr);
    uint32_t *ids   = malloc(nkmax * sizeof *ids);
    int       fails = 0;
    if (wd[0] == nullptr || wd[1] == nullptr || wd[2] == nullptr || xs == nullptr ||
        bias == nullptr || gf == nullptr || g1 == nullptr || yf == nullptr || y1 == nullptr ||
        yf2 == nullptr || gr == nullptr || ids == nullptr) {
        fails += geist_expect(false, "host memory");
        goto out;
    }
    for (size_t p = 0; p < 3 && fails == 0; p++)
        fails += make_experts(be,
                              q,
                              p < 2 ? inter : F_OUT,
                              p < 2 ? N_IN : inter,
                              &blob[p],
                              w[p],
                              &bw[p],
                              &ex[p],
                              wd[p]);
    for (size_t i = 0; i < M_MAX * N_IN; i++)
        xs[i] = (float) rng_u8() / 64.0f - 2.0f + (float) rng_u8() / 65536.0f;
    for (size_t i = 0; i < 2 * N_EXP * inter; i++)
        bias[i] = (float) rng_u8() / 128.0f - 1.0f;
    fails += geist_expect(
            fails == 0 &&
                    v->buffer_create(
                            be, M_MAX * N_IN * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bx) ==
                            GEIST_OK &&
                    v->buffer_create(
                            be, nkmax * inter * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bg) ==
                            GEIST_OK &&
                    v->buffer_create(
                            be, nkmax * inter * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bu) ==
                            GEIST_OK &&
                    v->buffer_create(
                            be, nkmax * F_OUT * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &by) ==
                            GEIST_OK &&
                    v->buffer_create(be, nkmax * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) ==
                            GEIST_OK &&
                    v->buffer_create(
                            be, 2 * N_EXP * inter * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bb) ==
                            GEIST_OK &&
                    v->buffer_upload(bx, M_MAX * N_IN * 4, (const uint8_t *) xs) == GEIST_OK &&
                    v->buffer_upload(bb, 2 * N_EXP * inter * 4, (const uint8_t *) bias) == GEIST_OK,
            "buffers");
    if (fails > 0)
        goto out;
    const struct geist_tensor tbg      = mat(bb, 0, N_EXP, inter);
    const struct geist_tensor tbu      = mat(bb, N_EXP * inter * sizeof(float), N_EXP, inter);
    double                    worst_g1 = 0.0, worst_y1 = 0.0, worst_gd = 0.0, worst_yd = 0.0;
    size_t                    same = 0, total = 0;
    for (int oai = 0; oai < 2 && fails == 0; oai++)
        for (size_t c = 0; c < F_CASES && fails == 0; c++) {
            const size_t m = cases[c][0], k = cases[c][1], nk = m * k;
            make_ids((int) cases[c][2], m, k, ids);
            for (size_t i = 0; i < nk * inter; i++)
                gf[i] = NAN;
            for (size_t i = 0; i < nk * F_OUT; i++)
                yf[i] = NAN;
            fails += geist_expect(
                    v->buffer_upload(bi, nk * 4, (const uint8_t *) ids) == GEIST_OK &&
                            v->buffer_upload(bg, nk * inter * 4, (const uint8_t *) gf) ==
                                    GEIST_OK &&
                            v->buffer_upload(by, nk * F_OUT * 4, (const uint8_t *) yf) == GEIST_OK,
                    "upload");
            struct geist_tensor             tx = mat(bx, 0, m, N_IN), tg = mat(bg, 0, nk, inter),
                                            tu = mat(bu, 0, nk, inter), ty = mat(by, 0, nk, F_OUT);
            const struct geist_glu_args     ga = {.act    = GEIST_GLU_SWIGLU_OAI,
                                                  .alpha  = GEIST_SWIGLU_OAI_ALPHA,
                                                  .limit  = GEIST_SWIGLU_OAI_LIMIT,
                                                  .x_bias = &tbg,
                                                  .z_bias = &tbu,
                                                  .ids    = bi};
            const struct geist_moe_ffn_args fa = {.n       = N_EXP,
                                                  .m       = m,
                                                  .k       = k,
                                                  .glu     = oai ? &ga : nullptr,
                                                  .w0      = {&w[0][0], &w[1][0], &w[2][0]},
                                                  .experts = {&ex[0], &ex[1], &ex[2]},
                                                  .ids     = bi,
                                                  .x       = &tx,
                                                  .g       = &tg,
                                                  .y       = &ty};
            if (!ffn_takes(be, q, m)) {
                fails += geist_expect(f->moe_ffn_ids(be, &fa) == GEIST_E_UNSUPPORTED,
                                      "rows the backend does not fuse: GEIST_E_UNSUPPORTED");
                continue;
            }
            for (int run = 0; run < 2; run++)
                fails += geist_expect(
                        f->moe_ffn_ids(be, &fa) == GEIST_OK &&
                                v->buffer_download(nk * F_OUT * 4,
                                                   (uint8_t *) (run ? yf2 : yf),
                                                   by) == GEIST_OK &&
                                v->buffer_download(nk * inter * 4, (uint8_t *) gf, bg) == GEIST_OK,
                        "moe_ffn_ids");
            fails += geist_expect(memcmp(yf, yf2, nk * F_OUT * 4) == 0,
                                  "a second run is bit-identical");
            /* The four calls it fuses, the arch's decomposed route. */
            fails += geist_expect(
                    f->linear_t_ids(be, N_EXP, m, k, false, &w[0][0], &ex[0], bi, &tx, &tg) ==
                                    GEIST_OK &&
                            f->linear_t_ids(
                                    be, N_EXP, m, k, false, &w[1][0], &ex[1], bi, &tx, &tu) ==
                                    GEIST_OK &&
                            (oai ? f->glu(be, &ga, &tg, &tu, &tg)
                                 : f->silu_mul(be, &tg, &tu, &tg)) == GEIST_OK &&
                            f->linear_t_ids(
                                    be, N_EXP, m, k, true, &w[2][0], &ex[2], bi, &tg, &ty) ==
                                    GEIST_OK &&
                            v->buffer_download(nk * inter * 4, (uint8_t *) g1, bg) == GEIST_OK &&
                            v->buffer_download(nk * F_OUT * 4, (uint8_t *) y1, by) == GEIST_OK,
                    "the four calls");
            double eg1 = 0.0, ey1 = 0.0, egd = 0.0, eyd = 0.0, sg = 0.0, sy = 0.0;
            size_t nan = 0;
            for (size_t s = 0; s < nk; s++) {
                const size_t e  = ids[s];
                const float *xr = xs + (s / k) * N_IN;
                for (size_t o = 0; o < inter; o++) {
                    const float *wg = wd[0] + (e * inter + o) * N_IN;
                    const float *wu = wd[1] + (e * inter + o) * N_IN;
                    double       a = 0.0, b = 0.0;
                    for (size_t i = 0; i < N_IN; i++) {
                        a += (double) wg[i] * (double) xr[i];
                        b += (double) wu[i] * (double) xr[i];
                    }
                    if (oai) {
                        a += bias[e * inter + o];
                        b += bias[(N_EXP + e) * inter + o];
                    }
                    gr[o]          = ref_act(oai != 0, a, b);
                    const double g = gf[s * inter + o];
                    nan += isnan(g);
                    egd = fmax(egd, fabs(gr[o] - g));
                    eg1 = fmax(eg1, fabs((double) g1[s * inter + o] - g));
                    sg  = fmax(sg, fabs(gr[o]));
                }
                for (size_t o = 0; o < F_OUT; o++) {
                    const float *wr  = wd[2] + (e * F_OUT + o) * inter;
                    double       ref = 0.0;
                    for (size_t i = 0; i < inter; i++)
                        ref += (double) wr[i] * gr[i];
                    const double y = yf[s * F_OUT + o];
                    nan += isnan(y);
                    eyd = fmax(eyd, fabs(ref - y));
                    ey1 = fmax(ey1, fabs((double) y1[s * F_OUT + o] - y));
                    sy  = fmax(sy, fabs(ref));
                }
            }
            same += memcmp(gf, g1, nk * inter * 4) == 0 && memcmp(yf, y1, nk * F_OUT * 4) == 0;
            total++;
            worst_g1      = fmax(worst_g1, sg > 0.0 ? eg1 / sg : 1.0);
            worst_y1      = fmax(worst_y1, sy > 0.0 ? ey1 / sy : 1.0);
            worst_gd      = fmax(worst_gd, sg > 0.0 ? egd / sg : 1.0);
            worst_yd      = fmax(worst_yd, sy > 0.0 ? eyd / sy : 1.0);
            const bool ok = nan == 0 && eg1 <= 1e-5 * sg && ey1 <= 1e-5 * sy && egd <= 2e-3 * sg &&
                            eyd <= 2e-3 * sy;
            if (!ok)
                fprintf(stderr,
                        "  %s %s m=%zu k=%zu ids %zu: %zu unwritten; g vs the four calls %.3g, "
                        "vs double %.3g of %.3g; y %.3g, %.3g of %.3g\n",
                        q->name,
                        oai ? "swiglu_oai" : "silu",
                        m,
                        k,
                        cases[c][2],
                        nan,
                        eg1,
                        egd,
                        sg,
                        ey1,
                        eyd,
                        sy);
            fails += geist_expect(nan == 0, "every slot's g and y row is written");
            fails += geist_expect(eg1 <= 1e-5 * sg && ey1 <= 1e-5 * sy,
                                  "moe_ffn_ids matches the four calls");
            fails += geist_expect(egd <= 2e-3 * sg && eyd <= 2e-3 * sy,
                                  "moe_ffn_ids matches double");
        }
    if (total > 0)
        printf("  %-5s moe_ffn_ids (inter %zu): g / y max |d|/scale %.3g / %.3g vs the four calls "
               "(%zu "
               "of %zu cases bit-identical), %.3g / %.3g vs double\n",
               q->name,
               inter,
               worst_g1,
               worst_y1,
               same,
               total,
               worst_gd,
               worst_yd);
    /* Gate and up of different dtypes: not here, at rows it otherwise takes. */
    const size_t              mt = ffn_takes(be, q, 9) ? 9 : 1;
    struct geist_tensor       tx = mat(bx, 0, mt, N_IN), tg = mat(bg, 0, mt, inter),
                              ty       = mat(by, 0, mt, F_OUT);
    struct geist_moe_ffn_args few      = {.n       = N_EXP,
                                          .m       = mt,
                                          .k       = 1,
                                          .w0      = {&w[0][0], &w[1][0], &w[2][0]},
                                          .experts = {&ex[0], &ex[1], &ex[2]},
                                          .ids     = bi,
                                          .x       = &tx,
                                          .g       = &tg,
                                          .y       = &ty};
    struct geist_weight       wu_other = w[1][0];
    wu_other.dtype =
            (uint16_t) (q->dtype == GEIST_DTYPE_Q8_0 ? GEIST_DTYPE_Q5_0 : GEIST_DTYPE_Q8_0);
    few.w0[1] = &wu_other;
    fails += geist_expect(f->moe_ffn_ids(be, &few) == GEIST_E_UNSUPPORTED,
                          "gate and up of different dtypes: GEIST_E_UNSUPPORTED");
out:
    struct geist_buffer *bs[] = {bw[0], bw[1], bw[2], bx, bg, bu, by, bi, bb};
    for (size_t i = 0; i < sizeof bs / sizeof bs[0]; i++)
        if (bs[i] != nullptr)
            v->buffer_destroy(be, bs[i]);
    for (size_t p = 0; p < 3; p++) {
        free(blob[p]);
        free(wd[p]);
    }
    free(xs);
    free(bias);
    free(gf);
    free(g1);
    free(yf);
    free(y1);
    free(yf2);
    free(gr);
    free(ids);
    return fails;
}

/* F32 experts have no grouped GEMM: the probe keeps a chunk on the host. */
static int test_f32_refused(struct geist_backend *be) {
    static float              wf[N_OUT * N_IN];
    struct geist_weight       w  = {.raw        = wf,
                                    .raw_nbytes = sizeof wf,
                                    .n_in       = N_IN,
                                    .n_out      = N_OUT,
                                    .dtype      = GEIST_DTYPE_F32};
    struct geist_fusion_query q1 = {
            .op = GEIST_FUSED_MOE_DEVICE, .m = 1, .gate_w = &w, .up_w = &w, .down_w = &w};
    struct geist_fusion_query qm = q1;
    qm.m                         = 512;
    return geist_expect(be->desc->fused->supported(be, &q1) && !be->desc->fused->supported(be, &qm),
                        "F32 experts: decode yes, a 512-row chunk no");
}

int main(void) {
    static const char *const names[] = {"metal", "vulkan"};
    int                      fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof names / sizeof names[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(names[b], nullptr, nullptr, &be) != GEIST_OK)
            continue;
        const struct geist_backend_fused *f = be->desc->fused;
        if (f == nullptr || f->moe_route == nullptr || f->linear_t_ids == nullptr ||
            f->supported == nullptr) {
            geist_backend_destroy(be);
            continue;
        }
        printf("%s:\n", names[b]);
        ran++;
        fails += test_route_chunk(be);
        static const struct qfmt qs[] = {
                {GEIST_DTYPE_Q4_K, 256, 144, false, "Q4_K"},
                {GEIST_DTYPE_Q6_K, 256, 210, true, "Q6_K"},
                {GEIST_DTYPE_Q8_0, 32, 34, false, "Q8_0"},
                {GEIST_DTYPE_MXFP4, 32, 17, false, "MXFP4"},
                {GEIST_DTYPE_Q5_0, 32, 22, false, "Q5_0"},
        };
        for (size_t i = 0; i < sizeof qs / sizeof qs[0]; i++)
            fails += test_grouped(be, &qs[i]);
        if (f->moe_ffn_ids != nullptr) /* down's n_in: whole blocks; 160 a ragged tile */
            for (size_t i = 0; i < sizeof qs / sizeof qs[0]; i++)
                fails += test_ffn(be, &qs[i], qs[i].elems == 256 ? 256 : 160);
        fails += test_f32_refused(be);
        geist_backend_destroy(be);
    }
    if (ran == 0) {
        printf("SKIP: no backend in this build routes a mixture of experts on the device\n");
        return 0;
    }
    if (fails == 0)
        printf("PASS: chunk routing, grouped expert GEMMs and the fused expert FFN\n");
    return fails == 0 ? 0 : 1;
}
