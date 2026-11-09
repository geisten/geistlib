/*
 * test_moe_device_unit - mixture-of-experts decode on the device (#201):
 * fused->moe_route, fused->linear_t_ids and fused->moe_weighted_sum on the
 * backends that have them (Metal), against host references.
 *
 * moe_route against moe_route (the host's router, the reference both
 * engines' routing is measured by) on #48's edge cases: random logits, ties
 * in the scores (ranked by id), k = n, an all-zero row, renormalised and
 * not; and (#373) with a router bias, SOFTMAX_WEIGHT gating (GPT-OSS: the
 * softmax over the selected logits alone, which the host computes in
 * double) and both; ids exact, weights within 1e-6. A non-finite logit sets
 * the flag word after the ids and still writes ids below n. (#376)
 * DeepSeek's router: sigmoid gating with a selection bias, renormalised and
 * not, a weight scale, softmax with a selection bias, and expert groups
 * (V3: 8 groups of which 4 stay, here over 64 experts and over 16 with ties
 * between groups), ids exact and weights within 1e-6 of the host's
 * moe_route; the probe agrees.
 *
 * linear_t_ids, for F32, Q4_K, Q6_K, Q8_0, MXFP4 and (#376) Q5_0: 10 experts back to back
 * in one buffer, the ids of two tokens' three slots (one expert twice, the
 * last expert), x one row per token (gate/up) and one row per slot (down);
 * against fused->linear_t of each slot's expert alone, to 1e-5 of the
 * output scale (the same matvec bodies; MXFP4's decode GEMV is another
 * kernel, the same arithmetic). MXFP4 also against a double-precision
 * product of the reference decoder's weights (dequant_mxfp4_row), to 1e-5
 * of the scale. supported(GEIST_FUSED_MOE_DEVICE) says yes for these dtypes
 * at m = 1 and at m = 8 (a batched decode's rows, #245), no at m = 0.
 *
 * glu (#373): silu and swiglu_oai (alpha 1.702, limit 7, inputs past the
 * limit), without biases, with bias rows by the routed ids and with one
 * expert's bias row for all rows, y aliasing x; against the formula in
 * double, to 1e-5 of the scale.
 *
 * moe_weighted_sum: two tokens of three slots onto their rows of `add`,
 * without and (#373) with the experts' output biases by the ids, against
 * the host sum in double to 1e-5 of its scale.
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

enum { N_EXP = 10, N_IN = 512, N_OUT = 136, M = 2, K = 3 };

static uint32_t rng = 0x13579bdfu;
static uint8_t  rng_u8(void) {
    rng = rng * 1664525u + 1013904223u;
    return (uint8_t) (rng >> 24);
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

/* Routes `rows` rows of `n` logits with k on the device and checks each
 * against the host's moe_route with the same router. */
static int check_route_opts(struct geist_backend    *be,
                            const char              *what,
                            size_t                   rows,
                            size_t                   n,
                            size_t                   k,
                            const struct route_opts *o,
                            const float             *logits,
                            bool                     expect_bad) {
    const struct geist_backend_vtbl  *v = be->desc->vtbl;
    const struct geist_backend_fused *f = be->desc->fused;
    struct geist_buffer *bl = nullptr, *bi = nullptr, *bw = nullptr, *bb = nullptr, *bs = nullptr;
    int                  fails = 0;
    const size_t         nk    = rows * k;
    if (nk > 64)
        return geist_expect(false, "route case fits the test's arrays");
    if (v->buffer_create(be, rows * n * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bl) != GEIST_OK ||
        v->buffer_create(be, (nk + 1) * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) != GEIST_OK ||
        v->buffer_create(be, nk * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bw) != GEIST_OK ||
        v->buffer_create(be, n * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bb) != GEIST_OK ||
        v->buffer_create(be, n * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bs) != GEIST_OK)
        return geist_expect(false, "route buffers");
    uint32_t zero[64 + 1] = {0};
    fails += geist_expect(
            v->buffer_upload(bl, rows * n * 4, (const uint8_t *) logits) == GEIST_OK &&
                    v->buffer_upload(bi, (nk + 1) * 4, (const uint8_t *) zero) == GEIST_OK &&
                    (o->bias == nullptr ||
                     v->buffer_upload(bb, n * 4, (const uint8_t *) o->bias) == GEIST_OK) &&
                    (o->sel == nullptr ||
                     v->buffer_upload(bs, n * 4, (const uint8_t *) o->sel) == GEIST_OK),
            "route upload");
    struct geist_tensor               tl = mat(bl, 0, rows, n), tw = vec(bw, 0, nk);
    struct geist_tensor               tb = vec(bb, 0, n), ts = vec(bs, 0, n);
    const struct geist_moe_route_args ra = {.m          = rows,
                                            .n          = n,
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
    uint32_t ids[64 + 1];
    float    w[64];
    fails += geist_expect(v->buffer_download((nk + 1) * 4, (uint8_t *) ids, bi) == GEIST_OK &&
                                  v->buffer_download(nk * 4, (uint8_t *) w, bw) == GEIST_OK,
                          "route download");
    fails += geist_expect((ids[nk] != 0) == expect_bad,
                          expect_bad ? "a non-finite logit sets the flag" : "no flag");
    double dw = 0.0;
    for (size_t r = 0; r < rows && fails == 0; r++) {
        uint32_t ref_ids[64];
        float    ref_w[64];
        if (expect_bad) {
            for (size_t j = 0; j < k; j++)
                fails += geist_expect(ids[r * k + j] < n, "ids stay below n");
            continue;
        }
        fails += geist_expect(moe_route(n, k, &hr, logits + r * n, ref_ids, ref_w),
                              "host reference");
        for (size_t j = 0; j < k; j++) {
            dw = fmax(dw, fabs((double) w[r * k + j] - ref_w[j]));
            if (ids[r * k + j] != ref_ids[j] || fabsf(w[r * k + j] - ref_w[j]) > 1e-6f) {
                fprintf(stderr,
                        "  %s row %zu slot %zu: id %u w %.8f, host id %u w %.8f\n",
                        what,
                        r,
                        j,
                        ids[r * k + j],
                        (double) w[r * k + j],
                        ref_ids[j],
                        (double) ref_w[j]);
                fails += geist_expect(false, what);
                break;
            }
        }
    }
    if (fails == 0 && !expect_bad)
        printf("  %s: ids exact, max |dw| %.3g\n", what, dw);
    v->buffer_destroy(be, bl);
    v->buffer_destroy(be, bi);
    v->buffer_destroy(be, bw);
    v->buffer_destroy(be, bb);
    v->buffer_destroy(be, bs);
    return fails;
}

static int check_route(struct geist_backend *be,
                       const char           *what,
                       size_t                rows,
                       size_t                n,
                       size_t                k,
                       bool                  norm,
                       bool                  sel_softmax,
                       const float          *bias,
                       const float          *logits,
                       bool                  expect_bad) {
    const struct route_opts o = {.gating = sel_softmax ? GEIST_MOE_ROUTE_SOFTMAX_WEIGHT
                                                       : GEIST_MOE_ROUTE_SOFTMAX,
                                 .norm   = norm,
                                 .bias   = bias};
    return check_route_opts(be, what, rows, n, k, &o, logits, expect_bad);
}

/* #376: DeepSeek's router against the host's. */
static int test_route_deepseek(struct geist_backend *be) {
    enum { NE = 64 };
    int   fails = 0;
    float lg[4 * NE], sel[NE], ties[2 * 16], sel16[16];
    for (size_t i = 0; i < 4 * NE; i++)
        lg[i] = (float) rng_u8() / 32.0f - 4.0f;
    for (size_t i = 0; i < NE; i++)
        sel[i] = (float) rng_u8() / 512.0f - 0.25f;
    /* Groups of 4 over 16 experts with equal group scores (the best two of
     * groups 0 and 2 tie) and equal expert scores across groups. */
    for (size_t i = 0; i < 16; i++) {
        ties[i]      = (float) (i % 4) - 1.0f;
        ties[16 + i] = i < 8 ? 1.0f : (float) rng_u8() / 64.0f - 2.0f;
        sel16[i]     = 0.0f;
    }
    const struct {
        const char       *what;
        size_t            rows, n, k;
        struct route_opts o;
        const float      *logits;
    } cases[] = {
            {"sigmoid, selection bias, renormalised (GigaChat3)",
             4,
             NE,
             4,
             {.gating = GEIST_MOE_ROUTE_SIGMOID, .norm = true, .sel = sel},
             lg},
            {"sigmoid, selection bias, not renormalised",
             4,
             NE,
             6,
             {.gating = GEIST_MOE_ROUTE_SIGMOID, .sel = sel},
             lg},
            {"sigmoid, renormalised, scale 2.5",
             4,
             NE,
             8,
             {.gating = GEIST_MOE_ROUTE_SIGMOID, .norm = true, .scale = 2.5f},
             lg},
            {"softmax, selection bias, scale 16",
             4,
             NE,
             6,
             {.gating = GEIST_MOE_ROUTE_SOFTMAX, .sel = sel, .scale = 16.0f},
             lg},
            {"V3: sigmoid, bias, 8 groups keep 4, renormalised, scale 2.5",
             4,
             NE,
             8,
             {.gating     = GEIST_MOE_ROUTE_SIGMOID,
              .norm       = true,
              .sel        = sel,
              .scale      = 2.5f,
              .n_group    = 8,
              .topk_group = 4},
             lg},
            {"softmax, 4 groups keep 2, k = 8 = all eligible",
             4,
             NE,
             8,
             {.gating = GEIST_MOE_ROUTE_SOFTMAX, .n_group = 4, .topk_group = 2},
             lg},
            {"groups with ties between groups and experts",
             2,
             16,
             3,
             {.gating = GEIST_MOE_ROUTE_SIGMOID, .sel = sel16, .n_group = 4, .topk_group = 2},
             ties},
            {"selected softmax with a selection bias",
             4,
             NE,
             4,
             {.gating = GEIST_MOE_ROUTE_SOFTMAX_WEIGHT, .sel = sel},
             lg},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        fails += check_route_opts(be,
                                  cases[i].what,
                                  cases[i].rows,
                                  cases[i].n,
                                  cases[i].k,
                                  &cases[i].o,
                                  cases[i].logits,
                                  false);
    }
    float bad[NE];
    memcpy(bad, lg, sizeof bad);
    bad[17]                   = INFINITY;
    const struct route_opts o = {.gating = GEIST_MOE_ROUTE_SIGMOID, .norm = true, .sel = sel};
    fails += check_route_opts(be, "sigmoid with an infinity", 1, NE, 4, &o, bad, true);
    return fails;
}

static int test_route(struct geist_backend *be) {
    int   fails = 0;
    float lg[4 * 16];
    for (size_t i = 0; i < 16; i++)
        lg[i] = (float) rng_u8() / 32.0f - 4.0f; /* random */
    for (size_t i = 0; i < 16; i++)
        lg[16 + i] = (float) (i % 4); /* ties: 3 at 3, 7, 11, 15 */
    for (size_t i = 0; i < 16; i++)
        lg[32 + i] = 0.0f; /* all zero */
    for (size_t i = 0; i < 16; i++)
        lg[48 + i] = (float) rng_u8() / 8.0f - 16.0f;
    float bias[16];
    for (size_t i = 0; i < 16; i++)
        bias[i] = (float) rng_u8() / 64.0f - 2.0f;
    fails += check_route(be, "route k=4 renormalised", 4, 16, 4, true, false, nullptr, lg, false);
    fails += check_route(be, "route k=4 softmax", 4, 16, 4, false, false, nullptr, lg, false);
    fails += check_route(be, "route k=n", 4, 16, 16, true, false, nullptr, lg, false);
    fails += check_route(be, "route k=1", 4, 16, 1, false, false, nullptr, lg, false);
    fails += check_route(be, "route with a bias", 4, 16, 4, false, false, bias, lg, false);
    fails += check_route(
            be, "route softmax over the selected", 4, 16, 4, false, true, nullptr, lg, false);
    fails += check_route(
            be, "route GPT-OSS (bias, selected)", 4, 16, 4, false, true, bias, lg, false);
    fails += check_route(be, "route selected, k=1", 4, 16, 1, false, true, bias, lg, false);
    float bad[16];
    memcpy(bad, lg, sizeof bad);
    bad[5] = NAN;
    fails += check_route(be, "route with a NaN", 1, 16, 4, true, false, nullptr, bad, true);
    fails += check_route(be, "route with a NaN and a bias", 1, 16, 4, false, true, bias, bad, true);

    /* #514: the router widths of OLMoE (64, top 8) and Qwen3-MoE (128, top
     * 8, renormalised), the widest the one-subgroup path takes on 32 lanes
     * (256) and one past it, with ties across lanes and a NaN. */
    static float wide[2 * 300], wbias[300];
    for (size_t i = 0; i < 300; i++) {
        wide[i]       = (float) rng_u8() / 32.0f - 4.0f;
        wide[300 + i] = (float) ((i * 7) % 5); /* ties: every fifth value */
        wbias[i]      = (float) rng_u8() / 64.0f - 2.0f;
    }
    static const size_t widths[] = {64, 128, 256, 257};
    for (size_t i = 0; i < sizeof widths / sizeof widths[0]; i++) {
        const size_t n = widths[i];
        float        rows[2 * 300];
        memcpy(rows, wide, n * sizeof(float));
        memcpy(rows + n, wide + 300, n * sizeof(float));
        char what[64];
        snprintf(what, sizeof what, "route n=%zu k=8", n);
        fails += check_route(be, what, 2, n, 8, false, false, nullptr, rows, false);
        snprintf(what, sizeof what, "route n=%zu k=8 renormalised, bias", n);
        fails += check_route(be, what, 2, n, 8, true, false, wbias, rows, false);
        rows[n / 2] = NAN;
        snprintf(what, sizeof what, "route n=%zu with a NaN", n);
        fails += check_route(be, what, 1, n, 8, true, false, nullptr, rows, true);
    }
    return fails;
}

struct qfmt {
    int         dtype;
    size_t      elems, bytes;
    bool        trailing_d; /* Q6_K: d is the trailing f16 */
    const char *name;
};

static void fill(uint8_t *dst, const struct qfmt *q, size_t rows) {
    if (q->dtype == GEIST_DTYPE_F32) {
        float *f = (float *) dst;
        for (size_t i = 0; i < rows * N_IN; i++)
            f[i] = (float) rng_u8() / 128.0f - 1.0f;
        return;
    }
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

static int test_ids(struct geist_backend *be, const struct qfmt *q) {
    const struct geist_backend_vtbl  *v = be->desc->vtbl;
    const struct geist_backend_fused *f = be->desc->fused;
    const size_t wb   = q->dtype == GEIST_DTYPE_F32 ? N_IN * N_OUT * sizeof(float)
                                                    : N_OUT * (N_IN / q->elems) * q->bytes;
    uint8_t     *blob = malloc(N_EXP * wb + 16384);
    uint8_t *al = blob != nullptr ? blob + (16384 - ((uintptr_t) blob % 16384)) % 16384 : nullptr;
    int      fails                 = 0;
    struct geist_weight   w[N_EXP] = {0};
    struct geist_buffer  *bw = nullptr, *bx = nullptr, *by = nullptr, *b1 = nullptr, *bi = nullptr;
    static const uint32_t ids[M * K] = {7, 2, 2, 0, 9, 5};
    if (blob == nullptr)
        return geist_expect(false, "host memory");
    fill(al, q, N_EXP * N_OUT);
    for (size_t e = 0; e < N_EXP; e++) {
        w[e] = (struct geist_weight) {.raw        = al + e * wb,
                                      .raw_nbytes = wb,
                                      .n_in       = N_IN,
                                      .n_out      = N_OUT,
                                      .dtype      = (uint16_t) q->dtype};
        fails += geist_expect(v->resolve_weight(be, &w[e]) == GEIST_OK, "resolve");
    }
    const struct geist_fusion_query qq = {
            .op = GEIST_FUSED_MOE_DEVICE, .m = 1, .gate_w = &w[0], .up_w = &w[0], .down_w = &w[0]};
    fails += geist_expect(f->supported(be, &qq), "supported(GEIST_FUSED_MOE_DEVICE)");
    struct geist_fusion_query q8 = qq, q0 = qq;
    q8.m = 8;
    q0.m = 0;
    fails += geist_expect(f->supported(be, &q8) && !f->supported(be, &q0),
                          "supported(GEIST_FUSED_MOE_DEVICE) at m = 8, not at m = 0");
    /* #376: the router it names; groups that do not divide the experts or
     * an unknown gating are refused. */
    const struct geist_moe_route_args v3         = {.n          = 256,
                                                    .k          = 8,
                                                    .gating     = GEIST_MOE_ROUTE_SIGMOID,
                                                    .norm       = true,
                                                    .w_scale    = 2.5f,
                                                    .n_group    = 8,
                                                    .topk_group = 4};
    struct geist_moe_route_args       bad_groups = v3, bad_gating = v3;
    bad_groups.n_group            = 7;
    bad_gating.gating             = (enum geist_moe_route_gating) 9;
    struct geist_fusion_query qv3 = qq, qbg = qq, qbt = qq;
    qv3.moe_route = &v3;
    qbg.moe_route = &bad_groups;
    qbt.moe_route = &bad_gating;
    fails += geist_expect(f->supported(be, &qv3) == (f->moe_route == nullptr ? false : true) &&
                                  !f->supported(be, &qbg) && !f->supported(be, &qbt),
                          "supported(GEIST_FUSED_MOE_DEVICE) answers for the router");
    float xs[M * K * N_IN];
    for (size_t i = 0; i < M * K * N_IN; i++)
        xs[i] = (float) rng_u8() / 64.0f - 2.0f;
    fails += geist_expect(
            v->buffer_create_aliased(be, al, N_EXP * wb, GEIST_BUFFER_WEIGHT, &bw) == GEIST_OK &&
                    v->buffer_create(be, sizeof xs, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bx) ==
                            GEIST_OK &&
                    v->buffer_create(
                            be, M * K * N_OUT * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &by) ==
                            GEIST_OK &&
                    v->buffer_create(
                            be, M * K * N_OUT * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &b1) ==
                            GEIST_OK &&
                    v->buffer_create(be, sizeof ids, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) ==
                            GEIST_OK,
            "buffers");
    if (fails > 0)
        goto out;
    fails += geist_expect(v->buffer_upload(bx, sizeof xs, (const uint8_t *) xs) == GEIST_OK &&
                                  v->buffer_upload(bi, sizeof ids, (const uint8_t *) ids) ==
                                          GEIST_OK,
                          "upload");
    const struct geist_tensor ex = {.buffer = bw,
                                    .offset = 0,
                                    .dtype  = (enum geist_dtype) q->dtype,
                                    .layout = q->dtype == GEIST_DTYPE_F32
                                                      ? GEIST_LAYOUT_DENSE
                                                      : GEIST_LAYOUT_BLOCK_QUANTIZED,
                                    .ndim   = 2,
                                    .shape  = {N_EXP * N_OUT, N_IN},
                                    .stride = {N_IN, 1}};
    for (int per_slot = 0; per_slot < 2 && fails == 0; per_slot++) {
        struct geist_tensor tx = mat(bx, 0, per_slot ? M * K : M, N_IN);
        struct geist_tensor ty = mat(by, 0, M * K, N_OUT);
        fails += geist_expect(
                f->linear_t_ids(be, N_EXP, M, K, per_slot != 0, &w[0], &ex, bi, &tx, &ty) ==
                        GEIST_OK,
                "linear_t_ids");
        for (size_t s = 0; s < M * K; s++) {
            const size_t        r  = per_slot ? s : s / K;
            struct geist_tensor x1 = mat(bx, r * N_IN * 4, 1, N_IN);
            struct geist_tensor o1 = mat(b1, s * N_OUT * 4, 1, N_OUT);
            struct geist_tensor w1 = ex;
            w1.offset              = ids[s] * wb;
            w1.shape[0]            = N_OUT;
            fails += geist_expect(f->linear_t(be, &x1, &w[ids[s]], &w1, 1, &o1) == GEIST_OK,
                                  "single linear_t");
        }
        float yg[M * K * N_OUT], y1[M * K * N_OUT];
        fails += geist_expect(v->buffer_download(sizeof yg, (uint8_t *) yg, by) == GEIST_OK &&
                                      v->buffer_download(sizeof y1, (uint8_t *) y1, b1) == GEIST_OK,
                              "download");
        double err = 0.0, scale = 0.0;
        for (size_t i = 0; i < M * K * N_OUT; i++) {
            err   = fmax(err, fabs((double) yg[i] - (double) y1[i]));
            scale = fmax(scale, fabs((double) y1[i]));
        }
        printf("  %-4s linear_t_ids (x per %s): max |d| %.3g of scale %.3g\n",
               q->name,
               per_slot ? "slot" : "token",
               err,
               scale);
        fails += geist_expect(scale > 0.0 && err <= 1e-5 * scale, "linear_t_ids matches linear_t");
        if (q->dtype != GEIST_DTYPE_MXFP4)
            continue;
        /* MXFP4: the exact weights (the reference decoder) times x, in double. */
        static float wrow[N_IN];
        double       errd = 0.0, scd = 0.0;
        for (size_t s2 = 0; s2 < M * K; s2++) {
            const size_t r = per_slot ? s2 : s2 / K;
            for (size_t o = 0; o < N_OUT; o++) {
                dequant_mxfp4_row(N_IN, al + ids[s2] * wb + o * (N_IN / 32) * 17, wrow);
                double ref = 0.0;
                for (size_t i = 0; i < N_IN; i++)
                    ref += (double) wrow[i] * (double) xs[r * N_IN + i];
                errd = fmax(errd, fabs(ref - (double) yg[s2 * N_OUT + o]));
                scd  = fmax(scd, fabs(ref));
            }
        }
        printf("  MXFP4 linear_t_ids vs double: max |d| %.3g of scale %.3g\n", errd, scd);
        fails += geist_expect(scd > 0.0 && errd <= 1e-5 * scd,
                              "MXFP4 linear_t_ids matches the double reference");
    }
out:
    struct geist_buffer *bs[] = {bw, bx, by, b1, bi};
    for (size_t i = 0; i < 5; i++)
        if (bs[i] != nullptr)
            v->buffer_destroy(be, bs[i]);
    free(blob);
    return fails;
}

static int test_sum(struct geist_backend *be, bool with_bias) {
    const struct geist_backend_vtbl  *v = be->desc->vtbl;
    const struct geist_backend_fused *f = be->desc->fused;
    enum { D = 72 };
    static const uint32_t ids[M * K] = {7, 2, 2, 0, 9, 5};
    float                 y[M * K * D], w[M * K], add[M * D], out[M * D], bias[N_EXP * D];
    struct geist_buffer  *by = nullptr, *bw = nullptr, *ba = nullptr, *bb = nullptr, *bi = nullptr;
    for (size_t i = 0; i < N_EXP * D; i++)
        bias[i] = (float) rng_u8() / 64.0f - 2.0f;
    for (size_t i = 0; i < M * K * D; i++)
        y[i] = (float) rng_u8() / 32.0f - 4.0f;
    for (size_t i = 0; i < M * K; i++)
        w[i] = (float) rng_u8() / 256.0f;
    for (size_t i = 0; i < M * D; i++)
        add[i] = (float) rng_u8() / 16.0f - 8.0f;
    int fails = geist_expect(
            v->buffer_create(be, sizeof y, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &by) == GEIST_OK &&
                    v->buffer_create(be, sizeof w, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bw) ==
                            GEIST_OK &&
                    v->buffer_create(be, sizeof add, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &ba) ==
                            GEIST_OK &&
                    v->buffer_upload(by, sizeof y, (const uint8_t *) y) == GEIST_OK &&
                    v->buffer_upload(bw, sizeof w, (const uint8_t *) w) == GEIST_OK &&
                    v->buffer_upload(ba, sizeof add, (const uint8_t *) add) == GEIST_OK &&
                    v->buffer_create(be, sizeof bias, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bb) ==
                            GEIST_OK &&
                    v->buffer_create(be, sizeof ids, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) ==
                            GEIST_OK &&
                    v->buffer_upload(bb, sizeof bias, (const uint8_t *) bias) == GEIST_OK &&
                    v->buffer_upload(bi, sizeof ids, (const uint8_t *) ids) == GEIST_OK,
            "sum buffers");
    if (fails == 0) {
        struct geist_tensor ty = mat(by, 0, M * K, D), tw = vec(bw, 0, M * K),
                            ta = mat(ba, 0, M, D), tb = mat(bb, 0, N_EXP, D);
        fails += geist_expect(f->moe_weighted_sum(be,
                                                  M,
                                                  K,
                                                  D,
                                                  &tw,
                                                  &ty,
                                                  &ta,
                                                  with_bias ? &tb : nullptr,
                                                  with_bias ? bi : nullptr,
                                                  &ta) == GEIST_OK,
                              "moe_weighted_sum (out = add)");
        fails += geist_expect(v->buffer_download(sizeof out, (uint8_t *) out, ba) == GEIST_OK,
                              "download");
        double err = 0.0, scale = 0.0;
        for (size_t t = 0; t < M; t++)
            for (size_t c = 0; c < D; c++) {
                double ref = add[t * D + c];
                for (size_t j = 0; j < K; j++)
                    ref += (double) w[t * K + j] *
                           ((double) y[(t * K + j) * D + c] +
                            (with_bias ? (double) bias[ids[t * K + j] * D + c] : 0.0));
                err   = fmax(err, fabs(ref - (double) out[t * D + c]));
                scale = fmax(scale, fabs(ref));
            }
        printf("  moe_weighted_sum%s: max |d| %.3g of scale %.3g\n",
               with_bias ? " with biases" : "",
               err,
               scale);
        fails += geist_expect(err <= 1e-5 * scale, "moe_weighted_sum matches the host");
    }
    struct geist_buffer *bs[] = {by, bw, ba, bb, bi};
    for (size_t i = 0; i < 5; i++)
        if (bs[i] != nullptr)
            v->buffer_destroy(be, bs[i]);
    return fails;
}

/* fused->glu against the formula in double: act 0 silu, 1 swiglu_oai;
 * bias 0 none, 1 by the routed ids, 2 one expert's row (e = 3) for all. */
static int test_glu(struct geist_backend *be, int act, int bias) {
    const struct geist_backend_vtbl  *v = be->desc->vtbl;
    const struct geist_backend_fused *f = be->desc->fused;
    enum { R = M * K, C = 200 };
    static const uint32_t ids[R] = {7, 2, 2, 0, 9, 5};
    static float          g[R * C], u[R * C], gb[N_EXP * C], ub[N_EXP * C], y[R * C];
    struct geist_buffer *bg = nullptr, *bu = nullptr, *bgb = nullptr, *bub = nullptr, *bi = nullptr;
    for (size_t i = 0; i < R * C; i++) {
        g[i] = (float) rng_u8() / 12.0f - 10.0f; /* past the limit both ways */
        u[i] = (float) rng_u8() / 12.0f - 10.0f;
    }
    for (size_t i = 0; i < N_EXP * C; i++) {
        gb[i] = (float) rng_u8() / 64.0f - 2.0f;
        ub[i] = (float) rng_u8() / 64.0f - 2.0f;
    }
    int fails = geist_expect(
            v->buffer_create(be, sizeof g, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bg) == GEIST_OK &&
                    v->buffer_create(be, sizeof u, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bu) ==
                            GEIST_OK &&
                    v->buffer_create(be, sizeof gb, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bgb) ==
                            GEIST_OK &&
                    v->buffer_create(be, sizeof ub, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bub) ==
                            GEIST_OK &&
                    v->buffer_create(be, sizeof ids, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) ==
                            GEIST_OK &&
                    v->buffer_upload(bg, sizeof g, (const uint8_t *) g) == GEIST_OK &&
                    v->buffer_upload(bu, sizeof u, (const uint8_t *) u) == GEIST_OK &&
                    v->buffer_upload(bgb, sizeof gb, (const uint8_t *) gb) == GEIST_OK &&
                    v->buffer_upload(bub, sizeof ub, (const uint8_t *) ub) == GEIST_OK &&
                    v->buffer_upload(bi, sizeof ids, (const uint8_t *) ids) == GEIST_OK,
            "glu buffers");
    if (fails == 0) {
        enum { E1 = 3 };
        struct geist_tensor tg = mat(bg, 0, R, C), tu = mat(bu, 0, R, C);
        struct geist_tensor tgb = bias == 2 ? mat(bgb, E1 * C * 4, 1, C) : mat(bgb, 0, N_EXP, C);
        struct geist_tensor tub = bias == 2 ? mat(bub, E1 * C * 4, 1, C) : mat(bub, 0, N_EXP, C);
        const struct geist_glu_args a = {.act    = act ? GEIST_GLU_SWIGLU_OAI : GEIST_GLU_SILU,
                                         .alpha  = 1.702f,
                                         .limit  = 7.0f,
                                         .x_bias = bias ? &tgb : nullptr,
                                         .z_bias = bias ? &tub : nullptr,
                                         .ids    = bias == 1 ? bi : nullptr};
        fails += geist_expect(f->glu(be, &a, &tg, &tu, &tg) == GEIST_OK, "glu (y = x)");
        fails += geist_expect(v->buffer_download(sizeof y, (uint8_t *) y, bg) == GEIST_OK,
                              "glu download");
        double err = 0.0, scale = 0.0;
        for (size_t r = 0; r < R; r++) {
            const size_t e = bias == 1 ? ids[r] : E1;
            for (size_t c = 0; c < C; c++) {
                double x = g[r * C + c], z = u[r * C + c], ref;
                if (bias) {
                    x += gb[e * C + c];
                    z += ub[e * C + c];
                }
                if (act) {
                    x   = fmin(x, 7.0);
                    z   = fmax(-7.0, fmin(z, 7.0));
                    ref = x / (1.0 + exp(-1.702 * x)) * (z + 1.0);
                } else {
                    ref = x / (1.0 + exp(-x)) * z;
                }
                err   = fmax(err, fabs(ref - (double) y[r * C + c]));
                scale = fmax(scale, fabs(ref));
            }
        }
        printf("  glu %s, %s: max |d| %.3g of scale %.3g\n",
               act ? "swiglu_oai" : "silu",
               bias == 0   ? "no bias"
               : bias == 1 ? "bias rows by id"
                           : "one expert's bias",
               err,
               scale);
        fails += geist_expect(scale > 0.0 && err <= 1e-5 * scale, "glu matches the double formula");
    }
    struct geist_buffer *bs[] = {bg, bu, bgb, bub, bi};
    for (size_t i = 0; i < 5; i++)
        if (bs[i] != nullptr)
            v->buffer_destroy(be, bs[i]);
    return fails;
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
            f->moe_weighted_sum == nullptr) {
            geist_backend_destroy(be);
            continue;
        }
        printf("%s:\n", names[b]);
        ran++;
        fails += test_route(be);
        fails += test_route_deepseek(be);
        static const struct qfmt qs[] = {
                {GEIST_DTYPE_F32, 1, 4, false, "F32"},
                {GEIST_DTYPE_Q4_K, 256, 144, false, "Q4_K"},
                {GEIST_DTYPE_Q6_K, 256, 210, true, "Q6_K"},
                {GEIST_DTYPE_Q8_0, 32, 34, false, "Q8_0"},
                {GEIST_DTYPE_MXFP4, 32, 17, false, "MXFP4"},
                {GEIST_DTYPE_Q5_0, 32, 22, false, "Q5_0"},
        };
        for (size_t i = 0; i < sizeof qs / sizeof qs[0]; i++)
            fails += test_ids(be, &qs[i]);
        fails += test_sum(be, false);
        fails += test_sum(be, true);
        for (int act = 0; act < 2 && f->glu != nullptr; act++)
            for (int bias = 0; bias < 3; bias++)
                fails += test_glu(be, act, bias);
        geist_backend_destroy(be);
    }
    if (ran == 0) {
        printf("SKIP: no backend in this build routes a mixture of experts on the device\n");
        return 0;
    }
    if (fails == 0)
        printf("PASS: device routing, id-indexed expert matvecs, glu, weighted sum\n");
    return fails == 0 ? 0 : 1;
}
