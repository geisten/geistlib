/*
 * test_moe_device_unit - mixture-of-experts decode on the device (#201):
 * fused->moe_route, fused->linear_t_ids and fused->moe_weighted_sum on the
 * backends that have them (Metal), against host references.
 *
 * moe_route against moe_route_topk (the host's router, the reference both
 * engines' routing is measured by) on #48's edge cases: random logits, ties
 * in the scores (ranked by id), k = n, an all-zero row, renormalised and
 * not; ids exact, weights within 1e-6. A non-finite logit sets the flag word
 * after the ids and still writes ids below n.
 *
 * linear_t_ids, for F32, Q4_K, Q6_K and Q8_0: 10 experts back to back in one
 * buffer, the ids of two tokens' three slots (one expert twice, the last
 * expert), x one row per token (gate/up) and one row per slot (down);
 * against fused->linear_t of each slot's expert alone, to 1e-5 of the
 * output scale (the same matvec bodies). supported(GEIST_FUSED_MOE_DEVICE)
 * says yes for these dtypes at m = 1.
 *
 * moe_weighted_sum: two tokens of three slots onto their rows of `add`,
 * against the host sum to 1e-5 of its scale.
 *
 * SKIPs when no backend in the build has the ops.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../src/archs/transformer/forward/blocks.h"
#include "test_helpers.h"

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

/* Routes `rows` rows of `n` logits with k on the device and checks each
 * against moe_route_topk. */
static int check_route(struct geist_backend *be,
                       const char           *what,
                       size_t                rows,
                       size_t                n,
                       size_t                k,
                       bool                  norm,
                       const float          *logits,
                       bool                  expect_bad) {
    const struct geist_backend_vtbl  *v  = be->desc->vtbl;
    const struct geist_backend_fused *f  = be->desc->fused;
    struct geist_buffer              *bl = nullptr, *bi = nullptr, *bw = nullptr;
    int                               fails = 0;
    const size_t                      nk    = rows * k;
    if (v->buffer_create(be, rows * n * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bl) != GEIST_OK ||
        v->buffer_create(be, (nk + 1) * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) != GEIST_OK ||
        v->buffer_create(be, nk * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bw) != GEIST_OK)
        return geist_expect(false, "route buffers");
    uint32_t zero[64 + 1] = {0};
    fails += geist_expect(
            v->buffer_upload(bl, rows * n * 4, (const uint8_t *) logits) == GEIST_OK &&
                    v->buffer_upload(bi, (nk + 1) * 4, (const uint8_t *) zero) == GEIST_OK,
            "route upload");
    struct geist_tensor tl = mat(bl, 0, rows, n), tw = vec(bw, 0, nk);
    fails += geist_expect(f->moe_route(be, rows, n, k, norm, &tl, bi, nk * 4, &tw) == GEIST_OK,
                          what);
    uint32_t ids[64 + 1];
    float    w[64];
    fails += geist_expect(v->buffer_download((nk + 1) * 4, (uint8_t *) ids, bi) == GEIST_OK &&
                                  v->buffer_download(nk * 4, (uint8_t *) w, bw) == GEIST_OK,
                          "route download");
    fails += geist_expect((ids[nk] != 0) == expect_bad,
                          expect_bad ? "a non-finite logit sets the flag" : "no flag");
    for (size_t r = 0; r < rows && fails == 0; r++) {
        uint32_t ref_ids[64];
        float    ref_w[64];
        if (expect_bad) {
            for (size_t j = 0; j < k; j++)
                fails += geist_expect(ids[r * k + j] < n, "ids stay below n");
            continue;
        }
        fails += geist_expect(moe_route_topk(n, k, norm, logits + r * n, ref_ids, ref_w),
                              "host reference");
        for (size_t j = 0; j < k; j++) {
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
    v->buffer_destroy(be, bl);
    v->buffer_destroy(be, bi);
    v->buffer_destroy(be, bw);
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
    fails += check_route(be, "route k=4 renormalised", 4, 16, 4, true, lg, false);
    fails += check_route(be, "route k=4 softmax", 4, 16, 4, false, lg, false);
    fails += check_route(be, "route k=n", 4, 16, 16, true, lg, false);
    fails += check_route(be, "route k=1", 4, 16, 1, false, lg, false);
    float bad[16];
    memcpy(bad, lg, sizeof bad);
    bad[5] = NAN;
    fails += check_route(be, "route with a NaN", 1, 16, 4, true, bad, true);
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
    }
out:
    struct geist_buffer *bs[] = {bw, bx, by, b1, bi};
    for (size_t i = 0; i < 5; i++)
        if (bs[i] != nullptr)
            v->buffer_destroy(be, bs[i]);
    free(blob);
    return fails;
}

static int test_sum(struct geist_backend *be) {
    const struct geist_backend_vtbl  *v = be->desc->vtbl;
    const struct geist_backend_fused *f = be->desc->fused;
    enum { D = 72 };
    float                y[M * K * D], w[M * K], add[M * D], out[M * D];
    struct geist_buffer *by = nullptr, *bw = nullptr, *ba = nullptr;
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
                    v->buffer_upload(ba, sizeof add, (const uint8_t *) add) == GEIST_OK,
            "sum buffers");
    if (fails == 0) {
        struct geist_tensor ty = mat(by, 0, M * K, D), tw = vec(bw, 0, M * K),
                            ta = mat(ba, 0, M, D);
        fails += geist_expect(f->moe_weighted_sum(be, M, K, D, &tw, &ty, &ta, &ta) == GEIST_OK,
                              "moe_weighted_sum (out = add)");
        fails += geist_expect(v->buffer_download(sizeof out, (uint8_t *) out, ba) == GEIST_OK,
                              "download");
        double err = 0.0, scale = 0.0;
        for (size_t t = 0; t < M; t++)
            for (size_t c = 0; c < D; c++) {
                double ref = add[t * D + c];
                for (size_t j = 0; j < K; j++)
                    ref += (double) w[t * K + j] * (double) y[(t * K + j) * D + c];
                err   = fmax(err, fabs(ref - (double) out[t * D + c]));
                scale = fmax(scale, fabs(ref));
            }
        printf("  moe_weighted_sum: max |d| %.3g of scale %.3g\n", err, scale);
        fails += geist_expect(err <= 1e-5 * scale, "moe_weighted_sum matches the host");
    }
    struct geist_buffer *bs[] = {by, bw, ba};
    for (size_t i = 0; i < 3; i++)
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
        static const struct qfmt qs[] = {
                {GEIST_DTYPE_F32, 1, 4, false, "F32"},
                {GEIST_DTYPE_Q4_K, 256, 144, false, "Q4_K"},
                {GEIST_DTYPE_Q6_K, 256, 210, true, "Q6_K"},
                {GEIST_DTYPE_Q8_0, 32, 34, false, "Q8_0"},
        };
        for (size_t i = 0; i < sizeof qs / sizeof qs[0]; i++)
            fails += test_ids(be, &qs[i]);
        fails += test_sum(be);
        geist_backend_destroy(be);
    }
    if (ran == 0) {
        printf("SKIP: no backend in this build routes a mixture of experts on the device\n");
        return 0;
    }
    if (fails == 0)
        printf("PASS: device routing, id-indexed expert matvecs, weighted sum\n");
    return fails == 0 ? 0 : 1;
}
