/*
 * test_backend_vulkan_linear_grouped_unit — fused->linear_t_grouped, the
 * decode step of several mixture-of-experts experts in one pass (#78 phase
 * 3), against its decomposed twin: one fused->linear_t per weight.
 *
 * For F32, Q4_K, Q5_K and Q6_K: 10 weights of one shape (two dispatches: the
 * kernels take VK_GROUP_MAX = 8 per dispatch), random but finite, resolved
 * on Vulkan; x [10, n_in] in VRAM. The grouped call and the 10 single calls
 * write [10, n_out] each; the grouped kernels are the single ones with the
 * weight read by device address, so the outputs must agree to rounding
 * (relative 1e-5 of the output scale). Also: supported() answers yes for
 * the resolved weights at m = 1 and no at m = 2, and a weight the registry
 * does not know makes the call GEIST_E_UNSUPPORTED before anything runs.
 *
 * rows_weighted_sum (#161, #163): per token, out[t] = add[t] + its rows of
 * y weighted, for two tokens of four shuffled rows each, the host's add
 * passed after a submit() of the grouped call that wrote y (nothing waited
 * for), against the host sum to 1e-5 of its scale. rows_gather: six rows
 * of x by index (one twice), bit for bit.
 *
 * linear_t_sel (#162): moe_route's top-3 of 10 logits (one tie) on the
 * device, submit_mark, the slots by those ids with two experts cold; the hot
 * slots against linear_t, the cold one untouched, map_marked's ids exact.
 *
 * SKIPs when the Vulkan backend is not built, has no device, or the device
 * has no bufferDeviceAddress (the op is then absent by design).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N = 10, N_IN = 512, N_OUT = 136 };

struct qfmt {
    int    dtype;
    size_t elems, bytes;
    bool   trailing_d; /* Q6_K: d is the trailing f16 */
};

static uint32_t rng = 0x2468ace1u;
static uint8_t  rng_u8(void) {
    rng = rng * 1664525u + 1013904223u;
    return (uint8_t) (rng >> 24);
}

/* Random-but-finite weight bytes: random quants, f16 scale fields pinned. */
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
        if (!q->trailing_d) {
            blk[2] = 0x00; /* dmin = fp16(0.5) */
            blk[3] = 0x38;
        }
    }
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

static int run_case(struct geist_backend *be, const struct qfmt *q, const char *name) {
    const struct geist_backend_vtbl  *v = be->desc->vtbl;
    const struct geist_backend_fused *f = be->desc->fused;
    const size_t        wb    = q->dtype == GEIST_DTYPE_F32 ? N_IN * N_OUT * sizeof(float)
                                                            : N_OUT * (N_IN / q->elems) * q->bytes;
    uint8_t            *blob  = malloc(N * wb);
    float              *xs    = malloc(N * N_IN * sizeof(float));
    float              *yg    = malloc(N * N_OUT * sizeof(float));
    float              *y1    = malloc(N * N_OUT * sizeof(float));
    int                 fails = 0;
    struct geist_weight w[N]  = {0};
    const struct geist_weight *wp[N];
    struct geist_buffer       *bx = nullptr, *by = nullptr, *b1 = nullptr;
    if (blob == nullptr || xs == nullptr || yg == nullptr || y1 == nullptr)
        return geist_expect(false, "host memory");
    fill(blob, q, N * N_OUT);
    for (size_t i = 0; i < N * N_IN; i++)
        xs[i] = (float) rng_u8() / 64.0f - 2.0f;
    for (size_t j = 0; j < N; j++) {
        w[j]  = (struct geist_weight) {.raw        = blob + j * wb,
                                       .raw_nbytes = wb,
                                       .n_in       = N_IN,
                                       .n_out      = N_OUT,
                                       .dtype      = (uint16_t) q->dtype};
        wp[j] = &w[j];
        fails += geist_expect(v->resolve_weight(be, &w[j]) == GEIST_OK, "resolve");
    }
    const struct geist_fusion_query q1 = {
            .op = GEIST_FUSED_LINEAR_GROUPED, .m = 1, .gate_w = &w[0], .up_w = &w[N - 1]};
    struct geist_fusion_query q2 = q1;
    q2.m                         = 2;
    fails += geist_expect(f->supported(be, &q1), "supported at m = 1");
    fails += geist_expect(!f->supported(be, &q2), "not supported at m = 2");

    fails += geist_expect(
            v->buffer_create(be, N * N_IN * 4, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &bx) ==
                            GEIST_OK &&
                    v->buffer_create(be, N * N_OUT * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &by) ==
                            GEIST_OK &&
                    v->buffer_create(be, N * N_OUT * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &b1) ==
                            GEIST_OK,
            "buffers");
    if (fails > 0)
        goto out;
    fails += geist_expect(v->buffer_upload(bx, N * N_IN * 4, (const uint8_t *) xs) == GEIST_OK,
                          "upload x");
    struct geist_tensor tx = mat(bx, 0, N, N_IN), ty = mat(by, 0, N, N_OUT);
    fails += geist_expect(f->linear_t_grouped(be, N, wp, &tx, &ty) == GEIST_OK, "grouped call");
    for (size_t j = 0; j < N; j++) {
        struct geist_tensor x1 = mat(bx, j * N_IN * 4, 1, N_IN);
        struct geist_tensor o1 = mat(b1, j * N_OUT * 4, 1, N_OUT);
        fails += geist_expect(f->linear_t(be, &x1, &w[j], nullptr, 1, &o1) == GEIST_OK,
                              "single linear_t");
    }
    fails += geist_expect(v->buffer_download(N * N_OUT * 4, (uint8_t *) yg, by) == GEIST_OK &&
                                  v->buffer_download(N * N_OUT * 4, (uint8_t *) y1, b1) == GEIST_OK,
                          "download");
    double err = 0.0, scale = 0.0;
    for (size_t i = 0; i < N * N_OUT; i++) {
        err   = fmax(err, fabs((double) yg[i] - (double) y1[i]));
        scale = fmax(scale, fabs((double) y1[i]));
    }
    printf("  %-4s: %d weights grouped vs single, max |d| %.3g of scale %.3g\n",
           name,
           N,
           err,
           scale);
    fails += geist_expect(scale > 0.0 && err <= 1e-5 * scale, "grouped matches single");

    /* #162: moe_route's top-3 of 10 logits (7, 4, 1) on the device, a mark,
     * then linear_t_sel by those ids with experts 4 and 8 cold (no weight):
     * the hot slots match linear_t of their expert, the cold slot keeps its
     * sentinel; map_marked hands back the ids. x: one row for every slot,
     * then a row per slot. */
    if (f->linear_t_sel != nullptr && f->moe_route != nullptr && f->submit_mark != nullptr &&
        f->map_marked != nullptr && fails == 0) {
        float lg[N]                   = {0};
        lg[7]                         = 3.0f;
        lg[4]                         = 2.5f;
        lg[1]                         = 2.5f; /* a tie: the lower id ranks first */
        lg[3]                         = -1.0f;
        static const uint32_t want[3] = {7, 1, 4};
        /* One array per case: the backend may key what it derives from the
         * array by its address (linear_t_sel's contract). */
        static const struct geist_weight *ws_case[8][N];
        static size_t                     n_case = 0;
        const struct geist_weight       **ws     = ws_case[n_case++ % 8];
        for (size_t e = 0; e < N; e++)
            ws[e] = e == 4 || e == 8 ? nullptr : &w[e];
        struct geist_buffer *bl = nullptr, *bi = nullptr, *bw = nullptr, *bs = nullptr;
        fails += geist_expect(
                v->buffer_create(be, sizeof lg, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bl) ==
                                GEIST_OK &&
                        v->buffer_create(be, 4 * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bi) ==
                                GEIST_OK &&
                        v->buffer_create(be, 3 * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bw) ==
                                GEIST_OK &&
                        v->buffer_create(
                                be, 3 * N_OUT * 4, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bs) ==
                                GEIST_OK,
                "sel buffers");
        for (int per_slot = 0; per_slot < 2 && fails == 0; per_slot++) {
            const uint32_t zero[4] = {0};
            float          sent[3 * N_OUT], got[3 * N_OUT], ref[N_OUT];
            for (size_t i = 0; i < 3 * N_OUT; i++)
                sent[i] = 1234.0f;
            fails += geist_expect(
                    v->buffer_upload(bl, sizeof lg, (const uint8_t *) lg) == GEIST_OK &&
                            v->buffer_upload(bi, sizeof zero, (const uint8_t *) zero) == GEIST_OK &&
                            v->buffer_upload(bs, sizeof sent, (const uint8_t *) sent) == GEIST_OK,
                    "sel upload");
            struct geist_tensor tl = mat(bl, 0, 1, N), tw = mat(bw, 0, 1, 3);
            struct geist_tensor tx1 = mat(bx, 0, per_slot ? 3 : 1, N_IN), ts = mat(bs, 0, 3, N_OUT);
            fails += geist_expect(f->moe_route(be, 1, N, 3, true, &tl, bi, 0, 12, &tw) == GEIST_OK,
                                  "moe_route");
            const uint64_t mark = f->submit_mark(be);
            fails += geist_expect(f->linear_t_sel(be, N, ws, 3, per_slot != 0, bi, 0, &tx1, &ts) ==
                                          GEIST_OK,
                                  "linear_t_sel");
            f->submit(be);
            const uint32_t *ids = f->map_marked(be, mark, bi);
            fails += geist_expect(ids != nullptr && memcmp(ids, want, sizeof want) == 0 &&
                                          ids[3] == 0,
                                  "map_marked: the routed ids, no flag");
            fails += geist_expect(v->buffer_download(sizeof got, (uint8_t *) got, bs) == GEIST_OK,
                                  "sel download");
            double err = 0.0, scale = 0.0;
            for (size_t sl = 0; sl < 3 && fails == 0; sl++) {
                if (ws[want[sl]] == nullptr) {
                    for (size_t c = 0; c < N_OUT; c++)
                        err = fmax(err, got[sl * N_OUT + c] == 1234.0f ? 0.0 : INFINITY);
                    continue;
                }
                struct geist_tensor x1 = mat(bx, (per_slot ? sl : 0) * N_IN * 4, 1, N_IN);
                struct geist_tensor o1 = mat(b1, 0, 1, N_OUT);
                fails += geist_expect(
                        f->linear_t(be, &x1, &w[want[sl]], nullptr, 1, &o1) == GEIST_OK &&
                                v->buffer_download(sizeof ref, (uint8_t *) ref, b1) == GEIST_OK,
                        "sel reference");
                for (size_t c = 0; c < N_OUT; c++) {
                    err   = fmax(err, fabs((double) got[sl * N_OUT + c] - (double) ref[c]));
                    scale = fmax(scale, fabs((double) ref[c]));
                }
            }
            printf("  %-4s: linear_t_sel (x per %s), max |d| %.3g of scale %.3g\n",
                   name,
                   per_slot ? "slot" : "token",
                   err,
                   scale);
            fails += geist_expect(scale > 0.0 && err <= 1e-5 * scale,
                                  "linear_t_sel matches linear_t, the cold slot untouched");
        }
        struct geist_buffer *bb[] = {bl, bi, bw, bs};
        for (size_t i = 0; i < 4; i++)
            if (bb[i] != nullptr)
                v->buffer_destroy(be, bb[i]);
    }

    /* rows_weighted_sum onto two tokens from y's first 8 rows, in a shuffled
     * order (token 0: rows 5 2 7 0, token 1: 1 6 3 4), started but not
     * waited for. */
    if (f->submit != nullptr && f->rows_weighted_sum != nullptr && fails == 0) {
        static const uint32_t start[3] = {0, 4, 8}, rows[8] = {5, 2, 7, 0, 1, 6, 3, 4};
        float                 wts[8], add[2 * N_OUT], out[2 * N_OUT];
        for (size_t j = 0; j < 8; j++)
            wts[j] = 0.25f * (float) (j + 1) - 1.0f;
        for (size_t c = 0; c < 2 * N_OUT; c++)
            add[c] = (float) rng_u8() / 32.0f - 4.0f;
        fails +=
                geist_expect(f->linear_t_grouped(be, N, wp, &tx, &ty) == GEIST_OK, "grouped again");
        f->submit(be);
        struct geist_tensor y8 = mat(by, 0, 8, N_OUT), o2 = mat(b1, 0, 2, N_OUT);
        fails += geist_expect(
                f->rows_weighted_sum(be, 2, N_OUT, 8, start, rows, wts, add, &y8, &o2) == GEIST_OK,
                "rows_weighted_sum");
        fails += geist_expect(v->buffer_download(sizeof out, (uint8_t *) out, b1) == GEIST_OK,
                              "download the sum");
        double e3 = 0.0, sc3 = 0.0;
        for (size_t t = 0; t < 2; t++)
            for (size_t c = 0; c < N_OUT; c++) {
                double ref = add[t * N_OUT + c];
                for (size_t q = start[t]; q < start[t + 1]; q++)
                    ref += (double) wts[q] * (double) yg[rows[q] * N_OUT + c];
                e3  = fmax(e3, fabs(ref - (double) out[t * N_OUT + c]));
                sc3 = fmax(sc3, fabs(ref));
            }
        printf("  %-4s: rows_weighted_sum (2 tokens) after submit, max |d| %.3g of scale %.3g\n",
               name,
               e3,
               sc3);
        fails += geist_expect(e3 <= 1e-5 * sc3, "rows_weighted_sum matches the host");
    }

    /* rows_gather: x's rows 3 1 0 7 7 2 (one twice), bit for bit. */
    if (f->rows_gather != nullptr && fails == 0) {
        static const uint32_t src[6] = {3, 1, 0, 7, 7, 2};
        float                 got[6 * N_IN];
        struct geist_buffer  *bg = nullptr;
        fails += geist_expect(
                v->buffer_create(be, sizeof got, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bg) ==
                        GEIST_OK,
                "gather buffer");
        struct geist_tensor xg = mat(bx, 0, N, N_IN), og = mat(bg, 0, 6, N_IN);
        fails +=
                geist_expect(f->rows_gather(be, 6, N_IN, src, &xg, &og) == GEIST_OK, "rows_gather");
        fails += geist_expect(v->buffer_download(sizeof got, (uint8_t *) got, bg) == GEIST_OK,
                              "download the gathered rows");
        size_t same = 0;
        for (size_t i = 0; i < 6; i++)
            same += memcmp(got + i * N_IN, xs + src[i] * N_IN, N_IN * sizeof(float)) == 0;
        printf("  %-4s: rows_gather, %zu of 6 rows bit for bit\n", name, same);
        fails += geist_expect(same == 6, "rows_gather copies the rows exactly");
        v->buffer_destroy(be, bg);
    }

    /* A weight the registry does not know: refused before anything runs. */
    struct geist_weight stranger = w[0];
    stranger.raw                 = blob + wb / 2;
    wp[3]                        = &stranger;
    fails += geist_expect(f->linear_t_grouped(be, N, wp, &tx, &ty) == GEIST_E_UNSUPPORTED,
                          "an unresolved weight is GEIST_E_UNSUPPORTED");
out:
    v->buffer_destroy(be, bx);
    v->buffer_destroy(be, by);
    v->buffer_destroy(be, b1);
    free(blob);
    free(xs);
    free(yg);
    free(y1);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "SKIP: vulkan backend unavailable\n");
        return GEIST_TEST_SKIP;
    }
    const struct geist_backend_fused *f               = be->desc->fused;
    float                             probe[N_IN * 2] = {0};
    struct geist_weight               pw              = {.raw        = probe,
                                                         .raw_nbytes = sizeof probe,
                                                         .n_in       = N_IN,
                                                         .n_out      = 2,
                                                         .dtype      = GEIST_DTYPE_F32};
    const struct geist_fusion_query pq = {.op = GEIST_FUSED_LINEAR_GROUPED, .m = 1, .gate_w = &pw};
    if (f == nullptr || f->linear_t_grouped == nullptr ||
        be->desc->vtbl->resolve_weight(be, &pw) != GEIST_OK || !f->supported(be, &pq)) {
        fprintf(stderr, "SKIP: no grouped linear on this device (bufferDeviceAddress)\n");
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    const struct qfmt fm[]    = {{GEIST_DTYPE_F32, 1, 4, false},
                                 {GEIST_DTYPE_Q4_K, 256, 144, false},
                                 {GEIST_DTYPE_Q5_K, 256, 176, false},
                                 {GEIST_DTYPE_Q6_K, 256, 210, true}};
    const char *const names[] = {"F32", "Q4_K", "Q5_K", "Q6_K"};
    int               fails   = 0;
    for (size_t i = 0; i < 4; i++)
        fails += run_case(be, &fm[i], names[i]);
    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("vulkan linear_t_grouped: pass\n");
    return GEIST_TEST_PASS;
}
