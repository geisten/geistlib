/*
 * test_backend_vulkan_stage_unit — fused->stage_reserve / stage_weights,
 * the staging slots a mixture-of-experts prefill copies cold experts into
 * (#239), against the same weights resident in VRAM.
 *
 * For F32, Q4_K, Q5_K and Q6_K (Q6_K is repacked for the GPU): four weights
 * of one shape, each resolved twice on Vulkan: resident, and as a host copy
 * (GEIST_W_HOST) of the same bytes at another address. Two of the host
 * copies are staged in set 0, two in set 1; linear_t over 24 rows of x by a
 * staged weight must equal linear_t by its resident twin bit for bit (the
 * same kernel on the same bytes). Three rounds restage set 0 in another
 * order without the host waiting in between, so each set's copies must
 * wait for the GEMMs that read its previous weights (on the transfer queue
 * of a real card; in the batch on lavapipe). Refusals, each
 * GEIST_E_UNSUPPORTED: more weights than slots, a resident weight, a weight
 * the backend does not know.
 *
 * SKIPs when the Vulkan backend is not built or has no device.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { K = 4, N_IN = 512, N_OUT = 136, M = 24 };

struct qfmt {
    int         dtype;
    size_t      elems, bytes;
    bool        trailing_d; /* Q6_K: d is the trailing f16 */
    const char *name;
};

static uint32_t rng = 0x1357acefu;
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

/* y[j] = x by weight w[j] for j in [0, K), into rows j of `out`. */
static int run_all(struct geist_backend      *be,
                   struct geist_buffer       *bx,
                   struct geist_buffer       *out,
                   const struct geist_weight *w) {
    const struct geist_backend_fused *f     = be->desc->fused;
    struct geist_tensor               tx    = mat(bx, 0, M, N_IN);
    int                               fails = 0;
    for (size_t j = 0; j < K; j++) {
        struct geist_tensor ty = mat(out, j * M * N_OUT * 4, M, N_OUT);
        fails += geist_expect(f->linear_t(be, &tx, &w[j], nullptr, M, &ty) == GEIST_OK, "linear_t");
    }
    return fails;
}

static int run_case(struct geist_backend *be, const struct qfmt *q) {
    const struct geist_backend_vtbl  *v = be->desc->vtbl;
    const struct geist_backend_fused *f = be->desc->fused;
    const size_t         wb = q->dtype == GEIST_DTYPE_F32 ? N_IN * N_OUT * sizeof(float)
                                                          : N_OUT * (N_IN / q->elems) * q->bytes;
    const size_t         yb = K * M * N_OUT * sizeof(float);
    uint8_t             *res_blob = malloc(K * wb), *host_blob = malloc(K * wb);
    float               *xs = malloc(M * N_IN * sizeof(float)), *yr = malloc(yb), *ys = malloc(yb);
    struct geist_weight  res[K] = {0}, host[K] = {0}, stray = {0};
    struct geist_buffer *bx = nullptr, *br = nullptr, *bs = nullptr;
    int                  fails = 0;
    if (res_blob == nullptr || host_blob == nullptr || xs == nullptr || yr == nullptr ||
        ys == nullptr) {
        fails += geist_expect(false, "host memory");
        goto out;
    }
    fill(res_blob, q, K * N_OUT);
    memcpy(host_blob, res_blob, K * wb);
    for (size_t i = 0; i < M * N_IN; i++)
        xs[i] = (float) rng_u8() / 64.0f - 2.0f;
    for (size_t j = 0; j < K; j++) {
        res[j]        = (struct geist_weight) {.raw        = res_blob + j * wb,
                                               .raw_nbytes = wb,
                                               .n_in       = N_IN,
                                               .n_out      = N_OUT,
                                               .dtype      = (uint16_t) q->dtype};
        host[j]       = res[j];
        host[j].raw   = host_blob + j * wb;
        host[j].flags = GEIST_W_HOST;
        fails += geist_expect(v->resolve_weight(be, &res[j]) == GEIST_OK &&
                                      v->resolve_weight(be, &host[j]) == GEIST_OK,
                              "resolve both copies");
    }
    const struct geist_weight *like[K] = {&host[0], &host[1], &host[2], &host[3]};
    fails += geist_expect(f->stage_reserve(be, K, K, like) == GEIST_OK, "a slot per weight");
    fails += geist_expect(
            v->buffer_create(be, M * N_IN * 4, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &bx) ==
                            GEIST_OK &&
                    v->buffer_create(be, yb, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &br) == GEIST_OK &&
                    v->buffer_create(be, yb, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &bs) == GEIST_OK,
            "buffers");
    if (fails > 0)
        goto out;
    fails += geist_expect(v->buffer_upload(bx, M * N_IN * 4, (const uint8_t *) xs) == GEIST_OK,
                          "upload x");
    fails += run_all(be, bx, br, res);
    fails += geist_expect(v->buffer_download(yb, (uint8_t *) yr, br) == GEIST_OK, "download");

    /* Each round stages the four in another slot order while the GEMMs of
     * the round before may still read the slots. */
    for (size_t round = 0; round < 3 && fails == 0; round++) {
        const struct geist_weight *st[K];
        for (size_t j = 0; j < K; j++)
            st[j] = &host[(j + round) % K];
        fails += geist_expect(f->stage_weights(be, K, st) == GEIST_OK, "stage");
        fails += run_all(be, bx, bs, host);
        f->submit(be);
    }
    fails += geist_expect(v->buffer_download(yb, (uint8_t *) ys, bs) == GEIST_OK, "download");
    if (fails == 0) {
        size_t diff = 0;
        for (size_t i = 0; i < K * M * N_OUT; i++)
            diff += memcmp(&yr[i], &ys[i], sizeof yr[i]) != 0;
        printf("%s: staged vs resident, %zu of %zu outputs differ\n",
               q->name,
               diff,
               (size_t) (K * M * N_OUT));
        fails += geist_expect(diff == 0, "a staged weight computes what its resident twin does");
    }

    /* Refusals. */
    const struct geist_weight *five[5]  = {&host[0], &host[1], &host[2], &host[3], &host[0]};
    const struct geist_weight *twice[2] = {&host[1], &host[1]};
    const struct geist_weight *resd[1]  = {&res[0]};
    stray                               = host[0];
    stray.raw                           = xs; /* never resolved */
    const struct geist_weight *unk[1]   = {&stray};
    fails += geist_expect(f->stage_weights(be, 5, five) == GEIST_E_UNSUPPORTED,
                          "more weights than slots: refused");
    fails += geist_expect(f->stage_weights(be, 2, twice) == GEIST_E_UNSUPPORTED,
                          "a weight twice: refused");
    fails += geist_expect(f->stage_weights(be, 1, resd) == GEIST_E_UNSUPPORTED,
                          "a resident weight: refused");
    fails += geist_expect(f->stage_weights(be, 1, unk) == GEIST_E_UNSUPPORTED,
                          "a weight the backend does not know: refused");
out:
    if (bx != nullptr)
        v->buffer_destroy(be, bx);
    if (br != nullptr)
        v->buffer_destroy(be, br);
    if (bs != nullptr)
        v->buffer_destroy(be, bs);
    free(res_blob);
    free(host_blob);
    free(xs);
    free(yr);
    free(ys);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &be) != GEIST_OK) {
        printf("SKIP: no Vulkan backend or device\n");
        return GEIST_TEST_SKIP;
    }
    if (be->desc->fused == nullptr || be->desc->fused->stage_weights == nullptr) {
        geist_backend_destroy(be);
        printf("FAIL: the Vulkan backend has no staging\n");
        return 1;
    }
    const struct qfmt fmts[] = {
            {GEIST_DTYPE_F32, 1, 4, false, "F32"},
            {GEIST_DTYPE_Q4_K, 256, 144, false, "Q4_K"},
            {GEIST_DTYPE_Q5_K, 256, 176, false, "Q5_K"},
            {GEIST_DTYPE_Q6_K, 256, 210, true, "Q6_K"},
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof fmts / sizeof fmts[0]; i++)
        fails += run_case(be, &fmts[i]);
    geist_backend_destroy(be);
    if (fails == 0)
        printf("PASS: staged weights compute as their resident twins; refusals hold\n");
    return fails == 0 ? 0 : 1;
}
