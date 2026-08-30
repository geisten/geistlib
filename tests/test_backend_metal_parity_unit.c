/*
 * test_backend_metal_parity_unit — numerical parity gate for the Metal
 * linear path (#181/#296): resolve_weight + linear_m1/linear_mN for Q4_0,
 * Q4_1, Q8_0, Q4_K, Q5_K, Q6_K and F32 weights, compared against cpu_scalar on the SAME
 * weight bytes. cpu_scalar dequantizes with an independent implementation
 * (src/formats/gguf), so agreement means the MSL dequant and the full
 * dispatch chain (pipeline creation, buffer registry, batching) are
 * correct.
 *
 * The Metal linear op falls back to a HOST compute path when a pointer is
 * not a registered backend buffer — a parity test on malloc memory would
 * silently gate the CPU, not the GPU. So x, w and y all come from
 * buffer_create + buffer_map: the registry lookup hits by construction and
 * the GPU path runs. buffer_download(y) drains the pending batch before
 * the compare.
 *
 * Tolerance: GPU sums in f32, reference in double — 1e-3 relative on
 * n_in=512 dots is generous headroom; a dequant bug is orders of magnitude.
 *
 * SKIPs (exit 0) when the metal backend is not built in or no device is
 * present (Linux legs, GPU-less macs).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>
#include "quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail++;
    }
}

static double max_abs(const float *a, const float *b, size_t n) {
    double out = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double) a[i] - b[i]);
        if (d > out)
            out = d;
    }
    return out;
}

enum {
    Q40_BB   = 18,
    Q41_BB   = 20,
    Q80_BB   = 34,
    Q4K_BB   = 144,
    Q5K_BB   = 176,
    IQ4NL_BB = 18,
    IQ4XS_BB = 136,
    Q3K_BB   = 110,
    IQ3S_BB  = 110,
    Q6K_BB   = 210,
    K_BLOCK  = 256
};

static uint32_t rng_state = 0x12345678u;
static uint8_t  rng_u8(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (uint8_t) (rng_state >> 24);
}

/* Random-but-finite quant blob: random bytes, f16 scale fields pinned. */
static void fill_blob(uint8_t *dst, size_t n_in, size_t n_out, int dtype) {
    const bool   small = dtype == GEIST_DTYPE_Q4_0 || dtype == GEIST_DTYPE_Q4_1 ||
                         dtype == GEIST_DTYPE_Q8_0 || dtype == GEIST_DTYPE_IQ4_NL;
    const size_t block = small ? 32u : K_BLOCK;
    const size_t bpr   = n_in / block;
    const size_t bb    = dtype == GEIST_DTYPE_Q4_0     ? Q40_BB
                         : dtype == GEIST_DTYPE_Q4_1   ? Q41_BB
                         : dtype == GEIST_DTYPE_Q8_0   ? Q80_BB
                         : dtype == GEIST_DTYPE_IQ4_NL ? IQ4NL_BB
                         : dtype == GEIST_DTYPE_IQ4_XS ? IQ4XS_BB
                         : dtype == GEIST_DTYPE_Q3_K   ? Q3K_BB
                         : dtype == GEIST_DTYPE_IQ3_S  ? IQ3S_BB
                         : dtype == GEIST_DTYPE_Q4_K   ? Q4K_BB
                         : dtype == GEIST_DTYPE_Q5_K   ? Q5K_BB
                                                       : Q6K_BB;
    for (size_t r = 0; r < n_out; r++) {
        for (size_t b = 0; b < bpr; b++) {
            uint8_t *blk = dst + (r * bpr + b) * bb;
            for (size_t i = 0; i < bb; i++) {
                blk[i] = rng_u8();
            }
            if (dtype == GEIST_DTYPE_Q3_K) {
                blk[108] = 0x00; /* d = fp16(1.0), trailing field */
                blk[109] = 0x3C;
            } else if (dtype == GEIST_DTYPE_IQ3_S) {
                blk[0] = 0x00; /* d = fp16(1.0) */
                blk[1] = 0x3C;
                /* scales nibbles <= 7 keep db = 1+2s <= 15; grid bytes
                 * reach ~46, so staged weights stay under the half
                 * integer lattice. */
                for (size_t sc = 106; sc < 110; sc++) {
                    blk[sc] = rng_u8() & 0x77u;
                }
            } else if (dtype == GEIST_DTYPE_IQ4_XS) {
                blk[0] = 0x00; /* d = fp16(1.0) */
                blk[1] = 0x3C;
                /* Pin the 6-bit sub-scales to |ls-32| <= 4 (scales_h 2-bit
                 * fields = 1, scales_l nibbles = 12..15): LUT values reach
                 * 127, and the GEMM's half staging is exact only under
                 * ~2048 — same reasoning as the Q5_K/Q6_K pins. */
                blk[2] = 0x55;
                blk[3] = 0x55;
                for (size_t sc = 4; sc < 8; sc++) {
                    blk[sc] = (uint8_t) (0xCCu | (rng_u8() & 0x33u));
                }
            } else if (dtype == GEIST_DTYPE_Q4_0 || dtype == GEIST_DTYPE_Q4_1 ||
                       dtype == GEIST_DTYPE_Q8_0 || dtype == GEIST_DTYPE_IQ4_NL) {
                blk[0] = 0x00; /* d = fp16(1.0) */
                blk[1] = 0x3C;
                if (dtype == GEIST_DTYPE_Q4_1) {
                    blk[2] = 0x00; /* m = fp16(0.5) */
                    blk[3] = 0x38;
                }
            } else if (dtype == GEIST_DTYPE_Q4_K || dtype == GEIST_DTYPE_Q5_K) {
                blk[0] = 0x00; /* d    = fp16(1.0)    */
                blk[1] = 0x3C;
                blk[2] = 0x00; /* dmin = fp16(0.5)    */
                blk[3] = 0x38;
                if (dtype == GEIST_DTYPE_Q5_K) {
                    /* Pin the packed 6-bit scales to <=7 (same reasoning as
                     * the Q6_K branch below): Q5_K quants reach 31, and
                     * 63*31 ~ 1953 sits where half's lattice is 1.0, so the
                     * GEMM's half staging would round 0.5 steps. Scales
                     * <=7 keep every staged weight exact. */
                    for (size_t s = 4; s < 12; s++) {
                        blk[s] = rng_u8() & 0x07u;
                    }
                    for (size_t s = 12; s < 16; s++) {
                        blk[s] = rng_u8() & 0x77u;
                    }
                }
            } else {
                blk[208] = 0x00; /* d = fp16(1.0), trailing field in Q6_K */
                blk[209] = 0x3C;
                /* Pin the 16 int8 block scales to small values: the metal
                 * simdgroup GEMM stages dequantized weights in HALF, whose
                 * integer lattice ends at 2048 — random ±127 scales put
                 * weights at ~4000 where half rounds, which is staging
                 * precision, not a dequant bug. Small scales keep every
                 * staged weight exactly representable, so parity is exact
                 * and any addressing bug still explodes. */
                for (size_t s = 192; s < 208; s++) {
                    blk[s] = (uint8_t) ((rng_u8() & 7u) - 3u);
                }
            }
        }
    }
}

/* Backend-owned, registry-visible allocation; returns the mapped host view. */
static void *dev_alloc(struct geist_backend *be, size_t bytes, struct geist_buffer **out) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v->buffer_create(be, bytes, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, out) != GEIST_OK) {
        return nullptr;
    }
    return v->buffer_map(*out);
}

static void run_case(struct geist_backend *mt,
                     struct geist_backend *ref,
                     int                   dtype,
                     const char           *name,
                     size_t                n_in,
                     size_t                n_out,
                     size_t                m) {
    const double tol = 1e-3;
    size_t       w_bytes;
    if (dtype == GEIST_DTYPE_F32) {
        w_bytes = n_in * n_out * sizeof(float);
    } else {
        const size_t block = (dtype == GEIST_DTYPE_Q4_0 || dtype == GEIST_DTYPE_Q4_1 ||
                              dtype == GEIST_DTYPE_Q8_0 || dtype == GEIST_DTYPE_IQ4_NL)
                                     ? 32u
                                     : K_BLOCK;
        const size_t bb    = dtype == GEIST_DTYPE_Q4_0     ? Q40_BB
                             : dtype == GEIST_DTYPE_Q4_1   ? Q41_BB
                             : dtype == GEIST_DTYPE_Q8_0   ? Q80_BB
                             : dtype == GEIST_DTYPE_IQ4_NL ? IQ4NL_BB
                             : dtype == GEIST_DTYPE_IQ4_XS ? IQ4XS_BB
                             : dtype == GEIST_DTYPE_Q3_K   ? Q3K_BB
                             : dtype == GEIST_DTYPE_IQ3_S  ? IQ3S_BB
                             : dtype == GEIST_DTYPE_Q4_K   ? Q4K_BB
                             : dtype == GEIST_DTYPE_Q5_K   ? Q5K_BB
                                                           : Q6K_BB;
        w_bytes            = n_out * (n_in / block) * bb;
    }
    const struct geist_backend_vtbl *v = mt->desc->vtbl;

    struct geist_buffer *bw = nullptr, *bx = nullptr, *by = nullptr;
    uint8_t             *blob = dev_alloc(mt, w_bytes, &bw);
    float               *x    = dev_alloc(mt, m * n_in * sizeof(float), &bx);
    float               *y_mt = dev_alloc(mt, m * n_out * sizeof(float), &by);
    float               *y_dl = malloc(m * n_out * sizeof(float));
    float               *y_rf = malloc(m * n_out * sizeof(float));
    if (blob == nullptr || x == nullptr || y_mt == nullptr || y_dl == nullptr || y_rf == nullptr) {
        check(false, "backend buffer alloc + map");
        return;
    }
    if (dtype == GEIST_DTYPE_F32) {
        float *wf = (float *) blob;
        for (size_t i = 0; i < n_in * n_out; i++) {
            wf[i] = ((float) rng_u8() - 127.5f) / 64.0f;
        }
    } else {
        fill_blob(blob, n_in, n_out, dtype);
    }
    for (size_t i = 0; i < m * n_in; i++) {
        x[i] = ((float) rng_u8() - 127.5f) / 32.0f;
    }

    struct geist_weight w_mt = {.raw        = blob,
                                .raw_nbytes = w_bytes,
                                .n_in       = (int32_t) n_in,
                                .n_out      = (int32_t) n_out,
                                .dtype      = (uint16_t) dtype};
    struct geist_weight w_rf = w_mt;

    check(mt->desc->vtbl->resolve_weight(mt, &w_mt) == GEIST_OK, "metal resolve_weight");
    check(ref->desc->vtbl->resolve_weight(ref, &w_rf) == GEIST_OK, "cpu_scalar resolve_weight");
    if (w_mt.linear_mN == nullptr || w_rf.linear_mN == nullptr) {
        check(false, "resolver installed no kernel");
        return;
    }
    if (m == 1) {
        w_mt.linear_m1(x, &w_mt, mt, y_mt);
        w_rf.linear_m1(x, &w_rf, ref, y_rf);
    } else {
        w_mt.linear_mN(m, x, &w_mt, mt, y_mt);
        w_rf.linear_mN(m, x, &w_rf, ref, y_rf);
    }
    /* Drains the pending GPU batch before reading y. */
    check(v->buffer_download(m * n_out * sizeof(float), (uint8_t *) y_dl, by) == GEIST_OK,
          "y download");

    double max_rel = 0.0;
    for (size_t i = 0; i < m * n_out; i++) {
        const double a   = y_dl[i];
        const double b   = y_rf[i];
        const double rel = fabs(a - b) / (fabs(b) > 1.0 ? fabs(b) : 1.0);
        if (rel > max_rel) {
            max_rel = rel;
        }
    }
    char label[128];
    snprintf(label,
             sizeof label,
             "%s m=%zu (%zux%zu) parity, max_rel=%.2e",
             name,
             m,
             n_out,
             n_in,
             max_rel);
    check(max_rel < tol, label);
    printf("  %-6s m=%-3zu  max_rel %.2e %s\n", name, m, max_rel, max_rel < tol ? "OK" : "FAIL");

    v->buffer_destroy(mt, bw);
    v->buffer_destroy(mt, bx);
    v->buffer_destroy(mt, by);
    free(y_dl);
    free(y_rf);
}

static void run_silu_case(struct geist_backend *mt) {
    const size_t                     n  = 257;
    const struct geist_backend_vtbl *v  = mt->desc->vtbl;
    struct geist_buffer             *bx = nullptr, *by = nullptr;
    float                           *x  = dev_alloc(mt, n * sizeof(float), &bx);
    float                           *y  = dev_alloc(mt, n * sizeof(float), &by);
    float                           *dl = malloc(n * sizeof(float));
    check(x != nullptr && y != nullptr && dl != nullptr, "SiLU buffers");
    if (x == nullptr || y == nullptr || dl == nullptr) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        x[i] = ((float) i - 128.0f) / 16.0f;
    }
    struct geist_tensor tx = {.buffer = bx,
                              .dtype  = GEIST_DTYPE_F32,
                              .layout = GEIST_LAYOUT_DENSE,
                              .ndim   = 1,
                              .shape  = {(int64_t) n},
                              .stride = {1}};
    struct geist_tensor ty = tx;
    ty.buffer              = by;
    check(mt->desc->prims->silu != nullptr, "Metal SiLU primitive installed");
    if (mt->desc->prims->silu != nullptr) {
        check(mt->desc->prims->silu(mt, &tx, &ty) == GEIST_OK, "Metal SiLU dispatch");
        check(v->buffer_download(n * sizeof(float), (uint8_t *) dl, by) == GEIST_OK,
              "Metal SiLU download");
        double max_abs = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double expected = (double) x[i] / (1.0 + exp(-(double) x[i]));
            const double err      = fabs((double) dl[i] - expected);
            if (err > max_abs) {
                max_abs = err;
            }
        }
        check(max_abs < 2e-6, "Metal SiLU parity");
        printf("  SiLU   n=%zu  max_abs %.2e %s\n", n, max_abs, max_abs < 2e-6 ? "OK" : "FAIL");
    }
    v->buffer_destroy(mt, bx);
    v->buffer_destroy(mt, by);
    free(dl);
}

static void run_embedding_case(struct geist_backend *mt, int dtype, const char *name) {
    const size_t vocab = 5, dim = 256, token = 3;
    const size_t block = (dtype == GEIST_DTYPE_Q4_0 || dtype == GEIST_DTYPE_Q8_0) ? 32u : K_BLOCK;
    const size_t bb    = dtype == GEIST_DTYPE_Q4_0   ? Q40_BB
                         : dtype == GEIST_DTYPE_Q8_0 ? Q80_BB
                         : dtype == GEIST_DTYPE_Q4_K ? Q4K_BB
                         : dtype == GEIST_DTYPE_Q5_K ? Q5K_BB
                                                     : Q6K_BB;
    const size_t bytes = vocab * (dim / block) * bb;
    const struct geist_backend_vtbl *v  = mt->desc->vtbl;
    struct geist_buffer             *bt = nullptr, *bo = nullptr;
    uint8_t                         *table    = dev_alloc(mt, bytes, &bt);
    float                           *out      = dev_alloc(mt, dim * sizeof(float), &bo);
    float                           *dl       = malloc(dim * sizeof(float));
    float                           *expected = malloc(dim * sizeof(float));
    check(table != nullptr && out != nullptr && dl != nullptr && expected != nullptr,
          "embedding buffers");
    if (table == nullptr || out == nullptr || dl == nullptr || expected == nullptr) {
        return;
    }
    fill_blob(table, dim, vocab, dtype);
    const uint8_t *row = table + token * (dim / block) * bb;
    if (dtype == GEIST_DTYPE_Q4_0) {
        dequant_q4_0_row(dim, row, expected);
    } else if (dtype == GEIST_DTYPE_Q8_0) {
        dequant_q8_0_row(dim, row, expected);
    } else if (dtype == GEIST_DTYPE_Q4_K) {
        dequant_q4_K_row(dim, row, expected);
    } else if (dtype == GEIST_DTYPE_Q5_K) {
        dequant_q5_K_row(dim, row, expected);
    } else {
        dequant_q6_K_row(dim, row, expected);
    }
    struct geist_tensor tt = {.buffer = bt,
                              .dtype  = (enum geist_dtype) dtype,
                              .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
                              .ndim   = 2,
                              .shape  = {(int64_t) vocab, (int64_t) dim}};
    struct geist_tensor to = {.buffer = bo,
                              .dtype  = GEIST_DTYPE_F32,
                              .layout = GEIST_LAYOUT_DENSE,
                              .ndim   = 1,
                              .shape  = {(int64_t) dim},
                              .stride = {1}};
    check(mt->desc->prims->embedding_lookup(mt, &tt, (geist_token_t) token, &to) == GEIST_OK,
          "Metal embedding dispatch");
    check(v->buffer_download(dim * sizeof(float), (uint8_t *) dl, bo) == GEIST_OK,
          "Metal embedding download");
    double max_abs = 0.0;
    for (size_t i = 0; i < dim; i++) {
        const double err = fabs((double) dl[i] - (double) expected[i]);
        if (err > max_abs) {
            max_abs = err;
        }
    }
    check(max_abs < 1e-6, "Metal embedding parity");
    printf("  embed %-4s max_abs %.2e %s\n", name, max_abs, max_abs < 1e-6 ? "OK" : "FAIL");
    v->buffer_destroy(mt, bt);
    v->buffer_destroy(mt, bo);
    free(dl);
    free(expected);
}

static void run_qwen35_attention_ops(struct geist_backend *mt) {
    enum { ROWS = 3, HEADS = 2, HD = 5, QOUT = HEADS * HD };
    const struct geist_backend_vtbl  *v  = mt->desc->vtbl;
    const struct geist_backend_fused *f  = geist_backend_fused_tbl(mt);
    struct geist_buffer              *bj = nullptr, *bq = nullptr, *bg = nullptr;
    float                            *joint = dev_alloc(mt, ROWS * 2 * QOUT * sizeof(float), &bj);
    float                            *q     = dev_alloc(mt, ROWS * QOUT * sizeof(float), &bq);
    float                            *gate  = dev_alloc(mt, ROWS * QOUT * sizeof(float), &bg);
    float                             got[ROWS * QOUT], expected[ROWS * QOUT];
    check(joint != nullptr && q != nullptr && gate != nullptr, "qwen35 attention-op buffers");
    if (joint == nullptr || q == nullptr || gate == nullptr)
        return;
    for (size_t i = 0; i < ROWS * 2 * QOUT; i++)
        joint[i] = (float) i * 0.07f - 1.1f;
    struct geist_tensor tj = {.buffer = bj,
                              .dtype  = GEIST_DTYPE_F32,
                              .layout = GEIST_LAYOUT_DENSE,
                              .ndim   = 2,
                              .shape  = {ROWS, 2 * QOUT},
                              .stride = {2 * QOUT, 1}};
    struct geist_tensor tq = {.buffer = bq,
                              .dtype  = GEIST_DTYPE_F32,
                              .layout = GEIST_LAYOUT_DENSE,
                              .ndim   = 2,
                              .shape  = {ROWS, QOUT},
                              .stride = {QOUT, 1}};
    struct geist_tensor tg = tq;
    tg.buffer              = bg;
    check(f->attn_qgate_split != nullptr && f->sigmoid_mul != nullptr,
          "qwen35 attention fusions installed");
    check(f->attn_qgate_split(mt, &tj, HEADS, HD, &tq, &tg) == GEIST_OK, "qgate split dispatch");
    check(mt->desc->prims->scale_f32(mt, &tq, 0.25f, &tq) == GEIST_OK, "q scale dispatch");
    check(f->sigmoid_mul(mt, &tq, &tg, &tq) == GEIST_OK, "sigmoid gate dispatch");
    check(v->buffer_download(sizeof got, (uint8_t *) got, bq) == GEIST_OK,
          "qwen35 attention-op download");
    for (size_t r = 0; r < ROWS; r++) {
        for (size_t h = 0; h < HEADS; h++) {
            for (size_t d = 0; d < HD; d++) {
                const size_t dst = r * QOUT + h * HD + d;
                const size_t src = r * 2 * QOUT + h * 2 * HD + d;
                const float  g   = joint[src + HD];
                expected[dst]    = joint[src] * 0.25f / (1.0f + expf(-g));
            }
        }
    }
    const double err = max_abs(got, expected, ROWS * QOUT);
    check(err < 2e-6, "qwen35 attention ops parity");
    printf("  qgate split + qscale + sigmoid_mul max_abs %.2e %s\n",
           err,
           err < 2e-6 ? "OK" : "FAIL");
    v->buffer_destroy(mt, bj);
    v->buffer_destroy(mt, bq);
    v->buffer_destroy(mt, bg);
}

int main(void) {
    struct geist_backend *mt = nullptr;
    enum geist_status     ms = geist_backend_create("metal", nullptr, nullptr, &mt);
    if (ms == GEIST_E_UNSUPPORTED || ms == GEIST_E_NOT_FOUND) {
        /* Not built in (Linux legs) or no Metal device — skip. */
        fprintf(stderr, "SKIP: metal backend unavailable (not built or no device)\n");
        return GEIST_TEST_SKIP;
    }
    struct geist_backend *ref = nullptr;
    check(mt != nullptr, "metal backend");
    check(geist_backend_create("cpu_scalar", nullptr, nullptr, &ref) == GEIST_OK,
          "cpu_scalar backend");
    if (mt == nullptr || ref == nullptr) {
        return 1;
    }

    /* n_in must be a multiple of 256 for k-quants; n_out deliberately not a
     * multiple of the threadgroup width to catch tail bugs. */
    run_case(mt, ref, GEIST_DTYPE_Q4_0, "Q4_0", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q4_0, "Q4_0", 512, 383, 8);
    /* m=33: two batch-row tiles of the simdgroup GEMM, both partial.
     * m=32 x n_out=384: full tiles -> the _fast interior variant. */
    run_case(mt, ref, GEIST_DTYPE_Q4_0, "Q4_0", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_Q4_0, "Q4_0", 512, 384, 32);
    run_case(mt, ref, GEIST_DTYPE_Q4_1, "Q4_1", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q4_1, "Q4_1", 512, 383, 8);
    run_case(mt, ref, GEIST_DTYPE_Q4_1, "Q4_1", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_Q4_1, "Q4_1", 512, 384, 32);
    run_case(mt, ref, GEIST_DTYPE_Q8_0, "Q8_0", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q8_0, "Q8_0", 512, 383, 8);
    run_case(mt, ref, GEIST_DTYPE_Q8_0, "Q8_0", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_Q8_0, "Q8_0", 512, 384, 32);
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 8);
    run_case(mt, ref, GEIST_DTYPE_Q5_K, "Q5_K", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q5_K, "Q5_K", 512, 383, 8);
    run_case(mt, ref, GEIST_DTYPE_Q5_K, "Q5_K", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_Q5_K, "Q5_K", 512, 384, 32);
    /* IQ4: n4 GEMV at m=1; the bounded simdgroup GEMM covers m>=2 (no
     * naive kernels) — m=4 pins the small-rows route, m=33 the partial
     * tiles. */
    run_case(mt, ref, GEIST_DTYPE_IQ4_NL, "IQ4NL", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_IQ4_NL, "IQ4NL", 512, 383, 4);
    run_case(mt, ref, GEIST_DTYPE_IQ4_NL, "IQ4NL", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_IQ4_XS, "IQ4XS", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_IQ4_XS, "IQ4XS", 512, 383, 4);
    run_case(mt, ref, GEIST_DTYPE_IQ4_XS, "IQ4XS", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_Q3_K, "Q3_K", 512, 384, 32);
    run_case(mt, ref, GEIST_DTYPE_Q3_K, "Q3_K", 512, 384, 8);
    run_case(mt, ref, GEIST_DTYPE_Q3_K, "Q3_K", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q3_K, "Q3_K", 512, 383, 4);
    run_case(mt, ref, GEIST_DTYPE_Q3_K, "Q3_K", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_IQ3_S, "IQ3_S", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_IQ3_S, "IQ3_S", 512, 383, 4);
    run_case(mt, ref, GEIST_DTYPE_IQ3_S, "IQ3_S", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 8);
    run_case(mt, ref, GEIST_DTYPE_F32, "F32", 256, 130, 1);
    run_case(mt, ref, GEIST_DTYPE_F32, "F32", 256, 130, 8);
    run_silu_case(mt);
    run_embedding_case(mt, GEIST_DTYPE_Q4_0, "Q4_0");
    run_embedding_case(mt, GEIST_DTYPE_Q8_0, "Q8_0");
    run_qwen35_attention_ops(mt);

    geist_backend_destroy(mt);
    geist_backend_destroy(ref);
    if (g_fail == 0) {
        printf("test_backend_metal_parity_unit: all checks passed\n");
    }
    return g_fail == 0 ? 0 : 1;
}
