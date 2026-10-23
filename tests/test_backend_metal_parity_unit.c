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

#ifdef GEIST_BACKEND_METAL
#include "../src/backends/metal/metal_internal.h"
#endif

static int g_fail = 0;
#define check(ok, what) (g_fail |= geist_expect((ok), (what)))

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
    PQ2_BB   = 34,
    Q6K_BB   = 210,
    K_BLOCK  = 256
};

static uint32_t rng_state = 0x12345678u;
static uint8_t  rng_u8(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (uint8_t) (rng_state >> 24);
}

/* Block geometry per quant dtype, from the canonical constants. */
static void block_geom(int dtype, size_t *elems, size_t *bytes) {
    switch (dtype) {
    case GEIST_DTYPE_PQ2_0:
        *elems = PQ2_0_BLOCK_ELEMS, *bytes = PQ2_BB;
        return;
    case GEIST_DTYPE_TQ2_0:
        *elems = TQ2_0_BLOCK_ELEMS, *bytes = TQ2_0_BLOCK_BYTES;
        return;
    case GEIST_DTYPE_I2_S:
        *elems = I2_S_BLOCK_ELEMS, *bytes = I2_S_BLOCK_BYTES;
        return;
    case GEIST_DTYPE_Q4_0:
        *elems = 32u, *bytes = Q40_BB;
        return;
    case GEIST_DTYPE_Q4_1:
        *elems = 32u, *bytes = Q41_BB;
        return;
    case GEIST_DTYPE_Q8_0:
        *elems = 32u, *bytes = Q80_BB;
        return;
    case GEIST_DTYPE_IQ4_NL:
        *elems = 32u, *bytes = IQ4NL_BB;
        return;
    case GEIST_DTYPE_IQ4_XS:
        *elems = K_BLOCK, *bytes = IQ4XS_BB;
        return;
    case GEIST_DTYPE_Q3_K:
        *elems = K_BLOCK, *bytes = Q3K_BB;
        return;
    case GEIST_DTYPE_IQ3_S:
        *elems = K_BLOCK, *bytes = IQ3S_BB;
        return;
    case GEIST_DTYPE_Q4_K:
        *elems = K_BLOCK, *bytes = Q4K_BB;
        return;
    case GEIST_DTYPE_Q5_K:
        *elems = K_BLOCK, *bytes = Q5K_BB;
        return;
    default:
        *elems = K_BLOCK, *bytes = Q6K_BB;
        return;
    }
}

/* F16 bits of a normal value exactly representable in half (the dense
 * cases' weights: odd multiples of 1/128 below 2). */
static uint16_t half_bits_exact(float v) {
    if (v == 0.0f) {
        return 0;
    }
    int         e;
    const float m = frexpf(fabsf(v), &e); /* |v| = m * 2^e, m in [0.5, 1) */
    return (uint16_t) ((v < 0.0f ? 0x8000u : 0u) | ((unsigned) (e + 14) << 10) |
                       (unsigned) ((m * 2.0f - 1.0f) * 1024.0f));
}

/* Random-but-finite quant blob: random bytes, f16 scale fields pinned. */
static void fill_blob(uint8_t *dst, size_t n_in, size_t n_out, int dtype) {
    size_t block = 0, bb = 0;
    block_geom(dtype, &block, &bb);
    const size_t bpr = n_in / block;
    for (size_t r = 0; r < n_out; r++) {
        for (size_t b = 0; b < bpr; b++) {
            uint8_t *blk = dst + (r * bpr + b) * bb;
            for (size_t i = 0; i < bb; i++) {
                blk[i] = rng_u8();
            }
            if (dtype == GEIST_DTYPE_PQ2_0) {
                /* d in {0.5, 1, 2, 4} per block (exact in half), so a kernel
                 * reading the wrong block's scale shows; codes stay random,
                 * 3 = +2 included. */
                blk[0] = 0x00;
                blk[1] = (uint8_t) (0x38u + 4u * (rng_u8() & 3u));
            } else if (dtype == GEIST_DTYPE_TQ2_0) {
                /* trailing d in {0.5, 1, 2, 4}, as for PQ2_0 */
                blk[64] = 0x00;
                blk[65] = (uint8_t) (0x38u + 4u * (rng_u8() & 3u));
            } else if (dtype == GEIST_DTYPE_Q3_K) {
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
    } else if (dtype == GEIST_DTYPE_F16 || dtype == GEIST_DTYPE_BF16) {
        w_bytes = n_in * n_out * sizeof(uint16_t);
    } else {
        size_t block = 0, bb = 0;
        block_geom(dtype, &block, &bb);
        w_bytes = n_out * (n_in / block) * bb;
        if (dtype == GEIST_DTYPE_I2_S) {
            w_bytes += sizeof(float); /* per-tensor scale after the blocks */
        }
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
    } else if (dtype == GEIST_DTYPE_F16 || dtype == GEIST_DTYPE_BF16) {
        /* (u8 - 127.5) / 64 is exact in both: 8 significant bits. */
        uint16_t *wh = (uint16_t *) blob;
        for (size_t i = 0; i < n_in * n_out; i++) {
            const float v = ((float) rng_u8() - 127.5f) / 64.0f;
            uint32_t    u;
            memcpy(&u, &v, sizeof u);
            wh[i] = dtype == GEIST_DTYPE_BF16 ? (uint16_t) (u >> 16) : half_bits_exact(v);
        }
    } else {
        fill_blob(blob, n_in, n_out, dtype);
        if (dtype == GEIST_DTYPE_I2_S) {
            const float scale = 0.75f; /* not 1, so a dropped scale shows */
            memcpy(blob + i2_s_scale_offset(n_in * n_out), &scale, sizeof scale);
        }
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
    /* relu_squared (BitNet b1.58 2B-4T FFN, #560): exact, it is one max and
     * one multiply in F32 like relu_squared_fp32. */
    check(mt->desc->prims->relu_squared != nullptr, "Metal relu_squared primitive installed");
    if (mt->desc->prims->relu_squared != nullptr) {
        check(mt->desc->prims->relu_squared(mt, &tx, &ty) == GEIST_OK,
              "Metal relu_squared dispatch");
        check(v->buffer_download(n * sizeof(float), (uint8_t *) dl, by) == GEIST_OK,
              "Metal relu_squared download");
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) {
            const float r = x[i] > 0.0f ? x[i] : 0.0f;
            bad += dl[i] != r * r;
        }
        check(bad == 0, "Metal relu_squared parity");
        printf("  relu2  n=%zu  mismatches %zu %s\n", n, bad, bad == 0 ? "OK" : "FAIL");
    }
    v->buffer_destroy(mt, bx);
    v->buffer_destroy(mt, by);
    free(dl);
}

static void run_embedding_case(struct geist_backend *mt, int dtype, const char *name) {
    const size_t vocab = 5, dim = 256, token = 3;
    size_t       block = 0, bb = 0;
    block_geom(dtype, &block, &bb);
    const size_t                     bytes = vocab * (dim / block) * bb;
    const struct geist_backend_vtbl *v     = mt->desc->vtbl;
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
    } else if (dtype == GEIST_DTYPE_PQ2_0) {
        dequant_pq2_0_row(dim, row, expected);
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
    check(geist_backend_fused_tbl(mt)->embedding_lookup_scaled(
                  mt, &tt, (geist_token_t) token, 1.0f, &to) == GEIST_OK,
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

#ifdef GEIST_BACKEND_METAL
/* #529: an upload into a buffer the open command sequence reads is staged
 * behind those reads. The sequence is not flushed; a copy encoded before
 * the upload sees the old bytes and one encoded after it the new. */
static void run_staged_upload(struct geist_backend *mt) {
    const struct geist_backend_vtbl *v  = mt->desc->vtbl;
    const struct metal_state        *st = mt->state;
    float                            old[64], neu[64], got[64];
    for (size_t i = 0; i < 64; i++) {
        old[i] = (float) i;
        neu[i] = 1000.0f + (float) i;
    }
    struct geist_buffer *x = nullptr, *y = nullptr, *z = nullptr;
    const bool           ok = dev_alloc(mt, sizeof old, &x) != nullptr &&
                              dev_alloc(mt, sizeof old, &y) != nullptr &&
                              dev_alloc(mt, sizeof old, &z) != nullptr &&
                              metal_ensure_attention_pipeline(mt) == GEIST_OK &&
                              v->buffer_upload(x, sizeof old, (const uint8_t *) old) == GEIST_OK;
    check(ok, "staged upload: buffers");
    if (ok) {
        const int tok = v->parallel_region_begin(mt, GEIST_REGION_PREFILL_BATCH);
        check(v->buffer_copy(y, 0, x, 0, sizeof old) == GEIST_OK, "staged upload: copy before");
        const int seq = st->sequence_token;
        check(v->buffer_upload(x, sizeof neu, (const uint8_t *) neu) == GEIST_OK,
              "staged upload: upload");
        check(st->sequence_token == seq, "staged upload: the sequence is not flushed");
        check(v->buffer_copy(z, 0, x, 0, sizeof neu) == GEIST_OK, "staged upload: copy after");
        v->parallel_region_end(mt, tok);
        check(v->buffer_download(sizeof got, (uint8_t *) got, y) == GEIST_OK &&
                      memcmp(got, old, sizeof got) == 0,
              "staged upload: the earlier copy sees the old bytes");
        check(v->buffer_download(sizeof got, (uint8_t *) got, z) == GEIST_OK &&
                      memcmp(got, neu, sizeof got) == 0,
              "staged upload: the later copy sees the new bytes");
    }
    v->buffer_destroy(mt, x);
    v->buffer_destroy(mt, y);
    v->buffer_destroy(mt, z);
}
#endif

/* #64: attention on an F32 KV cache must give the same bits as attention
 * on an F16 cache that holds the half-rounded values. Decode (q_rows 1,
 * head_dim 256 and 512, sliding window) reads the F32 cache directly and
 * rounds as it loads; the flash cases (4- and 8-simdgroup kernels, a
 * ragged last 8-row tile, GQA, a sliding window) go through the f16
 * staging copy. The K/V views start mid-buffer at an odd element. */
static void run_f32kv_attention_case(struct geist_backend *mt,
                                     size_t                q_rows,
                                     size_t                q_offset,
                                     size_t                q_heads,
                                     size_t                kv_heads,
                                     size_t                hd,
                                     size_t                window) {
    enum { PAD = 7 };
    const struct geist_backend_vtbl *v       = mt->desc->vtbl;
    const size_t                     k_rows  = q_offset + q_rows;
    const size_t                     q_elems = q_rows * q_heads * hd;
    const size_t                     kv      = k_rows * kv_heads * hd;
    struct geist_buffer             *bq = nullptr, *b32 = nullptr, *b16 = nullptr;
    struct geist_buffer             *bo32 = nullptr, *bo16 = nullptr;
    float                           *q   = dev_alloc(mt, q_elems * sizeof(float), &bq);
    float                           *c32 = dev_alloc(mt, (PAD + 2 * kv) * sizeof(float), &b32);
    uint16_t                        *c16 = dev_alloc(mt, (PAD + 2 * kv) * sizeof(uint16_t), &b16);
    float                           *o32 = dev_alloc(mt, q_elems * sizeof(float), &bo32);
    float                           *o16 = dev_alloc(mt, q_elems * sizeof(float), &bo16);
    const bool                       ok =
            q != nullptr && c32 != nullptr && c16 != nullptr && o32 != nullptr && o16 != nullptr;
    check(ok, "f32-KV attention buffers");
    if (ok) {
        for (size_t i = 0; i < q_elems; i++) {
            q[i] = ((float) rng_u8() - 127.5f) / 32.0f;
        }
        for (size_t i = 0; i < PAD + 2 * kv; i++) {
            c32[i] = ((float) rng_u8() - 127.5f) / 48.0f + (float) rng_u8() * 1e-5f;
            c16[i] = fp32_to_fp16_rne(c32[i]);
        }
        const struct geist_tensor tq = {
                .buffer = bq,
                .dtype  = GEIST_DTYPE_F32,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim   = 3,
                .shape  = {(int64_t) q_rows, (int64_t) q_heads, (int64_t) hd},
                .stride = {(int64_t) (q_heads * hd), (int64_t) hd, 1}};
        struct geist_tensor tk32 = {.buffer = b32,
                                    .dtype  = GEIST_DTYPE_F32,
                                    .layout = GEIST_LAYOUT_DENSE,
                                    .offset = PAD * sizeof(float),
                                    .ndim   = 3,
                                    .shape  = {(int64_t) k_rows, (int64_t) kv_heads, (int64_t) hd},
                                    .stride = {(int64_t) (kv_heads * hd), (int64_t) hd, 1}};
        struct geist_tensor tv32 = tk32;
        tv32.offset              = (PAD + kv) * sizeof(float);
        struct geist_tensor tk16 = tk32;
        tk16.buffer              = b16;
        tk16.dtype               = GEIST_DTYPE_F16;
        tk16.offset              = PAD * sizeof(uint16_t);
        struct geist_tensor tv16 = tk16;
        tv16.offset              = (PAD + kv) * sizeof(uint16_t);
        struct geist_tensor to32 = tq;
        to32.buffer              = bo32;
        struct geist_tensor to16 = tq;
        to16.buffer              = bo16;
        const struct geist_backend_primitives *pr = mt->desc->prims;
        check(pr->attention(mt, &tq, &tk32, &tv32, q_offset, window, &to32) == GEIST_OK &&
                      pr->attention(mt, &tq, &tk16, &tv16, q_offset, window, &to16) == GEIST_OK,
              "f32-KV attention dispatch");
        float *g32 = malloc(q_elems * sizeof(float));
        float *g16 = malloc(q_elems * sizeof(float));
        check(g32 != nullptr && g16 != nullptr &&
                      v->buffer_download(q_elems * sizeof(float), (uint8_t *) g32, bo32) ==
                              GEIST_OK &&
                      v->buffer_download(q_elems * sizeof(float), (uint8_t *) g16, bo16) ==
                              GEIST_OK,
              "f32-KV attention download");
        const bool same =
                g32 != nullptr && g16 != nullptr && memcmp(g32, g16, q_elems * sizeof(float)) == 0;
        check(same, "f32-KV attention bit-identical to f16 KV of the rounded values");
        printf("  attention f32 KV vs f16 KV rows %zu at %zu heads %zu/%zu hd %zu window %zu %s\n",
               q_rows,
               q_offset,
               q_heads,
               kv_heads,
               hd,
               window,
               same ? "OK" : "FAIL");
        free(g32);
        free(g16);
    }
    v->buffer_destroy(mt, bq);
    v->buffer_destroy(mt, b32);
    v->buffer_destroy(mt, b16);
    v->buffer_destroy(mt, bo32);
    v->buffer_destroy(mt, bo16);
}

/* #91: the prefill flash kernels (and, at one query row, the split-KV
 * decode kernels, #17) against cpu_scalar's attention on the
 * same half-rounded K/V. The Metal kernels round Q and the softmax weights
 * to half for the f16 matrix products and accumulate in f32, so the bound
 * is a tolerance, not bits: 1e-2 absolute on outputs of magnitude <= 2.7
 * (the half-rounded Q alone costs ~4e-3 on these data, with either
 * kernel); a masking or tiling bug is off by 0.1 or more.
 * The cases cover head_dims on 4 simdgroups (32/64/96/128/256) and on 8
 * (384/512), GQA, ragged last row tiles, sliding windows narrower than a
 * 64-key block and wider than the chunk, caches that are not a multiple of
 * 64 rows, two query rows, and caches under 64 rows (the scalar kernel). */
static void run_flash_ref_case(struct geist_backend *mt,
                               struct geist_backend *ref,
                               size_t                q_rows,
                               size_t                q_offset,
                               size_t                q_heads,
                               size_t                kv_heads,
                               size_t                hd,
                               size_t                window) {
    const struct geist_backend_vtbl *v       = mt->desc->vtbl;
    const struct geist_backend_vtbl *rv      = ref->desc->vtbl;
    const size_t                     k_rows  = q_offset + q_rows;
    const size_t                     q_elems = q_rows * q_heads * hd;
    const size_t                     kv      = k_rows * kv_heads * hd;
    struct geist_buffer             *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr;
    struct geist_buffer             *rq = nullptr, *rk = nullptr, *rvb = nullptr, *ro = nullptr;
    float                           *q   = dev_alloc(mt, q_elems * sizeof(float), &bq);
    uint16_t                        *k16 = dev_alloc(mt, kv * sizeof(uint16_t), &bk);
    uint16_t                        *v16 = dev_alloc(mt, kv * sizeof(uint16_t), &bv);
    float                           *o   = dev_alloc(mt, q_elems * sizeof(float), &bo);
    float                           *qr  = dev_alloc(ref, q_elems * sizeof(float), &rq);
    float                           *kr  = dev_alloc(ref, kv * sizeof(float), &rk);
    float                           *vr  = dev_alloc(ref, kv * sizeof(float), &rvb);
    float                           *orf = dev_alloc(ref, q_elems * sizeof(float), &ro);
    float                           *got = malloc(q_elems * sizeof(float));
    const bool ok = q != nullptr && k16 != nullptr && v16 != nullptr && o != nullptr &&
                    qr != nullptr && kr != nullptr && vr != nullptr && orf != nullptr &&
                    got != nullptr;
    check(ok, "flash reference buffers");
    if (ok) {
        /* scores of a few units: the softmax is neither flat nor one-hot */
        const float qs = 1.0f / (24.0f * sqrtf((float) hd));
        for (size_t i = 0; i < q_elems; i++) {
            q[i] = qr[i] = ((float) rng_u8() - 127.5f) * qs;
        }
        for (size_t i = 0; i < kv; i++) {
            k16[i] = fp32_to_fp16_rne(((float) rng_u8() - 127.5f) / 48.0f);
            v16[i] = fp32_to_fp16_rne(((float) rng_u8() - 127.5f) / 48.0f);
            kr[i]  = fp16_to_fp32(k16[i]);
            vr[i]  = fp16_to_fp32(v16[i]);
        }
        const struct geist_tensor tq = {
                .buffer = bq,
                .dtype  = GEIST_DTYPE_F32,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim   = 3,
                .shape  = {(int64_t) q_rows, (int64_t) q_heads, (int64_t) hd},
                .stride = {(int64_t) (q_heads * hd), (int64_t) hd, 1}};
        struct geist_tensor tk  = {.buffer = bk,
                                   .dtype  = GEIST_DTYPE_F16,
                                   .layout = GEIST_LAYOUT_DENSE,
                                   .ndim   = 3,
                                   .shape  = {(int64_t) k_rows, (int64_t) kv_heads, (int64_t) hd},
                                   .stride = {(int64_t) (kv_heads * hd), (int64_t) hd, 1}};
        struct geist_tensor tv  = tk;
        tv.buffer               = bv;
        struct geist_tensor to  = tq;
        to.buffer               = bo;
        struct geist_tensor tqr = tq;
        tqr.buffer              = rq;
        struct geist_tensor tkr = tk;
        tkr.buffer              = rk;
        tkr.dtype               = GEIST_DTYPE_F32;
        struct geist_tensor tvr = tkr;
        tvr.buffer              = rvb;
        struct geist_tensor tor = tq;
        tor.buffer              = ro;
        check(mt->desc->prims->attention(mt, &tq, &tk, &tv, q_offset, window, &to) == GEIST_OK &&
                      ref->desc->prims->attention(ref, &tqr, &tkr, &tvr, q_offset, window, &tor) ==
                              GEIST_OK,
              "flash reference dispatch");
        check(v->buffer_download(q_elems * sizeof(float), (uint8_t *) got, bo) == GEIST_OK,
              "flash reference download");
        (void) rv;
        /* Decode (one row) keeps Q and the weights in f32: only the
         * summation order differs from cpu_scalar. */
        const double err  = geist_test_max_abs(q_elems, got, orf);
        const bool   pass = err < (q_rows == 1 ? 2e-5 : 1e-2);
        check(pass, "flash attention matches cpu_scalar");
        printf("  flash vs cpu_scalar rows %zu at %zu heads %zu/%zu hd %zu window %zu max_abs "
               "%.2e %s\n",
               q_rows,
               q_offset,
               q_heads,
               kv_heads,
               hd,
               window,
               err,
               pass ? "OK" : "FAIL");
    }
    free(got);
    v->buffer_destroy(mt, bq);
    v->buffer_destroy(mt, bk);
    v->buffer_destroy(mt, bv);
    v->buffer_destroy(mt, bo);
    rv->buffer_destroy(ref, rq);
    rv->buffer_destroy(ref, rk);
    rv->buffer_destroy(ref, rvb);
    rv->buffer_destroy(ref, ro);
}

/* #17: rmsnorm and the fused rmsnorm_add against cpu_scalar's rmsnorm
 * (and a host add). One row of 1024..3072 is the register path at the
 * decode thread counts (cols/4); 5000 is wider than four per thread at
 * 1024 threads and 256-wide rows exercise the 256-thread path; several
 * rows are prefill (256 threads: up to 1024 columns in registers, the
 * loop beyond). The sums differ in order only: 1e-5 on outputs of a few
 * units, where a lane or a partial left out is off by 1e-2 or more. */
static void
run_norm_case(struct geist_backend *mt, struct geist_backend *ref, size_t rows, size_t cols) {
    const struct geist_backend_vtbl *v  = mt->desc->vtbl;
    const struct geist_backend_vtbl *rv = ref->desc->vtbl;
    const size_t                     n  = rows * cols;
    struct geist_buffer *bx = nullptr, *bw = nullptr, *br = nullptr, *by = nullptr, *bz = nullptr;
    struct geist_buffer *rx = nullptr, *rw = nullptr, *ry = nullptr;
    float               *x   = dev_alloc(mt, n * sizeof(float), &bx);
    float               *w   = dev_alloc(mt, cols * sizeof(float), &bw);
    float               *res = dev_alloc(mt, n * sizeof(float), &br);
    float               *y   = dev_alloc(mt, n * sizeof(float), &by);
    float               *z   = dev_alloc(mt, n * sizeof(float), &bz);
    float               *xr  = dev_alloc(ref, n * sizeof(float), &rx);
    float               *wr  = dev_alloc(ref, cols * sizeof(float), &rw);
    float               *yr  = dev_alloc(ref, n * sizeof(float), &ry);
    float               *got = malloc(n * sizeof(float));
    const bool           ok  = x != nullptr && w != nullptr && res != nullptr && y != nullptr &&
                               z != nullptr && xr != nullptr && wr != nullptr && yr != nullptr &&
                               got != nullptr;
    check(ok, "norm buffers");
    if (ok) {
        for (size_t i = 0; i < n; i++) {
            x[i] = xr[i] = ((float) rng_u8() - 127.5f) / 32.0f;
            res[i]       = ((float) rng_u8() - 127.5f) / 64.0f;
        }
        for (size_t i = 0; i < cols; i++) {
            w[i] = wr[i] = 0.5f + (float) rng_u8() / 256.0f;
        }
        const struct geist_tensor tx   = {.buffer = bx,
                                          .dtype  = GEIST_DTYPE_F32,
                                          .layout = GEIST_LAYOUT_DENSE,
                                          .ndim   = 2,
                                          .shape  = {(int64_t) rows, (int64_t) cols},
                                          .stride = {(int64_t) cols, 1}};
        const struct geist_tensor tw   = {.buffer = bw,
                                          .dtype  = GEIST_DTYPE_F32,
                                          .layout = GEIST_LAYOUT_DENSE,
                                          .ndim   = 1,
                                          .shape  = {(int64_t) cols},
                                          .stride = {1}};
        struct geist_tensor       tres = tx, ty = tx, tz = tx, txr = tx, twr = tw, tyr = tx;
        tres.buffer = br;
        ty.buffer   = by;
        tz.buffer   = bz;
        txr.buffer  = rx;
        twr.buffer  = rw;
        tyr.buffer  = ry;
        check(mt->desc->prims->rmsnorm(mt, &tx, &tw, 1e-6f, &ty) == GEIST_OK &&
                      mt->desc->fused->rmsnorm_add(mt, &tres, &tx, &tw, 1e-6f, &tz) == GEIST_OK &&
                      ref->desc->prims->rmsnorm(ref, &txr, &twr, 1e-6f, &tyr) == GEIST_OK,
              "norm dispatch");
        check(v->buffer_download(n * sizeof(float), (uint8_t *) got, by) == GEIST_OK,
              "norm download");
        const double err = geist_test_max_abs(n, got, yr);
        check(v->buffer_download(n * sizeof(float), (uint8_t *) got, bz) == GEIST_OK,
              "norm_add download");
        double err_add = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double d = fabs((double) got[i] - ((double) res[i] + (double) yr[i]));
            err_add        = d > err_add ? d : err_add;
        }
        const bool pass = err < 1e-5 && err_add < 1e-5;
        check(pass, "rmsnorm / rmsnorm_add match cpu_scalar");
        printf("  rmsnorm rows %zu cols %zu max_abs %.2e, rmsnorm_add %.2e %s\n",
               rows,
               cols,
               err,
               err_add,
               pass ? "OK" : "FAIL");
    }
    free(got);
    v->buffer_destroy(mt, bx);
    v->buffer_destroy(mt, bw);
    v->buffer_destroy(mt, br);
    v->buffer_destroy(mt, by);
    v->buffer_destroy(mt, bz);
    rv->buffer_destroy(ref, rx);
    rv->buffer_destroy(ref, rw);
    rv->buffer_destroy(ref, ry);
}

/* #17: rmsnorm_add_norm is rmsnorm_add (or add), scale_f32 and rmsnorm on
 * the same backend in one dispatch, and must give their bits: the decode
 * path swaps one for the other with greedy output unchanged. With w and a
 * scale (Gemma 4's PLE tail), and without w at scale 1 (Llama's residual
 * add). */
static void run_add_norm_case(struct geist_backend *mt, size_t rows, size_t cols, bool with_w) {
    const struct geist_backend_vtbl  *v     = mt->desc->vtbl;
    const struct geist_backend_fused *f     = geist_backend_fused_tbl(mt);
    const size_t                      n     = rows * cols;
    const float                       scale = with_w ? 0.6875f : 1.0f;
    struct geist_buffer              *bres = nullptr, *bx = nullptr, *bw = nullptr, *bzw = nullptr;
    struct geist_buffer              *by1 = nullptr, *bz1 = nullptr, *by2 = nullptr, *bz2 = nullptr;
    float                            *res = dev_alloc(mt, n * sizeof(float), &bres);
    float                            *x   = dev_alloc(mt, n * sizeof(float), &bx);
    float                            *w   = dev_alloc(mt, cols * sizeof(float), &bw);
    float                            *zw  = dev_alloc(mt, cols * sizeof(float), &bzw);
    const bool alloc = res != nullptr && x != nullptr && w != nullptr && zw != nullptr &&
                       dev_alloc(mt, n * sizeof(float), &by1) != nullptr &&
                       dev_alloc(mt, n * sizeof(float), &bz1) != nullptr &&
                       dev_alloc(mt, n * sizeof(float), &by2) != nullptr &&
                       dev_alloc(mt, n * sizeof(float), &bz2) != nullptr;
    float     *got1  = malloc(2 * n * sizeof(float));
    float     *got2  = malloc(2 * n * sizeof(float));
    const bool ok = alloc && got1 != nullptr && got2 != nullptr && f->rmsnorm_add_norm != nullptr;
    check(ok, "add_norm buffers");
    if (ok) {
        for (size_t i = 0; i < n; i++) {
            x[i]   = ((float) rng_u8() - 127.5f) / 32.0f;
            res[i] = ((float) rng_u8() - 127.5f) / 64.0f;
        }
        for (size_t i = 0; i < cols; i++) {
            w[i]  = 0.5f + (float) rng_u8() / 256.0f;
            zw[i] = 0.25f + (float) rng_u8() / 128.0f;
        }
        const struct geist_tensor trow = {.buffer = bx,
                                          .dtype  = GEIST_DTYPE_F32,
                                          .layout = GEIST_LAYOUT_DENSE,
                                          .ndim   = 2,
                                          .shape  = {(int64_t) rows, (int64_t) cols},
                                          .stride = {(int64_t) cols, 1}};
        const struct geist_tensor tvec = {.buffer = bw,
                                          .dtype  = GEIST_DTYPE_F32,
                                          .layout = GEIST_LAYOUT_DENSE,
                                          .ndim   = 1,
                                          .shape  = {(int64_t) cols},
                                          .stride = {1}};
        struct geist_tensor tx = trow, tres = trow, ty1 = trow, tz1 = trow, ty2 = trow, tz2 = trow,
                            tw = tvec, tzw = tvec;
        tres.buffer         = bres;
        ty1.buffer          = by1;
        tz1.buffer          = bz1;
        ty2.buffer          = by2;
        tz2.buffer          = bz2;
        tzw.buffer          = bzw;
        enum geist_status s = f->rmsnorm_add_norm(
                mt, 1e-6f, scale, &tres, &tx, with_w ? &tw : nullptr, &tzw, &ty1, &tz1);
        if (s == GEIST_OK)
            s = with_w ? f->rmsnorm_add(mt, &tres, &tx, &tw, 1e-6f, &ty2)
                       : mt->desc->prims->add(mt, &tres, &tx, &ty2);
        if (s == GEIST_OK && scale != 1.0f)
            s = mt->desc->prims->scale_f32(mt, &ty2, scale, &ty2);
        if (s == GEIST_OK)
            s = mt->desc->prims->rmsnorm(mt, &ty2, &tzw, 1e-6f, &tz2);
        check(s == GEIST_OK, "add_norm dispatch");
        check(v->buffer_download(n * sizeof(float), (uint8_t *) got1, by1) == GEIST_OK &&
                      v->buffer_download(n * sizeof(float), (uint8_t *) (got1 + n), bz1) ==
                              GEIST_OK &&
                      v->buffer_download(n * sizeof(float), (uint8_t *) got2, by2) == GEIST_OK &&
                      v->buffer_download(n * sizeof(float), (uint8_t *) (got2 + n), bz2) ==
                              GEIST_OK,
              "add_norm download");
        const bool same = memcmp(got1, got2, 2 * n * sizeof(float)) == 0;
        check(same, "rmsnorm_add_norm matches the separate ops bit for bit");
        printf("  rmsnorm_add_norm rows %zu cols %zu %s: %s\n",
               rows,
               cols,
               with_w ? "norm+scale" : "add",
               same ? "bit-identical OK" : "FAIL");
    }
    free(got1);
    free(got2);
    struct geist_buffer *bufs[] = {bres, bx, bw, bzw, by1, bz1, by2, bz2};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (bufs[i] != nullptr)
            v->buffer_destroy(mt, bufs[i]);
    }
}

/* A random quant weight of dtype in a fresh backend buffer, with its
 * tensor view; the bytes as fill_blob makes them. */
static uint8_t *quant_weight(struct geist_backend *mt,
                             int                   dtype,
                             size_t                n_in,
                             size_t                n_out,
                             struct geist_buffer **buf,
                             struct geist_tensor  *t) {
    size_t block = 0, bb = 0;
    block_geom(dtype, &block, &bb);
    uint8_t *blob = dev_alloc(mt, n_out * (n_in / block) * bb, buf);
    if (blob != nullptr) {
        fill_blob(blob, n_in, n_out, dtype);
        /* fill_blob pins d (and dmin) to powers of two, which multiply
         * exactly; random half scales in [0.5, 1) make those products
         * round as well (#17). */
        const size_t d_at = dtype == GEIST_DTYPE_Q6_K ? 208u : 0u;
        for (size_t b = 0; b < n_out * (n_in / block); b++) {
            uint8_t *blk  = blob + b * bb;
            blk[d_at]     = rng_u8();
            blk[d_at + 1] = (uint8_t) (0x38u | (rng_u8() & 3u));
            if (dtype == GEIST_DTYPE_Q4_K) {
                blk[2] = rng_u8();
                blk[3] = (uint8_t) (0x34u | (rng_u8() & 3u));
            }
        }
    }
    *t = (struct geist_tensor) {.buffer = *buf,
                                .dtype  = (uint16_t) dtype,
                                .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
                                .ndim   = 2,
                                .shape  = {(int64_t) n_out, (int64_t) n_in}};
    return blob;
}

static struct geist_tensor f32_row(struct geist_buffer *b, size_t n) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {1, (int64_t) n},
                                  .stride = {(int64_t) n, 1}};
}

/* The fused decode matvecs (#17) against the Metal ops they replace, bit
 * for bit: linear_t_triple against three linear_t, ffn_gate_up(_silu)
 * against two linear_t and gelu_tanh_mul / silu_mul. The separate ops are
 * held to cpu_scalar by run_case. */
static void run_fused_matvec_case(
        struct geist_backend *mt, int dt0, int dt1, int dt2, size_t n_in, size_t n_kv) {
    const struct geist_backend_vtbl  *v       = mt->desc->vtbl;
    const struct geist_backend_fused *f       = geist_backend_fused_tbl(mt);
    const int                         dts[3]  = {dt0, dt1, dt2};
    const size_t                      outs[3] = {n_in, n_kv, n_kv};
    struct geist_buffer              *bw[3] = {nullptr}, *by[3] = {nullptr}, *bz[3] = {nullptr};
    struct geist_tensor               tw[3], ty[3], tz[3];
    struct geist_weight               ww[3];
    struct geist_buffer              *bx  = nullptr;
    float                            *x   = dev_alloc(mt, n_in * sizeof(float), &bx);
    bool                              ok  = x != nullptr && f->linear_t_triple != nullptr;
    size_t                            tot = 0;
    for (size_t i = 0; i < 3 && ok; i++) {
        ok    = quant_weight(mt, dts[i], n_in, outs[i], &bw[i], &tw[i]) != nullptr &&
                dev_alloc(mt, outs[i] * sizeof(float), &by[i]) != nullptr &&
                dev_alloc(mt, outs[i] * sizeof(float), &bz[i]) != nullptr;
        ww[i] = (struct geist_weight) {
                .n_in = (int32_t) n_in, .n_out = (int32_t) outs[i], .dtype = (uint16_t) dts[i]};
        ty[i] = f32_row(by[i], outs[i]);
        tz[i] = f32_row(bz[i], outs[i]);
        tot += outs[i];
    }
    float *got = malloc(2 * tot * sizeof(float));
    check(ok && got != nullptr, "fused matvec buffers");
    if (ok && got != nullptr) {
        /* Not multiples of a power of two: the products and sums round,
         * so a kernel that orders or contracts them differently from the
         * matvec it replaces shows up here. */
        for (size_t i = 0; i < n_in; i++) {
            x[i] = ((float) rng_u8() - 127.5f) * 0.0372529f;
        }
        const struct geist_tensor tx = f32_row(bx, n_in);
        enum geist_status         s  = f->linear_t_triple(
                mt, &tx, &ww[0], &tw[0], &ww[1], &tw[1], &ww[2], &tw[2], 1, &ty[0], &ty[1], &ty[2]);
        for (size_t i = 0; i < 3 && s == GEIST_OK; i++) {
            s = f->linear_t(mt, &tx, &ww[i], &tw[i], 1, &tz[i]);
        }
        check(s == GEIST_OK, "linear_t_triple dispatch");
        size_t off = 0;
        for (size_t i = 0; i < 3; i++) {
            check(v->buffer_download(outs[i] * sizeof(float), (uint8_t *) (got + off), by[i]) ==
                                  GEIST_OK &&
                          v->buffer_download(outs[i] * sizeof(float),
                                             (uint8_t *) (got + tot + off),
                                             bz[i]) == GEIST_OK,
                  "linear_t_triple download");
            off += outs[i];
        }
        const bool same = memcmp(got, got + tot, tot * sizeof(float)) == 0;
        check(same, "linear_t_triple matches three linear_t bit for bit");
        printf("  linear_t_triple %d/%d/%d %zu->%zu,%zu: %s\n",
               dt0,
               dt1,
               dt2,
               n_in,
               n_in,
               n_kv,
               same ? "bit-identical OK" : "FAIL");

        /* gate/up: w1 and w2 share dtype and shape when dt1 == dt2. */
        for (int silu = 0; silu < 2 && dt1 == dt2; silu++) {
            struct geist_tensor tg = f32_row(by[0], n_kv), tu = f32_row(bz[0], n_kv);
            struct geist_tensor tf = f32_row(by[1], n_kv);
            s                      = silu ? f->ffn_gate_up_silu(mt, &tx, &tw[1], &tw[2], &tf)
                                          : f->ffn_gate_up(mt, &tx, &tw[1], &tw[2], &tf);
            if (s == GEIST_OK)
                s = f->linear_t(mt, &tx, &ww[1], &tw[1], 1, &tg);
            if (s == GEIST_OK)
                s = f->linear_t(mt, &tx, &ww[2], &tw[2], 1, &tu);
            if (s == GEIST_OK)
                s = silu ? f->silu_mul(mt, &tg, &tu, &tg) : f->gelu_tanh_mul(mt, &tg, &tu, &tg);
            check(s == GEIST_OK, "gate/up dispatch");
            check(v->buffer_download(n_kv * sizeof(float), (uint8_t *) got, by[1]) == GEIST_OK &&
                          v->buffer_download(n_kv * sizeof(float),
                                             (uint8_t *) (got + n_kv),
                                             by[0]) == GEIST_OK,
                  "gate/up download");
            /* The Q4_K GeGLU kernel (Gemma 4) predates #17 and sums through
             * dotb, which contracts the block scales differently from the
             * matvec: equal to rounding, not bit for bit. */
            const bool exact = silu || dt1 != GEIST_DTYPE_Q4_K;
            double     rel   = 0.0;
            for (size_t i = 0; i < n_kv; i++) {
                const double a = got[i], b = got[n_kv + i];
                const double r = fabs(a - b) / (fabs(b) > 1.0 ? fabs(b) : 1.0);
                rel            = r > rel ? r : rel;
            }
            const bool eq = exact ? memcmp(got, got + n_kv, n_kv * sizeof(float)) == 0 : rel < 1e-5;
            check(eq,
                  silu ? "ffn_gate_up_silu matches the separate ops bit for bit"
                       : "ffn_gate_up matches the separate ops");
            printf("  ffn_gate_up%s %d %zu->%zu: %s (max_rel %.1e)\n",
                   silu ? "_silu" : "",
                   dt1,
                   n_in,
                   n_kv,
                   eq ? (exact ? "bit-identical OK" : "OK") : "FAIL",
                   rel);
        }
    }
    free(got);
    for (size_t i = 0; i < 3; i++) {
        struct geist_buffer *bufs[] = {bw[i], by[i], bz[i]};
        for (size_t j = 0; j < 3; j++) {
            if (bufs[j] != nullptr)
                v->buffer_destroy(mt, bufs[j]);
        }
    }
    if (bx != nullptr)
        v->buffer_destroy(mt, bx);
}

static struct geist_tensor dense3(struct geist_buffer *b, int dt, size_t d0, size_t d1, size_t d2) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = (uint16_t) dt,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 3,
                                  .shape  = {(int64_t) d0, (int64_t) d1, (int64_t) d2},
                                  .stride = {(int64_t) (d1 * d2), (int64_t) d2, 1}};
}

/* attn_rope_append (#17) against the Metal ops it replaces, bit for bit:
 * per-head rmsnorm (with_norm), rope_apply over n_rot, scale_f32 and the
 * F16 or F32 KV append at row qp. */
static void run_rope_append_case(struct geist_backend *mt,
                                 size_t                rows,
                                 size_t                qh,
                                 size_t                kvh,
                                 size_t                hd,
                                 size_t                n_rot,
                                 size_t                qp,
                                 bool                  with_norm,
                                 bool                  f16) {
    const struct geist_backend_vtbl  *v  = mt->desc->vtbl;
    const struct geist_backend_fused *f  = geist_backend_fused_tbl(mt);
    const size_t                      nq = rows * qh * hd, nk = rows * kvh * hd;
    const size_t                      crow  = qp + rows;
    const size_t                      ncach = crow * kvh * hd;
    const size_t                      esz   = f16 ? sizeof(uint16_t) : sizeof(float);
    const float                       scale = 0.08838835f;
    struct geist_buffer *bq[2] = {nullptr}, *bk[2] = {nullptr}, *bv = nullptr, *bqw = nullptr,
                        *bkw = nullptr, *bc = nullptr, *bs = nullptr, *bkc[2] = {nullptr},
                        *bvc[2] = {nullptr};
    float *q                    = dev_alloc(mt, nq * sizeof(float), &bq[0]);
    float *k                    = dev_alloc(mt, nk * sizeof(float), &bk[0]);
    float *q2                   = dev_alloc(mt, nq * sizeof(float), &bq[1]);
    float *k2                   = dev_alloc(mt, nk * sizeof(float), &bk[1]);
    float *vv                   = dev_alloc(mt, nk * sizeof(float), &bv);
    float *qw                   = dev_alloc(mt, hd * sizeof(float), &bqw);
    float *kw                   = dev_alloc(mt, hd * sizeof(float), &bkw);
    float *co                   = dev_alloc(mt, rows * n_rot * sizeof(float), &bc);
    float *si                   = dev_alloc(mt, rows * n_rot * sizeof(float), &bs);
    bool   ok = q != nullptr && k != nullptr && q2 != nullptr && k2 != nullptr && vv != nullptr &&
                qw != nullptr && kw != nullptr && co != nullptr && si != nullptr &&
                f->attn_rope_append != nullptr;
    for (size_t i = 0; i < 2 && ok; i++) {
        uint8_t *a = dev_alloc(mt, ncach * esz, &bkc[i]);
        uint8_t *b = dev_alloc(mt, ncach * esz, &bvc[i]);
        ok         = a != nullptr && b != nullptr;
        if (ok) {
            memset(a, 0, ncach * esz);
            memset(b, 0, ncach * esz);
        }
    }
    const size_t tot = 2 * nq + 2 * nk + 4 * ncach * esz / sizeof(float) + 4;
    float       *got = malloc(tot * sizeof(float));
    check(ok && got != nullptr, "rope_append buffers");
    if (ok && got != nullptr) {
        for (size_t i = 0; i < nq; i++)
            q[i] = q2[i] = ((float) rng_u8() - 127.5f) * 0.0372529f;
        for (size_t i = 0; i < nk; i++) {
            k[i] = k2[i] = ((float) rng_u8() - 127.5f) * 0.0291f;
            vv[i]        = ((float) rng_u8() - 127.5f) * 0.0173f;
        }
        for (size_t i = 0; i < hd; i++) {
            qw[i] = 0.5f + (float) rng_u8() * 0.00731f;
            kw[i] = 0.25f + (float) rng_u8() * 0.00517f;
        }
        for (size_t i = 0; i < rows * n_rot; i++) {
            const float a = (float) (qp + i / n_rot) * 0.37f / (float) (1u + i % n_rot);
            co[i]         = cosf(a);
            si[i]         = sinf(a);
        }
        const int           cdt = f16 ? GEIST_DTYPE_F16 : GEIST_DTYPE_F32;
        struct geist_tensor tq  = dense3(bq[0], GEIST_DTYPE_F32, rows, qh, hd);
        struct geist_tensor tk  = dense3(bk[0], GEIST_DTYPE_F32, rows, kvh, hd);
        struct geist_tensor tv  = dense3(bv, GEIST_DTYPE_F32, rows, kvh, hd);
        struct geist_tensor tkc = dense3(bkc[0], cdt, crow, kvh, hd);
        struct geist_tensor tvc = dense3(bvc[0], cdt, crow, kvh, hd);
        struct geist_tensor tqw = {.buffer = bqw,
                                   .dtype  = GEIST_DTYPE_F32,
                                   .layout = GEIST_LAYOUT_DENSE,
                                   .ndim   = 1,
                                   .shape  = {(int64_t) hd},
                                   .stride = {1}};
        struct geist_tensor tkw = tqw;
        tkw.buffer              = bkw;
        struct geist_tensor tc  = {.buffer = bc,
                                   .dtype  = GEIST_DTYPE_F32,
                                   .layout = GEIST_LAYOUT_DENSE,
                                   .ndim   = 2,
                                   .shape  = {(int64_t) rows, (int64_t) n_rot},
                                   .stride = {(int64_t) n_rot, 1}};
        struct geist_tensor ts  = tc;
        ts.buffer               = bs;
        enum geist_status st    = f->attn_rope_append(mt,
                                                      &tq,
                                                      &tk,
                                                      &tv,
                                                      with_norm ? &tqw : nullptr,
                                                      with_norm ? &tkw : nullptr,
                                                      &tc,
                                                      &ts,
                                                      1e-6f,
                                                      scale,
                                                      qp,
                                                      &tkc,
                                                      &tvc);
        /* the separate ops on the copies */
        struct geist_tensor tq2 = dense3(bq[1], GEIST_DTYPE_F32, rows, qh, hd);
        struct geist_tensor tk2 = dense3(bk[1], GEIST_DTYPE_F32, rows, kvh, hd);
        struct geist_tensor qh2 = {.buffer = bq[1],
                                   .dtype  = GEIST_DTYPE_F32,
                                   .layout = GEIST_LAYOUT_DENSE,
                                   .ndim   = 2,
                                   .shape  = {(int64_t) (rows * qh), (int64_t) hd},
                                   .stride = {(int64_t) hd, 1}};
        struct geist_tensor kh2 = qh2;
        kh2.buffer              = bk[1];
        kh2.shape[0]            = (int64_t) (rows * kvh);
        if (st == GEIST_OK && with_norm)
            st = mt->desc->prims->rmsnorm(mt, &qh2, &tqw, 1e-6f, &qh2);
        if (st == GEIST_OK)
            st = mt->desc->prims->rope_apply(mt, &tq2, &tc, &ts);
        struct geist_tensor qrow = {.buffer = bq[1],
                                    .dtype  = GEIST_DTYPE_F32,
                                    .layout = GEIST_LAYOUT_DENSE,
                                    .ndim   = 2,
                                    .shape  = {(int64_t) rows, (int64_t) (qh * hd)},
                                    .stride = {(int64_t) (qh * hd), 1}};
        if (st == GEIST_OK)
            st = mt->desc->prims->scale_f32(mt, &qrow, scale, &qrow);
        if (st == GEIST_OK && with_norm)
            st = mt->desc->prims->rmsnorm(mt, &kh2, &tkw, 1e-6f, &kh2);
        if (st == GEIST_OK)
            st = mt->desc->prims->rope_apply(mt, &tk2, &tc, &ts);
        struct geist_tensor tkc2 = dense3(bkc[1], cdt, crow, kvh, hd);
        struct geist_tensor tvc2 = dense3(bvc[1], cdt, crow, kvh, hd);
        if (st == GEIST_OK && f16) {
            st = f->kv_append_f16(mt, &tk2, &tv, qp, &tkc2, &tvc2);
        } else if (st == GEIST_OK) {
            const size_t row_bytes = kvh * hd * sizeof(float);
            st = v->buffer_copy(bkc[1], qp * row_bytes, bk[1], 0, rows * row_bytes);
            if (st == GEIST_OK)
                st = v->buffer_copy(bvc[1], qp * row_bytes, bv, 0, rows * row_bytes);
        }
        check(st == GEIST_OK, "rope_append dispatch");
        float   *g   = got;
        size_t   bad = 0;
        uint8_t *c1 = malloc(ncach * esz), *c2 = malloc(ncach * esz);
        check(c1 != nullptr && c2 != nullptr &&
                      v->buffer_download(nq * sizeof(float), (uint8_t *) g, bq[0]) == GEIST_OK &&
                      v->buffer_download(nq * sizeof(float), (uint8_t *) (g + nq), bq[1]) ==
                              GEIST_OK,
              "rope_append q download");
        bad += memcmp(g, g + nq, nq * sizeof(float)) != 0;
        check(v->buffer_download(nk * sizeof(float), (uint8_t *) g, bk[0]) == GEIST_OK &&
                      v->buffer_download(nk * sizeof(float), (uint8_t *) (g + nk), bk[1]) ==
                              GEIST_OK,
              "rope_append k download");
        bad += 2 * (memcmp(g, g + nk, nk * sizeof(float)) != 0);
        for (size_t i = 0; i < 2 && c1 != nullptr && c2 != nullptr; i++) {
            struct geist_buffer *b1 = i == 0 ? bkc[0] : bvc[0], *b2 = i == 0 ? bkc[1] : bvc[1];
            check(v->buffer_download(ncach * esz, c1, b1) == GEIST_OK &&
                          v->buffer_download(ncach * esz, c2, b2) == GEIST_OK,
                  "rope_append cache download");
            bad += (4u << i) * (memcmp(c1, c2, ncach * esz) != 0);
        }
        free(c1);
        free(c2);
        check(bad == 0, "attn_rope_append matches the separate ops bit for bit");
        printf("  attn_rope_append rows %zu q %zu kv %zu hd %zu rot %zu qp %zu %s %s: %s (%zu)\n",
               rows,
               qh,
               kvh,
               hd,
               n_rot,
               qp,
               with_norm ? "norm" : "plain",
               f16 ? "f16" : "f32",
               bad == 0 ? "bit-identical OK" : "FAIL",
               bad);
    }
    free(got);
    struct geist_buffer *bufs[] = {
            bq[0], bq[1], bk[0], bk[1], bv, bqw, bkw, bc, bs, bkc[0], bkc[1], bvc[0], bvc[1]};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (bufs[i] != nullptr)
            v->buffer_destroy(mt, bufs[i]);
    }
}

#ifdef GEIST_BACKEND_METAL
/* #196: the block decode kernel merges its splits itself. Against the
 * separate combine pass (the test hook), bit for bit. */
static void run_dec_combine_case(struct geist_backend *mt,
                                 size_t                q_offset,
                                 size_t                qh,
                                 size_t                kvh,
                                 size_t                hd,
                                 size_t                window,
                                 bool                  f16) {
    const struct geist_backend_vtbl *v      = mt->desc->vtbl;
    struct metal_state              *st     = mt->state;
    const size_t                     k_rows = q_offset + 1, nq = qh * hd, kv = k_rows * kvh * hd;
    const size_t                     esz = f16 ? sizeof(uint16_t) : sizeof(float);
    struct geist_buffer             *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo[2] = {nullptr};
    float                           *q  = dev_alloc(mt, nq * sizeof(float), &bq);
    uint8_t                         *k  = dev_alloc(mt, kv * esz, &bk);
    uint8_t                         *vv = dev_alloc(mt, kv * esz, &bv);
    float                           *o0 = dev_alloc(mt, nq * sizeof(float), &bo[0]);
    float                           *o1 = dev_alloc(mt, nq * sizeof(float), &bo[1]);
    float                           *g  = malloc(2 * nq * sizeof(float));
    const bool ok = q != nullptr && k != nullptr && vv != nullptr && o0 != nullptr &&
                    o1 != nullptr && g != nullptr;
    check(ok, "decode combine buffers");
    if (ok) {
        const float qs = 1.0f / (24.0f * sqrtf((float) hd));
        for (size_t i = 0; i < nq; i++)
            q[i] = ((float) rng_u8() - 127.5f) * qs;
        for (size_t i = 0; i < kv; i++) {
            const float a = ((float) rng_u8() - 127.5f) / 48.0f;
            const float b = ((float) rng_u8() - 127.5f) / 48.0f;
            if (f16) {
                ((uint16_t *) k)[i]  = fp32_to_fp16_rne(a);
                ((uint16_t *) vv)[i] = fp32_to_fp16_rne(b);
            } else {
                ((float *) k)[i]  = a;
                ((float *) vv)[i] = b;
            }
        }
        const int           cdt = f16 ? GEIST_DTYPE_F16 : GEIST_DTYPE_F32;
        struct geist_tensor tq  = dense3(bq, GEIST_DTYPE_F32, 1, qh, hd);
        struct geist_tensor tk  = dense3(bk, cdt, k_rows, kvh, hd);
        struct geist_tensor tv  = dense3(bv, cdt, k_rows, kvh, hd);
        struct geist_tensor t0  = dense3(bo[0], GEIST_DTYPE_F32, 1, qh, hd);
        struct geist_tensor t1  = dense3(bo[1], GEIST_DTYPE_F32, 1, qh, hd);
        /* twice fused: the second call starts from the counts the first left */
        enum geist_status s = mt->desc->prims->attention(mt, &tq, &tk, &tv, q_offset, window, &t0);
        if (s == GEIST_OK)
            s = mt->desc->prims->attention(mt, &tq, &tk, &tv, q_offset, window, &t0);
        st->attn_dec_two_pass = true;
        if (s == GEIST_OK)
            s = mt->desc->prims->attention(mt, &tq, &tk, &tv, q_offset, window, &t1);
        st->attn_dec_two_pass = false;
        check(s == GEIST_OK &&
                      v->buffer_download(nq * sizeof(float), (uint8_t *) g, bo[0]) == GEIST_OK &&
                      v->buffer_download(nq * sizeof(float), (uint8_t *) (g + nq), bo[1]) ==
                              GEIST_OK,
              "decode combine dispatch");
        const bool same = memcmp(g, g + nq, nq * sizeof(float)) == 0;
        check(same, "fused decode combine matches the combine pass bit for bit");
        printf("  decode combine fused vs pass at %zu heads %zu/%zu hd %zu window %zu %s: %s\n",
               q_offset,
               qh,
               kvh,
               hd,
               window,
               f16 ? "f16" : "f32",
               same ? "bit-identical OK" : "FAIL");
    }
    free(g);
    struct geist_buffer *bufs[] = {bq, bk, bv, bo[0], bo[1]};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (bufs[i] != nullptr)
            v->buffer_destroy(mt, bufs[i]);
    }
}

/* #196: Gemma's attention prep (attn_qkv_prep) as one dispatch against the
 * q_norm_rope + kv_norm_append pair it replaces, bit for bit: q, k, v and
 * both caches. */
static void run_qkv_prep_case(struct geist_backend *mt,
                              size_t                rows,
                              size_t                qh,
                              size_t                kvh,
                              size_t                hd,
                              size_t                qp,
                              bool                  f16) {
    const struct geist_backend_vtbl  *v  = mt->desc->vtbl;
    const struct geist_backend_fused *f  = geist_backend_fused_tbl(mt);
    struct metal_state               *st = mt->state;
    const size_t                      nq = rows * qh * hd, nk = rows * kvh * hd;
    const size_t                      crow  = qp + rows;
    const size_t                      ncach = crow * kvh * hd;
    const size_t                      esz   = f16 ? sizeof(uint16_t) : sizeof(float);
    struct geist_buffer              *bq[2] = {nullptr}, *bk[2] = {nullptr}, *bv[2] = {nullptr},
                        *bw[3] = {nullptr}, *bc = nullptr, *bs = nullptr, *bkc[2] = {nullptr},
                        *bvc[2] = {nullptr};
    float *w[3];
    for (size_t i = 0; i < 3; i++)
        w[i] = dev_alloc(mt, hd * sizeof(float), &bw[i]);
    float *co      = dev_alloc(mt, rows * hd * sizeof(float), &bc);
    float *si      = dev_alloc(mt, rows * hd * sizeof(float), &bs);
    float *x[3][2] = {{nullptr}};
    bool   ok      = w[0] != nullptr && w[1] != nullptr && w[2] != nullptr && co != nullptr &&
                     si != nullptr && f->attn_qkv_prep != nullptr;
    for (size_t i = 0; i < 2 && ok; i++) {
        x[0][i]    = dev_alloc(mt, nq * sizeof(float), &bq[i]);
        x[1][i]    = dev_alloc(mt, nk * sizeof(float), &bk[i]);
        x[2][i]    = dev_alloc(mt, nk * sizeof(float), &bv[i]);
        uint8_t *a = dev_alloc(mt, ncach * esz, &bkc[i]);
        uint8_t *b = dev_alloc(mt, ncach * esz, &bvc[i]);
        ok = x[0][i] != nullptr && x[1][i] != nullptr && x[2][i] != nullptr && a != nullptr &&
             b != nullptr;
        if (ok) {
            memset(a, 0, ncach * esz);
            memset(b, 0, ncach * esz);
        }
    }
    uint8_t *c1 = malloc(ncach * esz + nq * sizeof(float));
    uint8_t *c2 = malloc(ncach * esz + nq * sizeof(float));
    check(ok && c1 != nullptr && c2 != nullptr, "qkv_prep buffers");
    if (ok && c1 != nullptr && c2 != nullptr) {
        for (size_t i = 0; i < nq; i++)
            x[0][0][i] = x[0][1][i] = ((float) rng_u8() - 127.5f) * 0.0372529f;
        for (size_t i = 0; i < nk; i++) {
            x[1][0][i] = x[1][1][i] = ((float) rng_u8() - 127.5f) * 0.0291f;
            x[2][0][i] = x[2][1][i] = ((float) rng_u8() - 127.5f) * 0.0173f;
        }
        for (size_t i = 0; i < hd; i++) {
            w[0][i] = 0.5f + (float) rng_u8() * 0.00731f;
            w[1][i] = 0.25f + (float) rng_u8() * 0.00517f;
            w[2][i] = 1.0f;
        }
        for (size_t i = 0; i < rows * hd; i++) {
            const float a = (float) (qp + i / hd) * 0.37f / (float) (1u + i % hd);
            co[i]         = cosf(a);
            si[i]         = sinf(a);
        }
        const int           cdt = f16 ? GEIST_DTYPE_F16 : GEIST_DTYPE_F32;
        struct geist_tensor tw  = {.buffer = bw[0],
                                   .dtype  = GEIST_DTYPE_F32,
                                   .layout = GEIST_LAYOUT_DENSE,
                                   .ndim   = 1,
                                   .shape  = {(int64_t) hd},
                                   .stride = {1}};
        struct geist_tensor tkw = tw, tvw = tw;
        tkw.buffer                 = bw[1];
        tvw.buffer                 = bw[2];
        struct geist_tensor tc     = {.buffer = bc,
                                      .dtype  = GEIST_DTYPE_F32,
                                      .layout = GEIST_LAYOUT_DENSE,
                                      .ndim   = 2,
                                      .shape  = {(int64_t) rows, (int64_t) hd},
                                      .stride = {(int64_t) hd, 1}};
        struct geist_tensor ts     = tc;
        ts.buffer                  = bs;
        enum geist_status s        = GEIST_OK;
        void *const       fused[2] = {st->qkv_norm_append_rows_pipeline,
                                      st->qkv_norm_append_rows_f16_pipeline};
        for (size_t i = 0; i < 2 && s == GEIST_OK; i++) {
            if (i == 1) { /* the separate kernels */
                st->qkv_norm_append_rows_pipeline     = nullptr;
                st->qkv_norm_append_rows_f16_pipeline = nullptr;
            }
            struct geist_tensor tq  = dense3(bq[i], GEIST_DTYPE_F32, rows, qh, hd);
            struct geist_tensor tk  = dense3(bk[i], GEIST_DTYPE_F32, rows, kvh, hd);
            struct geist_tensor tv  = dense3(bv[i], GEIST_DTYPE_F32, rows, kvh, hd);
            struct geist_tensor tkc = dense3(bkc[i], cdt, crow, kvh, hd);
            struct geist_tensor tvc = dense3(bvc[i], cdt, crow, kvh, hd);
            s                       = f->attn_qkv_prep(
                    mt, &tq, &tk, &tv, &tw, &tkw, &tvw, &tc, &ts, 1e-6f, qp, &tkc, &tvc);
        }
        st->qkv_norm_append_rows_pipeline     = fused[0];
        st->qkv_norm_append_rows_f16_pipeline = fused[1];
        check(s == GEIST_OK && fused[f16 ? 1 : 0] != nullptr, "qkv_prep dispatch");
        size_t                     bad         = 0;
        struct geist_buffer *const pairs[5][2] = {
                {bq[0], bq[1]}, {bk[0], bk[1]}, {bv[0], bv[1]}, {bkc[0], bkc[1]}, {bvc[0], bvc[1]}};
        const size_t bytes[5] = {nq * sizeof(float),
                                 nk * sizeof(float),
                                 nk * sizeof(float),
                                 ncach * esz,
                                 ncach * esz};
        for (size_t i = 0; i < 5; i++) {
            check(v->buffer_download(bytes[i], c1, pairs[i][0]) == GEIST_OK &&
                          v->buffer_download(bytes[i], c2, pairs[i][1]) == GEIST_OK,
                  "qkv_prep download");
            bad += (1u << i) * (memcmp(c1, c2, bytes[i]) != 0);
        }
        check(bad == 0, "attn_qkv_prep in one dispatch matches the two kernels bit for bit");
        printf("  attn_qkv_prep rows %zu q %zu kv %zu hd %zu qp %zu %s: %s (%zu)\n",
               rows,
               qh,
               kvh,
               hd,
               qp,
               f16 ? "f16" : "f32",
               bad == 0 ? "bit-identical OK" : "FAIL",
               bad);
    }
    free(c1);
    free(c2);
    struct geist_buffer *bufs[] = {bq[0],
                                   bq[1],
                                   bk[0],
                                   bk[1],
                                   bv[0],
                                   bv[1],
                                   bw[0],
                                   bw[1],
                                   bw[2],
                                   bc,
                                   bs,
                                   bkc[0],
                                   bkc[1],
                                   bvc[0],
                                   bvc[1]};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (bufs[i] != nullptr)
            v->buffer_destroy(mt, bufs[i]);
    }
}

static struct geist_tensor dense2(struct geist_buffer *b, size_t d0, size_t d1) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) d0, (int64_t) d1},
                                  .stride = {(int64_t) d1, 1}};
}

/* #196: the decode PLE gate and the F32 matvec (here its projection) with
 * a simdgroup per row, against ple_gate_f32 and matmul_f32, bit for bit.
 * hpl >= 1024 takes the 1024-thread layout of the projection. */
static void run_ple_rows_case(struct geist_backend *mt, size_t d_in, size_t hpl) {
    const struct geist_backend_vtbl  *v  = mt->desc->vtbl;
    const struct geist_backend_fused *f  = geist_backend_fused_tbl(mt);
    struct metal_state               *st = mt->state;
    struct geist_buffer              *bx = nullptr, *bgw = nullptr, *bple = nullptr, *bpw = nullptr,
                        *bres = nullptr, *bg[2] = {nullptr}, *bp[2] = {nullptr}, *by = nullptr;
    float *x   = dev_alloc(mt, d_in * sizeof(float), &bx);
    float *gw  = dev_alloc(mt, hpl * d_in * sizeof(float), &bgw);
    float *ple = dev_alloc(mt, hpl * sizeof(float), &bple);
    float *pw  = dev_alloc(mt, d_in * hpl * sizeof(float), &bpw);
    float *res = dev_alloc(mt, d_in * sizeof(float), &bres);
    float *yy  = dev_alloc(mt, d_in * sizeof(float), &by);
    bool ok = x != nullptr && gw != nullptr && ple != nullptr && pw != nullptr && res != nullptr &&
              yy != nullptr && f->ple_block != nullptr;
    for (size_t i = 0; i < 2 && ok; i++) {
        ok = dev_alloc(mt, hpl * sizeof(float), &bg[i]) != nullptr &&
             dev_alloc(mt, d_in * sizeof(float), &bp[i]) != nullptr;
    }
    float *g = malloc(2 * (hpl + d_in) * sizeof(float));
    check(ok && g != nullptr, "ple rows buffers");
    if (ok && g != nullptr) {
        for (size_t i = 0; i < d_in; i++) {
            x[i]   = ((float) rng_u8() - 127.5f) * 0.0213f;
            res[i] = ((float) rng_u8() - 127.5f) * 0.01f;
        }
        for (size_t i = 0; i < hpl * d_in; i++) {
            gw[i] = ((float) rng_u8() - 127.5f) * 0.00731f;
            pw[i] = ((float) rng_u8() - 127.5f) * 0.00517f;
        }
        for (size_t i = 0; i < hpl; i++)
            ple[i] = ((float) rng_u8() - 127.5f) * 0.0377f;
        struct geist_tensor tx = dense2(bx, 1, d_in), tgw = dense2(bgw, hpl, d_in);
        struct geist_tensor tpl = dense2(bple, 1, hpl), tpw = dense2(bpw, d_in, hpl);
        struct geist_tensor tr = dense2(bres, 1, d_in), ty = dense2(by, 1, d_in);
        void *const         pipes[2] = {st->f32_ple_gate_sg_pipeline, st->f32_matvec_rows_pipeline};
        enum geist_status   s        = GEIST_OK;
        for (size_t i = 0; i < 2 && s == GEIST_OK; i++) {
            if (i == 1) { /* the threadgroup-per-row kernels */
                st->f32_ple_gate_sg_pipeline = nullptr;
                st->f32_matvec_rows_pipeline = nullptr;
            }
            struct geist_tensor tg = dense2(bg[i], 1, hpl), tp = dense2(bp[i], 1, d_in);
            s = f->ple_block(mt, &tx, &tgw, &tpl, &tpw, &tr, nullptr, 1e-6f, &tg, &tp, &ty);
        }
        st->f32_ple_gate_sg_pipeline = pipes[0];
        st->f32_matvec_rows_pipeline = pipes[1];
        check(s == GEIST_OK && pipes[0] != nullptr && pipes[1] != nullptr &&
                      v->buffer_download(hpl * sizeof(float), (uint8_t *) g, bg[0]) == GEIST_OK &&
                      v->buffer_download(hpl * sizeof(float), (uint8_t *) (g + hpl), bg[1]) ==
                              GEIST_OK &&
                      v->buffer_download(d_in * sizeof(float), (uint8_t *) (g + 2 * hpl), bp[0]) ==
                              GEIST_OK &&
                      v->buffer_download(d_in * sizeof(float),
                                         (uint8_t *) (g + 2 * hpl + d_in),
                                         bp[1]) == GEIST_OK,
              "ple rows dispatch");
        /* Not the same bits: the old kernels' sums are not reproduced in
         * order under fast math. Rounding only, relative to the row's size. */
        const double gate_err = geist_test_max_abs(hpl, g, g + hpl);
        const double proj_err = geist_test_max_abs(d_in, g + 2 * hpl, g + 2 * hpl + d_in);
        const bool   pass     = gate_err < 2e-4 && proj_err < 2e-3;
        check(pass, "PLE gate / F32 matvec rows match the threadgroup kernels");
        printf("  ple_block rows d %zu hpl %zu: gate max_abs %.2e, proj max_abs %.2e %s\n",
               d_in,
               hpl,
               gate_err,
               proj_err,
               pass ? "OK" : "FAIL");
    }
    free(g);
    struct geist_buffer *bufs[] = {bx, bgw, bple, bpw, bres, by, bg[0], bg[1], bp[0], bp[1]};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (bufs[i] != nullptr)
            v->buffer_destroy(mt, bufs[i]);
    }
}
#endif

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
    const double err = geist_test_max_abs(ROWS * QOUT, got, expected);
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
    /* A ragged last row tile on the interior fast kernels (#35): n_out fills
     * whole 64-wide tiles, the rows do not. */
    static const size_t ragged[] = {13, 33, 47, 100};
    for (size_t r = 0; r < sizeof ragged / sizeof ragged[0]; r++) {
        run_case(mt, ref, GEIST_DTYPE_Q4_0, "Q4_0", 512, 384, ragged[r]);
        run_case(mt, ref, GEIST_DTYPE_Q5_K, "Q5_K", 512, 384, ragged[r]);
        run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 384, ragged[r]);
        run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 384, ragged[r]);
        run_case(mt, ref, GEIST_DTYPE_F32, "F32", 512, 384, ragged[r]);
        run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 512, 384, ragged[r]);
        run_case(mt, ref, GEIST_DTYPE_Q4_1, "Q4_1", 512, 384, ragged[r]);
        run_case(mt, ref, GEIST_DTYPE_Q8_0, "Q8_0", 512, 384, ragged[r]);
        run_case(mt, ref, GEIST_DTYPE_IQ4_XS, "IQ4XS", 512, 384, ragged[r]);
    }
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
    /* PQ2_0 (ternary Bonsai): n4 GEMV at m=1, the bounded GEMM for m>=2,
     * the interior _fast GEMM for full 32x64 tiles. */
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 512, 383, 4);
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 512, 384, 32);
    /* Whole 64-row tiles of the PQ2_0 fast kernel, one and two of them. */
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 512, 384, 64);
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 1024, 384, 128);
    /* n_out >= 6144 at m = 1 takes matvec_pq2_n8 (8 rows per simdgroup):
     * 6150 leaves a partial 16-row tile, 6144 an exact one. */
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2n8", 512, 6150, 1);
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2n8", 512, 6144, 1);
    /* n_out < 4 at m = 1 is the one shape the n4 GEMV refuses. The
     * gemm_only formats have no naive kernel under it, so the GEMM must
     * take it. */
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2tiny", 512, 3, 1);
    /* TQ2_0 (#559): GEMV with an odd row tail, GEMM bounded and full
     * tiles, and n_out < 4, which the GEMM takes at rows == 1. */
    run_case(mt, ref, GEIST_DTYPE_TQ2_0, "TQ2_0", 768, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_TQ2_0, "TQ2_0", 768, 383, 4);
    run_case(mt, ref, GEIST_DTYPE_TQ2_0, "TQ2_0", 768, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_TQ2_0, "TQ2_0", 768, 384, 32);
    run_case(mt, ref, GEIST_DTYPE_TQ2_0, "TQ2tiny", 768, 3, 1);
    /* F16 / BF16 (#564): decode, a bounded GEMM tile and an odd n_out. */
    run_case(mt, ref, GEIST_DTYPE_F16, "F16", 520, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_F16, "F16", 520, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_BF16, "BF16", 520, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_BF16, "BF16", 520, 383, 33);
    /* I2_S (#560): the same shapes; its scale is the tensor's tail. */
    run_case(mt, ref, GEIST_DTYPE_I2_S, "I2_S", 768, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_I2_S, "I2_S", 768, 383, 4);
    run_case(mt, ref, GEIST_DTYPE_I2_S, "I2_S", 768, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_I2_S, "I2_S", 768, 384, 32);
    run_case(mt, ref, GEIST_DTYPE_I2_S, "I2Stiny", 768, 3, 1);
    run_case(mt, ref, GEIST_DTYPE_IQ4_XS, "IQ4XStiny", 512, 1, 1);
    run_case(mt, ref, GEIST_DTYPE_Q3_K, "Q3Ktiny", 512, 2, 1);
    run_case(mt, ref, GEIST_DTYPE_IQ4_NL, "IQ4NLtiny", 512, 3, 1);
    run_case(mt, ref, GEIST_DTYPE_IQ3_S, "IQ3Stiny", 512, 2, 1);
    /* Full 64-output x 32-row tiles: matmul_q6k_mm_sg_fast (#17); m=96
     * splits into a 64-row head and a 32-row tail, both full tiles. */
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 384, 32);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 384, 96);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 8);
    run_case(mt, ref, GEIST_DTYPE_F32, "F32", 256, 130, 1);
    run_case(mt, ref, GEIST_DTYPE_F32, "F32", 256, 130, 8);
    /* The prefill kernels m = 1 and 8 do not reach. Appended last so the
     * cases above keep their random data.
     * Q4_K m=33 x n_out=383: n_out off the 64-column tile and an unaligned
     * y row stride keep the simdgroup GEMM out: the m16 kernel, last tile
     * partial.
     * Q4_K m=32 x n_out=384: full 32x64 tiles -> the simdgroup GEMM's _fast
     * variant, which a Q4_K prefill chunk with rows % 32 == 0 and
     * n_out % 64 == 0 takes by default.
     * Q6_K m=64 x n_out=384: rows % 64 == 0, n_out % 32 == 0 -> the Q6_K
     * simdgroup GEMM's _fast variant, likewise the default. */
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 33);
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 384, 32);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 384, 64);
    /* #578: a ragged batch splits into whole tiles on the _fast GEMM and a
     * tail at the y/x offset of its first row; up to 9 rows go row by row
     * through the matvec kernel, more through the bounded GEMM.
     * Q4_K m=34: 32 fast + 2 matvec. m=65: 64 fast + 1 matvec. m=45: 32
     * fast + 13 bounded. m=5: matvec only. m=12 x n_out=383: an unaligned
     * y row stride keeps the bounded GEMM's float4 epilogue out (m16).
     * Q6_K m=70: 64 fast + 6 matvec. m=80: 64 fast + 16 bounded. m=3:
     * matvec only. */
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 384, 34);
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 384, 65);
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 384, 45);
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 384, 5);
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 12);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 384, 70);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 384, 80);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 384, 3);
    run_silu_case(mt);
    /* Llama / Qwen3 attention prep (#17): Qwen3 0.6B decode (16/8 heads
     * of 128, q/k norms), Llama 3.2 3B (24/8 of 128, none), a prefill
     * chunk, a partial rotation and the F32 cache. */
    run_rope_append_case(mt, 1, 16, 8, 128, 128, 37, true, true);
    run_rope_append_case(mt, 1, 24, 8, 128, 128, 200, false, true);
    run_rope_append_case(mt, 7, 4, 2, 64, 64, 0, true, true);
    run_rope_append_case(mt, 3, 4, 1, 256, 64, 11, true, false);
    run_rope_append_case(mt, 1, 2, 2, 512, 512, 5, false, false);
    /* Decode q/k/v and gate/up fusion (#17): Gemma 4 / Llama Q4_K_M
     * (v Q6_K), all Q4_K, all Q8_0 (Qwen3, SmolLM2); n_kv off the
     * threadgroup tiles of 4 and 8 rows. */
    run_fused_matvec_case(mt, GEIST_DTYPE_Q4_K, GEIST_DTYPE_Q4_K, GEIST_DTYPE_Q6_K, 512, 258);
    run_fused_matvec_case(mt, GEIST_DTYPE_Q4_K, GEIST_DTYPE_Q4_K, GEIST_DTYPE_Q4_K, 1024, 254);
    run_fused_matvec_case(mt, GEIST_DTYPE_Q8_0, GEIST_DTYPE_Q8_0, GEIST_DTYPE_Q8_0, 1024, 260);
    run_fused_matvec_case(mt, GEIST_DTYPE_Q8_0, GEIST_DTYPE_Q8_0, GEIST_DTYPE_Q8_0, 512, 131);
    run_embedding_case(mt, GEIST_DTYPE_Q4_0, "Q4_0");
    run_embedding_case(mt, GEIST_DTYPE_Q8_0, "Q8_0");
    run_embedding_case(mt, GEIST_DTYPE_PQ2_0, "PQ2_0");
    run_qwen35_attention_ops(mt);
    run_norm_case(mt, ref, 1, 1536);
    run_norm_case(mt, ref, 1, 1024);
    run_norm_case(mt, ref, 1, 2048);
    run_norm_case(mt, ref, 1, 3072);
    run_norm_case(mt, ref, 1, 1000);
    run_norm_case(mt, ref, 1, 5000);
    run_norm_case(mt, ref, 1, 256);
    run_norm_case(mt, ref, 35, 256);
    run_norm_case(mt, ref, 3, 1536);
    run_norm_case(mt, ref, 7, 896);
    for (size_t i = 0; i < 2; i++) {
        const bool with_w = i == 0;
        run_add_norm_case(mt, 1, 1536, with_w);
        run_add_norm_case(mt, 1, 3072, with_w);
        run_add_norm_case(mt, 1, 1000, with_w);
        run_add_norm_case(mt, 1, 5000, with_w);
        run_add_norm_case(mt, 1, 256, with_w);
        run_add_norm_case(mt, 35, 256, with_w);
        run_add_norm_case(mt, 3, 1536, with_w);
        run_add_norm_case(mt, 7, 896, with_w);
    }
    run_f32kv_attention_case(mt, 1, 200, 8, 2, 256, 0);
    run_f32kv_attention_case(mt, 1, 500, 8, 2, 256, 128);
    run_f32kv_attention_case(mt, 1, 99, 4, 1, 512, 0);
    run_f32kv_attention_case(mt, 64, 0, 8, 2, 128, 0);
    run_f32kv_attention_case(mt, 13, 300, 4, 1, 256, 0);
    run_f32kv_attention_case(mt, 40, 100, 8, 2, 256, 64);
    run_f32kv_attention_case(mt, 24, 50, 2, 1, 512, 0);
    run_f32kv_attention_case(mt, 37, 600, 4, 1, 512, 256);
    run_flash_ref_case(mt, ref, 320, 700, 24, 4, 256, 0);
    run_flash_ref_case(mt, ref, 37, 600, 8, 1, 256, 512);
    run_flash_ref_case(mt, ref, 100, 0, 8, 1, 256, 32);
    run_flash_ref_case(mt, ref, 45, 300, 8, 1, 512, 0);
    run_flash_ref_case(mt, ref, 64, 0, 8, 2, 128, 0);
    run_flash_ref_case(mt, ref, 13, 51, 4, 2, 64, 0);
    run_flash_ref_case(mt, ref, 9, 20, 4, 1, 256, 0);
    run_flash_ref_case(mt, ref, 40, 100, 4, 2, 96, 0);
    run_flash_ref_case(mt, ref, 17, 450, 2, 1, 512, 128);
    run_flash_ref_case(mt, ref, 30, 70, 4, 1, 384, 0);
    run_flash_ref_case(mt, ref, 2, 98, 4, 4, 32, 0);
    run_flash_ref_case(mt, ref, 25, 15, 4, 2, 256, 0);
    /* Decode (#17): 32-key blocks for head_dim 64/128/256/512, windows
     * narrower than a block and wider than the cache, a window that starts
     * mid-block, more blocks than splits, and head_dim 96 (per-key kernel). */
    run_flash_ref_case(mt, ref, 1, 511, 8, 1, 256, 512);
    run_flash_ref_case(mt, ref, 1, 700, 8, 1, 256, 512);
    run_flash_ref_case(mt, ref, 1, 1000, 8, 1, 512, 0);
    run_flash_ref_case(mt, ref, 1, 40, 8, 2, 128, 0);
    run_flash_ref_case(mt, ref, 1, 63, 4, 4, 64, 0);
    run_flash_ref_case(mt, ref, 1, 2100, 24, 4, 256, 0);
    run_flash_ref_case(mt, ref, 1, 100, 4, 1, 96, 0);
    run_flash_ref_case(mt, ref, 1, 77, 2, 1, 512, 16);
    run_flash_ref_case(mt, ref, 1, 31, 8, 2, 128, 0);
#ifdef GEIST_BACKEND_METAL
    run_staged_upload(mt);
    /* #196 */
    run_dec_combine_case(mt, 511, 8, 1, 256, 512, true);
    run_dec_combine_case(mt, 1000, 8, 1, 512, 0, true);
    run_dec_combine_case(mt, 127, 8, 1, 256, 0, false);
    run_dec_combine_case(mt, 2100, 24, 4, 256, 0, true);
    run_dec_combine_case(mt, 40, 8, 2, 128, 0, false);
    run_dec_combine_case(mt, 31, 4, 4, 64, 0, true);
    run_qkv_prep_case(mt, 1, 8, 1, 256, 130, true);
    run_qkv_prep_case(mt, 1, 8, 1, 512, 77, false);
    run_qkv_prep_case(mt, 5, 8, 1, 256, 12, true);
    run_qkv_prep_case(mt, 3, 4, 2, 128, 0, false);
    run_ple_rows_case(mt, 1536, 256);
    run_ple_rows_case(mt, 384, 1024);
    run_ple_rows_case(mt, 2048, 250);
#endif

    /* 2..8 rows take matmul_pq2_m8 (#105): one and many K blocks per
     * simdgroup, a ragged last output tile. Last, so the shared random
     * stream of the cases above stays as it was. */
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2m8", 512, 383, 2);
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2m8", 1536, 64, 7);
    run_case(mt, ref, GEIST_DTYPE_PQ2_0, "PQ2m8", 384, 1000, 8);

    geist_backend_destroy(mt);
    geist_backend_destroy(ref);
    if (g_fail == 0) {
        printf("test_backend_metal_parity_unit: all checks passed\n");
    }
    return g_fail == 0 ? 0 : 1;
}
