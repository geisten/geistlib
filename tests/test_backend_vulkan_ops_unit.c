/* Vulkan elementwise / layout ops of the qwen35 family, each compared
 * with the host reference formula:
 *
 *   - rope_apply with a rotated prefix narrower than head_dim (partial
 *     rotary, 64 of 256) and with full rotation (regression);
 *   - silu, fused silu_mul (SwiGLU epilogue, in place), sigmoid_mul (the
 *     attention output gate, in place) and attn_qgate_split.
 *
 * Every operand lives in a KV_CACHE-role buffer, which this backend keeps in
 * VRAM without a host mapping: an op that silently fell back to a host loop
 * would fail with "bad inputs" instead of passing by accident.
 *
 * SKIPs (exit 77) when no Vulkan runtime/device is present. */
#include "test_helpers.h"

#include "hadamard.h"
#include "quant.h"

#include "gemma4_kernels.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define check(ok, what) (g_fail |= geist_expect((ok), (what)))

static struct geist_backend *g_be;

static struct geist_buffer *dev_buf(const float *src, size_t n) {
    const struct geist_backend_vtbl *v = g_be->desc->vtbl;
    struct geist_buffer             *b = nullptr;
    if (v->buffer_create(g_be, n * sizeof(float), GEIST_BUFFER_KV_CACHE, GEIST_MEMORY_AUTO, &b) !=
                GEIST_OK ||
        (src != nullptr &&
         v->buffer_upload(b, n * sizeof(float), (const uint8_t *) src) != GEIST_OK)) {
        return nullptr;
    }
    return b;
}

static bool download(struct geist_buffer *b, float *dst, size_t n) {
    return g_be->desc->vtbl->buffer_download(n * sizeof(float), (uint8_t *) dst, b) == GEIST_OK;
}

/* relu(x)^2 on the device, in place (VRAM-only buffer: a host fallback fails). */
static void test_relu2(size_t n) {
    float *x   = geist_test_fill(n, 0.37f, 0.3f, 3.0f, 0.0f);
    float *ref = malloc(n * sizeof(float));
    float *got = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        const float v = x[i] > 0.0f ? x[i] : 0.0f;
        ref[i]        = v * v;
    }
    struct geist_buffer *bx = dev_buf(x, n);
    check(bx != nullptr, "relu2 buffer");
    struct geist_tensor tx = geist_test_tensor_f32(bx, 1, (int64_t) n, 0, 0);
    check(g_be->desc->prims->relu_squared(g_be, &tx, &tx) == GEIST_OK, "relu2 dispatch");
    check(download(bx, got, n), "relu2 download");
    const double e = geist_test_max_abs(n, got, ref);
    printf("  relu2 n=%-6zu max_abs %.2e\n", n, e);
    check(e < 1e-6, "relu2");
    free(x);
    free(ref);
    free(got);
}

/* BitNet activation fake-quant: the device rows against the host formula. The
 * float ops differ in the last ulp between compilers, which can flip a value
 * sitting exactly on a rounding boundary by one quantum — so the bound is one
 * quantum (absmax/127) and flips must be rare. */
static void test_act_quant(size_t rows, size_t n) {
    float *x   = geist_test_fill(rows * n, 0.29f, 0.3f, 2.0f, 0.0f);
    float *ref = malloc(rows * n * sizeof(float));
    float *got = malloc(rows * n * sizeof(float));
    memcpy(ref, x, rows * n * sizeof(float));
    double quantum = 0.0;
    for (size_t t = 0; t < rows; t++) {
        float *row = ref + t * n, mx = 1e-5f;
        for (size_t j = 0; j < n; j++) {
            mx = fabsf(row[j]) > mx ? fabsf(row[j]) : mx;
        }
        const float sc = 127.0f / mx, inv = 1.0f / sc;
        quantum = mx / 127.0 > quantum ? mx / 127.0 : quantum;
        for (size_t j = 0; j < n; j++) {
            float q = row[j] * sc;
            q       = q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q);
            int qi  = (int) (q < 0.0f ? q - 0.5f : q + 0.5f);
            row[j]  = (float) qi * inv;
        }
    }
    struct geist_buffer *bx = dev_buf(x, rows * n);
    check(bx != nullptr, "act_quant buffer");
    struct geist_tensor tx = geist_test_tensor_f32(bx, 2, (int64_t) rows, (int64_t) n, 0);
    check(g_be->desc->fused->bitnet_act_quant != nullptr, "act_quant entry");
    check(g_be->desc->fused->bitnet_act_quant(g_be, &tx) == GEIST_OK, "act_quant dispatch");
    check(download(bx, got, rows * n), "act_quant download");
    size_t flips = 0;
    for (size_t i = 0; i < rows * n; i++) {
        flips += got[i] != ref[i];
    }
    const double e = geist_test_max_abs(rows * n, got, ref);
    printf("  act_quant %zux%zu max_abs %.2e (quantum %.2e) flips %zu\n",
           rows,
           n,
           e,
           quantum,
           flips);
    check(e <= quantum * 1.01, "act_quant within one quantum");
    check(flips * 200 <= rows * n, "act_quant flips rare");
    free(x);
    free(ref);
    free(got);
}

/* fused->embedding_lookup_scaled on a PQ2_0 table (repacked struct-of-arrays
 * on the device) against the host row dequant. The op never resolves a
 * table itself (#468): an unresolved one is declined, and the table runs on
 * the device once resolve_weight has copied it, as a tied lm_head is at
 * load. */
static uint8_t *test_embed_pq2_0(size_t vocab, size_t d, int32_t token) {
    const size_t nb = d / PQ2_0_BLOCK_ELEMS, row_bytes = nb * PQ2_0_BLOCK_BYTES;
    uint8_t     *blob = malloc(vocab * row_bytes);
    for (size_t i = 0; i < vocab * nb; i++) {
        uint8_t *blk = blob + i * PQ2_0_BLOCK_BYTES;
        for (size_t j = 0; j < PQ2_0_BLOCK_BYTES; j++) {
            blk[j] = (uint8_t) ((i * 131u + j * 17u + (i >> 3)) * 2654435761u >> 24);
        }
        blk[0] = 0x00; /* d = fp16(0.75) */
        blk[1] = 0x3A;
    }
    float *ref = malloc(d * sizeof(float)), *got = malloc(d * sizeof(float));
    dequant_pq2_0_row(d, blob + (size_t) token * row_bytes, ref);
    const float scale = 2.0f;
    for (size_t k = 0; k < d; k++) {
        ref[k] *= scale;
    }
    struct geist_buffer *tb = nullptr;
    check(g_be->desc->vtbl->buffer_create_aliased(
                  g_be, blob, vocab * row_bytes, GEIST_BUFFER_WEIGHT, &tb) == GEIST_OK,
          "embed table buffer");
    struct geist_tensor table = geist_test_tensor_f32(tb, 2, (int64_t) vocab, (int64_t) d, 0);
    table.dtype               = GEIST_DTYPE_PQ2_0;
    table.layout              = GEIST_LAYOUT_BLOCK_QUANTIZED;
    struct geist_buffer *bo   = dev_buf(nullptr, d);
    check(bo != nullptr, "embed out buffer");
    struct geist_tensor out = geist_test_tensor_f32(bo, 1, (int64_t) d, 0, 0);
    check(g_be->desc->fused->embedding_lookup_scaled != nullptr, "embed entry");
    check(g_be->desc->fused->embedding_lookup_scaled(g_be, &table, token, scale, &out) != GEIST_OK,
          "embed declines an unresolved table");
    struct geist_weight w = {.raw        = blob,
                             .raw_nbytes = vocab * row_bytes,
                             .n_in       = (int32_t) d,
                             .n_out      = (int32_t) vocab,
                             .dtype      = (uint16_t) GEIST_DTYPE_PQ2_0};
    check(g_be->desc->vtbl->resolve_weight(g_be, &w) == GEIST_OK, "embed table resolve");
    check(g_be->desc->fused->embedding_lookup_scaled(g_be, &table, token, scale, &out) == GEIST_OK,
          "embed dispatch");
    check(download(bo, got, d), "embed download");
    const double e = geist_test_max_abs(d, got, ref);
    printf("  embed pq2_0 %zux%zu tok %d max_abs %.2e\n", vocab, d, (int) token, e);
    check(e < 1e-6, "embed pq2_0");
    free(ref);
    free(got);
    /* The device copy stays registered under the table's address, so the
     * table outlives the remaining cases: a reused address would look
     * resolved. */
    return blob;
}

/* fused->hadamard_rotate on VRAM-only buffers against the host implementation
 * (geist_hadamard_rows): same butterfly order and the same float scale, so the
 * result must be bit-identical. */
static void test_hadamard(size_t      rows,
                          size_t      width,
                          size_t      block,
                          bool        signed_,
                          bool        inverse,
                          size_t      hd,
                          size_t      nk,
                          size_t      rep,
                          bool        in_place,
                          const char *name) {
    const size_t n   = rows * width;
    float       *x   = geist_test_fill(n, 0.23f, 0.3f, 1.5f, 0.0f);
    float       *sg  = malloc(width * sizeof(float));
    float       *ref = malloc(n * sizeof(float));
    float       *got = malloc(n * sizeof(float));
    for (size_t i = 0; i < width; i++) {
        sg[i] = ((i * 2654435761u) >> 13) & 1u ? -1.0f : 1.0f;
    }
    check(geist_hadamard_rows(
                  rows, width, block, hd, nk, rep, inverse, x, signed_ ? sg : nullptr, ref) ==
                  GEIST_OK,
          "hadamard host reference");
    struct geist_buffer *bx = dev_buf(x, n), *bs = signed_ ? dev_buf(sg, width) : nullptr,
                        *by = in_place ? bx : dev_buf(nullptr, n);
    check(bx && by && (!signed_ || bs), "hadamard buffers");
    struct geist_tensor tx = geist_test_tensor_f32(bx, 2, (int64_t) rows, (int64_t) width, 0),
                        ty = geist_test_tensor_f32(by, 2, (int64_t) rows, (int64_t) width, 0),
                        ts = geist_test_tensor_f32(bs, 1, (int64_t) width, 0, 0);
    const struct geist_hadamard_args args = {.x        = &tx,
                                             .signs    = signed_ ? &ts : nullptr,
                                             .y        = &ty,
                                             .block    = block,
                                             .perm_hd  = hd,
                                             .perm_nk  = nk,
                                             .perm_rep = rep,
                                             .inverse  = inverse};
    check(g_be->desc->fused->hadamard_rotate != nullptr, "hadamard entry");
    check(g_be->desc->fused->hadamard_rotate(g_be, &args) == GEIST_OK, "hadamard dispatch");
    check(download(by, got, n), "hadamard download");
    const double e = geist_test_max_abs(n, got, ref);
    printf("  hadamard %-26s max_abs %.2e\n", name, e);
    check(e == 0.0, name);
    free(x);
    free(sg);
    free(ref);
    free(got);
}

static void test_rope(size_t seq, size_t heads, size_t hd, size_t rot, const char *name) {
    float *x   = geist_test_fill(seq * heads * hd, 0.21f, 0.3f, 1.0f, 0.0f);
    float *cs  = geist_test_fill(seq * rot, 0.13f, 0.3f, 1.0f, 0.0f);
    float *sn  = geist_test_fill(seq * rot, 0.17f, 0.3f, 1.0f, 0.0f);
    float *ref = malloc(seq * heads * hd * sizeof(float));
    float *got = malloc(seq * heads * hd * sizeof(float));
    memcpy(ref, x, seq * heads * hd * sizeof(float));
    rope_apply(seq, heads, hd, rot, ref, cs, sn);

    struct geist_buffer *bx = dev_buf(x, seq * heads * hd), *bc = dev_buf(cs, seq * rot),
                        *bs = dev_buf(sn, seq * rot);
    check(bx && bc && bs, "rope buffers");
    struct geist_tensor tx =
            geist_test_tensor_f32(bx, 3, (int64_t) seq, (int64_t) heads, (int64_t) hd);
    struct geist_tensor     tc = geist_test_tensor_f32(bc, 2, (int64_t) seq, (int64_t) rot, 0);
    struct geist_tensor     ts = geist_test_tensor_f32(bs, 2, (int64_t) seq, (int64_t) rot, 0);
    const enum geist_status s  = g_be->desc->prims->rope_apply(g_be, &tx, &tc, &ts);
    check(s == GEIST_OK, "rope_apply dispatch");
    check(download(bx, got, seq * heads * hd), "rope download");
    const double e = geist_test_max_abs(seq * heads * hd, got, ref);
    printf("  rope %-22s max_abs %.2e\n", name, e);
    check(e < 2e-6, name);
    free(x);
    free(cs);
    free(sn);
    free(ref);
    free(got);
}

static void test_elementwise(size_t rows, size_t cols) {
    const size_t n = rows * cols;
    float       *a = geist_test_fill(n, 0.19f, 0.3f, 6.0f, 0.0f),
          *b = geist_test_fill(n, 0.11f, 0.3f, 3.0f, 0.0f), *got = malloc(n * sizeof(float));
    float *ref = malloc(n * sizeof(float));

    struct geist_buffer *ba = dev_buf(a, n), *bb = dev_buf(b, n), *by = dev_buf(nullptr, n);
    check(ba && bb && by, "ew buffers");
    struct geist_tensor ta = geist_test_tensor_f32(ba, 2, (int64_t) rows, (int64_t) cols, 0),
                        tb = geist_test_tensor_f32(bb, 2, (int64_t) rows, (int64_t) cols, 0),
                        ty = geist_test_tensor_f32(by, 2, (int64_t) rows, (int64_t) cols, 0);

    /* silu (out of place) */
    for (size_t i = 0; i < n; i++) {
        ref[i] = geist_test_silu(a[i]);
    }
    check(g_be->desc->prims->silu(g_be, &ta, &ty) == GEIST_OK, "silu dispatch");
    check(download(by, got, n), "silu download");
    double e = geist_test_max_abs(n, got, ref);
    printf("  silu        %zux%zu  max_abs %.2e\n", rows, cols, e);
    check(e < 2e-6, "silu");

    /* silu_mul, in place over a */
    const struct geist_backend_fused *f = geist_backend_fused_tbl(g_be);
    check(f->silu_mul != nullptr, "silu_mul present");
    for (size_t i = 0; i < n; i++) {
        ref[i] = geist_test_silu(a[i]) * b[i];
    }
    check(f->silu_mul(g_be, &ta, &tb, &ta) == GEIST_OK, "silu_mul dispatch");
    check(download(ba, got, n), "silu_mul download");
    e = geist_test_max_abs(n, got, ref);
    printf("  silu_mul    %zux%zu  max_abs %.2e (in place)\n", rows, cols, e);
    check(e < 2e-5, "silu_mul");

    /* sigmoid_mul, in place over a (re-upload the original a) */
    check(g_be->desc->vtbl->buffer_upload(ba, n * sizeof(float), (const uint8_t *) a) == GEIST_OK,
          "re-upload");
    check(f->sigmoid_mul != nullptr, "sigmoid_mul present");
    for (size_t i = 0; i < n; i++) {
        ref[i] = a[i] * (1.0f / (1.0f + expf(-b[i])));
    }
    check(f->sigmoid_mul(g_be, &ta, &tb, &ta) == GEIST_OK, "sigmoid_mul dispatch");
    check(download(ba, got, n), "sigmoid_mul download");
    e = geist_test_max_abs(n, got, ref);
    printf("  sigmoid_mul %zux%zu  max_abs %.2e (in place)\n", rows, cols, e);
    check(e < 2e-6, "sigmoid_mul");

    g_be->desc->vtbl->buffer_destroy(g_be, ba);
    g_be->desc->vtbl->buffer_destroy(g_be, bb);
    g_be->desc->vtbl->buffer_destroy(g_be, by);
    free(a);
    free(b);
    free(got);
    free(ref);
}

/* rmsnorm and rmsnorm_add on VRAM-only buffers, against a double-precision
 * host reference. Both shaders reduce with one shared slot per subgroup, so
 * the 8-lane llvmpipe (and Intel's 8/16) must not overrun an array sized
 * for 32-lane subgroups. */
static void test_rmsnorm(size_t rows, size_t feat) {
    const size_t n  = rows * feat;
    float       *x  = geist_test_fill(n, 0.13f, 0.3f, 4.0f, 0.0f),
          *r        = geist_test_fill(n, 0.07f, 0.3f, 2.0f, 0.0f),
          *w        = geist_test_fill(feat, 0.05f, 0.3f, 1.5f, 0.0f);
    float      *got = malloc(n * sizeof(float)), *ref = malloc(n * sizeof(float));
    const float eps = 1e-6f;
    for (size_t row = 0; row < rows; row++) {
        double ss = 0.0;
        for (size_t i = 0; i < feat; i++) {
            ss += (double) x[row * feat + i] * (double) x[row * feat + i];
        }
        const double inv = 1.0 / sqrt(ss / (double) feat + (double) eps);
        for (size_t i = 0; i < feat; i++) {
            ref[row * feat + i] = (float) ((double) x[row * feat + i] * inv * (double) w[i]);
        }
    }
    struct geist_buffer *bx = dev_buf(x, n), *br = dev_buf(r, n), *bw = dev_buf(w, feat),
                        *by = dev_buf(nullptr, n);
    check(bx && br && bw && by, "rmsnorm buffers");
    struct geist_tensor tx = geist_test_tensor_f32(bx, 2, (int64_t) rows, (int64_t) feat, 0),
                        tr = geist_test_tensor_f32(br, 2, (int64_t) rows, (int64_t) feat, 0),
                        tw = geist_test_tensor_f32(bw, 1, (int64_t) feat, 0, 0),
                        ty = geist_test_tensor_f32(by, 2, (int64_t) rows, (int64_t) feat, 0);

    check(g_be->desc->prims->rmsnorm(g_be, &tx, &tw, eps, &ty) == GEIST_OK, "rmsnorm dispatch");
    check(download(by, got, n), "rmsnorm download");
    double e = geist_test_max_abs(n, got, ref);
    printf("  rmsnorm     %zux%zu  max_abs %.2e\n", rows, feat, e);
    check(e < 1e-5, "rmsnorm");

    const struct geist_backend_fused *f = geist_backend_fused_tbl(g_be);
    check(f->rmsnorm_add != nullptr, "rmsnorm_add present");
    for (size_t i = 0; i < n; i++) {
        ref[i] += r[i];
    }
    check(f->rmsnorm_add(g_be, &tr, &tx, &tw, eps, &ty) == GEIST_OK, "rmsnorm_add dispatch");
    check(download(by, got, n), "rmsnorm_add download");
    e = geist_test_max_abs(n, got, ref);
    printf("  rmsnorm_add %zux%zu  max_abs %.2e\n", rows, feat, e);
    check(e < 1e-5, "rmsnorm_add");

    g_be->desc->vtbl->buffer_destroy(g_be, bx);
    g_be->desc->vtbl->buffer_destroy(g_be, br);
    g_be->desc->vtbl->buffer_destroy(g_be, bw);
    g_be->desc->vtbl->buffer_destroy(g_be, by);
    free(x);
    free(r);
    free(w);
    free(got);
    free(ref);
}

static void test_qgate(size_t rows, size_t heads, size_t hd) {
    const size_t q_out = heads * hd;
    float       *joint = geist_test_fill(rows * 2 * q_out, 0.07f, 0.3f, 2.0f, 0.0f);
    float       *q_ref = malloc(rows * q_out * sizeof(float)),
          *g_ref       = malloc(rows * q_out * sizeof(float));
    float *q_got       = malloc(rows * q_out * sizeof(float)),
          *g_got       = malloc(rows * q_out * sizeof(float));
    for (size_t t = 0; t < rows; t++) {
        for (size_t h = 0; h < heads; h++) {
            const float *src = joint + t * 2 * q_out + h * 2 * hd;
            memcpy(q_ref + t * q_out + h * hd, src, hd * sizeof(float));
            memcpy(g_ref + t * q_out + h * hd, src + hd, hd * sizeof(float));
        }
    }
    struct geist_buffer *bj = dev_buf(joint, rows * 2 * q_out),
                        *bq = dev_buf(nullptr, rows * q_out), *bg = dev_buf(nullptr, rows * q_out);
    check(bj && bq && bg, "qgate buffers");
    struct geist_tensor tj = geist_test_tensor_f32(bj, 2, (int64_t) rows, (int64_t) (2 * q_out), 0);
    struct geist_tensor tq = geist_test_tensor_f32(bq, 2, (int64_t) rows, (int64_t) q_out, 0);
    struct geist_tensor tg = geist_test_tensor_f32(bg, 2, (int64_t) rows, (int64_t) q_out, 0);
    const struct geist_backend_fused *f = geist_backend_fused_tbl(g_be);
    check(f->attn_qgate_split != nullptr, "attn_qgate_split present");
    check(f->attn_qgate_split(g_be, &tj, heads, hd, &tq, &tg) == GEIST_OK, "qgate dispatch");
    check(download(bq, q_got, rows * q_out) && download(bg, g_got, rows * q_out), "qgate download");
    const double eq = geist_test_max_abs(rows * q_out, q_got, q_ref),
                 eg = geist_test_max_abs(rows * q_out, g_got, g_ref);
    printf("  qgate_split %zu rows x %zu heads x %zu  q %.2e gate %.2e\n", rows, heads, hd, eq, eg);
    check(eq == 0.0 && eg == 0.0, "qgate_split is an exact copy");
    g_be->desc->vtbl->buffer_destroy(g_be, bj);
    g_be->desc->vtbl->buffer_destroy(g_be, bq);
    g_be->desc->vtbl->buffer_destroy(g_be, bg);
    free(joint);
    free(q_ref);
    free(g_ref);
    free(q_got);
    free(g_got);
}

/* prims->attention against the host reference attention_mqa_causal_kv, F32
 * KV. Shapes mirror the transformer families that use the generic (non
 * gemma) attention path. */
static void test_attention(size_t      n_q,
                           size_t      n_kv,
                           size_t      q_off,
                           size_t      qh,
                           size_t      kvh,
                           size_t      hd,
                           size_t      sliding,
                           const char *name) {
    float *q   = geist_test_fill(n_q * qh * hd, 0.031f, 0.3f, 0.6f, 0.0f),
          *k   = geist_test_fill(n_kv * kvh * hd, 0.023f, 0.3f, 0.6f, 0.0f);
    float *v   = geist_test_fill(n_kv * kvh * hd, 0.017f, 0.3f, 1.0f, 0.0f);
    float *ref = malloc(n_q * qh * hd * sizeof(float)),
          *got = malloc(n_q * qh * hd * sizeof(float));
    attention_mqa_causal_kv(n_q, n_kv, q_off, qh, kvh, hd, sliding, q, k, v, ref);
    struct geist_buffer *bq = dev_buf(q, n_q * qh * hd), *bk = dev_buf(k, n_kv * kvh * hd),
                        *bv = dev_buf(v, n_kv * kvh * hd), *bo = dev_buf(nullptr, n_q * qh * hd);
    check(bq && bk && bv && bo, "attention buffers");
    struct geist_tensor tq =
            geist_test_tensor_f32(bq, 3, (int64_t) n_q, (int64_t) qh, (int64_t) hd);
    struct geist_tensor tk =
            geist_test_tensor_f32(bk, 3, (int64_t) n_kv, (int64_t) kvh, (int64_t) hd);
    struct geist_tensor tv =
            geist_test_tensor_f32(bv, 3, (int64_t) n_kv, (int64_t) kvh, (int64_t) hd);
    struct geist_tensor to =
            geist_test_tensor_f32(bo, 3, (int64_t) n_q, (int64_t) qh, (int64_t) hd);
    check(g_be->desc->prims->attention(g_be, &tq, &tk, &tv, q_off, sliding, &to) == GEIST_OK,
          "attention dispatch");
    check(download(bo, got, n_q * qh * hd), "attention download");
    const double e = geist_test_max_abs(n_q * qh * hd, got, ref);
    printf("  attention %-28s max_abs %.2e\n", name, e);
    check(e < 1e-4, name);
    g_be->desc->vtbl->buffer_destroy(g_be, bq);
    g_be->desc->vtbl->buffer_destroy(g_be, bk);
    g_be->desc->vtbl->buffer_destroy(g_be, bv);
    g_be->desc->vtbl->buffer_destroy(g_be, bo);
    free(q);
    free(k);
    free(v);
    free(ref);
    free(got);
}

/* The same with F16 K/V, the KV-cache dtype the GPU path uses, so prefill
 * (n_q > 1) takes the tensor-core kernel for head_dim 128 / 256 / 512 and
 * sliding windows (#475). The reference sees the f16-rounded Q/K/V; the
 * kernel also rounds P to f16, hence the looser bound. */
static void test_attention_f16(size_t      n_q,
                               size_t      n_kv,
                               size_t      q_off,
                               size_t      qh,
                               size_t      kvh,
                               size_t      hd,
                               size_t      sliding,
                               float       q_amp,
                               const char *name) {
    const size_t nq = n_q * qh * hd, nk = n_kv * kvh * hd;
    float       *q = geist_test_fill(nq, 0.031f, 0.3f, q_amp, 0.0f),
          *k       = geist_test_fill(nk, 0.023f, 0.3f, 0.6f, 0.0f);
    float    *v    = geist_test_fill(nk, 0.017f, 0.3f, 1.0f, 0.0f);
    float    *ref = malloc(nq * sizeof(float)), *got = malloc(nq * sizeof(float));
    _Float16 *k16 = malloc(nk * sizeof(_Float16)), *v16 = malloc(nk * sizeof(_Float16));
    for (size_t i = 0; i < nq; i++) {
        q[i] = (float) (_Float16) q[i];
    }
    for (size_t i = 0; i < nk; i++) {
        k16[i] = (_Float16) k[i];
        v16[i] = (_Float16) v[i];
        k[i]   = (float) k16[i];
        v[i]   = (float) v16[i];
    }
    attention_mqa_causal_kv(n_q, n_kv, q_off, qh, kvh, hd, sliding, q, k, v, ref);
    const struct geist_backend_vtbl *vt = g_be->desc->vtbl;
    struct geist_buffer             *bq = dev_buf(q, nq), *bo = dev_buf(nullptr, nq), *bk = nullptr,
                        *bv = nullptr;
    check(bq && bo &&
                  vt->buffer_create(g_be, nk * 2, GEIST_BUFFER_KV_CACHE, GEIST_MEMORY_AUTO, &bk) ==
                          GEIST_OK &&
                  vt->buffer_create(g_be, nk * 2, GEIST_BUFFER_KV_CACHE, GEIST_MEMORY_AUTO, &bv) ==
                          GEIST_OK &&
                  vt->buffer_upload(bk, nk * 2, (const uint8_t *) k16) == GEIST_OK &&
                  vt->buffer_upload(bv, nk * 2, (const uint8_t *) v16) == GEIST_OK,
          "attention f16 buffers");
    struct geist_tensor tq =
            geist_test_tensor_f32(bq, 3, (int64_t) n_q, (int64_t) qh, (int64_t) hd);
    struct geist_tensor tk =
            geist_test_tensor_f32(bk, 3, (int64_t) n_kv, (int64_t) kvh, (int64_t) hd);
    struct geist_tensor tv =
            geist_test_tensor_f32(bv, 3, (int64_t) n_kv, (int64_t) kvh, (int64_t) hd);
    struct geist_tensor to =
            geist_test_tensor_f32(bo, 3, (int64_t) n_q, (int64_t) qh, (int64_t) hd);
    tk.dtype = GEIST_DTYPE_F16;
    tv.dtype = GEIST_DTYPE_F16;
    check(g_be->desc->prims->attention(g_be, &tq, &tk, &tv, q_off, sliding, &to) == GEIST_OK,
          "attention f16 dispatch");
    check(download(bo, got, nq), "attention f16 download");
    const double e = geist_test_max_abs(nq, got, ref);
    printf("  attention f16 %-24s max_abs %.2e\n", name, e);
    check(e < 1e-3, name);
    vt->buffer_destroy(g_be, bq);
    vt->buffer_destroy(g_be, bk);
    vt->buffer_destroy(g_be, bv);
    vt->buffer_destroy(g_be, bo);
    free(q);
    free(k);
    free(v);
    free(k16);
    free(v16);
    free(ref);
    free(got);
}

int main(void) {
    enum geist_status s = geist_backend_create("vulkan", nullptr, nullptr, &g_be);
    if (s == GEIST_E_NOT_FOUND || s == GEIST_E_UNSUPPORTED) {
        GEIST_SKIP("Vulkan backend unavailable");
    }
    if (s != GEIST_OK || g_be == nullptr) {
        fprintf(stderr, "FAIL: Vulkan create: %s\n", geist_last_create_error());
        return GEIST_TEST_FAIL;
    }
    test_rope(5, 3, 256, 64, "partial 64/256");
    test_rope(4, 2, 128, 128, "full 128/128");
    test_rope(3, 2, 96, 32, "partial 32/96");
    uint8_t *tables[3] = {test_embed_pq2_0(37, 384, 0),
                          test_embed_pq2_0(37, 384, 36),
                          test_embed_pq2_0(64, 1024, 17)};
    test_hadamard(3, 5120, 1024, true, false, 0, 0, 0, false, "fwd 5120/1024 signs");
    test_hadamard(3, 5120, 1024, true, true, 0, 0, 0, false, "inv 5120/1024 signs");
    test_hadamard(2, 6144, 1024, true, false, 128, 16, 3, false, "fwd 6144 grouped-v perm");
    test_hadamard(4, 17408, 1024, true, false, 0, 0, 0, true, "fwd 17408/1024 in place");
    test_hadamard(5, 384, 128, false, false, 0, 0, 0, true, "fwd 384/128 no signs");
    test_hadamard(2, 1536, 512, true, true, 0, 0, 0, false, "inv 1536/512 (inexact scale)");
    test_relu2(1000);
    test_relu2(4096 * 3);
    test_act_quant(5, 1536);
    test_act_quant(1, 4096);
    test_act_quant(9, 100);
    test_elementwise(7, 129);
    test_rmsnorm(3, 2048);
    test_rmsnorm(1, 1000);
    test_elementwise(1, 4096);
    test_qgate(5, 4, 64);
    test_qgate(1, 24, 256);
    test_attention(22, 22, 0, 8, 2, 256, 0, "qwen35 prefill 22 (8/2x256)");
    test_attention(1, 23, 22, 8, 2, 256, 0, "qwen35 decode kv=23");
    test_attention(9, 40, 31, 8, 2, 256, 0, "chunked prefill q_off=31");
    test_attention(16, 16, 0, 16, 16, 128, 0, "MHA 16x128");
    test_attention(1, 300, 299, 8, 2, 256, 0, "decode kv=300 (flash path)");
    test_attention_f16(37, 37, 0, 8, 2, 256, 0, 0.6f, "prefill 37 (8/2x256)");
    test_attention_f16(37, 37, 0, 16, 8, 128, 0, 0.6f, "prefill 37 (16/8x128)");
    test_attention_f16(37, 37, 0, 8, 1, 512, 0, 0.6f, "prefill 37 (8/1x512)");
    test_attention_f16(70, 100, 30, 8, 2, 256, 32, 0.6f, "chunk q_off=30, window 32");
    test_attention_f16(70, 100, 30, 8, 1, 512, 20, 0.6f, "x512 q_off=30, window 20");
    test_attention_f16(40, 40, 0, 16, 8, 128, 7, 0.6f, "x128 window 7 (< tile)");
    test_attention_f16(1, 50, 49, 8, 2, 256, 16, 0.6f, "decode kv=50, window 16");
    /* longer prefills: several 64-key steps, the n_kv tail, GQA 1/2/8 */
    test_attention_f16(200, 200, 0, 8, 8, 128, 0, 0.6f, "prefill 200 MHA x128");
    test_attention_f16(200, 200, 0, 8, 4, 256, 0, 0.6f, "prefill 200 GQA2 x256");
    test_attention_f16(200, 200, 0, 8, 1, 512, 0, 0.6f, "prefill 200 GQA8 x512");
    test_attention_f16(600, 700, 100, 8, 1, 256, 0, 0.6f, "600 q_off=100 GQA8 x256");
    test_attention_f16(600, 700, 100, 16, 8, 128, 0, 0.6f, "600 q_off=100 GQA2 x128");
    test_attention_f16(600, 650, 50, 4, 1, 512, 128, 0.6f, "600 q_off=50 x512 win 128");
    test_attention_f16(300, 300, 0, 8, 2, 256, 5, 0.6f, "300 x256 window 5 (< tile)");
    test_attention_f16(300, 340, 40, 8, 1, 256, 100, 0.6f, "300 q_off=40 window 100");
    /* larger logits: the running row max moves by more than the online
     * softmax's rescale threshold between key steps */
    test_attention_f16(300, 300, 0, 8, 2, 256, 0, 4.0f, "300 x256 large logits");
    test_attention_f16(260, 300, 40, 8, 1, 512, 0, 3.0f, "260 x512 large logits");
    test_attention_f16(260, 260, 0, 16, 8, 128, 64, 6.0f, "260 x128 win 64 large");
    /* decode (n_q == 1, kv > 192): the flash-decoding partial/combine path
     * (#475). One workgroup serves up to 4 q-heads of a GQA group (groups of
     * 1, 2, 3 and 4 in one batch; 8 and 16 in batches of 4); key spans start
     * at the window's first key and walk 16-, 32- or 64-key tiles, so the
     * cases put the range ends on and off tile and span boundaries, with
     * head_dim 64 / 96 / 128 / 256 / 512. */
    test_attention_f16(1, 300, 299, 8, 8, 64, 0, 0.6f, "decode kv=300 MHA x64");
    test_attention_f16(1, 512, 511, 16, 8, 128, 0, 0.6f, "decode kv=512 GQA2 x128");
    test_attention_f16(1, 513, 512, 16, 8, 128, 0, 0.6f, "decode kv=513 GQA2 x128");
    test_attention_f16(1, 1025, 1024, 24, 8, 96, 0, 0.6f, "decode kv=1025 GQA3 x96");
    test_attention_f16(1, 2049, 2048, 8, 1, 512, 0, 0.6f, "decode kv=2049 GQA8 x512");
    test_attention_f16(1, 2049, 2048, 8, 1, 256, 512, 0.6f, "decode 2049 GQA8 x256 w512");
    test_attention_f16(1, 700, 699, 8, 1, 256, 100, 3.0f, "decode 700 GQA8 w100 large");
    test_attention_f16(1, 1000, 999, 16, 1, 128, 0, 4.0f, "decode 1000 GQA16 large");
    test_attention_f16(1, 333, 332, 16, 4, 256, 17, 0.6f, "decode 333 GQA4 w17");
    test_attention_f16(1, 400, 250, 8, 2, 128, 0, 0.6f, "decode kv=400 q_off=250");
    test_attention_f16(1, 3072, 3071, 16, 8, 128, 0, 6.0f, "decode 3072 GQA2 large");
    geist_backend_destroy(g_be);
    for (size_t i = 0; i < sizeof tables / sizeof tables[0]; i++) {
        free(tables[i]);
    }
    if (g_fail == 0) {
        printf("PASS: Vulkan qwen35 ops (partial and interleaved rope, hadamard, PQ2_0 embed, "
               "relu2, "
               "act_quant, rmsnorm, rmsnorm_add, silu, "
               "silu_mul, "
               "sigmoid_mul, "
               "qgate_split)\n");
    }
    return g_fail == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
