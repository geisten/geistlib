/*
 * test_attention_int8_unit — attention_int8_via_buffers and
 * attention_int4_via_buffers against a double-precision reference, on the
 * one-head loop and the GQA-grouped passes, split or not.
 *
 * The kernels quantize each query head to int8 (scale amax/127, lrintf),
 * dot it exactly against the int8 (or unpacked int4) K rows, and softmax
 * and accumulate V in fp32, online over blocks of the context (decode
 * passes split into context chunks merged per head). The reference repeats
 * the same Q quantization and integer dots, then does the softmax and the
 * V sum in double: the only differences left are fp32 rounding, so the
 * bound is tight (1e-5 of the output scale; 1e-4 for the steep trends,
 * whose scores in the hundreds round coarser).
 *
 * Shapes are chosen to take each path under the current policy: passes of
 * 2, 3 and 4 heads (head_dim 128, head_dim 64 over a short and a long
 * context, MQA), decode split into chunks and decode too short to split,
 * and the one-head loop (3 heads per KV head, plain MHA) over one block and
 * over several; decode (n_q = 1) and prefill chunks; with and without a
 * sliding window; a context shorter than one block. Some shapes make the
 * scores trend along the context (every K row one pattern, K scales
 * rising), so the running max grows in every block for some heads (the
 * rescale path, and chunks with different maxima to merge) and never after
 * the first for others. Steep trends put scores hundreds apart, so that
 * most exponents, rescales and chunk merges fall below ATTN_EXP_FLOOR
 * (gemma4_kernels.h), where the kernels clamp the exponent or drop the
 * term. INT4 shapes pack K/V values in [-7, 7], which the 4-bit cache holds
 * exactly, and run the INT4 kernel's one-head loop.
 * Every output must be written (poisoned first, compared by bit pattern —
 * -ffast-math build); a 1-thread run must give the same bits as the
 * default team; an INT8 run without scratch (decode unsplit) must match
 * the reference too.
 *
 * The INT8 shapes run a second time through fused->attention_kv_int8 on
 * every backend in the build whose probe binds it (cpu_x86 must, and
 * cpu_neon where built with FEAT_DotProd): the
 * backend's own kernel on buffers and tensor views, held to the same
 * reference bound, the same poison and thread-count checks, and to the
 * portable loop's output (its decomposed twin) within that bound, and a
 * last query one position past the cache must be refused
 * (GEIST_E_INVALID_ARG) with the output untouched. Two shapes put -128 in
 * K, which the cache never writes but an int8 is.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "src/archs/transformer/forward/internal.h"
#include "src/archs/transformer/forward.h"

#include <geist.h>
#include <geist_backend.h>

#include "int4_kv.h"

#include "heap.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

static uint32_t g_rng = 0xB5297A4Du;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

struct shape {
    size_t n_q, n_q_heads, n_kv_heads, head_dim, n_kv, window;
    float  trend; /* 0, or one K row per KV head, K scales rising to ~0.1 * trend */
    bool   int4;  /* packed 4-bit K/V through attention_int4_via_buffers */
    bool   full;  /* K over the whole int8 range, -128 included */
};

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};

/* Whether backend `name` has to bind attention_kv_int8 in this build. */
static bool must_bind(const char *name) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (strcmp(name, "cpu_neon") == 0) {
        return true;
    }
#endif
    return strcmp(name, "cpu_x86") == 0;
}
constexpr size_t             N_BACKENDS = sizeof BACKENDS / sizeof BACKENDS[0];
static struct geist_backend *g_be[N_BACKENDS];

/* The kernel under test on this shape's cache. */
static void run(const struct shape *sh,
                size_t              n_scr,
                const float        *q,
                const int8_t       *k,
                const uint8_t      *k4,
                const float        *ks,
                const int8_t       *v,
                const uint8_t      *v4,
                const float        *vs,
                float              *out,
                float              *scratch) {
    const size_t q_offset = sh->n_kv - sh->n_q;
    if (sh->int4) {
        attention_int4_via_buffers(sh->n_q,
                                   sh->n_q_heads,
                                   sh->head_dim,
                                   sh->n_kv,
                                   sh->n_kv_heads,
                                   q_offset,
                                   sh->window,
                                   q,
                                   k4,
                                   ks,
                                   v4,
                                   vs,
                                   out);
    } else {
        attention_int8_via_buffers(sh->n_q,
                                   sh->n_q_heads,
                                   sh->head_dim,
                                   sh->n_kv,
                                   sh->n_kv_heads,
                                   n_scr,
                                   q_offset,
                                   sh->window,
                                   q,
                                   k,
                                   ks,
                                   v,
                                   vs,
                                   out,
                                   scratch);
    }
}

/* out_ref[t, h, :] in double, the kernel's int8 Q and integer dots. */
static void reference(const struct shape *sh,
                      const float        *q,
                      const int8_t       *k,
                      const float        *ks,
                      const int8_t       *v,
                      const float        *vs,
                      double             *out_ref,
                      double             *scores) {
    const size_t G = sh->n_q_heads / sh->n_kv_heads, hd = sh->head_dim;
    for (size_t t = 0; t < sh->n_q; t++) {
        const size_t q_pos = sh->n_kv - sh->n_q + t;
        const size_t s_lo = (sh->window > 0 && q_pos + 1 > sh->window) ? q_pos + 1 - sh->window : 0;
        for (size_t h = 0; h < sh->n_q_heads; h++) {
            const size_t kv_h = h / G;
            const float *qv   = q + (t * sh->n_q_heads + h) * hd;
            float        amax = 0.0f;
            for (size_t i = 0; i < hd; i++) {
                amax = fabsf(qv[i]) > amax ? fabsf(qv[i]) : amax;
            }
            float sq = amax / 127.0f;
            if (sq == 0.0f) {
                sq = 1.0f;
            }
            const float inv_q = 1.0f / sq;
            int8_t      qq[TRANSFORMER_HEAD_DIM_MAX];
            for (size_t i = 0; i < hd; i++) {
                qq[i] = (int8_t) lrintf(qv[i] * inv_q);
            }
            double mx = -INFINITY;
            for (size_t s = s_lo; s <= q_pos; s++) {
                const int8_t *kr  = k + (s * sh->n_kv_heads + kv_h) * hd;
                int64_t       dot = 0;
                for (size_t i = 0; i < hd; i++) {
                    dot += (int64_t) qq[i] * kr[i];
                }
                scores[s] = (double) ((float) dot * sq * ks[s * sh->n_kv_heads + kv_h]);
                mx        = scores[s] > mx ? scores[s] : mx;
            }
            double sum = 0.0;
            for (size_t s = s_lo; s <= q_pos; s++) {
                scores[s] = exp(scores[s] - mx);
                sum += scores[s];
            }
            double *o = out_ref + (t * sh->n_q_heads + h) * hd;
            for (size_t i = 0; i < hd; i++) {
                o[i] = 0.0;
            }
            for (size_t s = s_lo; s <= q_pos; s++) {
                const int8_t *vr = v + (s * sh->n_kv_heads + kv_h) * hd;
                const double  w  = scores[s] / sum * vs[s * sh->n_kv_heads + kv_h];
                for (size_t i = 0; i < hd; i++) {
                    o[i] += w * vr[i];
                }
            }
        }
    }
}

/* A DENSE view of `bytes` uploaded from `src`, or nullptr. */
static struct geist_buffer *upload(struct geist_backend *be, size_t bytes, const void *src) {
    const struct geist_backend_vtbl *vt = be->desc->vtbl;
    struct geist_buffer             *b  = nullptr;
    if (vt->buffer_create(be, bytes, GEIST_BUFFER_SCRATCH, 0, &b) != GEIST_OK || b == nullptr) {
        return nullptr;
    }
    void *p = vt->buffer_map(b);
    if (p == nullptr) {
        vt->buffer_destroy(be, b);
        return nullptr;
    }
    memcpy(p, src, bytes);
    vt->buffer_unmap(b);
    return b;
}

static struct geist_tensor
tensor(struct geist_buffer *b, enum geist_dtype dt, int ndim, size_t s0, size_t s1, size_t s2) {
    struct geist_tensor t = {.buffer = b, .dtype = dt, .layout = GEIST_LAYOUT_DENSE, .ndim = ndim};
    t.shape[0]            = (int64_t) s0;
    t.shape[1]            = (int64_t) s1;
    t.shape[2]            = (int64_t) s2;
    t.stride[ndim - 1]    = 1;
    t.stride[0]           = ndim == 3 ? (int64_t) (s1 * s2) : (int64_t) s1;
    if (ndim == 3) {
        t.stride[1] = (int64_t) s2;
    }
    return t;
}

/* fused->attention_kv_int8 of backend `bi` on this shape's cache, against
 * the reference and against the portable loop's `twin`. */
static int check_backend(size_t              bi,
                         const struct shape *sh,
                         const float        *q,
                         const int8_t       *k,
                         const float        *ks,
                         const int8_t       *v,
                         const float        *vs,
                         const double       *ref,
                         const float        *twin,
                         double              tol) {
    struct geist_backend             *be    = g_be[bi];
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    const struct geist_fusion_query   pq    = {.op         = GEIST_FUSED_ATTN_KV_INT8,
                                               .m          = sh->n_q,
                                               .head_dim   = sh->head_dim,
                                               .n_q_heads  = sh->n_q_heads,
                                               .n_kv_heads = sh->n_kv_heads};
    const bool bound = fused->attention_kv_int8 != nullptr && fused->supported != nullptr &&
                       fused->supported(be, &pq);
    if (!bound) {
        if (must_bind(BACKENDS[bi])) {
            fprintf(stderr,
                    "FAIL: %s does not bind attention_kv_int8 (hd=%zu)\n",
                    BACKENDS[bi],
                    sh->head_dim);
            return 1;
        }
        return 0;
    }
    const struct geist_backend_vtbl *vt       = be->desc->vtbl;
    const size_t                     kv_elems = sh->n_kv * sh->n_kv_heads * sh->head_dim;
    const size_t                     q_elems  = sh->n_q * sh->n_q_heads * sh->head_dim;
    const size_t                     n_sc     = sh->n_kv * sh->n_kv_heads;
    float                           *poison   = heap_alloc_array_aligned(float, q_elems);
    float                           *out      = heap_alloc_array_aligned(float, q_elems);
    float                           *out1     = heap_alloc_array_aligned(float, q_elems);
    struct geist_buffer *bq = nullptr, *bk = nullptr, *bks = nullptr, *bv = nullptr, *bvs = nullptr,
                        *bo = nullptr;
    int fails               = 0;
    if (poison == nullptr || out == nullptr || out1 == nullptr) {
        fprintf(stderr, "ERROR: allocation failed\n");
        fails = 1;
        goto done;
    }
    const float POISON = -7.5e30f;
    for (size_t i = 0; i < q_elems; i++) {
        poison[i] = POISON;
    }
    bq  = upload(be, q_elems * sizeof *q, q);
    bk  = upload(be, kv_elems, k);
    bks = upload(be, n_sc * sizeof *ks, ks);
    bv  = upload(be, kv_elems, v);
    bvs = upload(be, n_sc * sizeof *vs, vs);
    bo  = upload(be, q_elems * sizeof *out, poison);
    if (bq == nullptr || bk == nullptr || bks == nullptr || bv == nullptr || bvs == nullptr ||
        bo == nullptr) {
        fprintf(stderr, "ERROR: %s: buffers\n", BACKENDS[bi]);
        fails = 1;
        goto done;
    }
    const struct geist_tensor tq =
            tensor(bq, GEIST_DTYPE_F32, 3, sh->n_q, sh->n_q_heads, sh->head_dim);
    const struct geist_tensor tk =
            tensor(bk, GEIST_DTYPE_I8, 3, sh->n_kv, sh->n_kv_heads, sh->head_dim);
    const struct geist_tensor tv =
            tensor(bv, GEIST_DTYPE_I8, 3, sh->n_kv, sh->n_kv_heads, sh->head_dim);
    const struct geist_tensor tks = tensor(bks, GEIST_DTYPE_F32, 2, sh->n_kv, sh->n_kv_heads, 0);
    const struct geist_tensor tvs = tensor(bvs, GEIST_DTYPE_F32, 2, sh->n_kv, sh->n_kv_heads, 0);
    struct geist_tensor to = tensor(bo, GEIST_DTYPE_F32, 3, sh->n_q, sh->n_q_heads, sh->head_dim);
    const struct geist_attention_kv_int8_args args = {.q              = &tq,
                                                      .k              = &tk,
                                                      .k_scale        = &tks,
                                                      .v              = &tv,
                                                      .v_scale        = &tvs,
                                                      .out            = &to,
                                                      .q_offset       = sh->n_kv - sh->n_q,
                                                      .sliding_window = sh->window};
    enum geist_status                         st   = fused->attention_kv_int8(be, &args);
    memcpy(out, vt->buffer_map(bo), q_elems * sizeof *out);
    vt->buffer_unmap(bo);
    size_t unwritten = 0;
    double max_d = 0.0, max_t = 0.0;
    for (size_t i = 0; i < q_elems; i++) {
        if (memcmp(&out[i], &POISON, sizeof POISON) == 0) {
            unwritten++;
            continue;
        }
        const double d = fabs((double) out[i] - ref[i]);
        const double e = fabs((double) out[i] - (double) twin[i]);
        max_d          = d > max_d ? d : max_d;
        max_t          = e > max_t ? e : max_t;
    }
    if (st != GEIST_OK || unwritten != 0 || !(max_d <= tol) || !(max_t <= tol)) {
        fprintf(stderr,
                "FAIL: %s n_q=%zu heads=%zu/%zu hd=%zu n_kv=%zu window=%zu: status %d, %zu "
                "unwritten, max|d| %.3g vs the reference, %.3g vs the portable loop (tol %.3g)\n",
                BACKENDS[bi],
                sh->n_q,
                sh->n_q_heads,
                sh->n_kv_heads,
                sh->head_dim,
                sh->n_kv,
                sh->window,
                (int) st,
                unwritten,
                max_d,
                max_t,
                tol);
        fails++;
    }
#if defined(_OPENMP)
    const int team = omp_get_max_threads();
    omp_set_num_threads(1);
    st = fused->attention_kv_int8(be, &args);
    omp_set_num_threads(team);
    memcpy(out1, vt->buffer_map(bo), q_elems * sizeof *out1);
    vt->buffer_unmap(bo);
    if (st != GEIST_OK || memcmp(out, out1, q_elems * sizeof *out) != 0) {
        fprintf(stderr,
                "FAIL: %s n_q=%zu heads=%zu/%zu hd=%zu n_kv=%zu: 1 thread != %d threads\n",
                BACKENDS[bi],
                sh->n_q,
                sh->n_q_heads,
                sh->n_kv_heads,
                sh->head_dim,
                sh->n_kv,
                team);
        fails++;
    }
#endif
    /* The last query one position past the cache: refused, nothing written. */
    struct geist_attention_kv_int8_args past = args;
    past.q_offset                            = sh->n_kv - sh->n_q + 1;
    memcpy(out1, vt->buffer_map(bo), q_elems * sizeof *out1);
    vt->buffer_unmap(bo);
    st                 = fused->attention_kv_int8(be, &past);
    const bool touched = memcmp(out1, vt->buffer_map(bo), q_elems * sizeof *out1) != 0;
    vt->buffer_unmap(bo);
    if (st != GEIST_E_INVALID_ARG || touched) {
        fprintf(stderr,
                "FAIL: %s n_q=%zu n_kv=%zu q_offset=%zu: status %d (%s the output), not a "
                "refusal\n",
                BACKENDS[bi],
                sh->n_q,
                sh->n_kv,
                past.q_offset,
                (int) st,
                touched ? "wrote" : "left");
        fails++;
    }
done:
    for (struct geist_buffer **b = (struct geist_buffer *[]) {bq, bk, bks, bv, bvs, bo},
                             **e = b + 6;
         b < e;
         b++) {
        if (*b != nullptr) {
            vt->buffer_destroy(be, *b);
        }
    }
    safe_free((void **) &poison);
    safe_free((void **) &out);
    safe_free((void **) &out1);
    return fails;
}

static int check_shape(const struct shape *sh) {
    const size_t kv_elems = sh->n_kv * sh->n_kv_heads * sh->head_dim;
    const size_t q_elems  = sh->n_q * sh->n_q_heads * sh->head_dim;
    int8_t      *k        = heap_alloc_array_aligned(int8_t, kv_elems);
    int8_t      *v        = heap_alloc_array_aligned(int8_t, kv_elems);
    float       *ks       = heap_alloc_array_aligned(float, sh->n_kv * sh->n_kv_heads);
    float       *vs       = heap_alloc_array_aligned(float, sh->n_kv * sh->n_kv_heads);
    float       *q        = heap_alloc_array_aligned(float, q_elems);
    float       *out      = heap_alloc_array_aligned(float, q_elems);
    float       *out1     = heap_alloc_array_aligned(float, q_elems);
    double      *ref      = heap_alloc_array_aligned(double, q_elems);
    double      *scores   = heap_alloc_array_aligned(double, sh->n_kv);
    const size_t n_scr    = attention_int8_scratch_floats(sh->n_q_heads, sh->head_dim);
    float       *scratch  = heap_alloc_array_aligned(float, n_scr);
    float       *out2     = heap_alloc_array_aligned(float, q_elems);
    uint8_t     *k4       = heap_alloc_array_aligned(uint8_t, kv_elems / 2 + 1);
    uint8_t     *v4       = heap_alloc_array_aligned(uint8_t, kv_elems / 2 + 1);
    int          fails    = 0;
    if (k == nullptr || v == nullptr || ks == nullptr || vs == nullptr || q == nullptr ||
        out == nullptr || out1 == nullptr || ref == nullptr || scores == nullptr ||
        scratch == nullptr || out2 == nullptr || k4 == nullptr || v4 == nullptr) {
        fprintf(stderr, "ERROR: allocation failed\n");
        fails = 1;
        goto done;
    }
    /* INT4: values in [-7, 7], which the 4-bit cache holds exactly. */
    const uint32_t span = sh->int4 ? 15u : 255u;
    for (size_t i = 0; i < kv_elems; i++) {
        k[i] = (int8_t) ((int) (next_u32() % span) - (int) (span / 2));
        v[i] = (int8_t) ((int) (next_u32() % span) - (int) (span / 2));
        if (sh->full) {
            k[i] = (int8_t) (next_u32() % 3u == 0u ? -128 : (int) (next_u32() % 256u) - 128);
        }
    }
    for (size_t i = 0; i < sh->n_kv * sh->n_kv_heads; i++) {
        ks[i] = 0.0005f + (float) (next_u32() % 1000u) * 2e-6f;
        vs[i] = 0.0005f + (float) (next_u32() % 1000u) * 2e-6f;
    }
    if (sh->trend > 0.0f) {
        /* |score| grows from ~0.2 to ~35 times trend along the context (INT8;
         * ~17 times less for INT4's smaller values): rising for the heads
         * whose Q agrees with the pattern, falling for the others. */
        const float  top = 0.1f * sh->trend;
        const size_t row = sh->n_kv_heads * sh->head_dim;
        for (size_t s = 1; s < sh->n_kv; s++) {
            memcpy(k + s * row, k, row);
        }
        for (size_t s = 0; s < sh->n_kv; s++) {
            for (size_t h = 0; h < sh->n_kv_heads; h++) {
                ks[s * sh->n_kv_heads + h] = 0.0005f + top * (float) s / (float) sh->n_kv;
            }
        }
    }
    if (sh->int4) {
        float row[TRANSFORMER_HEAD_DIM_MAX];
        for (size_t r = 0; r < sh->n_kv * sh->n_kv_heads; r++) {
            for (size_t i = 0; i < sh->head_dim; i++) {
                row[i] = (float) k[r * sh->head_dim + i];
            }
            int4_pack_row(sh->head_dim, row, 1.0f, k4 + r * sh->head_dim / 2);
            for (size_t i = 0; i < sh->head_dim; i++) {
                row[i] = (float) v[r * sh->head_dim + i];
            }
            int4_pack_row(sh->head_dim, row, 1.0f, v4 + r * sh->head_dim / 2);
        }
    }
    for (size_t i = 0; i < q_elems; i++) {
        q[i] = ((float) (next_u32() % 2001u) - 1000.0f) * 1e-3f;
    }
    const float POISON = -7.5e30f;
    uint32_t    poison_bits;
    memcpy(&poison_bits, &POISON, sizeof poison_bits);
    for (size_t i = 0; i < q_elems; i++) {
        out[i] = POISON;
    }
    run(sh, n_scr, q, k, k4, ks, v, v4, vs, out, scratch);
    reference(sh, q, k, ks, v, vs, ref, scores);

    double scale = 0.0, max_d = 0.0;
    size_t unwritten = 0;
    for (size_t i = 0; i < q_elems; i++) {
        scale = fabs(ref[i]) > scale ? fabs(ref[i]) : scale;
    }
    for (size_t i = 0; i < q_elems; i++) {
        uint32_t bits;
        memcpy(&bits, &out[i], sizeof bits);
        if (bits == poison_bits) {
            unwritten++;
            continue;
        }
        const double d = fabs((double) out[i] - ref[i]);
        max_d          = d > max_d ? d : max_d;
    }
    /* Scores in the hundreds (steep trends) are a float ulp of 3e-5 apart,
     * and the kernel and this reference may multiply the three factors of a
     * score in different orders (they do under clang for aarch64): about
     * 1e-5 of the output, in any float kernel. */
    const double tol = (sh->trend > 1.0f ? 1e-4 : 1e-5) * scale + 1e-7;
    if (unwritten != 0 || !(max_d <= tol)) {
        fprintf(stderr,
                "FAIL: n_q=%zu heads=%zu/%zu hd=%zu n_kv=%zu window=%zu: %zu unwritten, "
                "max|d| %.3g (tol %.3g)\n",
                sh->n_q,
                sh->n_q_heads,
                sh->n_kv_heads,
                sh->head_dim,
                sh->n_kv,
                sh->window,
                unwritten,
                max_d,
                tol);
        fails++;
    }
    /* Without scratch decode runs unsplit: the reference bound holds too. */
    run(sh, 0, q, k, k4, ks, v, v4, vs, out2, nullptr);
    double max_d2 = 0.0;
    for (size_t i = 0; i < q_elems; i++) {
        const double d = fabs((double) out2[i] - ref[i]);
        max_d2         = d > max_d2 ? d : max_d2;
    }
    if (!(max_d2 <= tol)) {
        fprintf(stderr,
                "FAIL: n_q=%zu heads=%zu/%zu hd=%zu n_kv=%zu without scratch: max|d| %.3g "
                "(tol %.3g)\n",
                sh->n_q,
                sh->n_q_heads,
                sh->n_kv_heads,
                sh->head_dim,
                sh->n_kv,
                max_d2,
                tol);
        fails++;
    }
    if (!sh->int4) {
        for (size_t bi = 0; bi < N_BACKENDS; bi++) {
            if (g_be[bi] != nullptr) {
                fails += check_backend(bi, sh, q, k, ks, v, vs, ref, out, tol);
            }
        }
    }
#if defined(_OPENMP)
    /* Same bits whatever the team: each head is one thread's sequential work. */
    const int team = omp_get_max_threads();
    omp_set_num_threads(1);
    run(sh, n_scr, q, k, k4, ks, v, v4, vs, out1, scratch);
    omp_set_num_threads(team);
    if (memcmp(out, out1, q_elems * sizeof *out) != 0) {
        fprintf(stderr,
                "FAIL: n_q=%zu heads=%zu/%zu hd=%zu n_kv=%zu: 1 thread != %d threads\n",
                sh->n_q,
                sh->n_q_heads,
                sh->n_kv_heads,
                sh->head_dim,
                sh->n_kv,
                team);
        fails++;
    }
#endif
    printf("  n_q=%-2zu heads=%2zu/%zu hd=%-3zu n_kv=%-5zu window=%-3zu%s%s max|d|/scale %.2e\n",
           sh->n_q,
           sh->n_q_heads,
           sh->n_kv_heads,
           sh->head_dim,
           sh->n_kv,
           sh->window,
           sh->trend > 1.0f   ? " steep"
           : sh->trend > 0.0f ? " trend"
                              : "",
           sh->int4 ? " int4" : "",
           scale > 0.0 ? max_d / scale : max_d);
done:
    safe_free((void **) &k);
    safe_free((void **) &v);
    safe_free((void **) &ks);
    safe_free((void **) &vs);
    safe_free((void **) &q);
    safe_free((void **) &out);
    safe_free((void **) &out1);
    safe_free((void **) &ref);
    safe_free((void **) &scores);
    safe_free((void **) &scratch);
    safe_free((void **) &out2);
    safe_free((void **) &k4);
    safe_free((void **) &v4);
    return fails;
}

int main(void) {
    static const struct shape SHAPES[] = {
            {1, 16, 8, 128, 512, 0, 0.0f, false, false},    /* 2 heads per pass, decode */
            {16, 16, 8, 128, 700, 0, 0.0f, false, false},   /* ... prefill chunk */
            {16, 16, 8, 128, 200, 0, 0.0f, false, false},   /* ... context inside one block */
            {16, 24, 8, 128, 300, 0, 0.0f, false, false},   /* 3 heads per pass (head_dim 128) */
            {1, 32, 8, 64, 512, 0, 0.0f, false, false},     /* 4 heads per pass, decode */
            {16, 32, 8, 64, 2100, 0, 0.0f, false, false},   /* ... prefill chunk */
            {2, 16, 8, 64, 1600, 0, 0.0f, false, false},    /* 2 per pass: 1.6 MB of K/V */
            {4, 15, 5, 64, 2600, 0, 0.0f, false, false},    /* 3 per pass: 1.6 MB of K/V */
            {9, 15, 5, 64, 300, 0, 0.0f, false, false},     /* one-head loop: 192 KB of K/V */
            {1, 15, 5, 64, 300, 0, 0.0f, false, false},     /* ... decode */
            {1, 15, 5, 64, 1500, 0, 0.0f, false, false},    /* ... decode over 3 blocks */
            {4, 15, 5, 64, 1800, 700, 0.0f, false, false},  /* ... sliding window, 2 blocks */
            {16, 8, 1, 256, 600, 0, 0.0f, false, false},    /* MQA prefill: 4 per pass */
            {1, 8, 1, 256, 600, 0, 0.0f, false, false},     /* MQA decode: 4 per pass, 4 chunks */
            {1, 8, 1, 256, 200, 0, 0.0f, false, false},     /* ... too short to split: one-head */
            {1, 12, 2, 128, 500, 0, 0.0f, false, false},    /* decode: 3 per pass, 3 chunks */
            {1, 4, 2, 128, 400, 0, 0.0f, false, false},     /* decode: 2 per pass, 2 chunks */
            {3, 4, 4, 64, 100, 0, 0.0f, false, false},      /* MHA: one-head loop */
            {16, 16, 8, 128, 900, 256, 0.0f, false, false}, /* sliding window, grouped */
            {4, 32, 8, 64, 2500, 512, 0.0f, false, false},  /* sliding window, grouped */
            {1, 8, 1, 256, 2500, 512, 0.0f, false, false},  /* sliding window, decode in chunks */
            {1, 32, 8, 64, 3000, 0, 1.0f, false, false},    /* trending scores, decode */
            {1, 8, 1, 256, 3000, 0, 1.0f, false, false},    /* ... decode in chunks */
            {8, 16, 8, 128, 2000, 0, 1.0f, false, false},   /* ... prefill chunk */
            {1, 4, 4, 64, 2000, 0, 1.0f, false, false},     /* ... MHA, one-head over 4 blocks */
            {16, 16, 8, 128, 700, 0, 0.0f, true, false},    /* INT4: prefill chunk */
            {1, 32, 8, 64, 1500, 0, 0.0f, true, false},     /* INT4: decode over 3 blocks */
            {4, 8, 1, 256, 900, 300, 0.0f, true, false},    /* INT4: MQA, sliding window */
            {1, 4, 4, 64, 2000, 0, 1.0f, true, false},      /* INT4: trending scores */
            /* steep: scores hundreds apart, weights below the exp floor */
            {1, 32, 8, 64, 3000, 0, 10.0f, false, false},   /* grouped decode */
            {1, 8, 1, 256, 3000, 0, 10.0f, false, false},   /* ... in chunks: merge */
            {8, 16, 8, 128, 2000, 0, 10.0f, false, false},  /* grouped prefill chunk */
            {1, 4, 4, 64, 2000, 0, 10.0f, false, false},    /* one-head loop */
            {4, 15, 5, 64, 1800, 700, 10.0f, false, false}, /* ... sliding window */
            {1, 4, 4, 64, 2000, 0, 170.0f, true, false},    /* INT4 */
            /* K over the whole int8 range, -128 included */
            {1, 32, 8, 64, 700, 0, 0.0f, false, true},  /* grouped decode */
            {5, 16, 8, 80, 300, 0, 0.0f, false, true},  /* head_dim 80: the tails */
            {3, 12, 4, 72, 200, 0, 0.0f, false, false}, /* head_dim 72, 3 per pass */
            {2, 2, 2, 96, 150, 0, 0.0f, false, false},  /* MHA at head_dim 96 */
            {5, 6, 2, 24, 300, 0, 0.0f, false, true},   /* head_dim 24: dots all tail */
            {1, 4, 1, 8, 700, 0, 0.0f, false, false},   /* head_dim 8, split decode */
    };
    for (size_t bi = 0; bi < N_BACKENDS; bi++) {
        if (geist_backend_create(BACKENDS[bi], nullptr, nullptr, &g_be[bi]) != GEIST_OK) {
            g_be[bi] = nullptr; /* not in this build */
        }
    }
    int fails = 0;
    for (size_t i = 0; i < sizeof SHAPES / sizeof *SHAPES; i++) {
        fails += check_shape(&SHAPES[i]);
    }
    for (size_t bi = 0; bi < N_BACKENDS; bi++) {
        if (g_be[bi] != nullptr) {
            geist_backend_destroy(g_be[bi]);
        }
    }
    if (fails != 0) {
        fprintf(stderr, "FAIL: %d check(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: INT8 attention matches the double reference on every path, the backends' "
           "kernels too\n");
    return GEIST_TEST_PASS;
}
