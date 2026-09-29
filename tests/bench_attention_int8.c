/*
 * bench_attention_int8 — attention_int8_via_buffers, the INT8-KV attention
 * core every CPU backend runs (cpu_x86 defaults to the INT8 cache), timed on
 * its own over context lengths and head layouts.
 *
 * Model-level runs mix it with the GEMVs; this isolates the kernel. Random
 * int8 K/V with moderate scales and random Q keep the softmax spread out;
 * a peaked one costs the same since the exponent floor (ATTN_EXP_FLOOR).
 * The same calls go through fused->attention_kv_int8 of the first backend
 * in the build whose probe binds it (cpu_x86): the kernel_* columns, -1
 * where none does.
 *
 * Layouts: 32/8 heads hd 64 (Llama-3.2-1B), 16/8 hd 128 (Qwen3-0.6B),
 * 15/5 hd 64 (SmolLM2-360M), 8/1 hd 256 (MQA; Gemma 4 E2B's sliding layers
 * up to their window, 512), 32/32 hd 64 (MHA, SmolLM2-1.7B), 8/1 hd 512
 * (Gemma 4 E2B's full-attention layers). Decode is n_q = 1 at the end of
 * the context; prefill is one 64-token chunk ending there.
 * Reports the median of the timed calls, host loop and kernel alike.
 *
 * Usage: bench_attention_int8 [iters]   (default 20)
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "src/archs/transformer/forward/internal.h"
#include "src/archs/transformer/forward.h"

#include <geist.h>
#include <geist_backend.h>

#include "heap.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec * 1e-6;
}

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

struct layout {
    const char *name;
    size_t      n_q_heads, n_kv_heads, head_dim;
};

/* A buffer on `be` holding `bytes` from `src`, or nullptr. */
static struct geist_buffer *upload(struct geist_backend *be, size_t bytes, const void *src) {
    const struct geist_backend_vtbl *vt = be->desc->vtbl;
    struct geist_buffer             *b  = nullptr;
    if (vt->buffer_create(be, bytes, GEIST_BUFFER_SCRATCH, 0, &b) != GEIST_OK || b == nullptr) {
        return nullptr;
    }
    memcpy(vt->buffer_map(b), src, bytes);
    vt->buffer_unmap(b);
    return b;
}

static struct geist_tensor
view(struct geist_buffer *b, enum geist_dtype dt, int ndim, size_t s0, size_t s1, size_t s2) {
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

/* Median ms of fused->attention_kv_int8 on `be` over the same bytes, or -1
 * if the call cannot be set up. */
static double time_kernel(struct geist_backend *be,
                          const struct layout  *l,
                          size_t                n_kv,
                          size_t                n_q,
                          int                   iters,
                          const float          *q,
                          const int8_t         *k,
                          const float          *ks,
                          const int8_t         *v,
                          const float          *vs,
                          double               *t) {
    const struct geist_backend_vtbl  *vt       = be->desc->vtbl;
    const struct geist_backend_fused *fused    = geist_backend_fused_tbl(be);
    const size_t                      kv_elems = n_kv * l->n_kv_heads * l->head_dim;
    const size_t                      q_elems  = n_q * l->n_q_heads * l->head_dim;
    const size_t                      n_sc     = n_kv * l->n_kv_heads;
    struct geist_buffer              *b[6]     = {
            upload(be, q_elems * sizeof *q, q),
            upload(be, kv_elems, k),
            upload(be, n_sc * sizeof *ks, ks),
            upload(be, kv_elems, v),
            upload(be, n_sc * sizeof *vs, vs),
            upload(be, q_elems * sizeof *q, q),
    };
    double med = -1.0;
    if (b[0] != nullptr && b[1] != nullptr && b[2] != nullptr && b[3] != nullptr &&
        b[4] != nullptr && b[5] != nullptr) {
        const struct geist_tensor tq =
                view(b[0], GEIST_DTYPE_F32, 3, n_q, l->n_q_heads, l->head_dim);
        const struct geist_tensor tk =
                view(b[1], GEIST_DTYPE_I8, 3, n_kv, l->n_kv_heads, l->head_dim);
        const struct geist_tensor tks = view(b[2], GEIST_DTYPE_F32, 2, n_kv, l->n_kv_heads, 0);
        const struct geist_tensor tv =
                view(b[3], GEIST_DTYPE_I8, 3, n_kv, l->n_kv_heads, l->head_dim);
        const struct geist_tensor tvs = view(b[4], GEIST_DTYPE_F32, 2, n_kv, l->n_kv_heads, 0);
        struct geist_tensor to = view(b[5], GEIST_DTYPE_F32, 3, n_q, l->n_q_heads, l->head_dim);
        const struct geist_attention_kv_int8_args args = {.q        = &tq,
                                                          .k        = &tk,
                                                          .k_scale  = &tks,
                                                          .v        = &tv,
                                                          .v_scale  = &tvs,
                                                          .out      = &to,
                                                          .q_offset = n_kv - n_q};
        bool ok = fused->attention_kv_int8(be, &args) == GEIST_OK; /* warm */
        for (int it = 0; it < iters && ok; it++) {
            const double t0 = now_ms();
            ok              = fused->attention_kv_int8(be, &args) == GEIST_OK;
            t[it]           = now_ms() - t0;
        }
        if (ok) {
            qsort(t, (size_t) iters, sizeof *t, cmp_double);
            med = t[iters / 2];
        }
    }
    for (size_t i = 0; i < 6; i++) {
        if (b[i] != nullptr) {
            vt->buffer_destroy(be, b[i]);
        }
    }
    return med;
}

/* Whether `be` binds attention_kv_int8 for layout `l` at n_q queries. */
static bool binds(struct geist_backend *be, const struct layout *l, size_t n_q) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    const struct geist_fusion_query   pq    = {.op         = GEIST_FUSED_ATTN_KV_INT8,
                                               .m          = n_q,
                                               .head_dim   = l->head_dim,
                                               .n_q_heads  = l->n_q_heads,
                                               .n_kv_heads = l->n_kv_heads};
    return fused->attention_kv_int8 != nullptr && fused->supported != nullptr &&
           fused->supported(be, &pq);
}

struct timing {
    double host, kernel; /* median ms; kernel -1 where no backend binds it */
};

static struct timing
time_one(struct geist_backend *be, const struct layout *l, size_t n_kv, size_t n_q, int iters) {
    const size_t  kv_elems = n_kv * l->n_kv_heads * l->head_dim;
    const size_t  q_elems  = n_q * l->n_q_heads * l->head_dim;
    int8_t       *k        = heap_alloc_array_aligned(int8_t, kv_elems);
    int8_t       *v        = heap_alloc_array_aligned(int8_t, kv_elems);
    float        *ks       = heap_alloc_array_aligned(float, n_kv * l->n_kv_heads);
    float        *vs       = heap_alloc_array_aligned(float, n_kv * l->n_kv_heads);
    float        *q        = heap_alloc_array_aligned(float, q_elems);
    float        *out      = heap_alloc_array_aligned(float, q_elems);
    double       *t        = heap_alloc_array_aligned(double, (size_t) iters);
    const size_t  n_scr    = attention_int8_scratch_floats(l->n_q_heads, l->head_dim);
    float        *scratch  = heap_alloc_array_aligned(float, n_scr);
    struct timing med      = {-1.0, -1.0};
    if (k != nullptr && v != nullptr && ks != nullptr && vs != nullptr && q != nullptr &&
        out != nullptr && t != nullptr && scratch != nullptr) {
        for (size_t i = 0; i < kv_elems; i++) {
            k[i] = (int8_t) (next_u32() % 255u - 127u);
            v[i] = (int8_t) (next_u32() % 255u - 127u);
        }
        for (size_t i = 0; i < n_kv * l->n_kv_heads; i++) {
            ks[i] = 0.0005f + (float) (next_u32() % 1000u) * 1e-6f;
            vs[i] = 0.0005f + (float) (next_u32() % 1000u) * 1e-6f;
        }
        for (size_t i = 0; i < q_elems; i++) {
            q[i] = ((float) (next_u32() % 2001u) - 1000.0f) * 1e-3f;
        }
        const size_t q_offset = n_kv - n_q;
        attention_int8_via_buffers(n_q,
                                   l->n_q_heads,
                                   l->head_dim,
                                   n_kv,
                                   l->n_kv_heads,
                                   n_scr,
                                   q_offset,
                                   0,
                                   q,
                                   k,
                                   ks,
                                   v,
                                   vs,
                                   out,
                                   scratch); /* warm */
        for (int it = 0; it < iters; it++) {
            const double t0 = now_ms();
            attention_int8_via_buffers(n_q,
                                       l->n_q_heads,
                                       l->head_dim,
                                       n_kv,
                                       l->n_kv_heads,
                                       n_scr,
                                       q_offset,
                                       0,
                                       q,
                                       k,
                                       ks,
                                       v,
                                       vs,
                                       out,
                                       scratch);
            t[it] = now_ms() - t0;
        }
        qsort(t, (size_t) iters, sizeof *t, cmp_double);
        med.host = t[iters / 2];
        if (be != nullptr && binds(be, l, n_q)) {
            med.kernel = time_kernel(be, l, n_kv, n_q, iters, q, k, ks, v, vs, t);
        }
    }
    safe_free((void **) &k);
    safe_free((void **) &v);
    safe_free((void **) &ks);
    safe_free((void **) &vs);
    safe_free((void **) &q);
    safe_free((void **) &out);
    safe_free((void **) &t);
    safe_free((void **) &scratch);
    return med;
}

int main(int argc, char **argv) {
    const int                  iters = argc > 1 && atoi(argv[1]) > 0 ? atoi(argv[1]) : 20;
    static const struct layout L[]   = {
            {"llama32-1b 32/8 hd64", 32, 8, 64},
            {"qwen3-0.6b 16/8 hd128", 16, 8, 128},
            {"smollm2 15/5 hd64", 15, 5, 64},
            {"mqa 8/1 hd256", 8, 1, 256},
            {"mha 32/32 hd64", 32, 32, 64},
            {"gemma4-e2b full 8/1 hd512", 8, 1, 512},
    };
    static const size_t CTX[] = {512, 1024, 2048, 8192};
    /* The first backend in the build that binds the kernel, if any. */
    static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};
    struct geist_backend    *be         = nullptr;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof *BACKENDS && be == nullptr; b++) {
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK) {
            be = nullptr;
        } else if (!binds(be, &L[0], 1)) {
            geist_backend_destroy(be);
            be = nullptr;
        }
    }
    printf("{\"bench\":\"attention_int8\",\"iters\":%d,\"kernel\":\"%s\",\"rows\":[\n",
           iters,
           be != nullptr ? be->desc->name : "none");
    bool first = true;
    for (size_t li = 0; li < sizeof L / sizeof *L; li++) {
        for (size_t ci = 0; ci < sizeof CTX / sizeof *CTX; ci++) {
            const struct timing dec = time_one(be, &L[li], CTX[ci], 1, iters);
            const struct timing pre = time_one(be, &L[li], CTX[ci], 64, iters / 4 + 1);
            printf("%s {\"layout\":\"%s\",\"ctx\":%zu,\"decode_ms\":%.4f,\"prefill64_ms\":%.3f,"
                   "\"kernel_decode_ms\":%.4f,\"kernel_prefill64_ms\":%.3f}",
                   first ? " " : ",\n ",
                   L[li].name,
                   CTX[ci],
                   dec.host,
                   pre.host,
                   dec.kernel,
                   pre.kernel);
            first = false;
        }
    }
    printf("\n]}\n");
    geist_backend_destroy(be);
    return 0;
}
