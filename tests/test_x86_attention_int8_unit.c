/*
 * test_x86_attention_int8_unit — cpu_x86's two INT8-KV attention kernels,
 * AVX2 (cpu_x86_attention_kv_int8_run) and AVX-512 VNNI
 * (cpu_x86_attention_kv_int8_run_avx512_vnni), called directly on the same
 * caches, against a double-precision reference and against each other.
 *
 * test_attention_int8_unit runs the fused op, which takes one of the two:
 * the VNNI kernel wherever the dispatcher tier and cpuid allow it. This
 * test runs both on every host that can, over the shapes where the VNNI
 * kernel's code differs: each compile-time head_dim (64, 128, 256, 512)
 * and run-time ones with every tail of its 32-byte chunks; passes of 1 to
 * 4 heads, up to three a KV head, and at head_dim 512 passes of 3 and 4
 * (two groups of heads); contexts that leave 1 to 7 positions after its
 * steps of 4 and 8 and span several blocks of 512; split and unsplit
 * decode, prefill, sliding windows; prefill items of one, two and four
 * queries, the last item of a call with fewer, and queries whose spans end
 * in different blocks; scores that trend along the context (the running
 * max grows in every block); K over the whole int8 range, -128 included.
 *
 * The reference repeats the kernels' Q quantization and integer dots, then
 * does the softmax and the V sum in double: each kernel within 1e-5 of the
 * output scale. The two agree within 1e-5 of the scale too (the same
 * integer dots; the rest rounds differently, see
 * attention_int8_avx512_vnni.c). Every output must be written (poisoned
 * first, compared by bit pattern — this file builds with -ffast-math),
 * each kernel must give the same bits with one thread as with the default
 * team, and a decode without the part buffer (unsplit) must match the
 * reference too.
 *
 * The dispatch: fused->attention_kv_int8 must give the VNNI kernel's bits
 * where that may run, the AVX2 kernel's elsewhere and under
 * GEIST_FORCE_ISA=avx2. The dispatcher reads the variable once per
 * process, so the whole matrix runs in a child with it set, then in the
 * parent with the default dispatch. The CI SDE leg (-spr) runs the VNNI
 * kernel on hosts without it.
 */
#define _POSIX_C_SOURCE 200809L /* fork, setenv, waitpid */

#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#if !defined(GEIST_BACKEND_CPU_X86)
int main(void) {
    printf("SKIP: needs cpu_x86 in this build\n");
    return GEIST_TEST_SKIP;
}
#else

#define GEIST_INTERNAL_BACKEND_LAYER
#include "src/backends/cpu_x86/attention.h"
#include "src/backends/cpu_x86/kernel_w4a8.h" /* w4a8_dispatcher_tier */

#include "heap.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

struct shape {
    size_t n_q, n_q_heads, n_kv_heads, head_dim, n_kv, window;
    bool   trend; /* one K row per KV head, K scales rising along the context */
    bool   full;  /* K over the whole int8 range, -128 included */
};

typedef void run_fn(size_t        n_q,
                    size_t        n_q_heads,
                    size_t        head_dim,
                    size_t        n_kv,
                    size_t        n_kv_heads,
                    size_t        part_floats,
                    size_t        q_offset,
                    size_t        sliding_window,
                    const float  *q,
                    const int8_t *k,
                    const float  *k_scale,
                    const int8_t *v,
                    const float  *v_scale,
                    float        *out,
                    float        *part);

struct kernel {
    const char *name;
    run_fn     *run;
};

static const struct kernel AVX2 = {"avx2", cpu_x86_attention_kv_int8_run};
static const struct kernel VNNI = {"avx512_vnni", cpu_x86_attention_kv_int8_run_avx512_vnni};

/* Whether the VNNI kernel may run here: the dispatcher's rule
 * (ai8_vnni_usable in attention_int8.c). */
static bool vnni_usable(void) {
    return w4a8_dispatcher_tier() >= W4A8_ISA_AVX512_VNNI && __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
}

static uint32_t g_rng = 0x6D2B79F5u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

constexpr uint32_t POISON = 0x7FC0DEADu; /* a NaN no kernel produces */

static void poison(size_t n, float x[static n]) {
    for (size_t i = 0; i < n; i++) {
        memcpy(&x[i], &POISON, sizeof POISON);
    }
}

/* Index of the first element still poisoned, or n. */
static size_t first_poisoned(size_t n, const float x[static n]) {
    for (size_t i = 0; i < n; i++) {
        uint32_t b;
        memcpy(&b, &x[i], sizeof b);
        if (b == POISON) {
            return i;
        }
    }
    return n;
}

/* out_ref[t, h, :] in double, from the kernels' int8 Q and integer dots. */
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
            int8_t      qq[512];
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
                scores[s] = (double) dot * sq * ks[s * sh->n_kv_heads + kv_h];
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

/* max |out - ref| / max |ref| over n outputs. */
static double rel_err(size_t n, const float out[static n], const double ref[static n]) {
    double scale = 0.0, err = 0.0;
    for (size_t i = 0; i < n; i++) {
        scale = fmax(scale, fabs(ref[i]));
    }
    for (size_t i = 0; i < n; i++) {
        err = fmax(err, fabs((double) out[i] - ref[i]));
    }
    return err / (scale > 0.0 ? scale : 1.0);
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

/* fused->attention_kv_int8 of cpu_x86 on this shape's cache into `out`
 * (n_out floats); false if the call could not be made or failed. */
static bool call_fused(struct geist_backend *be,
                       const struct shape   *sh,
                       const float          *q,
                       const int8_t         *k,
                       const float          *ks,
                       const int8_t         *v,
                       const float          *vs,
                       size_t                n_out,
                       float                *out) {
    const struct geist_backend_vtbl *vt   = be->desc->vtbl;
    const size_t                     n_kv = sh->n_kv, kvh = sh->n_kv_heads, hd = sh->head_dim;
    struct geist_buffer             *bq  = upload(be, n_out * sizeof(float), q);
    struct geist_buffer             *bk  = upload(be, n_kv * kvh * hd, k);
    struct geist_buffer             *bv  = upload(be, n_kv * kvh * hd, v);
    struct geist_buffer             *bks = upload(be, n_kv * kvh * sizeof(float), ks);
    struct geist_buffer             *bvs = upload(be, n_kv * kvh * sizeof(float), vs);
    struct geist_buffer             *bo  = upload(be, n_out * sizeof(float), out);
    bool                             ok  = false;
    if (bq != nullptr && bk != nullptr && bv != nullptr && bks != nullptr && bvs != nullptr &&
        bo != nullptr) {
        const struct geist_tensor tq  = tensor(bq, GEIST_DTYPE_F32, 3, sh->n_q, sh->n_q_heads, hd);
        const struct geist_tensor tk  = tensor(bk, GEIST_DTYPE_I8, 3, n_kv, kvh, hd);
        const struct geist_tensor tv  = tensor(bv, GEIST_DTYPE_I8, 3, n_kv, kvh, hd);
        const struct geist_tensor tks = tensor(bks, GEIST_DTYPE_F32, 2, n_kv, kvh, 0);
        const struct geist_tensor tvs = tensor(bvs, GEIST_DTYPE_F32, 2, n_kv, kvh, 0);
        struct geist_tensor       to  = tensor(bo, GEIST_DTYPE_F32, 3, sh->n_q, sh->n_q_heads, hd);
        const struct geist_attention_kv_int8_args a = {.q              = &tq,
                                                       .k              = &tk,
                                                       .k_scale        = &tks,
                                                       .v              = &tv,
                                                       .v_scale        = &tvs,
                                                       .out            = &to,
                                                       .q_offset       = n_kv - sh->n_q,
                                                       .sliding_window = sh->window};
        ok = geist_backend_fused_tbl(be)->attention_kv_int8(be, &a) == GEIST_OK;
        if (ok) {
            memcpy(out, vt->buffer_map(bo), n_out * sizeof(float));
            vt->buffer_unmap(bo);
        }
    }
    struct geist_buffer *bufs[] = {bq, bk, bv, bks, bvs, bo};
    for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++) {
        if (bufs[i] != nullptr) {
            vt->buffer_destroy(be, bufs[i]);
        }
    }
    return ok;
}

static int check_shape(struct geist_backend *be, const struct shape *sh, bool vnni) {
    const size_t n_out = sh->n_q * sh->n_q_heads * sh->head_dim;
    const size_t n_c   = sh->n_kv * sh->n_kv_heads * sh->head_dim;
    const size_t n_s   = sh->n_kv * sh->n_kv_heads;
    const size_t pf    = cpu_x86_attention_kv_int8_part_floats(sh->n_q_heads, sh->head_dim);
    float       *q     = xmalloc(n_out * sizeof(float));
    int8_t      *k     = xmalloc(n_c);
    int8_t      *v     = xmalloc(n_c);
    float       *ks    = xmalloc(n_s * sizeof(float));
    float       *vs    = xmalloc(n_s * sizeof(float));
    float       *part  = xmalloc(pf * sizeof(float));
    float       *out[2], *out1 = xmalloc(n_out * sizeof(float));
    out[0]         = xmalloc(n_out * sizeof(float));
    out[1]         = xmalloc(n_out * sizeof(float));
    double *ref    = xmalloc(n_out * sizeof(double));
    double *scores = xmalloc(sh->n_kv * sizeof(double));

    for (size_t i = 0; i < n_out; i++) {
        q[i] = ((float) (next_u32() % 2001) - 1000.0f) / 500.0f;
    }
    for (size_t i = 0; i < n_c; i++) {
        const size_t hd_i = i % (sh->n_kv_heads * sh->head_dim); /* KV head and element */
        const size_t src  = sh->trend ? hd_i : i;                /* trend: position 0's row */
        k[i] = sh->full ? (int8_t) (next_u32() & 0xFF) : (int8_t) ((int) (next_u32() % 255) - 127);
        if (src != i) {
            k[i] = k[src];
        }
        v[i] = (int8_t) ((int) (next_u32() % 255) - 127);
    }
    for (size_t i = 0; i < n_s; i++) {
        const size_t s = i / sh->n_kv_heads;
        ks[i]          = sh->trend ? 0.002f * (1.0f + 4.0f * (float) s / (float) sh->n_kv)
                                   : 0.002f + (float) (next_u32() % 1000) * 1e-5f;
        vs[i]          = 0.01f + (float) (next_u32() % 1000) * 1e-5f;
    }
    reference(sh, q, k, ks, v, vs, ref, scores);

    const struct kernel kernels[2] = {AVX2, VNNI};
    const size_t        n_kernels  = vnni ? 2 : 1;
    const size_t        q_offset   = sh->n_kv - sh->n_q;
    int                 fails      = 0;
    for (size_t ki = 0; ki < n_kernels; ki++) {
        const struct kernel *kn = &kernels[ki];
        poison(n_out, out[ki]);
        kn->run(sh->n_q,
                sh->n_q_heads,
                sh->head_dim,
                sh->n_kv,
                sh->n_kv_heads,
                pf,
                q_offset,
                sh->window,
                q,
                k,
                ks,
                v,
                vs,
                out[ki],
                part);
        const size_t unwritten = first_poisoned(n_out, out[ki]);
        const double err       = rel_err(n_out, out[ki], ref);
        if (unwritten != n_out || !(err <= 1e-5)) {
            fprintf(stderr,
                    "FAIL: %s n_q=%zu %zu/%zu hd=%zu n_kv=%zu window=%zu: %s, error %.3e of the "
                    "scale\n",
                    kn->name,
                    sh->n_q,
                    sh->n_q_heads,
                    sh->n_kv_heads,
                    sh->head_dim,
                    sh->n_kv,
                    sh->window,
                    unwritten != n_out ? "outputs left unwritten" : "outputs written",
                    err);
            fails++;
            continue;
        }
#if defined(_OPENMP)
        const int team = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        poison(n_out, out1);
        kn->run(sh->n_q,
                sh->n_q_heads,
                sh->head_dim,
                sh->n_kv,
                sh->n_kv_heads,
                pf,
                q_offset,
                sh->window,
                q,
                k,
                ks,
                v,
                vs,
                out1,
                part);
#if defined(_OPENMP)
        omp_set_num_threads(team);
#endif
        if (memcmp(out1, out[ki], n_out * sizeof(float)) != 0) {
            fprintf(stderr,
                    "FAIL: %s hd=%zu n_kv=%zu: one thread gives other bits\n",
                    kn->name,
                    sh->head_dim,
                    sh->n_kv);
            fails++;
        }
        if (sh->n_q == 1) {
            /* Unsplit: no part buffer. */
            poison(n_out, out1);
            kn->run(sh->n_q,
                    sh->n_q_heads,
                    sh->head_dim,
                    sh->n_kv,
                    sh->n_kv_heads,
                    0,
                    q_offset,
                    sh->window,
                    q,
                    k,
                    ks,
                    v,
                    vs,
                    out1,
                    nullptr);
            const double err1 = rel_err(n_out, out1, ref);
            if (first_poisoned(n_out, out1) != n_out || !(err1 <= 1e-5)) {
                fprintf(stderr,
                        "FAIL: %s hd=%zu n_kv=%zu unsplit: error %.3e\n",
                        kn->name,
                        sh->head_dim,
                        sh->n_kv,
                        err1);
                fails++;
            }
        }
    }
    double agree = 0.0;
    if (fails == 0 && vnni) {
        double scale = 0.0;
        for (size_t i = 0; i < n_out; i++) {
            scale = fmax(scale, fabs((double) out[0][i]));
            agree = fmax(agree, fabs((double) out[0][i] - out[1][i]));
        }
        agree /= scale > 0.0 ? scale : 1.0;
        if (!(agree <= 1e-5)) {
            fprintf(stderr,
                    "FAIL: hd=%zu n_kv=%zu: the kernels differ by %.3e of the scale\n",
                    sh->head_dim,
                    sh->n_kv,
                    agree);
            fails++;
        }
    }
    /* The fused op dispatches to the VNNI kernel where it may run. */
    if (fails == 0) {
        poison(n_out, out1);
        if (!call_fused(be, sh, q, k, ks, v, vs, n_out, out1)) {
            fprintf(stderr,
                    "FAIL: hd=%zu n_kv=%zu: attention_kv_int8 failed\n",
                    sh->head_dim,
                    sh->n_kv);
            fails++;
        } else if (memcmp(out1, out[vnni ? 1 : 0], n_out * sizeof(float)) != 0) {
            fprintf(stderr,
                    "FAIL: hd=%zu n_kv=%zu: attention_kv_int8 is not the %s kernel\n",
                    sh->head_dim,
                    sh->n_kv,
                    kernels[vnni ? 1 : 0].name);
            fails++;
        }
    }
    if (fails == 0) {
        printf("  n_q=%-2zu %2zu/%-2zu hd=%-3zu n_kv=%-4zu window=%-4zu%-11s ",
               sh->n_q,
               sh->n_q_heads,
               sh->n_kv_heads,
               sh->head_dim,
               sh->n_kv,
               sh->window,
               sh->trend ? (sh->full ? " trend full" : " trend") : (sh->full ? " full" : ""));
        if (vnni) {
            printf("the kernels agree to %.2e of the scale\n", agree);
        } else {
            printf("ok\n");
        }
    }
    safe_free((void **) &q);
    safe_free((void **) &k);
    safe_free((void **) &v);
    safe_free((void **) &ks);
    safe_free((void **) &vs);
    safe_free((void **) &part);
    safe_free((void **) &out[0]);
    safe_free((void **) &out[1]);
    safe_free((void **) &out1);
    safe_free((void **) &ref);
    safe_free((void **) &scores);
    return fails;
}

static int run_all(void) {
    static const struct shape SHAPES[] = {
            /* compile-time head_dims */
            {1, 32, 8, 64, 1030, 0, false, false},   /* 4 heads a pass, split decode */
            {17, 32, 8, 64, 300, 0, false, false},   /* ... prefill */
            {1, 15, 5, 64, 777, 0, false, false},    /* 3 a pass */
            {9, 15, 5, 64, 200, 64, false, false},   /* ... prefill, window */
            {1, 4, 4, 64, 3000, 0, false, false},    /* 1 a pass (MHA), 6 blocks */
            {1, 16, 8, 128, 1500, 0, false, false},  /* 2 a pass */
            {5, 32, 8, 128, 515, 0, false, false},   /* 4 a pass, 2 blocks */
            {1, 8, 1, 256, 2049, 512, false, false}, /* Gemma 4 sliding: 2 passes of 4 */
            {7, 8, 1, 256, 600, 512, false, false},  /* ... prefill */
            {1, 12, 1, 256, 700, 0, false, false},   /* 3 passes of 4 */
            {2, 6, 2, 256, 90, 0, false, false},     /* 3 a pass */
            {1, 8, 1, 512, 1100, 0, false, false},   /* Gemma 4 full: 4 a pass, 2 groups */
            {6, 8, 1, 512, 530, 0, false, false},    /* ... prefill */
            {1, 6, 2, 512, 301, 0, false, false},    /* 3 a pass: groups of 2 and 1 */
            {3, 4, 4, 512, 129, 0, false, false},    /* 1 a pass */
            /* run-time head_dims: every tail of the 32-byte chunks */
            {1, 4, 1, 1, 64, 0, false, false},
            {3, 6, 2, 31, 97, 0, false, false},
            {1, 10, 2, 33, 1031, 0, false, false}, /* 5 a KV head: one-head passes */
            {4, 12, 4, 80, 258, 0, false, false},
            {1, 9, 3, 100, 513, 0, false, false},
            {2, 8, 2, 200, 77, 0, false, false},
            {1, 4, 2, 300, 1029, 0, false, false},
            {1, 2, 1, 384, 150, 0, false, false},
            {2, 4, 1, 511, 70, 0, false, false},
            /* the running max grows in every block; K over the whole range */
            {1, 8, 2, 64, 1600, 0, true, false},
            {3, 8, 1, 256, 1100, 0, true, true},
            {1, 16, 4, 128, 900, 0, false, true},
            {2, 8, 1, 512, 700, 0, true, true},
            /* prefill items of several queries: over 1 MB of K and V, spans
             * past one block */
            {7, 16, 8, 64, 2100, 0, false, false},    /* 4 an item, then 3 */
            {4, 15, 5, 64, 2600, 0, false, false},    /* 2 an item (5 items a query) */
            {6, 16, 8, 64, 2600, 1500, false, false}, /* 4 then 2, window */
            {7, 8, 1, 512, 1100, 0, false, false},    /* 2 an item, then 1 */
            {5, 12, 4, 80, 2000, 0, false, false},    /* 4 then 1, head_dim at run time */
            {8, 16, 4, 64, 2200, 0, true, false},     /* 4 an item, max grows */
    };
    const bool            vnni = vnni_usable();
    struct geist_backend *be   = nullptr;
    if (geist_backend_create("cpu_x86", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "FAIL: cpu_x86 did not create\n");
        return GEIST_TEST_FAIL;
    }
    printf("%s: %s\n",
           getenv("GEIST_FORCE_ISA") != nullptr ? "GEIST_FORCE_ISA=avx2" : "default dispatch",
           vnni ? "both kernels, the fused op on the AVX-512 VNNI one"
                : "the AVX2 kernel, the fused op on it (no AVX-512 VNNI allowed)");
    int fails = 0;
    for (size_t i = 0; i < sizeof SHAPES / sizeof *SHAPES; i++) {
        fails += check_shape(be, &SHAPES[i], vnni);
    }
    geist_backend_destroy(be);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

int main(void) {
    /* Nothing has probed the ISA yet: the child's clamp takes effect. */
    fflush(stdout);
    const pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return GEIST_TEST_ERROR;
    }
    if (pid == 0) {
        setenv("GEIST_FORCE_ISA", "avx2", 1);
        const int rc = run_all();
        fflush(stdout);
        _exit(rc);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) {
        perror("waitpid");
        return GEIST_TEST_ERROR;
    }
    const int forced = WIFEXITED(status) ? WEXITSTATUS(status) : GEIST_TEST_ERROR;
    const int native = run_all();
    if (forced != GEIST_TEST_PASS || native != GEIST_TEST_PASS) {
        fprintf(stderr, "FAIL: GEIST_FORCE_ISA=avx2 run %d, default run %d\n", forced, native);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: both INT8-KV attention kernels match the double reference and each other, "
           "and the fused op takes the one the dispatcher allows\n");
    return GEIST_TEST_PASS;
}

#endif
