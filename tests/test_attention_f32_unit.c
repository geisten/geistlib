/*
 * test_attention_f32_unit — every CPU backend's F32 attention op, the path
 * of the FP32 KV cache, against a double-precision reference.
 *
 * desc->prims->attention is what the transformer calls with an FP32 cache.
 * Each backend has its own: cpu_neon a NEON kernel that falls back to the
 * gemma4_kernels.c reference cpu_scalar runs when there are fewer than 16
 * (query, head) pairs, cpu_x86 an AVX2 kernel that takes every shape and
 * plans its work (attention.c: passes of up to four heads of a KV group,
 * prefill items of up to four queries, decode split into chunks). The
 * shapes cover every branch: MQA, GQA and MHA, one query (decode) and many
 * (prefill), head_dim 64, 128 and 256 (compiled for) and others (20, 40
 * and 80 with their tails, 512), contexts on both sides of 512-position
 * blocks, sliding windows, and each of cpu_x86's plans: passes of 1-4
 * heads, one, two or four queries an item (the last one short), decode
 * split into 2-4 chunks or not.
 *
 * K and V rows that no query may see (before every window) hold NaN, and
 * the output starts as NaN: a kernel that reads past the mask, or leaves a
 * row unwritten, fails; the check reads the bits, since tests build with
 * -ffast-math, under which isfinite may fold to true. Some shapes scale the
 * keys from position 512 on by 300, so a later block's scores exceed the
 * first block's by hundreds: exp overflows unless the softmax's running
 * max follows them. Scores that large carry rounding of about 1e-5 into
 * the output in any float kernel (the reference, too), so those shapes get
 * 1e-4 rather than 2e-5.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_rng = 0x2545F4914F6CDD1Dull;
static float    urand(void) { /* [-1, 1) */
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (float) (g_rng >> 40) / (float) (1u << 23) - 1.0f;
}

static bool finite_f32(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return (u & 0x7f800000u) != 0x7f800000u;
}

struct shape {
    size_t n_q, n_q_heads, n_kv_heads, head_dim, n_kv, window;
    float  peak; /* 0, or the factor on the keys from position 512 on */
};

/* out[t,h,:] = softmax_s(q[t,h,:] . k[s,kv(h),:]) v[s,kv(h),:] over the
 * causal window, in double. */
static void reference(const struct shape *sh,
                      size_t              q_offset,
                      const float        *q,
                      const float        *k,
                      const float        *v,
                      double             *out,
                      double             *scores) {
    const size_t group = sh->n_q_heads / sh->n_kv_heads;
    for (size_t t = 0; t < sh->n_q; t++) {
        const size_t q_pos = q_offset + t;
        const size_t s_lo  = sh->window > 0 && q_pos + 1 > sh->window ? q_pos + 1 - sh->window : 0;
        for (size_t h = 0; h < sh->n_q_heads; h++) {
            const size_t kv_h = h / group;
            const float *qv   = q + (t * sh->n_q_heads + h) * sh->head_dim;
            double       mx   = -INFINITY;
            for (size_t s = s_lo; s <= q_pos; s++) {
                const float *kv  = k + (s * sh->n_kv_heads + kv_h) * sh->head_dim;
                double       dot = 0.0;
                for (size_t i = 0; i < sh->head_dim; i++) {
                    dot += (double) qv[i] * kv[i];
                }
                scores[s] = dot;
                mx        = dot > mx ? dot : mx;
            }
            double sum = 0.0;
            for (size_t s = s_lo; s <= q_pos; s++) {
                scores[s] = exp(scores[s] - mx);
                sum += scores[s];
            }
            double *o = out + (t * sh->n_q_heads + h) * sh->head_dim;
            for (size_t i = 0; i < sh->head_dim; i++) {
                o[i] = 0.0;
            }
            for (size_t s = s_lo; s <= q_pos; s++) {
                const float *vv = v + (s * sh->n_kv_heads + kv_h) * sh->head_dim;
                for (size_t i = 0; i < sh->head_dim; i++) {
                    o[i] += scores[s] / sum * vv[i];
                }
            }
        }
    }
}

static struct geist_tensor tensor3(struct geist_buffer *b, size_t n, size_t heads, size_t hd) {
    return (struct geist_tensor) {
            .buffer = b,
            .offset = 0,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 3,
            .shape  = {(int64_t) n, (int64_t) heads, (int64_t) hd},
            .stride = {(int64_t) (heads * hd), (int64_t) hd, 1},
    };
}

static struct geist_buffer *upload(struct geist_backend *be, const float *x, size_t n) {
    struct geist_buffer *b = nullptr;
    if (be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &b) !=
                GEIST_OK ||
        be->desc->vtbl->buffer_upload(b, n * sizeof(float), (const uint8_t *) x) != GEIST_OK) {
        fprintf(stderr, "buffer create/upload failed\n");
        exit(GEIST_TEST_ERROR);
    }
    return b;
}

/* Runs one shape on one backend; returns 1 on a mismatch. */
static int run(struct geist_backend *be, const char *backend, const struct shape *sh) {
    const size_t q_offset = sh->n_kv - sh->n_q; /* the new queries end the context */
    const size_t nq       = sh->n_q * sh->n_q_heads * sh->head_dim;
    const size_t nkv      = sh->n_kv * sh->n_kv_heads * sh->head_dim;
    float       *q        = xmalloc(nq * sizeof *q);
    float       *k        = xmalloc(nkv * sizeof *k);
    float       *v        = xmalloc(nkv * sizeof *v);
    float       *out      = xmalloc(nq * sizeof *out);
    double      *ref      = xmalloc(nq * sizeof *ref);
    double      *scores   = xmalloc(sh->n_kv * sizeof *scores);
    const float  qs       = 1.0f / sqrtf((float) sh->head_dim);
    for (size_t i = 0; i < nq; i++) {
        q[i]   = urand() * qs;
        out[i] = NAN;
    }
    for (size_t i = 0; i < nkv; i++) {
        k[i] = urand();
        v[i] = urand();
        if (sh->peak > 0.0f && i >= 512 * sh->n_kv_heads * sh->head_dim) {
            k[i] *= sh->peak;
        }
    }
    /* Rows before the first query's window are outside every mask. */
    const size_t unseen =
            sh->window > 0 && q_offset + 1 > sh->window ? q_offset + 1 - sh->window : 0;
    for (size_t i = 0; i < unseen * sh->n_kv_heads * sh->head_dim; i++) {
        k[i] = NAN;
        v[i] = NAN;
    }
    reference(sh, q_offset, q, k, v, ref, scores);

    struct geist_buffer *bq = upload(be, q, nq), *bk = upload(be, k, nkv), *bv = upload(be, v, nkv);
    struct geist_buffer *bo = upload(be, out, nq);
    struct geist_tensor  tq = tensor3(bq, sh->n_q, sh->n_q_heads, sh->head_dim);
    struct geist_tensor  tk = tensor3(bk, sh->n_kv, sh->n_kv_heads, sh->head_dim);
    struct geist_tensor  tv = tensor3(bv, sh->n_kv, sh->n_kv_heads, sh->head_dim);
    struct geist_tensor  to = tensor3(bo, sh->n_q, sh->n_q_heads, sh->head_dim);
    const enum geist_status st =
            be->desc->prims->attention(be, &tq, &tk, &tv, q_offset, sh->window, &to);
    if (st == GEIST_OK) {
        (void) be->desc->vtbl->buffer_download(nq * sizeof(float), (uint8_t *) out, bo);
    }
    double worst = 0.0;
    size_t at = 0, nonfinite = 0;
    for (size_t i = 0; st == GEIST_OK && i < nq; i++) {
        if (!finite_f32(out[i])) {
            at = nonfinite++ == 0 ? i : at;
            continue;
        }
        const double e = fabs((double) out[i] - ref[i]);
        if (e > worst) {
            worst = e;
            at    = nonfinite == 0 ? i : at;
        }
    }
    const double tol = sh->peak > 0.0f ? 1e-4 : 2e-5;
    be->desc->vtbl->buffer_destroy(be, bq);
    be->desc->vtbl->buffer_destroy(be, bk);
    be->desc->vtbl->buffer_destroy(be, bv);
    be->desc->vtbl->buffer_destroy(be, bo);
    char what[200];
    snprintf(what,
             sizeof what,
             "%s: n_q %zu, heads %zu/%zu, head_dim %zu, n_kv %zu, window %zu, peak %g: status "
             "%d, %zu non-finite, max error %.3g (limit %.0e) at %zu",
             backend,
             sh->n_q,
             sh->n_q_heads,
             sh->n_kv_heads,
             sh->head_dim,
             sh->n_kv,
             sh->window,
             (double) sh->peak,
             (int) st,
             nonfinite,
             worst,
             tol,
             at);
    free(q);
    free(k);
    free(v);
    free(out);
    free(ref);
    free(scores);
    return geist_expect(st == GEIST_OK && nonfinite == 0 && worst <= tol, what);
}

int main(void) {
    static const struct shape SHAPES[] = {
            /* decode, fewer than 16 (query, head) pairs: cpu_neon's reference path */
            {1, 4, 1, 64, 1, 0, 0.0f},
            {1, 4, 2, 64, 700, 0, 0.0f},
            /* decode, 16+ pairs: the SIMD kernels, MQA and GQA */
            {1, 16, 1, 64, 1, 0, 0.0f},
            {1, 16, 1, 128, 511, 0, 0.0f},
            {1, 16, 1, 256, 512, 0, 0.0f},
            {1, 16, 4, 128, 513, 0, 0.0f},
            {1, 16, 4, 64, 1500, 0, 0.0f},
            {1, 32, 8, 256, 1100, 0, 0.0f},
            {1, 16, 2, 128, 1500, 700, 0.0f},
            {1, 16, 1, 64, 2000, 512, 0.0f},
            /* prefill */
            {3, 4, 2, 64, 3, 0, 0.0f},
            {24, 4, 2, 64, 24, 0, 0.0f},
            {40, 8, 1, 128, 600, 0, 0.0f},
            {40, 8, 4, 64, 600, 128, 0.0f},
            {17, 16, 1, 256, 1040, 0, 0.0f},
            {64, 4, 4, 128, 64, 16, 0.0f},
            /* later blocks outscore the first by hundreds */
            {1, 4, 2, 64, 1300, 0, 300.0f},
            {1, 16, 4, 128, 1500, 0, 300.0f},
            {1, 16, 1, 256, 1100, 0, 300.0f},
            {20, 8, 2, 64, 1200, 0, 300.0f},
            {24, 8, 2, 64, 1300, 0, 300.0f},
            /* cpu_x86's plans: passes of three heads (groups of 3 and 6),
             * four queries an item (the last one short) and two, a window
             * that differs between an item's queries, decode split into 2-4
             * chunks and not, head_dim with tails and 512 */
            {1, 12, 4, 64, 700, 0, 0.0f},
            {1, 12, 2, 64, 1000, 0, 0.0f},
            {30, 15, 5, 64, 1100, 0, 0.0f},
            {23, 8, 1, 256, 1300, 0, 0.0f},
            {30, 16, 4, 64, 2000, 1000, 0.0f},
            {20, 8, 8, 64, 1100, 0, 0.0f},
            {1, 32, 8, 64, 300, 0, 0.0f},
            {1, 8, 2, 80, 900, 0, 0.0f},
            {33, 8, 2, 80, 1200, 0, 0.0f},
            {1, 6, 3, 40, 600, 0, 0.0f},
            {9, 6, 3, 40, 700, 0, 0.0f},
            {1, 4, 4, 20, 530, 0, 0.0f},
            {17, 4, 1, 20, 600, 0, 0.0f},
            {1, 8, 1, 512, 1500, 0, 0.0f},
            {12, 8, 1, 512, 900, 0, 0.0f},
    };
    static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};
    int                      fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue;
        }
        ran++;
        for (size_t i = 0; i < sizeof SHAPES / sizeof SHAPES[0]; i++) {
            fails += run(be, BACKENDS[b], &SHAPES[i]);
        }
        geist_backend_destroy(be);
    }
    if (ran == 0) {
        printf("SKIP: no CPU backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: F32 attention matches the double reference on every CPU backend\n");
    return GEIST_TEST_PASS;
}
