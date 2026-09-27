/*
 * test_attention_int8_unit — attention_int8_via_buffers against a
 * double-precision reference, on the one-head loop and the GQA-grouped
 * passes.
 *
 * The kernel quantizes each query head to int8 (scale amax/127, lrintf),
 * dots it exactly against the int8 K rows, softmaxes and accumulates V in
 * fp32 (the grouped passes with an online softmax over blocks of the
 * context). The reference repeats the same Q quantization and integer dots,
 * then does the softmax and the V sum in double: the only differences left
 * are fp32 rounding, so the bound is tight (1e-5 of the output scale).
 *
 * Shapes are chosen to take each path under the current policy: passes of
 * 2, 3 and 4 heads (head_dim 128, head_dim 64 over a short and a long
 * context, MQA prefill) and the one-head loop (3 heads per KV head over a
 * short context, MQA decode with too few items, plain MHA); decode
 * (n_q = 1) and prefill chunks; with and without a sliding window; a context
 * shorter than one online-softmax block. Two shapes make the scores trend
 * along the context (every K row one pattern, K scales rising), so the
 * running max grows in every block for some heads (the rescale path) and
 * never after the first for others. Every output must be written
 * (poisoned first, compared by bit pattern — -ffast-math build), and a
 * 1-thread run must give the same bits as the default team.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "src/archs/transformer/forward/internal.h"

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
    bool   trend; /* one K pattern per KV head, K scales rising along the context */
};

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
    int          fails    = 0;
    if (k == nullptr || v == nullptr || ks == nullptr || vs == nullptr || q == nullptr ||
        out == nullptr || out1 == nullptr || ref == nullptr || scores == nullptr) {
        fprintf(stderr, "ERROR: allocation failed\n");
        fails = 1;
        goto done;
    }
    for (size_t i = 0; i < kv_elems; i++) {
        k[i] = (int8_t) (next_u32() % 255u - 127u);
        v[i] = (int8_t) (next_u32() % 255u - 127u);
    }
    for (size_t i = 0; i < sh->n_kv * sh->n_kv_heads; i++) {
        ks[i] = 0.0005f + (float) (next_u32() % 1000u) * 2e-6f;
        vs[i] = 0.0005f + (float) (next_u32() % 1000u) * 2e-6f;
    }
    if (sh->trend) {
        /* |score| grows from ~0.2 to ~35 along the context: rising for the
         * heads whose Q agrees with the pattern, falling for the others. */
        const size_t row = sh->n_kv_heads * sh->head_dim;
        for (size_t s = 1; s < sh->n_kv; s++) {
            memcpy(k + s * row, k, row);
        }
        for (size_t s = 0; s < sh->n_kv; s++) {
            for (size_t h = 0; h < sh->n_kv_heads; h++) {
                ks[s * sh->n_kv_heads + h] = 0.0005f + 0.1f * (float) s / (float) sh->n_kv;
            }
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
    const size_t q_offset = sh->n_kv - sh->n_q;
    attention_int8_via_buffers(sh->n_q,
                               sh->n_q_heads,
                               sh->head_dim,
                               sh->n_kv,
                               sh->n_kv_heads,
                               q_offset,
                               sh->window,
                               q,
                               k,
                               ks,
                               v,
                               vs,
                               out);
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
    const double tol = 1e-5 * scale + 1e-7;
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
#if defined(_OPENMP)
    /* Same bits whatever the team: each head is one thread's sequential work. */
    const int team = omp_get_max_threads();
    omp_set_num_threads(1);
    attention_int8_via_buffers(sh->n_q,
                               sh->n_q_heads,
                               sh->head_dim,
                               sh->n_kv,
                               sh->n_kv_heads,
                               q_offset,
                               sh->window,
                               q,
                               k,
                               ks,
                               v,
                               vs,
                               out1);
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
    printf("  n_q=%-2zu heads=%2zu/%zu hd=%-3zu n_kv=%-5zu window=%-3zu%s max|d|/scale %.2e\n",
           sh->n_q,
           sh->n_q_heads,
           sh->n_kv_heads,
           sh->head_dim,
           sh->n_kv,
           sh->window,
           sh->trend ? " trend" : "",
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
    return fails;
}

int main(void) {
    static const struct shape SHAPES[] = {
            {1, 16, 8, 128, 512, 0, false},    /* 2 heads per pass (head_dim 128) */
            {16, 16, 8, 128, 700, 0, false},   /* ... prefill chunk */
            {16, 16, 8, 128, 200, 0, false},   /* ... context inside one block */
            {16, 24, 8, 128, 300, 0, false},   /* 3 heads per pass (head_dim 128) */
            {1, 32, 8, 64, 512, 0, false},     /* 4 heads per pass, decode */
            {16, 32, 8, 64, 2100, 0, false},   /* ... prefill chunk */
            {2, 16, 8, 64, 1600, 0, false},    /* 2 per pass: 1.6 MB of K/V */
            {4, 15, 5, 64, 2600, 0, false},    /* 3 per pass: 1.6 MB of K/V */
            {9, 15, 5, 64, 300, 0, false},     /* one-head loop: 192 KB of K/V */
            {16, 8, 1, 256, 600, 0, false},    /* MQA prefill: 4 per pass */
            {1, 8, 1, 256, 600, 0, false},     /* MQA decode: one-head loop */
            {3, 4, 4, 64, 100, 0, false},      /* MHA: one-head loop */
            {16, 16, 8, 128, 900, 256, false}, /* sliding window, grouped */
            {4, 32, 8, 64, 2500, 512, false},  /* sliding window, grouped */
            {1, 32, 8, 64, 3000, 0, true},     /* trending scores, 6 blocks */
            {8, 16, 8, 128, 2000, 0, true},    /* ... prefill chunk */
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof SHAPES / sizeof *SHAPES; i++) {
        fails += check_shape(&SHAPES[i]);
    }
    if (fails != 0) {
        fprintf(stderr, "FAIL: %d check(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: INT8 attention matches the double reference on every path\n");
    return GEIST_TEST_PASS;
}
