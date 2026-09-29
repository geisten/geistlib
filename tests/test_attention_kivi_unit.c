/*
 * test_attention_kivi_unit — attention_kivi_via_buffers, the KIVI cache's
 * attention, against a double-precision reference, with the cache filled
 * the way a session fills it: every position through the FP32 residual,
 * and each full group of KIVI_K_GROUP_SIZE positions drained into the 2-bit
 * cache by kivi_drain_one_layer (as transformer_kivi_drain_full does).
 *
 * The reference sees what the cache holds: drained positions' K and V
 * dequantized by the kivi.h primitives, one KV head at a time, whatever
 * the layout of the drained groups; residual positions in FP32. Its
 * softmax and V sum run in double. Shapes: MQA and GQA, decode and a
 * prefill chunk, a context inside the first group and contexts over
 * several groups with a residual tail, a sliding window, and peaked scores
 * (queries scaled so that scores sit hundreds apart). The output starts as
 * NaN and is compared by bit pattern (-ffast-math build), so every row must
 * be written.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "src/archs/transformer/forward/internal.h"
#include "src/archs/transformer/forward.h"

#include "kivi.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
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
    float  qscale; /* 1: scores within a few units; 300: hundreds apart */
};

/* What the cache holds for K and V at every position, [n_kv][n_kv_heads][hd]:
 * drained positions (the first `drained`) dequantized per KV head, the rest
 * as given. */
static void cache_values(const struct shape *sh,
                         size_t              drained,
                         const float        *k,
                         const float        *v,
                         float              *kd,
                         float              *vd) {
    const size_t R = KIVI_K_GROUP_SIZE, hd = sh->head_dim, kvh = sh->n_kv_heads;
    float       *grp = xmalloc(R * hd * sizeof *grp);
    uint8_t     *q   = xmalloc(R * hd / 4);
    float       *sc  = xmalloc(hd * sizeof *sc);
    float       *ze  = xmalloc(hd * sizeof *ze);
    memcpy(kd, k, sh->n_kv * kvh * hd * sizeof *kd);
    memcpy(vd, v, sh->n_kv * kvh * hd * sizeof *vd);
    for (size_t g = 0; g < drained / R; g++) {
        for (size_t h = 0; h < kvh; h++) {
            for (size_t t = 0; t < R; t++) {
                memcpy(grp + t * hd, k + ((g * R + t) * kvh + h) * hd, hd * sizeof *grp);
            }
            kivi_pack_k_group(R, hd, grp, q, sc, ze);
            kivi_unpack_k_group(R, hd, q, sc, ze, grp);
            for (size_t t = 0; t < R; t++) {
                memcpy(kd + ((g * R + t) * kvh + h) * hd, grp + t * hd, hd * sizeof *grp);
            }
        }
    }
    for (size_t s = 0; s < drained; s++) {
        for (size_t h = 0; h < kvh; h++) {
            float vs = 0.0f, vz = 0.0f;
            kivi_pack_v_row(hd, v + (s * kvh + h) * hd, q, &vs, &vz);
            kivi_unpack_v_row(hd, q, vs, vz, vd + (s * kvh + h) * hd);
        }
    }
    free(grp);
    free(q);
    free(sc);
    free(ze);
}

/* out[t,h,:] = softmax_s(q[t,h,:] . k[s,kv(h),:]) v[s,kv(h),:] over the
 * causal window, in double. */
static void reference(const struct shape *sh,
                      const float        *q,
                      const float        *k,
                      const float        *v,
                      double             *out,
                      double             *scores) {
    const size_t group = sh->n_q_heads / sh->n_kv_heads, hd = sh->head_dim;
    for (size_t t = 0; t < sh->n_q; t++) {
        const size_t q_pos = sh->n_kv - sh->n_q + t;
        const size_t s_lo  = sh->window > 0 && q_pos + 1 > sh->window ? q_pos + 1 - sh->window : 0;
        for (size_t h = 0; h < sh->n_q_heads; h++) {
            const size_t kv_h = h / group;
            const float *qv   = q + (t * sh->n_q_heads + h) * hd;
            double       mx   = -INFINITY;
            for (size_t s = s_lo; s <= q_pos; s++) {
                const float *kv  = k + (s * sh->n_kv_heads + kv_h) * hd;
                double       dot = 0.0;
                for (size_t i = 0; i < hd; i++) {
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
            double *o = out + (t * sh->n_q_heads + h) * hd;
            for (size_t i = 0; i < hd; i++) {
                o[i] = 0.0;
            }
            for (size_t s = s_lo; s <= q_pos; s++) {
                const float *vv = v + (s * sh->n_kv_heads + kv_h) * hd;
                for (size_t i = 0; i < hd; i++) {
                    o[i] += scores[s] / sum * vv[i];
                }
            }
        }
    }
}

static int check_shape(const struct shape *sh) {
    const size_t R = KIVI_K_GROUP_SIZE, hd = sh->head_dim, kvh = sh->n_kv_heads;
    const size_t nkv = sh->n_kv * kvh * hd, nq = sh->n_q * sh->n_q_heads * hd;
    const size_t groups = sh->n_kv / R + 1;
    float       *k      = xmalloc(nkv * sizeof *k);
    float       *v      = xmalloc(nkv * sizeof *v);
    float       *kd     = xmalloc(nkv * sizeof *kd);
    float       *vd     = xmalloc(nkv * sizeof *vd);
    float       *k_res  = xmalloc(nkv * sizeof *k_res);
    float       *v_res  = xmalloc(nkv * sizeof *v_res);
    uint8_t     *k_q4   = xmalloc(nkv / 4 + 1);
    uint8_t     *v_q4   = xmalloc(nkv / 4 + 1);
    float       *k_sc   = xmalloc(groups * kvh * hd * sizeof *k_sc);
    float       *k_ze   = xmalloc(groups * kvh * hd * sizeof *k_ze);
    float       *v_sc   = xmalloc(sh->n_kv * kvh * sizeof *v_sc);
    float       *v_ze   = xmalloc(sh->n_kv * kvh * sizeof *v_ze);
    float       *q      = xmalloc(nq * sizeof *q);
    float       *out    = xmalloc(nq * sizeof *out);
    float       *scores = xmalloc(sh->n_kv * sizeof *scores);
    double      *ref    = xmalloc(nq * sizeof *ref);
    double      *rsc    = xmalloc(sh->n_kv * sizeof *rsc);
    for (size_t i = 0; i < nkv; i++) {
        k[i] = urand();
        v[i] = urand();
    }
    const float qs = sh->qscale / sqrtf((float) hd);
    for (size_t i = 0; i < nq; i++) {
        q[i]   = urand() * qs;
        out[i] = NAN;
    }

    /* The session's path: all positions through the residual, full groups
     * drained, the survivors moved to the front of the residual. */
    memcpy(k_res, k, nkv * sizeof *k_res);
    memcpy(v_res, v, nkv * sizeof *v_res);
    size_t drained = 0, residual = sh->n_kv;
    while (residual >= R) {
        kivi_drain_one_layer(
                drained, residual, R, hd, kvh, k_res, v_res, k_q4, v_q4, k_sc, k_ze, v_sc, v_ze);
        drained += R;
        residual -= R;
    }
    attention_kivi_via_buffers(sh->n_q,
                               sh->n_q_heads,
                               hd,
                               sh->n_kv,
                               kvh,
                               sh->n_kv - sh->n_q,
                               sh->window,
                               drained,
                               R,
                               q,
                               k_q4,
                               k_sc,
                               k_ze,
                               v_q4,
                               v_sc,
                               v_ze,
                               k_res,
                               v_res,
                               scores,
                               out);

    cache_values(sh, drained, k, v, kd, vd);
    reference(sh, q, kd, vd, ref, rsc);
    double scale = 0.0, worst = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < nq; i++) {
        scale = fabs(ref[i]) > scale ? fabs(ref[i]) : scale;
    }
    for (size_t i = 0; i < nq; i++) {
        if (!finite_f32(out[i])) {
            nonfinite++;
            continue;
        }
        const double e = fabs((double) out[i] - ref[i]);
        worst          = e > worst ? e : worst;
    }
    /* Scores in the hundreds carry float rounding of about 1e-5 into the
     * weights in any float kernel. */
    const double tol = (sh->qscale > 1.0f ? 1e-4 : 1e-5) * scale + 1e-7;
    char         what[200];
    snprintf(what,
             sizeof what,
             "n_q %zu, heads %zu/%zu, head_dim %zu, n_kv %zu (%zu drained), window %zu, qscale "
             "%g: %zu non-finite, max error %.3g (limit %.3g)",
             sh->n_q,
             sh->n_q_heads,
             kvh,
             hd,
             sh->n_kv,
             drained,
             sh->window,
             (double) sh->qscale,
             nonfinite,
             worst,
             tol);
    free(k);
    free(v);
    free(kd);
    free(vd);
    free(k_res);
    free(v_res);
    free(k_q4);
    free(v_q4);
    free(k_sc);
    free(k_ze);
    free(v_sc);
    free(v_ze);
    free(q);
    free(out);
    free(scores);
    free(ref);
    free(rsc);
    return geist_expect(nonfinite == 0 && worst <= tol, what);
}

int main(void) {
    static const struct shape SHAPES[] = {
            {1, 8, 1, 64, 100, 0, 1.0f},    /* MQA decode inside the first group */
            {1, 8, 1, 64, 300, 0, 1.0f},    /* MQA decode, two groups drained */
            {1, 4, 2, 64, 300, 0, 1.0f},    /* GQA decode, two groups drained */
            {1, 8, 4, 128, 520, 0, 1.0f},   /* GQA decode, four groups drained */
            {16, 8, 2, 64, 400, 0, 1.0f},   /* GQA prefill chunk */
            {4, 8, 1, 256, 700, 200, 1.0f}, /* MQA, head_dim 256, sliding window */
            {1, 8, 1, 64, 600, 0, 300.0f},  /* MQA, peaked scores */
            {4, 8, 2, 128, 600, 0, 300.0f}, /* GQA, peaked scores */
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof SHAPES / sizeof SHAPES[0]; i++) {
        fails += check_shape(&SHAPES[i]);
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: KIVI attention matches the double reference of the cache it reads\n");
    return GEIST_TEST_PASS;
}
