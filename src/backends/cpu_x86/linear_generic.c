/*
 * src/backends/cpu_x86/linear_generic.c — cpu_x86's linear for every dtype
 * that has no native x86 kernel.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * These dtypes (Q8_0, Q4_0, Q4_1, Q5_K, Q3_K, the IQ formats, TQ2_0, PQ2_0,
 * BF16, and F16 prefill) used to stay on cpu_scalar's resolver: the
 * correctness oracle, which allocates a row buffer on the heap per call,
 * dequantizes into it and dots in double, on one thread. On x86 that was the
 * production path — 225 heap allocations and no thread scaling per decoded
 * token on a Q8_0 model (#410, #504).
 *
 * This is the floor, not the goal: each OpenMP thread dequantizes its rows
 * with the format's own row decoder (quant.h) into a private row of the
 * calling thread's workspace (L1-resident for any realistic n_in), then dots
 * it in fp32 with AVX2/FMA. The M>1 path dequantizes each weight row once,
 * four at a time, and dots them against all m activation rows in blocks of
 * 4 rows by 3 tokens. No heap allocation once the workspace has grown to
 * the shape. A dtype that earns a native int8 kernel
 * moves off this path; nothing else changes for it.
 *
 * Numerics: fp32 accumulation in a fixed per-dot order — 4 x 8 lanes for one
 * token, 1 x 8 lanes per dot for M>1 — then a fixed reduction, so a result
 * depends neither on the thread count nor, for M>1, on the block it lands
 * in. It differs from cpu_scalar's double accumulation by float rounding
 * only — the dequantized weights are the same bits.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_generic.h"

#include "backend_state.h"

#include "checked.h"
#include "linear_ref.h"
#include "quant.h"

#include <geist_backend.h>
#include <geist_types.h>

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

typedef void (*row_dequant_fn)(size_t n, const void *src, float *out);

/* F16C is part of the x86-64-v3 baseline this backend builds for. */
static void f16_row(size_t n, const void *src, float *out) {
    const uint16_t *h = (const uint16_t *) src;
    size_t          i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (h + i))));
    }
    for (; i < n; i++) {
        out[i] = _cvtsh_ss(h[i]);
    }
}

static void bf16_row(size_t n, const void *src, float *out) {
    const uint16_t *h = (const uint16_t *) src;
    for (size_t i = 0; i < n; i++) {
        const uint32_t b = (uint32_t) h[i] << 16;
        memcpy(&out[i], &b, sizeof b);
    }
}

static row_dequant_fn row_dequant_for(uint16_t dtype) {
    switch ((enum geist_dtype) dtype) {
    case GEIST_DTYPE_Q4_0:
        return dequant_q4_0_row;
    case GEIST_DTYPE_Q4_1:
        return dequant_q4_1_row;
    case GEIST_DTYPE_Q8_0:
        return dequant_q8_0_row;
    case GEIST_DTYPE_Q3_K:
        return dequant_q3_K_row;
    case GEIST_DTYPE_Q4_K:
        return dequant_q4_K_row;
    case GEIST_DTYPE_Q5_K:
        return dequant_q5_K_row;
    case GEIST_DTYPE_Q6_K:
        return dequant_q6_K_row;
    case GEIST_DTYPE_IQ2_S:
        return dequant_iq2_s_row;
    case GEIST_DTYPE_IQ3_S:
        return dequant_iq3_s_row;
    case GEIST_DTYPE_IQ4_NL:
        return dequant_iq4_nl_row;
    case GEIST_DTYPE_IQ4_XS:
        return dequant_iq4_xs_row;
    case GEIST_DTYPE_TQ2_0:
        return dequant_tq2_0_row;
    case GEIST_DTYPE_PQ2_0:
        return dequant_pq2_0_row;
    case GEIST_DTYPE_F16:
        return f16_row;
    case GEIST_DTYPE_BF16:
        return bf16_row;
    default:
        return nullptr;
    }
}

/* 4 x 8 independent FMA lanes (FMA latency, not throughput, bounds a single
 * accumulator), reduced in a fixed order. `row` is private workspace, so it
 * never aliases `x`. */
static inline float dot_f32(size_t n, const float *restrict x, const float *restrict row) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    size_t i  = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(row + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(row + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(row + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(row + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(row + i), s0);
    }
    const __m256 s  = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128       s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4              = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4              = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    float acc       = _mm_cvtss_f32(s4);
    for (; i < n; i++) {
        acc += x[i] * row[i];
    }
    return acc;
}

static inline size_t team_max(void) {
#if defined(_OPENMP)
    return (size_t) omp_get_max_threads();
#else
    return 1;
#endif
}

static inline size_t team_id(void) {
#if defined(_OPENMP)
    return (size_t) omp_get_thread_num();
#else
    return 0;
#endif
}

/* Everything a call needs, or false with nothing acquired. per_thread
 * dequantized rows per possible thread, each padded to a 64-byte multiple
 * so no two threads' rows share a cache line. */
struct generic_plan {
    row_dequant_fn deq;
    size_t         row_bytes;
    size_t         stride; /* floats between thread rows */
    float         *rows;
};

[[nodiscard]] static bool plan_call(size_t                     per_thread,
                                    const struct geist_weight *w,
                                    struct geist_backend      *be,
                                    struct generic_plan       *p) {
    const size_t n_in = (size_t) w->n_in;
    p->deq            = row_dequant_for(w->dtype);
    if (p->deq == nullptr || be == nullptr || be->state == nullptr ||
        quant_raw_bytes((enum geist_dtype) w->dtype, n_in, &p->row_bytes)) {
        return false;
    }
    p->stride    = (n_in + 15u) & ~(size_t) 15u;
    size_t bytes = 0;
    if (ckd_mul(&bytes, team_max(), p->stride) || ckd_mul(&bytes, bytes, per_thread) ||
        ckd_mul(&bytes, bytes, sizeof(float))) {
        return false;
    }
    struct cpu_x86_workspace *ws =
            cpu_x86_ws_acquire_mN((struct cpu_x86_state *) be->state, 0, 0, 0, bytes);
    if (ws == nullptr) {
        return false;
    }
    p->rows = (float *) ws->mN_aux;
    return true;
}

static void cpu_x86_linear_generic_m1(const float               *x,
                                      const struct geist_weight *w,
                                      struct geist_backend      *be,
                                      float                     *y) {
    const size_t        n_in  = (size_t) w->n_in;
    const size_t        n_out = (size_t) w->n_out;
    struct generic_plan p;
    if (!plan_call(1, w, be, &p)) {
        geist_linear_ref(1, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    const uint8_t *raw = (const uint8_t *) w->raw;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        float *row = p.rows + team_id() * p.stride;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t j = 0; j < n_out; j++) {
            p.deq(n_in, raw + j * p.row_bytes, row);
            y[j] = dot_f32(n_in, x, row);
        }
    }
}

/* The M>1 dots, a block of MN_ROWS dequantized weight rows against
 * MN_TOKENS activation rows at a time: 12 accumulators, and per 8
 * elements 7 loads for 12 FMAs where a dot at a time loads twice per FMA
 * and waits on the load ports. AVX2 has 16 ymm: 12 accumulators, the 3
 * activation vectors, a weight vector. Each dot is one 8-lane accumulator
 * in k order, reduced in a fixed order, then the scalar tail: its result
 * does not depend on the block it lands in, nor on the thread count. */
constexpr size_t MN_ROWS   = 4;
constexpr size_t MN_TOKENS = 3;

static inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s        = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s        = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* y[t * n_out + r] = row r . x row t for r < nr <= MN_ROWS, t < nt <=
 * MN_TOKENS; rows `stride` floats apart, x rows n apart. Inlined with
 * constant nr and nt, so that the accumulators stay in registers. */
[[gnu::always_inline]] static inline void dots_block(size_t       nr,
                                                     size_t       nt,
                                                     size_t       n,
                                                     size_t       stride,
                                                     const float *rows,
                                                     const float *x,
                                                     size_t       n_out,
                                                     float       *y) {
    __m256 acc[MN_ROWS][MN_TOKENS];
    for (size_t r = 0; r < nr; r++) {
        for (size_t t = 0; t < nt; t++) {
            acc[r][t] = _mm256_setzero_ps();
        }
    }
    size_t k = 0;
    for (; k + 8 <= n; k += 8) {
        __m256 xv[MN_TOKENS];
        for (size_t t = 0; t < nt; t++) {
            xv[t] = _mm256_loadu_ps(x + t * n + k);
        }
        for (size_t r = 0; r < nr; r++) {
            const __m256 wv = _mm256_loadu_ps(rows + r * stride + k);
            for (size_t t = 0; t < nt; t++) {
                acc[r][t] = _mm256_fmadd_ps(xv[t], wv, acc[r][t]);
            }
        }
    }
    for (size_t r = 0; r < nr; r++) {
        for (size_t t = 0; t < nt; t++) {
            float s = hsum8(acc[r][t]);
            for (size_t kk = k; kk < n; kk++) {
                s += x[t * n + kk] * rows[r * stride + kk];
            }
            y[t * n_out + r] = s;
        }
    }
}

/* All m tokens against nr <= MN_ROWS rows: blocks of MN_TOKENS, then the
 * tail. */
static void dots_rows(size_t       nr,
                      size_t       m,
                      size_t       n,
                      size_t       stride,
                      const float *rows,
                      const float *x,
                      size_t       n_out,
                      float       *y) {
    size_t t = 0;
    for (; t + MN_TOKENS <= m; t += MN_TOKENS) {
        switch (nr) {
        case 4:
            dots_block(4, 3, n, stride, rows, x + t * n, n_out, y + t * n_out);
            break;
        case 3:
            dots_block(3, 3, n, stride, rows, x + t * n, n_out, y + t * n_out);
            break;
        case 2:
            dots_block(2, 3, n, stride, rows, x + t * n, n_out, y + t * n_out);
            break;
        default:
            dots_block(1, 3, n, stride, rows, x + t * n, n_out, y + t * n_out);
            break;
        }
    }
    for (; t < m; t++) {
        switch (nr) {
        case 4:
            dots_block(4, 1, n, stride, rows, x + t * n, n_out, y + t * n_out);
            break;
        case 3:
            dots_block(3, 1, n, stride, rows, x + t * n, n_out, y + t * n_out);
            break;
        case 2:
            dots_block(2, 1, n, stride, rows, x + t * n, n_out, y + t * n_out);
            break;
        default:
            dots_block(1, 1, n, stride, rows, x + t * n, n_out, y + t * n_out);
            break;
        }
    }
}

static void cpu_x86_linear_generic_mN(size_t                     m,
                                      const float               *x,
                                      const struct geist_weight *w,
                                      struct geist_backend      *be,
                                      float                     *y) {
    const size_t        n_in  = (size_t) w->n_in;
    const size_t        n_out = (size_t) w->n_out;
    struct generic_plan p;
    if (!plan_call(MN_ROWS, w, be, &p)) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    const uint8_t *raw    = (const uint8_t *) w->raw;
    const size_t   blocks = (n_out + MN_ROWS - 1) / MN_ROWS;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        float *rows = p.rows + team_id() * MN_ROWS * p.stride;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t b = 0; b < blocks; b++) {
            const size_t j0 = b * MN_ROWS;
            const size_t nr = n_out - j0 < MN_ROWS ? n_out - j0 : MN_ROWS;
            for (size_t r = 0; r < nr; r++) {
                p.deq(n_in, raw + (j0 + r) * p.row_bytes, rows + r * p.stride);
            }
            dots_rows(nr, m, n_in, p.stride, rows, x, n_out, y + j0);
        }
    }
}

bool cpu_x86_linear_generic_bind(struct geist_weight *w) {
    size_t row_bytes = 0;
    if (w == nullptr || w->n_in <= 0 || row_dequant_for(w->dtype) == nullptr ||
        quant_raw_bytes((enum geist_dtype) w->dtype, (size_t) w->n_in, &row_bytes)) {
        return false;
    }
    w->linear_m1 = cpu_x86_linear_generic_m1;
    w->linear_mN = cpu_x86_linear_generic_mN;
    return true;
}
