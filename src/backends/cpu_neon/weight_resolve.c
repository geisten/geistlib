/*
 * src/backends/cpu_neon/weight_resolve.c — load-time kernel resolution.
 *
 * Layer: BACKEND (cpu_neon). Implements geist_backend_vtbl::resolve_weight
 * for the new geist_weight flow (refactor v2, P1.1.b).
 *
 * Pre-resolves the (M=1 decode-style, M>1 prefill-style) kernel pair for
 * each weight tensor at model load. The forward loop then calls
 *
 *     w->linear_mN(x, w, m, y);
 *
 * without dtype dispatch or vtable indirection.
 *
 * Trampolines: thin wrappers around the existing kernels in
 * src/backends/common/gguf_quant.c that translate the new signature
 * (struct geist_weight *) back into the kernels' (raw, n_in, n_out)
 * style. Once all callers are on the new flow (P1.1.c..d), the kernels
 * can be inlined here and the wrapper indirection removed.
 *
 * Supported dtypes (full M=1 + M>1 coverage):
 *   Q3_K, Q4_K, Q6_K, IQ2_S, IQ3_S
 *
 * Partial coverage (M=1 only; M>1 falls through to legacy path):
 *   Q8_0
 *
 * Not supported yet (returns GEIST_E_UNSUPPORTED → legacy path runs):
 *   Q5_K, F32 dense, F16 dense, BF16 dense, IQ2_XXS, IQ2_M (mixed)
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"
#include "parallel.h"
#include "tl1.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "quant.h"
#include "heap.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* GEIST_PROFILE_QUANT=1: separate the W4A8/W6A8 activation-quantization wall
 * time from the matmul wall time in the m>1 (prefill) path, to decide whether a
 * quantize-once-per-activation refactor is worth it. Calls happen one matmul at
 * a time from the (single-threaded) forward loop, so plain accumulators suffice. */
static uint64_t g_qprof_quant_ns = 0;
static uint64_t g_qprof_mm_ns    = 0;
static int      g_qprof_state    = -1;
static void     qprof_print(void) {
    const double q   = (double) g_qprof_quant_ns / 1e6;
    const double mm  = (double) g_qprof_mm_ns / 1e6;
    const double tot = q + mm;
    fprintf(stderr,
            "quant-profile: activation-quant %.2f ms (%.1f%%), matmul %.2f ms (%.1f%%)\n",
            q,
            tot > 0 ? 100.0 * q / tot : 0.0,
            mm,
            tot > 0 ? 100.0 * mm / tot : 0.0);
}
static inline bool qprof_on(void) {
    if (g_qprof_state < 0) {
        const char *e = getenv("GEIST_PROFILE_QUANT");
        g_qprof_state = (e != nullptr && e[0] == '1') ? 1 : 0;
        if (g_qprof_state) {
            atexit(qprof_print);
        }
    }
    return g_qprof_state != 0;
}
static inline uint64_t qprof_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

/* Dense F32 projections (e.g. the PLE model_proj path) route through the
 * geist_gemm facade: Accelerate/AMX on Mac, OpenBLAS on Linux, or the native
 * fallback in a BLAS-free build. */
#include "geist_gemm.h"

/* TQ2_0 q8a scratch storage lives on the backend's cpu_neon_workspace
 * (see internal.h). Kernels reach it via the `be` parameter; the
 * `ws->m{1,N}_*` field names below are populated/grown by the kernels
 * exactly the way the previous file-scope _Thread_local globals were.
 * Lifetime is bound to the backend struct, not the OMP runtime — so a
 * model unload/reload cycle in a long-running process actually returns
 * the memory at backend destroy. */

/* ---- M=1 (decode) trampolines ---------------------------------------- */

static void cpu_neon_w_q3k_m1(const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    (void) be;
    linear_q3k_decode_w3a8(x, w->raw, (size_t) w->n_in, (size_t) w->n_out, y);
}

static void cpu_neon_w_q4k_m1(const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    (void) be;
    /* The raw SDOT decode GEMV beats the predecoded-block path here: the latter
     * re-quantizes + allocates per call and its m=1 form is a GEMM kernel, not a
     * tuned GEMV (measured ~21 vs ~33 tg128 on M1 Max). Keep raw. */
    linear_q4k_decode_w4a8(x, w->raw, (size_t) w->n_in, (size_t) w->n_out, y);
}

static void cpu_neon_w_q4k_pair_m1(const float               *x,
                                   const struct geist_weight *w0,
                                   const struct geist_weight *w1,
                                   struct geist_backend      *be,
                                   float                     *y0,
                                   float                     *y1) {
    (void) be;
    if (w0->n_in != w1->n_in) {
        return;
    }
    linear_q4k_decode_w4a8_pair(
            x, w0->raw, w1->raw, (size_t) w0->n_in, (size_t) w0->n_out, (size_t) w1->n_out, y0, y1);
}

static void cpu_neon_w_q6k_m1(const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    (void) be;
    if (w->backend_layout == GEIST_W_LAYOUT_Q6_K_X8_GEMV && w->aux_fp32 != nullptr) {
        linear_q6k_decode_w6a8_x8(x, w->aux_fp32, (size_t) w->n_in, (size_t) w->n_out, y);
        return;
    }
    linear_q6k_decode_w6a8(x, w->raw, (size_t) w->n_in, (size_t) w->n_out, y);
}

static void cpu_neon_w_q8_0_m1(const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    (void) be;
    linear_q8_0_decode_w8a8(x, w->raw, (size_t) w->n_in, (size_t) w->n_out, y);
}

static void cpu_neon_w_iq2s_m1(const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    (void) be;
    linear_iq2s_decode_w2a8(x, w->raw, (size_t) w->n_in, (size_t) w->n_out, y);
}

static void cpu_neon_w_iq3s_m1(const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    (void) be;
    linear_iq3s_decode_w3a8(x, w->raw, (size_t) w->n_in, (size_t) w->n_out, y);
}

/* F32 dense (P1.1.e): cblas-backed SGEMV / SGEMM. Row-major weight is
 * [n_out, n_in]; we compute y = W @ x as a sgemv with TransA=NoTrans
 * since the row-major layout already has the right shape. */
static void cpu_neon_w_f32_m1(const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    (void) be;
    geist_sgemv(GEIST_OP_N,
                (int) w->n_out,
                (int) w->n_in,
                1.0f,
                (const float *) w->raw,
                (int) w->n_in,
                x,
                1,
                0.0f,
                y,
                1);
}

static void cpu_neon_w_f32_mN(const float               *x,
                              const struct geist_weight *w,
                              size_t                     m,
                              struct geist_backend      *be,
                              float                     *y) {
    (void) be;
    /* Y [m, n_out] = X [m, n_in] @ W^T   (W row-major [n_out, n_in]). */
    geist_sgemm(GEIST_OP_N,
                GEIST_OP_T,
                (int) m,
                (int) w->n_out,
                (int) w->n_in,
                1.0f,
                x,
                (int) w->n_in,
                (const float *) w->raw,
                (int) w->n_in,
                0.0f,
                y,
                (int) w->n_out);
}

/* ---- M>1 (prefill) trampolines --------------------------------------- */

static void cpu_neon_w_q3k_mN(const float               *x,
                              const struct geist_weight *w,
                              size_t                     m,
                              struct geist_backend      *be,
                              float                     *y) {
    (void) be;
    linear_q3k_w3a8_prefill(x, w->raw, m, (size_t) w->n_in, (size_t) w->n_out, y);
}

static bool cpu_neon_qk_mN_workspace_prepare(struct cpu_neon_workspace *ws, size_t m, size_t n_in) {
    const size_t xq_need    = m * n_in;
    const size_t sum_need   = m * (n_in / 32);
    const size_t scale_need = m * (n_in / Q4_K_BLOCK_ELEMS);
    if (ws->qk_mN_xq_cap < xq_need) {
        safe_free((void **) &ws->qk_mN_xq);
        ws->qk_mN_xq = heap_alloc_array_aligned(int8_t, xq_need);
        if (ws->qk_mN_xq == nullptr) {
            ws->qk_mN_xq_cap = 0;
            return false;
        }
        ws->qk_mN_xq_cap = xq_need;
    }
    if (ws->qk_mN_sum32_cap < sum_need) {
        safe_free((void **) &ws->qk_mN_sum32);
        ws->qk_mN_sum32 = heap_alloc_array_aligned(int32_t, sum_need);
        if (ws->qk_mN_sum32 == nullptr) {
            ws->qk_mN_sum32_cap = 0;
            return false;
        }
        ws->qk_mN_sum32_cap = sum_need;
    }
    if (ws->qk_mN_sc_cap < scale_need) {
        safe_free((void **) &ws->qk_mN_sc);
        ws->qk_mN_sc = heap_alloc_array_aligned(float, scale_need);
        if (ws->qk_mN_sc == nullptr) {
            ws->qk_mN_sc_cap = 0;
            return false;
        }
        ws->qk_mN_sc_cap = scale_need;
    }
    return true;
}

static void
cpu_neon_qk_mN_quantize_x(struct cpu_neon_workspace *ws, const float *x, size_t m, size_t n_in) {
#if defined(_OPENMP)
    if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < m; i++) {
            ws->qk_mN_sc[i] = quantize_x_for_q4k(
                    x + i * n_in, n_in, ws->qk_mN_xq + i * n_in, ws->qk_mN_sum32 + i * (n_in / 32));
        }
        return;
    }
#pragma omp parallel for schedule(static) if (m >= 4)
#endif
    for (size_t i = 0; i < m; i++) {
        ws->qk_mN_sc[i] = quantize_x_for_q4k(
                x + i * n_in, n_in, ws->qk_mN_xq + i * n_in, ws->qk_mN_sum32 + i * (n_in / 32));
    }
}

static void cpu_neon_qk_mN_quantize_x_blocks(struct cpu_neon_workspace *ws,
                                             const float               *x,
                                             size_t                     m,
                                             size_t                     n_in) {
    const size_t n_blocks = n_in / Q4_K_BLOCK_ELEMS;
#if defined(_OPENMP)
    if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < m; i++) {
            quantize_x_for_q4k_blocks(x + i * n_in,
                                      n_in,
                                      ws->qk_mN_xq + i * n_in,
                                      ws->qk_mN_sum32 + i * (n_in / 32),
                                      ws->qk_mN_sc + i * n_blocks);
        }
        return;
    }
#pragma omp parallel for schedule(static) if (m >= 4)
#endif
    for (size_t i = 0; i < m; i++) {
        quantize_x_for_q4k_blocks(x + i * n_in,
                                  n_in,
                                  ws->qk_mN_xq + i * n_in,
                                  ws->qk_mN_sum32 + i * (n_in / 32),
                                  ws->qk_mN_sc + i * n_blocks);
    }
}

static bool q4k_weight_predecoded(const struct geist_weight *w) {
    return w != nullptr &&
           (w->backend_layout == GEIST_W_LAYOUT_Q4_K_PREDECODE ||
            w->backend_layout == GEIST_W_LAYOUT_Q4_K_PREDECODE_NTILE4) &&
           w->aux_fp32 != nullptr;
}

static bool q4k_weight_ntile4(const struct geist_weight *w) {
    return w != nullptr && w->backend_layout == GEIST_W_LAYOUT_Q4_K_PREDECODE_NTILE4;
}

static bool q6k_weight_ntile4(const struct geist_weight *w) {
    return w != nullptr && w->backend_layout == GEIST_W_LAYOUT_Q6_K_PREDECODE_NTILE4 &&
           w->aux_fp32 != nullptr;
}

static bool q6k_weight_ntile4_stream(const struct geist_weight *w) {
    return w != nullptr && w->backend_layout == GEIST_W_LAYOUT_Q6_K_PREDECODE_NTILE4_STREAM &&
           w->aux_fp32 != nullptr;
}

static void cpu_neon_q4k_run_prequantized(const struct cpu_neon_state     *st,
                                          const struct cpu_neon_workspace *ws,
                                          const struct geist_weight       *w,
                                          bool                             use_block_scales,
                                          size_t                           m,
                                          float                           *y) {
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if (q4k_weight_predecoded(w)) {
        if (q4k_weight_ntile4(w)) {
            /* mtile8_ntile4 = same packed format, wider M-tile (8 rows
             * per inner iter). Falls back to mtile4_ntile4_packed when
             * m<8 internally. */
            linear_q4k_w4a8_prefill_predecoded_mtile8_ntile4_packed(
                    ws->qk_mN_xq, ws->qk_mN_sc, ws->qk_mN_sum32, m, w->aux_fp32, n_in, n_out, y);
        } else if (use_block_scales) {
            linear_q4k_w4a8_prefill_predecoded_mtile4_bscale(
                    ws->qk_mN_xq, ws->qk_mN_sc, ws->qk_mN_sum32, m, w->aux_fp32, n_in, n_out, y);
        } else if (st->policy.q4k_mtile_prefill) {
            if (st->policy.q4k_ntile_prefill) {
                linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4(ws->qk_mN_xq,
                                                                 ws->qk_mN_sc,
                                                                 ws->qk_mN_sum32,
                                                                 m,
                                                                 w->aux_fp32,
                                                                 n_in,
                                                                 n_out,
                                                                 y);
            } else {
                /* mtile8 is bit-identical to mtile4 and falls back to
                 * mtile4 for m<8; small win on Mac, larger expected
                 * on Pi5 where per-loop overhead is more visible. */
                linear_q4k_w4a8_prefill_predecoded_mtile8(ws->qk_mN_xq,
                                                          ws->qk_mN_sc,
                                                          ws->qk_mN_sum32,
                                                          m,
                                                          w->aux_fp32,
                                                          n_in,
                                                          n_out,
                                                          y);
            }
        } else {
            linear_q4k_w4a8_prefill_predecoded(
                    ws->qk_mN_xq, ws->qk_mN_sc, ws->qk_mN_sum32, m, w->aux_fp32, n_in, n_out, y);
        }
    } else {
        linear_q4k_w4a8_prefill_pre(
                ws->qk_mN_xq, ws->qk_mN_sc, ws->qk_mN_sum32, m, w->raw, n_in, n_out, y);
    }
}

/* SGEMM-prefill path (m ≥ threshold): dequant W tile-by-tile into a
 * workspace-resident fp32 scratch, call cblas_sgemm per tile.
 * Implementation defined further down (after DEQ_TILE_ROWS + dequant_tile).
 * Activation x is consumed as fp32 — no quantize_x_for_q4k. */
static bool
cpu_neon_dequant_w_workspace_prepare(struct cpu_neon_workspace *ws, size_t tile_rows, size_t n_in);
static void   cpu_neon_qk_sgemm_run(const float               *x,
                                    const struct geist_weight *w,
                                    size_t                     m,
                                    size_t                     tile_rows,
                                    float                     *tile_fp32,
                                    float                     *y);
static size_t qk_sgemm_tile_rows_for(const struct cpu_neon_state *st);

static void cpu_neon_w_q4k_mN(const float               *x,
                              const struct geist_weight *w,
                              size_t                     m,
                              struct geist_backend      *be,
                              float                     *y) {
    struct cpu_neon_state     *st   = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws   = &st->workspace;
    const size_t               n_in = (size_t) w->n_in;
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;

    if (st->policy.q4k_sgemm_prefill && m >= st->policy.qk_sgemm_threshold &&
        cpu_neon_dequant_w_workspace_prepare(ws, qk_sgemm_tile_rows_for(st), n_in)) {
        cpu_neon_qk_sgemm_run(x, w, m, qk_sgemm_tile_rows_for(st), ws->dequant_w_fp32, y);
        return;
    }

    if (!cpu_neon_qk_mN_workspace_prepare(ws, m, n_in))
        return;

    const bool use_block_scales = st->policy.q4k_mtile_prefill && st->policy.q4k_block_q8_prefill &&
                                  q4k_weight_predecoded(w) && !q4k_weight_ntile4(w);
    const bool qp               = qprof_on();
    uint64_t   t0               = qp ? qprof_now_ns() : 0;
    if (use_block_scales) {
        cpu_neon_qk_mN_quantize_x_blocks(ws, x, m, n_in);
    } else {
        cpu_neon_qk_mN_quantize_x(ws, x, m, n_in);
    }
    if (qp) {
        g_qprof_quant_ns += qprof_now_ns() - t0;
        t0 = qprof_now_ns();
    }
    cpu_neon_q4k_run_prequantized(st, ws, w, use_block_scales, m, y);
    if (qp) {
        g_qprof_mm_ns += qprof_now_ns() - t0;
    }
}

static void cpu_neon_w_q4k_pair_mN(const float               *x,
                                   const struct geist_weight *w0,
                                   const struct geist_weight *w1,
                                   size_t                     m,
                                   struct geist_backend      *be,
                                   float                     *y0,
                                   float                     *y1) {
    struct cpu_neon_state     *st   = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws   = &st->workspace;
    const size_t               n_in = (size_t) w0->n_in;
    if (m == 0 || m > GEIST_QUANT_M_CAP || w0->n_in != w1->n_in)
        return;

    /* SGEMM-prefill pair: reuse the same dequant_w_fp32 scratch for both
     * weights. Each gets its own dequant+sgemm tile-loop; the activation
     * x is shared and stays in L2 across the two sgemm calls. */
    if (st->policy.q4k_sgemm_prefill && m >= st->policy.qk_sgemm_threshold &&
        cpu_neon_dequant_w_workspace_prepare(ws, qk_sgemm_tile_rows_for(st), n_in)) {
        const size_t tile_rows = qk_sgemm_tile_rows_for(st);
        cpu_neon_qk_sgemm_run(x, w0, m, tile_rows, ws->dequant_w_fp32, y0);
        cpu_neon_qk_sgemm_run(x, w1, m, tile_rows, ws->dequant_w_fp32, y1);
        return;
    }

    if (!cpu_neon_qk_mN_workspace_prepare(ws, m, n_in))
        return;

    const bool use_block_scales = st->policy.q4k_mtile_prefill && st->policy.q4k_block_q8_prefill &&
                                  q4k_weight_predecoded(w0) && q4k_weight_predecoded(w1) &&
                                  !q4k_weight_ntile4(w0) && !q4k_weight_ntile4(w1);
    if (use_block_scales) {
        cpu_neon_qk_mN_quantize_x_blocks(ws, x, m, n_in);
    } else {
        cpu_neon_qk_mN_quantize_x(ws, x, m, n_in);
    }

    if (!use_block_scales && q4k_weight_ntile4(w0) && q4k_weight_ntile4(w1) &&
        w0->n_out == w1->n_out) {
        linear_q4k_w4a8_prefill_pair_predecoded_mtile4_ntile4_packed(ws->qk_mN_xq,
                                                                     ws->qk_mN_sc,
                                                                     ws->qk_mN_sum32,
                                                                     m,
                                                                     w0->aux_fp32,
                                                                     w1->aux_fp32,
                                                                     n_in,
                                                                     (size_t) w0->n_out,
                                                                     y0,
                                                                     y1);
    } else {
        cpu_neon_q4k_run_prequantized(st, ws, w0, use_block_scales, m, y0);
        cpu_neon_q4k_run_prequantized(st, ws, w1, use_block_scales, m, y1);
    }
}

static void cpu_neon_w_q6k_mN(const float               *x,
                              const struct geist_weight *w,
                              size_t                     m,
                              struct geist_backend      *be,
                              float                     *y) {
    struct cpu_neon_state     *st   = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws   = &st->workspace;
    const size_t               n_in = (size_t) w->n_in;
    if (m == 0 || m > GEIST_QUANT_M_CAP)
        return;

    /* SGEMM-prefill path (m ≥ threshold): dequant Q6_K + AMX SGEMM. */
    if (st->policy.q6k_sgemm_prefill && m >= st->policy.qk_sgemm_threshold &&
        cpu_neon_dequant_w_workspace_prepare(ws, qk_sgemm_tile_rows_for(st), n_in)) {
        cpu_neon_qk_sgemm_run(x, w, m, qk_sgemm_tile_rows_for(st), ws->dequant_w_fp32, y);
        return;
    }

    if (!cpu_neon_qk_mN_workspace_prepare(ws, m, n_in))
        return;
    const bool qp6 = qprof_on();
    uint64_t   t6  = qp6 ? qprof_now_ns() : 0;
    for (size_t i = 0; i < m; i++) {
        ws->qk_mN_sc[i] = quantize_x_int8_sym(x + i * n_in, n_in, ws->qk_mN_xq + i * n_in);
    }
    if (qp6) {
        g_qprof_quant_ns += qprof_now_ns() - t6;
        t6 = qprof_now_ns();
    }
    if (q6k_weight_ntile4_stream(w)) {
        linear_q6k_w6a8_prefill_predecoded_ntile4_stream(
                ws->qk_mN_xq, ws->qk_mN_sc, m, w->aux_fp32, n_in, (size_t) w->n_out, y);
    } else if (q6k_weight_ntile4(w)) {
        linear_q6k_w6a8_prefill_predecoded_ntile4(
                ws->qk_mN_xq, ws->qk_mN_sc, m, w->aux_fp32, n_in, (size_t) w->n_out, y);
    } else {
        linear_q6k_w6a8_prefill_pre(
                ws->qk_mN_xq, ws->qk_mN_sc, m, w->raw, n_in, (size_t) w->n_out, y);
    }
    if (qp6) {
        g_qprof_mm_ns += qprof_now_ns() - t6;
    }
}

static void cpu_neon_w_iq2s_mN(const float               *x,
                               const struct geist_weight *w,
                               size_t                     m,
                               struct geist_backend      *be,
                               float                     *y) {
    (void) be;
    linear_iq2s_w2a8_prefill(x, w->raw, m, (size_t) w->n_in, (size_t) w->n_out, y);
}

static void cpu_neon_w_iq3s_mN(const float               *x,
                               const struct geist_weight *w,
                               size_t                     m,
                               struct geist_backend      *be,
                               float                     *y) {
    (void) be;
    linear_iq3s_w3a8_prefill(x, w->raw, m, (size_t) w->n_in, (size_t) w->n_out, y);
}

/* P2.b: native Q5_K W5A8 NEON kernels. M=1 always wins over the
 * dequant trampoline. M>1 is platform-dependent — Mac AMX SGEMM beats
 * native NEON for high M, Pi 5 (no AMX) is the open question. The
 * env toggle below selects between them. */
static void cpu_neon_w_q5k_m1(const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    (void) be;
    linear_q5k_decode_w5a8(x, w->raw, (size_t) w->n_in, (size_t) w->n_out, y);
}
static void cpu_neon_w_q5k_mN(const float               *x,
                              const struct geist_weight *w,
                              size_t                     m,
                              struct geist_backend      *be,
                              float                     *y) {
    (void) be;
    linear_q5k_w5a8_prefill(x, w->raw, m, (size_t) w->n_in, (size_t) w->n_out, y);
}

/* P2.d: native Q8_0 W8A8 prefill kernel (M>1). M=1 already native
 * (cpu_neon_w_q8_0_m1 above). Same platform cross-over pattern as
 * Q5_K — Mac AMX SGEMM may win at high M, Pi 5 native expected to
 * win since OpenBLAS lags. */
static void cpu_neon_w_q8_0_mN(const float               *x,
                               const struct geist_weight *w,
                               size_t                     m,
                               struct geist_backend      *be,
                               float                     *y) {
    (void) be;
    linear_q8_0_w8a8_prefill(x, w->raw, m, (size_t) w->n_in, (size_t) w->n_out, y);
}

/* P2: dequant-and-cblas trampolines (Q5_K / F16 / BF16 / Q8_0 M>1).
 *
 * These don't have native NEON quantized-dot kernels (yet); they
 * cover the resolver path so the legacy v->linear() fallback can be
 * retired. Per-call cost is one dequant pass over the weight tile
 * plus a cblas call. Memory-efficient: tile by output rows (TILE
 * rows at a time) so the FP32 scratch stays bounded regardless of
 * tensor size — the largest Gemma 4 weight (lm_head 262144×1536)
 * would otherwise demand 1.5 GB transient.
 *
 * The performance ceiling here is the dequant pass + a cblas SGEMV
 * — for Q5_K the legacy `linear()` did the same thing as a single
 * full-tensor dequant + sgemm with a 1.5 GB malloc, so this is
 * functionally identical with a much smaller working set. A native
 * W5A8 kernel (Q5_K analog of Q4_K) is the obvious next step. */
#define DEQ_TILE_ROWS_DEFAULT 32

static size_t qk_sgemm_tile_rows_for(const struct cpu_neon_state *st) {
    if (st == nullptr || st->policy.qk_sgemm_tile_rows == 0) {
        return DEQ_TILE_ROWS_DEFAULT;
    }
    return st->policy.qk_sgemm_tile_rows;
}

typedef void (*dequant_row_fn)(const uint8_t *blocks, float *out, size_t n_elems);
static size_t blk_bytes_for(enum geist_dtype dt) {
    switch (dt) {
    case GEIST_DTYPE_Q4_0:
        return Q4_0_BLOCK_BYTES;
    case GEIST_DTYPE_Q4_K:
        return Q4_K_BLOCK_BYTES;
    case GEIST_DTYPE_Q5_K:
        return Q5_K_BLOCK_BYTES;
    case GEIST_DTYPE_Q6_K:
        return Q6_K_BLOCK_BYTES;
    case GEIST_DTYPE_Q8_0:
        return Q8_0_BLOCK_BYTES;
    case GEIST_DTYPE_TQ2_0:
        return TQ2_0_BLOCK_BYTES;
    default:
        return 0; /* F16/BF16: not block-quantized */
    }
}
static size_t blk_elems_for(enum geist_dtype dt) {
    switch (dt) {
    case GEIST_DTYPE_Q4_0:
        return Q4_0_BLOCK_ELEMS;
    case GEIST_DTYPE_Q4_K:
        return Q4_K_BLOCK_ELEMS;
    case GEIST_DTYPE_Q5_K:
        return Q5_K_BLOCK_ELEMS;
    case GEIST_DTYPE_Q6_K:
        return Q6_K_BLOCK_ELEMS;
    case GEIST_DTYPE_Q8_0:
        return Q8_0_BLOCK_ELEMS;
    case GEIST_DTYPE_TQ2_0:
        return TQ2_0_BLOCK_ELEMS;
    default:
        return 1;
    }
}
static dequant_row_fn dequant_row_fn_for(enum geist_dtype dt) {
    switch (dt) {
    case GEIST_DTYPE_Q4_0:
        return (dequant_row_fn) dequant_q4_0_row;
    case GEIST_DTYPE_Q4_K:
        return (dequant_row_fn) dequant_q4_K_row;
    case GEIST_DTYPE_Q5_K:
        return (dequant_row_fn) dequant_q5_K_row;
    case GEIST_DTYPE_Q6_K:
        return (dequant_row_fn) dequant_q6_K_row;
    case GEIST_DTYPE_Q8_0:
        return (dequant_row_fn) dequant_q8_0_row;
    case GEIST_DTYPE_TQ2_0:
        return (dequant_row_fn) dequant_tq2_0_row;
    default:
        return nullptr; /* F16/BF16 handled inline */
    }
}

/* Materialize TILE rows of the weight into `tile_fp32` (row-major,
 * [TILE, n_in]). Handles block-quantized k-quants + half-precision
 * (F16/BF16) sources. */
static void
dequant_tile(const struct geist_weight *w, size_t row_start, size_t tile_rows, float *tile_fp32) {
    const enum geist_dtype dt   = (enum geist_dtype) w->dtype;
    const size_t           n_in = (size_t) w->n_in;
    const dequant_row_fn   fn   = dequant_row_fn_for(dt);
    if (fn != nullptr) {
        const size_t   blk_bytes = blk_bytes_for(dt);
        const size_t   blk_elems = blk_elems_for(dt);
        const size_t   row_bytes = (n_in / blk_elems) * blk_bytes;
        const uint8_t *src       = (const uint8_t *) w->raw + row_start * row_bytes;
        for (size_t r = 0; r < tile_rows; r++) {
            fn(src + r * row_bytes, tile_fp32 + r * n_in, n_in);
        }
        return;
    }
    if (dt == GEIST_DTYPE_F16) {
        /* F16 row = 2 bytes per element. Use vcvt_f16_f32 if NEON,
         * scalar otherwise. */
        const uint16_t *src = (const uint16_t *) w->raw + row_start * n_in;
        for (size_t r = 0; r < tile_rows; r++) {
            float *dst = tile_fp32 + r * n_in;
            size_t i   = 0;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
            for (; i + 4 <= n_in; i += 4) {
                float16x4_t h = vld1_f16((const __fp16 *) (src + i));
                vst1q_f32(dst + i, vcvt_f32_f16(h));
            }
#endif
            for (; i < n_in; i++) {
                /* Scalar fallback — emulate F16-to-F32 via IEEE bit twiddle. */
                uint16_t h     = src[i];
                uint32_t sign  = (uint32_t) (h & 0x8000) << 16;
                uint32_t exp16 = (h >> 10) & 0x1F;
                uint32_t mant  = h & 0x3FF;
                uint32_t bits;
                if (exp16 == 0) {
                    bits = sign | (mant ? (((uint32_t) (mant) << 13) | 0x38800000u) : 0u);
                } else if (exp16 == 0x1F) {
                    bits = sign | 0x7F800000u | (mant << 13);
                } else {
                    bits = sign | ((exp16 + 112u) << 23) | (mant << 13);
                }
                memcpy(dst + i, &bits, 4);
            }
            src += n_in;
        }
        return;
    }
    if (dt == GEIST_DTYPE_BF16) {
        /* BF16 → F32: shift up by 16 bits, low half is zero. */
        const uint16_t *src = (const uint16_t *) w->raw + row_start * n_in;
        for (size_t r = 0; r < tile_rows; r++) {
            float *dst = tile_fp32 + r * n_in;
            for (size_t i = 0; i < n_in; i++) {
                uint32_t bits = (uint32_t) src[i] << 16;
                memcpy(dst + i, &bits, 4);
            }
            src += n_in;
        }
        return;
    }
}

/* M=1: tile through output rows, dequant each tile + sgemv. The tile
 * is heap-allocated once (tile_rows × n_in floats) and reused
 * across the output-row iterations of this single call. ~192 KB for
 * Gemma 4 d_model=1536; ~1.5 MB for FFN n_in=12288. */
static void cpu_neon_w_dequant_trampoline_m1(const float               *x,
                                             const struct geist_weight *w,
                                             struct geist_backend      *be,
                                             float                     *y) {
    const struct cpu_neon_state *st =
            (be != nullptr) ? (const struct cpu_neon_state *) be->state : nullptr;
    const size_t tile_rows = qk_sgemm_tile_rows_for(st);
    const size_t n_in      = (size_t) w->n_in;
    const size_t n_out     = (size_t) w->n_out;
    /* Decode-path optimization (P3.9): parallelize tile-by-tile across
     * cores. Each iteration dequants its own tile and does an
     * independent sgemv into a disjoint y[r0..r0+tr) slice — no shared
     * state. Pi 5 win: ~2× decode tok/s on TQ2_0. Mac builds without
     * libomp so _OPENMP is undefined and this collapses to the serial
     * loop. We use a private tile per thread instead of the previous
     * single shared tile so concurrent threads don't trample. */
#ifdef _OPENMP
#pragma omp parallel
    {
        float *tile = heap_alloc_array_aligned(float, tile_rows *n_in);
        if (tile == nullptr) {
            /* No room for per-thread tile — silently skip; decode will
             * see stale y values. Caller can't propagate. */
        } else {
#pragma omp for schedule(static) nowait
            for (size_t r0 = 0; r0 < n_out; r0 += tile_rows) {
                const size_t tr = (n_out - r0 < tile_rows) ? (n_out - r0) : tile_rows;
                dequant_tile(w, r0, tr, tile);
                /* Force OpenBLAS to use 1 thread inside the parallel region
                 * to avoid 4×4 = 16-way oversubscription. Set once per call
                 * via openblas_set_num_threads — cheap. */
                geist_sgemv(GEIST_OP_N,
                            (int) tr,
                            (int) n_in,
                            1.0f,
                            tile,
                            (int) n_in,
                            x,
                            1,
                            0.0f,
                            y + r0,
                            1);
            }
            safe_free((void **) &tile);
        }
    }
#else
    float *tile = heap_alloc_array_aligned(float, tile_rows *n_in);
    if (tile == nullptr)
        return;
    for (size_t r0 = 0; r0 < n_out; r0 += tile_rows) {
        const size_t tr = (n_out - r0 < tile_rows) ? (n_out - r0) : tile_rows;
        dequant_tile(w, r0, tr, tile);
        geist_sgemv(
                GEIST_OP_N, (int) tr, (int) n_in, 1.0f, tile, (int) n_in, x, 1, 0.0f, y + r0, 1);
    }
    safe_free((void **) &tile);
#endif
}

/* Fused F16 × A32 GEMV (M=1) — reads the f16 weight once and converts
 * 4-at-a-time in-register (vcvt_f32_f16), accumulating directly against the
 * fp32 activation. Avoids the dequant-trampoline's full f32 materialization
 * (which reads 656 MB + writes 1.3 GB + BLAS-reads 1.3 GB per call on the
 * BitNet-2B-4T tied f16 lm_head — the decode bottleneck). Bandwidth-bound:
 * one pass over the f16 weight. */
#if defined(__ARM_NEON)
void cpu_neon_w_f16_m1(const float               *x,
                       const struct geist_weight *w,
                       struct geist_backend      *be,
                       float                     *y) {
    (void) be;
    const size_t     n_in  = (size_t) w->n_in;
    const size_t     n_out = (size_t) w->n_out;
    const float16_t *W     = (const float16_t *) w->raw;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_out; r++) {
        const float16_t *wr   = W + r * n_in;
        float32x4_t      acc0 = vdupq_n_f32(0.0f);
        float32x4_t      acc1 = vdupq_n_f32(0.0f);
        size_t           k    = 0;
        for (; k + 8 <= n_in; k += 8) {
            acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vld1_f16(wr + k)), vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vld1_f16(wr + k + 4)), vld1q_f32(x + k + 4));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; k < n_in; k++) {
            sum += (float) wr[k] * x[k];
        }
        y[r] = sum;
    }
}
#endif

/* Definitions of the forward-declared SGEMM-prefill helpers. */
static bool
cpu_neon_dequant_w_workspace_prepare(struct cpu_neon_workspace *ws, size_t tile_rows, size_t n_in) {
    const size_t need = tile_rows * n_in;
    if (ws->dequant_w_fp32_cap >= need)
        return true;
    safe_free((void **) &ws->dequant_w_fp32);
    ws->dequant_w_fp32 = heap_alloc_array_aligned(float, need);
    if (ws->dequant_w_fp32 == nullptr) {
        ws->dequant_w_fp32_cap = 0;
        return false;
    }
    ws->dequant_w_fp32_cap = need;
    return true;
}

static void cpu_neon_qk_sgemm_run(const float               *x,
                                  const struct geist_weight *w,
                                  size_t                     m,
                                  size_t                     tile_rows,
                                  float                     *tile_fp32,
                                  float                     *y) {
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    /* Tile-loop parallelization: each thread dequants its own slice of
     * weight rows into its OWN stack-allocated tile then SGEMMs. This is
     * critical on Mac — Accelerate's internal threading doesn't engage
     * for small N (per-tile sizes are ~64x32x1536 which Accelerate runs
     * single-threaded), so we have to parallelize at this level. */
    (void) tile_fp32; /* workspace fallback used when OMP unavailable */
    const size_t n_tiles = (n_out + tile_rows - 1) / tile_rows;
    (void) n_tiles; /* only used in the _OPENMP tile loop below */
#if defined(_OPENMP)
#pragma omp parallel
    {
        /* Per-thread tile buffer on heap (32-row tile fp32). Reused across
         * the thread's iterations. Avoid stack VLA — n_in can be large. */
        float *t_tile = heap_alloc_array_aligned(float, tile_rows *n_in);
        if (t_tile != nullptr) {
#pragma omp for schedule(dynamic, 1)
            for (size_t t = 0; t < n_tiles; t++) {
                const size_t r0 = t * tile_rows;
                const size_t tr = (n_out - r0 < tile_rows) ? (n_out - r0) : tile_rows;
                dequant_tile(w, r0, tr, t_tile);
                geist_sgemm(GEIST_OP_N,
                            GEIST_OP_T,
                            (int) m,
                            (int) tr,
                            (int) n_in,
                            1.0f,
                            x,
                            (int) n_in,
                            t_tile,
                            (int) n_in,
                            0.0f,
                            y + r0,
                            (int) n_out);
            }
            safe_free((void **) &t_tile);
        }
    }
#else
    for (size_t r0 = 0; r0 < n_out; r0 += tile_rows) {
        const size_t tr = (n_out - r0 < tile_rows) ? (n_out - r0) : tile_rows;
        dequant_tile(w, r0, tr, tile_fp32);
        geist_sgemm(GEIST_OP_N,
                    GEIST_OP_T,
                    (int) m,
                    (int) tr,
                    (int) n_in,
                    1.0f,
                    x,
                    (int) n_in,
                    tile_fp32,
                    (int) n_in,
                    0.0f,
                    y + r0,
                    (int) n_out);
    }
#endif
}

/* M>1: tile through output rows, dequant each tile + sgemm against
 * the full activation block. */
static void cpu_neon_w_dequant_trampoline_mN(const float               *x,
                                             const struct geist_weight *w,
                                             size_t                     m,
                                             struct geist_backend      *be,
                                             float                     *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    /* Used for F32/F16/BF16 dense (notably Gemma's PLE model_proj, n×1536→8960).
     * Parallelize the output-row tiles across OMP threads — each thread dequants
     * its tile and runs a single-threaded cblas_sgemm. This threads via libgomp
     * (NOT OpenBLAS's pthread pool, which won't spawn alongside libgomp), so it
     * scales where a single big cblas_sgemm would stay single-threaded. The
     * model_proj was previously a fully serial tile loop. Gated to non-Accelerate
     * (Pi/Linux): on Mac, Accelerate threads its own cblas, so we keep the serial
     * tile loop there to avoid OMP×Accelerate oversubscription. */
#if defined(_OPENMP) && !defined(HAVE_ACCELERATE)
    const size_t n_tiles = (n_out + DEQ_TILE_ROWS_DEFAULT - 1) / DEQ_TILE_ROWS_DEFAULT;
#pragma omp parallel
    {
        float *tile = heap_alloc_array_aligned(float, DEQ_TILE_ROWS_DEFAULT *n_in);
        if (tile != nullptr) {
#pragma omp for schedule(dynamic, 1)
            for (size_t ti = 0; ti < n_tiles; ti++) {
                const size_t r0 = ti * DEQ_TILE_ROWS_DEFAULT;
                const size_t tr =
                        (n_out - r0 < DEQ_TILE_ROWS_DEFAULT) ? (n_out - r0) : DEQ_TILE_ROWS_DEFAULT;
                dequant_tile(w, r0, tr, tile);
                geist_sgemm(GEIST_OP_N,
                            GEIST_OP_T,
                            (int) m,
                            (int) tr,
                            (int) n_in,
                            1.0f,
                            x,
                            (int) n_in,
                            tile,
                            (int) n_in,
                            0.0f,
                            y + r0,
                            (int) n_out);
            }
            safe_free((void **) &tile);
        }
    }
#else
    float *tile = heap_alloc_array_aligned(float, DEQ_TILE_ROWS_DEFAULT *n_in);
    if (tile == nullptr)
        return;
    for (size_t r0 = 0; r0 < n_out; r0 += DEQ_TILE_ROWS_DEFAULT) {
        const size_t tr =
                (n_out - r0 < DEQ_TILE_ROWS_DEFAULT) ? (n_out - r0) : DEQ_TILE_ROWS_DEFAULT;
        dequant_tile(w, r0, tr, tile);
        geist_sgemm(GEIST_OP_N,
                    GEIST_OP_T,
                    (int) m,
                    (int) tr,
                    (int) n_in,
                    1.0f,
                    x,
                    (int) n_in,
                    tile,
                    (int) n_in,
                    0.0f,
                    y + r0,
                    (int) n_out);
    }
    safe_free((void **) &tile);
#endif
}

/* ---- Resolver -------------------------------------------------------- */

/* Kernel table — single source of truth for "which (m1, mN) pair is
 * installed for which dtype under which ISA". Scanned in declaration
 * order; the first row whose dtype matches AND whose `requires` is a
 * subset of the active host's ISA mask is installed.
 *
 * Adding a dtype = adding a row. Adding an ISA-specific specialization
 * = adding a row before the more general row for the same dtype. The
 * resolver itself stays a ~20-line generic dispatch.
 *
 * Policy overrides (Q5_K / Q8_0 / TQ2_0 native-vs-trampoline mN) and
 * the TQ2_0 TL1 install are applied AFTER the table match in
 * apply_resolver_post_hooks below — they are orthogonal to ISA
 * capability and don't belong in the capability table. */
static const struct cpu_neon_kernel_entry CPU_NEON_KERNELS[] = {
        /* K-series: all NEON-baseline. */
        {GEIST_DTYPE_Q3_K, CPU_NEON_ISA_NEON, cpu_neon_w_q3k_m1, cpu_neon_w_q3k_mN, "q3_K"},
        {GEIST_DTYPE_Q4_K, CPU_NEON_ISA_NEON, cpu_neon_w_q4k_m1, cpu_neon_w_q4k_mN, "q4_K"},
        {GEIST_DTYPE_Q5_K, CPU_NEON_ISA_NEON, cpu_neon_w_q5k_m1, cpu_neon_w_q5k_mN, "q5_K"},
        {GEIST_DTYPE_Q6_K, CPU_NEON_ISA_NEON, cpu_neon_w_q6k_m1, cpu_neon_w_q6k_mN, "q6_K"},
        {GEIST_DTYPE_Q8_0, CPU_NEON_ISA_NEON, cpu_neon_w_q8_0_m1, cpu_neon_w_q8_0_mN, "q8_0"},

        /* IQ-series. */
        {GEIST_DTYPE_IQ2_S, CPU_NEON_ISA_NEON, cpu_neon_w_iq2s_m1, cpu_neon_w_iq2s_mN, "iq2_s"},
        {GEIST_DTYPE_IQ3_S, CPU_NEON_ISA_NEON, cpu_neon_w_iq3s_m1, cpu_neon_w_iq3s_mN, "iq3_s"},

        /* F32 native both paths. */
        {GEIST_DTYPE_F32, CPU_NEON_ISA_NEON, cpu_neon_w_f32_m1, cpu_neon_w_f32_mN, "f32"},

/* Dequant-and-cblas trampolines for formats without a native NEON
 * kernel: F16 / BF16 / Q4_0. M>1 prefill via OpenBLAS / Accelerate
 * SGEMM after dequant; M=1 via the same trampoline. */
/* F16: fused in-register-convert GEMV for decode (M=1) — avoids the
 * trampoline's full f32 materialization (the BitNet-2B-4T tied lm_head is
 * f16 and dominates decode). M>1 prefill stays on the SGEMM trampoline. */
#if defined(__ARM_NEON)
        {GEIST_DTYPE_F16,
         CPU_NEON_ISA_NEON,
         cpu_neon_w_f16_m1,
         cpu_neon_w_dequant_trampoline_mN,
         "f16/fused-m1"},
#else
        {GEIST_DTYPE_F16,
         CPU_NEON_ISA_NEON,
         cpu_neon_w_dequant_trampoline_m1,
         cpu_neon_w_dequant_trampoline_mN,
         "f16/trampoline"},
#endif
        {GEIST_DTYPE_BF16,
         CPU_NEON_ISA_NEON,
         cpu_neon_w_dequant_trampoline_m1,
         cpu_neon_w_dequant_trampoline_mN,
         "bf16/trampoline"},
        {GEIST_DTYPE_Q4_0,
         CPU_NEON_ISA_NEON,
         cpu_neon_w_dequant_trampoline_m1,
         cpu_neon_w_dequant_trampoline_mN,
         "q4_0/trampoline"},

/* TQ2_0: ternary BitNet b1.58. Preferred (q8a, requires dotprod)
 * first, fp32 fallback second. The compile-time #if keeps the
 * dotprod-using symbols out of pre-ARMv8.2 builds entirely. */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        {GEIST_DTYPE_TQ2_0,
         CPU_NEON_ISA_NEON | CPU_NEON_ISA_DOTPROD,
         cpu_neon_w_tq2_0_q8a_m1,
         cpu_neon_w_tq2_0_q8a_mN,
         "tq2_0/q8a"},
#endif
        {GEIST_DTYPE_TQ2_0,
         CPU_NEON_ISA_NEON,
         cpu_neon_w_tq2_0_m1,
         cpu_neon_w_dequant_trampoline_mN,
         "tq2_0/fp32"},

/* I2_S: BitNet b1.58 official ternary (Microsoft 2B-4T). Dotprod-only —
 * the SDOT i2_s kernels assume ARMv8.2; no fp32 fallback row yet (every
 * geist ARM target enables +dotprod). */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        {GEIST_DTYPE_I2_S,
         CPU_NEON_ISA_NEON | CPU_NEON_ISA_DOTPROD,
         cpu_neon_w_i2_s_q8a_m1,
         cpu_neon_w_i2_s_q8a_mN,
         "i2_s/q8a"},
#endif
};
static_assert(sizeof(CPU_NEON_KERNELS) / sizeof(CPU_NEON_KERNELS[0]) > 0,
              "kernel table must not be empty");

/* TQ2_0 install hook: optional TL1 LUT-GEMV M=1 specialization (review
 * §Mac-decode catch-up). Allocates packed TL1 bytes via heap.h when the
 * shape is supported AND the policy enables it; falls back silently to
 * the table-selected q8a/fp32 kernel otherwise. */
static enum geist_status
install_tq2_0_tl1_if_eligible(struct geist_weight *w, const struct cpu_neon_kernel_policy *policy) {
    const size_t bytes = tl1_pack_size_bytes((size_t) w->n_in, (size_t) w->n_out);
    if (!cpu_neon_should_install_tl1(policy, (size_t) w->n_in, (size_t) w->n_out, bytes) ||
        bytes == 0 || bytes > (size_t) INT32_MAX) { /* aux_n is int32_t */
        return GEIST_OK;
    }
    void *tl1_buf = heap_alloc_aligned(bytes, 64);
    if (tl1_buf == nullptr) {
        return GEIST_OK;
    } /* stay on the q8a path */
    if (tl1_pack_from_tq2_0(w->raw, (size_t) w->n_in, (size_t) w->n_out, tl1_buf) != 0) {
        /* Pack rejected: free and stay on the q8a path. */
        safe_free(&tl1_buf);
        return GEIST_OK;
    }
    w->aux_fp32 = (const float *) tl1_buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->backend_layout    = GEIST_W_LAYOUT_TQ2_0_TL1;
    w->backend_alignment = 64;
    w->linear_m1         = cpu_neon_w_tl1_m1;
    /* Ternary prefill (m>1) stays on the mature SDOT q8a_mN path. The TL1
     * LUT-GEMM mN kernel loses end-to-end on A76 (Pi 5 4t BitNet 2B-4T seq128
     * prefill 21.0 TL1 vs 33.6 q8a) and paid ~383 MB extra RSS, so its opt-in
     * flag (GEIST_TL1_PREFILL) was removed. */
    return GEIST_OK;
}

static enum geist_status
install_q4k_predecode_if_eligible(struct geist_weight                 *w,
                                  const struct cpu_neon_kernel_policy *policy) {
    if (policy == nullptr || !policy->q4k_predecode || w == nullptr ||
        w->dtype != GEIST_DTYPE_Q4_K || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_OK;
    }
    const bool   use_ntile_pack = policy->q4k_mtile_prefill && policy->q4k_ntile_prefill;
    const size_t bytes =
            use_ntile_pack ? q4k_predecode_ntile4_size_bytes((size_t) w->n_in, (size_t) w->n_out)
                           : q4k_predecode_size_bytes((size_t) w->n_in, (size_t) w->n_out);
    if (bytes == 0 || bytes > (size_t) INT32_MAX) {
        return GEIST_OK;
    }
    void *buf = heap_alloc_aligned(bytes, 64);
    if (buf == nullptr) {
        return GEIST_OK;
    }
    const int pack_status =
            use_ntile_pack
                    ? q4k_predecode_ntile4_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf)
                    : q4k_predecode_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf);
    if (pack_status != 0) {
        safe_free(&buf);
        return GEIST_OK;
    }
    w->aux_fp32 = (const float *) buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->backend_layout =
            use_ntile_pack ? GEIST_W_LAYOUT_Q4_K_PREDECODE_NTILE4 : GEIST_W_LAYOUT_Q4_K_PREDECODE;
    w->backend_alignment = 64;
    return GEIST_OK;
}

static enum geist_status
install_q6k_ntile_if_eligible(struct geist_weight *w, const struct cpu_neon_kernel_policy *policy) {
    if (policy == nullptr || !policy->q6k_ntile_prefill || w == nullptr ||
        w->dtype != GEIST_DTYPE_Q6_K || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_OK;
    }
    /* Target FFN-down-style Q6_K matrices. Repacking very large output
     * tensors such as token embeddings/lm_head adds hundreds of MB and
     * does not address the measured Gemma prefill bottleneck. */
    if (w->n_in < 4096 || w->n_out > 4096) {
        return GEIST_OK;
    }

    const bool   use_stream = policy->q6k_ntile4_stream_prefill;
    const size_t bytes =
            use_stream ? q6k_predecode_ntile4_stream_size_bytes((size_t) w->n_in, (size_t) w->n_out)
                       : q6k_predecode_ntile4_size_bytes((size_t) w->n_in, (size_t) w->n_out);
    if (bytes == 0 || bytes > (size_t) INT32_MAX) {
        return GEIST_OK;
    }
    void *buf = heap_alloc_aligned(bytes, 64);
    if (buf == nullptr) {
        return GEIST_OK;
    }
    const int pack_status =
            use_stream
                    ? q6k_predecode_ntile4_stream_pack(
                              w->raw, (size_t) w->n_in, (size_t) w->n_out, buf)
                    : q6k_predecode_ntile4_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf);
    if (pack_status != 0) {
        safe_free(&buf);
        return GEIST_OK;
    }
    w->aux_fp32 = (const float *) buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->backend_layout    = use_stream ? GEIST_W_LAYOUT_Q6_K_PREDECODE_NTILE4_STREAM
                                      : GEIST_W_LAYOUT_Q6_K_PREDECODE_NTILE4;
    w->backend_alignment = 64;
    return GEIST_OK;
}

static enum geist_status
install_q6k_x8_gemv_if_eligible(struct geist_weight                 *w,
                                const struct cpu_neon_kernel_policy *policy) {
    (void) policy;
#if defined(GEIST_TARGET_PI5)
    const char *enabled = getenv("GEIST_Q6K_X8_GEMV");
    if (enabled == nullptr || enabled[0] != '1') {
        return GEIST_OK;
    }
    if (w == nullptr || w->dtype != GEIST_DTYPE_Q6_K || w->n_in <= 0 || w->n_out <= 0 ||
        w->n_out < 32768 || (w->n_out % 8) != 0) {
        return GEIST_OK;
    }
    const size_t bytes = q6k_x8_gemv_size_bytes((size_t) w->n_in, (size_t) w->n_out);
    if (bytes == 0 || bytes > (size_t) INT32_MAX) {
        return GEIST_OK;
    }
    void *buf = heap_alloc_aligned(bytes, 64);
    if (buf == nullptr) {
        return GEIST_OK;
    }
    if (q6k_x8_gemv_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf) != 0) {
        safe_free(&buf);
        return GEIST_OK;
    }
    w->aux_fp32 = (const float *) buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->backend_layout    = GEIST_W_LAYOUT_Q6_K_X8_GEMV;
    w->backend_alignment = 64;
#else
    (void) w;
#endif
    return GEIST_OK;
}

/* Apply per-dtype policy overrides after the table match. These are
 * platform-tuning decisions (native NEON vs Accelerate/OpenBLAS dequant
 * trampoline for M>1), not ISA-capability decisions. Q5_K + Q8_0 +
 * TQ2_0 all follow the same pattern: keep native mN if policy says so,
 * otherwise replace with the trampoline. See P2.c bench comments
 * preserved below.
 *
 *   Mac M1  trampoline (AMX SGEMM)   wins ~2-3× over native NEON M>1
 *   Pi 5    native NEON              wins ~1.3× over OpenBLAS SGEMM
 *
 * GEIST_*_NATIVE_MN env vars override the platform default per dtype. */
static void apply_resolver_post_hooks(struct geist_weight                 *w,
                                      const struct cpu_neon_kernel_policy *policy) {
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K:
        (void) install_q4k_predecode_if_eligible(w, policy);
        return;
    case GEIST_DTYPE_Q6_K:
        (void) install_q6k_x8_gemv_if_eligible(w, policy);
        if (w->backend_layout == GEIST_W_LAYOUT_Q6_K_X8_GEMV) {
            return;
        }
        (void) install_q6k_ntile_if_eligible(w, policy);
        return;
    case GEIST_DTYPE_Q5_K:
        if (!policy->q5k_native_mn) {
            w->linear_mN = cpu_neon_w_dequant_trampoline_mN;
        }
        return;
    case GEIST_DTYPE_Q8_0:
        if (!policy->q8_0_native_mn) {
            w->linear_mN = cpu_neon_w_dequant_trampoline_mN;
        }
        return;
    case GEIST_DTYPE_TQ2_0:
        /* Native q8a_mN vs trampoline (only meaningful when the q8a
         * row was installed, i.e. host has dotprod). On the fp32
         * fallback row, linear_mN is already the trampoline. */
        if (w->linear_m1 == cpu_neon_w_tq2_0_q8a_m1 && !policy->tq2_0_native_mn) {
            w->linear_mN = cpu_neon_w_dequant_trampoline_mN;
        }
        if (w->linear_m1 == cpu_neon_w_tq2_0_q8a_m1) {
            w->backend_layout    = GEIST_W_LAYOUT_TQ2_0_Q8A;
            w->backend_alignment = 64;
        }
        /* Optional TL1 LUT-GEMV M=1 over-installer. */
        if (w->linear_m1 == cpu_neon_w_tq2_0_q8a_m1) {
            (void) install_tq2_0_tl1_if_eligible(w, policy);
        }
        return;
    default:
        return;
    }
}

[[nodiscard]] enum geist_status cpu_neon_resolve_weight(struct geist_backend *be,
                                                        struct geist_weight  *w) {
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    /* Fail-fast on missing backend state: the policy carries the runtime
     * ISA bits that gate kernel installation. Silently falling back to a
     * synthesized default policy here would mean a test stub or partly-
     * initialized backend gets the wrong kernels with no diagnostic
     * (see code-review finding #5 / V7). The engine guarantees
     * geist_backend_create finishes before resolve_weight is called. */
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const struct cpu_neon_state        *bst    = (const struct cpu_neon_state *) be->state;
    const struct cpu_neon_kernel_policy policy = bst->policy;
    const cpu_neon_isa_mask             host_isa =
            (policy.has_dotprod ? CPU_NEON_ISA_DOTPROD : 0u) |
            (policy.has_fp16 ? CPU_NEON_ISA_FP16 : 0u) |
            CPU_NEON_ISA_NEON; /* cpu_neon backend only registers on NEON */

    const size_t n = sizeof(CPU_NEON_KERNELS) / sizeof(CPU_NEON_KERNELS[0]);
    for (size_t i = 0; i < n; i++) {
        const struct cpu_neon_kernel_entry *e = &CPU_NEON_KERNELS[i];
        if (e->dtype != (enum geist_dtype) w->dtype) {
            continue;
        }
        if ((e->requires & host_isa) != e->requires) {
            continue;
        }
        w->linear_m1 = e->linear_m1;
        w->linear_mN = e->linear_mN;
        if (w->dtype == GEIST_DTYPE_Q4_K) {
            w->linear_pair_m1 = cpu_neon_w_q4k_pair_m1;
            w->linear_pair_mN = cpu_neon_w_q4k_pair_mN;
        }
        if (w->backend_layout == 0) {
            w->backend_layout = GEIST_W_LAYOUT_SOURCE;
        }
        apply_resolver_post_hooks(w, &policy);
        return GEIST_OK;
    }
    /* No row matched: either the dtype is unknown to cpu_neon
     * (legacy vtable v->linear() fallback applies upstream), or every
     * matching row required an ISA the host lacks. */
    return GEIST_E_UNSUPPORTED;
}
