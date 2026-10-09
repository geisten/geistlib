/*
 * src/backends/cpu_neon/weight_resolve.c — load-time kernel resolution.
 *
 * Layer: BACKEND (cpu_neon). Implements geist_backend_vtbl::resolve_weight.
 *
 * Pre-resolves the (M=1 decode-style, M>1 prefill-style) kernel pair for
 * each weight tensor at model load. The forward loop then calls
 *
 *     w->linear_mN(m, x, w, be, y);
 *
 * without dtype dispatch or vtable indirection.
 *
 * The kernels live in kernels/, one file per weight format. Which pair a
 * dtype gets on which ISA is the table CPU_NEON_KERNELS below; a dtype
 * without a row returns GEIST_E_UNSUPPORTED. The dequant trampolines
 * (dequantize a tile of rows, then geist_sgemm) are the M>1 path where a
 * format has no native one or the policy picks them.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"
#include "parallel.h"
#include "tl1.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "linear_ref.h"
#include "quant.h"
#include "selected_rows.h"

#include <pthread.h>
#include <stdatomic.h>
#include "checked.h"
#include "heap.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* GEIST_PROFILE_QUANT=1: split the m>1 (prefill) wall time into W4A8/W6A8
 * activation quantization and matmul, printed at exit. The accumulators are
 * atomic: concurrent sessions share them. */
static _Atomic uint64_t g_qprof_quant_ns = 0;
static _Atomic uint64_t g_qprof_mm_ns    = 0;
static _Atomic int      g_qprof_state    = -1;
static pthread_once_t   g_qprof_once     = PTHREAD_ONCE_INIT;
static void             qprof_print(void) {
    const double q   = (double) atomic_load_explicit(&g_qprof_quant_ns, memory_order_relaxed) / 1e6;
    const double mm  = (double) atomic_load_explicit(&g_qprof_mm_ns, memory_order_relaxed) / 1e6;
    const double tot = q + mm;
    fprintf(stderr,
            "quant-profile: activation-quant %.2f ms (%.1f%%), matmul %.2f ms (%.1f%%)\n",
            q,
            tot > 0 ? 100.0 * q / tot : 0.0,
            mm,
            tot > 0 ? 100.0 * mm / tot : 0.0);
}
/* pthread_once, not a racy first-use check: two threads both seeing the
 * uninitialized state would each register atexit(qprof_print). */
static void qprof_init_once(void) {
    const char *e  = getenv("GEIST_PROFILE_QUANT");
    const int   on = (e != nullptr && e[0] == '1') ? 1 : 0;
    atomic_store_explicit(&g_qprof_state, on, memory_order_relaxed);
    if (on) {
        atexit(qprof_print);
    }
}

static inline bool qprof_on(void) {
    (void) pthread_once(&g_qprof_once, qprof_init_once);
    return atomic_load_explicit(&g_qprof_state, memory_order_relaxed) != 0;
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

/* ---- M=1 (decode) trampolines ---------------------------------------- */

/* Quantize `m` activation rows into the thread's workspace instead of a
 * per-call malloc, one scale per `group` elements (quant.h, #698: 256 for
 * the K-quants and IQ formats, 32 for Q4_0 / Q4_1 / Q8_0 / IQ4_NL).
 * geist_weight.h declares linear_m1 / linear_mN allocation-free; the
 * kernels' convenience wrappers in quant.h are not, so the resolver binds
 * the _pre variants and supplies the scratch. With `out_sum32` non-null the
 * per-32-element activation sums are filled too (Q4_0 x8, Q4_1, Q5_K).
 *
 * Returns nullptr when that scratch cannot be had, having computed y with
 * geist_linear_ref, which needs none: the kernel signature is void, and a
 * zeroed y would read as an answer. */
static const int8_t *ws_quantize_act(struct geist_backend      *be,
                                     size_t                     m,
                                     size_t                     n_in,
                                     size_t                     group,
                                     const float               *x,
                                     const struct geist_weight *w,
                                     float                     *y,
                                     float                    **out_scales,
                                     int32_t                  **out_sum32) {
    struct cpu_neon_state *st = (struct cpu_neon_state *) be->state;
    if (st == nullptr) {
        geist_linear_ref(m, x, w, y);
        return nullptr;
    }
    struct cpu_neon_workspace *ws       = cpu_neon_ws(st);
    const size_t               n_groups = geist_act_groups(n_in, group);
    size_t                     xq_need  = 0;
    size_t                     sc_need  = 0;
    size_t                     sum_need = 0;
    if (ws == nullptr || ckd_mul(&xq_need, m, n_in) || ckd_mul(&sc_need, m, n_groups) ||
        ckd_mul(&sum_need, m, n_in / 32u) ||
        !cpu_neon_grow_i8(&ws->act_xq, &ws->act_xq_cap, xq_need) ||
        !cpu_neon_grow_f32(&ws->act_scale, &ws->act_scale_cap, sc_need) ||
        (out_sum32 != nullptr &&
         !cpu_neon_grow_i32(&ws->act_sum32, &ws->act_sum32_cap, sum_need))) {
        geist_linear_ref(m, x, w, y);
        return nullptr;
    }
    const size_t blocks = n_in / 32u;
    for (size_t i = 0; i < m; i++) {
        quantize_x_q8_groups(n_in,
                             group,
                             x + i * n_in,
                             ws->act_xq + i * n_in,
                             ws->act_scale + i * n_groups,
                             out_sum32 != nullptr ? ws->act_sum32 + i * blocks : nullptr);
    }
    *out_scales = ws->act_scale;
    if (out_sum32 != nullptr) {
        *out_sum32 = ws->act_sum32;
    }
    return ws->act_xq;
}

/* The _pre prefill kernels of Q3_K, Q5_K, Q8_0, IQ2_S and IQ3_S keep one
 * accumulator per row on the stack, GEIST_QUANT_M_CAP of them, and return
 * without writing y for more rows. caps.max_m keeps the engine below that;
 * a direct caller gets y from the reference, and true. */
static bool ref_past_m_cap(size_t m, const float *x, const struct geist_weight *w, float *y) {
    if (m <= GEIST_QUANT_M_CAP) {
        return false;
    }
    geist_linear_ref(m, x, w, y);
    return true;
}

static void cpu_neon_w_q3k_m1(const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_q3k_decode_w3a8_pre(n_in, n_out, sc, xq, w->raw, y);
}

static void cpu_neon_w_q4k_m1(const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    (void) be;
    /* The raw SDOT decode GEMV beats the predecoded-block path, whose m=1
     * form is a GEMM kernel (M1 Max tg128: ~33 vs ~21). */
    linear_q4k_decode_w4a8((size_t) w->n_in, (size_t) w->n_out, x, w->raw, y);
}

static void cpu_neon_w_q4k_pair_m1(const float               *x,
                                   const struct geist_weight *w0,
                                   const struct geist_weight *w1,
                                   struct geist_backend      *be,
                                   float                     *y0,
                                   float                     *y1) {
    (void) be;
    if (w0->n_in != w1->n_in) {
        geist_linear_ref(1, x, w0, y0);
        geist_linear_ref(1, x, w1, y1);
        return;
    }
    linear_q4k_decode_w4a8_pair(
            (size_t) w0->n_in, (size_t) w0->n_out, (size_t) w1->n_out, x, w0->raw, w1->raw, y0, y1);
}

static void cpu_neon_w_q6k_m1(const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    /* The _pre kernels on the workspace scratch; same quantize_x_q8_groups
     * as the quant.h wrappers, so the same bits. */
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    if (w->backend_layout == GEIST_W_LAYOUT_Q6_K_X8_GEMV && w->aux_fp32 != nullptr) {
        linear_q6k_decode_w6a8_x8_pre(n_in, n_out, sc, xq, w->aux_fp32, y);
        return;
    }
    linear_q6k_decode_w6a8_pre(n_in, n_out, sc, xq, w->raw, y);
}

static void cpu_neon_w_q8_0_m1(const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8_0_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_q8_0_decode_w8a8_pre(n_in, n_out, sc, xq, w->raw, y);
}

static void cpu_neon_w_q4_0_m1(const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    const size_t n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    if (w->backend_layout == GEIST_W_LAYOUT_Q4_0_X8_GEMV && w->aux_fp32 != nullptr) {
        float        *sc  = nullptr;
        int32_t      *s32 = nullptr;
        const int8_t *xq  = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8_0_ELEMS, x, w, y, &sc, &s32);
        if (xq == nullptr) {
            return;
        }
        linear_q4_0_decode_w4a8_x8_pre(n_in, n_out, sc, xq, s32, w->aux_fp32, y);
        return;
    }
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8_0_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_q4_0_decode_w4a8_pre(n_in, n_out, sc, xq, w->raw, y);
}

static void cpu_neon_w_q4_1_m1(const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc  = nullptr;
    int32_t      *s32 = nullptr;
    const int8_t *xq  = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8_0_ELEMS, x, w, y, &sc, &s32);
    if (xq == nullptr) {
        return;
    }
    linear_q4_1_decode_w4a8_pre(n_in, n_out, sc, xq, s32, w->raw, y);
}

static void cpu_neon_w_q4_0_mN(size_t                     m,
                               const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    const size_t n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    /* int8 GEMM on the x8 layout beats both the row-major SDOT sweep and
     * the dequant+SGEMM trampoline: one weight pass, 4 tokens per block
     * load (#295). Falls through when the x8 aux is absent. */
    if (w->backend_layout == GEIST_W_LAYOUT_Q4_0_X8_GEMV && w->aux_fp32 != nullptr) {
        float        *sc  = nullptr;
        int32_t      *s32 = nullptr;
        const int8_t *xq  = ws_quantize_act(be, m, n_in, GEIST_ACT_Q8_0_ELEMS, x, w, y, &sc, &s32);
        if (xq == nullptr) {
            return;
        }
        linear_q4_0_w4a8_prefill_x8_pre(m, n_in, n_out, xq, sc, s32, w->aux_fp32, y);
        return;
    }
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, m, n_in, GEIST_ACT_Q8_0_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_q4_0_w4a8_prefill_pre(m, n_in, n_out, xq, sc, w->raw, y);
}

static void cpu_neon_w_q4_1_mN(size_t                     m,
                               const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc  = nullptr;
    int32_t      *s32 = nullptr;
    const int8_t *xq  = ws_quantize_act(be, m, n_in, GEIST_ACT_Q8_0_ELEMS, x, w, y, &sc, &s32);
    if (xq == nullptr) {
        return;
    }
    linear_q4_1_w4a8_prefill_pre(m, n_in, n_out, xq, sc, s32, w->raw, y);
}

static void cpu_neon_w_iq2s_m1(const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_iq2s_decode_w2a8_pre(n_in, n_out, sc, xq, w->raw, y);
}

static void cpu_neon_w_iq3s_m1(const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_iq3s_decode_w3a8_pre(n_in, n_out, sc, xq, w->raw, y);
}

static void cpu_neon_w_iq4xs_m1(const float               *x,
                                const struct geist_weight *w,
                                struct geist_backend      *be,
                                float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_iq4xs_decode_w4a8_pre(n_in, n_out, sc, xq, w->raw, y);
}

static void cpu_neon_w_iq4xs_mN(size_t                     m,
                                const float               *x,
                                const struct geist_weight *w,
                                struct geist_backend      *be,
                                float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, m, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_iq4xs_w4a8_prefill_pre(m, n_in, n_out, xq, sc, w->raw, y);
}

static void cpu_neon_w_iq4nl_m1(const float               *x,
                                const struct geist_weight *w,
                                struct geist_backend      *be,
                                float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8_0_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_iq4nl_decode_w4a8_pre(n_in, n_out, sc, xq, w->raw, y);
}

/* F32 dense: SGEMV / SGEMM through geist_gemm. The weight is row-major
 * [n_out, n_in], so y = W @ x is a sgemv with TransA=NoTrans. */
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

static void cpu_neon_w_f32_mN(size_t                     m,
                              const float               *x,
                              const struct geist_weight *w,
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

static void cpu_neon_w_q3k_mN(size_t                     m,
                              const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    if (ref_past_m_cap(m, x, w, y)) {
        return;
    }
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, m, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_q3k_w3a8_prefill_pre(m, n_in, n_out, xq, sc, w->raw, y);
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

/* One Q4_K M>1 activation quantization, for qk_mN_quantize_rows. */
struct qk_mN_quantize_job {
    const float *x;
    size_t       n_in;
    int8_t      *xq;
    float       *sc;
    int32_t     *sum32;
};

/* Rows [i0, i1) of cpu_neon_qk_mN_quantize_x. */
static void qk_mN_quantize_rows(void *ctx, size_t i0, size_t i1) {
    const struct qk_mN_quantize_job *job      = ctx;
    const float                     *x        = job->x;
    const size_t                     n_in     = job->n_in;
    const size_t                     n_groups = n_in / Q4_K_BLOCK_ELEMS;
    int8_t                          *xq       = job->xq;
    float                           *sc       = job->sc;
    int32_t                         *sum32    = job->sum32;
    for (size_t i = i0; i < i1; i++) {
        quantize_x_q8_groups(n_in,
                             GEIST_ACT_Q8K_ELEMS,
                             x + i * n_in,
                             xq + i * n_in,
                             sc + i * n_groups,
                             sum32 + i * (n_in / 32));
    }
}

/* Q4_K M>1 activation quantization into the qk_mN workspace: one scale per
 * 256-element super-block (#698) and the per-32 sums of the min-offset term.
 * Every Q4_K prefill kernel reads that layout, so the block-scale policy
 * (q4k_block_q8_prefill) now only picks between two bit-identical kernels. */
static void
cpu_neon_qk_mN_quantize_x(struct cpu_neon_workspace *ws, const float *x, size_t m, size_t n_in) {
    struct qk_mN_quantize_job job = {
            .x = x, .n_in = n_in, .xq = ws->qk_mN_xq, .sc = ws->qk_mN_sc, .sum32 = ws->qk_mN_sum32};
    if (m >= 4) {
        geist_par_for(m, qk_mN_quantize_rows, &job);
    } else {
        qk_mN_quantize_rows(&job, 0, m);
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
                    m, n_in, n_out, ws->qk_mN_xq, ws->qk_mN_sc, ws->qk_mN_sum32, w->aux_fp32, y);
        } else if (use_block_scales) {
            linear_q4k_w4a8_prefill_predecoded_mtile4_bscale(
                    m, n_in, n_out, ws->qk_mN_xq, ws->qk_mN_sc, ws->qk_mN_sum32, w->aux_fp32, y);
        } else if (st->policy.q4k_mtile_prefill) {
            if (st->policy.q4k_ntile_prefill) {
                linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4(m,
                                                                 n_in,
                                                                 n_out,
                                                                 ws->qk_mN_xq,
                                                                 ws->qk_mN_sc,
                                                                 ws->qk_mN_sum32,
                                                                 w->aux_fp32,
                                                                 y);
            } else {
                /* mtile8 is bit-identical to mtile4 and falls back to
                 * mtile4 for m<8. */
                linear_q4k_w4a8_prefill_predecoded_mtile8(m,
                                                          n_in,
                                                          n_out,
                                                          ws->qk_mN_xq,
                                                          ws->qk_mN_sc,
                                                          ws->qk_mN_sum32,
                                                          w->aux_fp32,
                                                          y);
            }
        } else {
            linear_q4k_w4a8_prefill_predecoded(
                    m, n_in, n_out, ws->qk_mN_xq, ws->qk_mN_sc, ws->qk_mN_sum32, w->aux_fp32, y);
        }
    } else {
        linear_q4k_w4a8_prefill_pre(
                m, n_in, n_out, ws->qk_mN_xq, ws->qk_mN_sc, ws->qk_mN_sum32, w->raw, y);
    }
}

/* SGEMM-prefill path (m ≥ threshold): dequant W tile-by-tile into a
 * workspace-resident fp32 scratch, call cblas_sgemm per tile. Defined
 * below dequant_tile. Activation x is consumed as fp32. */
static bool
cpu_neon_dequant_w_workspace_prepare(struct cpu_neon_workspace *ws, size_t tile_rows, size_t n_in);
static void   cpu_neon_qk_sgemm_run(const float               *x,
                                    const struct geist_weight *w,
                                    size_t                     m,
                                    size_t                     tile_rows,
                                    float                     *tile_fp32,
                                    float                     *y);
static size_t qk_sgemm_tile_rows_for(const struct cpu_neon_state *st);

static void cpu_neon_w_q4k_mN(size_t                     m,
                              const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    struct cpu_neon_state     *st   = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws   = cpu_neon_ws(st);
    const size_t               n_in = (size_t) w->n_in;
    if (m == 0)
        return;
    /* No workspace (the thread's first call, and the heap refused it), or
     * more rows than the kernels' stack accumulators: the reference. */
    if (ws == nullptr || m > GEIST_QUANT_M_CAP) {
        geist_linear_ref(m, x, w, y);
        return;
    }

    if (st->policy.q4k_sgemm_prefill && m >= st->policy.qk_sgemm_threshold &&
        cpu_neon_dequant_w_workspace_prepare(ws, qk_sgemm_tile_rows_for(st), n_in)) {
        cpu_neon_qk_sgemm_run(x, w, m, qk_sgemm_tile_rows_for(st), ws->dequant_w_fp32, y);
        return;
    }

    if (!cpu_neon_qk_mN_workspace_prepare(ws, m, n_in)) {
        geist_linear_ref(m, x, w, y);
        return;
    }

    const bool use_block_scales = st->policy.q4k_mtile_prefill && st->policy.q4k_block_q8_prefill &&
                                  q4k_weight_predecoded(w) && !q4k_weight_ntile4(w);
    const bool qp               = qprof_on();
    uint64_t   t0               = qp ? qprof_now_ns() : 0;
    cpu_neon_qk_mN_quantize_x(ws, x, m, n_in);
    if (qp) {
        atomic_fetch_add_explicit(&g_qprof_quant_ns, qprof_now_ns() - t0, memory_order_relaxed);
        t0 = qprof_now_ns();
    }
    cpu_neon_q4k_run_prequantized(st, ws, w, use_block_scales, m, y);
    if (qp) {
        atomic_fetch_add_explicit(&g_qprof_mm_ns, qprof_now_ns() - t0, memory_order_relaxed);
    }
}

static void cpu_neon_w_q4k_pair_mN(size_t                     m,
                                   const float               *x,
                                   const struct geist_weight *w0,
                                   const struct geist_weight *w1,
                                   struct geist_backend      *be,
                                   float                     *y0,
                                   float                     *y1) {
    struct cpu_neon_state     *st   = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws   = cpu_neon_ws(st);
    const size_t               n_in = (size_t) w0->n_in;
    if (m == 0)
        return;
    if (ws == nullptr || m > GEIST_QUANT_M_CAP || w0->n_in != w1->n_in) {
        geist_linear_ref(m, x, w0, y0);
        geist_linear_ref(m, x, w1, y1);
        return;
    }

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

    if (!cpu_neon_qk_mN_workspace_prepare(ws, m, n_in)) {
        geist_linear_ref(m, x, w0, y0);
        geist_linear_ref(m, x, w1, y1);
        return;
    }

    const bool use_block_scales = st->policy.q4k_mtile_prefill && st->policy.q4k_block_q8_prefill &&
                                  q4k_weight_predecoded(w0) && q4k_weight_predecoded(w1) &&
                                  !q4k_weight_ntile4(w0) && !q4k_weight_ntile4(w1);
    cpu_neon_qk_mN_quantize_x(ws, x, m, n_in);

    if (!use_block_scales && q4k_weight_ntile4(w0) && q4k_weight_ntile4(w1) &&
        w0->n_out == w1->n_out) {
        linear_q4k_w4a8_prefill_pair_predecoded_mtile4_ntile4_packed(m,
                                                                     n_in,
                                                                     (size_t) w0->n_out,
                                                                     ws->qk_mN_xq,
                                                                     ws->qk_mN_sc,
                                                                     ws->qk_mN_sum32,
                                                                     w0->aux_fp32,
                                                                     w1->aux_fp32,
                                                                     y0,
                                                                     y1);
    } else {
        cpu_neon_q4k_run_prequantized(st, ws, w0, use_block_scales, m, y0);
        cpu_neon_q4k_run_prequantized(st, ws, w1, use_block_scales, m, y1);
    }
}

static void cpu_neon_w_q6k_mN(size_t                     m,
                              const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    struct cpu_neon_state     *st   = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws   = cpu_neon_ws(st);
    const size_t               n_in = (size_t) w->n_in;
    if (m == 0)
        return;
    if (ws == nullptr || m > GEIST_QUANT_M_CAP) {
        geist_linear_ref(m, x, w, y);
        return;
    }

    /* SGEMM-prefill path (m ≥ threshold): dequant Q6_K + AMX SGEMM. */
    if (st->policy.q6k_sgemm_prefill && m >= st->policy.qk_sgemm_threshold &&
        cpu_neon_dequant_w_workspace_prepare(ws, qk_sgemm_tile_rows_for(st), n_in)) {
        cpu_neon_qk_sgemm_run(x, w, m, qk_sgemm_tile_rows_for(st), ws->dequant_w_fp32, y);
        return;
    }

    if (!cpu_neon_qk_mN_workspace_prepare(ws, m, n_in)) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    const bool qp6 = qprof_on();
    uint64_t   t6  = qp6 ? qprof_now_ns() : 0;
    for (size_t i = 0; i < m; i++) {
        quantize_x_q8_groups(n_in,
                             GEIST_ACT_Q8K_ELEMS,
                             x + i * n_in,
                             ws->qk_mN_xq + i * n_in,
                             ws->qk_mN_sc + i * (n_in / Q6_K_BLOCK_ELEMS),
                             nullptr);
    }
    if (qp6) {
        atomic_fetch_add_explicit(&g_qprof_quant_ns, qprof_now_ns() - t6, memory_order_relaxed);
        t6 = qprof_now_ns();
    }
    if (q6k_weight_ntile4_stream(w)) {
        linear_q6k_w6a8_prefill_predecoded_ntile4_stream(
                m, n_in, (size_t) w->n_out, ws->qk_mN_xq, ws->qk_mN_sc, w->aux_fp32, y);
    } else if (q6k_weight_ntile4(w)) {
        linear_q6k_w6a8_prefill_predecoded_ntile4(
                m, n_in, (size_t) w->n_out, ws->qk_mN_xq, ws->qk_mN_sc, w->aux_fp32, y);
    } else {
        linear_q6k_w6a8_prefill_pre(
                m, n_in, (size_t) w->n_out, ws->qk_mN_xq, ws->qk_mN_sc, w->raw, y);
    }
    if (qp6) {
        atomic_fetch_add_explicit(&g_qprof_mm_ns, qprof_now_ns() - t6, memory_order_relaxed);
    }
}

static void cpu_neon_w_iq2s_mN(size_t                     m,
                               const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    if (ref_past_m_cap(m, x, w, y)) {
        return;
    }
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, m, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_iq2s_w2a8_prefill_pre(m, n_in, n_out, xq, sc, w->raw, y);
}

static void cpu_neon_w_iq3s_mN(size_t                     m,
                               const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    if (ref_past_m_cap(m, x, w, y)) {
        return;
    }
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, m, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_iq3s_w3a8_prefill_pre(m, n_in, n_out, xq, sc, w->raw, y);
}

/* Native Q5_K W5A8 kernels. M=1 always beats the dequant trampoline;
 * for M>1 the q5k_native_mn policy picks (Mac AMX SGEMM wins at high M). */
static void cpu_neon_w_q5k_m1(const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc  = nullptr;
    int32_t      *s32 = nullptr;
    const int8_t *xq  = ws_quantize_act(be, 1, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, &s32);
    if (xq == nullptr) {
        return;
    }
    linear_q5k_decode_w5a8_pre(n_in, n_out, sc, xq, s32, w->raw, y);
}
static void cpu_neon_w_q5k_mN(size_t                     m,
                              const float               *x,
                              const struct geist_weight *w,
                              struct geist_backend      *be,
                              float                     *y) {
    if (ref_past_m_cap(m, x, w, y)) {
        return;
    }
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc  = nullptr;
    int32_t      *s32 = nullptr;
    const int8_t *xq  = ws_quantize_act(be, m, n_in, GEIST_ACT_Q8K_ELEMS, x, w, y, &sc, &s32);
    if (xq == nullptr) {
        return;
    }
    linear_q5k_w5a8_prefill_pre(m, n_in, n_out, xq, sc, s32, w->raw, y);
}

/* Native Q8_0 W8A8 prefill kernel (M>1); same q8_0_native_mn policy
 * cross-over as Q5_K. */
static void cpu_neon_w_q8_0_mN(size_t                     m,
                               const float               *x,
                               const struct geist_weight *w,
                               struct geist_backend      *be,
                               float                     *y) {
    if (ref_past_m_cap(m, x, w, y)) {
        return;
    }
    const size_t  n_in = (size_t) w->n_in, n_out = (size_t) w->n_out;
    float        *sc = nullptr;
    const int8_t *xq = ws_quantize_act(be, m, n_in, GEIST_ACT_Q8_0_ELEMS, x, w, y, &sc, nullptr);
    if (xq == nullptr) {
        return;
    }
    linear_q8_0_w8a8_prefill_pre(m, n_in, n_out, xq, sc, w->raw, y);
}

/* Dequant-and-cblas trampolines: for every format without a native path
 * (or where the policy prefers SGEMM). Per call, one dequant pass over the
 * weight plus a cblas call, tiled by output rows so the FP32 scratch stays
 * bounded (a 262144×1536 lm_head would otherwise need 1.5 GB). */
#define DEQ_TILE_ROWS_DEFAULT 32

static size_t qk_sgemm_tile_rows_for(const struct cpu_neon_state *st) {
    if (st == nullptr || st->policy.qk_sgemm_tile_rows == 0) {
        return DEQ_TILE_ROWS_DEFAULT;
    }
    return st->policy.qk_sgemm_tile_rows;
}

/* Exactly the dequant family's signature: a call through an incompatible
 * function pointer type is undefined (AGENT.md §6). */
typedef void (*dequant_row_fn)(size_t n_elems, const void *blocks, float *out);
static size_t blk_bytes_for(enum geist_dtype dt) {
    switch (dt) {
    case GEIST_DTYPE_Q4_0:
        return Q4_0_BLOCK_BYTES;
    case GEIST_DTYPE_Q4_1:
        return Q4_1_BLOCK_BYTES;
    case GEIST_DTYPE_Q5_0:
        return Q5_0_BLOCK_BYTES;
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
    case GEIST_DTYPE_PQ2_0:
        return PQ2_0_BLOCK_BYTES;
    case GEIST_DTYPE_IQ4_NL:
        return IQ4_NL_BLOCK_BYTES;
    case GEIST_DTYPE_IQ4_XS:
        return IQ4_XS_BLOCK_BYTES;
    default:
        return 0; /* F16/BF16: not block-quantized */
    }
}
static size_t blk_elems_for(enum geist_dtype dt) {
    switch (dt) {
    case GEIST_DTYPE_Q4_0:
        return Q4_0_BLOCK_ELEMS;
    case GEIST_DTYPE_Q4_1:
        return Q4_1_BLOCK_ELEMS;
    case GEIST_DTYPE_Q5_0:
        return Q5_0_BLOCK_ELEMS;
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
    case GEIST_DTYPE_PQ2_0:
        return PQ2_0_BLOCK_ELEMS;
    case GEIST_DTYPE_IQ4_NL:
        return IQ4_NL_BLOCK_ELEMS;
    case GEIST_DTYPE_IQ4_XS:
        return IQ4_XS_BLOCK_ELEMS;
    default:
        return 1;
    }
}
static dequant_row_fn dequant_row_fn_for(enum geist_dtype dt) {
    switch (dt) {
    case GEIST_DTYPE_Q4_0:
        return dequant_q4_0_row;
    case GEIST_DTYPE_Q4_1:
        return dequant_q4_1_row;
    case GEIST_DTYPE_Q5_0:
        return dequant_q5_0_row;
    case GEIST_DTYPE_Q4_K:
        return dequant_q4_K_row;
    case GEIST_DTYPE_Q5_K:
        return dequant_q5_K_row;
    case GEIST_DTYPE_Q6_K:
        return dequant_q6_K_row;
    case GEIST_DTYPE_Q8_0:
        return dequant_q8_0_row;
    case GEIST_DTYPE_TQ2_0:
        return dequant_tq2_0_row;
    case GEIST_DTYPE_PQ2_0:
        return dequant_pq2_0_row;
    case GEIST_DTYPE_IQ4_NL:
        return dequant_iq4_nl_row;
    case GEIST_DTYPE_IQ4_XS:
        return dequant_iq4_xs_row;
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
            fn(n_in, src + r * row_bytes, tile_fp32 + r * n_in);
        }
        return;
    }
    if (dt == GEIST_DTYPE_F16) {
        /* vcvt_f32_f16 with NEON fp16, scalar otherwise. */
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

/* One dequant trampoline call, for deq_tiles. */
struct deq_tiles_job {
    const float               *x;
    const struct geist_weight *w;
    size_t                     m, n_in, n_out, tile_rows, n_tiles;
    bool                       gemv;    /* the M=1 trampoline: sgemv */
    bool                       dynamic; /* tiles handed out one at a time */
    float                     *tiles;   /* geist_par_max_threads() slots */
    atomic_size_t              slot;    /* the next free slot */
    atomic_size_t              next;    /* dynamic: the next tile */
    float                     *y;
};

/* Output-row tiles of a dequant trampoline, each dequantized into this
 * range's slot of the caller's workspace tiles, then an sgemv (the M=1
 * trampoline) or sgemm into its disjoint y columns: tiles [t0, t1), or,
 * when the job is dynamic, the next tile from a shared counter until none
 * is left (OpenMP's schedule(dynamic, 1)). The slots are the caller's, not
 * per worker thread, so a call either has every tile it needs or none and
 * takes the reference whole. BLAS runs one thread inside these bodies,
 * which avoids 4×4 = 16-way oversubscription: geist_sgemv pins it once per
 * process, before any thread reaches cblas. */
static void deq_tiles(void *ctx, size_t t0, size_t t1) {
    struct deq_tiles_job      *job       = ctx;
    const float               *x         = job->x;
    const struct geist_weight *w         = job->w;
    const size_t               m         = job->m;
    const size_t               n_in      = job->n_in;
    const size_t               n_out     = job->n_out;
    const size_t               tile_rows = job->tile_rows;
    const size_t               n_tiles   = job->n_tiles;
    const bool                 gemv      = job->gemv;
    const bool                 dynamic   = job->dynamic;
    float                     *y         = job->y;
    float *tile = job->tiles +
                  atomic_fetch_add_explicit(&job->slot, 1, memory_order_relaxed) * tile_rows * n_in;
    for (size_t t = t0;;) {
        if (dynamic) {
            t = atomic_fetch_add_explicit(&job->next, 1, memory_order_relaxed);
            if (t >= n_tiles) {
                break;
            }
        } else if (t >= t1) {
            break;
        }
        const size_t r0 = t * tile_rows;
        const size_t tr = (n_out - r0 < tile_rows) ? (n_out - r0) : tile_rows;
        dequant_tile(w, r0, tr, tile);
        if (gemv) {
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
        } else {
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
        if (!dynamic) {
            t++;
        }
    }
}

/* Runs a dequant trampoline's tiles on `tiles` (prepared by
 * cpu_neon_dequant_w_workspace_prepare for this tile_rows and n_in). */
static void deq_tiles_run(const float               *x,
                          const struct geist_weight *w,
                          size_t                     m,
                          size_t                     tile_rows,
                          bool                       gemv,
                          bool                       dynamic,
                          float                     *tiles,
                          float                     *y) {
    const size_t         n_out = (size_t) w->n_out;
    struct deq_tiles_job job   = {.x         = x,
                                  .w         = w,
                                  .m         = m,
                                  .n_in      = (size_t) w->n_in,
                                  .n_out     = n_out,
                                  .tile_rows = tile_rows,
                                  .n_tiles   = (n_out + tile_rows - 1) / tile_rows,
                                  .gemv      = gemv,
                                  .dynamic   = dynamic,
                                  .tiles     = tiles,
                                  .y         = y};
    atomic_init(&job.slot, 0);
    atomic_init(&job.next, 0);
    const size_t threads = geist_par_max_threads();
    geist_par_for(dynamic && job.n_tiles > threads ? threads : job.n_tiles, deq_tiles, &job);
}

/* M=1: tile through output rows, dequant each tile + sgemv, the tiles
 * statically split. A tile (tile_rows × n_in floats) per thread, in the
 * calling thread's workspace and kept across calls: ~192 KB each for
 * Gemma 4 d_model=1536; ~1.5 MB for FFN n_in=12288. Without it, the
 * reference. */
static void cpu_neon_w_dequant_trampoline_m1(const float               *x,
                                             const struct geist_weight *w,
                                             struct geist_backend      *be,
                                             float                     *y) {
    struct cpu_neon_state     *st = (be != nullptr) ? (struct cpu_neon_state *) be->state : nullptr;
    struct cpu_neon_workspace *ws = st != nullptr ? cpu_neon_ws(st) : nullptr;
    const size_t               tile_rows = qk_sgemm_tile_rows_for(st);
    if (ws == nullptr || !cpu_neon_dequant_w_workspace_prepare(ws, tile_rows, (size_t) w->n_in)) {
        geist_linear_ref(1, x, w, y);
        return;
    }
    deq_tiles_run(x, w, 1, tile_rows, true, false, ws->dequant_w_fp32, y);
}

/* Fused F16 × A32 GEMV (M=1): one pass over the f16 weight, converted
 * in-register (vcvt_f32_f16) and accumulated against the fp32 activation,
 * instead of the trampoline's full f32 materialization (the BitNet-2B-4T
 * tied f16 lm_head dominates decode). */
#if defined(__ARM_NEON)
/* One cpu_neon_w_f16_m1 call, for f16_m1_rows. */
struct f16_m1_job {
    const float16_t *W;
    const float     *x;
    size_t           n_in;
    float           *y;
};

/* Output rows [r0, r1) of cpu_neon_w_f16_m1. */
static void f16_m1_rows(void *ctx, size_t r0, size_t r1) {
    const struct f16_m1_job *job  = ctx;
    const float16_t         *W    = job->W;
    const float             *x    = job->x;
    const size_t             n_in = job->n_in;
    float                   *y    = job->y;
    for (size_t r = r0; r < r1; r++) {
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

void cpu_neon_w_f16_m1(const float               *x,
                       const struct geist_weight *w,
                       struct geist_backend      *be,
                       float                     *y) {
    (void) be;
    struct f16_m1_job job = {
            .W = (const float16_t *) w->raw, .x = x, .n_in = (size_t) w->n_in, .y = y};
    geist_par_for((size_t) w->n_out, f16_m1_rows, &job);
}
#endif

/* Definitions of the forward-declared SGEMM-prefill helpers. */
static bool
cpu_neon_dequant_w_workspace_prepare(struct cpu_neon_workspace *ws, size_t tile_rows, size_t n_in) {
    /* One tile per thread of the call (deq_tiles claims them by slot). */
    size_t need = 0;
    if (ckd_mul(&need, tile_rows, n_in) || ckd_mul(&need, need, geist_par_max_threads())) {
        return false;
    }
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
    /* Each thread dequants its slice of weight rows into its tile, then
     * SGEMMs. Accelerate runs these small tile shapes single-threaded,
     * so the parallelism has to come from this level. */
    deq_tiles_run(x, w, m, tile_rows, false, true, tile_fp32, y);
}

/* M>1: tile through output rows, dequant each tile + sgemm against
 * the full activation block. */
static void cpu_neon_w_dequant_trampoline_mN(size_t                     m,
                                             const float               *x,
                                             const struct geist_weight *w,
                                             struct geist_backend      *be,
                                             float                     *y) {
    /* Used for F16/BF16 dense and the quantized formats without a native
     * mN kernel (Q5_K / Q8_0 / Q4_0 / Q4_1 on Mac). The output-row tiles
     * run in parallel, handed out one at a time, each a single-threaded
     * cblas_sgemm: neither Accelerate nor OpenBLAS threads these tile
     * shapes. */
    struct cpu_neon_state     *st = (be != nullptr) ? (struct cpu_neon_state *) be->state : nullptr;
    struct cpu_neon_workspace *ws = st != nullptr ? cpu_neon_ws(st) : nullptr;
    if (ws == nullptr ||
        !cpu_neon_dequant_w_workspace_prepare(ws, DEQ_TILE_ROWS_DEFAULT, (size_t) w->n_in)) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    deq_tiles_run(x, w, m, DEQ_TILE_ROWS_DEFAULT, false, true, ws->dequant_w_fp32, y);
}

/* ---- Resolver -------------------------------------------------------- */

/* Kernel table — single source of truth for "which (m1, mN) pair is
 * installed for which dtype under which ISA". Scanned in declaration
 * order; the first row whose dtype matches AND whose `requires` is a
 * subset of the active host's ISA mask is installed.
 *
 * An ISA-specific specialization goes before the more general row for
 * the same dtype.
 *
 * Policy overrides (native-vs-trampoline mN, layout repacks, TL1) are
 * applied after the table match in apply_resolver_post_hooks: they are
 * tuning decisions, not ISA capability. */
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
        /* IQ4: native W4A8 decode GEMVs (vqtbl1q LUT + SDOT); IQ4_NL
         * prefill on the dequant+SGEMM trampoline. */
        {GEIST_DTYPE_IQ4_NL,
         CPU_NEON_ISA_NEON,
         cpu_neon_w_iq4nl_m1,
         cpu_neon_w_dequant_trampoline_mN,
         "iq4_nl/w4a8-m1"},
        {GEIST_DTYPE_IQ4_XS,
         CPU_NEON_ISA_NEON,
         cpu_neon_w_iq4xs_m1,
         cpu_neon_w_iq4xs_mN,
         "iq4_xs/w4a8"},

        /* F32 native both paths. */
        {GEIST_DTYPE_F32, CPU_NEON_ISA_NEON, cpu_neon_w_f32_m1, cpu_neon_w_f32_mN, "f32"},

/* F16: fused in-register-convert GEMV for decode (M=1); prefill and BF16
 * on the dequant+SGEMM trampoline. */
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
        {GEIST_DTYPE_Q4_0, CPU_NEON_ISA_NEON, cpu_neon_w_q4_0_m1, cpu_neon_w_q4_0_mN, "q4_0/w4a8"},
        {GEIST_DTYPE_Q4_1, CPU_NEON_ISA_NEON, cpu_neon_w_q4_1_m1, cpu_neon_w_q4_1_mN, "q4_1/w4a8"},
        /* Q5_0: the odd tensors llama-quantize keeps in Q4_K_M models; no
         * native kernel yet, so both paths dequantize. */
        {GEIST_DTYPE_Q5_0,
         CPU_NEON_ISA_NEON,
         cpu_neon_w_dequant_trampoline_m1,
         cpu_neon_w_dequant_trampoline_mN,
         "q5_0/trampoline"},

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

/* PQ2_0: PrismML ternary (Ternary-Bonsai). Decode through the SDOT
 * W2A8 GEMV, prefill on the dequant+SGEMM trampoline (the x8 repack
 * replaces both, see install_pq2_0_x8_gemv_if_eligible). Hosts without
 * dotprod take the trampoline for both. */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        {GEIST_DTYPE_PQ2_0,
         CPU_NEON_ISA_NEON | CPU_NEON_ISA_DOTPROD,
         cpu_neon_w_pq2_0_q8a_m1,
         cpu_neon_w_dequant_trampoline_mN,
         "pq2_0/q8a-m1"},
#endif
        {GEIST_DTYPE_PQ2_0,
         CPU_NEON_ISA_NEON,
         cpu_neon_w_dequant_trampoline_m1,
         cpu_neon_w_dequant_trampoline_mN,
         "pq2_0/trampoline"},

/* I2_S: BitNet b1.58 official ternary (Microsoft 2B-4T). Dotprod-only:
 * no fp32 fallback row (every geist ARM target enables +dotprod). */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        {GEIST_DTYPE_I2_S,
         CPU_NEON_ISA_NEON | CPU_NEON_ISA_DOTPROD,
         cpu_neon_w_i2_s_q8a_m1,
         cpu_neon_w_i2_s_q8a_mN,
         "i2_s/q8a"},
#endif
};
/* Resolution counters: with GEIST_LOG_KERNELS=1 the resolver counts the
 * tensors bound per table row and prints them at backend destroy, so the
 * binary says which kernels a model executes (#327). Diagnostic only:
 * relaxed increments on an opt-in path, never read by the engine. */
static _Atomic uint32_t g_kernel_hits[sizeof(CPU_NEON_KERNELS) / sizeof(CPU_NEON_KERNELS[0])];

static bool kernel_log_enabled(void) {
    const char *e = getenv("GEIST_LOG_KERNELS");
    return e != nullptr && e[0] == '1';
}

void cpu_neon_dump_kernel_hits(void) {
    if (!kernel_log_enabled()) {
        return;
    }
    const size_t n = sizeof(CPU_NEON_KERNELS) / sizeof(CPU_NEON_KERNELS[0]);
    fprintf(stderr, "[geist] resolved kernels (tensors per catalog row):\n");
    for (size_t i = 0; i < n; i++) {
        const uint32_t hits = atomic_load_explicit(&g_kernel_hits[i], memory_order_relaxed);
        if (hits > 0) {
            fprintf(stderr, "[geist]   %-24s %6u\n", CPU_NEON_KERNELS[i].name, hits);
        }
    }
}

static_assert(sizeof(CPU_NEON_KERNELS) / sizeof(CPU_NEON_KERNELS[0]) > 0,
              "kernel table must not be empty");

/* TQ2_0 install hook: optional TL1 LUT-GEMV M=1 specialization.
 * Allocates packed TL1 bytes via heap.h when the shape is supported AND
 * the policy enables it; otherwise keeps the table-selected kernel. */
static enum geist_status
install_tq2_0_tl1_if_eligible(struct geist_weight *w, const struct cpu_neon_kernel_policy *policy) {
    const size_t bytes = tl1_pack_size_bytes((size_t) w->n_in, (size_t) w->n_out);
    if (!cpu_neon_should_install_tl1(policy, (size_t) w->n_in, (size_t) w->n_out, bytes) ||
        bytes == 0 || bytes > (size_t) INT32_MAX) { /* aux_n is int32_t */
        return GEIST_OK;
    }
    void *tl1_buf = heap_alloc_pages(bytes);
    if (tl1_buf == nullptr) {
        return GEIST_OK;
    } /* stay on the q8a path */
    if (tl1_pack_from_tq2_0(w->raw, (size_t) w->n_in, (size_t) w->n_out, tl1_buf) != 0) {
        /* Pack rejected: free and stay on the q8a path. */
        heap_free_pages(&tl1_buf, bytes);
        return GEIST_OK;
    }
    w->aux_fp32 = (const float *) tl1_buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_PAGES | GEIST_W_AUX_BACKEND_REPACK;
    w->backend_layout    = GEIST_W_LAYOUT_TQ2_0_TL1;
    w->backend_alignment = 64;
    w->linear_m1         = cpu_neon_w_tl1_m1;
    /* Prefill (m>1) stays on the SDOT q8a_mN path: a TL1 LUT-GEMM loses
     * there on A76 (Pi 5 BitNet 2B-4T prefill 21.0 vs 33.6 t/s). */
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
    void *buf = heap_alloc_pages(bytes);
    if (buf == nullptr) {
        return GEIST_OK;
    }
    const int pack_status =
            use_ntile_pack
                    ? q4k_predecode_ntile4_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf)
                    : q4k_predecode_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf);
    if (pack_status != 0) {
        heap_free_pages(&buf, bytes);
        return GEIST_OK;
    }
    w->aux_fp32 = (const float *) buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_PAGES | GEIST_W_AUX_BACKEND_REPACK;
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
    void *buf = heap_alloc_pages(bytes);
    if (buf == nullptr) {
        return GEIST_OK;
    }
    const int pack_status =
            use_stream
                    ? q6k_predecode_ntile4_stream_pack(
                              w->raw, (size_t) w->n_in, (size_t) w->n_out, buf)
                    : q6k_predecode_ntile4_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf);
    if (pack_status != 0) {
        heap_free_pages(&buf, bytes);
        return GEIST_OK;
    }
    w->aux_fp32 = (const float *) buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_PAGES | GEIST_W_AUX_BACKEND_REPACK;
    w->backend_layout    = use_stream ? GEIST_W_LAYOUT_Q6_K_PREDECODE_NTILE4_STREAM
                                      : GEIST_W_LAYOUT_Q6_K_PREDECODE_NTILE4;
    w->backend_alignment = 64;
    return GEIST_OK;
}

static enum geist_status
install_q6k_x8_gemv_if_eligible(struct geist_weight                 *w,
                                const struct cpu_neon_kernel_policy *policy) {
#if defined(__ARM_NEON)
    /* Interleaved-8-row Q6_K GEMV. Costs a heap copy of the tensor, so
     * the n_out gate limits it to lm_head-class tensors; the mmap'd
     * source pages go cold after warmup. Default on with Accelerate
     * (4B lm_head 16.1 -> 11.9 ms/token); opt-in elsewhere via
     * GEIST_Q6K_X8_GEMV=1 for RSS headroom. */
    if (!policy->q6k_x8_gemv) {
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
    void *buf = heap_alloc_pages(bytes);
    if (buf == nullptr) {
        return GEIST_OK;
    }
    if (q6k_x8_gemv_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf) != 0) {
        heap_free_pages(&buf, bytes);
        return GEIST_OK;
    }
    w->aux_fp32 = (const float *) buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_PAGES | GEIST_W_AUX_BACKEND_REPACK;
    w->backend_layout    = GEIST_W_LAYOUT_Q6_K_X8_GEMV;
    w->backend_alignment = 64;
#else
    (void) w;
    (void) policy;
#endif
    return GEIST_OK;
}

static enum geist_status
install_q4_0_x8_gemv_if_eligible(struct geist_weight                 *w,
                                 const struct cpu_neon_kernel_policy *policy) {
#if defined(__ARM_NEON)
    /* Interleaved-8-row Q4_0 GEMV. Unlike the Q6_K variant this hits
     * every projection tensor, so the heap copies sum to ~1x the model's
     * Q4_0 bytes; GEIST_W_RAW_COLD lets the loader drop the mmap'd
     * source pages (#729). See kernel_catalog.c for the default. */
    if (!policy->q4_0_x8_gemv) {
        return GEIST_OK;
    }
    if (w == nullptr || w->dtype != GEIST_DTYPE_Q4_0 || w->n_in <= 0 || w->n_out <= 0 ||
        w->n_out < 1024 || (w->n_out % 8) != 0 || ((size_t) w->n_in % 32) != 0) {
        return GEIST_OK;
    }
    const size_t bytes = q4_0_x8_gemv_size_bytes((size_t) w->n_in, (size_t) w->n_out);
    if (bytes == 0 || bytes > (size_t) INT32_MAX) {
        return GEIST_OK;
    }
    void *buf = heap_alloc_pages(bytes);
    if (buf == nullptr) {
        return GEIST_OK;
    }
    if (q4_0_x8_gemv_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf) != 0) {
        heap_free_pages(&buf, bytes);
        return GEIST_OK;
    }
    w->aux_fp32 = (const float *) buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_PAGES | GEIST_W_AUX_BACKEND_REPACK |
                GEIST_W_RAW_COLD;
    w->backend_layout    = GEIST_W_LAYOUT_Q4_0_X8_GEMV;
    w->backend_alignment = 64;
#else
    (void) w;
    (void) policy;
#endif
    return GEIST_OK;
}

/* PQ2_0 x8 interleaved decode GEMV over-installer; see kernels/pq2_0.c.
 * Any refusal (policy off, shape, OOM) keeps the table's row kernel. */
static void install_pq2_0_x8_gemv_if_eligible(struct geist_weight                 *w,
                                              const struct cpu_neon_kernel_policy *policy) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (!policy->pq2_0_x8_gemv || w->linear_m1 != cpu_neon_w_pq2_0_q8a_m1 || w->n_in <= 0 ||
        w->n_out <= 0) {
        return;
    }
    const size_t bytes = pq2_0_x8_size_bytes((size_t) w->n_in, (size_t) w->n_out);
    if (bytes == 0 || bytes > (size_t) INT32_MAX) {
        return;
    }
    void *buf = heap_alloc_pages(bytes);
    if (buf == nullptr) {
        return;
    }
    pq2_0_x8_pack(w->raw, (size_t) w->n_in, (size_t) w->n_out, buf);
    w->aux_fp32 = (const float *) buf;
    w->aux_n    = (int32_t) bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_PAGES | GEIST_W_AUX_BACKEND_REPACK |
                GEIST_W_RAW_COLD;
    w->backend_layout    = GEIST_W_LAYOUT_PQ2_0_X8_GEMV;
    w->backend_alignment = 64;
    w->linear_m1         = cpu_neon_w_pq2_0_x8_m1;
    w->linear_mN         = cpu_neon_w_pq2_0_x8_mN;
    w->linear_pair_m1    = cpu_neon_w_pq2_0_x8_pair_m1;
    w->linear_pair_mN    = cpu_neon_w_pq2_0_x8_pair_mN;
#else
    (void) w;
    (void) policy;
#endif
}

/* Apply per-dtype policy overrides after the table match: layout repacks,
 * and native NEON vs dequant+SGEMM trampoline for M>1. The trampoline wins
 * ~2-3× with AMX SGEMM on Mac; native NEON wins ~1.3× over OpenBLAS on
 * Pi 5. GEIST_*_NATIVE_MN env vars override the default per dtype. */
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
    case GEIST_DTYPE_IQ4_XS:
        /* Same crossover as Q4_0/Q8_0: Accelerate's dequant+SGEMM wins
         * on Mac; Pi/Linux keeps the native tile kernel. */
        if (!policy->iq4xs_native_mn) {
            w->linear_mN = cpu_neon_w_dequant_trampoline_mN;
        }
        return;
    case GEIST_DTYPE_Q8_0:
        if (!policy->q8_0_native_mn) {
            w->linear_mN = cpu_neon_w_dequant_trampoline_mN;
        }
        return;
    case GEIST_DTYPE_Q4_0:
    case GEIST_DTYPE_Q4_1:
        /* Same crossover as Q8_0: Mac AMX SGEMM beats the SDOT W4A8
         * prefill kernels; Pi/Linux keeps them (OpenBLAS lags). m1
         * decode stays native either way — with the x8 interleave
         * over-install where eligible (Q4_0 only). */
        if (!policy->q4_01_native_mn) {
            w->linear_mN = cpu_neon_w_dequant_trampoline_mN;
        }
        if (w->dtype == GEIST_DTYPE_Q4_0) {
            (void) install_q4_0_x8_gemv_if_eligible(w, policy);
            if (w->backend_layout == GEIST_W_LAYOUT_Q4_0_X8_GEMV) {
                /* With the x8 aux present, the int8 mN GEMM replaces the
                 * dequant+SGEMM trampoline (#295). */
                w->linear_mN = cpu_neon_w_q4_0_mN;
            }
        }
        return;
    case GEIST_DTYPE_PQ2_0:
        install_pq2_0_x8_gemv_if_eligible(w, policy);
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

/* Runtime ISA bits this host offers, in the same form the table's
 * `requires` masks are written in. */
static cpu_neon_isa_mask host_isa_mask(const struct cpu_neon_kernel_policy *policy) {
    return (policy->has_dotprod ? CPU_NEON_ISA_DOTPROD : 0u) |
           (policy->has_fp16 ? CPU_NEON_ISA_FP16 : 0u) |
           CPU_NEON_ISA_NEON; /* cpu_neon backend only registers on NEON */
}

/* First table row that matches `dtype` and whose ISA requirements this
 * host meets — the same "first match wins" scan cpu_neon_resolve_weight
 * performs, so a capability answer and the kernel actually installed can
 * never disagree. Returns nullptr when no row applies. */
static const struct cpu_neon_kernel_entry *lookup_kernel_entry(enum geist_dtype  dtype,
                                                               cpu_neon_isa_mask host_isa) {
    const size_t n = sizeof(CPU_NEON_KERNELS) / sizeof(CPU_NEON_KERNELS[0]);
    for (size_t i = 0; i < n; i++) {
        const struct cpu_neon_kernel_entry *e = &CPU_NEON_KERNELS[i];
        if (e->dtype != dtype) {
            continue;
        }
        if ((e->requires & host_isa) != e->requires) {
            continue;
        }
        return e;
    }
    return nullptr;
}

enum cpu_neon_linear_support_kind cpu_neon_linear_support(const struct geist_backend *be,
                                                          enum geist_dtype            w_dtype) {
    if (be == nullptr || be->state == nullptr) {
        return CPU_NEON_SUPPORT_NONE;
    }
    const struct cpu_neon_state        *bst = (const struct cpu_neon_state *) be->state;
    const struct cpu_neon_kernel_entry *e =
            lookup_kernel_entry(w_dtype, host_isa_mask(&bst->policy));
    if (e == nullptr) {
        return CPU_NEON_SUPPORT_NONE;
    }
    /* EMULATED: both paths are the generic dequant trampoline. A
     * purpose-built kernel on either path (f16's fused m1 GEMV) counts as
     * NATIVE. */
    const bool m1_generic = e->linear_m1 == cpu_neon_w_dequant_trampoline_m1;
    const bool mN_generic = e->linear_mN == cpu_neon_w_dequant_trampoline_mN;
    return (m1_generic && mN_generic) ? CPU_NEON_SUPPORT_EMULATED : CPU_NEON_SUPPORT_NATIVE;
}

[[nodiscard]] enum geist_status cpu_neon_resolve_weight(struct geist_backend *be,
                                                        struct geist_weight  *w) {
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0 || w->raw_nbytes == 0u) {
        return GEIST_E_INVALID_ARG;
    }
    /* The source extent, before anything reads it: the repacks (Q4_K
     * predecode, Q6_K n-tile, TL1) stream the whole tensor, so raw_nbytes
     * must cover (dtype, n_in, n_out). */
    if (!quant_weight_extent_ok(w)) {
        return GEIST_E_FORMAT;
    }
    /* Fail fast without backend state: the policy carries the runtime ISA
     * bits that gate kernel installation, and a synthesized default could
     * install the wrong kernels silently. */
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct cpu_neon_state *bst = (struct cpu_neon_state *) be->state;
    /* Fold an applied calibration into the policy exactly once, at the
     * first kernel binding (the apply window just closed — the engine
     * set be->calibration.locked before calling us). */
    if (!bst->policy_calibrated) {
        if (be->calibration.n_values > 0) {
            bst->policy = cpu_neon_kernel_policy_effective(&bst->hw, be);
        }
        bst->policy_calibrated = true;
    }
    const struct cpu_neon_kernel_policy policy   = bst->policy;
    const cpu_neon_isa_mask             host_isa = host_isa_mask(&policy);

    const size_t n = sizeof(CPU_NEON_KERNELS) / sizeof(CPU_NEON_KERNELS[0]);
    for (size_t i = 0; i < n; i++) {
        const struct cpu_neon_kernel_entry *e = &CPU_NEON_KERNELS[i];
        if (e->dtype != (enum geist_dtype) w->dtype) {
            continue;
        }
        if ((e->requires & host_isa) != e->requires) {
            continue;
        }
        if (kernel_log_enabled()) {
            atomic_fetch_add_explicit(&g_kernel_hits[i], 1u, memory_order_relaxed);
        }
        w->linear_m1 = e->linear_m1;
        w->linear_mN = e->linear_mN;
        if (w->dtype == GEIST_DTYPE_Q4_K) {
            w->linear_pair_m1 = cpu_neon_w_q4k_pair_m1;
            w->linear_pair_mN = cpu_neon_w_q4k_pair_mN;
        }
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        /* The row kernel's pair. mN stays the dequant trampoline here,
         * which has no pair form; install_pq2_0_x8_gemv_if_eligible
         * replaces both slots from the post hooks below when the x8
         * repack takes over. Keyed on the kernel the table actually
         * installed, not on the dtype: a host without dotprod took the
         * trampoline row and must keep it. */
        if (w->linear_m1 == cpu_neon_w_pq2_0_q8a_m1) {
            w->linear_pair_m1 = cpu_neon_w_pq2_0_q8a_pair_m1;
        }
#endif
        if (w->backend_layout == 0) {
            w->backend_layout = GEIST_W_LAYOUT_SOURCE;
        }
        apply_resolver_post_hooks(w, &policy);
        if (w->backend_layout == GEIST_W_LAYOUT_PQ2_0_X8_GEMV) {
            w->linear_rows      = geist_cpu_selected_rows;
            w->linear_rows_tile = 8;
        } else if (w->backend_layout == GEIST_W_LAYOUT_SOURCE &&
                   (w->flags & GEIST_W_AUX_BACKEND_REPACK) == 0 &&
                   w->linear_m1 != cpu_neon_w_dequant_trampoline_m1 &&
                   w->dtype != GEIST_DTYPE_F32 && w->dtype != GEIST_DTYPE_I2_S) {
            w->linear_rows      = geist_cpu_selected_rows;
            w->linear_rows_tile = 1;
        }
        return GEIST_OK;
    }
    /* No row matched: the dtype is unknown to cpu_neon, or every matching
     * row requires an ISA the host lacks. */
    return GEIST_E_UNSUPPORTED;
}
