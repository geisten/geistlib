/*
 * src/backends/cpu_x86/backend.c — x86_64 backend, Phase 1a Step 5.
 *
 * Layer: BACKEND.
 *
 * cpu_x86's vtbl reuses cpu_scalar's slots for everything except create,
 * destroy, and resolve_weight: those three are overridden to thread the
 * per-instance state needed by the W4A8 hot path (acts + sum_a scratch),
 * and to install the cpu_x86 Q4_K M=1 kernel via cpu_x86_linear_q4k_resolve.
 *
 * The vtbl is initialized at module load via __attribute__((constructor)):
 *   1. Struct-copy cpu_scalar_vtbl into cpu_x86_vtbl.
 *   2. Override .create / .destroy / .resolve_weight.
 * Constructor runs before main, so the descriptor's vtbl pointer is
 * always valid by the time the engine calls geist_backend_create.
 *
 * Non-Q4_K dtypes fall through to cpu_scalar's resolver via the same
 * cpu_scalar_resolve_weight function — the descriptor remains drop-in
 * compatible with every Gemma 4 weight, and only the Q4_K decode path
 * routes through W4A8 + VPDPBUSD today.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "backend.h"

#include "backend_state.h"
#include "elementwise.h"
#include "kernel_f16_gemv.h"
#include "kernel_i2s.h"
#include "linear_f32q.h"
#include "linear_q4k.h"
#include "linear_q6k.h"

#include "geist_gemm.h"
#include "heap.h"

#include <geist_backend.h>
#include <geist_weight.h>

#include <stddef.h>
#include <string.h>

/* Borrowed from cpu_scalar/backend.c (exported there). */
extern const struct geist_backend_vtbl cpu_scalar_vtbl;
extern enum geist_status               cpu_scalar_resolve_weight(struct geist_backend *be,
                                                                 struct geist_weight  *w);

/* Mutable vtbl filled in at module-init time. */
static struct geist_backend_vtbl cpu_x86_vtbl;

/* ---------- Lifecycle ---------- */

[[nodiscard]] static enum geist_status cpu_x86_create(struct geist_backend            *be,
                                                      const struct geist_backend_opts *opts) {
    (void) opts;
    struct cpu_x86_state *st = geist_backend_alloc(be, sizeof(*st), alignof(struct cpu_x86_state));
    if (st == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "cpu_x86: failed to allocate %zu-byte state", sizeof(*st));
        return GEIST_E_OOM;
    }
    *st       = (struct cpu_x86_state) {0};
    be->state = st;
    return GEIST_OK;
}

static void cpu_x86_destroy(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct cpu_x86_state *st = (struct cpu_x86_state *) be->state;
    safe_free((void **) &st->acts_scratch);
    safe_free((void **) &st->sum_a_scratch);
    safe_free((void **) &st->acts_mtile);
    safe_free((void **) &st->sum_a_mtile);
    safe_free((void **) &st->scale_x_mtile);
    geist_backend_free(be, st);
    be->state = nullptr;
}

/* ---------- Resolver ---------- */

/* F32 dense (PLE gate/proj, small dense heads): cblas via geist_gemm. The
 * cpu_scalar fallback is a single-threaded naive triple loop (~10× slower);
 * for Gemma 4 the per-layer PLE projections run F32, so this matters at
 * prefill. Weight is row-major [n_out, n_in]; y = W @ x. */
static void cpu_x86_linear_f32_m1(const float               *x,
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

static void cpu_x86_linear_f32_mN(const float               *x,
                                  const struct geist_weight *w,
                                  size_t                     m,
                                  struct geist_backend      *be,
                                  float                     *y) {
    (void) be;
    /* Y [m, n_out] = X [m, n_in] @ W^T  (W row-major [n_out, n_in]). */
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

/* BitNet b1.58 I2_S ternary decode: native packed-2-bit VPDPBUSD GEMV
 * (kernel_i2s). The single per-tensor fp32 scale lives at the tail of the
 * weight blob (offset n_in*n_out/4). */
static void cpu_x86_linear_i2s_m1(const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    (void) be;
    const size_t   n_in  = (size_t) w->n_in;
    const size_t   n_out = (size_t) w->n_out;
    const uint8_t *raw   = (const uint8_t *) w->raw;
    float          scale;
    memcpy(&scale, raw + n_in * n_out / 4, sizeof scale);
    i2s_gemv_m1(n_out, n_in, x, raw, scale, y);
}

/* F16 dense decode (BitNet's tied lm_head, 657 MB read once per token):
 * OMP + F16C GEMV, far faster than the serial cpu_scalar dequant-dot. */
static void cpu_x86_linear_f16_m1(const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    (void) be;
    f16_gemv_m1((size_t) w->n_out, (size_t) w->n_in, x, (const uint16_t *) w->raw, y);
}

[[nodiscard]] static enum geist_status cpu_x86_resolve_weight(struct geist_backend *be,
                                                              struct geist_weight  *w) {
    /* Start from the cpu_scalar mapping: covers every dtype + sets m1/_mN
     * to the cpu_scalar kernels. Bail out on any error there. */
    enum geist_status base = cpu_scalar_resolve_weight(be, w);
    if (base != GEIST_OK) {
        return base;
    }
    /* Override the M=1 path per dtype. M>1 stays on cpu_scalar's slow
     * path for now — Phase 1b wires the W4A8 prefill kernel (see
     * docs/LINUX_X86_SPEC.md §"Prefill kernel topology").
     *
     * Q4_K → W4A8 + VPDPBUSD. Q6_K → fp32 predecode + cblas_sgemv (the
     * typical Gemma 4 tied lm_head). Other dtypes stay on cpu_scalar. */
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K: {
        struct cpu_x86_state *st = (struct cpu_x86_state *) be->state;
        (void) cpu_x86_linear_q4k_resolve(st, w); /* OOM → keep scalar m1 */
        break;
    }
    case GEIST_DTYPE_Q6_K:
        (void) cpu_x86_linear_q6k_resolve(w); /* OOM → keep scalar m1 */
        break;
    case GEIST_DTYPE_I2_S:
        /* Decode: native packed-2-bit VNNI GEMV. M>1 prefill stays on
         * cpu_scalar's dequant path until the lane-parallel ternary GEMM. */
        w->linear_m1 = cpu_x86_linear_i2s_m1;
        break;
    case GEIST_DTYPE_F16:
        /* Tied lm_head on BitNet: OMP + F16C GEMV for the M=1 head. */
        w->linear_m1 = cpu_x86_linear_f16_m1;
        break;
    case GEIST_DTYPE_F32:
        /* Quantize F32 dense (Gemma 4 PLE gate/proj) to W8A8 and run the
         * VPDPBUSD GEMM — far faster than skinny cblas. Falls back to cblas
         * when not applicable (n_in % 16 != 0) or on OOM. */
        if (cpu_x86_linear_f32q_resolve(w) != GEIST_OK) {
            w->linear_m1 = cpu_x86_linear_f32_m1;
            w->linear_mN = cpu_x86_linear_f32_mN;
        }
        break;
    default:
        break;
    }
    return GEIST_OK;
}

/* ---------- Vtbl init ---------- */

__attribute__((constructor)) static void cpu_x86_init_vtbl(void) {
    cpu_x86_vtbl                      = cpu_scalar_vtbl;
    cpu_x86_vtbl.create               = cpu_x86_create;
    cpu_x86_vtbl.destroy              = cpu_x86_destroy;
    cpu_x86_vtbl.resolve_weight       = cpu_x86_resolve_weight;
    cpu_x86_vtbl.gelu_tanh            = cpu_x86_gelu_tanh;
    cpu_x86_vtbl.gelu_tanh_mul        = cpu_x86_gelu_tanh_mul;
    cpu_x86_vtbl.gelu_tanh_mul_scaled = cpu_x86_gelu_tanh_mul_scaled;
}

const struct geist_backend_descriptor geist_backend_cpu_x86 = {
        .name   = "cpu_x86",
        .vtbl   = &cpu_x86_vtbl,
        .caps   = nullptr,
        .n_caps = 0,
};
