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
#include <stdlib.h>
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

static void cpu_x86_linear_i2s_mN(const float               *x,
                                  const struct geist_weight *w,
                                  size_t                     m,
                                  struct geist_backend      *be,
                                  float                     *y) {
    (void) be;
    const size_t   n_in  = (size_t) w->n_in;
    const size_t   n_out = (size_t) w->n_out;
    const uint8_t *raw   = (const uint8_t *) w->raw;
    float          scale;
    memcpy(&scale, raw + n_in * n_out / 4, sizeof scale);
    i2s_gemm_mN(m, n_out, n_in, x, raw, scale, y);
}

/* I2_S x4 row-interleaved fast path: codes live in w->aux_fp32 (the
 * i2s_to_x4 blob), the per-tensor scale at the tail of w->raw. */
static void cpu_x86_linear_i2s_x4_m1(const float               *x,
                                     const struct geist_weight *w,
                                     struct geist_backend      *be,
                                     float                     *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    float        scale;
    memcpy(&scale, (const uint8_t *) w->raw + n_in * n_out / 4, sizeof scale);
    i2s_x4_gemv_m1(n_out, n_in, x, (const uint8_t *) w->aux_fp32, scale, y);
}

static void cpu_x86_linear_i2s_x4_mN(const float               *x,
                                     const struct geist_weight *w,
                                     size_t                     m,
                                     struct geist_backend      *be,
                                     float                     *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    float        scale;
    memcpy(&scale, (const uint8_t *) w->raw + n_in * n_out / 4, sizeof scale);
    i2s_x4_gemm_mN(m, n_out, n_in, x, (const uint8_t *) w->aux_fp32, scale, y);
}

/* Install the x4 fast path: repack into the row-interleaved blob (aux,
 * heap-owned) and route both kernels there. Returns false (keep the packed
 * path) on shape mismatch or OOM. */
static bool cpu_x86_linear_i2s_x4_resolve(struct geist_weight *w) {
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if (n_out % 4 != 0 || n_in % 64 != 0 || !i2s_isa_is_vnni()) {
        return false;
    }
    const size_t blob_bytes = n_out * n_in / 4;
    uint8_t     *blob       = heap_alloc_aligned(blob_bytes, OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return false;
    }
    i2s_to_x4(n_out, n_in, (const uint8_t *) w->raw, blob);
    w->aux_fp32  = (const float *) blob;
    w->aux_n     = (int32_t) blob_bytes;
    w->flags     = (uint16_t) (w->flags | GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK);
    w->linear_m1 = cpu_x86_linear_i2s_x4_m1;
    w->linear_mN = cpu_x86_linear_i2s_x4_mN;
    return true;
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

/* Q8-weight decode: int8 weights (0.5× the f16 read) in w->aux_fp32, per-row
 * scales right after them. The BW lever for the lm_head. */
static inline size_t q8w_scales_offset(size_t n_out, size_t n_in) {
    return (n_out * n_in + 63u) & ~(size_t) 63u; /* int8 wq, then 64-aligned scales */
}

static void cpu_x86_linear_q8w_m1(const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    (void) be;
    const size_t  n_in   = (size_t) w->n_in;
    const size_t  n_out  = (size_t) w->n_out;
    const int8_t *wq     = (const int8_t *) w->aux_fp32;
    const float  *scales = (const float *) (wq + q8w_scales_offset(n_out, n_in));
    q8w_gemv_m1(n_out, n_in, x, wq, scales, y);
}

/* Quantize an F16 weight to per-row int8 at load (aux, heap-owned). Returns
 * false (keep the f16 GEMV) on OOM. Only worthwhile for big weights (the
 * 657 MB tied lm_head); small F16 weights keep the exact f16 path. */
static bool cpu_x86_linear_q8w_resolve(struct geist_weight *w) {
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if ((uint64_t) n_out * n_in < (1u << 20)) {
        return false; /* tiny F16 dense → not worth the BW/quant tradeoff */
    }
    {
        const char *e = getenv("GEIST_Q8_LMHEAD"); /* =0 keeps the exact F16C GEMV */
        if (e != nullptr && e[0] == '0') {
            return false;
        }
    }
    const size_t scales_off = q8w_scales_offset(n_out, n_in);
    const size_t blob_bytes = scales_off + n_out * sizeof(float);
    uint8_t     *blob       = heap_alloc_aligned(blob_bytes, OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return false;
    }
    f16_to_q8w(
            n_out, n_in, (const uint16_t *) w->raw, (int8_t *) blob, (float *) (blob + scales_off));
    w->aux_fp32  = (const float *) blob;
    w->aux_n     = (int32_t) blob_bytes;
    w->flags     = (uint16_t) (w->flags | GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK);
    w->linear_m1 = cpu_x86_linear_q8w_m1;
    return true;
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
        /* Fast path: 4-row-interleaved x4 layout (one act load feeds 4 rows).
         * Falls back to the packed VPDPBUSD kernels on shape mismatch / OOM /
         * non-VNNI. */
        if (!cpu_x86_linear_i2s_x4_resolve(w)) {
            w->linear_m1 = cpu_x86_linear_i2s_m1;
            w->linear_mN = cpu_x86_linear_i2s_mN;
        }
        break;
    case GEIST_DTYPE_F16:
        /* Tied lm_head on BitNet: Q8 weight (half the BW) for the M=1 head;
         * F16C GEMV fallback on OOM / tiny weights. */
        if (!cpu_x86_linear_q8w_resolve(w)) {
            w->linear_m1 = cpu_x86_linear_f16_m1;
        }
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
