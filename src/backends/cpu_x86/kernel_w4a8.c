/*
 * src/backends/cpu_x86/kernel_w4a8.c — W4A8 dispatcher.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Phase 1a Step 1 (current): scalar-only dispatcher. The function-pointer
 * indirection is in place from day one so adding the AVX-512+VNNI / AVX-512
 * / AVX2 tiers is "fill the slot at init" and not "rewrite the caller". See
 * docs/LINUX_X86_SPEC.md §"Architecture commits" (commit 3).
 *
 * Init policy:
 *   1. Probe ISA via geist_hw_probe.
 *   2. Apply GEIST_FORCE_ISA env clamp (down-only — never lifts above probe).
 *   3. Resolve to the highest available variant whose TU is compiled in.
 *
 * Compiled at the baseline -march=x86-64-v3 (Haswell+), with no AVX-512
 * intrinsics, so this TU is safe to call before any cpuid check.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_w4a8.h"

#include "hw_probe.h"
#include "quant.h" /* quantize_x_int8_sym */

#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

typedef float (*w4a8_dot_fn)(size_t        n_blocks,
                             const uint8_t weights[static n_blocks * W4A8_BLOCK_BYTES_WEIGHTS],
                             const float   w_scales[static n_blocks],
                             const float   w_offsets[static n_blocks],
                             const int8_t  acts[static n_blocks * W4A8_BLOCK_ELEMS],
                             const int32_t sum_a_per_block[static n_blocks],
                             float         scale_x);

/* Forward decls of the ISA-specific variants. Each lives in its own TU
 * compiled with the matching -march= flag; see mk/backend-cpu_x86.mk. */
[[nodiscard]] float
w4a8_dot_avx512_vnni(size_t        n_blocks,
                     const uint8_t weights[static n_blocks * W4A8_BLOCK_BYTES_WEIGHTS],
                     const float   w_scales[static n_blocks],
                     const float   w_offsets[static n_blocks],
                     const int8_t  acts[static n_blocks * W4A8_BLOCK_ELEMS],
                     const int32_t sum_a_per_block[static n_blocks],
                     float         scale_x);

/* Dispatcher state. Once written, never mutated again — readers see a
 * consistent snapshot under any thread interleaving (writes happen at
 * backend create, reads from the inference hot path). */
static w4a8_dot_fn   g_dot     = nullptr;
static enum w4a8_isa g_current = W4A8_ISA_SCALAR;
static int           g_inited  = 0;

static enum w4a8_isa probe_best_isa(void) {
    struct geist_hw_probe hw;
    geist_hw_probe_fill(&hw);
    if (hw.has_avx512_vnni) {
        return W4A8_ISA_AVX512_VNNI;
    }
    if (hw.has_avx512f) {
        return W4A8_ISA_AVX512;
    }
    if (hw.has_avx2) {
        return W4A8_ISA_AVX2;
    }
    return W4A8_ISA_SCALAR;
}

static enum w4a8_isa parse_force_env(void) {
    const char *e = getenv("GEIST_FORCE_ISA");
    if (e == nullptr || e[0] == '\0') {
        return W4A8_ISA_AVX512_BF16; /* sentinel: "no override". */
    }
    if (strcmp(e, "scalar") == 0) {
        return W4A8_ISA_SCALAR;
    }
    if (strcmp(e, "avx2") == 0) {
        return W4A8_ISA_AVX2;
    }
    if (strcmp(e, "avx512") == 0) {
        return W4A8_ISA_AVX512;
    }
    if (strcmp(e, "avx512_vnni") == 0) {
        return W4A8_ISA_AVX512_VNNI;
    }
    if (strcmp(e, "avx512_bf16") == 0) {
        return W4A8_ISA_AVX512_BF16;
    }
    /* Unknown value: treat as "no override" rather than fail loudly. */
    return W4A8_ISA_AVX512_BF16;
}

enum w4a8_isa w4a8_dispatcher_init(void) {
    if (g_inited != 0) {
        return g_current;
    }

    enum w4a8_isa best   = probe_best_isa();
    enum w4a8_isa forced = parse_force_env();
    /* Clamp the override down — never lift above what the host supports. */
    enum w4a8_isa chosen = (forced < best) ? forced : best;

    /* Wire the function-pointer slot. AVX-512 (no VNNI) and AVX2 variants
     * arrive in Phase 1a Step 3/4; until then they fall back to scalar. */
    switch (chosen) {
    case W4A8_ISA_AVX512_BF16:
    case W4A8_ISA_AVX512_VNNI:
        g_dot     = w4a8_dot_avx512_vnni;
        g_current = W4A8_ISA_AVX512_VNNI;
        break;
    case W4A8_ISA_AVX512:
    case W4A8_ISA_AVX2:
    case W4A8_ISA_SCALAR:
    default:
        g_dot     = w4a8_dot_scalar;
        g_current = W4A8_ISA_SCALAR;
        break;
    }

    g_inited = 1;
    return g_current;
}

enum w4a8_isa w4a8_dispatcher_current(void) {
    return g_current;
}

[[nodiscard]] float w4a8_dot(size_t        n_blocks,
                             const uint8_t weights[static n_blocks * W4A8_BLOCK_BYTES_WEIGHTS],
                             const float   w_scales[static n_blocks],
                             const float   w_offsets[static n_blocks],
                             const int8_t  acts[static n_blocks * W4A8_BLOCK_ELEMS],
                             const int32_t sum_a_per_block[static n_blocks],
                             float         scale_x) {
    /* Lazy init: any caller hitting w4a8_dot before the backend bind path
     * still gets a correct dispatch. Subsequent inits are no-ops. */
    if (g_inited == 0) {
        (void) w4a8_dispatcher_init();
    }
    return g_dot(n_blocks, weights, w_scales, w_offsets, acts, sum_a_per_block, scale_x);
}

void w4a8_gemv(size_t        n_rows,
               size_t        n_blocks_per_row,
               const uint8_t weights[static n_rows * n_blocks_per_row * W4A8_BLOCK_BYTES_WEIGHTS],
               const float   w_scales[static n_rows * n_blocks_per_row],
               const float   w_offsets[static n_rows * n_blocks_per_row],
               const int8_t  acts[static n_blocks_per_row * W4A8_BLOCK_ELEMS],
               const int32_t sum_a_per_block[static n_blocks_per_row],
               float         scale_x,
               float         out[static n_rows]) {
    /* Ensure the dispatcher is wired before the OMP region: lazy init from
     * inside #pragma omp parallel would race on first-use. */
    if (g_inited == 0) {
        (void) w4a8_dispatcher_init();
    }
    const size_t bytes_per_row  = n_blocks_per_row * W4A8_BLOCK_BYTES_WEIGHTS;
    const size_t scales_per_row = n_blocks_per_row;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t m = 0; m < n_rows; m++) {
        const uint8_t *w_row = weights + m * bytes_per_row;
        const float   *s_row = w_scales + m * scales_per_row;
        const float   *o_row = w_offsets + m * scales_per_row;
        out[m] = g_dot(n_blocks_per_row, w_row, s_row, o_row, acts, sum_a_per_block, scale_x);
    }
}

[[nodiscard]] float
w4a8_quantize_acts_row(size_t      n_in,
                       const float x[static n_in],
                       int8_t      acts_out[static n_in],
                       int32_t     sum_a_per_block_out[static n_in / W4A8_BLOCK_ELEMS]) {
    const float scale_x = quantize_x_int8_sym(x, n_in, acts_out);

    /* Single pass over the freshly-written int8 buffer (hot in L1) to
     * compute per-block sums. The block stride matches W4A8's 32-element
     * quantum from kernel_w4a8.h. */
    const size_t n_blocks = n_in / W4A8_BLOCK_ELEMS;
    for (size_t b = 0; b < n_blocks; b++) {
        const int8_t *blk = acts_out + b * W4A8_BLOCK_ELEMS;
        int32_t       s   = 0;
        for (size_t i = 0; i < W4A8_BLOCK_ELEMS; i++) {
            s += (int32_t) blk[i];
        }
        sum_a_per_block_out[b] = s;
    }
    return scale_x;
}

const char *w4a8_isa_name(enum w4a8_isa isa) {
    switch (isa) {
    case W4A8_ISA_SCALAR:
        return "scalar";
    case W4A8_ISA_AVX2:
        return "avx2";
    case W4A8_ISA_AVX512:
        return "avx512";
    case W4A8_ISA_AVX512_VNNI:
        return "avx512_vnni";
    case W4A8_ISA_AVX512_BF16:
        return "avx512_bf16";
    default:
        return "?";
    }
}
