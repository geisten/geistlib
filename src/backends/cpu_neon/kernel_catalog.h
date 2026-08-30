/*
 * src/backends/cpu_neon/kernel_catalog.h - cpu_neon kernel policy.
 *
 * The catalog turns hardware facts + environment overrides into resolver
 * choices. Kernel implementations stay in weight_resolve.c; this layer
 * keeps platform selection out of the dtype switch.
 *
 * Hardware bits (has_dotprod, has_fp16) carry the RUNTIME probe result
 * so the resolver can refuse to install an ISA-incompatible kernel at
 * load time instead of SIGILL'ing at decode. They are not env-toggleable.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_KERNEL_CATALOG_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_KERNEL_CATALOG_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_neon/kernel_catalog.h is internal to the backend layer."
#endif

#include "hw_probe.h"
#include <geist_types.h>

#include <geist.h>        /* enum geist_dtype */
#include <geist_weight.h> /* struct geist_weight */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Required-ISA bitmask. Used by the kernel-entry table to declare
 * "this row uses vdotq_s32" / "this row needs hardware fp16" etc.,
 * checked once at resolve time against the runtime hw_probe so a
 * binary cross-built with a feature its host lacks fails-loudly at
 * model load instead of SIGILL'ing at first decode. */
typedef uint32_t cpu_neon_isa_mask;
enum {
    CPU_NEON_ISA_NEON    = 1u << 0,
    CPU_NEON_ISA_DOTPROD = 1u << 1,
    CPU_NEON_ISA_FP16    = 1u << 2,
};

/* Kernel-table entry. The resolver scans CPU_NEON_KERNELS in order and
 * installs the first row whose dtype matches the requested dtype AND
 * whose `requires` bitmask is a subset of the active host's ISA mask.
 *
 * `linear_m1` and `linear_mN` are the decode (M=1) and prefill (M>1)
 * function pointers stored on the resolved weight. Either may be the
 * trampoline (cpu_neon_w_dequant_trampoline_*); both must be non-null.
 *
 * `name` is for diagnostic logging only. */
struct cpu_neon_kernel_entry {
    enum geist_dtype dtype;
    cpu_neon_isa_mask
        requires;
    /* The canonical types from geist_weight.h, not a hand-spelled copy.
     * This table re-declared the signature inline and drifted out of step
     * with the typedef the moment the parameter order changed; naming the
     * typedef makes that impossible. */
    geist_kernel_linear_m1_fn linear_m1;
    geist_kernel_linear_mN_fn linear_mN;
    const char               *name;
};

struct cpu_neon_kernel_policy {
    /* Runtime ISA facts — set from hw_probe, never from env. */
    bool has_dotprod;
    bool has_fp16;

    /* Tunables — platform default + GEIST_* env overrides. */
    bool q5k_native_mn;
    bool q4k_predecode;
    bool q4k_mtile_prefill;
    bool q4k_ntile_prefill;
    bool q4k_block_q8_prefill;
    /* Mac AMX/Accelerate dequant→cblas_sgemm prefill path. When set AND
     * m ≥ qk_sgemm_threshold, the Q4_K/Q6_K W*A8 NEON kernels are bypassed
     * in favor of dequant_q?_K_row + cblas_sgemm (tiled DEQ_TILE_ROWS=32).
     * Default true when has_accelerate (Mac); false on Pi 5/Linux. */
    bool q4k_sgemm_prefill;
    bool q6k_sgemm_prefill;
    /* M-threshold for SGEMM-path crossover. Default 32 (empirical Mac M1
     * crossover where dequant overhead is amortized by AMX speedup).
     * Env: GEIST_QK_SGEMM_THRESHOLD. */
    size_t qk_sgemm_threshold;
    /* Output rows per dequant→SGEMM tile. Default 64; larger tiles reduce
     * cblas call overhead, smaller tiles reduce fp32 scratch/cache pressure.
     * Env: GEIST_QK_SGEMM_TILE_ROWS. */
    size_t qk_sgemm_tile_rows;
    bool   q6k_ntile_prefill;
    /* Interleaved-8-row Q6_K decode GEMV (packed heap copy of the
     * tensor; installed only for n_out >= 32768 = lm_head class).
     * Default on where Accelerate implies desktop-class RAM; opt-in
     * elsewhere. Env: GEIST_Q6K_X8_GEMV. */
    bool q6k_x8_gemv;
    /* Same interleaved-8-row GEMV for Q4_0 — applies to every eligible
     * Q4_0 tensor (FFN/attention/DN projections), so the packed copies
     * sum to ~1x the model's Q4_0 bytes on heap; the mmap'd source goes
     * cold after warmup. Env: GEIST_Q4_0_X8_GEMV. */
    bool q4_0_x8_gemv;
    bool q6k_ntile4_stream_prefill;
    bool q8_0_native_mn;
    bool q4_01_native_mn;
    bool iq4xs_native_mn;
    bool tq2_0_native_mn;
    bool tq2_0_tl1_m1;
};

struct cpu_neon_kernel_policy cpu_neon_kernel_policy_default(const struct geist_hw_probe *hw);

bool cpu_neon_should_install_tl1(const struct cpu_neon_kernel_policy *policy,
                                 size_t                               n_in,
                                 size_t                               n_out,
                                 size_t                               tl1_bytes);

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_KERNEL_CATALOG_H */
