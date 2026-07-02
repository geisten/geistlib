/*
 * src/backends/cpu_neon/kernel_catalog.c - cpu_neon kernel policy.
 *
 * Strict env parsing convention: a knob is "on" iff its value starts with
 * '1'. This matches the legacy per-knob `env[0] == '1'` checks that
 * existed before the catalog was introduced — anything else (empty,
 * 'true', 'yes', '0') is treated as "use the platform default". Bools
 * that need an explicit "force off" use a separate _FORCE-off var by
 * convention rather than overloading truthiness.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_catalog.h"

#include <stdlib.h>

bool cpu_neon_env_bool(const char *name, bool fallback) {
    const char *env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    /* Strict: only "1..." is true. Matches the pre-catalog convention so
     * scripts that exported `GEIST_*_NATIVE_MN=true` continue to mean
     * "fallback (platform default)" rather than silently flipping to
     * native NEON. */
    return env[0] == '1';
}

struct cpu_neon_kernel_policy cpu_neon_kernel_policy_default(const struct geist_hw_probe *hw) {

    const bool has_accelerate = hw != nullptr && hw->has_accelerate;

    struct cpu_neon_kernel_policy p = {
            /* Runtime ISA facts — surfaced verbatim from the probe so the
             * resolver can refuse incompatible kernels. */
            .has_dotprod = hw != nullptr && hw->has_dotprod,
            .has_fp16    = hw != nullptr && hw->has_fp16,

            /* Apple AMX/Accelerate wins for high-M dequant+SGEMM; Pi/Linux
             * wins with native per-row NEON kernels for these tile sizes. */
            .q5k_native_mn        = !has_accelerate,
            .q4k_predecode        = has_accelerate,
            .q4k_mtile_prefill    = has_accelerate,
            .q4k_ntile_prefill    = has_accelerate,
            .q4k_block_q8_prefill = false,
            /* The Pi5 OpenBLAS SGEMM-prefill path is currently not
             * quality-safe for Gemma Q4_K/Q6_K end-to-end generation. Keep
             * it opt-in via GEIST_Q{4,6}K_SGEMM_PREFILL=1 until the
             * dequant+sgemm path has a full-logits parity gate on Pi. */
            .q4k_sgemm_prefill         = has_accelerate,
            .q6k_sgemm_prefill         = has_accelerate,
            .qk_sgemm_threshold        = has_accelerate ? 64 : 32,
            .qk_sgemm_tile_rows        = 64,
            .q6k_ntile_prefill         = has_accelerate,
            .q6k_ntile4_stream_prefill = has_accelerate,
            .q8_0_native_mn            = !has_accelerate,
            .tq2_0_native_mn           = !has_accelerate,
            .tq2_0_tl1_m1              = false,
    };

    p.q5k_native_mn     = cpu_neon_env_bool("GEIST_Q5K_NATIVE_MN", p.q5k_native_mn);
    p.q4k_predecode     = cpu_neon_env_bool("GEIST_Q4K_PREDECODE", p.q4k_predecode);
    p.q4k_mtile_prefill = cpu_neon_env_bool("GEIST_Q4K_MTILE_PREFILL", p.q4k_mtile_prefill);
    p.q4k_ntile_prefill = cpu_neon_env_bool("GEIST_Q4K_NTILE_PREFILL", p.q4k_ntile_prefill);
    p.q4k_block_q8_prefill =
            cpu_neon_env_bool("GEIST_Q4K_BLOCK_Q8_PREFILL", p.q4k_block_q8_prefill);
    p.q6k_ntile_prefill = cpu_neon_env_bool("GEIST_Q6K_NTILE_PREFILL", p.q6k_ntile_prefill);
    p.q6k_ntile4_stream_prefill =
            cpu_neon_env_bool("GEIST_Q6K_NTILE4_STREAM_PREFILL", p.q6k_ntile4_stream_prefill);
    p.q4k_sgemm_prefill = cpu_neon_env_bool("GEIST_Q4K_SGEMM_PREFILL", p.q4k_sgemm_prefill);
    p.q6k_sgemm_prefill = cpu_neon_env_bool("GEIST_Q6K_SGEMM_PREFILL", p.q6k_sgemm_prefill);
    const char *thresh  = getenv("GEIST_QK_SGEMM_THRESHOLD");
    if (thresh != nullptr && *thresh != '\0') {
        char               *end = nullptr;
        const unsigned long v   = strtoul(thresh, &end, 10);
        if (end != thresh && v > 0 && v < (1u << 20)) {
            p.qk_sgemm_threshold = (size_t) v;
        }
    }
    const char *tile_rows = getenv("GEIST_QK_SGEMM_TILE_ROWS");
    if (tile_rows != nullptr && *tile_rows != '\0') {
        char               *end = nullptr;
        const unsigned long v   = strtoul(tile_rows, &end, 10);
        if (end != tile_rows && v >= 4 && v <= 256) {
            p.qk_sgemm_tile_rows = (size_t) v;
        }
    }
    p.q8_0_native_mn  = cpu_neon_env_bool("GEIST_Q8_0_NATIVE_MN", p.q8_0_native_mn);
    p.tq2_0_native_mn = cpu_neon_env_bool("GEIST_TQ2_0_NATIVE_MN", p.tq2_0_native_mn);
    p.tq2_0_tl1_m1    = cpu_neon_env_bool("GEIST_TL1", p.tq2_0_tl1_m1);
    return p;
}

bool cpu_neon_should_install_tl1(const struct cpu_neon_kernel_policy *policy,
                                 size_t                               n_in,
                                 size_t                               n_out,
                                 size_t                               tl1_bytes) {
    if (policy == nullptr || !policy->tq2_0_tl1_m1) {
        return false;
    }
    return n_in > 0 && n_out > 0 && tl1_bytes > 0;
}
