/*
 * src/archs/transformer/prefill_tuning.h — per-model prefill knobs.
 *
 * Layer: ARCHITECTURE (transformer).
 *
 * Two knobs decide how a prompt is prefilled on a CPU backend, and the
 * right value for both depends on the model, not only on the machine:
 *
 *   m_max      rows per prefill chunk. Every chunk dequantizes every
 *              weight again, so a bigger chunk amortizes that over more
 *              tokens and feeds the GEMM a taller batch — until the
 *              activation scratch stops fitting. Measured on the 27B
 *              ternary Bonsai (M1 Max, pp512): 64 -> 128 is +5-14 %, 256
 *              gains nothing more. Small models do not pay the dequant
 *              enough to notice.
 *   blocktime  how long an idle OpenMP worker spins before it sleeps,
 *              in ms (libomp reads KMP_BLOCKTIME once, at its first
 *              parallel region). Infinite (the OMP_WAIT_POLICY=active the
 *              cpu_neon backend asks for) is what decode wants: ~200 tiny
 *              regions per token, a sleeping worker's wake-up costs more
 *              than the region. A prefill SGEMM runs 30-50 ms though, and
 *              on Apple Accelerate threads it on the very cores the other
 *              workers spin on: KMP_BLOCKTIME=0 measured +21-33 % prefill
 *              on the 27B, and -16 % decode on a 0.8B. Switching it per
 *              phase at run time (kmp_set_blocktime) measured nothing but
 *              its own cost, so the knob is one value per process, applied
 *              from the table of the FIRST model loaded, before its weight
 *              packing runs the first parallel region.
 *
 * Resolution is DEFAULT + DELTA + OVERRIDE, in that order:
 *   default   what the platform and backend already choose (arch_state.c:
 *             64, or caps.preferred_m_max); blocktime 0
 *   delta     the row below matching (family, weight bytes): a signed
 *             number of rows added to m_max, and a blocktime for that
 *             model class (-1 = keep the default)
 *   override  GEIST_M_MAX and GEIST_PREFILL_BLOCKTIME_MS, absolute
 *
 * A row with family "*" matches every family; the first match wins, so
 * specific rows go first. Sizes are the GGUF's tensor bytes, the one
 * number every model carries without a parameter count in its metadata.
 */
#ifndef GEIST_INTERNAL_TRANSFORMER_PREFILL_TUNING_H
#define GEIST_INTERNAL_TRANSFORMER_PREFILL_TUNING_H

#include <stddef.h>

struct transformer_prefill_tuning {
    const char *family;    /* arch family name, or "*" */
    size_t      min_bytes; /* weight bytes, inclusive */
    size_t      max_bytes; /* exclusive; SIZE_MAX = open */
    int         m_max_delta;
    int         prefill_blocktime_ms; /* -1: default */
};

struct transformer_prefill_resolved {
    size_t m_max;
    int    prefill_blocktime_ms; /* -1: leave the runtime's policy alone */
    bool   m_max_from_env;
};

/* Resolves both knobs for (family, weight_bytes) from `base_m_max` and
 * the backend's row cap (0 = uncapped), reading GEIST_M_MAX and
 * GEIST_PREFILL_BLOCKTIME_MS from `getenv_fn` (nullptr = getenv; tests
 * inject). Never fails: an unknown model gets the defaults. */
/* Applies the resolved blocktime as the process's KMP_BLOCKTIME. libomp
 * reads it once, at its first parallel region, so this has to run before
 * weight packing; an explicit KMP_BLOCKTIME in the environment wins, and
 * so does the first model's value in a process that loads several. -1
 * leaves everything alone. */
void transformer_prefill_apply_blocktime(const struct transformer_prefill_resolved *r);

struct transformer_prefill_resolved
transformer_prefill_resolve(const char *family,
                            size_t      weight_bytes,
                            size_t      base_m_max,
                            size_t      cap_m_max,
                            const char *(*getenv_fn)(const char *) );

#endif /* GEIST_INTERNAL_TRANSFORMER_PREFILL_TUNING_H */
