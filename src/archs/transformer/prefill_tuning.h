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
 *              activation scratch stops fitting. The 27B ternary Bonsai
 *              gains from 64 -> 128, not beyond; small models do not pay
 *              the dequant enough to notice.
 *   blocktime  how long an idle OpenMP worker spins before it sleeps,
 *              in ms (KMP_BLOCKTIME; libomp reads it once, at its first
 *              OpenMP API call). The default is the idle bound of
 *              omp_idle.h: long enough for decode's ~200 tiny regions per
 *              token, short enough that a waiting process goes idle. A
 *              prefill SGEMM runs 30-50 ms though, and on Apple Accelerate
 *              threads it on the very cores the other workers spin on:
 *              KMP_BLOCKTIME=0 speeds up the 27B's prefill (-9.6 %) but
 *              slows its decode (+18.3 %) and a 0.8B's decode more still,
 *              so no model sets it today. Switching it per phase at run time
 *              (kmp_set_blocktime) costs more than it saves, so the knob
 *              is one value per process, taken from the FIRST model
 *              loaded, before its weight packing runs the first parallel
 *              region.
 *
 * Resolution is DEFAULT + DELTA + OVERRIDE, in that order:
 *   default   what the platform and backend already choose (arch_state.c:
 *             64, or caps.preferred_m_max); blocktime GEIST_IDLE_SPIN_MS
 *   delta     per model class (prefill_tuning.c): rows added to m_max,
 *             and optionally a blocktime; today only qwen35 from 4 GiB up
 *             has one (+64 rows, no blocktime of its own)
 *   override  GEIST_M_MAX and GEIST_PREFILL_BLOCKTIME_MS, absolute
 *
 * Sizes are the GGUF's tensor bytes, the one number every model carries
 * without a parameter count in its metadata.
 */
#ifndef GEIST_INTERNAL_TRANSFORMER_PREFILL_TUNING_H
#define GEIST_INTERNAL_TRANSFORMER_PREFILL_TUNING_H

#include <stddef.h>

struct transformer_prefill_resolved {
    size_t m_max;
    int    prefill_blocktime_ms; /* >= 0 */
    bool   m_max_from_env;
};

/* Applies the resolved blocktime as the process's KMP_BLOCKTIME
 * (geist_omp_blocktime_apply): before weight packing, replacing the idle
 * default but never an explicit KMP_BLOCKTIME; the first model decides. */
void transformer_prefill_apply_blocktime(const struct transformer_prefill_resolved *r);

/* Resolves both knobs for (family, weight_bytes) from `base_m_max` and
 * the backend's row cap (0 = uncapped), reading GEIST_M_MAX,
 * GEIST_PREFILL_BLOCKTIME_MS and GEIST_IDLE_SPIN_MS from `getenv_fn`
 * (nullptr = getenv; tests inject). Never fails: an unknown model gets
 * the defaults. */
struct transformer_prefill_resolved
transformer_prefill_resolve(const char *family,
                            size_t      weight_bytes,
                            size_t      base_m_max,
                            size_t      cap_m_max,
                            const char *(*getenv_fn)(const char *) );

#endif /* GEIST_INTERNAL_TRANSFORMER_PREFILL_TUNING_H */
