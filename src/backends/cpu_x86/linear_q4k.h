/*
 * src/backends/cpu_x86/linear_q4k.h — cpu_x86 Q4_K linear M=1 (decode) path.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * The resolver in backend.c calls cpu_x86_linear_q4k_resolve() per Q4_K
 * weight: it predecodes the GGUF Q4_K layout into the W4A8 SoA the inner
 * kernel consumes (one allocation per weight via heap.h, stored in
 * w->aux_fp32 reinterpreted as a byte blob), grows the backend's
 * activation scratch if needed, and installs cpu_x86_linear_q4k_m1 as
 * w->linear_m1.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4K_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4K_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_q4k.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_weight.h>

#include <stddef.h>

struct cpu_x86_state;

/* Predecode one Q4_K weight to the W4A8 layout, install the M=1 kernel
 * pointer, and grow the per-backend activation scratch if needed.
 *
 * Returns:
 *   GEIST_OK          — weight predecoded, scratch ensured, kernel pointer set.
 *   GEIST_E_OOM       — heap.h allocation failed; w->aux_fp32 left unset.
 *   GEIST_E_INVALID_ARG — w->n_in not a positive multiple of Q4_K_BLOCK_ELEMS.
 *
 * Caller: cpu_x86_resolve_weight in backend.c.
 */
[[nodiscard]] enum geist_status cpu_x86_linear_q4k_resolve(struct cpu_x86_state *st,
                                                           struct geist_weight  *w);

/* The M=1 (decode) kernel installed into w->linear_m1 by the resolver. */
void cpu_x86_linear_q4k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y);

/* The M>1 (prefill) kernel installed into w->linear_mN by the resolver.
 *
 * Phase-1b-Step-1 implementation: serial loop calling the M=1 kernel
 * m times. Correct but does not amortize the weight-read across the
 * batch (every row re-streams the W4A8 SoA). The next iteration in
 * Phase 1b fuses the inner so each weight block is read once per
 * m-tile; benchmarks against cpu_scalar should still show a large win
 * here because cpu_scalar dequants Q4_K → fp32 per row inside the
 * inner. */
void cpu_x86_linear_q4k_mN(
        const float *x, const struct geist_weight *w, size_t m, struct geist_backend *be, float *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4K_H */
