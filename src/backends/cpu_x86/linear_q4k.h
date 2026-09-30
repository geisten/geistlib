/*
 * src/backends/cpu_x86/linear_q4k.h — cpu_x86 Q4_K linear (decode and prefill).
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * The resolver in backend.c calls cpu_x86_linear_q4k_resolve() per Q4_K
 * weight: it repacks the GGUF Q4_K layout into the one layout the kernels
 * read for that shape (Q4_Kx8 when n_out % 8 == 0, else the W4A8 SoA; one
 * allocation per weight via heap.h, stored in w->aux_fp32 reinterpreted
 * as a byte blob) and installs cpu_x86_linear_q4k_m1 / _mN. The
 * activation scratch comes from the per-thread workspace at call time.
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

/* Repack one Q4_K weight (see above) and install the M=1 and M>1 kernel
 * pointers.
 *
 * Returns:
 *   GEIST_OK          — weight repacked, kernel pointers set.
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

/* The M>1 (prefill) kernel installed into w->linear_mN by the resolver:
 * the Q4_Kx8 GEMM over whole groups of 4 rows, the M=1 kernel for the
 * last m % 4 rows and for a weight in the W4A8 layout. */
void cpu_x86_linear_q4k_mN(
        size_t m, const float *x, const struct geist_weight *w, struct geist_backend *be, float *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_Q4K_H */
