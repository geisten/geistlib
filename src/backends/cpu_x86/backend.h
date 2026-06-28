/*
 * src/backends/cpu_x86/backend.h — x86_64 backend (AVX2 / AVX-512 / +VNNI / +BF16).
 *
 * Layer: BACKEND.
 *
 * Phase 0 (this commit): empty shell. The descriptor exists and the
 * registry slot is wired (gated by GEIST_BACKEND_CPU_X86), but the vtbl
 * points to cpu_scalar's implementation — i.e. cpu_x86 today is a renamed
 * cpu_scalar. Phase 1a/1b/2 (see docs/LINUX_X86_SPEC.md) replace individual
 * vtbl slots with native VPDPBUSD / VDPBF16PS kernels.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/backend.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_types.h>

struct geist_backend_descriptor;

extern const struct geist_backend_descriptor geist_backend_cpu_x86;

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_H */
