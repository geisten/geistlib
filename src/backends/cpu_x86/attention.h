/*
 * src/backends/cpu_x86/attention.h — cpu_x86 attention override.
 *
 * Layer: BACKEND (cpu_x86, internal). See attention.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/attention.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_backend.h>

#include <stddef.h>

[[nodiscard]] enum geist_status cpu_x86_attention(struct geist_backend      *be,
                                                  const struct geist_tensor *q,
                                                  const struct geist_tensor *k,
                                                  const struct geist_tensor *v,
                                                  size_t                     q_offset,
                                                  size_t                     sliding_window,
                                                  struct geist_tensor       *out);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_H */
