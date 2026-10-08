/*
 * src/backends/cpu_x86/backend_state.h — per-instance state held in
 * geist_backend->state.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Kernel scratch (int8 activations + per-block sum_a, written by
 * w4a8_quantize_acts_row and read by the GEMV within one linear call) is
 * per calling thread, since concurrent sessions may share one backend
 * (geist_arch.h thread contract). The backend owns and frees every node;
 * buffers grow on demand at kernel time.
 *
 * Lifecycle:
 *   - cpu_x86_create (in backend.c) zero-inits the state and mints a
 *     ws_generation so a recycled state address can't satisfy another
 *     state's thread-local cache entry.
 *   - kernels call cpu_x86_ws_acquire(st, n_in), which finds or mints
 *     the calling thread's workspace and ensures scratch_cap >= n_in.
 *   - cpu_x86_destroy walks ws_head and frees every node.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_BACKEND_STATE_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_BACKEND_STATE_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/backend_state.h is internal to the backend layer."
#endif

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

struct cpu_x86_workspace {
    int8_t  *acts_scratch;  /* int8 activation buffer; heap_alloc_aligned. */
    int32_t *sum_a_scratch; /* per-block sum_a int32 buffer; heap-aligned. */
    size_t   scratch_cap;   /* max n_in (in fp32 elements) the scratch covers. */
    /* M>1 (prefill) high-water scratch, aligned for AVX-512 loads. Four
     * buffers rather than one arena: each has its own element type, and
     * `mN_aux` stays live alongside `mN_acts` (i2s_gemm_avx512_vnni permutes
     * one into the other). Capacities are in bytes. */
    int8_t  *mN_acts;
    size_t   mN_acts_cap;
    int32_t *mN_sum_a;
    size_t   mN_sum_a_cap;
    float   *mN_scale;
    size_t   mN_scale_cap;
    uint8_t *mN_aux;
    size_t   mN_aux_cap;
    /* Split-decode partials of the INT8-KV attention (attention_int8.c). */
    float *attn_part;
    size_t attn_part_cap; /* bytes */
};

/* One node per (backend, thread) pair; ws_head is a lock-free push-only
 * list so acquire never blocks a concurrent session. */
struct cpu_x86_ws_node {
    pthread_t tid;
    struct cpu_x86_ws_node *_Atomic next;
    struct cpu_x86_workspace ws;
};

struct cpu_x86_state {
    int      max_threads; /* geist_backend_opts.max_threads: caps every region; 0 = no cap */
    uint64_t ws_generation;
    struct cpu_x86_ws_node *_Atomic ws_head;
};

uint64_t cpu_x86_ws_next_generation(void);

/* The calling thread's workspace with scratch_cap >= n_in, or nullptr on
 * OOM (callers zero their output and return, matching cpu_neon). */
struct cpu_x86_workspace *cpu_x86_ws_acquire(struct cpu_x86_state *st, size_t n_in);

/* The calling thread's workspace with the M>1 scratch grown to cover the
 * requested byte counts (0 = buffer unused), or nullptr on OOM or size
 * overflow; callers then fall back to their M=1 loop. */
struct cpu_x86_workspace *cpu_x86_ws_acquire_mN(struct cpu_x86_state *st,
                                                size_t                acts_bytes,
                                                size_t                sum_a_bytes,
                                                size_t                scale_bytes,
                                                size_t                aux_bytes);

/* The calling thread's attention part buffer, grown to `floats` floats;
 * nullptr if that cannot be had (the caller then runs unsplit). */
float *cpu_x86_ws_attn_part(struct cpu_x86_state *st, size_t floats);

void cpu_x86_ws_destroy_all(struct cpu_x86_state *st);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_BACKEND_STATE_H */
