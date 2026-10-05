/*
 * src/backends/cpu_x86/linear_util.h — small inline helpers the cpu_x86
 * linear kernels share: horizontal sums, the int8 activation quantizer's
 * steps, the activation workspace, the VNNI gate and the OpenMP team.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Everything here is static inline, so each TU compiles what it uses with its
 * own ISA flags and nothing becomes a call across TUs. The *_avx512_vnni.c
 * TUs may include this header for hsum512_ps; vnni_tiles_usable is not
 * visible there, because the decision whether those TUs may run has to stay
 * outside them (mk/backend-cpu_x86.mk).
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_UTIL_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_UTIL_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/linear_util.h is internal to the backend layer."
#endif

#include "backend_state.h"
#include "kernel_w4a8.h" /* w4a8_dispatcher_tier: the ISA gate, GEIST_FORCE_ISA-clamped */

#include "checked.h"

#include <geist_backend.h>

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Sum of 8 lanes: halves, then 4 -> 2 -> 1. A fixed tree, so a row's result
 * does not depend on anything but its inputs. */
static inline float hsum_ps(__m256 s) {
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4        = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4        = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    return _mm_cvtss_f32(s4);
}

/* Sum of 8 lanes by two hadds: a different tree from hsum_ps (pairs of
 * neighbours first), kept for the kernels whose results were fixed with it. */
static inline float hsum_ps_hadd(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128       s  = _mm_add_ps(lo, hi);
    s               = _mm_hadd_ps(s, s);
    s               = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

#if defined(__AVX512F__) && defined(__AVX512DQ__)
/* Sum of 16 lanes: the halves added, then hsum_ps. */
static inline float hsum512_ps(__m512 v) {
    return hsum_ps(_mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1)));
}
#endif

/* The calling thread's workspace with room for m quantized activation rows
 * of n_in: int8 values, one fp32 scale per scale_block elements and, unless
 * sum_block is 0, one int32 sum per sum_block elements. nullptr when the
 * backend has no state, on overflow, or on OOM. */
static inline struct cpu_x86_workspace *acquire_acts(
        struct geist_backend *be, size_t m, size_t n_in, size_t scale_block, size_t sum_block) {
    size_t acts_bytes = 0, n_scales = 0, scale_bytes = 0, n_sums = 0, sum_bytes = 0;
    if (be == nullptr || be->state == nullptr || ckd_mul(&acts_bytes, m, n_in) ||
        ckd_mul(&n_scales, m, n_in / scale_block) ||
        ckd_mul(&scale_bytes, n_scales, sizeof(float)) ||
        (sum_block != 0 &&
         (ckd_mul(&n_sums, m, n_in / sum_block) || ckd_mul(&sum_bytes, n_sums, sizeof(int32_t))))) {
        return nullptr;
    }
    return cpu_x86_ws_acquire_mN(
            (struct cpu_x86_state *) be->state, acts_bytes, sum_bytes, scale_bytes, 0);
}

#if !defined(__AVX512F__)
/* Whether this host may run the *_avx512_vnni.c tiles: the dispatcher tier
 * (which honours GEIST_FORCE_ISA) and every AVX-512 subset those TUs are
 * compiled for. Decided outside them — see mk/backend-cpu_x86.mk. */
static inline bool vnni_tiles_usable(void) {
    return w4a8_dispatcher_tier() >= W4A8_ISA_AVX512_VNNI && __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
}
#endif

/* The OpenMP team: the most threads a region may get, this thread's index,
 * and the size of the current one. 1 / 0 / 1 without OpenMP. */
static inline size_t team_max(void) {
#if defined(_OPENMP)
    return (size_t) omp_get_max_threads();
#else
    return 1;
#endif
}

static inline size_t team_id(void) {
#if defined(_OPENMP)
    return (size_t) omp_get_thread_num();
#else
    return 0;
#endif
}

static inline size_t team_size(void) {
#if defined(_OPENMP)
    return (size_t) omp_get_num_threads();
#else
    return 1;
#endif
}

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_LINEAR_UTIL_H */
