/*
 * src/engine/hw_probe.h - runtime hardware feature summary.
 *
 * Layer: ENGINE.
 *
 * All feature bits (has_neon / has_dotprod / has_fp16) reflect the
 * CURRENT host as probed via getauxval (Linux) / sysctlbyname (macOS) /
 * __builtin_cpu_supports (x86_64). Compile-time macros are used only as
 * a last-resort fallback on platforms without a runtime probe source.
 *
 * `has_accelerate` is a build-time link decision and remains compile-time.
 *
 * `logical_cores` is the OS hardware core count, not the OpenMP team
 * size (use omp_get_max_threads() if you want OMP_NUM_THREADS).
 *
 * Backends consult this at create time so the resolver can refuse to
 * install an ISA-incompatible kernel — preventing SIGILL at decode on
 * cross-built binaries.
 */
#ifndef GEIST_INTERNAL_HW_PROBE_H
#define GEIST_INTERNAL_HW_PROBE_H

#include <stdbool.h>
#include <stddef.h>

enum geist_hw_os {
    GEIST_HW_OS_UNKNOWN = 0,
    GEIST_HW_OS_MACOS,
    GEIST_HW_OS_LINUX,
    GEIST_HW_OS_WINDOWS,
};

enum geist_hw_cpu {
    GEIST_HW_CPU_UNKNOWN = 0,
    GEIST_HW_CPU_ARM64_GENERIC,
    GEIST_HW_CPU_APPLE_SILICON,
    GEIST_HW_CPU_X86_64_GENERIC,
};

struct geist_hw_probe {
    enum geist_hw_os  os;
    enum geist_hw_cpu cpu;

    bool is_apple_silicon;
    bool has_neon;
    bool has_dotprod;
    bool has_fp16;
    bool has_avx2;
    bool has_avx512f;
    bool has_avx512_vnni;
    bool has_amx_int8;
    bool has_accelerate;
    bool has_openmp;

    size_t logical_cores;  /* 0 when unknown. */
    size_t physical_cores; /* 0 when unknown. SMT collapsed (Linux only today). */
    size_t n_l3_domains;   /* 0 unknown, 1 = single L3, N = AMD multi-CCD / Intel
                            * P/E cluster. Used by Phase-1a CCD-aware threading
                            * (see docs/LINUX_X86_SPEC.md). */
};

void geist_hw_probe_fill(struct geist_hw_probe *out);

#endif /* GEIST_INTERNAL_HW_PROBE_H */
