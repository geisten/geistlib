/*
 * src/base/hw_probe.h - runtime hardware feature summary.
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
    bool has_fma;
    bool has_bmi2;
    bool has_avx512f;
    bool has_avx512_vnni;
    bool has_amx_int8;
    bool has_accelerate;
    bool has_openmp;
    /* Running as a guest under a hypervisor. Probed on Apple silicon only
     * (kern.hv_vmm_present), false elsewhere. The guest has no AMX, so
     * Accelerate's sgemm runs at NEON speed there (#456). */
    bool is_virtual_machine;

    size_t logical_cores;  /* 0 when unknown. */
    size_t physical_cores; /* 0 when unknown. SMT collapsed (Linux only today). */
    size_t n_l3_domains;   /* 0 unknown, 1 = single L3, N = AMD multi-CCD / Intel
                            * P/E cluster. Part of the calibration key only:
                            * pinning decode to one L3 domain measured slower
                            * (benchmark/results/X86.md). */

    /* Normalized µarch identity of THIS machine: core-type fingerprint
     * plus topology, e.g. "arm64:41.d0b*4" (Pi 5, 4x Cortex-A76) or
     * "apple:Apple-M1-Max:8P+2E". big.LITTLE shows up as a '+'-joined
     * multiset. Consumed by the calibration key — treat as opaque
     * outside the probe; the format may grow more factors. Empty when
     * no identity source exists (calibration then refuses to key). */
    char uarch[96];
};

void geist_hw_probe_fill(struct geist_hw_probe *out);

/* The instruction-set bits only (has_neon ... has_amx_int8), everything
 * else zero: no /sys reads, so cheap enough to run on every backend
 * create. geist_hw_probe_fill starts from it. */
void geist_hw_probe_isa(struct geist_hw_probe *out);

/* The first instruction-set feature this build may execute that `hw` does
 * not have, or nullptr when it has them all. The target flags
 * (mk/target-*.mk: -march=armv8.2-a+fp16+dotprod, -mcpu=cortex-a76,
 * -march=x86-64-v3) let the compiler use a feature in any function, not
 * only in the kernels written for it, so no kernel table can route around
 * a missing one: the host needs every one of them. geist_backend_create
 * refuses a host that lacks one, before any code that might use it runs. */
const char *geist_hw_build_isa_missing(const struct geist_hw_probe *hw);

#endif /* GEIST_INTERNAL_HW_PROBE_H */
