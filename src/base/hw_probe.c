/*
 * src/engine/hw_probe.c — runtime hardware feature summary.
 *
 * Every feature bit (NEON, dotprod, FP16, Accelerate) is probed at the
 * actual host where the process is running, not at the build host. This
 * is the contract the resolver and kernel_catalog rely on to refuse to
 * install ISA-incompatible kernels at model-load time, instead of
 * SIGILL'ing at decode.
 *
 * Detection sources:
 *   Linux / glibc       getauxval(AT_HWCAP / AT_HWCAP2)
 *   macOS               sysctlbyname("hw.optional.arm.FEAT_*")
 *   x86_64              __builtin_cpu_supports() (gcc / clang)
 *   Compile-time macros are used ONLY as the fallback when neither runtime
 *   source is available (BSD / bare ARM Linux without getauxval), and a
 *   conservative "absent" answer is preferred over a falsely-true one.
 *
 * `logical_cores` always reflects the OS hardware count when available;
 * OMP_NUM_THREADS / omp_get_max_threads is a thread-pool config, not a
 * hardware fact, and has no place in the probe.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "hw_probe.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/auxv.h>
/* HWCAP_ASIMDDP and HWCAP_ASIMDHP / HWCAP_FPHP are stable on aarch64. */
#if defined(__aarch64__)
#include <asm/hwcap.h>
#endif
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

/* Read a boolean sysctl on Apple; returns the value or `fallback` on failure. */
#if defined(__APPLE__)
static bool sysctl_bool(const char *name, bool fallback) {
    int    value = 0;
    size_t len   = sizeof(value);
    if (sysctlbyname(name, &value, &len, nullptr, 0) != 0) {
        return fallback;
    }
    return value != 0;
}
#endif

void geist_hw_probe_fill(struct geist_hw_probe *out) {
    if (out == nullptr) {
        return;
    }
    *out = (struct geist_hw_probe) {0};

    /* ----- OS ------------------------------------------------------- */
#if defined(__APPLE__)
    out->os = GEIST_HW_OS_MACOS;
#elif defined(__linux__)
    out->os = GEIST_HW_OS_LINUX;
#elif defined(_WIN32)
    out->os = GEIST_HW_OS_WINDOWS;
#else
    out->os = GEIST_HW_OS_UNKNOWN;
#endif

    /* ----- CPU family ---------------------------------------------- */
#if defined(__aarch64__) || defined(__arm64__)
#if defined(__APPLE__)
    out->cpu              = GEIST_HW_CPU_APPLE_SILICON;
    out->is_apple_silicon = true;
#else
    out->cpu = GEIST_HW_CPU_ARM64_GENERIC;
#endif
#elif defined(__x86_64__) || defined(_M_X64)
    out->cpu = GEIST_HW_CPU_X86_64_GENERIC;
#else
    out->cpu = GEIST_HW_CPU_UNKNOWN;
#endif

    /* ----- Feature bits — RUNTIME probe ---------------------------- */
#if defined(__aarch64__) && defined(__linux__)
    /* Linux/aarch64: glibc getauxval is the authoritative path. */
    {
        const unsigned long hw = getauxval(AT_HWCAP);
        /* HWCAP_ASIMD covers the NEON ISA; every aarch64 has it but we
         * still record it to keep the contract honest. */
#ifdef HWCAP_ASIMD
        if ((hw & HWCAP_ASIMD) != 0) {
            out->has_neon = true;
        }
#else
        out->has_neon = true; /* aarch64 baseline */
#endif
#ifdef HWCAP_ASIMDDP
        if ((hw & HWCAP_ASIMDDP) != 0) {
            out->has_dotprod = true;
        }
#endif
#ifdef HWCAP_FPHP
        if ((hw & HWCAP_FPHP) != 0) {
            out->has_fp16 = true;
        }
#endif
#ifdef HWCAP_ASIMDHP
        /* Either of FPHP or ASIMDHP indicates hardware fp16 arithmetic. */
        if ((hw & HWCAP_ASIMDHP) != 0) {
            out->has_fp16 = true;
        }
#endif
    }
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    /* Apple Silicon: sysctl is authoritative. M1+ has dotprod and fp16
     * unconditionally, but we still probe so a future M-series capability
     * lookup (e.g. SME / SVE) lands here uniformly. */
    out->has_neon    = true; /* baseline */
    out->has_dotprod = sysctl_bool("hw.optional.arm.FEAT_DotProd", true);
    out->has_fp16    = sysctl_bool("hw.optional.arm.FEAT_FP16", true);
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    /* x86_64 via compiler builtins. NEON / dotprod are ARM-only; record
     * them as false so any consumer that checks them on x86 fails-closed. */
    out->has_neon    = false;
    out->has_dotprod = false;
    out->has_fp16    = __builtin_cpu_supports("f16c");
    out->has_avx2    = __builtin_cpu_supports("avx2");
    out->has_avx512f = __builtin_cpu_supports("avx512f");
#if defined(__GNUC__) || defined(__clang__)
    out->has_avx512_vnni = __builtin_cpu_supports("avx512vnni");
    out->has_amx_int8    = __builtin_cpu_supports("amx-int8");
#endif
#else
    /* Compile-time fallback for platforms without a runtime probe path.
     * Conservative: only set what the compile target guarantees. */
#if defined(__ARM_NEON) || defined(__NEON__)
    out->has_neon = true;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    out->has_dotprod = true;
#endif
#if (defined(__ARM_FP) && (__ARM_FP & 2)) || defined(__ARM_FP16_FORMAT_IEEE)
    out->has_fp16 = true;
#endif
#endif

#if defined(HAVE_ACCELERATE)
    /* Accelerate is a build-time link decision, not a runtime fact. */
    out->has_accelerate = true;
#endif

#if defined(_OPENMP)
    out->has_openmp = true;
#endif

    /* ----- Logical cores — OS hardware count, NOT the OMP team ----- */
#if defined(_WIN32)
    {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        out->logical_cores = (size_t) info.dwNumberOfProcessors;
    }
#elif defined(_SC_NPROCESSORS_ONLN)
    {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n > 0) {
            out->logical_cores = (size_t) n;
        }
    }
#endif

    /* ----- Physical cores + L3 domains (CCDs) — Linux /sys ----------
     *
     * physical_cores = # of unique topology/thread_siblings_list strings.
     * n_l3_domains   = # of unique cache/index3/shared_cpu_list strings.
     *
     * On AMD 9950X this resolves to 16 physical cores, 2 L3 domains
     * (CCDs). On Pi 5 / single-socket Intel client it's 1 L3 domain.
     * Used by the Phase-1a CCD-aware threading default in cpu_x86
     * (decode pins to one L3 domain, prefill to all). */
#if defined(__linux__)
    if (out->logical_cores > 0 && out->logical_cores <= 256) {
        char   siblings_seen[64][64] = {0};
        char   l3_seen[16][64]       = {0};
        size_t n_physical            = 0;
        size_t n_l3                  = 0;
        for (size_t cpu = 0; cpu < out->logical_cores; cpu++) {
            char path[128];
            char buf[64];

            snprintf(path,
                     sizeof(path),
                     "/sys/devices/system/cpu/cpu%zu/topology/thread_siblings_list",
                     cpu);
            FILE *f = fopen(path, "r");
            if (f != nullptr) {
                if (fgets(buf, sizeof(buf), f) != nullptr) {
                    /* Strip newline. */
                    buf[strcspn(buf, "\n")] = '\0';
                    bool seen               = false;
                    for (size_t i = 0; i < n_physical && i < 64; i++) {
                        if (strcmp(siblings_seen[i], buf) == 0) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen && n_physical < 64) {
                        snprintf(siblings_seen[n_physical], 64, "%s", buf);
                        n_physical++;
                    }
                }
                fclose(f);
            }

            snprintf(path,
                     sizeof(path),
                     "/sys/devices/system/cpu/cpu%zu/cache/index3/shared_cpu_list",
                     cpu);
            f = fopen(path, "r");
            if (f != nullptr) {
                if (fgets(buf, sizeof(buf), f) != nullptr) {
                    buf[strcspn(buf, "\n")] = '\0';
                    bool seen               = false;
                    for (size_t i = 0; i < n_l3 && i < 16; i++) {
                        if (strcmp(l3_seen[i], buf) == 0) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen && n_l3 < 16) {
                        snprintf(l3_seen[n_l3], 64, "%s", buf);
                        n_l3++;
                    }
                }
                fclose(f);
            }
        }
        out->physical_cores = n_physical;
        out->n_l3_domains   = n_l3;
    }
#endif
}

const char *geist_hw_os_name(enum geist_hw_os os) {
    switch (os) {
    case GEIST_HW_OS_MACOS:
        return "macos";
    case GEIST_HW_OS_LINUX:
        return "linux";
    case GEIST_HW_OS_WINDOWS:
        return "windows";
    default:
        return "unknown";
    }
}

const char *geist_hw_cpu_name(enum geist_hw_cpu cpu) {
    switch (cpu) {
    case GEIST_HW_CPU_ARM64_GENERIC:
        return "arm64";
    case GEIST_HW_CPU_ARM64_CORTEX_A76:
        return "cortex-a76";
    case GEIST_HW_CPU_APPLE_SILICON:
        return "apple-silicon";
    case GEIST_HW_CPU_X86_64_GENERIC:
        return "x86_64";
    default:
        return "unknown";
    }
}
