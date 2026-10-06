# mk/target-linux.mk — generic Linux target.
#
#   * aarch64 / arm64  → cpu_neon + cpu_scalar. Generic ARMv8.2 tuning
#                        (Graviton, Ampere, generic ARM64 servers/SBCs). For a
#                        Raspberry Pi 5 specifically, prefer `make TARGET=pi5`
#                        (cortex-a76).
#   * x86_64           → cpu_x86 + cpu_scalar (AVX-512/VNNI tiers runtime-
#                        dispatched over an x86-64-v3 baseline; see #96/#108).
#
# Stack mirrors pi5: OpenBLAS (cblas, dense fp32), OpenMP; FFT vendored.
# Override OpenBLAS location via OPENBLAS_LIBS (see `make help`).

LINUX_ARCH := $(shell uname -m)

# Compiler — gcc-14+ or clang-16+ for C23 support.
CC ?= cc

# ----- x86_64 path ----------------------------------------------------------
#
# Native cpu_x86 backend is the default (#108). Every AVX-512 kernel is
# runtime-gated behind hw_probe/cpuid (#96), so the x86-64-v3 baseline below
# is the only hardware floor. BACKENDS="cpu_scalar" builds the portable
# reference. Remember `make clean` when switching BACKENDS.
# The -Wno-vla-parameter relaxation below is a GCC-only name; clang treats
# unknown -Wno- options as errors under -Werror. Detect the compiler family once.
# For clang, also keep INFINITY well-defined under -ffast-math
# (-fno-finite-math-only, same as the mac-omp target) — clang 18+ makes
# INFINITY-with-finite-math a hard error, and the attention mask needs it.
ifeq (,$(findstring clang,$(CC)))
WARN_RELAX := -Wno-vla-parameter
else
WARN_RELAX := -fno-finite-math-only
endif

ifeq ($(LINUX_ARCH),x86_64)

BACKENDS ?= cpu_x86 cpu_scalar

# Baseline x86-64-v3 (Haswell+: AVX2, FMA, BMI2). Per-TU -march= flags in
# mk/backend-cpu_x86.mk override this for the AVX-512 / +VNNI / +BF16 tiers.
CFLAGS_TARGET := -march=x86-64-v3 -mtune=generic -fopenmp -ffast-math \
                 $(WARN_RELAX)
LDFLAGS_TARGET := -fopenmp
LDLIBS_TARGET  := -lm
GEMM_PROVIDER ?= openblas

else ifneq (,$(filter $(LINUX_ARCH),i686 i386))

$(error TARGET=linux on $(LINUX_ARCH): 32-bit x86 is not supported.)

else

# ----- ARM64 path (Graviton2+, Ampere Altra, generic ARMv8.2) --------------
BACKENDS ?= cpu_neon cpu_scalar

# Generic ARMv8.2-A tuning — runs on Graviton2+, Ampere Altra, and ARM64
# SBCs with dotprod and fp16 (Cortex-A55/A76 and later). No -mcpu pin so the
# same binary is portable across those cores. A Cortex-A53 or A72 (Raspberry
# Pi 3/4) has neither feature: geist_backend_create refuses it, naming the
# missing one, rather than letting the first SDOT raise SIGILL. The cpu_neon
# kernels need dotprod to compile at all; for those cores build
#   make TARGET=linux BACKENDS=cpu_scalar CFLAGS_TARGET="-mcpu=cortex-a72 ..."
# (the rest of CFLAGS_TARGET as below).
# See target-pi5.mk for the rationale behind -ffast-math and the
# -Wno-vla-parameter relaxation.
CFLAGS_TARGET := -march=armv8.2-a+fp16+dotprod -fopenmp -ffast-math \
                 $(WARN_RELAX)

LDFLAGS_TARGET := -fopenmp
LDLIBS_TARGET  := -lm

# Dense fp32 GEMM provider. Default OpenBLAS (cblas); the openblas fragment
# resolves and links it. Use GEMM_PROVIDER=native for a dependency-free binary
# (libc/libm/libgomp only) — the musl-static CI artifact. Audio FFT is vendored
# either way (no FFTW3).
GEMM_PROVIDER ?= openblas

endif

# Feature-test macro, both architectures, every Linux libc. `-std=c23` (not
# gnu23) defines __STRICT_ANSI__, and musl then hides everything outside ISO C
# (`strdup` included). Set here, not by each caller.
CFLAGS_TARGET += -D_GNU_SOURCE
