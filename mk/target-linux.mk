# mk/target-linux.mk — generic Linux target.
#
# geist's compute kernels are currently NEON-only: arm_neon.h is included
# unconditionally across src/formats/gguf/, src/backends/common/, and
# src/formats/ptqtp/. The x86 backend (src/backends/cpu_x86/) is a policy
# skeleton, not a working compute path yet. Therefore:
#
#   * aarch64 / arm64  → supported. Generic ARMv8.2 tuning (Graviton, Ampere,
#                        generic ARM64 servers/SBCs). For a Raspberry Pi 5
#                        specifically, prefer `make TARGET=pi5` (cortex-a76).
#   * x86_64           → not supported yet. We fail fast with guidance rather
#                        than emitting a wall of arm_neon.h compile errors.
#
# Stack mirrors pi5: OpenBLAS (cblas, dense fp32), OpenMP; FFT vendored.
# Override OpenBLAS location via OPENBLAS_LIBS (see `make help`).

LINUX_ARCH := $(shell uname -m)

# Compiler — gcc-13+ or clang-16+ for C23 support.
CC ?= cc

# ----- x86_64 path (Phase 0 of docs/LINUX_X86_SPEC.md) ---------------------
#
# Native cpu_x86 backend is under construction on feat/cpu-x86 (gated, opt-in
# via BACKENDS="cpu_x86 cpu_scalar"). Default x86_64 build today is cpu_scalar
# only — slow but correct — until the Phase-2 win criteria are measured and
# this file flips x86_64 default BACKENDS to "cpu_x86 cpu_scalar". See
# docs/LINUX_X86_SPEC.md §"Build target" and §"Branch and merge strategy".
ifeq ($(LINUX_ARCH),x86_64)

BACKENDS ?= cpu_scalar

# Baseline x86-64-v3 (Haswell+: AVX2, FMA, BMI2). Per-TU -march= flags in
# mk/backend-cpu_x86.mk override this for the AVX-512 / +VNNI / +BF16 tiers.
CFLAGS_TARGET := -march=x86-64-v3 -mtune=generic -fopenmp -ffast-math \
                 -Wno-nonnull-compare -Wno-vla-parameter
LDFLAGS_TARGET := -fopenmp
LDLIBS_TARGET  := -lm
GEMM_PROVIDER ?= openblas

else ifneq (,$(filter $(LINUX_ARCH),i686 i386))

$(error TARGET=linux on $(LINUX_ARCH): 32-bit x86 is not supported.)

else

# ----- ARM64 path (existing — Graviton2+, Ampere Altra, generic ARMv8.2) ----
BACKENDS ?= cpu_neon cpu_scalar

# Generic ARMv8.2-A tuning — runs on Graviton2+, Ampere Altra, and most
# ARM64 SBCs. No -mcpu pin so the same binary is portable across cores.
# See target-pi5.mk for the rationale behind -ffast-math and the
# -Wno-nonnull-compare / -Wno-vla-parameter relaxations under stricter GCC.
CFLAGS_TARGET := -march=armv8.2-a+fp16+dotprod -fopenmp -ffast-math \
                 -Wno-nonnull-compare -Wno-vla-parameter

LDFLAGS_TARGET := -fopenmp
LDLIBS_TARGET  := -lm

# Dense fp32 GEMM provider. Default OpenBLAS (cblas); the openblas fragment
# resolves and links it. Use GEMM_PROVIDER=native for a dependency-free binary
# (libc/libm/libgomp only) — the musl-static CI artifact. Audio FFT is vendored
# either way (no FFTW3).
GEMM_PROVIDER ?= openblas

endif
