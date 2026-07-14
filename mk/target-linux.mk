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

# Compiler — gcc-13+ or clang-16+ for C23 support.
CC ?= cc

# ----- x86_64 path ----------------------------------------------------------
#
# Native cpu_x86 backend is the DEFAULT (#108). The old opt-in gate ("until
# the Phase-2 win criteria are measured") is long met: matches-to-beats
# llama.cpp on Gemma/Llama and beats bitnet.cpp by +61 % prefill / +90 %
# decode on BitNet (benchmark/BENCHMARK_X86.md); every AVX-512 kernel is
# runtime-gated behind hw_probe/cpuid (#96) so the x86-64-v3 baseline below
# remains the only hardware floor; the CI x86 int/e2e strand (native +
# GEIST_FORCE_ISA=avx2) is required. A plain `make` previously shipped the
# ~200× slower scalar path — the same trap #102 fixed in bench_perf_sweep.
# BACKENDS="cpu_scalar" still builds the portable reference (dedicated CI
# job). Remember `make clean` when switching BACKENDS.
ifeq ($(LINUX_ARCH),x86_64)

BACKENDS ?= cpu_x86 cpu_scalar

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
