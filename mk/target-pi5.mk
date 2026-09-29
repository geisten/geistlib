# mk/target-pi5.mk — Raspberry Pi 5 / Cortex-A76 target settings.
#
# Audience: Pi 5 (ARM64, Cortex-A76, 4 cores).
# Stack: OpenBLAS for cblas (dense fp32), OpenMP for threading; FFT is vendored.
# parallel kernels (m>1 prefill loops in NEON backend).
#
# Dependencies resolved via pkg-config with manual override via OPENBLAS_LIBS,
# OPENBLAS_LIBS environment variable (see `make help`).

# Compiler — prefer gcc-13 for proper C23 constexpr support.
# Cross-compile: override with CC=aarch64-linux-gnu-gcc-13.
CC ?= gcc

BACKENDS ?= cpu_neon cpu_scalar

# Cortex-A76 specialization — best codegen for NEON kernels.
#
# `-Wno-vla-parameter`: GCC (clang has no such warning) flags a parameter
# that one declaration spells `[static n]` and another a plain pointer —
# mostly gemma4_kernels.c against its header. Both spell the same type.
# -Wnonnull-compare stays on: a null check on a `[static n]` parameter is a
# contradiction, and GCC and clang settle it by deleting the check at -O1
# and above (AGENT.md §1).
# `-ffast-math` enables fp reassociation + finite-math assumptions,
# unlocking substantially more aggressive NEON autovectorization in
# the softmax / activation / elementwise kernels. +12% Pi 5 decode on
# BitNet 2B-4T at t=4 active wait. Greedy decode and WikiText PPL match
# strict-math within noise (verified on bitnet-2b4t-TQ2_0-v2.gguf).
# GCC's `-ffast-math` defines `__FAST_MATH__`; some code paths may opt
# out via `#pragma STDC FENV_ACCESS ON` if exact rounding ever matters.
CFLAGS_TARGET := -DGEIST_TARGET_PI5=1 -mcpu=cortex-a76 -fopenmp -ffast-math -Wno-vla-parameter

# Same rationale as mk/target-linux.mk: -std=c23 defines __STRICT_ANSI__,
# under which POSIX symbols (mkstemp, strdup, ...) vanish without a feature
# macro. It belongs here rather than in each caller — the pi5 target missed
# it when linux gained it in 0.8.0, and TARGET=pi5 broke unseen because no
# CI leg builds it (#244).
CFLAGS_TARGET += -D_GNU_SOURCE

LDFLAGS_TARGET := -fopenmp
LDLIBS_TARGET  := -lm

# Dense fp32 GEMM provider. Default OpenBLAS (cblas); the openblas fragment
# resolves and links it. Use GEMM_PROVIDER=native for a dependency-free binary
# (libc/libm/libgomp only) — the musl-static CI artifact. Audio FFT is vendored
# either way (no FFTW3).
GEMM_PROVIDER ?= openblas
