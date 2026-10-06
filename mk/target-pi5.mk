# mk/target-pi5.mk — Raspberry Pi 5 / Cortex-A76 target settings.
#
# Audience: Pi 5 (ARM64, Cortex-A76, 4 cores).
# Stack: OpenBLAS for cblas (dense fp32), OpenMP for threading; FFT is vendored.
#
# OpenBLAS is resolved via pkg-config; override with OPENBLAS_LIBS
# (see `make help`).

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
# `-ffast-math` enables fp reassociation + finite-math assumptions, which
# unlocks NEON autovectorization in the softmax / activation / elementwise
# kernels. Greedy decode and WikiText PPL match strict-math within noise.
CFLAGS_TARGET := -DGEIST_TARGET_PI5=1 -mcpu=cortex-a76 -fopenmp -ffast-math -Wno-vla-parameter

# Same rationale as mk/target-linux.mk: -std=c23 defines __STRICT_ANSI__,
# under which POSIX symbols (mkstemp, strdup, ...) vanish without a feature
# macro. No CI leg builds TARGET=pi5 (#244), so keep it in step by hand.
CFLAGS_TARGET += -D_GNU_SOURCE

LDFLAGS_TARGET := -fopenmp
LDLIBS_TARGET  := -lm

# Dense fp32 GEMM provider. Default OpenBLAS (cblas); the openblas fragment
# resolves and links it. Use GEMM_PROVIDER=native for a dependency-free binary
# (libc/libm/libgomp only) — the musl-static CI artifact. Audio FFT is vendored
# either way (no FFTW3).
GEMM_PROVIDER ?= openblas
