# mk/target-mac-omp.mk — macOS / Apple Silicon target WITH OpenMP.
#
# The custom NEON kernels in cpu_neon run their loops as OpenMP regions
# here (geist_par_for, src/base/par.h); target-mac.mk runs them on GCD.
#
# Dependency: Homebrew libomp (`brew install libomp`). Default path
# is /opt/homebrew/opt/libomp; override via LIBOMP_PREFIX.
#
# Build: `make TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar"`
# Run:   `bin/mac-omp/release/<binary>`
#        - The cpu_neon backend sets OMP_WAIT_POLICY=active (decode runs one
#          parallel region per matmul, and passive wait pays a thread-pool
#          wake-up for each) and KMP_BLOCKTIME=200, so idle workers sleep
#          0.2 s after the last region (src/base/omp_idle.h). An explicit
#          setting in the environment wins; KMP_BLOCKTIME=infinite keeps an
#          idle process at 100 % on every worker core.
#        - 6 threads on M-class (8 P-cores); 8+ contends on DRAM bandwidth.

CC ?= clang

# Apple silicon: NEON pair. Intel Mac: x86 pair, same baseline as
# target-mac.mk / target-linux.mk.
ifeq ($(shell uname -m),x86_64)
BACKENDS ?= cpu_x86 cpu_scalar
MAC_ARCH_CFLAGS := -march=x86-64-v3 -mtune=generic
else
BACKENDS ?= cpu_neon cpu_scalar
MAC_ARCH_CFLAGS :=
endif

# Dense fp32 GEMM via Accelerate's cblas (linked below for vDSP anyway).
# GEMM_PROVIDER=native opts out to the dependency-free path.
GEMM_PROVIDER ?= accelerate

LIBOMP_PREFIX ?= /opt/homebrew/opt/libomp

# Use -isystem (not -I) for the libomp header to suppress -Wundef
# errors from libomp's `#if __cplusplus` checks. -Xpreprocessor is
# required because Apple-clang doesn't accept -fopenmp directly.
#
# `-ffast-math -fno-finite-math-only`: see target-mac.mk for the
# rationale; same flag set with OpenMP added.
CFLAGS_TARGET  := -DHAVE_ACCELERATE=1 \
                  -Xpreprocessor -fopenmp \
                  -isystem $(LIBOMP_PREFIX)/include \
                  -ffast-math -fno-finite-math-only \
                  $(MAC_ARCH_CFLAGS)

# Default links libomp dynamically (the dev workflow). GEIST_STATIC_OMP=1 links
# the static libomp.a instead, so a release binary depends only on system
# frameworks (Accelerate/libSystem, always present on macOS) — self-contained.
ifeq ($(GEIST_STATIC_OMP),1)
  LDFLAGS_TARGET := -framework Accelerate $(LIBOMP_PREFIX)/lib/libomp.a
else
  LDFLAGS_TARGET := -framework Accelerate -L$(LIBOMP_PREFIX)/lib -lomp
endif
LDLIBS_TARGET  := -lm
