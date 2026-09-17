# mk/target-mac-omp.mk — macOS / Apple Silicon target WITH OpenMP.
#
# Audience: Mac M1+ users who want the 2-2.2× decode speedup from
# parallel BitLinear matmuls (and any other OpenMP-gated kernel in
# the cpu_neon backend). Stock target-mac.mk uses Accelerate's
# internal threading for sgemm, but the W1.58 × A8 (TQ2_0) kernels
# are custom NEON — they only parallelize when libomp is linked.
#
# Dependency: Homebrew libomp (`brew install libomp`). Default path
# is /opt/homebrew/opt/libomp; override via LIBOMP_PREFIX.
#
# Build: `make TARGET=mac-omp BACKENDS="cpu_neon cpu_scalar"`
# Run:   `OMP_WAIT_POLICY=active OMP_NUM_THREADS=6 bin/mac-omp/release/<binary>`
#        - active wait policy (workers spin instead of sleep) is CRITICAL:
#          BitNet decode has 210 matmul calls/token → 210 omp parallel
#          regions; default passive wait incurs ~60% overhead from
#          repeated thread-pool wake-ups. Sample profiling: __kmp_*
#          drops from 62k samples to background noise. KMP_BLOCKTIME=infinite
#          is equivalent.
#        - 6 threads optimal on M-class (8 P-cores); 4 also strong, 8+
#          contends on DRAM bandwidth and regresses.
#
# BitNet 2B-4T decode @ seq=128, with OMP_WAIT_POLICY=active and the
# post-P9 build flags below (-ffast-math -fno-finite-math-only):
#   mac single-thread:                 17.4 tps  (P9 baseline)
#   mac-omp t=4, default wait:         ~28 tps
#   mac-omp t=4, WAIT_POLICY=active:   49.1 tps
#   mac-omp t=6, WAIT_POLICY=active: **60.9 tps**  (recommended)
#   bitnet.cpp t=4 (reference):        65.5 tps
# Gap to bitnet.cpp: ~1.07× behind — within OS/measurement noise of
# parity. Most of the closure since P3.13 (13.2 tps baseline) came
# from libomp + active wait + ffast-math vectorization unlocks.

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
