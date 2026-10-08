# mk/target-mac.mk — macOS / Apple Silicon target settings, NO OpenMP.
#
# Audience: Mac M1+ (Apple-clang via Xcode) without Homebrew libomp,
# or for serial debugging.
# Stack: Accelerate framework (BLAS via cblas, vDSP for FFT).
#
# Most users want `make TARGET=mac-omp` instead — `mk/detect-target.sh`
# auto-picks it when /opt/homebrew/opt/libomp/lib/libomp.dylib exists.
# Without libomp, the cpu_neon kernels run their loops on GCD
# (geist_par_for, src/base/par.h; #618). That path is not yet measured
# against mac-omp on Apple Silicon; mac-omp is the benchmarked build.
$(warning building plain mac target without OpenMP — the kernels run on GCD, not yet benchmarked against TARGET=mac-omp (#618). For the measured build: brew install libomp.)

CC ?= clang

# Apple silicon builds the NEON pair. An Intel Mac builds the x86 pair with
# the ISA baseline of target-linux.mk: the cpu_x86 intrinsics need AVX2/F16C
# at compile time; AVX-512 kernels are dispatched at runtime.
ifeq ($(shell uname -m),x86_64)
BACKENDS ?= cpu_x86 cpu_scalar
MAC_ARCH_CFLAGS := -march=x86-64-v3 -mtune=generic
else
BACKENDS ?= cpu_neon cpu_scalar
MAC_ARCH_CFLAGS :=
endif

# On Apple silicon, Apple-clang already targets the host CPU optimally with
# -O3; no -march needed there.
#
# `-ffast-math -fno-finite-math-only`: enables fp reassociation +
# vectorizer-friendly assumptions while keeping `-INFINITY` semantics
# (Apple-clang's `-ffast-math` alone rejects `-INFINITY`, which the
# attention softmax uses as its init sentinel). Greedy decode is
# bit-identical on the test prompts.
CFLAGS_TARGET  := -DHAVE_ACCELERATE=1 -ffast-math -fno-finite-math-only $(MAC_ARCH_CFLAGS)

# Accelerate framework provides BLAS + vDSP (FFT).
LDFLAGS_TARGET := -framework Accelerate
LDLIBS_TARGET  := -lm

# Dense fp32 GEMM via Accelerate's cblas (linked above for vDSP anyway).
# GEMM_PROVIDER=native opts out to the dependency-free path.
GEMM_PROVIDER ?= accelerate
