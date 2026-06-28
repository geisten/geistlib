# mk/backend-cpu_x86.mk — x86_64 backend sources.
#
# Phase 0 (current): backend.c shell + kernel_catalog policy table. The vtbl
# is cpu_scalar's; per-ISA kernel TUs land in Phase 1a (W4A8 VPDPBUSD), 1b
# (BF16-SGEMM trampoline), 2 (native VDPBF16PS-SGEMM).
# See docs/LINUX_X86_SPEC.md.
#
# Opt-in via `make BACKENDS="cpu_x86 cpu_scalar"` (default Linux x86_64 build
# remains cpu_scalar-only until the win-criteria pass and target-linux.mk
# flips the default).

BACKEND_SOURCES += \
    src/backends/cpu_x86/backend.c \
    src/backends/cpu_x86/elementwise.c \
    src/backends/cpu_x86/kernel_catalog.c \
    src/backends/cpu_x86/kernel_w4a8.c \
    src/backends/cpu_x86/kernel_w4a8_scalar.c \
    src/backends/cpu_x86/kernel_w4a8_avx512_vnni.c \
    src/backends/cpu_x86/q4k_to_w4a8.c \
    src/backends/cpu_x86/linear_q4k.c \
    src/backends/cpu_x86/linear_q6k.c \
    src/backends/cpu_x86/linear_f32q.c \
    src/backends/cpu_x86/kernel_w8a8.c \
    src/backends/cpu_x86/kernel_w8a8_scalar.c \
    src/backends/cpu_x86/kernel_w8a8_avx512_vnni.c \
    src/backends/cpu_x86/kernel_q6k_gemv.c \
    src/backends/cpu_x86/q6k_to_w8a8.c \
    src/backends/cpu_x86/kernel_bf16_gemm_scalar.c \
    src/backends/cpu_x86/kernel_bf16_gemm_avx512_bf16.c \
    src/backends/cpu_x86/q4k_to_q4kx8.c \
    src/backends/cpu_x86/q8_kx4.c \
    src/backends/cpu_x86/kernel_q4kx8_gemm_scalar.c \
    src/backends/cpu_x86/kernel_q4kx8_gemm_avx512.c \
    src/backends/cpu_x86/kernel_q4kx8_gemm_avx512_full.c

# Per-TU ISA flags. CFLAGS_STRICT is set globally in mk/common.mk with `:=`,
# but the compile recipe expands $(CFLAGS_STRICT) at recipe-run time, so the
# target-specific `+=` below is in effect for those .o targets.
#
# The variant TUs only run on hosts whose hw_probe + dispatcher have already
# verified the matching cpuid feature bits — no SIGILL risk.
$(BUILD_DIR)/src/backends/cpu_x86/kernel_w4a8_avx512_vnni.o: CFLAGS_STRICT += \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
$(BUILD_DIR)/src/backends/cpu_x86/kernel_w8a8_avx512_vnni.o: CFLAGS_STRICT += \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
$(BUILD_DIR)/src/backends/cpu_x86/kernel_bf16_gemm_avx512_bf16.o: CFLAGS_STRICT += \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512bf16
$(BUILD_DIR)/src/backends/cpu_x86/kernel_q4kx8_gemm_avx512_full.o: CFLAGS_STRICT += \
    -mavx2 -mavx -mf16c -mfma -mavx512f -mavx512bw -mavx512dq -mavx512vl
