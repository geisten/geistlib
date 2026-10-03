# mk/backend-cpu_x86.mk — x86_64 backend sources.
#
# backend.c selects kernels directly in its resolver; per-ISA kernel TUs:
# W4A8 VPDPBUSD, BF16-SGEMM trampoline, native VDPBF16PS-SGEMM, I2_S ternary,
# Q8_0 prefill tiles, INT8-KV attention, PQ2_0 prefill on AMX-INT8.
#
# The default on Linux x86_64 (target-linux.mk: BACKENDS ?= cpu_x86
# cpu_scalar). BACKENDS=cpu_scalar builds the portable reference alone.

BACKEND_SOURCES += \
    src/backends/cpu_x86/backend.c \
    src/backends/cpu_x86/workspace.c \
    src/backends/cpu_x86/threads.c \
    src/backends/cpu_x86/elementwise.c \
    src/backends/cpu_x86/attention.c \
    src/backends/cpu_x86/attention_int8.c \
    src/backends/cpu_x86/attention_int8_avx512_vnni.c \
    src/backends/cpu_x86/kernel_w4a8.c \
    src/backends/cpu_x86/kernel_w4a8_scalar.c \
    src/backends/cpu_x86/kernel_w4a8_avx512_vnni.c \
    src/backends/cpu_x86/q4k_to_w4a8.c \
    src/backends/cpu_x86/linear_q4k.c \
    src/backends/cpu_x86/linear_q6k.c \
    src/backends/cpu_x86/linear_f32q.c \
    src/backends/cpu_x86/linear_generic.c \
    src/backends/cpu_x86/linear_q4_0.c \
    src/backends/cpu_x86/linear_q8_0.c \
    src/backends/cpu_x86/linear_pq2_0.c \
    src/backends/cpu_x86/kernel_pq2_0_amx.c \
    src/backends/cpu_x86/kernel_q8_0_avx512_vnni.c \
    src/backends/cpu_x86/kernel_w8a8.c \
    src/backends/cpu_x86/kernel_w8a8_scalar.c \
    src/backends/cpu_x86/kernel_w8a8_avx512_vnni.c \
    src/backends/cpu_x86/kernel_q6k_gemv.c \
    src/backends/cpu_x86/q6k_to_w8a8.c \
    src/backends/cpu_x86/kernel_i2s.c \
    src/backends/cpu_x86/kernel_i2s_avx512_vnni.c \
    src/backends/cpu_x86/kernel_f16_gemv.c \
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
# verified the matching cpuid feature bits — no SIGILL risk. That holds only
# while the deciding code stays OUT of these TUs: -mavx512* lets the compiler
# emit EVEX anywhere in them, prologues included (MODE=asan does it for the
# shadow poisoning), so a guard placed inside would run after the first
# illegal instruction, not before it. q4kx8 keeps its guard and shape
# dispatch in kernel_q4kx8_gemm_avx512.c, which is not in this list.
$(BUILD_DIR)/src/backends/cpu_x86/kernel_w4a8_avx512_vnni.o: CFLAGS_STRICT += \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
$(BUILD_DIR)/src/backends/cpu_x86/kernel_w8a8_avx512_vnni.o: CFLAGS_STRICT += \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
$(BUILD_DIR)/src/backends/cpu_x86/kernel_i2s_avx512_vnni.o: CFLAGS_STRICT += \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
$(BUILD_DIR)/src/backends/cpu_x86/kernel_q8_0_avx512_vnni.o: CFLAGS_STRICT += \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
$(BUILD_DIR)/src/backends/cpu_x86/attention_int8_avx512_vnni.o: CFLAGS_STRICT += \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
$(BUILD_DIR)/src/backends/cpu_x86/kernel_q4kx8_gemm_avx512_full.o: CFLAGS_STRICT += \
    -mavx2 -mavx -mf16c -mfma -mavx512f -mavx512bw -mavx512dq -mavx512vl
# linear_pq2_0.c's amx_usable() also checks the kernel granted the tile data.
$(BUILD_DIR)/src/backends/cpu_x86/kernel_pq2_0_amx.o: CFLAGS_STRICT += \
    -mamx-tile -mamx-int8 -mavx512f -mavx512bw
