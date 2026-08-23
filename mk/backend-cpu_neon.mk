# mk/backend-cpu_neon.mk — ARM NEON CPU backend (Raspberry Pi 5, Apple Silicon).
#
# Enabled via `make BACKENDS="... cpu_neon"` (the default on ARM targets).
# Registered at runtime in src/engine/backend_registry.c under
# GEIST_BACKEND_CPU_NEON. The intrinsic-heavy TUs self-disable behind
# `#ifdef __ARM_NEON`, so the backend stays harmless (compiles to empty) when
# this is enabled on a non-ARM target via the cross-build matrix.

BACKEND_SOURCES += \
    src/backends/cpu_neon/backend.c \
    src/backends/cpu_neon/elementwise.c \
    src/backends/cpu_neon/kernel_catalog.c \
    src/backends/cpu_neon/kernels/iq2_s.c \
    src/backends/cpu_neon/kernels/iq3_s.c \
    src/backends/cpu_neon/kernels/q3_K.c \
    src/backends/cpu_neon/kernels/q4_01.c \
    src/backends/cpu_neon/kernels/q4_K.c \
    src/backends/cpu_neon/kernels/q5_K.c \
    src/backends/cpu_neon/kernels/q6_K.c \
    src/backends/cpu_neon/kernels/q8_0.c \
    src/backends/cpu_neon/kernels/tq2_0.c \
    src/backends/cpu_neon/parallel.c \
    src/backends/cpu_neon/tl1.c \
    src/backends/cpu_neon/transformer_ops.c \
    src/backends/cpu_neon/weight_resolve.c \
    src/backends/cpu_neon/workspace.c
