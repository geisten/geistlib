# mk/backend-metal.mk — Apple Metal GPU backend (Apple Silicon).
#
# Enabled via `make BACKENDS="... metal"`. Registered at runtime in
# src/engine/backend_registry.c under GEIST_BACKEND_METAL. The backend loads
# Metal.framework and libobjc symbols at runtime via dlopen/dlsym, so it needs
# no -framework at link time — only libdl. Harmless to compile on non-Apple
# targets (create() fails to find the Metal device and returns an error).

BACKEND_SOURCES += \
    src/backends/metal/backend.c

LDLIBS += -ldl

# The old fine-grained GPU op implementations (command_sequence_*, ple_block,
# ffn_geglu_block, attention_block, greedy_head, matmul_q4k/q6k) are kept
# compilable in backend.c during the port but are not yet all wired into the
# main-contract vtbl (the GEMM encoders get reused by resolve_weight in a later
# stage; the rest are removed in the Stage-6 cleanup). Suppress unused-function
# only for this TU until then. ponytail: drop this line once Stage 6 lands.
$(BUILD_DIR)/src/backends/metal/backend.o: CFLAGS_STRICT += -Wno-unused-function
