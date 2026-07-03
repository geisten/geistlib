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
