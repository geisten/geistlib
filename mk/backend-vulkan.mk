# Vulkan compute backend. libvulkan is dlopen'd at runtime (no link-time
# dependency); building needs only the Vulkan headers (VK_NO_PROTOTYPES).
# Shaders: src/backends/vulkan/shaders/*.comp are compiled to SPIR-V and
# committed as *_spv.h headers — `make vulkan-shaders` (needs the pinned glslc,
# see GLSLC_VERSION) regenerates them; end-user builds never invoke glslc.
BACKEND_SOURCES += src/backends/vulkan/lifecycle.c \
    src/backends/vulkan/resources.c \
    src/backends/vulkan/pipelines.c \
    src/backends/vulkan/sequence.c \
    src/backends/vulkan/ops.c
LDLIBS += -ldl

VULKAN_COMP := $(wildcard src/backends/vulkan/shaders/*.comp)
# Pinned shader compiler: Debian 13's glslc package. Another glslc emits
# different (equivalent) SPIR-V for the same source, so the checked-in
# headers would churn with whoever regenerated them last. CI regenerates
# with exactly this version and fails on any diff (vulkan-lavapipe job).
# Without it locally:
#   docker run --rm -v "$$PWD:/src" -w /src debian:13 sh -c \
#     'apt-get update && apt-get install -y glslc=$(GLSLC_DEB) make &&
#      make GEMM_PROVIDER=native BACKENDS=vulkan vulkan-shaders'
# GLSLC_VERSION_CHECK=0 skips the check (for experiments, never commits).
GLSLC ?= glslc
GLSLC_DEB := 2025.2-1
GLSLC_VERSION := shaderc 2025.2-1 spirv-tools 2025.1~rc1-1 glslang 15.1.0+1.4.309.0-1
GLSLC_VERSION_CHECK ?= 1
.PHONY: vulkan-shaders
vulkan-shaders:
	@if [ "$(GLSLC_VERSION_CHECK)" != 0 ]; then \
	  have=$$($(GLSLC) --version 2>/dev/null | head -n 3 | tr -s '\n' ' ' | sed 's/ *$$//'); \
	  if [ "$$have" != "$(GLSLC_VERSION)" ]; then \
	    echo "vulkan-shaders: $(GLSLC) is '$$have'," >&2; \
	    echo "  the headers are pinned to '$(GLSLC_VERSION)'" >&2; \
	    echo "  (Debian 13 package glslc=$(GLSLC_DEB); see mk/backend-vulkan.mk)." >&2; \
	    echo "  GLSLC_VERSION_CHECK=0 overrides; do not commit headers built that way." >&2; \
	    exit 1; \
	  fi; \
	fi
	@set -e; for f in $(VULKAN_COMP); do \
	  out="$${f%.comp}_spv.h"; sym=$$(basename $${f%.comp})_spv; \
	  case "$$f" in \
	    *_cm.comp|*_cm32.comp) opt="" ;; \
	    *) opt="-O" ;; \
	  esac; \
	  $(GLSLC) $$opt --target-env=vulkan1.3 -mfmt=num "$$f" -o "$$out.tmp"; \
	  { echo "/* generated from $$f — make vulkan-shaders */"; \
	    echo "static const uint32_t $$sym[] = {"; cat "$$out.tmp"; echo "};"; } > "$$out"; \
	  rm -f "$$out.tmp"; echo "  $$out"; \
	done
# NOTE: *_cm.comp (cooperative matrix) compile WITHOUT -O — spirv-opt
# miscompiles coopmat kernels (outputs silently become zero).
