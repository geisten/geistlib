# Vulkan compute backend. libvulkan is dlopen'd at runtime (no link-time
# dependency); building needs only the Vulkan headers (VK_NO_PROTOTYPES).
# Shaders: src/backends/vulkan/shaders/*.comp are compiled to SPIR-V and
# committed as *_spv.h headers — `make vulkan-shaders` (needs glslc)
# regenerates them; end-user builds never invoke glslc.
BACKEND_SOURCES += src/backends/vulkan/backend.c
LDLIBS += -ldl

VULKAN_COMP := $(wildcard src/backends/vulkan/shaders/*.comp)
.PHONY: vulkan-shaders
vulkan-shaders:
	@set -e; for f in $(VULKAN_COMP); do \
	  out="$${f%.comp}_spv.h"; sym=$$(basename $${f%.comp})_spv; \
	  glslc -O --target-env=vulkan1.3 -mfmt=num "$$f" -o "$$out.tmp"; \
	  { echo "/* generated from $$f — make vulkan-shaders */"; \
	    echo "static const uint32_t $$sym[] = {"; cat "$$out.tmp"; echo "};"; } > "$$out"; \
	  rm -f "$$out.tmp"; echo "  $$out"; \
	done
