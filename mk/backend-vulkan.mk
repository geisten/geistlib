# Vulkan compute backend. libvulkan is dlopen'd at runtime (no link-time
# dependency); building needs only the Vulkan headers (VK_NO_PROTOTYPES).
# Shaders: src/backends/vulkan/shaders/*.comp are compiled to SPIR-V and
# committed as *_spv.h headers — `make vulkan-shaders` (needs glslc)
# regenerates them; end-user builds never invoke glslc.
BACKEND_SOURCES += src/backends/vulkan/backend.c
LDLIBS += -ldl

VULKAN_COMP := $(wildcard src/backends/vulkan/shaders/*.comp)
# *_dp4a.comp needs GL_EXT_integer_dot_product, which the distro
# glslc/glslang (shaderc 2023.x / glslang 15) predate — point GLSLANG at a
# glslang >= 16 binary (KhronosGroup/glslang main-tot release) to regenerate.
GLSLANG ?= glslang
.PHONY: vulkan-shaders
vulkan-shaders:
	@set -e; for f in $(VULKAN_COMP); do \
	  out="$${f%.comp}_spv.h"; sym=$$(basename $${f%.comp})_spv; \
	  case "$$f" in \
	    *_dp4a.comp) $(GLSLANG) -V --target-env vulkan1.3 -x -o "$$out.tmp" "$$f" >/dev/null; \
	      grep -v '^[[:space:]]*//' "$$out.tmp" > "$$out.tmp2"; mv "$$out.tmp2" "$$out.tmp" ;; \
	    *_cm.comp|*_cm32.comp) glslc --target-env=vulkan1.3 -mfmt=num "$$f" -o "$$out.tmp" ;; \
	    *) glslc -O --target-env=vulkan1.3 -mfmt=num "$$f" -o "$$out.tmp" ;; \
	  esac; \
	  { echo "/* generated from $$f — make vulkan-shaders */"; \
	    echo "static const uint32_t $$sym[] = {"; cat "$$out.tmp"; echo "};"; } > "$$out"; \
	  rm -f "$$out.tmp"; echo "  $$out"; \
	done
# NOTE: *_cm.comp (cooperative matrix) compile WITHOUT -O — spirv-opt
# miscompiles coopmat kernels (outputs silently become zero). *_dp4a.comp
# also skips spirv-opt (glslang emits unoptimized SPIR-V; the driver
# optimizes anyway, same as the coopmat kernels).
