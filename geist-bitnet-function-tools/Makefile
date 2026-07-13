# geist-bitnet-function-tools — function tooling on top of the geist engine.
#
# The geist inference engine is IMPORTED as a prebuilt static library — this
# project compiles none of the engine sources itself:
#
#   lib/geistenlib.a         geist's libgeist.a, imported under this project's name
#   include/geisten/         geist's public ABI headers (geist.h, geist_util.h, ...)
#   include/geisten/tools/   geist's header-only agent layer (agent.h + schema v1)
#
# Usage:
#   make import              # clone/build geisten, import geistenlib.a + headers
#   make                     # import (if needed) + build ./geist-fn
#   make selftest            # run the function-tool registry selftest (no model)
#   make model               # fetch the BitNet b1.58 2B-4T GGUF (~1.2 GB)
#   make run ARGS='"Add 5 and 7"'
#
# Where the engine comes from:
#   - Inside a geisten checkout (this directory as a subproject): uses `..`.
#   - Standalone: clones $(GEISTEN_REPO) shallowly into vendor/geisten.
#   Override with GEISTEN_DIR=/path/to/geisten.

PROJECT := geist-fn

GEISTEN_REPO ?= https://github.com/geisten/geisten.git
GEISTEN_REF  ?= main
GEISTEN_DIR  ?= $(if $(wildcard ../include/geist.h),..,vendor/geisten)

MODE   ?= release
# geisten's own host detection (mac | mac-omp | pi5 | linux). Falls back to
# `linux` only while the vendor clone does not exist yet; every recipe that
# uses TARGET runs after the clone, so it re-evaluates correctly.
TARGET ?= $(shell sh $(GEISTEN_DIR)/mk/detect-target.sh 2>/dev/null || echo linux)

GEISTENLIB := lib/geistenlib.a
HDR        := include/geisten

# Reuse geisten's real target settings (CC, CFLAGS_TARGET, LDLIBS_TARGET) so
# this binary links with exactly the same flags as the library — no duplicated
# platform knowledge. Guarded: on the very first standalone run the fragments
# appear only after the clone; `all` recurses so the second parse sees them.
ifneq ($(wildcard $(GEISTEN_DIR)/mk/target-$(TARGET).mk),)
  include $(GEISTEN_DIR)/mk/target-$(TARGET).mk
  GEMM_PROVIDER ?= native
  include $(GEISTEN_DIR)/mk/gemm-$(GEMM_PROVIDER).mk
endif

# Same compiler floor as geisten: C23 (gcc >= 14 or clang >= 16); pass e.g.
# CC=gcc-14 when the system cc is older.
# -D_DEFAULT_SOURCE: strict ISO mode hides the POSIX/BSD surface the function
# tools use (localtime_r, getloadavg, sysconf) — re-enable it.
CFLAGS  := -std=c23 -O2 -Wall -Wextra -D_DEFAULT_SOURCE -I$(HDR) -I$(HDR)/tools $(CFLAGS_TARGET) $(GEMM_CFLAGS)
LDFLAGS := $(LDFLAGS_TARGET)
LDLIBS  := $(LDLIBS_TARGET) $(GEMM_LDLIBS) -lm

.PHONY: all import selftest model run clean distclean help

all:
	@$(MAKE) --no-print-directory import
	@$(MAKE) --no-print-directory $(PROJECT)

# ---- import: geisten -> geistenlib.a + headers ------------------------------

import: $(GEISTENLIB)

$(GEISTEN_DIR)/Makefile:
	git clone --depth 1 --branch $(GEISTEN_REF) $(GEISTEN_REPO) $(GEISTEN_DIR)

# Header-only agent layer: agent.h plus the two schema headers it includes.
AGENT_HEADERS := agent.h json_schema_v1.h dynamic_arguments_v1.h

$(GEISTENLIB): | $(GEISTEN_DIR)/Makefile
	$(MAKE) -C $(GEISTEN_DIR) lib TARGET=$(TARGET) MODE=$(MODE)
	@mkdir -p lib $(HDR)/tools
	cp $(GEISTEN_DIR)/lib/$(TARGET)/$(MODE)/libgeist.a $(GEISTENLIB)
	cp $(GEISTEN_DIR)/include/*.h $(HDR)/
	cp $(addprefix $(GEISTEN_DIR)/tools/,$(AGENT_HEADERS)) $(HDR)/tools/
	@echo "imported: $(GEISTENLIB) ($(TARGET)/$(MODE)) + $(HDR)/"

# ---- build ------------------------------------------------------------------

$(PROJECT): src/main.c src/function_tools.h $(GEISTENLIB)
	$(CC) $(CFLAGS) -o $@ src/main.c $(GEISTENLIB) $(LDFLAGS) $(LDLIBS)

# Registry + schema-validation roundtrip, no model file needed (CI-friendly).
selftest: all
	./$(PROJECT) --selftest

# ---- model: BitNet b1.58 2B-4T (ternary i2_s GGUF) --------------------------

MODEL_DIR   ?= models
BITNET_FILE ?= bitnet-2b4t.i2_s.gguf
BITNET_URL  ?= https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf
MODEL_PATH  := $(MODEL_DIR)/$(BITNET_FILE)

model: $(MODEL_PATH)
	@echo "BitNet model ready: $(MODEL_PATH)"

$(MODEL_PATH):
	@command -v curl >/dev/null 2>&1 || { echo "model: curl not found in PATH" >&2; exit 1; }
	@mkdir -p $(MODEL_DIR)
	curl -fL --retry 3 --retry-delay 2 -C - -o "$@.part" "$(BITNET_URL)"
	@mv "$@.part" "$@"

# `make run ARGS='"What time is it?"'` — build, fetch the model, run a request.
run: all $(MODEL_PATH)
	OMP_WAIT_POLICY=active ./$(PROJECT) -m $(MODEL_PATH) $(ARGS)

# ---- housekeeping -----------------------------------------------------------

clean:
	rm -f $(PROJECT)

distclean: clean
	rm -rf lib/geistenlib.a $(HDR) vendor

help:
	@echo "make [all]      import geistenlib.a + build ./$(PROJECT)"
	@echo "make import     build geisten and import lib/geistenlib.a + include/geisten/"
	@echo "make selftest   run the tool registry selftest (no model needed)"
	@echo "make model      download the BitNet b1.58 2B-4T GGUF to $(MODEL_PATH)"
	@echo "make run ARGS='\"Add 5 and 7\"'   run a request against BitNet"
	@echo "vars: GEISTEN_DIR=$(GEISTEN_DIR) TARGET=$(TARGET) MODE=$(MODE)"
