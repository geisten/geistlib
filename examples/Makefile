# examples/Makefile — build the example programs against a built libgeist.a.
#
# Reuses the repo's real target settings (mk/target-$(TARGET).mk) so the
# example links with exactly the same compiler, flags, and libraries as the
# library itself — no duplicated platform knowledge here.
#
#   make -C examples                  # auto-detect target, MODE=release
#   make -C examples TARGET=pi5        # override target
#   make -C examples MODE=debug        # override mode
#
# If libgeist.a for the chosen TARGET/MODE is missing, it is built first.

TARGET ?= $(shell ../mk/detect-target.sh)
MODE   ?= release

include ../mk/target-$(TARGET).mk

# Mirror the library build: the GEMM provider fragment supplies the dense-fp32
# BLAS link (e.g. -lopenblas on Linux/Pi). target-*.mk sets the per-target
# default (openblas on Linux, accelerate on macOS); native = no cblas link.
GEMM_PROVIDER ?= native
include ../mk/gemm-$(GEMM_PROVIDER).mk

LIB := ../lib/$(TARGET)/$(MODE)/libgeist.a

CFLAGS  := -std=c23 -O2 -Wall -Wextra -I../include $(CFLAGS_TARGET) $(GEMM_CFLAGS)
LDFLAGS := $(LDFLAGS_TARGET)
LDLIBS  := $(LDLIBS_TARGET) $(GEMM_LDLIBS)

EXAMPLES := simple_generate

.PHONY: all clean
all: $(EXAMPLES)

# Build the library on demand for the selected target/mode.
$(LIB):
	$(MAKE) -C .. lib TARGET=$(TARGET) MODE=$(MODE)

%: %.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS) $(LDLIBS)

clean:
	@rm -f $(EXAMPLES)
