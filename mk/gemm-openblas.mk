# mk/gemm-openblas.mk — dense fp32 GEMM via OpenBLAS (cblas).
#
# Selected via `make GEMM_PROVIDER=openblas` (the default on Linux / Pi 5).
# geist_gemm.c forwards geist_sgemm / geist_sgemv to cblas_sgemm / cblas_sgemv;
# OpenBLAS provides those symbols. Location is resolved via pkg-config with a
# plain -lopenblas fallback; override either var on the command line.

OPENBLAS_LIBS   ?= $(shell pkg-config --libs   openblas 2>/dev/null || echo '-lopenblas')
OPENBLAS_CFLAGS ?= $(shell pkg-config --cflags openblas 2>/dev/null)

# Fail here rather than at the first link. Without this, a machine that lacks
# OpenBLAS compiles every object first and then dies on `cannot find
# -lopenblas` with no hint that a BLAS-free build exists, minutes in.
#
# But only for goals that actually link. `make lib` archives objects and never
# resolves -lopenblas, and the README's quickstart is exactly `make lib` with
# gcc and make as the only stated prerequisites — erroring there would break
# the documented build on every Linux box without OpenBLAS. Verified in a
# clean alpine:3.21: `make lib` stays green, `make test-unit` gets the message.
OPENBLAS_LINKLESS_GOALS := lib clean distclean help format format-check \
                           fetch-model fetch-bench-model fetch-llama-model
OPENBLAS_GOALS := $(if $(MAKECMDGOALS),$(MAKECMDGOALS),__default__)
ifneq ($(strip $(filter-out $(OPENBLAS_LINKLESS_GOALS),$(OPENBLAS_GOALS))),)
# The probe is the same link the build will attempt: resolving OPENBLAS_LIBS
# with this CC is the only check that cannot disagree with the real failure.
# Costs one empty-main compile per make invocation.
OPENBLAS_PROBE := $(shell t=$$(mktemp -d) && printf 'int main(void){return 0;}' > $$t/p.c && $(CC) $$t/p.c $(OPENBLAS_LIBS) -o $$t/p >/dev/null 2>&1 && echo ok; rm -rf $$t)
ifneq ($(OPENBLAS_PROBE),ok)
$(error OpenBLAS not found (tried: $(OPENBLAS_LIBS)). Install it (apk add openblas-dev / apt install libopenblas-dev / brew install openblas), or build without it: make GEMM_PROVIDER=native)
endif
endif

GEMM_CFLAGS := $(OPENBLAS_CFLAGS) -DGEIST_GEMM_OPENBLAS
GEMM_LDLIBS := $(OPENBLAS_LIBS)
