# mk/gemm-openblas.mk — dense fp32 GEMM via OpenBLAS (cblas).
#
# Selected via `make GEMM_PROVIDER=openblas` (the default on Linux / Pi 5).
# geist_gemm.c forwards geist_sgemm / geist_sgemv to cblas_sgemm / cblas_sgemv;
# OpenBLAS provides those symbols. Location is resolved via pkg-config with a
# plain -lopenblas fallback; override either var on the command line.

OPENBLAS_LIBS   ?= $(shell pkg-config --libs   openblas 2>/dev/null || echo '-lopenblas')
OPENBLAS_CFLAGS ?= $(shell pkg-config --cflags openblas 2>/dev/null)

GEMM_CFLAGS := $(OPENBLAS_CFLAGS) -DGEIST_GEMM_OPENBLAS
GEMM_LDLIBS := $(OPENBLAS_LIBS)
