# geist Makefile — entry point.
#
# Usage:
#   make                       # auto-detect TARGET, MODE=release
#   make TARGET=pi5            # cross-compile (override CC if needed)
#   make MODE=asan             # sanitizer build
#   make MODE=asan TARGET=pi5  # combinations
#   make test                  # unit + integration + python suites
#   make bench                 # reproducible benchmark vs llama.cpp / bitnet.cpp
#   make help                  # show all options
#
# Output layout (per-target, per-mode segregated):
#   build/$(TARGET)/$(MODE)/   *.o, *.d
#   lib/$(TARGET)/$(MODE)/     libgeist.a
#   bin/$(TARGET)/$(MODE)/     tests/test_*, tests/bench_*, tools/eval_geist, ...
#
# Adding a new target architecture:
#   1. Create mk/target-<name>.mk (set CC, CFLAGS_TARGET, LDFLAGS_TARGET, LDLIBS_TARGET)
#   2. Run: make TARGET=<name>

# Auto-detect target (mac on macOS, pi5 on ARM64 Linux, linux otherwise).
TARGET ?= $(shell mk/detect-target.sh)
MODE   ?= release

# Phony targets — do not match files.
.PHONY: all lib bin run agent-contract-smoke release-check release-state-check bench-smoke fetch-bench-model clean distclean help test test-unit test-int test-e2e test-all test-py test-dequant fetch-model fetch-llama-model fetch-qwen3-model fetch-qwen35-model fetch-e4b-model fetch-audio-tower bench bench-small bench-detailed bench-quality-small bench-quality-detailed bench-compare-ref bench-mmlu bench-vision bench-video bench-audio bench-mm format format-check

# Default goal. `lib` is the deliverable; `bin` builds the in-tree test and
# evaluation tools under bin/<target>/<mode>/. This repository ships no CLI.
all: lib bin

# Pull in target settings (CC, CFLAGS_TARGET, LDFLAGS_TARGET, LDLIBS_TARGET).
# Guard first: an empty or unknown TARGET otherwise dies on a missing
# `mk/target-.mk`, which tells the reader nothing about what went wrong.
ifeq ($(wildcard mk/target-$(TARGET).mk),)
  $(error unsupported TARGET '$(TARGET)' — no mk/target-$(TARGET).mk. \
    Available: $(patsubst mk/target-%.mk,%,$(wildcard mk/target-*.mk)). \
    Pick one with `make TARGET=<name>`)
endif
include mk/target-$(TARGET).mk

# Pull in common build rules (LIB_FILE, BIN_TARGETS, object/link rules).
include mk/common.mk

lib: $(LIB_FILE)

bin: $(BIN_TARGETS)

# Agent-runtime API contract (docs/API_CONTRACT.md). An out-of-tree agent
# runtime links these symbols across a release boundary, so a signature change
# must fail HERE, not in the consumer's build. Compiling pins the signatures,
# linking pins their existence.
# This is the last thing in the engine that knows an agent exists, and it is
# deliberate: it is a promise, not a dependency.
# The gate a release passes before the tag: the four version sites agree, the
# API contract holds, no `STABLE since` names a version newer than this one,
# and nothing is stranded under CHANGELOG [Unreleased]. NOT a per-PR check —
# promoting a symbol to a future version is legitimate in a PR and wrong only
# at the moment of tagging. See scripts/check-release.sh.
release-check:
	@sh scripts/check-release.sh --pre-tag

# Networked postcondition: what /releases/latest serves agrees with main and
# carries the complete asset set. CI runs this daily and after publication.
release-state-check:
	@sh scripts/check-published-release.sh

AGENT_CONTRACT_SMOKE := $(BIN_DIR)/examples/agent_contract_smoke
agent-contract-smoke: $(AGENT_CONTRACT_SMOKE)
	@$(AGENT_CONTRACT_SMOKE)
$(AGENT_CONTRACT_SMOKE): examples/agent_contract_smoke.c $(LIB_FILE) include/geist.h include/geist_util.h
	@mkdir -p $(@D)
	$(CC) -std=c23 -Wall -Wextra -Wpedantic -Werror -Iinclude $(LDFLAGS) -o $@ $< $(LIB_FILE) $(LDLIBS)

# `make run ARGS='model.gguf "your prompt"'` — build the smallest useful
# program against the library and run it. The engine ships no CLI of its own
# any more; examples/simple_generate.c is the STABLE-core demo.
run: lib
	@$(MAKE) -C examples TARGET=$(TARGET) MODE=$(MODE)
	@OMP_WAIT_POLICY=active examples/simple_generate $(ARGS)

# Test runner — invokes mk/run-tests.sh against the test bin directory.
# FILTER is an optional substring; e.g. `make test FILTER=q3k` runs only
# tests whose binary name contains "q3k".
TEST_BIN_DIR := $(BIN_DIR)/tests
# Run artifacts live OUTSIDE the repo so the working tree stays clean; override
# with `make bench-small BENCH_OUT_DIR=...` to put them elsewhere.
BENCH_OUT_DIR ?= $(HOME)/bench-geistlib/quality_perf
# Overrides for the bench-* quality/perf suites; passed through via BENCH_PY
# below (apple-perf.yml sets BENCH_GGUF in CI).
BENCH_GGUF ?=
BENCH_THREADS ?=
BENCH_REF_GGUF ?=
BENCH_REF_BIN ?=

# ---- Reference test model -------------------------------------------------
# The _int / _e2e / bench suites load a real GGUF via GEIST_GGUF_PATH and skip
# cleanly when it is absent (see tests/test_helpers.h). `make fetch-model`
# downloads it once into MODEL_DIR; the suites below auto-point
# GEIST_GGUF_PATH at it when present (unless the caller already set it).
#
# Source: unsloth's Gemma 4 E2B-it GGUF (Q4_K_M, ~3.1 GB). Overridable —
# point MODEL_URL elsewhere, or pass HF_TOKEN=... for a gated mirror.
MODEL_DIR     ?= gguf_artifacts
MODEL_FILE    ?= gemma4-e2b-Q4_K_M.gguf
MODEL_PATH    := $(MODEL_DIR)/$(MODEL_FILE)
MODEL_HF_REPO ?= unsloth/gemma-4-E2B-it-GGUF
MODEL_HF_FILE ?= gemma-4-E2B-it-Q4_K_M.gguf
MODEL_URL     ?= https://huggingface.co/$(MODEL_HF_REPO)/resolve/main/$(MODEL_HF_FILE)

# `make test` / test-int / test-e2e auto-fetch the model when it is missing,
# then point GEIST_GGUF_PATH at it so the model-gated suites actually run
# instead of skipping. Set AUTO_FETCH_MODEL=0 to keep the network out of
# `make test` (suites then skip cleanly when the model is absent — handy for
# CI / offline). A caller-provided GEIST_GGUF_PATH always wins and suppresses
# the download. MODEL_PREREQ is the on-demand download dependency; it is the
# real file target ($(MODEL_PATH)), so it no-ops when the model already exists.
AUTO_FETCH_MODEL ?= 1
ifeq ($(strip $(GEIST_GGUF_PATH)),)
  ifeq ($(AUTO_FETCH_MODEL),1)
    MODEL_PREREQ := $(MODEL_PATH)
  endif
endif

# Shell prelude for GGUF-consuming recipes, evaluated at recipe time (after any
# on-demand download): prefer a caller-set GEIST_GGUF_PATH, else use the
# reference model if present. Absolute path so it resolves regardless of cwd.
GGUF_ENV = if [ -z "$$GEIST_GGUF_PATH" ] && [ -f "$(MODEL_PATH)" ]; then \
               export GEIST_GGUF_PATH="$(abspath $(MODEL_PATH))"; \
           fi;

# `make test` chains unit + int + py — daily-iteration default. The model is
# listed FIRST so the on-demand download (if any) happens up front, before the
# unit tests run, rather than mid-run between unit and int suites.
test: $(MODEL_PREREQ) test-unit test-int test-py

test-unit: bin
	@$(GGUF_ENV) mk/run-tests.sh $(TEST_BIN_DIR) "_unit"

test-int: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) mk/run-tests.sh $(TEST_BIN_DIR) "_int"

test-e2e: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) mk/run-tests.sh $(TEST_BIN_DIR) "_e2e"

# Python-side tests (algorithm reference impls — PTQTP, quantization tooling).
# Hermetic: no GGUF, no network. Exit non-zero on any failure.
test-py:
	@status=0; \
	for f in $(wildcard tests/test_*.py); do \
		echo "=== $$f ==="; \
		python3 "$$f" || status=$$?; \
	done; \
	if [ $$status -ne 0 ]; then echo "test-py: FAIL"; exit $$status; fi; \
	echo "test-py: PASS"

# Dequant parity against gguf-py, the canonical reference implementation.
# Kept out of test-py, which is hermetic by contract — this one needs a real
# GGUF. Skips (77) when the model or the `gguf` package is missing, so it is
# safe to run anywhere; pass GGUF files as arguments to check specific ones.
test-dequant: bin
	@python3 tests/scripts/validate_gguf_dequant.py $(DEQUANT_MODELS); \
	rc=$$?; \
	if [ $$rc -eq 77 ]; then echo "test-dequant: SKIPPED"; exit 0; fi; \
	exit $$rc

# `make test-all` adds e2e but excludes benches. Model first (see `test`).
test-all: $(MODEL_PREREQ) test-unit test-int test-py test-e2e

# Download the reference GGUF (~3.1 GB) once into MODEL_DIR. Idempotent: the
# file rule no-ops when the model already exists, so it is safe to depend on
# and cheap to re-run. Downloads to a .part file and renames on success so an
# interrupted transfer never leaves a truncated model at the final path
# (curl -C - resumes the .part on the next run). Override source via MODEL_URL;
# pass HF_TOKEN=... for gated mirrors.
fetch-model: $(MODEL_PATH)
	@echo "Reference model ready: $(MODEL_PATH)"

# Shared download recipe for the model-file rules below: curl into $@.part and
# rename on success, so an interrupted transfer never leaves a truncated file
# at the final path (-C - resumes the .part next run). Honors HF_TOKEN for
# gated mirrors. $(1) = human-readable size, $(2) = URL.
define fetch_gguf
	@command -v curl >/dev/null 2>&1 || { echo "$@: curl not found in PATH" >&2; exit 1; }
	@mkdir -p $(@D)
	@echo "Downloading $(@F) ($(1)) from:"
	@echo "  $(2)"
	@curl -fL --retry 3 --retry-delay 2 -C - \
	  $(if $(HF_TOKEN),-H "Authorization: Bearer $(HF_TOKEN)",) \
	  -o "$@.part" "$(2)"
	@mv "$@.part" "$@"
endef

# SHA-256 pin check for the fetch-* targets: prints both hashes on mismatch so
# a truncated download, a corrupted CI cache and a silently changed upstream
# all fail loudly BEFORE a test runs. Changing a model means changing its pin,
# which also rotates any CI cache key derived from it. $(1) = file, $(2) = pin.
define verify_sha256
	@hash=$$( (command -v sha256sum >/dev/null && sha256sum "$(1)" || shasum -a 256 "$(1)") | cut -d' ' -f1 ); \
	if [ "$$hash" != "$(2)" ]; then \
	  echo "$@: SHA-256 mismatch for $(1)" >&2; \
	  echo "  expected $(2)" >&2; \
	  echo "  actual   $$hash" >&2; \
	  echo "  (truncated/corrupt download or cache -> delete the file and re-run;" >&2; \
	  echo "   persists -> upstream content changed, re-pin deliberately)" >&2; \
	  exit 1; \
	fi
endef

$(MODEL_PATH):
	$(call fetch_gguf,~3.1 GB,$(MODEL_URL))

# Benches are timing tools, not tests — separate target. Each bench prints
# its own metrics; runner just reports run/skip/fail status.
bench-smoke: bin
	@$(GGUF_ENV) GEIST_INCLUDE_BENCH=1 mk/run-tests.sh $(TEST_BIN_DIR) "bench_"

# ---- make bench: the reproducible benchmark ------------------------------
# One command, one report, meant to be run by someone who does not trust the
# numbers yet. Protocol frozen in tools/bench_reproduce.py; every row in
# benchmark/reference_runs.json came from it.
#
# Its own model, deliberately: MODEL_* above is the *test fixture* (Gemma,
# 3.1 GB, wired into seven test targets). The reproducer takes the smallest
# model that carries a published claim instead of making a curious visitor
# wait for three gigabytes.
BENCH_MODEL_DIR  ?= gguf_artifacts
BENCH_MODEL_FILE ?= bitnet-2b4t-i2_s.gguf
BENCH_MODEL_PATH := $(BENCH_MODEL_DIR)/$(BENCH_MODEL_FILE)
BENCH_MODEL_URL  ?= https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf

$(BENCH_MODEL_PATH):
	$(call fetch_gguf,~1.1 GB,$(BENCH_MODEL_URL))

fetch-bench-model: $(BENCH_MODEL_PATH)
	@echo "Benchmark model ready: $(BENCH_MODEL_PATH)"

# Gemma 4 E4B variant (4.6 GB) — the second gemma4 geometry (42 layers,
# d_model 2560, 5+1 sliding pattern). Exists so the metadata-driven family
# populator (#258) has an executing check; the weekly e4b-smoke workflow
# fetches it, not the default test flow.
E4B_MODEL_DIR  ?= gguf_artifacts
E4B_MODEL_FILE ?= gemma4-e4b-Q4_K_M.gguf
E4B_MODEL_PATH := $(E4B_MODEL_DIR)/$(E4B_MODEL_FILE)
E4B_MODEL_URL  ?= https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf

$(E4B_MODEL_PATH):
	$(call fetch_gguf,~4.6 GB,$(E4B_MODEL_URL))

fetch-e4b-model: $(E4B_MODEL_PATH)
	@echo "E4B model ready: $(E4B_MODEL_PATH)"

# Llama-family reference model. Small on purpose (369 MB): it exists to prove
# the engine reads a SECOND architecture family and tokenizer mode, not to
# measure anything, so test_llama_{load,e2e}_int stop skipping. Not part of
# AUTO_FETCH_MODEL -- a local `make test` should not grow by another download
# on top of the 3.1 GB reference; CI fetches it explicitly.
LLAMA_MODEL_DIR  ?= gguf_artifacts
LLAMA_MODEL_FILE ?= smollm2-360m-instruct-q8_0.gguf
LLAMA_MODEL_PATH := $(LLAMA_MODEL_DIR)/$(LLAMA_MODEL_FILE)
LLAMA_MODEL_URL  ?= https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf
# Pinned content hash (= the upstream LFS oid, Apache-2.0 model). Verified on
# every fetch-llama-model run, so a truncated download, a corrupted CI cache
# and a silently changed upstream all fail loudly BEFORE a test runs — and
# each prints which of the two hashes disagrees. Changing the model means
# changing this pin, which also rotates the CI cache key derived from it.
LLAMA_MODEL_SHA256 := 48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201

$(LLAMA_MODEL_PATH):
	$(call fetch_gguf,~369 MB,$(LLAMA_MODEL_URL))

# Qwen3 reference model (#275). Small third family (609 MB): proves the
# qwen3 populator (metadata head_dim 128 ≠ d_model/n_heads, per-head
# QK-norm) and the qwen2 pretokenizer parity path. SHA-pinned like the
# llama fixture; CI fetches it explicitly.
QWEN3_MODEL_DIR  ?= gguf_artifacts
QWEN3_MODEL_FILE ?= qwen3-0.6b-q8_0.gguf
QWEN3_MODEL_PATH := $(QWEN3_MODEL_DIR)/$(QWEN3_MODEL_FILE)
QWEN3_MODEL_URL  ?= https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
# Upstream LFS oid, verified 2026-08-19.
QWEN3_MODEL_SHA256 := 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031

$(QWEN3_MODEL_PATH):
	$(call fetch_gguf,~609 MB,$(QWEN3_MODEL_URL))

fetch-qwen3-model: $(QWEN3_MODEL_PATH)
	$(call verify_sha256,$(QWEN3_MODEL_PATH),$(QWEN3_MODEL_SHA256))
	@echo "Qwen3 reference model ready (SHA-256 verified): $(QWEN3_MODEL_PATH)"

# Qwen3.5 hybrid reference model (#281). 0.8B Q8_0 (~780 MB): proves the
# per-layer token-mixer dispatch (gated DeltaNet + gated attention) and
# the qwen35 pretokenizer. SHA-pinned; CI fetches it explicitly.
QWEN35_MODEL_DIR  ?= gguf_artifacts
QWEN35_MODEL_FILE ?= qwen3.5-0.8b-q8_0.gguf
QWEN35_MODEL_PATH := $(QWEN35_MODEL_DIR)/$(QWEN35_MODEL_FILE)
QWEN35_MODEL_URL  ?= https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
# Upstream LFS oid, verified 2026-08-23.
QWEN35_MODEL_SHA256 := 0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c

$(QWEN35_MODEL_PATH):
	$(call fetch_gguf,~780 MB,$(QWEN35_MODEL_URL))

fetch-qwen35-model: $(QWEN35_MODEL_PATH)
	$(call verify_sha256,$(QWEN35_MODEL_PATH),$(QWEN35_MODEL_SHA256))
	@echo "Qwen3.5 reference model ready (SHA-256 verified): $(QWEN35_MODEL_PATH)"

fetch-llama-model: $(LLAMA_MODEL_PATH)
	$(call verify_sha256,$(LLAMA_MODEL_PATH),$(LLAMA_MODEL_SHA256))
	@echo "Llama reference model ready (SHA-256 verified): $(LLAMA_MODEL_PATH)"

# Gemma 4 audio tower (~590 MB), extracted from the public checkpoint via
# HTTP Range requests — tools/fetch_audio_tower.py pulls only the
# model.audio_tower.* / model.embed_audio.* byte ranges out of the 9.7 GB
# file. Content-pinned like fetch-llama-model: the script refuses an output
# whose SHA-256 differs, so a changed upstream fails before a test runs.
# Not part of AUTO_FETCH_MODEL; CI's audio-smoke job fetches it explicitly.
AUDIO_TOWER_PATH   ?= audio_bench/audio_tower.safetensors
AUDIO_TOWER_SHA256 := d6c45a6c276212dc3a793e66dfc588d89c12d1ac92c0e4b85494390ca848cd77

$(AUDIO_TOWER_PATH):
	@python3 tools/fetch_audio_tower.py -o "$@" --sha256 $(AUDIO_TOWER_SHA256)

fetch-audio-tower: $(AUDIO_TOWER_PATH)
	@echo "Audio tower ready (SHA-256 verified on fetch): $(AUDIO_TOWER_PATH)"

bench: bin $(BENCH_MODEL_PATH)
	@python3 tools/bench_reproduce.py --gguf "$(BENCH_MODEL_PATH)" \
	  --target "$(TARGET)" --mode "$(MODE)" $(BENCH_ARGS)

# Modality-specific multimodal benches — runnable separately so a user
# benching the vision pipeline doesn't pay for audio/quality suites.
# Each just filters the bench_<modality>_* binaries; argument-less
# invocation hits the default test asset paths.
bench-vision: bin
	@GEIST_INCLUDE_BENCH=1 mk/run-tests.sh $(TEST_BIN_DIR) "bench_vision_"

bench-video: bin
	@GEIST_INCLUDE_BENCH=1 mk/run-tests.sh $(TEST_BIN_DIR) "bench_video_"

bench-audio: bin
	@GEIST_INCLUDE_BENCH=1 mk/run-tests.sh $(TEST_BIN_DIR) "bench_audio_"

# All multimodal encoders end-to-end. Useful as a single CI gate.
bench-mm: bench-vision bench-video bench-audio

# Reproducible quality/performance suites — all drive tools/bench_quality_perf.py,
# which records a row into benchmark/results/APPLE.md only when the run sets a new best
# for that (model, host, os, target/mode, threads) key. BENCH_REF_* are read by the
# quality/compare suites only; harmless (empty) for the plain perf suites.
BENCH_PY = BENCH_GGUF="$(BENCH_GGUF)" BENCH_THREADS="$(BENCH_THREADS)" \
           BENCH_REF_GGUF="$(BENCH_REF_GGUF)" BENCH_REF_BIN="$(BENCH_REF_BIN)" \
           python3 tools/bench_quality_perf.py --target "$(TARGET)" --mode "$(MODE)" \
             --bin-dir "$(TEST_BIN_DIR)" --out-dir "$(BENCH_OUT_DIR)" \
             --benchmark-md benchmark/results/APPLE.md --record --suite

bench-small:            bin ; @$(BENCH_PY) small
bench-detailed:         bin ; @$(BENCH_PY) detailed
bench-quality-small:    bin ; @$(BENCH_PY) quality-small
bench-quality-detailed: bin ; @$(BENCH_PY) quality-detailed
bench-compare-ref:      bin ; @$(BENCH_PY) compare-ref

# Quality: MMLU accuracy via the self-contained tools/eval_mmlu.py harness
# (drives the eval_geist REPL, tokenizes with the model's OWN GGUF tokenizer —
# no external HF tokenizer, no chat-template parity issue; 5-shot base-completion
# cloze). Needs `pip install datasets` for the real cais/mmlu set. Override
# MMLU_LIMIT/MMLU_SHOTS; MMLU_LIMIT=0 runs the full ~14k-question set.
MMLU_LIMIT ?= 200
MMLU_SHOTS ?= 5
bench-mmlu: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) OMP_WAIT_POLICY=active python3 tools/eval_mmlu.py \
	  --bin $(BIN_DIR)/tools/eval_geist \
	  --gguf "$${GEIST_GGUF_PATH:-$(abspath $(MODEL_PATH))}" \
	  --hf --shuffle --limit $(MMLU_LIMIT) --shots $(MMLU_SHOTS)

# Cleanup.
clean:
	@rm -rf build/$(TARGET)/$(MODE) lib/$(TARGET)/$(MODE) bin/$(TARGET)/$(MODE)
	@echo "Cleaned $(TARGET)/$(MODE)."

distclean:
	@rm -rf build lib bin
	@rm -f *.npy *.bin test_* eval_geist bench_sgemv summary.json module_tree.txt tokens_ref.txt
	@echo "Cleaned all targets, modes, and temporary files."

# Code formatting via clang-format. Reads .clang-format from repo root.
# `make format` rewrites in place; `make format-check` is dry-run for CI.
# Covers the whole src/ tree (recursive), tests/, and any root-level *.c/*.h.
# third_party/ is vendored and intentionally excluded, as are the GENERATED
# SPIR-V shader headers (*_spv.h, emitted by `make vulkan-shaders`) — machine
# output is not subject to the style gate and reformatting it would churn.
FORMAT_FILES := $(wildcard *.c *.h tests/*.c tests/*.h) \
                $(shell find src \( -name '*.c' -o -name '*.h' \) ! -name '*_spv.h')

format:
	@clang-format -i $(FORMAT_FILES)
	@echo "Formatted $(words $(FORMAT_FILES)) files."

format-check:
	@clang-format --dry-run --Werror $(FORMAT_FILES) && \
	echo "All $(words $(FORMAT_FILES)) files conform to .clang-format"

# Help text.
help:
	@printf '%s\n' \
	"geist build system   (detected TARGET=$(TARGET), MODE=$(MODE), CC=$(CC))" \
	"" \
	"Build & run:" \
	"  make                       lib + dev binaries for this TARGET/MODE" \
	"  make run ARGS='m.gguf \"hi\"'      build + run examples/simple_generate" \
	"  make lib | bin             only the static lib | only the binaries" \
	"  make MODE=debug|asan|tsan|cov|perf   gdb | ASan+UBSan | TSan races | coverage | -O3+g profiling" \
	"  make clean | distclean     remove current TARGET/MODE | remove everything" \
	"" \
	"Test:" \
	"  make test                  unit + int + py  (auto-fetches model; AUTO_FETCH_MODEL=0 to skip)" \
	"  make test-unit|test-int|test-e2e|test-all   [FILTER=substr]" \
	"  make fetch-model [HF_TOKEN=..]              download reference GGUF (~3.1 GB)" \
	"  make fetch-llama-model                      download SmolLM2 fixture (~369 MB, SHA-pinned)" \
	"  make fetch-qwen3-model                      download Qwen3-0.6B fixture (~609 MB, SHA-pinned)" \
	"  make fetch-qwen35-model                     download Qwen3.5-0.8B hybrid fixture (~780 MB, SHA-pinned)" \
	"  make fetch-audio-tower                      extract Gemma 4 audio tower (~590 MB, SHA-pinned)" \
	"  make fetch-e4b-model [HF_TOKEN=..]          download Gemma 4 E4B GGUF (~4.6 GB)" \
	"" \
	"Bench (timing/quality tools, not pass/fail):" \
	"  make bench                                  reproducible cross-engine benchmark" \
	"  make fetch-bench-model                      download the BitNet GGUF bench needs" \
	"  make bench-smoke | bench-mm                 raw probes | multimodal encoders" \
	"  make bench-small | bench-detailed           record perf to benchmark/results/APPLE.md" \
	"  make bench-quality-small|-detailed          MMLU acc -> benchmark/results/APPLE.md" \
	"  make bench-compare-ref BENCH_REF_URL=...    MMLU vs a running llama-server" \
	"  make bench-mmlu [MMLU_LIMIT=0]              MMLU accuracy (pip install datasets)" \
	"" \
	"Format:  make format | format-check          (clang-format, reads .clang-format)" \
	"" \
	"Release: make release-check                  candidate metadata and published ancestry" \
	"         make release-state-check            GitHub latest tag, ancestry and assets" \
	"" \
	"Targets: mac, mac-omp (Accelerate), pi5 (OpenBLAS+OpenMP), linux" \
	"  cross-compile:   make TARGET=pi5 CC=aarch64-linux-gnu-gcc-14" \
	"  dependency-free static ARM build:   make TARGET=pi5 GEMM_PROVIDER=native EXTRA_LDFLAGS=-static"
