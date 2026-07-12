# geist Makefile — entry point.
#
# Usage:
#   make                       # auto-detect TARGET, MODE=release
#   make TARGET=pi5            # cross-compile (override CC if needed)
#   make MODE=asan             # sanitizer build
#   make MODE=asan TARGET=pi5  # combinations
#   make test                  # placeholder (Phase E implements)
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
.PHONY: all lib bin run dynamic-example-host clean distclean help test test-unit test-int test-e2e test-all test-py fetch-model bench bench-small bench-detailed bench-quality-small bench-quality-detailed bench-compare-ref bench-mmlu bench-tooling bench-agent bench-agent-live bench-agent-judge format format-check

# Default goal. The `geist` symlink (built after common.mk pins BIN_DIR) points
# `./geist` at the freshly built CLI so you never type the bin/<target>/<mode> path.
all: lib bin geist

# Pull in target settings (CC, CFLAGS_TARGET, LDFLAGS_TARGET, LDLIBS_TARGET).
include mk/target-$(TARGET).mk

# Pull in common build rules (LIB_FILE, BIN_TARGETS, object/link rules).
include mk/common.mk

# Convenience aggregate goals. `bin` is declared AFTER the EMBED block below:
# make expands prerequisite lists immediately at parse time, and the EMBED
# block swaps the CLI's entry in BIN_TARGETS — declaring `bin` here would
# capture the pre-swap list and link a broken plain tools/geist from an
# embedded geist.o.
lib: $(LIB_FILE)

# `./$(EMBED_NAME)` → the built CLI for the current TARGET/MODE, so the demo is
#   make && OMP_WAIT_POLICY=active ./geist -m model.gguf "What is the capital of France?"
# Re-pointed on every build (cheap); removed by `make distclean`.
#
# EMBED_NAME names the symlink AND the binary under $(BIN_DIR)/tools/. Embedded
# builds default to "geist-embedded", never "geist": if the embedded binary
# lived at tools/geist, any later plain `make` would relink that path WITHOUT
# the model — the ./geist-bitnet symlink would then point at a plain binary
# that parses the prompt as a model path ("model_load failed"). Distinct paths,
# no clobber. A distinct name also avoids the "which one needs a model?"
# confusion — the embedded build takes no model-path argument, unlike `geist`:
#   make EMBED_MODEL=bitnet-2b4t.i2_s.gguf EMBED_NAME=geist-bitnet
#   ./geist-bitnet "The capital of France is"      # no model path — it's baked in
ifneq ($(strip $(EMBED_MODEL)),)
  EMBED_NAME ?= geist-embedded
else
  EMBED_NAME ?= geist
endif
GEIST_BIN := $(BIN_DIR)/tools/$(EMBED_NAME)
geist: $(GEIST_BIN)
	@ln -sf $(GEIST_BIN) $(EMBED_NAME) && echo "./$(EMBED_NAME) -> $(GEIST_BIN)"

# `make run ARGS='-m model.gguf "your prompt" -n 40'` — build, then run the CLI
# with OMP_WAIT_POLICY=active (matters for multi-thread perf on mac-omp).
run: geist
	@OMP_WAIT_POLICY=active ./$(EMBED_NAME) $(ARGS)

# Host-neutral dynamic-tools example. It links no model/runtime and no adapter
# adapter; any application can build the same request/validation contract.
DYNAMIC_EXAMPLE_HOST := $(BIN_DIR)/examples/dynamic_tools_host
dynamic-example-host: $(DYNAMIC_EXAMPLE_HOST)
$(DYNAMIC_EXAMPLE_HOST): examples/dynamic_tools_host.c tools/dynamic_tools_v1.h tools/json_schema_v1.h
	@mkdir -p $(@D)
	$(CC) -std=c23 -Wall -Wextra -Wpedantic -Werror -I. $< -o $@

# ---- Optional: embed a model into the geist CLI (single-binary deploy) -----
# `make EMBED_MODEL=path/to/model.gguf` bakes the GGUF into ./geist via an
# .incbin stub, so the binary needs no model file — the CLI then takes only a
# prompt:  ./geist "The capital of France is".  Zero-copy: weights alias the
# in-binary .rodata blob. Small models only — the binary grows by the model
# size; >~1.5 GB exceeds the 2 GB release limit. The GGUF must carry its own
# tokenizer (no sibling file is searched). Text-only (no external vision/audio).
#
# Toggling EMBED_MODEL flips -DGEIST_EMBEDDED_MODEL on geist.o, which make can't
# see from the source mtime alone (stale geist.o -> link error, or a binary that
# silently ignores the embed). Track the embed state in a stamp file and depend
# geist.o on it: the stamp is rewritten only when the state changes, so switching
# between embedded/file mode rebuilds geist.o automatically and nothing churns
# otherwise. Applies in both branches, so it lives outside the ifneq.
EMBED_TAG         := $(if $(strip $(EMBED_MODEL)),embedded:$(abspath $(EMBED_MODEL)),none)
GEIST_EMBED_STAMP := $(BUILD_DIR)/tools/.geist-embed-state
# When the embed state changes, DELETE geist.o + this mode's binary so they
# rebuild with the right -DGEIST_EMBEDDED_MODEL. We can't rely on a stamp
# prerequisite's mtime: macOS ships GNU make 3.81, whose timestamp comparison is
# whole-second, so a stamp rewritten in the same second as the prior build looks
# "not newer" and the rebuild is skipped. Deleting sidesteps mtime entirely.
# Only $(GEIST_BIN) is deleted — the OTHER mode's binary (e.g. tools/geist-bitnet
# when switching back to plain) stays valid: it was linked from a geist.o that
# matched its own embed state. Runs at parse time.
$(shell mkdir -p $(BUILD_DIR)/tools 2>/dev/null; \
        if [ "$$(cat $(GEIST_EMBED_STAMP) 2>/dev/null)" != "$(EMBED_TAG)" ]; then \
            printf '%s' "$(EMBED_TAG)" > $(GEIST_EMBED_STAMP); \
            rm -f $(BUILD_DIR)/tools/geist.o $(GEIST_BIN); \
        fi)

ifneq ($(strip $(EMBED_MODEL)),)
  ifeq ($(wildcard $(EMBED_MODEL)),)
    $(error EMBED_MODEL='$(EMBED_MODEL)' not found)
  endif
  EMBED_ABS  := $(abspath $(EMBED_MODEL))
  EMBED_SIZE := $(shell wc -c < "$(EMBED_ABS)")
  $(info embedding $(EMBED_MODEL) ($(shell echo $$(($(EMBED_SIZE)/1048576))) MB) into ./$(EMBED_NAME) — runs with no model-path argument)
  ifeq ($(shell test $(EMBED_SIZE) -gt 1610612736 && echo big),big)
    $(warning EMBED_MODEL >1.5 GB — the binary will exceed the 2 GB GitHub-release limit and be unwieldy)
  endif
  EMBED_OBJ := $(BUILD_DIR)/src/engine/embedded_model.o
  # Stub TU: -DGEIST_EMBED_MODEL_PATH points .incbin at the model; rebuild if it changes.
  $(EMBED_OBJ): EMBED_CFLAGS := -DGEIST_EMBED_MODEL_PATH='"$(EMBED_ABS)"'
  $(EMBED_OBJ): $(EMBED_ABS)
  # geist CLI: compile its TU with the embed flag and link the stub object in.
  $(BUILD_DIR)/tools/geist.o: CFLAGS += -DGEIST_EMBEDDED_MODEL
  # Link to tools/$(EMBED_NAME), NOT tools/geist: geist.o is shared between the
  # plain and embedded builds (the stamp above rebuilds it on every mode
  # switch), but if the BINARIES shared a path too, any later plain `make`
  # would clobber the embedded binary — leaving the embed symlink pointing at
  # a model-less CLI. Swap the CLI's entry in BIN_TARGETS so `make bin` builds
  # the embedded path (and never a broken plain link against an embedded
  # geist.o). The generic $(BIN_DIR)/% rule in common.mk can't produce this
  # target (there is no tools/$(EMBED_NAME).o), so link explicitly from geist.o.
  BIN_TARGETS := $(patsubst $(BIN_DIR)/tools/geist,$(GEIST_BIN),$(BIN_TARGETS))
  $(GEIST_BIN): $(BUILD_DIR)/tools/geist.o $(EMBED_OBJ) $(LIB_FILE)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $(BUILD_DIR)/tools/geist.o $(EMBED_OBJ) $(LIB_FILE) $(LDLIBS)
endif

# Declared after the EMBED block on purpose — see the note at `lib:` above.
bin: $(BIN_TARGETS)

# Test runner — invokes mk/run-tests.sh against the test bin directory.
# FILTER is an optional substring; e.g. `make test FILTER=q3k` runs only
# tests whose binary name contains "q3k".
TEST_BIN_DIR := $(BIN_DIR)/tests
BENCH_OUT_DIR ?= bench_runs/quality_perf
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

$(MODEL_PATH):
	@command -v curl >/dev/null 2>&1 || { echo "fetch-model: curl not found in PATH" >&2; exit 1; }
	@mkdir -p $(MODEL_DIR)
	@echo "Downloading $(MODEL_FILE) (~3.1 GB) from:"
	@echo "  $(MODEL_URL)"
	@curl -fL --retry 3 --retry-delay 2 -C - \
	  $(if $(HF_TOKEN),-H "Authorization: Bearer $(HF_TOKEN)",) \
	  -o "$@.part" "$(MODEL_URL)"
	@mv "$@.part" "$@"
	@echo "Saved to $@"

# Benches are timing tools, not tests — separate target. Each bench prints
# its own metrics; runner just reports run/skip/fail status.
bench: bin
	@$(GGUF_ENV) GEIST_INCLUDE_BENCH=1 mk/run-tests.sh $(TEST_BIN_DIR) "bench_"

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
# which records a row into benchmark/BENCHMARK.md only when the run sets a new best
# for that (model, host, os, target/mode, threads) key. BENCH_REF_* are read by the
# quality/compare suites only; harmless (empty) for the plain perf suites.
BENCH_PY = BENCH_GGUF="$(BENCH_GGUF)" BENCH_THREADS="$(BENCH_THREADS)" \
           BENCH_REF_GGUF="$(BENCH_REF_GGUF)" BENCH_REF_BIN="$(BENCH_REF_BIN)" \
           python3 tools/bench_quality_perf.py --target "$(TARGET)" --mode "$(MODE)" \
             --bin-dir "$(TEST_BIN_DIR)" --out-dir "$(BENCH_OUT_DIR)" \
             --benchmark-md benchmark/BENCHMARK.md --record --suite

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

# Quality: function-calling + JSON-generation via tools/eval_tooling.py (also
# self-contained — drives the eval_geist GEN command, no dataset needed). The
# probe set is curated and validates extracted JSON (valid + schema + correct
# function/arguments). TOOLING_SUITE = json | func | all. TOOLING_MIN is a
# quality gate: 0 = report only; >0 fails if correct/total drops below it (CI).
TOOLING_SUITE ?= all
TOOLING_MIN ?= 0
bench-tooling: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) OMP_WAIT_POLICY=active python3 tools/eval_tooling.py \
	  --bin $(BIN_DIR)/tools/eval_geist \
	  --gguf "$${GEIST_GGUF_PATH:-$(abspath $(MODEL_PATH))}" \
	  --suite $(TOOLING_SUITE) --min-correct $(TOOLING_MIN)

# Quality: agent-LAYER reliability (routing / arg lifting / chains) via the
# self-contained tests/bench_agent_eval binary — mechanical per-stage scoring
# over tests/data/agent_eval/cases.jsonl, greedy decode, web tools stubbed
# in-process (no network). AGENT_EVAL_MODE = forced | free | both.
# AGENT_EVAL_MIN gates the forced-mode pass count (exit 1 below it). The fixed
# threshold 43/48 is the level achieved on bitnet-2b4t-i2_s (2026-07): single
# 15/15, chains 6/8, ambig 2/4, neg 3/3, e2e 17/18 (answer-content checks incl.
# the memory roundtrip, multi-turn context carry, and the stock_movers cases).
# Recalibrate (or pass AGENT_EVAL_MIN=0) when evaluating a different model.
AGENT_EVAL_MODE ?= both
AGENT_EVAL_MIN ?= 43
bench-agent: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) $(TEST_BIN_DIR)/bench_agent_eval --mode $(AGENT_EVAL_MODE) --min-pass $(AGENT_EVAL_MIN)

# Manual live-web smoke: the SAME harness with the real web_search/web_fetch
# (curl + DuckDuckGo) over a tiny stable corpus — checks the stub assumptions
# against reality. Network-dependent: report-only, never wired into CI.
# DuckDuckGo rate-limits back-to-back requests quickly; set
# GEIST_SEARX_ENDPOINT=<searxng-url> for a stable search backend.
bench-agent-live: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) $(TEST_BIN_DIR)/bench_agent_eval --mode forced --live-web \
	  tests/data/agent_eval/cases_live.jsonl

# Advisory answer-coherence judge: a second AI (local Ollama, JUDGE_MODEL)
# reads every forced-mode answer and says whether it is a coherent response —
# the gap the mechanical expect-substring check cannot see. Report-only, exit
# 0 always: LLM judges drift, the deterministic gate stays authoritative.
JUDGE_MODEL ?= gemma4:26b
bench-agent-judge: bin $(MODEL_PREREQ)
	@mkdir -p bench_runs/agent_eval
	@$(GGUF_ENV) $(TEST_BIN_DIR)/bench_agent_eval --mode forced \
	  --dump bench_runs/agent_eval/answers.jsonl
	@python3 tools/eval_agent_judge.py \
	  --answers bench_runs/agent_eval/answers.jsonl --model $(JUDGE_MODEL)

# Cleanup.
clean:
	@rm -rf build/$(TARGET)/$(MODE) lib/$(TARGET)/$(MODE) bin/$(TARGET)/$(MODE)
	@echo "Cleaned $(TARGET)/$(MODE)."

distclean:
	@rm -rf build lib bin
	@rm -f geist $(EMBED_NAME) *.npy *.bin test_* eval_geist profile_decode dump_llamacpp_logits bench_sgemv summary.json module_tree.txt tokens_ref.txt
	@echo "Cleaned all targets, modes, and temporary files."

# Code formatting via clang-format. Reads .clang-format from repo root.
# `make format` rewrites in place; `make format-check` is dry-run for CI.
# Covers the whole src/ tree (recursive), tests/, and any root-level *.c/*.h.
# third_party/ is vendored and intentionally excluded.
FORMAT_FILES := $(wildcard *.c *.h tests/*.c tests/*.h) \
                $(shell find src -name '*.c' -o -name '*.h')

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
	"  make                       lib + binaries + ./geist symlink for this TARGET/MODE" \
	"  OMP_WAIT_POLICY=active ./geist -m model.gguf \"prompt\" -n 40    run the CLI" \
	"  make run ARGS='-m m.gguf \"hi\"'   build, then run ./geist with OMP_WAIT_POLICY=active" \
	"  make lib | bin             only the static lib | only the binaries" \
	"  make MODE=debug|asan|perf  -O0+g for gdb | ASan+UBSan | -O3+g for profilers" \
	"  make clean | distclean     remove current TARGET/MODE | remove everything" \
	"" \
	"Test:" \
	"  make test                  unit + int + py  (auto-fetches model; AUTO_FETCH_MODEL=0 to skip)" \
	"  make test-unit|test-int|test-e2e|test-all   [FILTER=substr]" \
	"  make fetch-model [HF_TOKEN=..]              download reference GGUF (~3.1 GB)" \
	"" \
	"Bench (timing/quality tools, not pass/fail):" \
	"  make bench | bench-mm                       raw probes | multimodal encoders" \
	"  make bench-small | bench-detailed           record perf to benchmark/BENCHMARK.md" \
	"  make bench-quality-small|-detailed          MMLU acc -> benchmark/BENCHMARK.md" \
	"  make bench-compare-ref BENCH_REF_URL=...    MMLU vs a running llama-server" \
	"  make bench-mmlu [MMLU_LIMIT=0] | bench-tooling         accuracy (pip install datasets)" \
	"  make bench-agent [AGENT_EVAL_MODE=both]     agent-layer routing/args/chains eval" \
	"" \
	"Format:  make format | format-check          (clang-format, reads .clang-format)" \
	"" \
	"Targets: mac, mac-omp (Accelerate), pi5 (OpenBLAS+OpenMP), linux" \
	"  cross-compile:   make TARGET=pi5 CC=aarch64-linux-gnu-gcc-14" \
	"  dependency-free static ARM build:   make TARGET=pi5 GEMM_PROVIDER=native EXTRA_LDFLAGS=-static"
