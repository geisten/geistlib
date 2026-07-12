# geist Test-Suite

Tests live in `tests/` and are built as standalone binaries via the root
Makefile. Each test is its own `main()`. There is no test framework — the
contract is purely the **exit code** plus optional stdout output.

## Running Tests

```sh
make test          # all unit + integration tests (excludes _e2e)
make test-unit     # only fast kernel-level tests (suffix _unit)
make test-int      # only multi-module integration tests (suffix _int)
make test-e2e      # only end-to-end tests (suffix _e2e — slow, may need GGUF)
make test-all      # unit + int + e2e
make test-ha       # HA adapter, dynamic tools, install/rollback contracts
make dynamic-example-host # compile the independent Phase-3 host

# Filtering
make test FILTER=q3k       # only tests whose name contains "q3k"
make test FILTER=audio     # only audio-related tests

# With sanitizers
make MODE=asan test
```

`make test-ha` also validates `repository.yaml` and the HA app scaffold:
architectures, disabled privileges/APIs/ports/mounts, AppArmor boundaries,
`/data` persistence, healthcheck, and build-only multi-arch workflow.

Dynamic-tools safety is split across `test_json_schema_v1_unit`,
`test_dynamic_tools_v1_unit`, `test_dynamic_request_v1_unit`,
`test_dynamic_arguments_v1_unit`, `test_dynamic_host_v1_unit`,
`test_agent_unit`, `test_ha_health.py`, `test_ha_history.py`,
`test_ha_operability.py`, and
`test_ha_dynamic_tools_v1.py`. A real model/socket/host
transcript is recorded in
`docs/benchmarks/dynamic-tools-v1-e2e-20260712.txt`.

## Exit-Code Convention (automake-compatible)

| Code | Meaning |
|------|---------|
| 0    | PASS — test verified the behavior |
| 77   | SKIPPED — pre-condition not met (no GGUF, wrong hardware, etc.) |
| 99   | ERROR — test harness broke (allocation failed, missing dep) |
| other| FAIL — assertion or comparison failed |

stdout is informational. The exit code is authoritative.

## Test-Naming Convention

| Suffix       | Speed | Scope | Example |
|--------------|-------|-------|---------|
| `_unit`      | <1s   | one kernel/function | `test_q3k_unit.c` |
| `_int`       | <30s  | several modules together | `test_attention_int.c` |
| `_e2e`       | minutes | full LM forward, GGUF-bound | `test_chat_audio_e2e.c` |

Bench-style timing probes (no pass/fail) live alongside as `bench_*.c` and
are run via `make bench` rather than the test runner.

## Helpers

`tests/test_helpers.h` provides:

- Exit-code constants: `GEIST_TEST_PASS`, `GEIST_TEST_FAIL`, `GEIST_TEST_SKIP`, `GEIST_TEST_ERROR`
- `GEIST_SKIP("reason")` — emit reason + exit 77
- `GEIST_SKIP_IF(cond, "reason")` — conditional skip
- `GEIST_REQUIRE_GGUF(var)` — declare `const char* var = geist_test_find_gguf()` and skip if not found
- `geist_fp32_close(a, b, rtol, atol)` — combined rel+abs tolerance
- `geist_fp32_close_array(a, b, n, rtol, atol)` — vector variant; returns first-failing index or -1

Recommended tolerances:

| Pfad                  | rtol  | atol |
|-----------------------|-------|------|
| FP32-FP32 reference   | 1e-5  | 1e-7 |
| Quantized W3A8/W4A8   | 1e-3  | 1e-2 |

## Environment Variables

| Var                | Effect |
|--------------------|--------|
| `GEIST_GGUF_PATH`  | Override GGUF search; tests skip cleanly if file is missing. |

## Writing a New Test

```c
#include "test_helpers.h"

int main(int argc, char** argv) {
    /* Optional: skip on wrong hardware */
    #if !defined(__ARM_NEON)
    GEIST_SKIP("requires NEON");
    #endif

    /* Optional: skip if GGUF needed */
    GEIST_REQUIRE_GGUF(model_path);

    /* ... do the test ... */

    if (some_failure) {
        fprintf(stderr, "expected X, got Y\n");
        return GEIST_TEST_FAIL;
    }
    return GEIST_TEST_PASS;
}
```
