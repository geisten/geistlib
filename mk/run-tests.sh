#!/bin/sh
# mk/run-tests.sh — discover, run, and report on test binaries.
#
# Usage:
#   mk/run-tests.sh <bin_dir> [filter]
#
# Args:
#   bin_dir    Directory containing test binaries (e.g. bin/mac/release/tests)
#   filter     Optional substring; only binaries containing it are run.
#              Empty / unset means run all.
#
# Exit-code convention (per binary, automake-style):
#   0    PASS
#   77   SKIPPED (with reason on stdout)
#   99   ERROR (test harness broke, not the code-under-test)
#   *    FAIL
#
# This script returns:
#   0 if all selected tests passed (skips count as success)
#   1 if any test failed
#   2 if the bin_dir does not exist or contains no tests
#
# POSIX sh, no bash. The build dropped its bash dependency (mk/detect-target.sh
# is #!/bin/sh); this script was the last thing forcing it, so on an image
# without bash — Alpine's build-base does not pull one in — the suite died with
# a shell error instead of a test result. Selected binaries ride in the
# positional parameters instead of an array, and failure details go to a temp
# file instead of parallel indexed arrays.

set -u

BIN_DIR="${1:-}"
FILTER="${2:-}"

if [ -z "$BIN_DIR" ]; then
    echo "Usage: $0 <bin_dir> [filter]" >&2
    exit 99
fi

if [ ! -d "$BIN_DIR" ]; then
    echo "ERROR: bin_dir '$BIN_DIR' does not exist." >&2
    echo "       Build tests first via 'make bin'." >&2
    exit 2
fi

# Collect candidates. Benches (bench_*) are NOT included by default — they
# are timing tools, not pass/fail tests; route via `make bench`. Set
# GEIST_INCLUDE_BENCH=1 to also pick up bench_* binaries.
#
# A glob matching nothing expands to itself, so every candidate is checked with
# -x below; that drops the literal pattern along with any non-executable.
set -- "$BIN_DIR"/test_*
if [ "${GEIST_INCLUDE_BENCH:-0}" = "1" ]; then
    set -- "$@" "$BIN_DIR"/bench_*
fi

# Filter in place: shift each candidate off the front and append the keepers to
# the back, exactly $# times, so what remains is the kept set.
n_candidates=$#
i=0
while [ "$i" -lt "$n_candidates" ]; do
    b=$1
    shift
    i=$((i + 1))
    [ -x "$b" ] || continue
    if [ -n "$FILTER" ]; then
        case $(basename "$b") in
            *"$FILTER"*) ;;
            *) continue ;;
        esac
    fi
    set -- "$@" "$b"
done

if [ $# -eq 0 ]; then
    if [ -n "$FILTER" ]; then
        echo "No tests in '$BIN_DIR' match filter '$FILTER' (skipping)."
    else
        echo "No tests built in '$BIN_DIR'. Run 'make bin' first."
        exit 2
    fi
    # Empty filter-result is informational — exit 0 so `make test-unit` is
    # benign before the unit-suffix migration (Phase E-4) lands.
    exit 0
fi

PASS=0
FAIL=0
SKIP=0
ERROR=0

# Failure details accumulate in a temp file: POSIX sh has no arrays, and a
# failing test's output is multi-line, so a delimiter-joined variable would be
# fragile.
FAILED_LOG=$(mktemp) || exit 99
trap 'rm -f "$FAILED_LOG"' EXIT HUP INT TERM

# ANSI colors (optional — tee-friendly).
if [ -t 1 ]; then
    esc=$(printf '\033')
    C_GREEN="${esc}[32m"
    C_RED="${esc}[31m"
    C_YELLOW="${esc}[33m"
    C_GREY="${esc}[90m"
    C_RESET="${esc}[0m"
else
    C_GREEN=
    C_RED=
    C_YELLOW=
    C_GREY=
    C_RESET=
fi

START_TIME=$(date +%s)
TOTAL=$#

for bin in "$@"; do
    name=$(basename "$bin")
    out=$("$bin" 2>&1)
    rc=$?

    case $rc in
        0)
            PASS=$((PASS + 1))
            printf "  ${C_GREEN}PASS${C_RESET}  %s\n" "$name"
            ;;
        77)
            SKIP=$((SKIP + 1))
            reason=$(echo "$out" | head -1)
            printf "  ${C_YELLOW}SKIP${C_RESET}  %s ${C_GREY}(%s)${C_RESET}\n" "$name" "$reason"
            ;;
        99)
            ERROR=$((ERROR + 1))
            { printf '\n--- %s ---\n' "$name"; echo "$out" | sed 's/^/  /'; } >>"$FAILED_LOG"
            printf "  ${C_RED}ERR${C_RESET}   %s\n" "$name"
            ;;
        *)
            FAIL=$((FAIL + 1))
            { printf '\n--- %s ---\n' "$name"; echo "$out" | sed 's/^/  /'; } >>"$FAILED_LOG"
            printf "  ${C_RED}FAIL${C_RESET}  %s ${C_GREY}(exit=%d)${C_RESET}\n" "$name" "$rc"
            ;;
    esac
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo
echo "Ran $TOTAL test(s) in ${ELAPSED}s: ${C_GREEN}${PASS} passed${C_RESET}, ${C_YELLOW}${SKIP} skipped${C_RESET}, ${C_RED}${FAIL} failed${C_RESET}, ${C_RED}${ERROR} error${C_RESET}"

if [ -s "$FAILED_LOG" ]; then
    echo
    echo "=== Failure details ==="
    cat "$FAILED_LOG"
fi

if [ "$FAIL" -gt 0 ] || [ "$ERROR" -gt 0 ]; then
    exit 1
fi
exit 0
