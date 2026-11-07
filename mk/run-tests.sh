#!/bin/sh
# mk/run-tests.sh — discover, run, and report on test binaries.
#
# Usage:
#   mk/run-tests.sh <bin_dir> [filter [substr]]
#
# Args:
#   bin_dir    Directory containing test binaries (e.g. bin/mac/release/tests)
#   filter     Optional substring; only binaries containing it are run.
#              Empty / unset means run all.
#   substr     Optional literal substring ANDed with filter; the Makefile
#              passes FILTER here (`make test-unit FILTER=q3k`).
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
# A passing test shows its LAST line beside the verdict — where suites print
# their summary numbers (cosine, WER, tokens/s). GEIST_TEST_VERBOSE=1 shows
# every line of every test.
#
# With GITHUB_STEP_SUMMARY set, the verdict line and the skip table are also
# appended there, so skips are visible.
# GEIST_EXPECTED_SKIPS=<file> (one test name per line, # comments) turns any
# skip NOT on the list into a failure — the list is what a leg is allowed to
# skip, so a fixture that goes missing fails instead of shrinking the suite.
# GEIST_TEST_JOBS=N (default 1) runs N tests at a time; GEIST_TEST_FIRST
# (names) starts those first. Each line carries the test's wall time, [Ns].
#
# POSIX sh, no bash (minimal images such as Alpine ship none). Selected
# binaries ride in the positional parameters; verdicts and failure details go
# to a temp dir.

set -u

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

# run_one <bin> <dir>: run one test binary and print its verdict line. Leaves
# <dir>/<name>.verdict (PASS, SKIP, FAIL or ERR), and the details the summary
# needs in <dir>/<name>.failed and <dir>/<name>.skipped. The line is printed
# whole, in one printf, so parallel runners do not interleave inside it.
run_one() {
    bin=$1
    dir=$2
    name=$(basename "$bin")
    t0=$(date +%s)
    out=$("$bin" 2>&1)
    rc=$?
    secs=$(($(date +%s) - t0))

    verbose=
    if [ "${GEIST_TEST_VERBOSE:-0}" = "1" ] && [ -n "$out" ]; then
        verbose=$(printf '\n--- %s ---\n%s' "$name" "$(echo "$out" | sed 's/^/  /')")
    fi

    case $rc in
        0)
            verdict=PASS
            # A passing test's summary is its last line, by convention across
            # this suite. Showing it keeps the table one-line-per-test while
            # the numbers stay visible; GEIST_TEST_VERBOSE=1 shows the rest.
            summary=$(echo "$out" | grep -v '^[[:space:]]*$' | tail -1)
            if [ -n "$summary" ]; then
                line=$(printf "  ${C_GREEN}PASS${C_RESET}  %s ${C_GREY}[%ss] (%s)${C_RESET}" "$name" "$secs" "$summary")
            else
                line=$(printf "  ${C_GREEN}PASS${C_RESET}  %s ${C_GREY}[%ss]${C_RESET}" "$name" "$secs")
            fi
            ;;
        77)
            verdict=SKIP
            reason=$(echo "$out" | grep -v '^[[:space:]]*$' | head -1)
            printf '%s\t%s\n' "$name" "$reason" >"$dir/$name.skipped"
            if [ -n "${GEIST_EXPECTED_SKIPS:-}" ] && ! grep -qx "$name" "$GEIST_EXPECTED_SKIPS"; then
                verdict=FAIL
                { printf '\n--- %s ---\n' "$name"
                  echo "  UNEXPECTED SKIP: not listed in $GEIST_EXPECTED_SKIPS"
                  echo "$out" | sed 's/^/  /'; } >"$dir/$name.failed"
                line=$(printf "  ${C_RED}FAIL${C_RESET}  %s ${C_GREY}(unexpected skip: %s)${C_RESET}" "$name" "$reason")
            else
                line=$(printf "  ${C_YELLOW}SKIP${C_RESET}  %s ${C_GREY}(%s)${C_RESET}" "$name" "$reason")
            fi
            ;;
        99)
            verdict=ERR
            { printf '\n--- %s ---\n' "$name"; echo "$out" | sed 's/^/  /'; } >"$dir/$name.failed"
            line=$(printf "  ${C_RED}ERR${C_RESET}   %s ${C_GREY}[%ss]${C_RESET}" "$name" "$secs")
            ;;
        *)
            verdict=FAIL
            { printf '\n--- %s ---\n' "$name"; echo "$out" | sed 's/^/  /'; } >"$dir/$name.failed"
            line=$(printf "  ${C_RED}FAIL${C_RESET}  %s ${C_GREY}(exit=%d) [%ss]${C_RESET}" "$name" "$rc" "$secs")
            ;;
    esac
    echo "$verdict" >"$dir/$name.verdict"
    if [ -n "$verbose" ]; then
        printf '%s\n%s\n' "$verbose" "$line"
    else
        printf '%s\n' "$line"
    fi
}

# Re-entry for GEIST_TEST_JOBS > 1: xargs runs `run-tests.sh --run-one <dir>
# <bin>` per test.
if [ "${1:-}" = "--run-one" ]; then
    run_one "$3" "$2"
    exit 0
fi

BIN_DIR="${1:-}"
FILTER="${2:-}"
SUBSTR="${3:-}"

if [ -z "$BIN_DIR" ]; then
    echo "Usage: $0 <bin_dir> [filter [substr]]" >&2
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
        # The three suite classes match the NAME SUFFIX, a user filter any
        # substring, so `_int` does not pick up *_int8_unit.
        case "$FILTER" in
            _unit | _int | _e2e) pat="*$FILTER" ;;
            *) pat="*$FILTER*" ;;
        esac
        case $(basename "$b") in
            $pat) ;;
            *) continue ;;
        esac
    fi
    # Quoted, so SUBSTR matches literally; empty matches everything.
    case ${b##*/} in
        *"$SUBSTR"*) ;;
        *) continue ;;
    esac
    set -- "$@" "$b"
done

if [ $# -eq 0 ]; then
    if [ -n "$FILTER$SUBSTR" ]; then
        echo "No tests in '$BIN_DIR' match filter '$FILTER'${SUBSTR:+ and '$SUBSTR'} (skipping)."
    else
        echo "No tests built in '$BIN_DIR'. Run 'make bin' first."
        exit 2
    fi
    # An empty filter result is informational, not a failure.
    exit 0
fi

PASS=0
FAIL=0
SKIP=0
ERROR=0

# Per-test verdicts and details land in a temp dir (run_one); the summary
# below reads them back in suite order. POSIX sh has no arrays, and a failing
# test's output is multi-line, so a delimiter-joined variable would be fragile.
WORK_DIR=$(mktemp -d) || exit 99
FAILED_LOG="$WORK_DIR/failed.log"
SKIPPED_LOG="$WORK_DIR/skipped.log"
: >"$FAILED_LOG"
: >"$SKIPPED_LOG"
trap 'rm -rf "$WORK_DIR"' EXIT HUP INT TERM

if [ -n "${GEIST_EXPECTED_SKIPS:-}" ] && [ ! -r "$GEIST_EXPECTED_SKIPS" ]; then
    echo "ERROR: GEIST_EXPECTED_SKIPS='$GEIST_EXPECTED_SKIPS' is not readable." >&2
    exit 99
fi

START_TIME=$(date +%s)
TOTAL=$#
JOBS=${GEIST_TEST_JOBS:-1}
case $JOBS in
    '' | *[!0-9]* | 0) echo "ERROR: GEIST_TEST_JOBS='$JOBS' is not a positive integer." >&2; exit 99 ;;
esac

if [ "$JOBS" -gt 1 ]; then
    # GEIST_TEST_JOBS=N runs N tests at a time, verdicts in completion order.
    # The tests start in suite order, except that GEIST_TEST_FIRST (test names,
    # space-separated) start first: the longest ones, so the run does not wait
    # on one that started last. Only for suites whose tests share no fixed
    # file names; gcov merges each process's counters into the .gcda files
    # under a file lock, so a coverage build may run in parallel too.
    {
        for f in ${GEIST_TEST_FIRST:-}; do
            for bin in "$@"; do
                [ "${bin##*/}" = "$f" ] && printf '%s\n' "$bin"
            done
        done
        for bin in "$@"; do
            case " ${GEIST_TEST_FIRST:-} " in
                *" ${bin##*/} "*) ;;
                *) printf '%s\n' "$bin" ;;
            esac
        done
    } | xargs -I {} -P "$JOBS" sh "$0" --run-one "$WORK_DIR" {}
else
    for bin in "$@"; do
        run_one "$bin" "$WORK_DIR"
    done
fi

for bin in "$@"; do
    name=$(basename "$bin")
    verdict=$(cat "$WORK_DIR/$name.verdict" 2>/dev/null)
    case $verdict in
        PASS) PASS=$((PASS + 1)) ;;
        SKIP) SKIP=$((SKIP + 1)) ;;
        FAIL) FAIL=$((FAIL + 1)) ;;
        ERR) ERROR=$((ERROR + 1)) ;;
        *)
            # The runner itself died before recording a verdict.
            ERROR=$((ERROR + 1))
            printf '\n--- %s ---\n  no verdict recorded (runner died)\n' "$name" >>"$FAILED_LOG"
            ;;
    esac
    [ -f "$WORK_DIR/$name.failed" ] && cat "$WORK_DIR/$name.failed" >>"$FAILED_LOG"
    [ -f "$WORK_DIR/$name.skipped" ] && cat "$WORK_DIR/$name.skipped" >>"$SKIPPED_LOG"
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

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo "### tests \`${FILTER:-all}\` in \`$BIN_DIR\`: $PASS passed, $SKIP skipped, $FAIL failed, $ERROR error"
        if [ -s "$SKIPPED_LOG" ]; then
            echo
            echo "| skipped | reason |"
            echo "| :-- | :-- |"
            # awk, not sed: BSD sed has no \t. Pipes escaped for the table.
            awk -F'\t' '{ gsub(/\|/, "\\|"); print "| " $1 " | " $2 " |" }' "$SKIPPED_LOG"
        fi
        echo
    } >>"$GITHUB_STEP_SUMMARY"
fi

if [ "$FAIL" -gt 0 ] || [ "$ERROR" -gt 0 ]; then
    exit 1
fi
exit 0
