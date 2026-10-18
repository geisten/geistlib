#!/bin/sh
# ci-changes.sh — does a pull request change code, or only documentation?
#
# Prints "code=true" or "code=false" for $GITHUB_OUTPUT. The build and test
# jobs of ci.yml run only for code=true; a skipped job reports success, so
# main's required checks still pass for a docs-only pull request.
#
# Documentation: *.md, docs/, benchmark/results/ (raw data), assets/,
# CITATION.cff, LICENSE, NOTICE, and include/geist.h when only its
# GEIST_VERSION_* lines change (a release PR). A documentation file whose path
# a test names (tests/test_bench_tools_py.py reads benchmark/results/APPLE.md)
# counts as code. Anything else, a push, or a missing base: code.
#
# Usage: sh scripts/ci-changes.sh [BASE_SHA]   (default: $BASE_SHA)
#        sh scripts/ci-changes.sh --test       self-check of the path rule
set -eu

# Reads paths on stdin; exit 0 if every one is documentation.
docs_only() {
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        case "$p" in
            include/geist.h)
                # Changed lines other than the version defines are code.
                [ -n "${BASE:-}" ] || return 1
                if git diff -U0 "$BASE"...HEAD -- include/geist.h | grep '^[-+]' \
                        | grep -v '^[-+][-+]' | grep -qv 'GEIST_VERSION_'; then
                    return 1
                fi
                continue ;;
            *.md | docs/* | benchmark/results/* | assets/* | CITATION.cff | LICENSE | NOTICE) ;;
            *) return 1 ;;
        esac
        # ponytail: a literal path match, enough for the one test that reads a doc.
        if [ -z "${NO_TEST_REFS:-}" ] && git grep -qF "$p" -- tests/; then
            return 1
        fi
    done
    return 0
}

if [ "${1:-}" = "--test" ]; then
    export NO_TEST_REFS=1
    ok() { printf '%s\n' "$@" | docs_only || { echo "FAIL: expected docs: $*"; exit 1; }; }
    no() { if printf '%s\n' "$@" | docs_only; then echo "FAIL: expected code: $*"; exit 1; fi; }
    ok README.md docs/OFFLOAD.md benchmark/results/raw/x/run.log CITATION.cff
    ok assets/neuron.png CHANGELOG.md
    no src/engine/session.c
    no docs/OFFLOAD.md Makefile
    no .gitea/workflows/ci.yml
    no include/geist.h   # no BASE: cannot prove a version-only change
    unset NO_TEST_REFS
    printf 'benchmark/results/APPLE.md\n' | docs_only && { echo "FAIL: APPLE.md is read by a test"; exit 1; }
    echo "ci-changes: self-check OK"
    exit 0
fi

BASE=${1:-${BASE_SHA:-}}
if [ -z "$BASE" ] || ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
    echo "code=true"
    exit 0
fi
if git diff --name-only "$BASE"...HEAD | docs_only; then
    echo "code=false"
else
    echo "code=true"
fi
