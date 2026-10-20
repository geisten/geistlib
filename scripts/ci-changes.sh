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
# --tier prints which CI tier runs (docs/CI_COVERAGE.md "Tiers"):
# "full=true" runs the post-merge tier as well — on anything but a pull
# request, on GitHub (the mirror's pull requests run everything on GitHub's
# runners) and for a pull request labelled ci:full; "label=true" when that
# label is set (the coverage job runs on a pull request only then).
#
# Usage: sh scripts/ci-changes.sh [BASE_SHA]   (default: $BASE_SHA)
#        sh scripts/ci-changes.sh --tier       (GITHUB_EVENT_NAME, GITHUB_SERVER_URL, GITHUB_EVENT_PATH)
#        sh scripts/ci-changes.sh --test       self-check of the path and tier rules
set -eu

# Labels of the pull request in $GITHUB_EVENT_PATH, one per line.
pr_labels() {
    [ -r "${GITHUB_EVENT_PATH:-}" ] || return 0
    python3 -c 'import json, sys
pr = json.load(open(sys.argv[1])).get("pull_request") or {}
for label in pr.get("labels") or []:
    print(label.get("name", ""))' "$GITHUB_EVENT_PATH"
}

tier() {
    label=false
    if pr_labels | grep -qx 'ci:full'; then label=true; fi
    full=$label
    [ "${GITHUB_EVENT_NAME:-}" = pull_request ] || full=true
    [ "${GITHUB_SERVER_URL:-}" != https://github.com ] || full=true
    echo "full=$full"
    echo "label=$label"
}

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
    ok assets/neuron.png CHANGELOG.md changes/123-fix.md
    no src/engine/session.c
    no docs/OFFLOAD.md Makefile
    no .gitea/workflows/ci.yml
    no include/geist.h   # no BASE: cannot prove a version-only change
    unset NO_TEST_REFS
    printf 'benchmark/results/APPLE.md\n' | docs_only && { echo "FAIL: APPLE.md is read by a test"; exit 1; }
    ev=$(mktemp)
    t() {  # t <event> <server> <labels json> <expected tier output>
        printf '{"pull_request": {"labels": %s}}' "$3" > "$ev"
        got=$(GITHUB_EVENT_NAME=$1 GITHUB_SERVER_URL=$2 GITHUB_EVENT_PATH=$ev tier | tr '\n' ' ')
        [ "$got" = "$4" ] || { echo "FAIL: tier $1 $2 $3: got '$got', want '$4'"; rm -f "$ev"; exit 1; }
    }
    G=https://git.geisten.net
    t pull_request $G '[]' 'full=false label=false '
    t pull_request $G '[{"name": "automerge"}]' 'full=false label=false '
    t pull_request $G '[{"name": "ci:full"}]' 'full=true label=true '
    t pull_request https://github.com '[]' 'full=true label=false '
    t push $G '[]' 'full=true label=false '
    t schedule $G '[]' 'full=true label=false '
    t workflow_dispatch $G '[]' 'full=true label=false '
    rm -f "$ev"
    echo "ci-changes: self-check OK"
    exit 0
fi

if [ "${1:-}" = "--tier" ]; then
    tier
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
