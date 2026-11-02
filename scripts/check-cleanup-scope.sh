#!/bin/sh
# check-cleanup-scope.sh — GEIST_CLEANUP (src/base/cleanup.h, #349) is for
# load, setup and error paths. Kernels and forward passes are hot: their
# exits stay explicit, so neither the macro nor a bare [[gnu::cleanup]] may
# appear under a kernels/ or forward/ directory.
set -eu

if hits=$(grep -rnE 'GEIST_CLEANUP|gnu::cleanup|cleanup\.h' --include='*.[ch]' \
    $(find src -type d \( -name kernels -o -name forward \))); then
    echo "check-cleanup-scope: scope-exit cleanup in a hot path:" >&2
    echo "$hits" >&2
    echo "Free explicitly there; GEIST_CLEANUP is for cold code (AGENT.md §3)." >&2
    exit 1
fi
echo "cleanup scope OK: no GEIST_CLEANUP under kernels/ or forward/"
