#!/bin/sh
# scripts/check_handles.sh - a handle of the wrong type fails to compile (#347).
#
# Each tests/static/*_wrong_handle.c must compile as is, and fail with an
# incompatible-pointer diagnostic for each GEIST_WRONG_HANDLE case in it.
# The clean compile proves a case fails for its wrong handle, not for a
# broken include. CC and CFLAGS come from `make check-handles`.
set -u
cd "$(dirname "$0")/.." || exit 99
cc=${CC:-cc}
flags="${CFLAGS:-} -fsyntax-only -Werror=incompatible-pointer-types"
log=$(mktemp) || exit 99
trap 'rm -f "$log"' EXIT

total=0
for src in tests/static/*_wrong_handle.c; do
    # shellcheck disable=SC2086 # flags is a word list
    if ! $cc $flags "$src" >"$log" 2>&1; then
        cat "$log"
        echo "check-handles: $src does not compile with its handles in place"
        exit 1
    fi
    cases=$(sed -n 's/^#elif GEIST_WRONG_HANDLE == \([0-9]*\)$/\1/p' "$src")
    n=0
    for c in $cases; do
        n=$((n + 1))
        # shellcheck disable=SC2086
        if $cc $flags -DGEIST_WRONG_HANDLE="$c" "$src" >"$log" 2>&1; then
            echo "check-handles: case $c of $src compiled; its wrong handle went unnoticed"
            exit 1
        fi
        if ! grep -qi 'incompatible pointer type' "$log"; then
            cat "$log"
            echo "check-handles: case $c of $src failed, but not on the handle type"
            exit 1
        fi
    done
    [ "$n" -gt 0 ] || { echo "check-handles: no cases in $src"; exit 1; }
    total=$((total + n))
done
[ "$total" -gt 0 ] || { echo "check-handles: no tests/static/*_wrong_handle.c"; exit 1; }
echo "check-handles: $total wrong handles rejected at compile time"
