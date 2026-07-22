#!/bin/sh
# check-version.sh — fail if the version drifts across the docs.
#
# Single source of truth: include/geist.h GEIST_VERSION_STRING (what
# `geist --version` prints). The README carries the same version in three
# spots (badge, Status line, citation); they must all agree. Run in CI on
# every PR, and locally before a release.
#
# ponytail: a guard, not a generator — it does not edit anything, it just
# makes drift a build failure. Bump include/geist.h and the README together.
set -eu

hdr=$(sed -n 's/.*GEIST_VERSION_STRING "\([0-9][0-9.]*\)".*/\1/p' include/geist.h | head -1)
[ -n "$hdr" ] || { echo "check-version: could not read GEIST_VERSION_STRING from include/geist.h"; exit 2; }

fail=0
check() {  # $1=label  $2=found
  if [ "$2" != "$hdr" ]; then
    echo "  MISMATCH: $1 = '${2:-<not found>}', expected '$hdr'"
    fail=1
  fi
}

badge=$(sed -n 's/.*status-experimental%20(v\([0-9][0-9.]*\)).*/\1/p' README.md | head -1)
body=$(sed -n 's/.*`geistlib` is \*\*v\([0-9][0-9.]*\).*/\1/p' README.md | head -1)
cite=$(sed -n 's/.*version = {\([0-9][0-9.]*\)}.*/\1/p' README.md | head -1)

check "README status badge" "$badge"
check "README Status line"  "$body"
check "README citation"     "$cite"

if [ "$fail" -eq 0 ]; then
  echo "version OK: $hdr  (include/geist.h == README badge/status/citation)"
else
  echo "→ source of truth is include/geist.h ($hdr); update the README to match."
  exit 1
fi
