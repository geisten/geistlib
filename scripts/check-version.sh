#!/bin/sh
# check-version.sh — fail if the version drifts across the docs.
#
# Single source of truth: include/geist.h GEIST_VERSION_STRING (what
# `geist --version` prints). The README carries the same version in three spots
# (badge, Status line, citation) and CITATION.cff in a fourth; they must all
# agree. The comment used to say three and the code checked four — which is the
# harmless version of the drift this script exists to prevent. Run in CI on
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

# CITATION.cff feeds the repository's "Cite this repository" button. It was not
# checked here, and drifted two releases behind unnoticed.
cff=$(sed -n 's/^version: "\([0-9][0-9.]*\)".*/\1/p' CITATION.cff | head -1)
check "CITATION.cff version" "$cff"

# The numeric components are a second source of the same fact, and
# geist_version_components() is what a consumer version-gates on. In 0.7.0
# MINOR still read 6, so the library reported 0.6.0 while the string said
# 0.7.0. Reassemble and compare rather than trusting them to be edited together.
maj=$(sed -n 's/.*GEIST_VERSION_MAJOR \([0-9][0-9]*\).*/\1/p' include/geist.h | head -1)
min=$(sed -n 's/.*GEIST_VERSION_MINOR \([0-9][0-9]*\).*/\1/p' include/geist.h | head -1)
pat=$(sed -n 's/.*GEIST_VERSION_PATCH \([0-9][0-9]*\).*/\1/p' include/geist.h | head -1)
check "GEIST_VERSION_MAJOR/MINOR/PATCH" "$maj.$min.$pat"

if [ "$fail" -eq 0 ]; then
  echo "version OK: $hdr  (geist.h string == components == README == CITATION.cff)"
else
  echo "→ source of truth is include/geist.h GEIST_VERSION_STRING ($hdr);"
  echo "  update the mismatching spots above to match."
  exit 1
fi
