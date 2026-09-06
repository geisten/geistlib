#!/bin/sh
# check-version.sh — fail if the version drifts across the docs.
#
# Single source of truth for the source candidate: include/geist.h
# GEIST_VERSION_STRING (what `geist --version` prints). CITATION.cff mirrors it
# and must agree. The README intentionally obtains the latest *published*
# version from GitHub instead of copying this candidate version: source state
# and publication state are different facts. Run in CI on every PR, and locally
# before a release.
#
# ponytail: a guard, not a generator — it does not edit anything, it just
# makes drift a build failure. Bump include/geist.h and CITATION.cff together.
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

# CITATION.cff feeds the repository's "Cite this repository" button.
cff=$(sed -n 's/^version: "\([0-9][0-9.]*\)".*/\1/p' CITATION.cff | head -1)
check "CITATION.cff version" "$cff"

# The README describes a moving development branch. A literal current version
# there recreates the source-vs-published drift this split is designed to avoid.
if grep -qE 'status-experimental%20\(v[0-9]|`geistlib` is \*\*v[0-9]|version = \{[0-9]' README.md; then
  echo "  MISMATCH: README hard-codes a release version; link releases/latest instead"
  fail=1
fi

# The numeric components are a second source of the same fact, and
# geist_version_components() is what a consumer version-gates on. In 0.7.0
# MINOR still read 6, so the library reported 0.6.0 while the string said
# 0.7.0. Reassemble and compare rather than trusting them to be edited together.
maj=$(sed -n 's/.*GEIST_VERSION_MAJOR \([0-9][0-9]*\).*/\1/p' include/geist.h | head -1)
min=$(sed -n 's/.*GEIST_VERSION_MINOR \([0-9][0-9]*\).*/\1/p' include/geist.h | head -1)
pat=$(sed -n 's/.*GEIST_VERSION_PATCH \([0-9][0-9]*\).*/\1/p' include/geist.h | head -1)
check "GEIST_VERSION_MAJOR/MINOR/PATCH" "$maj.$min.$pat"

if [ "$fail" -eq 0 ]; then
  echo "version OK: $hdr  (geist.h string == components == CITATION.cff; README is dynamic)"
else
  echo "→ source of truth is include/geist.h GEIST_VERSION_STRING ($hdr);"
  echo "  update the mismatching spots above to match."
  exit 1
fi
