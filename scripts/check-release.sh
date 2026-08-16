#!/bin/sh
# check-release.sh — the gate a release must pass, run before the tag.
#
# WHY THIS EXISTS. 0.9.0 nearly shipped wrong twice in one sitting:
#
#   * docs/API_CONTRACT.md and a header said "STABLE since 0.9.0" while
#     GEIST_VERSION_STRING still read 0.8.2 and no tag carried 0.9.0. The
#     promise lived in the source and in no artefact — a consumer pinning the
#     newest tag got a library that did not have it.
#   * CITATION.cff was a fourth version site that check-version.sh checks and
#     its own header comment did not mention. Trusting the comment over the
#     script would have shipped a stale citation.
#
# Both are the same failure: a release is several files agreeing, and agreement
# is not something to verify by memory at the moment you type `git tag`.
#
# ponytail: a guard, not a generator. It edits nothing; it makes a wrong
# release a failed command. Run it via `make release-check`.
#
# Usage:
#   sh scripts/check-release.sh              # validate the tree
#   sh scripts/check-release.sh --pre-tag    # ... and require the tag be unused
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

pre_tag=0
[ "${1:-}" = "--pre-tag" ] && pre_tag=1

fail=0
note() { echo "  $*" >&2; fail=1; }

version=$(sed -n 's/.*GEIST_VERSION_STRING "\([0-9][0-9.]*\)".*/\1/p' include/geist.h | head -1)
[ -n "$version" ] || { echo "check-release: no GEIST_VERSION_STRING in include/geist.h" >&2; exit 2; }
echo "check-release: candidate $version"

# X.Y.Z -> a comparable integer. Two digits per field is plenty for 0.x and
# keeps this arithmetic in POSIX sh.
vnum() {
    IFS=. read -r a b c <<EOF
$1
EOF
    printf '%d' $(( ${a:-0} * 10000 + ${b:-0} * 100 + ${c:-0} ))
}

# ---- 1. the existing guards, so one command is the whole gate --------------
sh scripts/check-version.sh      || fail=1
sh scripts/check-api-contract.sh || fail=1

# ---- 2. no promise dated after the release that carries it ----------------
# A `STABLE since V` with V newer than this version is a commitment no shipped
# artefact fulfils. It is legitimate INSIDE a PR — the promotion lands before
# the release that names it — and wrong at the moment of release, which is why
# this check lives here and not in the per-PR CI.
cand=$(vnum "$version")
for v in $(grep -rhoE 'STABLE since [0-9]+\.[0-9]+\.[0-9]+' include docs \
           | sed 's/STABLE since //' | sort -u); do
    if [ "$(vnum "$v")" -gt "$cand" ]; then
        note "PROMISE AHEAD OF RELEASE: 'STABLE since $v' but this release is $version"
        grep -rn "STABLE since $v" include docs | sed 's/^/    /' >&2
    fi
done

# ---- 3. the changelog names this version, and nothing is stranded ---------
if ! grep -qE "^## \[$(echo "$version" | sed 's/\./\\./g')\] — [0-9]{4}-[0-9]{2}-[0-9]{2}" CHANGELOG.md; then
    note "CHANGELOG.md has no dated '## [$version] — YYYY-MM-DD' section"
fi
# Entries left under [Unreleased] ship without appearing in any release's notes.
unreleased=$(awk '/^## \[Unreleased\]/{f=1;next} /^## \[/{f=0} f' CHANGELOG.md | grep -c '[^[:space:]]' || true)
if [ "$unreleased" -gt 0 ]; then
    note "CHANGELOG.md still has $unreleased line(s) under [Unreleased] — move them into [$version] or they ship undocumented"
fi

# ---- 4. the tag is free (only when about to create it) --------------------
if [ "$pre_tag" = "1" ]; then
    if git rev-parse -q --verify "refs/tags/v$version" >/dev/null; then
        note "tag v$version already exists"
    fi
    git fetch -q --tags origin 2>/dev/null || true
    if git rev-parse -q --verify "refs/tags/v$version" >/dev/null; then
        note "tag v$version exists on origin"
    fi
    if [ -n "$(git status --porcelain)" ]; then
        note "working tree is dirty; a tag would not describe what was tested"
    fi
fi

if [ "$fail" != "0" ]; then
    echo "check-release: FAIL" >&2
    exit 1
fi
echo "check-release: OK — $version is consistent and ready to tag"
