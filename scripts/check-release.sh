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

# The first released-looking section must be the candidate. Older source-only
# milestones deliberately have unlinked headings, because no tag/release exists
# for them. This prevents a version bump at the bottom of a still-unreleased
# changelog from passing.
first_release=$(sed -n 's/^## \[\([0-9][^]]*\)\].*/\1/p' CHANGELOG.md | head -1)
if [ "$first_release" != "$version" ]; then
    note "FIRST CHANGELOG RELEASE: '$first_release', expected '$version'"
fi

# Every linked heading needs a reference definition. In particular, this catches
# the old state where [0.10.7] existed but [Unreleased] still compared from
# v0.10.6 and no [0.10.7] link was defined.
for heading in $(sed -n 's/^## \[\([^]]*\)\].*/\1/p' CHANGELOG.md); do
    if ! grep -qF "[$heading]:" CHANGELOG.md; then
        note "CHANGELOG LINK MISSING: [$heading] has no reference definition"
    fi
done

# ---- 4. the published base and candidate link agree -----------------------
# Tags are publication history, not source milestones. Fetch them before a
# release check, then require the newest published tag to be in this commit's
# ancestry and use it as the candidate comparison base.
previous_tag=
if [ "$pre_tag" = "1" ]; then
    git fetch -q --tags origin 2>/dev/null || true
fi
for tag in $(git tag --list 'v[0-9]*' --sort=-version:refname); do
    if [ "$tag" != "v$version" ]; then
        previous_tag=$tag
        break
    fi
done
if [ -z "$previous_tag" ]; then
    note "no previous release tag is available"
else
    if ! git merge-base --is-ancestor "$previous_tag^{}" HEAD; then
        note "PUBLISHED HISTORY DISCONNECTED: $previous_tag is not an ancestor of HEAD"
    fi

    previous=${previous_tag#v}
    if [ "$(vnum "$version")" -le "$(vnum "$previous")" ]; then
        note "candidate $version must be newer than published $previous"
    fi
    expected_unreleased="[Unreleased]: https://github.com/geisten/geistlib/compare/v$version...HEAD"
    expected_candidate="[$version]: https://github.com/geisten/geistlib/compare/v$previous...v$version"
    grep -qFx "$expected_unreleased" CHANGELOG.md \
        || note "CHANGELOG [Unreleased] link must be: $expected_unreleased"
    grep -qFx "$expected_candidate" CHANGELOG.md \
        || note "CHANGELOG [$version] link must be: $expected_candidate"
fi

# ---- 5. the tag is free (only when about to create it) --------------------
if [ "$pre_tag" = "1" ]; then
    if git rev-parse -q --verify "refs/tags/v$version" >/dev/null; then
        note "tag v$version already exists"
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
