#!/bin/sh
# Verify the public state that users reach through /releases/latest. Unlike
# check-version.sh, this deliberately crosses the repository/GitHub boundary.
# Run after publishing and from the scheduled release-state workflow.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

repo=${GITHUB_REPOSITORY:-geisten/geistlib}
version=$(sed -n 's/.*GEIST_VERSION_STRING "\([0-9][0-9.]*\)".*/\1/p' include/geist.h | head -1)
[ -n "$version" ] || { echo "check-published-release: no GEIST_VERSION_STRING" >&2; exit 2; }
expected_tag="v$version"

command -v gh >/dev/null 2>&1 \
    || { echo "check-published-release: gh is required" >&2; exit 2; }

latest_tag=$(gh api "repos/$repo/releases/latest" --jq .tag_name)
if [ "$latest_tag" != "$expected_tag" ]; then
    echo "check-published-release: latest is $latest_tag, source expects $expected_tag" >&2
    exit 1
fi

draft=$(gh release view "$latest_tag" --repo "$repo" --json isDraft --jq .isDraft)
prerelease=$(gh release view "$latest_tag" --repo "$repo" --json isPrerelease --jq .isPrerelease)
if [ "$draft" != false ] || [ "$prerelease" != false ]; then
    echo "check-published-release: $latest_tag is draft=$draft prerelease=$prerelease" >&2
    exit 1
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/geist-release-state.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM

printf '%s\n' \
    SHA256SUMS \
    geist-bitnet-linux-arm64 \
    geist-bitnet-linux-x86_64 \
    geist-linux-arm64 \
    geist-linux-x86_64 \
    libgeist-linux-arm64.tar.gz \
    libgeist-linux-x86_64.tar.gz \
    libgeist-macos-arm64.tar.gz | sort > "$tmpdir/expected-assets"
gh release view "$latest_tag" --repo "$repo" --json assets \
    --jq '.assets[].name' | sort > "$tmpdir/published-assets"
if ! diff -u "$tmpdir/expected-assets" "$tmpdir/published-assets"; then
    echo "check-published-release: public asset set is incomplete or unexpected" >&2
    exit 1
fi

git fetch -q --tags origin main
tag_sha=$(git rev-list -n 1 "$latest_tag")
if ! git merge-base --is-ancestor "$tag_sha" origin/main; then
    echo "check-published-release: $latest_tag ($tag_sha) is not an ancestor of origin/main" >&2
    exit 1
fi

echo "published release OK: $latest_tag ($tag_sha), complete assets, ancestor of main"
