#!/bin/sh
# Verify the public state that users reach through
# https://geisten.net/download/geistlib/latest/. Unlike check-version.sh, this
# deliberately crosses the repository/web boundary. Run after publishing and
# from the scheduled release-state workflow.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

base=${GEIST_DOWNLOAD_URL:-https://geisten.net/download/geistlib}/latest
version=$(sed -n 's/.*GEIST_VERSION_STRING "\([0-9][0-9.]*\)".*/\1/p' include/geist.h | head -1)
[ -n "$version" ] || { echo "check-published-release: no GEIST_VERSION_STRING" >&2; exit 2; }
expected_tag="v$version"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/geist-release-state.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM

latest=$(curl -fsS "$base/VERSION")
if [ "v$latest" != "$expected_tag" ]; then
    echo "check-published-release: latest is v$latest, source expects $expected_tag" >&2
    exit 1
fi

printf '%s\n' \
    SHA256SUMS \
    geist-linux-arm64 \
    geist-linux-x86_64 \
    libgeist-linux-arm64.tar.gz \
    libgeist-linux-x86_64.tar.gz \
    libgeist-macos-arm64.tar.gz | sort > "$tmpdir/expected-assets"
curl -fsS -o "$tmpdir/SHA256SUMS" "$base/SHA256SUMS"
{ echo SHA256SUMS; awk '{print $2}' "$tmpdir/SHA256SUMS"; } | sort > "$tmpdir/published-assets"
if ! diff -u "$tmpdir/expected-assets" "$tmpdir/published-assets"; then
    echo "check-published-release: public asset set is incomplete or unexpected" >&2
    exit 1
fi
for f in $(awk '{print $2}' "$tmpdir/SHA256SUMS"); do
    curl -fsS -o "$tmpdir/$f" "$base/$f"
done
if ! (cd "$tmpdir" && sha256sum -c --quiet SHA256SUMS 2>/dev/null || shasum -a 256 -c --quiet SHA256SUMS); then
    echo "check-published-release: a published file does not match SHA256SUMS" >&2
    exit 1
fi

git fetch -q --tags origin main
tag_sha=$(git rev-list -n 1 "$expected_tag")
if ! git merge-base --is-ancestor "$tag_sha" origin/main; then
    echo "check-published-release: $expected_tag ($tag_sha) is not an ancestor of origin/main" >&2
    exit 1
fi

echo "published release OK: $expected_tag ($tag_sha), complete and verified at $base, ancestor of main"
