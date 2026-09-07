#!/bin/sh
# fetch-dep.sh NAME REPO SHA [PATCH]
# Clone REPO into build/deps/NAME, check SHA out detached, apply PATCH.
# The state is rebuilt on every run; local edits under build/deps are lost.
# Only full 40-hex SHAs pin: names move, a SHA does not.
# Shared verbatim by geist repositories; the canonical copy is geistlib's.
set -eu
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo 'usage: fetch-dep.sh NAME REPO SHA [PATCH]' >&2
    exit 2
fi
name=$1 repo=$2 sha=$3 patch=${4:-}
if ! printf '%s\n' "$sha" | grep -Eq '^[0-9a-f]{40}$'; then
    echo "fetch-dep.sh: SHA must be 40 hex characters, got '$sha'" >&2
    exit 2
fi
dir=build/deps/$name
if [ -n "$patch" ]; then
    patch=$(cd "$(dirname "$patch")" && pwd)/$(basename "$patch")
fi
mkdir -p build/deps
if [ -d "$dir/.git" ]; then
    git -C "$dir" remote set-url origin "$repo"
else
    git clone -q "$repo" "$dir"
fi
git -C "$dir" cat-file -e "$sha^{commit}" 2>/dev/null || git -C "$dir" fetch -q origin "$sha"
git -C "$dir" checkout -q -f --detach "$sha"
git -C "$dir" clean -qfdx
test "$(git -C "$dir" rev-parse HEAD)" = "$sha"
stamp="$sha unpatched"
if [ -n "$patch" ]; then
    git -C "$dir" apply --check -p1 "$patch"
    git -C "$dir" apply -p1 "$patch"
    stamp="$sha $(cksum < "$patch" | cut -d' ' -f1)"
fi
printf '%s\n' "$stamp" > "$dir/.fetch-dep"
printf '%s %s\n' "$name" "$stamp"
