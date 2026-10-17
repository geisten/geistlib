#!/bin/sh
# package-sdk.sh — lay out a libgeist SDK and prove a consumer can link it.
#
# Copies <lib_dir>/libgeist.a, include/*.h and LICENSE into dist/<name>/, then
# links and runs the three SDK smokes (embed, agent contract, runtime
# contract) against the packaged copy with the consumer's link flags. The
# release workflow runs it once per platform; it tars dist/<name> itself.
#
# The embed smoke links the whole archive (--whole-archive, ld64's
# -force_load): a static link only pulls the objects it references, so a plain
# smoke link can pass while an object the smoke never reaches still has
# unresolved dependencies (geistlib#144). Forcing every object in makes the
# link itself assert that the full archive resolves with the documented flags.
#
# Usage (from the repository root):
#   sh scripts/package-sdk.sh <name> <lib_dir> <consumer link flags...>
#   sh scripts/package-sdk.sh libgeist-linux-arm64 lib/linux/release -fopenmp -lm
set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 <name> <lib_dir> <consumer link flags...>" >&2
    exit 2
fi
name=$1
lib=$2
shift 2

d=dist/$name
mkdir -p "$d/include"
cp "$lib/libgeist.a" "$d/"
cp include/*.h "$d/include/"
cp LICENSE "$d/"

if [ "$(uname -s)" = Darwin ]; then
    whole_start="-Wl,-force_load,$d/libgeist.a"
    whole_end=
else
    whole_start="-Wl,--whole-archive $d/libgeist.a"
    whole_end=-Wl,--no-whole-archive
fi

# shellcheck disable=SC2086  # whole_* are deliberately split into words
cc -std=c23 -I "$d/include" examples/embed_smoke.c $whole_start $whole_end "$@" -o /tmp/sdk_smoke
/tmp/sdk_smoke
cc -std=c23 -I "$d/include" examples/agent_contract_smoke.c "$d/libgeist.a" "$@" -o /tmp/agent_smoke
/tmp/agent_smoke
cc -std=c23 -I "$d/include" examples/runtime_contract_smoke.c "$d/libgeist.a" "$@" -o /tmp/runtime_smoke
/tmp/runtime_smoke
