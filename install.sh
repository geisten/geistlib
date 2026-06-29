#!/bin/sh
# geist installer — downloads the single-file `geist-bitnet` (BitNet b1.58 2B-4T
# baked in) for your platform and drops it on your PATH. One file, no model file,
# no BLAS, no Python, no CUDA.
#
#   curl -fsSL https://raw.githubusercontent.com/geisten/geistlib/main/install.sh | sh
#
# Env knobs: GEIST_BINDIR (install dir, default ~/.local/bin).
set -eu

REPO=geisten/geistlib
BINDIR=${GEIST_BINDIR:-$HOME/.local/bin}

# Map uname -> the release artifact. Pi 5 / any ARM64 Linux share one binary.
os=$(uname -s)
arch=$(uname -m)
case "$os/$arch" in
  Darwin/arm64)               plat=macos-arm64 ;;
  Linux/aarch64|Linux/arm64)  plat=linux-arm64 ;;
  *)
    echo "geist: no prebuilt binary for $os/$arch yet." >&2
    echo "       Prebuilt binaries are ARM64; x86-64 Linux (AVX-512) builds from source:" >&2
    echo "       https://github.com/$REPO#-getting-started" >&2
    exit 1 ;;
esac

asset="geist-bitnet-$plat.tar.gz"
url="https://github.com/$REPO/releases/latest/download/$asset"

command -v curl >/dev/null 2>&1 || { echo "geist: curl is required" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $asset (~1.1 GB — the model is baked in) …"
curl -fL --progress-bar -o "$tmp/$asset" "$url"
tar -C "$tmp" -xzf "$tmp/$asset"

bin=$(find "$tmp" -type f -name geist-bitnet | head -1)
[ -n "$bin" ] || { echo "geist: geist-bitnet not found in $asset" >&2; exit 1; }
mkdir -p "$BINDIR"
cp "$bin" "$BINDIR/geist-bitnet"
chmod +x "$BINDIR/geist-bitnet"

echo "Installed: $BINDIR/geist-bitnet"
case ":$PATH:" in
  *":$BINDIR:"*)
    echo 'Run:  geist-bitnet "The capital of France is"' ;;
  *)
    echo "Add to PATH:  export PATH=\"$BINDIR:\$PATH\""
    echo "Then run:     geist-bitnet \"The capital of France is\"" ;;
esac
