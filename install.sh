#!/bin/sh
# geist installer — downloads the single-file `geist-bitnet` (BitNet b1.58 2B-4T
# baked in) for your platform and drops it on your PATH. One file, no model file,
# no BLAS, no Python, no CUDA.
#
#   curl -fsSL https://raw.githubusercontent.com/geisten/geisten/main/install.sh | sh
#
# Env knobs: GEIST_BINDIR (install dir, default ~/.local/bin).
set -eu

REPO=geisten/geisten
BINDIR=${GEIST_BINDIR:-$HOME/.local/bin}

# Map uname -> the release artifact. Pi 5 / any ARM64 Linux share one binary;
# x86-64 Linux ships a musl-static AVX-512 build (runtime-dispatched, so it also
# runs on any x86-64-v3 CPU).
os=$(uname -s)
arch=$(uname -m)
case "$os/$arch" in
  Darwin/arm64)               plat=macos-arm64 ;;
  Linux/aarch64|Linux/arm64)  plat=linux-arm64 ;;
  Linux/x86_64|Linux/amd64)   plat=linux-x86_64 ;;
  *)
    echo "geist: no prebuilt binary for $os/$arch." >&2
    echo "       Prebuilt: macOS arm64, Linux arm64, Linux x86_64. Others build from source:" >&2
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

# Integrity: verify the download against the release SHA256SUMS before extracting.
# If the release ships SHA256SUMS, a missing entry or a mismatch ABORTS. Releases
# that predate checksums (SHA256SUMS absent) only warn — the file ships from the
# next release (geisten#87), after which verification is always enforced here.
sums_url="https://github.com/$REPO/releases/latest/download/SHA256SUMS"
if curl -fsSL -o "$tmp/SHA256SUMS" "$sums_url" 2>/dev/null; then
  if command -v sha256sum >/dev/null 2>&1; then sha='sha256sum'
  elif command -v shasum >/dev/null 2>&1; then sha='shasum -a 256'
  else echo "geist: need sha256sum or shasum to verify the download" >&2; exit 1
  fi
  # SHA256SUMS lines are "<hex>  <name>" (text) or "<hex>  *<name>" (binary mode).
  expected=$(awk -v a="$asset" '$2 == a || $2 == "*" a { print $1 }' "$tmp/SHA256SUMS" | head -1)
  [ -n "$expected" ] || { echo "geist: $asset not listed in SHA256SUMS — refusing to install" >&2; exit 1; }
  actual=$($sha "$tmp/$asset" | awk '{ print $1 }')
  if [ "$expected" != "$actual" ]; then
    echo "geist: SHA-256 mismatch for $asset — refusing to install" >&2
    echo "       expected $expected" >&2
    echo "       actual   $actual" >&2
    exit 1
  fi
  echo "Verified SHA-256."
else
  echo "geist: WARNING — this release has no SHA256SUMS; integrity NOT verified." >&2
  echo "       Checksums ship from the next release (geisten#87)." >&2
fi

tar -C "$tmp" -xzf "$tmp/$asset"

bin=$(find "$tmp" -type f -name geist-bitnet | head -1)
[ -n "$bin" ] || { echo "geist: geist-bitnet not found in $asset" >&2; exit 1; }
mkdir -p "$BINDIR"
cp "$bin" "$BINDIR/geist-bitnet"
chmod +x "$BINDIR/geist-bitnet"

echo "Installed: $BINDIR/geist-bitnet"
case ":$PATH:" in
  *":$BINDIR:"*)
    echo 'Run:  geist-bitnet "What is the capital of France?"' ;;
  *)
    echo "Add to PATH:  export PATH=\"$BINDIR:\$PATH\""
    echo "Then run:     geist-bitnet \"The capital of France is\"" ;;
esac
