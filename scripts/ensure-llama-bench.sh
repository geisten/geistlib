#!/bin/sh
# ensure-llama-bench.sh VARIANT [REPO] [PIN] — print the path of a pinned
# llama-bench build for the cross-engine benchmarks, building it if needed.
#
# VARIANT: cpu | vulkan | metal. The cpu build must not link a GPU backend,
# or bench_cross_engine.py's CPU-only validation rejects it. Checkouts live
# outside the workspace (one per repository, so a fork build never shadows
# the upstream pin's) and are rebuilt only when the pin moves. Build output
# goes to stderr; stdout is the binary path alone.
set -e
VARIANT=$1
REPO=${2:-ggml-org/llama.cpp}
PIN=${3:-d81235049384534c167caea52b85a694f6103d14}

if [ "$REPO" = "ggml-org/llama.cpp" ]; then
    DIR="$HOME/llama.cpp-cross-engine"
else
    DIR="$HOME/llama.cpp-cross-engine-$(echo "$REPO" | tr '/' '_')"
fi
BUILD="$DIR/build-$VARIANT"

{
# Each build directory records the commit it was built from: the variants
# share one checkout, so a checkout already at the pin says nothing about a
# build-vulkan last built at an older pin (bench-matrix run 850: every
# upstream Vulkan cell refused its stale llama-bench).
WANT=$(git -C "$DIR" rev-parse "$PIN^{commit}" 2>/dev/null || true)
if [ ! -x "$BUILD/bin/llama-bench" ] || [ -z "$WANT" ] ||
   [ "$(cat "$BUILD/.geist-built-from" 2>/dev/null)" != "$WANT" ] ||
   [ "$(git -C "$DIR" rev-parse HEAD)" != "$WANT" ]; then
    [ -d "$DIR/.git" ] || git clone "https://github.com/$REPO" "$DIR"
    # A full SHA fetches directly; a short one or a branch needs the history.
    git -C "$DIR" fetch origin "$PIN" 2>/dev/null || git -C "$DIR" fetch origin
    git -C "$DIR" checkout -q "$PIN"
    rm -f "$BUILD/.geist-built-from"
    case $VARIANT in
    vulkan)
        # ggml-vulkan #includes spirv/unified1/spirv.hpp directly, so the
        # headers must be on the compiler's include path.
        SPIRV_INC="$HOME/.local/include"
        if [ ! -f "$SPIRV_INC/spirv/unified1/spirv.hpp" ]; then
            [ -d "$DIR/spirv-headers/.git" ] || git clone --depth 1 \
                https://github.com/KhronosGroup/SPIRV-Headers "$DIR/spirv-headers"
            SPIRV_INC="$DIR/spirv-headers/include"
        fi
        set -- -DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH="$HOME/.local" -DCMAKE_CXX_FLAGS="-I$SPIRV_INC" ;;
    metal)
        set -- -DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple ;;
    cpu)
        if [ "$(uname)" = Darwin ]; then
            set -- -DGGML_NATIVE=ON -DGGML_METAL=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple
        else
            set -- -DGGML_NATIVE=ON -DGGML_VULKAN=OFF
        fi ;;
    *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
    esac
    cmake -S "$DIR" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF "$@"
    # Bounded by RAM, not only cores: an unbounded -j took the 4 GB Pi
    # down (llama-model.cpp and friends need ~1.5 GB per compiler).
    CORES=$(getconf _NPROCESSORS_ONLN)
    MEM=$(sysctl -n hw.memsize 2>/dev/null || echo $(( $(getconf _PHYS_PAGES) * $(getconf PAGESIZE) )))
    MEM_JOBS=$(( MEM / 1536 / 1048576 ))
    JOBS=$(( MEM_JOBS < CORES ? MEM_JOBS : CORES ))
    cmake --build "$BUILD" --target llama-bench -j "$(( JOBS > 0 ? JOBS : 1 ))"
    git -C "$DIR" rev-parse HEAD > "$BUILD/.geist-built-from"
fi
echo "llama.cpp $REPO @ $(git -C "$DIR" rev-parse HEAD) ($VARIANT)"
} >&2
echo "$BUILD/bin/llama-bench"
