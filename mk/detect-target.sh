#!/bin/sh
# Detect the build target based on the host system.
#
# POSIX sh on purpose, not bash: this runs before anything else in the build,
# and minimal containers (Alpine, distroless) ship no bash. With a bash
# shebang, detection failed silently there and make died on `mk/target-.mk`.
# Output: mac | mac-omp | pi5 | linux | unknown
# Override at make-time: make TARGET=mac
#
# On Darwin, mac-omp is preferred when Homebrew libomp is installed —
# without it geist's per-matmul `#pragma omp parallel for` directives
# are silently ignored, capping multi-thread perf at ~1/4 of llama.cpp.
# The plain `mac` target stays available as a fallback when libomp is
# missing or for serial debugging.

case "$(uname -s)" in
    Darwin)
        libomp_prefix="${LIBOMP_PREFIX:-/opt/homebrew/opt/libomp}"
        if [ -f "$libomp_prefix/lib/libomp.dylib" ]; then
            echo "mac-omp"
        else
            echo "mac"
        fi
        ;;
    Linux)
        case "$(uname -m)" in
            aarch64|arm64) echo "pi5" ;;
            *)             echo "linux" ;;
        esac
        ;;
    *)
        echo "unknown"
        ;;
esac
