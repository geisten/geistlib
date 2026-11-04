#!/bin/sh
# toolchain_probe.sh — which C23 features this toolchain really has (#341).
#
# Compiles, links and runs one small program per feature with the project's
# warning flags (-std=c23 -Wall -Wextra -Wpedantic -Werror), so "accepted"
# means accepted under the flags the tree builds with, and "works" means the
# program got the right answer:
#
#   stdbit          <stdbit.h>: stdc_count_ones & co., type-generic and _ui
#   counted_by      __attribute__((counted_by(n))) on a flexible array member,
#                   and whether __builtin_dynamic_object_size then sees it
#   fixed_enum_cxx  enum e : int32_t / : uint8_t in a header, compiled as C23
#                   and as C++17 -pedantic-errors (check-headers' flags)
#   gnu_cleanup     [[gnu::cleanup(fn)]] runs on every way out of the scope
#   typeof          typeof and typeof_unqual
#   stdckdint       <stdckdint.h> ckd_mul (src/base/checked.h falls back
#                   without it)
#   embed           #embed (the shader headers stay generated without it)
#
# Prints one line per feature: `probe <feature> <yes|no> <detail>`.
#
#   sh tools/toolchain_probe.sh [--id NAME] [--expect FILE]
#
# --expect compares the yes/no column with FILE's `<id> <feature> <yes|no>`
# lines for this --id and exits 1 on any difference: CI runs it on each
# toolchain of docs/TOOLCHAIN_PROBE.md, so a runner image that gains or
# loses a feature fails until the document and FILE say so. The comparison
# holds for the compiler major version FILE records (`<id> cc_major <n>`,
# from `$CC -dumpversion`); another compiler, or an id FILE does not list, is
# reported and not compared (GitHub's runners carry an older Apple clang).
# NAME is lowercased and every run of other characters becomes one '-', so a
# CI matrix name can be passed as is.
#
# CC (default cc) and CXX (default c++) pick the compilers.
set -eu

CC=${CC:-cc}
CXX=${CXX:-c++}
id=""
expect=""
while [ $# -gt 0 ]; do
    case "$1" in
        --id) id=$(printf '%s' "$2" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9][^a-z0-9]*/-/g; s/^-//; s/-$//'); shift 2 ;;
        --expect) expect=$2; shift 2 ;;
        *) echo "usage: $0 [--id NAME] [--expect FILE]" >&2; exit 2 ;;
    esac
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
CFLAGS="-std=c23 -O2 -Wall -Wextra -Wpedantic -Werror"
CXXFLAGS="-std=c++17 -Wall -Wextra -pedantic-errors -Werror"
results=""

say() {
    printf 'probe %-15s %-3s %s\n' "$1" "$2" "$3"
    results="$results$1 $2
"
}

# $1 = name of a .c file in $tmp; compile, link, run. Sets $why on failure.
run_c() {
    why=""
    if ! $CC $CFLAGS -o "$tmp/$1.bin" "$tmp/$1" >"$tmp/$1.log" 2>&1; then
        why="compile/link: $(grep -m1 -E 'error|undefined' "$tmp/$1.log" | sed 's/^[^:]*:[0-9:]* *//' | cut -c1-90)"
        return 1
    fi
    rc=0
    "$tmp/$1.bin" || rc=$?
    if [ "$rc" -ne 0 ]; then
        why="runs, wrong result (exit $rc)"
        return 1
    fi
}

echo "toolchain ${id:-unnamed}"
echo "cc  $($CC --version 2>&1 | head -n1)"
echo "cxx $($CXX --version 2>/dev/null | head -n1)"
if ldd --version >/dev/null 2>&1; then
    echo "libc $(ldd --version 2>&1 | head -n1)"
elif [ "$(uname -s)" = Darwin ]; then
    echo "libc macOS $(sw_vers -productVersion), SDK $(xcrun --show-sdk-version 2>/dev/null || echo '?')"
else
    echo "libc $(ldd 2>&1 | head -n2 | tr "\n" " ")"   # musl: name, then "Version x"
fi

# ---- <stdbit.h> -----------------------------------------------------------
cat >"$tmp/stdbit.c" <<'EOF'
#include <stdbit.h>
#include <stdint.h>
int main(void) {
    unsigned x = 0xF0u;
    if (stdc_count_ones_ui(x) != 4) return 1;
    if (stdc_trailing_zeros_ui(x) != 4) return 2;
    if (stdc_leading_zeros_ull(1ull) != 63) return 3;
    if (stdc_bit_width_ui(x) != 8) return 4;
    if (stdc_bit_ceil_ui(5u) != 8) return 5;
    if (stdc_count_ones((uint64_t) 0xFF) != 8) return 6; /* type-generic */
    return 0;
}
EOF
if printf '#include <stdbit.h>\nint x;\n' | $CC -std=c23 -x c -fsyntax-only - >/dev/null 2>&1; then
    if run_c stdbit.c; then say stdbit yes "header, _ui/_ull and type-generic forms"
    else say stdbit no "header present; $why"; fi
else
    say stdbit no "no <stdbit.h>"
fi

# ---- counted_by -----------------------------------------------------------
cat >"$tmp/counted_by.c" <<'EOF'
#include <stddef.h>
#include <stdlib.h>
struct buf { size_t n; int v[] __attribute__((counted_by(n))); };
int main(void) {
    struct buf *b = malloc(sizeof *b + 4 * sizeof(int));
    if (!b) return 9;
    b->n = 4;
    size_t sz = __builtin_dynamic_object_size(b->v, 1);
    free(b);
    return sz == 4 * sizeof(int) ? 0 : 1;
}
EOF
if run_c counted_by.c; then
    say counted_by yes "accepted; __builtin_dynamic_object_size sees n"
elif [ -x "$tmp/counted_by.c.bin" ]; then
    say counted_by no "accepted, but no effect on __builtin_dynamic_object_size"
else
    say counted_by no "$why"
fi

# ---- fixed underlying enum in a header, C23 and C++17 ----------------------
cat >"$tmp/fixed_enum.h" <<'EOF'
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum probe_status : int32_t { PROBE_OK = 0, PROBE_E_ARG = -1 };
enum probe_small : uint8_t { PROBE_A, PROBE_B };
typedef enum probe_status probe_status_t;
probe_status_t probe_fn(enum probe_small s);
static_assert(sizeof(enum probe_status) == 4, "int32_t width");
static_assert(sizeof(enum probe_small) == 1, "uint8_t width");
#ifdef __cplusplus
}
#endif
EOF
printf '#include "fixed_enum.h"\nint main(void) { return PROBE_OK; }\n' >"$tmp/fixed_enum.c"
printf '#include "fixed_enum.h"\nint main() { return PROBE_OK; }\n' >"$tmp/fixed_enum.cc"
if ! run_c fixed_enum.c; then
    say fixed_enum_cxx no "C23: $why"
elif ! $CXX --version >/dev/null 2>&1; then
    say fixed_enum_cxx no "C23 yes; no C++ compiler to check C++17"
elif $CXX $CXXFLAGS -o "$tmp/fixed_enum_cc.bin" "$tmp/fixed_enum.cc" >"$tmp/fe.log" 2>&1; then
    say fixed_enum_cxx yes "C23 and C++17, sizes 4 and 1"
else
    say fixed_enum_cxx no "C++17: $(grep -m1 error "$tmp/fe.log" | cut -c1-90)"
fi

# ---- [[gnu::cleanup]] --------------------------------------------------------
cat >"$tmp/cleanup.c" <<'EOF'
static int ran;
static void done(int *p) { ran += *p; }
static int f(int early) {
    [[gnu::cleanup(done)]] int x = 1;
    if (early)
        return 1;
    x = 2;
    return 0;
}
int main(void) {
    int r = f(1) + f(0);
    return (r == 1 && ran == 3) ? 0 : 1;
}
EOF
if run_c cleanup.c; then say gnu_cleanup yes "runs on early and normal return"
else say gnu_cleanup no "$why"; fi

# ---- typeof / typeof_unqual -------------------------------------------------
cat >"$tmp/typeof.c" <<'EOF'
int main(void) {
    const int a = 1;
    typeof(a) b = 2;            /* const int */
    typeof_unqual(a) c = 3;     /* int */
    c = 4;
    typeof(int[4]) arr = {0};
    return (a + b + c + (int) sizeof arr == 1 + 2 + 4 + 4 * (int) sizeof(int)) ? 0 : 1;
}
EOF
if run_c typeof.c; then say typeof yes "typeof and typeof_unqual"
else say typeof no "$why"; fi

# ---- <stdckdint.h> -----------------------------------------------------------
cat >"$tmp/stdckdint.c" <<'EOF'
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
int main(void) {
    size_t r;
    if (!ckd_mul(&r, SIZE_MAX / 2, (size_t) 3)) return 1;
    if (ckd_mul(&r, (size_t) 6, (size_t) 7) || r != 42) return 2;
    return 0;
}
EOF
if run_c stdckdint.c; then say stdckdint yes "ckd_mul detects overflow"
else say stdckdint no "$why"; fi

# ---- #embed ------------------------------------------------------------------
printf 'AB' >"$tmp/blob.bin"
cat >"$tmp/embed.c" <<'EOF'
static const unsigned char blob[] = {
#embed "blob.bin"
};
int main(void) { return (sizeof blob == 2 && blob[0] == 'A') ? 0 : 1; }
EOF
if run_c embed.c; then say embed yes "#embed of a 2-byte file"
else say embed no "$why"; fi

# ---- compare --------------------------------------------------------------
[ -n "$expect" ] || exit 0
if [ -z "$id" ] || ! grep -q "^$id " "$expect"; then
    echo "toolchain_probe: '${id:-}' is not listed in $expect; not compared"
    exit 0
fi
major=$($CC -dumpversion 2>/dev/null | cut -d. -f1)
want_major=$(awk -v i="$id" '$1 == i && $2 == "cc_major" { print $3 }' "$expect")
if [ "$major" != "$want_major" ]; then
    echo "toolchain_probe: $id: compiler major $major, $expect has ${want_major:-none}; not compared"
    exit 0
fi
bad=0
while read -r feat got; do
    want=$(awk -v i="$id" -v f="$feat" '$1 == i && $2 == f { print $3 }' "$expect")
    if [ "$want" != "$got" ]; then
        echo "toolchain_probe: $id $feat: expected '${want:-unlisted}', got '$got'." \
             "Update $expect and docs/TOOLCHAIN_PROBE.md." >&2
        bad=1
    fi
done <<EOF2
$results
EOF2
[ "$bad" -eq 0 ] && echo "toolchain_probe: $id matches $expect"
exit $bad
