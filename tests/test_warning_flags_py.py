#!/usr/bin/env python3
"""Each warning #343 turned on fails the build (#343).

Takes the flags the library is built with (from `make -n`), and compiles two
snippets per warning: one the warning has to reject under -Werror, and the
same code written the accepted way, which has to compile. A warning that is
dropped from mk/common.mk, or that a compiler does not know, fails here. A
make variable given to `make test-py` (CC, TARGET) reaches `make -n` through
MAKEFLAGS.
"""
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = re.compile(r"\s(src/\S+\.c)(?:\s|$)")

# warning: (rejected, accepted)
CASES = {
    "-Wimplicit-fallthrough": (
        """
int f(int x);
int f(int x) {
    int r = 0;
    switch (x) {
    case 0:
        r = 1;
    case 1:
        r += 2;
        break;
    default:
        break;
    }
    return r;
}
""",
        """
int f(int x);
int f(int x) {
    int r = 0;
    switch (x) {
    case 0:
        r = 1;
        [[fallthrough]];
    case 1:
        r += 2;
        break;
    default:
        break;
    }
    return r;
}
""",
    ),
    "-Wmissing-prototypes": (
        "int twice(int x) { return 2 * x; }\n",
        "static int half(int x) { return x / 2; }\n"
        "int twice(int x);\n"
        "int twice(int x) { return 4 * half(x); }\n",
    ),
    "-Wformat=2": (
        """
#include <stdarg.h>
#include <stdio.h>
int say(char *buf, size_t n, const char *fmt, int x);
int say(char *buf, size_t n, const char *fmt, int x) {
    return snprintf(buf, n, fmt, x);
}
""",
        """
#include <stdarg.h>
#include <stdio.h>
[[gnu::format(printf, 3, 4)]] int say(char *buf, size_t n, const char *fmt, ...);
int say(char *buf, size_t n, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(buf, n, fmt, ap);
    va_end(ap);
    return r;
}
""",
    ),
    # gcc follows the null through its optimizer, clang reports only a
    # dereference of a literal null; the probe is one both reject.
    "-Wnull-dereference": (
        """
int load(void);
int load(void) {
    return *(int *) 0;
}
""",
        """
int load(const int *p);
int load(const int *p) {
    return p != nullptr ? *p : 0;
}
""",
    ),
}


# Directories that build with -Wconversion -Wsign-conversion (#343), and the
# probes: (rejected, accepted).
CONVERSION_DIRS = (
    "src/base/",
    "src/io/",
    "src/engine/",
    "src/archs/vision_siglip/",
    "src/backends/cpu_scalar/",
)
CONVERSION_CASES = {
    "-Wsign-conversion": (
        "unsigned f(int x);\nunsigned f(int x) { return x; }\n",
        "unsigned f(unsigned x);\nunsigned f(unsigned x) { return x; }\n",
    ),
    "-Wconversion": (
        "short g(long x);\nshort g(long x) { return x; }\n",
        "long g(short x);\nlong g(short x) { return x; }\n",
    ),
}


def library_flags(prefix="src/"):
    """The compiler and flags of one library object under prefix, with -Werror."""
    out = subprocess.run(
        ["make", "-n", "-B", "lib"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        src = SOURCE.search(line)
        if " -c " in line and " -o " in line and src and src.group(1).startswith(prefix):
            argv = shlex.split(line)
            keep = []
            skip = False
            for a in argv:
                if skip:
                    skip = False
                elif a in ("-o", "-MF", "-MT"):
                    skip = True
                elif a in ("-c", "-MMD", "-MP") or SOURCE.fullmatch(" " + a):
                    continue
                else:
                    keep.append(a)
            return keep
    raise AssertionError(f"make -n lib printed no compile command under {prefix}")


def compiles(flags, code):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.c"
        src.write_text(code)
        # A real compile, not -fsyntax-only: gcc runs -Wimplicit-fallthrough
        # in the middle end, which -fsyntax-only never reaches.
        return subprocess.run(
            flags + ["-c", "-o", os.devnull, str(src)], cwd=ROOT, capture_output=True, text=True
        )


class WarningFlags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flags = library_flags()
        probe = compiles(cls.flags, "int x;\n")
        if probe.returncode != 0 and "-std=c23" in probe.stderr:
            raise unittest.SkipTest(f"{cls.flags[0]} has no C23: make test-py CC=<compiler>")
        assert probe.returncode == 0, probe.stderr
        assert "-Werror" in cls.flags, cls.flags

    def test_each_warning(self):
        for warning, (rejected, accepted) in CASES.items():
            with self.subTest(warning=warning):
                self.assertIn(warning, self.flags)
                bad = compiles(self.flags, rejected)
                self.assertNotEqual(bad.returncode, 0, f"{warning} let the probe through")
                # clang names -Wfoo, gcc under -Werror -Werror=foo
                self.assertIn(warning.split("=")[0].removeprefix("-W"), bad.stderr)
                good = compiles(self.flags, accepted)
                self.assertEqual(good.returncode, 0, good.stderr)

    def test_conversion_per_directory(self):
        for prefix in CONVERSION_DIRS:
            flags = library_flags(prefix)
            for warning, (rejected, accepted) in CONVERSION_CASES.items():
                with self.subTest(dir=prefix, warning=warning):
                    self.assertIn(warning, flags)
                    bad = compiles(flags, rejected)
                    self.assertNotEqual(bad.returncode, 0, f"{warning} let the probe through")
                    self.assertIn(warning.removeprefix("-W"), bad.stderr)
                    good = compiles(flags, accepted)
                    self.assertEqual(good.returncode, 0, good.stderr)


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=1))
