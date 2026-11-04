#!/usr/bin/env python3
"""One spelling per concept (#348): the C23 form, once the tree has moved to it.

Fails on `_Static_assert`, on `__builtin_memcpy` and its kin (the standard
name is the same builtin), and on an `__attribute__` that has an equivalent
`[[...]]` or `alignas`, in code; a comment may name them. KEEP lists what has
not moved yet, each with its reason.
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP = {
    "always_inline": "next batch of #348",
    "constructor": "next batch of #348",
    "target": "next batch of #348: built through macros in the x86 kernels",
}
ATTR = re.compile(r"__attribute__\s*\(\(\s*(\w+)")
STATIC = re.compile(r"\b_Static_assert\b")
# A builtin with a standard spelling the tree uses everywhere else.
BUILTIN = re.compile(r"\b__builtin_(memcpy|memset|memmove|memcmp|strlen)\b")
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def problems(text):
    """(line, what) of each spelling that has to go, comments aside."""
    code = COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    found = []
    for m in STATIC.finditer(code):
        found.append((code.count("\n", 0, m.start()) + 1, "_Static_assert: use static_assert"))
    for m in ATTR.finditer(code):
        if m.group(1) not in KEEP:
            line = code.count("\n", 0, m.start()) + 1
            found.append((line, f"__attribute__(({m.group(1)})): use the [[...]] or alignas form"))
    for m in BUILTIN.finditer(code):
        found.append((code.count("\n", 0, m.start()) + 1, f"__builtin_{m.group(1)}: use {m.group(1)}"))
    return found


class Spelling(unittest.TestCase):
    def test_tree(self):
        names = subprocess.run(
            ["git", "ls-files", "src", "include", "tools", "tests", "examples"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        bad = []
        for name in names:
            if name.endswith((".c", ".h", ".m")):
                text = (ROOT / name).read_text(errors="replace")
                bad += [f"{name}:{line}: {what}" for line, what in problems(text)]
        self.assertEqual(bad, [])

    def test_seeded(self):
        self.assertEqual(len(problems("_Static_assert(1, \"x\");\n")), 1)
        self.assertEqual(len(problems("int a __attribute__((aligned(8)));\n")), 1)
        self.assertEqual(problems("/* _Static_assert __attribute__((packed)) */\n"), [])
        self.assertEqual(len(problems("__builtin_memcpy(&f, &b, 4);\n")), 1)
        self.assertEqual(problems("__builtin_prefetch(p);\n"), [])
        self.assertEqual(problems("__attribute__((target(\"avx2\"))) void f(void);\n"), [])


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=1))
