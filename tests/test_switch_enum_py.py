#!/usr/bin/env python3
"""A new enum value fails the build at every switch over that enum (#346).

Copies include/, adds a probe value to a public enum, and compiles the whole
library with its own flags under -fsyntax-only. Each switch over that enum
(here: each `case` of the enum's last value) has to be reported, in the .c
files and in the headers they include. Sources this target does not build
(cpu_x86 and Vulkan on a Mac) are not in the compile commands. A make
variable given to `make test-py` (CC, TARGET) reaches `make -n` through
MAKEFLAGS.
"""
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# enum, header, its last value (a switch that names it is exhaustive)
ENUMS = [
    ("geist_dtype", "geist_types.h", "GEIST_DTYPE_COUNT"),
    ("geist_fused_op", "geist_backend.h", "GEIST_FUSED_MOE_DEVICE"),
]
SOURCE = re.compile(r"\s(src/\S+\.(?:c|m))(?:\s|$)")


def compile_commands():
    """{source: argv} of the library build, without running it."""
    out = subprocess.run(
        ["make", "-n", "-B", "lib"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    commands = {}
    for line in out.splitlines():
        if " -c " not in line or " -o " not in line:
            continue
        found = SOURCE.search(line)
        if found:
            argv = [a for a in shlex.split(line) if a != "-Werror"]
            commands[found.group(1)] = argv
    return commands


def syntax_only(argv, source, include):
    args = []
    skip = False
    for a in argv:
        if skip:
            skip = False
        elif a == "-o":
            skip = True
        elif a == "-c" or a == source:
            continue
        elif a in ("-MMD", "-MP"):
            continue
        elif a == "-MF" or a == "-MT":
            skip = True
        elif a == "-Iinclude":
            args.append(f"-I{include}")
        else:
            args.append(a)
    args += ["-fsyntax-only", source]
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def reported(stderr, value):
    """(file, line) of each switch the compiler says misses `value`."""
    sites = set()
    for line in stderr.splitlines():
        m = re.match(r"(\S+?):(\d+):\d+: (?:warning|error): .*\b" + value + r"\b", line)
        if m and "switch" in line:
            sites.add((m.group(1), int(m.group(2))))
    return sites


class SwitchEnum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.commands = compile_commands()
        source, argv = next(iter(cls.commands.items()))
        probe = syntax_only(argv, source, ROOT / "include")
        if "-std=c23" in probe.stderr:
            raise unittest.SkipTest(f"{argv[0]} has no C23: make test-py CC=<compiler>")

    def probe(self, enum, header, last):
        value = "SWITCH_ENUM_PROBE"
        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "include"
            shutil.copytree(ROOT / "include", include)
            path = include / header
            text = path.read_text()
            at = text.index("\n};", text.index(f"enum {enum} : uint32_t {{"))
            path.write_text(text[:at] + f"\n    {value}," + text[at:])
            with ThreadPoolExecutor(max_workers=8) as pool:
                runs = pool.map(
                    lambda item: syntax_only(item[1], item[0], include), self.commands.items()
                )
                sites = set().union(*(reported(r.stderr, value) for r in runs))
        built = set(self.commands) | {f for f, _ in sites}
        for name in built:
            need = (ROOT / name).read_text(errors="replace").count(f"case {last}:")
            got = sum(1 for f, _ in sites if f == name)
            self.assertEqual(got, need, f"{name}: {need} switches, {got} reported")
        self.assertTrue(sites, f"{value} went unnoticed")

    def test_dtype(self):
        self.probe(*ENUMS[0])

    def test_fused_op(self):
        self.probe(*ENUMS[1])

    def test_the_tree_compiles(self):
        for source, argv in self.commands.items():
            r = syntax_only(argv, source, ROOT / "include")
            self.assertEqual(r.returncode, 0, r.stderr[:2000])
            self.assertNotIn("switch", r.stderr)
            break


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=1))
