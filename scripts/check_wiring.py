#!/usr/bin/env python3
"""check_wiring.py - family and backend wiring fail the build, not a load (#345).

No models, no GPU. Part of `make check-static`.

  cases    every tests/wiring/*.c compiles with -DWIRING_OK and fails
           without it, with the message its `// expect: <regex>` names;
           make refuses a family list that disagrees with the units; no
           family table is defined without TRANSFORMER_FAMILY
  matrix   each family alone (make FAMILIES=<one>) builds, and a program
           that pulls in every object of the library links: a reference
           from shared code to one family's table is an undefined symbol

  check_wiring.py            both
  check_wiring.py cases      only the cases
  check_wiring.py matrix     only the matrix

The matrix builds under build/wiring/ and lib/wiring/; objects are reused
between runs. MAKE names make; JOBS the parallelism.
"""
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, "tests", "wiring")
FAMILY_LIST = os.path.join(ROOT, "src", "archs", "transformer", "family_list.h")
MAKE = os.environ.get("MAKE", "make")
JOBS = os.environ.get("JOBS", str(os.cpu_count() or 1))


def make(*args, **kw):
    return subprocess.run(
        [MAKE, "--no-print-directory", *args], cwd=ROOT, capture_output=True, text=True, **kw
    )


def compile_flags():
    """The compiler and flags this host builds the family registry with."""
    r = make("-n", "-B", "lib")
    if r.returncode != 0:
        sys.exit(f"check_wiring: make -n lib failed\n{r.stderr}")
    for line in r.stdout.splitlines():
        if "src/archs/transformer/arch_family.c" in line and " -c " in line:
            args = shlex.split(line)
            i = args.index("-c")
            rest = args[:i] + args[i + 4 :]  # drop "-c <src> -o <obj>"
            return [a for a in rest if a not in ("-MMD", "-MP")]
    sys.exit("check_wiring: no compile command for arch_family.c in make -n lib")


def cases():
    flags = compile_flags()
    bad = 0
    n = 0
    for name in sorted(os.listdir(CASES)):
        if not name.endswith(".c"):
            continue
        n += 1
        path = os.path.join(CASES, name)
        with open(path) as f:
            m = re.search(r"// expect: (.+)", f.read())
        if m is None:
            print(f"cases: tests/wiring/{name}: no // expect: line")
            bad += 1
            continue
        want = re.compile(m[1].strip())
        cmd = [*flags, "-c", path, "-o", os.devnull]
        ok = subprocess.run([*cmd, "-DWIRING_OK"], cwd=ROOT, capture_output=True, text=True)
        broken = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if ok.returncode != 0:
            print(f"cases: tests/wiring/{name}: the control does not compile\n{ok.stderr}")
            bad += 1
        elif broken.returncode == 0:
            print(f"cases: tests/wiring/{name}: the mistake compiled")
            bad += 1
        elif not want.search(broken.stderr):
            print(f"cases: tests/wiring/{name}: failed, but not with /{m[1].strip()}/\n"
                  f"{broken.stderr}")
            bad += 1
    bad += family_list_cases()
    bad += bare_tables()
    print(f"cases: {n} compile cases and 2 family-list cases, {bad} wrong")
    return bad


def bare_tables():
    """A family table defined without TRANSFORMER_FAMILY escapes its checks."""
    tracked = subprocess.run(
        ["git", "ls-files", "src", "tests", "tools"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.split()
    bare = re.compile(r"struct\s+transformer_family\s+\w+\s*(\[[^]]*\]\s*)?=")
    bad = 0
    for path in tracked:
        if path.endswith((".c", ".h")):
            with open(os.path.join(ROOT, path), errors="replace") as f:
                for no, line in enumerate(f, 1):
                    if bare.search(line):
                        print(f"cases: {path}:{no}: a family table without TRANSFORMER_FAMILY")
                        bad += 1
    return bad


def family_list_cases():
    """make stops on a list without a unit's line, and on a line without a unit."""
    with open(FAMILY_LIST) as f:
        text = f.read()
    dropped = re.sub(r"\n *X\(olmoe,[^\n]*", "", text)
    extra = text.replace("    X(gemma4,", '    X(nosuch, NOSUCH, nosuch, "nosuch") \\\n    X(gemma4,')
    assert dropped != text and extra != text, "family_list.h layout changed; update the cases"
    bad = 0
    for what, body, want in (
        ("a unit without its line", dropped, "a unit not listed 'olmoe'"),
        ("a line without its unit", extra, "listed without a unit 'nosuch'"),
    ):
        with tempfile.NamedTemporaryFile("w", suffix=".h", delete=False) as f:
            f.write(body)
        try:
            r = make("-n", "lib", f"FAMILY_LIST={f.name}")
        finally:
            os.unlink(f.name)
        if r.returncode == 0 or want not in r.stderr:
            print(f"cases: family list with {what}: make did not stop with \"{want}\"\n"
                  f"{r.stderr}")
            bad += 1
    return bad


def all_families():
    r = make("-s", "-p", "-n", "lib")
    m = re.search(r"^ALL_FAMILIES := (.*)$", r.stdout, re.M)
    if m is None:
        sys.exit(f"check_wiring: ALL_FAMILIES not found\n{r.stderr}")
    return m[1].split()


def link_all(variables, tmp):
    """Link a program that pulls in every object of the library `make lib`
    built with these variables: the shared-library link line of `make -n
    shared` (force_load / whole-archive), turned into an executable. An
    undefined symbol anywhere in the library fails it."""
    r = make("-n", "shared", *variables)
    line = next((l for l in r.stdout.splitlines() if "force_load" in l or "whole-archive" in l), None)
    if r.returncode != 0 or line is None:
        return f"no link line from make -n shared\n{r.stdout}{r.stderr}"
    main_c = os.path.join(tmp, "main.c")
    with open(main_c, "w") as f:
        f.write("int main(void) { return 0; }\n")
    args = shlex.split(line)
    args = [main_c if a in ("-dynamiclib", "-shared") else a for a in args]
    args[args.index("-o") + 1] = os.path.join(tmp, "a.out")
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return None if r.returncode == 0 else r.stderr[-3000:]


def build_and_link(label, variables, tmp):
    t0 = time.monotonic()
    r = make("-s", f"-j{JOBS}", "lib", *variables)
    err = f"{r.stdout[-3000:]}{r.stderr[-3000:]}" if r.returncode != 0 else link_all(variables, tmp)
    dt = time.monotonic() - t0
    if err:
        print(f"matrix: {label} does not build and link alone ({dt:.0f} s)\n{err}")
        return 1
    print(f"matrix: {label} ok ({dt:.0f} s)")
    return 0


def matrix():
    bad = 0
    out = ["BUILD_DIR=build/wiring/families", "LIB_DIR=lib/wiring/families"]
    with tempfile.TemporaryDirectory() as tmp:
        for fam in all_families():
            bad += build_and_link(f"FAMILIES={fam}", [f"FAMILIES={fam}", *out], tmp)
    return bad


def main():
    what = sys.argv[1:] or ["cases", "matrix"]
    bad = 0
    for w in what:
        if w == "cases":
            bad += cases()
        elif w == "matrix":
            bad += matrix()
        else:
            sys.exit(f"check_wiring: unknown step '{w}' (cases, matrix)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
