#!/usr/bin/env python3
"""check_static.py - clang-tidy over the library, with a findings ratchet (#342).

No models, no GPU: the compile commands come from `make -n`, clang-tidy reads
the sources. A finding is (file, check, message) with a count; line numbers
stay out, so moving code does not disturb the baseline.

  check_static.py           fail on a finding the baseline lacks, and on a
                            baseline entry that is gone: it only shrinks
  check_static.py --update  rewrite the baseline
  check_static.py --selftest
                            tests/static/ must report exactly its seeded
                            defects: a missed one fails, and so does an
                            unmarked one

A NOLINT needs its reason on the same line: NOLINT(check): why. So does an
analyzer finding in the baseline, in static-reasons.txt.

CLANG_TIDY names the binary; JOBS the parallelism.
"""
import argparse
import collections
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "scripts", "static-baseline.txt")
REASONS = os.path.join(ROOT, "scripts", "static-reasons.txt")
FINDING = re.compile(r"^(.+?):(\d+):(\d+): (warning|error): (.*) \[([^\]]+)\]$")
NOLINT = re.compile(r"NOLINT(?:NEXTLINE|BEGIN|END)?(\([^)]*\))?(.*)")
REPORTED = ("src/", "include/", "tests/static/")


def compile_commands():
    """Each library source with the flags this host builds it with."""
    out = subprocess.run(
        ["make", "-n", "-B", "lib"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    cmds = []
    for line in out.splitlines():
        if " -c " not in line or ".c -o " not in line:
            continue
        args = shlex.split(line)
        src = args[args.index("-c") + 1]
        if src.startswith("src/"):
            cmds.append({"directory": ROOT, "arguments": args, "file": src})
    return cmds


def extra_args():
    if sys.platform != "darwin":
        return []
    sdk = subprocess.run(
        ["xcrun", "--show-sdk-path"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return ["--extra-arg=-isysroot" + sdk]


def tidy(tool, db, src):
    r = subprocess.run(
        [tool, "-p", db, "--quiet", *extra_args(), src],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode not in (0, 1):
        raise RuntimeError(f"{tool} on {src}: exit {r.returncode}\n{r.stderr[-2000:]}")
    return r.stdout


def scan(tool, cmds):
    """{(file, line, column, check, message)}; a header's finding counts once."""
    with tempfile.TemporaryDirectory() as db:
        with open(os.path.join(db, "compile_commands.json"), "w") as f:
            json.dump(cmds, f)
        jobs = int(os.environ.get("JOBS", os.cpu_count() or 1))
        with ThreadPoolExecutor(jobs) as pool:
            outputs = list(pool.map(lambda c: tidy(tool, db, c["file"]), cmds))
    seen = set()
    for text in outputs:
        for line in text.splitlines():
            m = FINDING.match(line)
            if m is None:
                continue
            path = os.path.relpath(os.path.normpath(os.path.join(ROOT, m[1])), ROOT)
            if path.startswith(REPORTED):
                seen.add((path, int(m[2]), int(m[3]), m[6], m[5]))
    return seen


def findings(tool, cmds):
    """{(file, check, message): count}; line numbers stay out."""
    return collections.Counter((p, check, msg) for p, _, _, check, msg in scan(tool, cmds))


def read_baseline():
    counts = collections.Counter()
    if os.path.exists(BASELINE):
        with open(BASELINE) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                n, path, check, msg = line.rstrip("\n").split("\t", 3)
                counts[(path, check, msg)] = int(n)
    return counts


def unexplained(base):
    """Analyzer findings in the baseline without a line in static-reasons.txt."""
    with open(REASONS) as f:
        known = {tuple(l.split("\t")[:2]) for l in f if not l.startswith("#")}
    keys = {(p, c) for p, c, _ in base if c.startswith("clang-analyzer-")}
    return sorted(keys - known)


def ratchet(now, base):
    """(findings above the baseline, baseline entries no longer found)."""
    return now - base, base - now


def write_baseline(counts):
    with open(BASELINE, "w") as f:
        f.write("# scripts/check_static.py: known findings; this list only shrinks.\n")
        f.write("# count, file, check, message. Regenerate with --update.\n")
        for (path, check, msg), n in sorted(counts.items()):
            f.write(f"{n}\t{path}\t{check}\t{msg}\n")


def nolint_ok(line):
    m = NOLINT.search(line)
    return m is None or re.match(r"\s*:\s*\w", m[2]) is not None


def nolint_without_reason():
    tracked = subprocess.run(
        ["git", "ls-files", "src", "include", "tests", "tools"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    bad = []
    for path in tracked:
        if not path.endswith((".c", ".h", ".m")):
            continue
        with open(os.path.join(ROOT, path), errors="replace") as f:
            for no, line in enumerate(f, 1):
                if not nolint_ok(line):
                    bad.append(f"{path}:{no}: NOLINT without a reason")
    return bad


def expectations(path, text):
    """{(file, line, check)} from the `// expect: <check>` marks of a fixture."""
    return {
        (path, no, c)
        for no, line in enumerate(text.splitlines(), 1)
        for c in re.findall(r"// expect: (\S+)", line)
    }


def selftest_diff(want, got):
    """(seeded defects not reported, findings on no `// expect:` line).

    Both must be empty: a seeded defect the analyzer stops finding fails, and
    so does a defect added to a fixture without its mark -- the fixtures say
    exactly what the job catches."""
    return sorted(want - got), sorted(got - want)


def selftest(tool):
    """tests/static/ must report exactly its `// expect:` marks."""
    cmds = compile_commands()
    flags = [a for a in cmds[0]["arguments"] if a.startswith(("-std=", "-I", "-D"))]
    fixtures = sorted(
        os.path.join("tests/static", n)
        for n in os.listdir(os.path.join(ROOT, "tests/static"))
        if n.endswith(".c")
    )
    want = set()
    for path in fixtures:
        with open(os.path.join(ROOT, path)) as f:
            want |= expectations(path, f.read())
    seeded = [
        {"directory": ROOT, "arguments": ["cc", *flags, "-c", p, "-o", "/dev/null"], "file": p}
        for p in fixtures
    ]
    got = {(p, line, check) for p, line, _, check, _ in scan(tool, seeded)}
    missed, unmarked = selftest_diff(want, got)
    for path, line, check in missed:
        print(f"selftest: {path}:{line}: {check} was not reported")
    for path, line, check in unmarked:
        print(f"selftest: {path}:{line}: {check} reported, but the line has no // expect: mark")
    print(f"selftest: {len(want) - len(missed)} of {len(want)} seeded defects found, "
          f"{len(unmarked)} unmarked")
    return 1 if missed or unmarked else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    tool = os.environ.get("CLANG_TIDY", "clang-tidy")
    if a.selftest:
        return selftest(tool)

    bad = nolint_without_reason()
    bad += [f"{p}: {c}: in the baseline without a reason" for p, c in unexplained(read_baseline())]
    for line in bad:
        print(line)
    now = findings(tool, compile_commands())
    if a.update:
        write_baseline(now)
        print(f"baseline: {sum(now.values())} findings")
        return 1 if bad else 0
    new, gone = ratchet(now, read_baseline())
    for (path, check, msg), n in sorted(new.items()):
        print(f"new: {path}: {check}: {msg} (+{n})")
    for (path, check, msg), n in sorted(gone.items()):
        print(f"gone, drop it from the baseline: {path}: {check}: {msg} (-{n})")
    print(f"check_static: {sum(now.values())} findings, {sum(new.values())} new, "
          f"{sum(gone.values())} gone")
    return 1 if bad or new or gone else 0


if __name__ == "__main__":
    sys.exit(main())
