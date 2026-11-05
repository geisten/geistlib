#!/usr/bin/env python3
"""check_layers.py - module boundaries from the compiler, against one table (#344).

tools/layers.json names the layers of src/ and what each may use. This tool
checks four things against it, with a baseline of known violations that may
only shrink (tools/layers_baseline.txt):

  include  every include edge, from the compiler's own include tree (-H) of
           each library source and each internal header, built with the flags
           `make -n -B lib` prints. A text scan would not see the edges:
           -Isrc/base and friends make `#include "heap.h"` mean any layer.
  symbol   every undefined symbol of every built object (nm), against the
           layer that defines it. A symbol declared in an interface layer's
           header (include/, src/quant/) belongs to that layer, whoever
           defines it. Catches a call across a layer that declares the
           prototype itself and needs no include.
  header   every internal header compiles on its own (like check-headers for
           the public ones), with the flags of its directory.
  if       the portability #ifs per file outside the wrappers (src/base/ and
           the backends, tools/layers.json "portability"): a
           #if/#ifdef/#ifndef/#elif whose condition names a reserved
           identifier (__APPLE__, __ARM_NEON, _OPENMP, ...), counted with the
           lexer of tools/c23_metrics.py. The count of a file may only fall
           (AGENT.md §5).

  check_layers.py --build-dir DIR [VAR=value ...]
                         all four; DIR holds the objects of `make lib`, the
                         VAR=value are passed on to `make -n -B lib`
  check_layers.py --update ...
                         rewrite the baseline: the lines this run judged are
                         replaced by what it found, the others stay
  check_layers.py --doc  write the table into docs/ARCHITECTURE.md (every run
                         fails while the two differ)
  check_layers.py --selftest
                         the violations seeded in tests/layers/ must be
                         reported, and nothing else; no binary runs

The build configuration is whatever `make -n -B lib` gives: `make
check-layers` runs it with the TARGET/BACKENDS/CC of the build. Files this
configuration does not compile (cpu_x86 on arm64, vulkan without its
headers) are not judged, and their baseline lines are left alone. A line
about a judged file that this run did not find is only reported (another
configuration may see it), except an `if` line: the lexer sees every file
the same everywhere, so a count below its ceiling fails until the ceiling
is lowered.

JOBS sets the parallelism; CC the compiler of the selftest (default cc).
Stdlib only, plus a C compiler and nm.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "tools" / "layers.json"
BASELINE = ROOT / "tools" / "layers_baseline.txt"
FIXTURE = ROOT / "tests" / "layers"
DOC = ROOT / "docs" / "ARCHITECTURE.md"
DOC_BEGIN = "<!-- layers:begin (generated from tools/layers.json by tools/check_layers.py --doc) -->"
DOC_END = "<!-- layers:end -->"

_spec = importlib.util.spec_from_file_location("c23_metrics", ROOT / "tools" / "c23_metrics.py")
c23_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c23_metrics)

UNDEFINED = ("U", "w")  # nm -P types of an undefined (or weak undefined) symbol
PP_IF = ("if", "ifdef", "ifndef", "elif", "elifdef", "elifndef")
RESERVED = re.compile(r"^_[A-Z_]")


def jobs() -> int:
    return int(os.environ.get("JOBS", os.cpu_count() or 1))


def pmap(fn, items):
    with ThreadPoolExecutor(jobs()) as pool:
        return list(pool.map(fn, items))


# ---- the table -------------------------------------------------------------

class Table:
    def __init__(self, path: Path):
        doc = json.loads(path.read_text())
        self.layers = {l["name"]: l for l in doc["layers"]}
        self.prefixes = sorted(
            ((p, l["name"]) for l in doc["layers"] for p in l["paths"]),
            key=lambda pl: -len(pl[0]),
        )
        self.composition = doc.get("composition", {})
        self.header_defines = doc.get("header_defines", [])
        self.generated = tuple(doc.get("generated_headers", []))
        # headers that are templates of their includers, by design: not
        # compiled alone (their includers' edges are checked as usual)
        self.fragments = doc.get("fragments", {})
        port = doc.get("portability", {})
        self.wrappers = tuple(port.get("wrappers", []))
        self.not_portability = set(port.get("not_portability", []))
        # The -H tree shows a header only where it is first opened: when a.c
        # includes x.h, which includes s.h, a later `#include "s.h"` in a.c
        # prints nothing. With may_use closed under "uses", such a hidden edge
        # is allowed whenever the edges it hides behind are, so no violation
        # hides; the table must therefore be closed.
        for l in doc["layers"]:
            for u in l["may_use"]:
                if u not in self.layers:
                    raise SystemExit(f"{path}: {l['name']} may use unknown layer {u}")
                missing = set(self.layers[u]["may_use"]) - set(l["may_use"]) - {l["name"]}
                if missing:
                    raise SystemExit(f"{path}: {l['name']} may use {u}, which may use "
                                     f"{', '.join(sorted(missing))}: list those too")

    def layer_of(self, rel: str) -> str | None:
        for p, name in self.prefixes:
            if rel.startswith(p):
                return name
        return None

    def interfaces(self) -> list[str]:
        return [n for n, l in self.layers.items() if l.get("interface")]

    def allowed(self, src: str, dst_layer: str | None) -> bool:
        """May file `src` use something of layer `dst_layer`?"""
        src_layer = self.layer_of(src)
        if dst_layer is None or src_layer is None:
            return False
        if dst_layer == src_layer or dst_layer in self.layers[src_layer]["may_use"]:
            return True
        # a composition point may use the layers it names, and what they may
        return any(dst_layer == c or dst_layer in self.layers[c]["may_use"]
                   for c in self.composition.get(src, ()))


def render(table: Table) -> str:
    """The table as docs/ARCHITECTURE.md shows it."""
    rows = ["| layer | files | may use | |", "| :-- | :-- | :-- | :-- |"]
    for name, l in table.layers.items():
        paths = ", ".join(f"`{p}`" for p in l["paths"])
        uses = ", ".join(l["may_use"]) or "nothing"
        what = l.get("what", "") + (" (interface)" if l.get("interface") else "")
        rows.append(f"| {name} | {paths} | {uses} | {what} |")
    rows.append("")
    for f, ls in table.composition.items():
        rows.append(f"Composition point `{f}` may also use {', '.join(ls)}.")
        rows.append("")
    return "\n".join(rows)


def doc_with_table(table: Table) -> tuple[str, str]:
    """(docs/ARCHITECTURE.md as it is, as it should be)."""
    text = DOC.read_text()
    s, e = text.find(DOC_BEGIN), text.find(DOC_END)
    if s < 0 or e < s:
        raise SystemExit(f"check_layers: {DOC.relative_to(ROOT)} lacks the {DOC_BEGIN} markers")
    return text, text[:s + len(DOC_BEGIN)] + "\n" + render(table) + text[e:]


# ---- compile commands --------------------------------------------------------

def lib_compile_commands(root: Path = ROOT, make_args=()) -> list[tuple[str, list[str]]]:
    """(source, argv) of each library source under src/, as `make lib` builds it."""
    out = subprocess.run(
        ["make", "-n", "-B", "lib", *make_args], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout
    cmds = []
    for line in out.splitlines():
        if " -c " not in line or " -o " not in line:
            continue
        argv = shlex.split(line)
        src = argv[argv.index("-c") + 1]
        if src.startswith("src/") and src.endswith(".c"):
            cmds.append((src, argv))
    return cmds


def object_of(argv: list[str]) -> str:
    return argv[argv.index("-o") + 1]


def flags_of(argv: list[str]) -> list[str]:
    """The argv without its input, output and dependency-file options."""
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a in ("-c", "-o", "-MF", "-MT", "-MQ"):
            skip = True
            continue
        if a in ("-MMD", "-MD", "-MP"):
            continue
        out.append(a)
    return out


# ---- include edges ------------------------------------------------------------

def rel(root: Path, path: str) -> str | None:
    """`path` relative to root, or None outside of it (a system header)."""
    p = Path(os.path.normpath(root / path))
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return None


def include_tree(root: Path, top: str, stderr: str) -> set[tuple[str, str]]:
    """(includer, included) of the -H tree, both inside root; `top` is depth 0."""
    edges = set()
    stack = [top]
    for line in stderr.splitlines():
        m = re.match(r"^(\.+) (.+)$", line)
        if m is None:
            continue
        depth = len(m[1])
        path = rel(root, m[2].strip())
        del stack[depth:]
        parent = stack[-1] if stack else None
        if parent is not None and path is not None:
            edges.add((parent, path))
        stack.append(path)
    return edges


def tu_edges(root: Path, cmds) -> tuple[set, list[str]]:
    """Include edges of every source; and the sources that did not preprocess."""

    def one(c):
        src, argv = c
        r = subprocess.run([*flags_of(argv), "-H", "-E", src, "-o", os.devnull],
                           cwd=root, capture_output=True, text=True)
        return src, r.returncode, include_tree(root, src, r.stderr), r.stderr

    edges, failed = set(), []
    for src, rc, e, err in pmap(one, cmds):
        edges |= e
        if rc != 0:
            failed.append(f"{src}: does not preprocess\n{err[-1500:]}")
    return edges, failed


def files_under(root: Path, *dirs: str) -> list[str]:
    """Every file below the given directories, relative to root. A walk, not
    `git ls-files`: CI's Alpine container has no git, and a new file is
    checked before it is added."""
    out = []
    for d in dirs:
        for base, subdirs, names in os.walk(root / d):
            subdirs.sort()
            out += [(Path(base) / n).relative_to(root).as_posix() for n in names]
    return sorted(out)


def internal_headers(root: Path, table: Table) -> list[str]:
    return [f for f in files_under(root, "src") if f.endswith(".h")
            and not f.endswith(table.generated) and f not in table.fragments]


def flags_for_header(table: Table, cmds, header: str) -> list[str] | None:
    """The flags of a source in the header's directory, else of its layer."""
    hdir = os.path.dirname(header)
    layer = table.layer_of(header)
    best, best_len = None, -1
    for src, argv in cmds:
        if table.layer_of(src) != layer:
            continue
        common = os.path.commonpath([hdir, os.path.dirname(src)])
        if len(common) > best_len:
            best, best_len = argv, len(common)
    return flags_of(best) if best is not None else None


def header_checks(root: Path, table: Table, cmds, headers) -> tuple[set, list, list]:
    """(include edges of the headers, headers that do not compile alone, skipped)."""
    defines = [f"-D{d}=" for d in table.header_defines]  # as `#define X` would

    def one(h):
        flags = flags_for_header(table, cmds, h)
        if flags is None:
            return h, None, set(), ""
        # the declaration keeps a header of macros only from being an empty TU
        stub = f'#include "{root / h}"\ntypedef int check_layers_stub;\n'
        r = subprocess.run([*flags, *defines, "-H", "-fsyntax-only", "-x", "c", "-"],
                           cwd=root, input=stub, capture_output=True, text=True)
        return h, r.returncode, include_tree(root, "<stub>", r.stderr), r.stderr

    edges, bad, skipped = set(), [], []
    for h, rc, e, err in pmap(one, headers):
        if rc is None:
            skipped.append(h)
            continue
        edges |= {(a, b) for a, b in e if a != "<stub>"}
        if rc != 0:
            bad.append((h, err))
    return edges, bad, skipped


# ---- symbols --------------------------------------------------------------------

def nm(obj: Path) -> tuple[set[str], set[str]]:
    """(defined, undefined) external symbols of an object, C names."""
    out = subprocess.run(["nm", "-g", "-P", str(obj)], capture_output=True, text=True,
                         check=True).stdout
    with open(obj, "rb") as f:
        strip = f.read(4) != b"\x7fELF"  # Mach-O (and COFF) prefix C names with _
    defined, undefined = set(), set()
    for line in out.splitlines():
        f = line.split()
        if len(f) < 2 or f[0].endswith(":"):
            continue
        name = f[0][1:] if strip and f[0].startswith("_") else f[0]
        (undefined if f[1] in UNDEFINED else defined).add(name)
    return defined, undefined


def declared_names(root: Path, flags: list[str], header: str) -> set[str]:
    """Functions and extern objects the header itself declares, after the
    preprocessor: names at file scope followed by `(`, and the last name of an
    `extern` declaration."""
    r = subprocess.run([*flags, "-E", "-x", "c", "-"], cwd=root,
                       input=f'#include "{root / header}"\n', capture_output=True, text=True)
    if r.returncode != 0:
        return set()
    names = set()
    target = os.path.normpath(root / header)
    cur = None
    depth = 0  # (), [] and {} together
    ext, last, last_id = False, None, None
    for t_line in r.stdout.splitlines():
        m = re.match(r'^#\s*(?:line\s+)?\d+\s+"([^"]*)"', t_line)
        if m:
            cur = os.path.normpath(os.path.join(root, m[1]))
            continue
        if t_line.startswith("#"):
            continue
        toks = c23_metrics.lex(t_line)
        for i, t in enumerate(toks):
            s = t.text
            if s in ("(", "[", "{"):
                if s == "(" and depth == 0 and cur == target and last is not None \
                        and last.kind == "id" and not last.text.startswith("__") \
                        and last.text not in ("alignas", "typeof", "static_assert"):
                    names.add(last.text)
                depth += 1
            elif s in (")", "]", "}"):
                depth -= 1
            elif depth == 0 and s == "extern":
                ext = True
            elif depth == 0 and s in (";", "=", ","):
                if ext and cur == target and last_id is not None:
                    names.add(last_id)
                if s == ";":
                    ext = False
            if depth == 0 and t.kind == "id":
                last_id = s
            last = t
    return names


def interface_symbols(root: Path, table: Table, cmds) -> dict[str, str]:
    """{symbol: interface layer} for every name an interface header declares."""
    if not cmds:
        return {}
    flags = flags_of(cmds[0][1])
    headers = [f for f in files_under(root, *[
        p for n in table.interfaces() for p in table.layers[n]["paths"]]) if f.endswith(".h")]
    owner = {}
    for h, names in zip(headers, pmap(lambda h: declared_names(root, flags, h), headers)):
        for n in names:
            owner.setdefault(n, table.layer_of(h))
    return owner


def symbol_violations(root: Path, table: Table, cmds) -> set[str]:
    objs = [(src, root / object_of(argv)) for src, argv in cmds]
    missing = [str(o) for _, o in objs if not o.exists()]
    if missing:
        raise SystemExit(f"check_layers: {len(missing)} objects missing, run `make lib` "
                         f"first (e.g. {missing[0]})")
    syms = pmap(lambda so: nm(so[1]), objs)
    defined_in = {}
    for (src, _), (d, _) in zip(objs, syms):
        for s in d:
            defined_in.setdefault(s, src)
    owner = interface_symbols(root, table, cmds)
    bad = set()
    for (src, _), (_, und) in zip(objs, syms):
        for s in und:
            where = defined_in.get(s)
            if where is None:  # libc, OpenMP, BLAS, the OS
                continue
            layer = owner.get(s) or table.layer_of(where)
            if not table.allowed(src, layer):
                bad.add(f"symbol {src} -> {s} ({where})")
    return bad


# ---- portability #ifs -------------------------------------------------------------

def portability_ifs(root: Path, table: Table, files) -> dict[str, int]:
    counts = {}
    for f in files:
        if f.startswith(table.wrappers):
            continue
        toks = c23_metrics.lex((root / f).read_text(encoding="utf-8", errors="replace"))
        n = 0
        for i, t in enumerate(toks):
            if not (t.bol and t.text == "#" and i + 1 < len(toks) and toks[i + 1].text in PP_IF):
                continue
            j = i + 2
            while j < len(toks) and not toks[j].bol:
                s = toks[j].text
                if toks[j].kind == "id" and RESERVED.match(s) and s not in table.not_portability:
                    n += 1
                    break
                j += 1
        if n:
            counts[f] = n
    return counts


def lexed_sources(root: Path, table: Table) -> list[str]:
    """c23_metrics.py's scope: the C sources of src/ and include/."""
    return [f for f in files_under(root, "src", "include")
            if f.endswith((".c", ".h", ".m")) and not f.endswith(table.generated)]


# ---- the whole check ------------------------------------------------------------------

def analyse(root: Path, table: Table, cmds, symbols: bool):
    """(violations, analysed files, errors). Violations are baseline lines."""
    errors = []
    edges, failed = tu_edges(root, cmds)
    errors += failed
    headers = internal_headers(root, table)
    h_edges, h_bad, h_skipped = header_checks(root, table, cmds, headers)
    edges |= h_edges
    v = set()
    for a, b in edges:
        if not table.allowed(a, table.layer_of(b)):
            v.add(f"include {a} -> {b}")
    v |= {f"header {h}" for h, _ in h_bad}
    built_headers = set(headers) - set(h_skipped)
    files = lexed_sources(root, table)
    analysed = {  # per kind, the files whose lines this run can judge
        "include": {src for src, _ in cmds} | built_headers,
        "header": built_headers,
        "symbol": {src for src, _ in cmds} if symbols else set(),
        "if": set(files),
    }
    if symbols:
        v |= symbol_violations(root, table, cmds)
    for f, n in portability_ifs(root, table, files).items():
        v.add(f"if {f} {n}")
    return v, analysed, errors, h_bad, h_skipped


def subject(line: str) -> str:
    """The file a baseline line is about."""
    return line.split()[1]


def judged(line: str, analysed: dict) -> bool:
    """Did this run analyse what the baseline line is about?"""
    return subject(line) in analysed.get(line.split()[0], ())


def read_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {l.rstrip("\n") for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")}


def compare(now: set[str], base: set[str], analysed: dict) -> tuple[list, list]:
    """(new violations, baseline lines about an analysed file no longer seen).
    An `if` line is a ceiling: a count above it is new, one below it stale."""
    ceil = {}
    for l in base:
        k, f, *rest = l.split()
        if k == "if":
            ceil[f] = int(rest[0])
    new, gone = [], []
    for l in sorted(now - base):
        k, f, *rest = l.split()
        if k == "if" and int(rest[0]) <= ceil.get(f, 0):
            continue
        new.append(l)
    for l in sorted(base - now):
        if judged(l, analysed):
            gone.append(l)
    return new, gone


def write_baseline(path: Path, now: set[str], base: set[str], analysed: dict):
    keep = {l for l in base if not judged(l, analysed)}
    lines = sorted(now | keep, key=lambda l: (l.split()[0], l))
    with open(path, "w") as f:
        f.write("# tools/check_layers.py: known violations of tools/layers.json; this list\n")
        f.write("# only shrinks. `if <file> <n>` is a ceiling. Regenerate with --update.\n")
        for l in lines:
            f.write(l + "\n")


def report(new, gone, h_bad, errors):
    for e in errors:
        print(f"error: {e}")
    alone = {h: err for h, err in h_bad}
    for l in new:
        print(f"new: {l}")
        if l.startswith("header "):
            print("    " + "\n    ".join(alone[subject(l)].strip().splitlines()[-6:]))
    for l in gone:
        if l.startswith("if "):
            print(f"lower, lower the ceiling (--update): {l}")
        else:  # may be another configuration's: a note, not a failure
            print(f"not seen here; if gone everywhere, drop it from the baseline: {l}")


def selftest() -> int:
    """tests/layers/ is a small tree with one violation of each kind seeded
    (expected.txt). Every one must be reported, nothing else may be, and with
    them in a baseline the check must pass. Only the compiler and nm run."""
    table = Table(FIXTURE / "layers.json")
    spec = json.loads((FIXTURE / "layers.json").read_text())["selftest"]
    cc = shlex.split(os.environ.get("CC", "cc"))
    want = read_baseline(FIXTURE / "expected.txt")
    with tempfile.TemporaryDirectory() as tmp:
        cmds = [(s, [*cc, *spec["flags"], "-c", s, "-o", str(Path(tmp) / (s + ".o"))])
                for s in spec["sources"]]
        for s, argv in cmds:
            os.makedirs(os.path.dirname(object_of(argv)), exist_ok=True)
            r = subprocess.run(argv, cwd=FIXTURE, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"selftest: {s} does not compile:\n{r.stderr}")
                return 1
        now, analysed, errors, _, skipped = analyse(FIXTURE, table, cmds, True)
    status = 0
    for l in sorted(want - now):
        print(f"selftest: not reported: {l}")
        status = 1
    for l in sorted(now - want):
        print(f"selftest: reported, but not seeded: {l}")
        status = 1
    for e in errors + [f"{h}: skipped" for h in skipped]:
        print(f"selftest: {e}")
        status = 1
    new, gone = compare(now, want, analysed)
    if new or gone:
        print("selftest: the expected violations as a baseline do not pass")
        status = 1
    lower = {l if not l.startswith("if ") else f"{l.rsplit(' ', 1)[0]} 9" for l in want}
    new, _ = compare(now, lower, analysed)
    if new:
        print("selftest: an `if` count under its ceiling failed")
        status = 1
    print(f"selftest: {len(want & now)} of {len(want)} seeded violations found"
          + ("" if status else ", nothing else"))
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-dir", help="where `make lib` put the objects; without it, "
                    "no symbol check")
    ap.add_argument("--update", action="store_true", help="rewrite the baseline")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--doc", action="store_true",
                    help="write the table into docs/ARCHITECTURE.md")
    ap.add_argument("make_args", nargs="*", help="passed on to `make -n -B lib`")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    table = Table(TABLE)
    have, want = doc_with_table(table)
    if a.doc:
        DOC.write_text(want)
        return 0
    if have != want:
        print(f"check_layers: the table in {DOC.relative_to(ROOT)} differs from "
              f"{TABLE.relative_to(ROOT)}; run tools/check_layers.py --doc")
        return 1
    cmds = lib_compile_commands(ROOT, a.make_args)
    if not cmds:
        print("check_layers: `make -n -B lib` printed no compile commands")
        return 1
    if a.build_dir:
        outside = [o for _, argv in cmds if not (ROOT / object_of(argv)).is_relative_to(
            ROOT / a.build_dir)]
        if outside:
            print(f"check_layers: make builds {outside[0]}, not under {a.build_dir}")
            return 1
    now, analysed, errors, h_bad, skipped = analyse(ROOT, table, cmds, bool(a.build_dir))
    base = read_baseline(BASELINE)
    if a.update:
        write_baseline(BASELINE, now, base, analysed)
        print(f"check_layers: {BASELINE.relative_to(ROOT)} has {len(read_baseline(BASELINE))} lines")
        return 1 if errors else 0
    new, gone = compare(now, base, analysed)
    report(new, gone, h_bad, errors)
    kinds = {}
    for l in now:
        kinds[l.split()[0]] = kinds.get(l.split()[0], 0) + 1
    n_if = sum(int(l.split()[2]) for l in now if l.startswith("if "))
    print(f"check_layers: {len(cmds)} sources and {len(analysed['header'])} "
          f"internal headers ({len(skipped)} more not built in this configuration); "
          f"{kinds.get('include', 0)} include, {kinds.get('symbol', 0)} symbol and "
          f"{kinds.get('header', 0)} header violations, {n_if} portability #ifs in "
          f"{kinds.get('if', 0)} files; {len(new)} new, {len(gone)} gone")
    stale_if = [l for l in gone if l.startswith("if ")]
    return 1 if new or stale_if or errors else 0


if __name__ == "__main__":
    sys.exit(main())
