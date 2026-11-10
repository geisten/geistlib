#!/usr/bin/env python3
"""include_graph.py — the include graph of src/ and include/, per layer (#341).

A text scan of `#include` lines cannot see where a header comes from: with
-Isrc/base and friends, `#include "heap.h"` reads the same in every layer.
This tool asks the compiler. Every .c, .m and .h file is preprocessed on
its own with `clang -E -dI`, which keeps each #include directive of the
active branches in the output, followed by a line marker naming the file it
opened. A directive whose header an include guard already saw opens
nothing; it is resolved by the search rule (the includer's directory for
"...", then -I in order), and that rule is checked against the compiler on
every directive the compiler did open, so a mismatch fails the run.
Each edge is direct: includer -> included, as written in the includer.

Host independence: -nostdinc, and the system headers the tree names are
empty stand-ins in a temporary -isystem directory (found first with
-M -MG, which lists missing headers instead of failing). Only the project's
-I directories count, and the target comes from --target, not the host.
CPATH and friends are dropped from the environment. Apple clang 21 and
LLVM clang 22 give the same graph. gcc cannot preprocess for a foreign
target, so the tool needs clang.

Three configurations cover the backends the Makefile builds:

    macos-arm64    cpu_neon cpu_scalar metal   (TARGET=mac)
    linux-arm64    cpu_neon cpu_scalar vulkan  (TARGET=linux on aarch64)
    linux-x86_64   cpu_x86 cpu_scalar vulkan   (TARGET=linux on x86_64,
                                                with the AVX-512/AMX flags
                                                mk/backend-cpu_x86.mk gives
                                                single TUs, on every TU)

An edge is recorded with the configurations it occurs in; a header outside
a configuration's backends is not preprocessed there. The -I list is
read from CFLAGS_BASE in mk/common.mk, so it cannot drift from the build.

    write   write build/include_graph/include_graph.json and INCLUDE_GRAPH.md
    show    print the layer matrix

A report, not a gate: nothing is committed (#442). An exact-match check on a
committed edge list made every pair of PRs that touch includes conflict;
tools/check_layers.py (#344) enforces the edges that matter, from the same
compiler output.

Library use: `layer_of(path)` maps a file to its layer; `graph(cc)` returns
the document written to the JSON.

Stdlib only; runs on Python 3.9.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "build" / "include_graph"
JSON_OUT = OUT_DIR / "include_graph.json"
MD_OUT = OUT_DIR / "INCLUDE_GRAPH.md"
GENERATED = ("_spv.h", "metal_shaders.h")  # data only, include nothing
ALL_BACKENDS = ("cpu_neon", "cpu_scalar", "cpu_x86", "metal", "vulkan")
FAMILIES = ("bitnet", "deepseek2", "gemma3", "gemma4", "glm4moe", "gpt_oss", "granitemoe",
            "llama", "llama4", "olmoe", "qwen3", "qwen35", "qwen3moe")
# What a header needs defined to be preprocessed on its own: internal
# headers #error without their layer's guard.
HEADER_DEFINES = ("GEIST_INTERNAL_ENGINE_LAYER=", "GEIST_INTERNAL_ARCH_LAYER=",
                  "GEIST_INTERNAL_BACKEND_LAYER=")

CONFIGS = {
    "macos-arm64": {
        "target": "arm64-apple-macos14",
        "flags": ["-DHAVE_ACCELERATE=1"],
        "backends": ["cpu_neon", "cpu_scalar", "metal"],
    },
    "linux-arm64": {
        "target": "aarch64-linux-gnu",
        "flags": ["-march=armv8.2-a+fp16+dotprod", "-D_GNU_SOURCE",
                  "-DGEIST_GEMM_OPENBLAS"],
        "backends": ["cpu_neon", "cpu_scalar", "vulkan"],
    },
    "linux-x86_64": {
        "target": "x86_64-linux-gnu",
        "flags": ["-march=x86-64-v3", "-mf16c", "-mavx512f", "-mavx512bw",
                  "-mavx512dq", "-mavx512vl", "-mavx512vnni", "-mamx-tile",
                  "-mamx-int8", "-D_GNU_SOURCE", "-DGEIST_GEMM_OPENBLAS"],
        "backends": ["cpu_x86", "cpu_scalar", "vulkan"],
    },
}

# A layer is a directory: src/<a> for the flat layers, src/<a>/<b> under the
# directories that group several (formats, backends, archs).
GROUPED = ("formats", "backends", "archs")


def layer_of(path: str) -> str:
    """The layer a repository-relative path belongs to."""
    parts = path.split("/")
    if parts[0] == "src" and len(parts) > 2:
        if parts[1] in GROUPED and len(parts) > 3:
            return "/".join(parts[:3])
        return "/".join(parts[:2])
    if parts[0] == "third_party" and len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0]


def include_dirs() -> list[str]:
    """The -I list of CFLAGS_BASE in mk/common.mk."""
    text = (ROOT / "mk" / "common.mk").read_text()
    m = re.search(r"^CFLAGS_BASE\s*:=((?:.*\\\n)*.*)$", text, re.M)
    if not m:
        sys.exit("include_graph: CFLAGS_BASE not found in mk/common.mk")
    dirs = re.findall(r"-I(\S+)", m.group(1))
    if not dirs:
        sys.exit("include_graph: CFLAGS_BASE has no -I")
    return dirs


def sources() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "src", "include"],
                         cwd=ROOT, capture_output=True, check=True).stdout
    files = [f for f in out.decode().split("\0") if f]
    return sorted(f for f in files
                  if f.endswith((".c", ".h", ".m")) and not f.endswith(GENERATED))


def in_config(path: str, backends: list[str]) -> bool:
    parts = path.split("/")
    if parts[:2] == ["src", "backends"] and parts[2] in ALL_BACKENDS:
        return parts[2] in backends
    return not path.endswith(".m")


def check_cc(cc: str) -> str:
    exe = shutil.which(cc)
    if not exe:
        sys.exit(f"include_graph: {cc} not found (needs clang; set CC_CLANG)")
    ver = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    if "clang" not in ver:
        sys.exit(f"include_graph: {cc} is not clang; gcc cannot preprocess for "
                 "a foreign --target")
    return exe


def base_cmd(cc: str, cfg: dict, idirs: list[str], path: str) -> list[str]:
    objc = path.endswith(".m") or path.startswith("src/backends/metal/")
    cmd = [cc, f"--target={cfg['target']}", *cfg["flags"],
           "-x", "objective-c" if objc else "c", "-std=c23", "-nostdinc"]
    cmd += [f"-I{d}" for d in idirs]
    cmd += [f"-DGEIST_BACKEND_{b.upper()}=1" for b in cfg["backends"]]
    cmd += [f"-DGEIST_FAMILY_{f.upper()}=1" for f in FAMILIES]
    cmd += ["-DGEIST_TOWERS=1"]
    if path.endswith(".h"):
        cmd += [f"-D{d}" for d in HEADER_DEFINES]
    return cmd


# The compiler adds these to the search path even under -nostdinc (macOS's
# /usr/bin/python3 sets CPATH=/usr/local/include).
ENV = {k: v for k, v in os.environ.items()
       if k not in ("CPATH", "C_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
                    "CPLUS_INCLUDE_PATH", "CCC_OVERRIDE_OPTIONS")}


def run(cmd: list[str], path: str) -> str:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        sys.exit(f"include_graph: preprocessing {path} failed:\n"
                 f"{' '.join(cmd)}\n{r.stderr}")
    return r.stdout


def missing(cc: str, cfg: dict, idirs: list[str], path: str) -> set[str]:
    """Headers the file names that are not in the tree (the system's), as
    spelled; -MG lists them instead of failing."""
    out = run(base_cmd(cc, cfg, idirs, path) + ["-M", "-MG", "-MF", "-", path], path)
    deps = out.replace("\\\n", " ").split(":", 1)[1].split()
    return {d for d in deps if not (ROOT / d).is_file()}


def rel(p: str) -> str:
    return os.path.relpath(os.path.normpath(ROOT / p), ROOT)


def resolve(idirs: list[str], stub: str, includer: str, quote: bool, name: str):
    """Where the preprocessor finds `name`: the includer's directory for a
    quoted include, then -I in order, then the stub system directory."""
    dirs = ([os.path.dirname(includer)] if quote else []) + idirs + [stub]
    for d in dirs:
        cand = os.path.join(d, name)
        if (ROOT / cand).is_file():
            return cand
    return None


DIRECTIVE = re.compile(r'^#\s*(?:include|import)\s*([<"])([^>"]+)[>"].*/\* clang -E -dI \*/$')
MARKER = re.compile(r'^# \d+ "((?:[^"\\]|\\.)*)"((?: \d)*)$')


def direct(cc: str, cfg: dict, idirs: list[str], stub: str, path: str) -> list[tuple[str, str]]:
    """Direct (includer, included) edges of every file the preprocessor reads
    for `path`. `-dI` keeps each #include directive in the output, also the
    ones an include guard skips; the line marker after a directive says
    which file it opened. A skipped one is resolved by the search rule, and
    that rule is checked against the compiler on every directive it did
    open, so it cannot drift from the compiler's."""
    out = run(base_cmd(cc, cfg, idirs, path) + ["-isystem", stub, "-E", "-dI", path], path)
    stubabs = os.path.realpath(stub)
    edges = []
    stack: list[str] = []
    pending = None  # (includer, quote, name) of the directive just seen

    def settle(opened):
        nonlocal pending
        inc, quote, name = pending
        pending = None
        found = resolve(idirs, stub, inc, quote, name)
        if found is None:
            sys.exit(f"include_graph: {inc}: cannot resolve {name}")
        found = "<system>" if os.path.realpath(found).startswith(stubabs + os.sep) \
            else rel(found)
        if opened is not None and opened != found:
            sys.exit(f"include_graph: {inc}: {name} opened {opened}, search rule "
                     f"says {found}; the rule in resolve() is out of step")
        if found != "<system>" and not inc.startswith("<"):
            edges.append((inc, found))

    for line in out.splitlines():
        m = MARKER.match(line)
        if m:
            name, flags = m.group(1), m.group(2).split()
            f = name if name.startswith("<") else (
                "<system>" if os.path.realpath(name).startswith(stubabs + os.sep) else rel(name))
            if "1" in flags:
                if pending is not None:
                    settle(f)
                stack.append(f)
                continue
            if "2" in flags:  # back in the includer
                if pending is not None:
                    settle(None)
                if stack:
                    stack.pop()
            if stack:
                stack[-1] = f
            else:
                stack.append(f)
            continue
        if pending is not None:
            settle(None)
        m = DIRECTIVE.match(line)
        if m and stack:
            pending = (stack[-1], m.group(1) == '"', m.group(2))
    if pending is not None:
        settle(None)
    return edges


def graph(cc: str = "clang", jobs: int | None = None) -> dict:
    """The include graph as the document benchmark/include_graph.json holds."""
    cc = check_cc(cc)
    idirs = include_dirs()
    files = sources()
    tasks = [(name, cfg, f) for name, cfg in CONFIGS.items()
             for f in files if in_config(f, cfg["backends"])]
    with ThreadPoolExecutor(max_workers=jobs or os.cpu_count() or 4) as ex, \
            tempfile.TemporaryDirectory(prefix="include_graph.") as stub:
        names = set().union(*ex.map(lambda t: missing(cc, t[1], idirs, t[2]), tasks))
        for n in names:  # empty stand-ins, so -E runs without the system's
            (Path(stub) / n).parent.mkdir(parents=True, exist_ok=True)
            (Path(stub) / n).touch()
        results = list(ex.map(lambda t: (t[0], direct(cc, t[1], idirs, stub, t[2])), tasks))
    edges: dict[str, dict[str, set]] = {}
    for name, es in results:
        for a, b in es:
            edges.setdefault(a, {}).setdefault(b, set()).add(name)
    nodes = sorted(set(files) | set(edges) | {b for v in edges.values() for b in v})
    layers: dict[str, int] = {}
    for n in nodes:
        layers[layer_of(n)] = layers.get(layer_of(n), 0) + 1
    lmatrix: dict[str, dict[str, int]] = {}
    for a, tos in edges.items():
        for b in tos:
            la, lb = layer_of(a), layer_of(b)
            row = lmatrix.setdefault(la, {})
            row[lb] = row.get(lb, 0) + 1
    order = list(CONFIGS)
    return {
        "generator": "tools/include_graph.py (#341)",
        "include_dirs": idirs,
        "configs": CONFIGS,
        "layers": dict(sorted(layers.items())),
        "layer_edges": {a: dict(sorted(r.items())) for a, r in sorted(lmatrix.items())},
        "edges": {a: {b: sorted(cs, key=order.index) for b, cs in sorted(tos.items())}
                  for a, tos in sorted(edges.items())},
    }


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


def report(doc: dict) -> str:
    """INCLUDE_GRAPH.md from the document."""
    layers = list(doc["layers"])
    order = list(doc["configs"])
    nedges = sum(len(v) for v in doc["edges"].values())
    out = ["# Include graph", "",
           "Generated by `tools/include_graph.py` (#341) from the preprocessor's "
           "own output (`clang -E -dI`), not from a text scan; regenerate with "
           "`make include-graph`. The data, edge by edge, is in "
           "`include_graph.json` next to this file. "
           "This file reports what is there; which layer may include which is "
           "decided elsewhere (#344).", "",
           f"{len(doc['edges'])} including files, {nedges} direct include edges, "
           f"{len(layers)} layers, configurations: "
           + ", ".join(f"`{c}` ({' '.join(doc['configs'][c]['backends'])})" for c in order)
           + ".", "",
           "## Layers", "",
           "| layer | files | includes from other layers | included by other layers |",
           "|---|--:|--:|--:|"]
    le = doc["layer_edges"]
    for la in layers:
        outgoing = sum(n for lb, n in le.get(la, {}).items() if lb != la)
        incoming = sum(r.get(la, 0) for lx, r in le.items() if lx != la)
        out.append(f"| `{la}` | {doc['layers'][la]} | {outgoing} | {incoming} |")
    out += ["", "## Layer matrix", "",
            "Direct include edges from the row's layer into the column's. "
            "Columns are numbered like the rows; `.` is none.", ""]
    out.append("| from \\ to | " + " | ".join(str(i + 1) for i in range(len(layers))) + " |")
    out.append("|---|" + "--:|" * len(layers))
    for i, la in enumerate(layers):
        row = le.get(la, {})
        cells = [str(row.get(lb, ".")) for lb in layers]
        out.append(f"| {i + 1} `{la}` | " + " | ".join(cells) + " |")
    out += ["", "## Cross-layer edges", "",
            "Every edge whose two ends are in different layers, grouped by "
            "layer pair. An edge that is missing from a configuration which "
            "preprocesses its including file says where it is seen.", ""]
    groups: dict[tuple[str, str], list[str]] = {}
    for a, tos in doc["edges"].items():
        for b, cs in tos.items():
            la, lb = layer_of(a), layer_of(b)
            if la == lb:
                continue
            where = [c for c in order if in_config(a, doc["configs"][c]["backends"])]
            note = "" if cs == where else f" (only {', '.join(cs)})"
            groups.setdefault((la, lb), []).append(f"`{a}` → `{b}`{note}")
    for (la, lb), items in sorted(groups.items()):
        out.append(f"### `{la}` → `{lb}` ({len(items)})")
        out.append("")
        out += [f"- {s}" for s in items]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("write", "show"))
    ap.add_argument("--cc", default=os.environ.get("CC_CLANG", "clang"),
                    help="clang to preprocess with (default: $CC_CLANG or clang)")
    ap.add_argument("-j", "--jobs", type=int, default=None)
    a = ap.parse_args()
    doc = graph(a.cc, a.jobs)
    js, md = dump(doc), report(doc)
    if a.cmd == "write":
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        JSON_OUT.write_text(js)
        MD_OUT.write_text(md)
        print(f"wrote {JSON_OUT.relative_to(ROOT)} and {MD_OUT.relative_to(ROOT)}")
        return 0
    layers = list(doc["layers"])
    w = max(len(x) for x in layers)
    print(f"{'':{w}}  " + " ".join(f"{i + 1:>4}" for i in range(len(layers))))
    for i, la in enumerate(layers):
        row = doc["layer_edges"].get(la, {})
        print(f"{la:{w}}  " + " ".join(f"{row.get(lb, '.'):>4}" for lb in layers)
              + f"   ({i + 1})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
