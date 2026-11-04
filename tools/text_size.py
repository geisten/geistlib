#!/usr/bin/env python3
"""text_size.py — .text bytes of a build, per layer and per library (#341).

#333's gate: ".text size at most +1 % per library, explained if above".
This tool measures it from a finished build, no rebuild:

    measure BUILD_DIR [LIB ...]   JSON on stdout: .text of every object under
                                  BUILD_DIR/{src,third_party}, summed per
                                  layer and in total (= libgeist.a), plus the
                                  .text of each LIB given (libgeist.so/.dylib)
    compare BASE.json NEW.json    per-layer and per-library change; exit 1
                                  when a library grows by more than 1 %

.text is every section named .text or .text.* (ELF; gcc puts cold code in
.text.unlikely) or __text (Mach-O). Sizes come from `llvm-size -A` or GNU
`size -A` (Apple's /usr/bin/size has no -A; `xcrun llvm-size` does), picked
in that order; SIZE overrides.

Typical use, base and new built in separate worktrees:

    make TARGET=linux lib shared
    python3 tools/text_size.py measure build/linux/release lib/linux/release/libgeist.so > new.json

Stdlib only; runs on Python 3.9.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

LIMIT_PCT = 1.0
GROUPED = ("formats", "backends", "archs")


def layer_of(rel: str) -> str:
    """src/<a>, or src/<group>/<b> under formats/, backends/, archs/ (the
    layers of tools/include_graph.py)."""
    parts = rel.split("/")
    if parts[0] == "src" and len(parts) > 2:
        if parts[1] in GROUPED and len(parts) > 3:
            return "/".join(parts[:3])
        return "/".join(parts[:2])
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def size_tool() -> list[str]:
    if os.environ.get("SIZE"):
        return os.environ["SIZE"].split()
    for cand in (["llvm-size"], ["xcrun", "llvm-size"], ["size"]):
        if shutil.which(cand[0]):
            r = subprocess.run(cand + ["--version"], capture_output=True, text=True)
            if r.returncode == 0:
                return cand
    sys.exit("text_size: no llvm-size or GNU size found; set SIZE")


def text_sizes(paths: list[str]) -> dict[str, int]:
    """{path: .text bytes} from one `size -A` call per batch of files."""
    out: dict[str, int] = {}
    tool = size_tool()
    for i in range(0, len(paths), 200):
        batch = paths[i:i + 200]
        r = subprocess.run(tool + ["-A", *batch], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"text_size: {' '.join(tool)} -A failed:\n{r.stderr}")
        cur = None
        for line in r.stdout.splitlines():
            m = re.match(r"^(\S.*?)\s+:\s*$", line)
            if m:
                cur = m.group(1)
                out[cur] = 0
                continue
            f = line.split()
            if cur and len(f) >= 2 and f[1].isdigit() and (
                    f[0] in (".text", "__text") or f[0].startswith(".text.")):
                out[cur] += int(f[1])
    missing = [p for p in paths if p not in out]
    if missing:
        sys.exit(f"text_size: no size output for {missing[:3]}")
    return out


def measure(build_dir: str, libs: list[str]) -> dict:
    base = Path(build_dir)
    objs = sorted(str(p) for d in ("src", "third_party") for p in (base / d).rglob("*.o"))
    if not objs:
        sys.exit(f"text_size: no objects under {build_dir}/src")
    sizes = text_sizes(objs + libs)
    layers: dict[str, int] = {}
    for o in objs:
        layer = layer_of(os.path.relpath(o, base))
        layers[layer] = layers.get(layer, 0) + sizes[o]
    return {
        "objects": len(objs),
        "libgeist.a": sum(sizes[o] for o in objs),
        "layers": dict(sorted(layers.items())),
        "shared": {os.path.basename(p): sizes[p] for p in libs},
    }


def pct(a: int, b: int) -> str:
    return f"{(b - a) / a * 100:+.2f} %" if a else "new"


def compare(base: dict, new: dict) -> int:
    bad = []
    rows = [("libgeist.a", base["libgeist.a"], new["libgeist.a"])]
    rows += [(k, base["shared"].get(k, 0), v) for k, v in new["shared"].items()]
    for name, a, b in rows:
        flag = a and (b - a) / a * 100 > LIMIT_PCT
        print(f"{name:32} {a:>10} -> {b:>10}  {pct(a, b)}{'  > +1 %' if flag else ''}")
        if flag:
            bad.append(name)
    for layer in sorted(set(base["layers"]) | set(new["layers"])):
        a, b = base["layers"].get(layer, 0), new["layers"].get(layer, 0)
        if a != b:
            print(f"  {layer:30} {a:>10} -> {b:>10}  {pct(a, b)}")
    return 1 if bad else 0


def main() -> int:
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "measure":
        print(json.dumps(measure(a[1], a[2:]), indent=1))
        return 0
    if len(a) == 3 and a[0] == "compare":
        return compare(json.loads(Path(a[1]).read_text()), json.loads(Path(a[2]).read_text()))
    print(__doc__.split("\n\n")[2], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
