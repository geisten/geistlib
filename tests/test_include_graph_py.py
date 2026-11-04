#!/usr/bin/env python3
"""tools/include_graph.py takes direct edges from the preprocessor (#341).

A tiny tree where a text scan and a naive -H tree go wrong: an include that
an include guard skips, a quoted name that exists both next to the includer
and on the -I path, an include in an inactive #if branch, a system header.
Needs clang; skipped without it.
"""
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("include_graph", ROOT / "tools" / "include_graph.py")
ig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ig)

FILES = {
    "src/base/heap.h": "#ifndef HEAP_H\n#define HEAP_H\n#include <stddef.h>\n#endif\n",
    "src/base/util.h": "#pragma once\n#include \"heap.h\"\n",
    "src/engine/heap.h": "#pragma once\n",  # shadows src/base/heap.h for engine/
    "src/engine/model.c": (
        "#include \"util.h\"\n"      # -> src/base/util.h -> src/base/heap.h
        "#include \"heap.h\"\n"      # next to the includer: src/engine/heap.h
        "#include <heap.h>\n"        # angle: -I only -> src/base/heap.h, guarded
        "#if defined(NOT_SET)\n#include \"missing_on_purpose.h\"\n#endif\n"
        "#include <stdio.h>\n"
    ),
}


class LayerOf(unittest.TestCase):
    def test_layers(self):
        self.assertEqual(ig.layer_of("src/base/heap.h"), "src/base")
        self.assertEqual(ig.layer_of("src/backends/cpu_neon/kernels/q4_K.c"), "src/backends/cpu_neon")
        self.assertEqual(ig.layer_of("src/archs/transformer/forward/step.c"), "src/archs/transformer")
        self.assertEqual(ig.layer_of("src/formats/gguf/q8_0.c"), "src/formats/gguf")
        self.assertEqual(ig.layer_of("include/geist.h"), "include")
        self.assertEqual(ig.layer_of("third_party/stb/stb_image.h"), "third_party/stb")

    def test_include_dirs_follow_common_mk(self):
        dirs = ig.include_dirs()
        self.assertEqual(dirs[:2], ["include", "."])
        self.assertIn("src/base", dirs)


@unittest.skipUnless(shutil.which("clang"), "needs clang")
class Direct(unittest.TestCase):
    def test_edges(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stub:
            for p, text in FILES.items():
                (Path(tmp) / p).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / p).write_text(text)
            saved = ig.ROOT
            ig.ROOT = Path(tmp)
            try:
                cfg = ig.CONFIGS["linux-arm64"]
                idirs = ["src/base"]
                names = ig.missing("clang", cfg, idirs, "src/engine/model.c")
                self.assertEqual(names, {"stdio.h", "stddef.h"})
                for n in names:
                    (Path(stub) / n).touch()
                edges = sorted(set(ig.direct("clang", cfg, idirs, stub, "src/engine/model.c")))
            finally:
                ig.ROOT = saved
        self.assertEqual(edges, [
            ("src/base/util.h", "src/base/heap.h"),
            ("src/engine/model.c", "src/base/heap.h"),   # skipped by the guard, still an edge
            ("src/engine/model.c", "src/base/util.h"),
            ("src/engine/model.c", "src/engine/heap.h"),
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
