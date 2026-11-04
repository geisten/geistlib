#!/usr/bin/env python3
"""tools/check_layers.py proves the layer table of tools/layers.json (#344).

The selftest compiles the small tree in tests/layers/, with one violation of
each kind seeded: an upward include, a call across a layer that repeats the
prototype instead of including it, a header that does not compile alone and
portability #ifs outside a wrapper. All four must be reported and nothing
else; no binary runs. The rest checks the parts that need no compiler: the
table must be closed, docs/ARCHITECTURE.md must show it, and the #if count
and the ratchet must behave.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_layers", ROOT / "tools" / "check_layers.py")
layers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layers)


class Selftest(unittest.TestCase):
    def test_seeded_violations_are_found(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_layers.py"), "--selftest"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("4 of 4 seeded violations found, nothing else", r.stdout)


class Table(unittest.TestCase):
    def table(self, doc):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(doc, f)
        try:
            return layers.Table(Path(f.name))
        finally:
            os.unlink(f.name)

    def test_may_use_must_be_closed(self):
        # a -> b -> c without a -> c: an include of c.h in a.c behind b.h would
        # not show in the -H tree, so the table is refused
        doc = {"layers": [
            {"name": "c", "paths": ["c/"], "may_use": []},
            {"name": "b", "paths": ["b/"], "may_use": ["c"]},
            {"name": "a", "paths": ["a/"], "may_use": ["b"]},
        ]}
        with self.assertRaises(SystemExit):
            self.table(doc)
        doc["layers"][2]["may_use"] = ["b", "c"]
        t = self.table(doc)
        self.assertTrue(t.allowed("a/x.c", "c"))
        self.assertFalse(t.allowed("c/x.c", "a"))

    def test_longest_prefix_and_composition(self):
        t = layers.Table(layers.TABLE)
        self.assertEqual(t.layer_of("src/backends/common/kivi.h"), "backend_common")
        self.assertEqual(t.layer_of("src/backends/cpu_neon/backend.c"), "cpu_neon")
        self.assertIsNone(t.layer_of("tests/test_helpers.h"))
        self.assertTrue(t.allowed("src/engine/backend_registry.c", "cpu_neon"))
        self.assertFalse(t.allowed("src/engine/session.c", "cpu_neon"))
        self.assertFalse(t.allowed("src/archs/transformer/arch.c", "engine"))

    def test_architecture_md_shows_the_table(self):
        have, want = layers.doc_with_table(layers.Table(layers.TABLE))
        self.assertEqual(have, want, "run tools/check_layers.py --doc (docs/ARCHITECTURE.md)")


class PortabilityIfs(unittest.TestCase):
    def count(self, src, name="src/engine/x.c"):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.parent.mkdir(parents=True)
            p.write_text(src)
            return layers.portability_ifs(Path(d), layers.Table(layers.TABLE), [name]).get(name, 0)

    def test_reserved_names_count_once_per_directive(self):
        self.assertEqual(self.count(
            "#if defined(__APPLE__) && defined(__aarch64__)\n#elif defined(_WIN32)\n#endif\n"
            "#ifdef GEIST_BACKEND_METAL\n#endif\n"  # a project macro: not portability
            "#ifndef X_H\n#define X_H\n#endif\n"  # an include guard
            "#ifdef __cplusplus\n#endif\n"  # language linkage, not a platform
            "/* #ifdef __linux__ */\n"), 2)

    def test_wrappers_do_not_count(self):
        self.assertEqual(self.count("#ifdef __linux__\n#endif\n", "src/base/par.c"), 0)


class Ratchet(unittest.TestCase):
    def test_new_gone_and_ceilings(self):
        analysed = {"include": {"a.c"}, "if": {"b.c", "c.c"}, "symbol": set(), "header": set()}
        base = {"include a.c -> x.h", "include z.c -> x.h", "if b.c 3", "if c.c 2"}
        now = {"include a.c -> y.h", "if b.c 2", "if c.c 3"}
        new, gone = layers.compare(now, base, analysed)
        self.assertEqual(new, ["if c.c 3", "include a.c -> y.h"])
        # z.c was not analysed here: its line stands
        self.assertEqual(gone, ["if b.c 3", "if c.c 2", "include a.c -> x.h"])


if __name__ == "__main__":
    unittest.main()
