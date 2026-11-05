#!/usr/bin/env python3
"""tools/c23_metrics.py counts tokens, not text (#341).

Each case is C source where a text search miscounts: a construct inside a
comment or string literal, a line splice, a `default:` that belongs to
_Generic or to an inner switch, a cast next to a call. Then the tree has to stay within
the ceilings of benchmark/c23_metrics.json, and two runs have to agree.
"""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("c23_metrics", ROOT / "tools" / "c23_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def count(src):
    return metrics.count(metrics.lex(src))


class Lexer(unittest.TestCase):
    def test_comments_and_literals_do_not_count(self):
        m = count(
            '/* nullptr goto void * */\n'
            '// switch (x) { default: }\n'
            'const char *s = "goto nullptr (void *) __attribute__";\n'
            "char c = '\"'; int x = 1; /* \" */ void *p = nullptr;\n"
        )
        self.assertEqual(m["goto"], 0)
        self.assertEqual(m["switch"], 0)
        self.assertEqual(m["__attribute__"], 0)
        self.assertEqual(m["void_ptr"], 1)
        self.assertEqual(m["nullptr"], 1)

    def test_line_comment_continues_over_a_splice(self):
        m = count("// a comment \\\n goto x;\nint y = 0;\n")
        self.assertEqual(m["goto"], 0)

    def test_switch_default(self):
        m = count(
            "int f(int x) {\n"
            "    switch (x) {\n"  # outer: no default of its own
            "    case 0:\n"
            "        switch (x + 1) { case 1: break; default: break; }\n"
            "        return _Generic(x, int: 1, default: 2);\n"
            "    case 1: { return 3; }\n"
            "    }\n"
            "    switch (x) case 4: return 4;\n"  # no braces
            "    switch (x) { case 5: return 5; default: return 6; }\n"
            "}\n"
        )
        self.assertEqual(m["switch"], 4)
        self.assertEqual(m["switch_with_default"], 2)

    def test_enums(self):
        m = count(
            "enum a { A0 };\n"
            "enum b : uint32_t { B0 };\n"
            "enum { C0 = 4 };\n"
            "typedef enum { D0 } d_t;\n"
            "enum a x;\n"  # a use, not a definition
            "struct s { enum a f : 3; };\n"
        )
        self.assertEqual(m["enum_named"], 2)
        self.assertEqual(m["enum_named_unfixed"], 1)
        self.assertEqual(m["enum_anonymous"], 1)

    def test_defines(self):
        m = count(
            "#ifndef H\n"
            "#define H\n"  # include guard: no replacement
            "#define N 4\n"
            "#define LONG \\\n    (1 + 2)\n"
            "#define F(x) ((x) + 1)\n"
            "#define G (x)\n"  # object-like: a space before the parenthesis
            "#if N\n#elif 1\n#endif\n"
            "#endif\n"
        )
        self.assertEqual(m["define_object"], 3)
        self.assertEqual(m["define_function"], 1)
        self.assertEqual(m["pp_conditional"], 2)
        self.assertEqual(m["pp_elif"], 1)

    def test_casts(self):
        m = count(
            "void f(void *);\n"  # declarator
            "void g(void) {\n"
            "    f((void *) p);\n"
            "    f((const void *) p);\n"
            "    size_t n = sizeof(void *);\n"
            "    safe_free((void **) &p);\n"
            "    void (*cb)(void *) = 0;\n"
            "    return (void *) p;\n"
            "}\n"
        )
        self.assertEqual(m["void_ptr_cast"], 3)
        self.assertEqual(m["void_pp_cast"], 1)
        self.assertEqual(m["void_ptr"], 7)

    def test_attributes(self):
        m = count(
            "[[nodiscard]] int f(int n, int a[static n]);\n"
            "[[gnu::always_inline]] static inline int g(void);\n"
            "__attribute__((unused)) static int h;\n"
            "_Static_assert(1, \"\"); static_assert(1);\n"
            "int x = __builtin_popcount(3); bool o = ckd_add(&x, x, 1);\n"
        )
        self.assertEqual(
            [m[k] for k in ("nodiscard", "static_array_param", "std_attribute",
                            "__attribute__", "_Static_assert", "static_assert",
                            "__builtin", "ckd")],
            [1, 1, 1, 1, 1, 1, 1, 1],
        )


class Committed(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "c23_metrics.py"), *args],
            cwd=ROOT, capture_output=True, text=True,
        )

    def test_committed_file_is_current(self):
        r = self.run_tool("check")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_ratchet_fails_only_on_growth(self):
        totals = dict.fromkeys(metrics.GAPS, 5)
        ceiling = {k: {"max": 5} for k in metrics.GAPS}
        self.assertEqual(metrics.ratchet(ceiling, totals), ([], []))
        ceiling["goto"]["max"] = 4
        ceiling["void_ptr"]["max"] = 6
        self.assertEqual(metrics.ratchet(ceiling, totals), (["goto"], ["void_ptr"]))
        del ceiling["goto"]
        self.assertEqual(metrics.ratchet(ceiling, totals)[0], ["goto"])

    def test_write_only_raises(self):
        totals = dict.fromkeys(metrics.GAPS, 5)
        ceiling = {k: {"max": 9} for k in metrics.GAPS}
        ceiling["goto"]["max"] = 3
        new = metrics.raised(ceiling, totals)
        self.assertEqual(new["goto"]["max"], 5)       # exceeded: raised
        self.assertEqual(new["void_ptr"]["max"], 9)   # not exceeded: kept, not lowered

    def test_two_runs_agree(self):
        a = self.run_tool("show", "--json")
        b = self.run_tool("show", "--json")
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(a.stdout, b.stdout)
        self.assertGreater(json.loads(a.stdout)["totals"]["files"], 0)


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=1))
