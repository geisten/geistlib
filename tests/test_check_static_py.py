#!/usr/bin/env python3
"""test_check_static_py - the ratchet and the NOLINT rule of scripts/check_static.py (#342)."""
import collections
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import check_static  # noqa: E402

A = ("src/a.c", "bugprone-x", "m")
B = ("src/b.c", "bugprone-y", "n")


class Ratchet(unittest.TestCase):
    def test_equal_is_clean(self):
        c = collections.Counter({A: 2})
        self.assertEqual(check_static.ratchet(c, c), (collections.Counter(), collections.Counter()))

    def test_one_more_of_a_known_finding_is_new(self):
        new, gone = check_static.ratchet(collections.Counter({A: 3}), collections.Counter({A: 2}))
        self.assertEqual(new[A], 1)
        self.assertFalse(gone)

    def test_a_fixed_finding_must_leave_the_baseline(self):
        new, gone = check_static.ratchet(collections.Counter(), collections.Counter({B: 1}))
        self.assertFalse(new)
        self.assertEqual(gone[B], 1)


class Reasons(unittest.TestCase):
    def test_every_baselined_analyzer_finding_has_a_reason(self):
        scripts = os.path.dirname(check_static.BASELINE)
        for name in sorted(os.listdir(scripts)):
            if name.startswith("static-baseline"):
                base = check_static.read_baseline(os.path.join(scripts, name))
                self.assertTrue(base, name)
                self.assertEqual(check_static.unexplained(base), [], name)

    def test_an_unexplained_analyzer_finding_is_reported(self):
        key = ("src/none.c", "clang-analyzer-core.NullDereference", "m")
        self.assertEqual(check_static.unexplained({key: 1}), [key[:2]])


class Selftest(unittest.TestCase):
    FIXTURE = "int f(void) {\n    int *p = nullptr;\n    return *p; // expect: clang-analyzer-core.NullDereference\n}\n"
    MARK = ("tests/static/x.c", 3, "clang-analyzer-core.NullDereference")

    def test_marks_parse_with_their_line(self):
        self.assertEqual(check_static.expectations("tests/static/x.c", self.FIXTURE), {self.MARK})

    def test_exact_match_is_clean(self):
        self.assertEqual(check_static.selftest_diff({self.MARK}, {self.MARK}), ([], []))

    def test_a_seeded_defect_not_reported_fails(self):
        self.assertEqual(check_static.selftest_diff({self.MARK}, set()), ([self.MARK], []))

    def test_an_unmarked_defect_in_a_fixture_fails(self):
        uaf = ("tests/static/x.c", 9, "clang-analyzer-unix.Malloc")
        self.assertEqual(check_static.selftest_diff({self.MARK}, {self.MARK, uaf}), ([], [uaf]))


class Nolint(unittest.TestCase):
    def test_reason_required(self):
        self.assertTrue(check_static.nolint_ok("int x; // NOLINT(bugprone-x): the header wants it"))
        self.assertFalse(check_static.nolint_ok("int x; // NOLINT(bugprone-x)"))
        self.assertFalse(check_static.nolint_ok("int x; // NOLINT"))
        self.assertTrue(check_static.nolint_ok("int x;"))

    def test_a_finding_line_parses(self):
        m = check_static.FINDING.match("src/a.c:3:4: warning: bad thing [bugprone-x]")
        self.assertEqual((m[1], m[5], m[6]), ("src/a.c", "bad thing", "bugprone-x"))


if __name__ == "__main__":
    unittest.main()
