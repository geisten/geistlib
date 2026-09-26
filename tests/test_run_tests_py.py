#!/usr/bin/env python3
"""mk/run-tests.sh: the expected-skip allowlist and the step-summary table.

Hermetic — fakes a bin dir with two-line shell scripts standing in for test
binaries, so the runner's own verdict logic is what gets checked.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "mk" / "run-tests.sh"


def fake_bin(root: Path, name: str, rc: int, line: str) -> None:
    p = root / name
    p.write_text(f"#!/bin/sh\necho '{line}'\nexit {rc}\n")
    p.chmod(0o755)


class RunTestsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = Path(self.tmp.name) / "bin"
        self.bin.mkdir()
        fake_bin(self.bin, "test_ok_unit", 0, "cosine 0.9999")
        fake_bin(self.bin, "test_listed_unit", 77, "SKIP: no fixture")
        fake_bin(self.bin, "test_unlisted_unit", 77, "SKIP: tower | missing")
        self.allow = Path(self.tmp.name) / "allow.txt"
        self.allow.write_text("# comment\ntest_listed_unit\n")

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, **env) -> subprocess.CompletedProcess:
        full = {**os.environ, **env}
        full.pop("GITHUB_STEP_SUMMARY", None)
        full.update(env)
        return subprocess.run(["sh", str(RUNNER), str(self.bin)], capture_output=True, text=True, env=full)

    def test_skips_pass_without_an_allowlist(self):
        r = self.run_it()
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("2 skipped", r.stdout)

    def test_unlisted_skip_fails_listed_skip_passes(self):
        r = self.run_it(GEIST_EXPECTED_SKIPS=str(self.allow))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("unexpected skip", r.stdout)
        self.assertIn("test_unlisted_unit", r.stdout)
        self.assertIn("1 skipped, 1 failed", r.stdout)

    def test_missing_allowlist_is_a_harness_error(self):
        r = self.run_it(GEIST_EXPECTED_SKIPS=str(self.allow) + ".nope")
        self.assertEqual(r.returncode, 99)

    def test_step_summary_gets_the_skip_table(self):
        summary = Path(self.tmp.name) / "summary.md"
        r = self.run_it(GITHUB_STEP_SUMMARY=str(summary))
        self.assertEqual(r.returncode, 0, r.stdout)
        text = summary.read_text()
        self.assertIn("1 passed, 2 skipped", text)
        self.assertIn("| test_listed_unit | SKIP: no fixture |", text)
        self.assertIn(r"SKIP: tower \| missing", text)  # pipes escaped for the table


if __name__ == "__main__":
    unittest.main()
