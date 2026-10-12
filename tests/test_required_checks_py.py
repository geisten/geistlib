#!/usr/bin/env python3
"""tests/test_required_checks_py.py — the required-status-checks list.

The stdlib parser in scripts/required_checks.py must expand matrix include
entries into status names, not mistake step names for matrix entries, fall
back to the job key without a name, and agree with the repository's own
.gitea/workflows/ci.yml and .gitea/required-checks.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import required_checks as rc  # noqa: E402

fails = 0


def expect(cond: bool, label: str) -> None:
    global fails
    print(f"  {'✓' if cond else '✗'} {label}")
    fails += not cond


WORKFLOW = """name: CI
on:
  pull_request:
jobs:
  plain:
    name: plain job # a comment
    runs-on: x
    steps:
      - name: not a matrix entry
        run: true
  build:
    name: build (${{ matrix.name }})
    strategy:
      matrix:
        include:
          - name: Linux arm64
            runner: a
            flags: "x"
          # an Intel leg, commented
          - runner: b
            name: "macOS arm64"
    steps:
      - name: also not a matrix entry
  docs:
    name: documented build (${{ matrix.image }})
    strategy:
      matrix:
        include:
          - image: debian:13
          - image: alpine:3.21
  unnamed:
    runs-on: x
"""

got = rc.expected(WORKFLOW)
expect(got == sorted([
    "CI / plain job (pull_request)",
    "CI / build (Linux arm64) (pull_request)",
    "CI / build (macOS arm64) (pull_request)",
    "CI / documented build (debian:13) (pull_request)",
    "CI / documented build (alpine:3.21) (pull_request)",
    "CI / unnamed (pull_request)",
]), "matrix entries expand, steps and comments are ignored, a job without name uses its key")

real = rc.expected(rc.WORKFLOW.read_text())
expect(len(real) == len(set(real)) and len(real) >= 20, "the real workflow yields distinct names")
expect(real == rc.listed(), ".gitea/required-checks.txt matches .gitea/workflows/ci.yml")

print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
