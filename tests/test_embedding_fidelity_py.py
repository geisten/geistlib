#!/usr/bin/env python3
"""test_embedding_fidelity_py.py — the embedding parity gate and the .gemb
format contract between its two ends.

Hermetic by contract: no GGUF, no network, no model weights.

Two things are covered, and the second is the one that would break silently:

  1. tools/eval_embedding_fidelity.py's gate logic, via its own --selftest.
  2. The .gemb container. tools/dump_geist_embedding.c writes it and this
     Python reads it, so the header layout is a cross-language contract with
     nothing to enforce it at compile time. The bytes below are assembled
     exactly as write_header() lays them out — magic, then three
     little-endian uint32 — so a change on either side fails here instead of
     producing a plausible array of the wrong shape.
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from eval_embedding_fidelity import (  # noqa: E402
    DEFAULT_MIN_COSINE,
    gate,
    read_gemb,
    selftest,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else f' — {detail}'}")
    if not ok:
        FAILURES.append(name)


def make_gemb(rows: np.ndarray, *, magic: bytes = b"GEMB", version: int = 1,
              count: int | None = None, dim: int | None = None) -> bytes:
    """Assemble a .gemb exactly as dump_geist_embedding.c's write_header does."""
    n, d = rows.shape
    header = magic + struct.pack("<III", version, n if count is None else count,
                                 d if dim is None else dim)
    return header + rows.astype(np.float32).tobytes()


def test_gate_selftest() -> None:
    check("the gate's own selftest passes", selftest() == 0)


def test_gemb_round_trip() -> None:
    rng = np.random.default_rng(5)
    rows = rng.standard_normal((6, 32)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "e.gemb"
        p.write_bytes(make_gemb(rows))
        got = read_gemb(p)
        check("a .gemb round-trips to the same shape", got.shape == rows.shape, str(got.shape))
        check("a .gemb round-trips to the same values", np.array_equal(got, rows))


def test_gemb_rejects_malformed() -> None:
    rng = np.random.default_rng(6)
    rows = rng.standard_normal((3, 8)).astype(np.float32)

    cases = {
        "wrong magic": make_gemb(rows, magic=b"XXXX"),
        "future version": make_gemb(rows, version=2),
        "count larger than payload": make_gemb(rows, count=99),
        "dim larger than payload": make_gemb(rows, dim=99),
        "truncated header": b"GEMB\x01\x00",
    }
    for name, blob in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.gemb"
            p.write_bytes(blob)
            try:
                read_gemb(p)
            except SystemExit:
                check(f"rejects {name}", True)
            else:
                check(f"rejects {name}", False, "accepted silently")


def test_gate_thresholds() -> None:
    """The floor must actually be the decision boundary."""
    a = np.eye(2, 4)
    # Rotate one row just enough to land under any sane floor.
    b = a.copy()
    b[0] = [0.6, 0.8, 0.0, 0.0]
    passed_strict, _ = gate(a, b, DEFAULT_MIN_COSINE)
    passed_loose, _ = gate(a, b, 0.5)
    check("a 0.6 cosine fails the default floor", not passed_strict)
    check("the same pair passes a 0.5 floor", passed_loose)


def main() -> int:
    print("=== embedding fidelity gate ===")
    for fn in (
        test_gate_selftest,
        test_gemb_round_trip,
        test_gemb_rejects_malformed,
        test_gate_thresholds,
    ):
        fn()
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
