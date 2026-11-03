#!/usr/bin/env python3
"""tests/test_ppl_vs_llama_py.py — ppl_vs_llama.py's BOS guard (#400).

geist's add-BOS decision (eval_geist BOS) must match llama.cpp's
(llama-tokenize of a probe with and without --no-bos); a mismatch skews
geist's perplexity silently, so the harness aborts. Hermetic: no model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from ppl_vs_llama import bos_mismatch  # noqa: E402

fails = 0


def expect(cond: bool, label: str) -> None:
    global fails
    print(f"  {'✓' if cond else '✗'} {label}")
    fails += not cond


expect(bos_mismatch(128000, [128000, 64], [64]) is None, "both add BOS 128000: agree")
expect(bos_mismatch(None, [64], [64]) is None, "neither adds BOS: agree")
why = bos_mismatch(128000, [64], [64]) or ""
expect("geist adds BOS 128000" in why and "llama.cpp adds no BOS" in why, "geist adds, llama.cpp not: abort")
why = bos_mismatch(None, [128000, 64], [64]) or ""
expect("geist adds no BOS" in why and "llama.cpp adds BOS 128000" in why, "llama.cpp adds, geist not: abort")
expect(bos_mismatch(1, [128000, 64], [64]) is not None, "different BOS ids: abort")

print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
