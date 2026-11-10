#!/usr/bin/env python3
"""Tests of the expert selection in tools/moe_cold_requant.py (#247): usage
lines that stop at the last expert seen are padded, cold experts are the
least used per layer on the profile set (ties by id), the random control
keeps the count and is reproducible. The ggml part needs a llama.cpp build
and is not tested here. Stdlib only."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import moe_cold_requant as mcr  # noqa: E402

USAGE = """\
# header lines are ignored
USAGE m profile a.txt 10 0 5,0,3,1
USAGE m profile b.txt 10 0 1,0,0
USAGE m profile a.txt 10 1 0,9,9,2
USAGE m heldout c.txt 10 0 4,4,1,1
USAGE m heldout c.txt 10 1 1,1,1,1
USAGE other profile a.txt 10 0 9,9,9,9
"""


def main():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "usage.txt"
        p.write_text(USAGE)
        u = mcr.load_usage(str(p), "m", 2, 4)
    assert u["profile"][0] == [6, 0, 3, 1], u["profile"][0]  # padded, summed over files
    assert u["profile"][1] == [0, 9, 9, 2]
    assert u["heldout"][0] == [4, 4, 1, 1]

    n, cold = mcr.pick_cold(u, 2, 4, 0.5, "least")
    assert n == 2
    assert cold == [{1, 3}, {0, 3}], cold

    n, none = mcr.pick_cold(u, 2, 4, 0.0, "least")
    assert n == 0 and none == [set(), set()]

    _, r1 = mcr.pick_cold(u, 2, 4, 0.5, "random")
    _, r2 = mcr.pick_cold(u, 2, 4, 0.5, "random")
    assert r1 == r2 and all(len(c) == 2 for c in r1)
    print("moe_cold_requant: ok")


if __name__ == "__main__":
    main()
