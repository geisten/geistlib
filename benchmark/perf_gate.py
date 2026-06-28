#!/usr/bin/env python3
"""perf_gate.py — read a bench_perf_sweep JSON line from stdin and exit non-zero
if prefill/decode tok/s fall below floors.

A COARSE cliff-detector for CI: shared cloud runners are noisy (±~20%), so the
floors are set well below the real number — this catches gross regressions
(scalar fallback, -O0 sneaking in, OpenMP off, a kernel returning early), not
subtle ones. The Raspberry Pi 5 stays the precise arbiter (benchmark/BENCHMARK_PI5.md).

  bench_perf_sweep --gguf M --seq-lens 128 --decode-n 16 ... | \
    python3 benchmark/perf_gate.py --min-prefill 20 --min-decode 4
  python3 benchmark/perf_gate.py --selftest
"""
import argparse
import json
import sys


def gate(line: str, min_prefill: float, min_decode: float) -> str | None:
    """Return an error string if below a floor, else None."""
    d = json.loads(line)
    pp, tg = float(d["prefill_tps"]), float(d["decode_tps"])
    print(f"perf: prefill={pp:.1f} tok/s (floor {min_prefill}), "
          f"decode={tg:.1f} tok/s (floor {min_decode})")
    bad = []
    if pp < min_prefill:
        bad.append(f"prefill {pp:.1f} < {min_prefill}")
    if tg < min_decode:
        bad.append(f"decode {tg:.1f} < {min_decode}")
    return "; ".join(bad) if bad else None


def _selftest() -> None:
    ok = '{"prefill_tps": 40.0, "decode_tps": 8.0}'
    assert gate(ok, 20, 4) is None
    assert gate('{"prefill_tps": 5.0, "decode_tps": 8.0}', 20, 4) is not None   # prefill cliff
    assert gate('{"prefill_tps": 40.0, "decode_tps": 1.0}', 20, 4) is not None  # decode cliff
    print("perf_gate selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prefill", type=float, default=0.0)
    ap.add_argument("--min-decode", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()

    line = ""
    for ln in sys.stdin:
        ln = ln.strip()
        if ln.startswith("{"):
            line = ln  # last JSON line wins (one seq-len)
    if not line:
        sys.exit("perf_gate: no JSON line on stdin (did bench_perf_sweep run?)")
    err = gate(line, args.min_prefill, args.min_decode)
    if err:
        sys.exit(f"PERF REGRESSION: {err}")
    print("perf gate: ok")


if __name__ == "__main__":
    main()
