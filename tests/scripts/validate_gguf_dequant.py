#!/usr/bin/env python3
"""validate_gguf_dequant.py — bit-exact parity of geist's dequant kernels
against gguf-py's reference implementation.

Run via `make test-dequant`, or directly:

    python3 tests/scripts/validate_gguf_dequant.py [model.gguf ...]

With no arguments it checks every GGUF in gguf_artifacts/. For each file it
picks one tensor per distinct dtype and compares geist's FP32 output against
gguf-py's, element for element.

Deriving the cases from the file rather than hardcoding tensor names is
deliberate: a fixed list goes stale when a re-quantised model changes a
tensor's dtype, and then reports its own staleness as a kernel failure.

Exit codes follow tests/README.md: 0 PASS, 77 SKIPPED, 99 harness error.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP, ERROR = 77, 99

# Per-tensor comparison budget in elements (~16 MB of FP32 per side). Large
# enough that a broken kernel cannot hide, small enough to run on a Pi.
ELEM_BUDGET = 4_000_000


def find_binary():
    """Locate test_gguf_dequant_unit for any built target, newest first."""
    cands = sorted(
        ROOT.glob("bin/*/*/tests/test_gguf_dequant_unit"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def cases_from(reader, path):
    """One tensor per distinct dtype — the smallest of each, to stay quick."""
    by_dtype = {}
    for t in reader.tensors:
        prev = by_dtype.get(t.tensor_type)
        if prev is None or t.n_elements < prev.n_elements:
            by_dtype[t.tensor_type] = t
    return [(path, t.name, t.tensor_type) for t in by_dtype.values()]


def row_budget(tensor):
    """How many leading rows to compare, and the row length.

    An embedding table dequantises to tens of GB of FP32 — gemma-4-E2B's
    per_layer_token_embd is 2.3e9 elements, which neither a Pi nor a CI runner
    can hold, on either side of the comparison. A prefix of full rows exercises
    the same kernels; 0 rows means "the whole tensor fits, take it all".
    """
    row_elems = int(tensor.shape[0])
    rows = int(tensor.n_elements) // row_elems if row_elems else 0
    capped = max(1, ELEM_BUDGET // row_elems) if row_elems else rows
    return (0 if capped >= rows else capped), row_elems


def reference(gguf, np, tensor, dtype, want_rows):
    """gguf-py's view of the same bits, as FP32 — sliced before dequantising,
    so the reference never materialises more than we asked the engine for."""
    data = tensor.data[:want_rows] if want_rows else tensor.data
    if dtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return data.astype(np.float32).reshape(-1)
    if dtype == gguf.GGMLQuantizationType.BF16:
        # gguf-py hands BF16 back as raw bytes and has no dequant for it.
        # BF16 is the top 16 bits of an FP32, so widen and shift.
        halves = data.view(np.uint16).reshape(-1).astype(np.uint32)
        return (halves << 16).view(np.float32)
    return gguf.quants.dequantize(data, dtype).astype(np.float32).reshape(-1)


def main():
    try:
        import gguf
        import numpy as np
    except ImportError as e:
        print(f"SKIP: {e.name} not installed (pip install gguf numpy)")
        return SKIP

    binary = find_binary()
    if binary is None:
        print("SKIP: test_gguf_dequant_unit not built — run `make bin` first")
        return SKIP

    models = [Path(a) for a in sys.argv[1:]] or sorted(
        (ROOT / "gguf_artifacts").glob("*.gguf")
    )
    models = [m for m in models if m.is_file()]
    if not models:
        print("SKIP: no GGUF found — run `make fetch-model`")
        return SKIP

    failures = checked = 0
    for model in models:
        try:
            rd = gguf.GGUFReader(str(model))
        except Exception as e:
            # gguf-py rejects dtypes it doesn't know — notably geist's ternary
            # i2_s (type 36). No reference means no comparison, not a failure.
            print(f"=== {model.name}\n  skip whole file: gguf-py cannot parse it — {e}")
            continue

        print(f"=== {model.name}")
        for path, name, dtype in cases_from(rd, model):
            tname = str(dtype).split(".")[-1]
            tensor = next(x for x in rd.tensors if x.name == name)
            want_rows, row_elems = row_budget(tensor)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "out.bin"
                proc = subprocess.run(
                    [str(binary), str(path), name, str(out), str(want_rows)],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    why = (proc.stderr.strip().splitlines() or ["no output"])[-1]
                    print(f"  FAIL {name} ({tname}): {why}")
                    failures += 1
                    continue
                ours = np.frombuffer(out.read_bytes(), dtype=np.float32)

            try:
                ref = reference(gguf, np, tensor, dtype, want_rows)
            except Exception as e:
                print(f"  skip {name} ({tname}): gguf-py cannot decode it — {e}")
                continue

            checked += 1
            if ours.shape != ref.shape:
                print(f"  FAIL {name} ({tname}): {ours.shape} vs {ref.shape}")
                failures += 1
                continue
            # Both sides decode the same bits, so equality is exact. A tolerance
            # here would hide a real kernel divergence.
            bad = int((ours != ref).sum())
            if bad:
                worst = float(np.abs(ours - ref).max())
                print(f"  FAIL {name} ({tname}): {bad}/{ref.size} differ, max|Δ|={worst:.3e}")
                failures += 1
            else:
                scope = f"{want_rows} of {int(tensor.n_elements)//row_elems} rows" if want_rows \
                    else f"{ref.size} elems"
                print(f"  ok   {name} ({tname}, {scope}) bit-exact")

    if not checked:
        print("SKIP: no tensor was comparable on both sides")
        return SKIP
    print(f"{'FAIL' if failures else 'PASS'}: {checked} tensors checked, {failures} mismatched")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
