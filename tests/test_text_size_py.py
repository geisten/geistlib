#!/usr/bin/env python3
"""tools/text_size.py sums .text per layer and gates growth at 1 % (#341).

A fake `size -A` prints ELF and Mach-O style sections; only .text, .text.*
and __text count, and compare fails exactly when a library grows > 1 %.
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("text_size", ROOT / "tools" / "text_size.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

FAKE = r'''#!/bin/sh
shift  # -A
for f in "$@"; do
  echo "$f  :"
  echo "section size addr"
  case "$f" in
    *heap.o) echo ".text 100 0"; echo ".text.unlikely 20 0"; echo ".data 999 0" ;;
    *q4_K.o) echo "__text 300 0"; echo "__const 50 0" ;;
    *) echo ".text 7 0" ;;
  esac
  echo "Total 1"; echo
done
'''


class TextSize(unittest.TestCase):
    def test_measure_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            for p in ("src/base/heap.o", "src/backends/cpu_neon/kernels/q4_K.o",
                      "third_party/stb/stb_impl.o"):
                (t / "b" / p).parent.mkdir(parents=True, exist_ok=True)
                (t / "b" / p).write_bytes(b"")
            (t / "libgeist.so").write_bytes(b"")
            fake = t / "size"
            fake.write_text(FAKE)
            fake.chmod(0o755)
            os.environ["SIZE"] = str(fake)
            try:
                doc = ts.measure(str(t / "b"), [str(t / "libgeist.so")])
            finally:
                del os.environ["SIZE"]
        self.assertEqual(doc["layers"], {"src/backends/cpu_neon": 300, "src/base": 120,
                                         "third_party/stb": 7})
        self.assertEqual(doc["libgeist.a"], 427)
        self.assertEqual(doc["shared"], {"libgeist.so": 7})
        grown = json.loads(json.dumps(doc))
        grown["libgeist.a"] = 427 + 4  # +0.94 %
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ts.compare(doc, grown), 0)
            grown["libgeist.a"] = 427 + 5  # +1.17 %
            self.assertEqual(ts.compare(doc, grown), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
