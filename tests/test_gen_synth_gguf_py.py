#!/usr/bin/env python3
"""Hermetic checks for tools/gen_synth_gguf.py.

The generator exists so perf baselines need no download; a malformed file
would make every number taken with it meaningless, so the structure is pinned
here with an independent minimal GGUF reader: header, metadata, tensor table,
alignment, payload sizes, valid fp16 block scales, and seed determinism.
"""
import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen_synth_gguf", ROOT / "tools/gen_synth_gguf.py")
gen = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gen)

TINY = ["--layers", "1", "--d-model", "256", "--heads", "4", "--kv-heads", "2",
        "--ffn", "512", "--vocab", "512", "--wtype", "q4_k", "--embd-type", "q8_0"]


def read_gguf(path: Path) -> tuple[dict, list[tuple], int, bytes]:
    data = path.read_bytes()
    magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", data, 0)
    assert magic == 0x46554747 and version == 3
    p = 24

    def string() -> str:
        nonlocal p
        (n,) = struct.unpack_from("<Q", data, p)
        s = data[p + 8:p + 8 + n].decode()
        p += 8 + n
        return s

    meta = {}
    for _ in range(n_kv):
        key = string()
        (vt,) = struct.unpack_from("<I", data, p)
        p += 4
        if vt == 4:
            (meta[key],) = struct.unpack_from("<I", data, p)
            p += 4
        elif vt == 6:
            (meta[key],) = struct.unpack_from("<f", data, p)
            p += 4
        elif vt == 8:
            meta[key] = string()
        elif vt == 9:
            et, n = struct.unpack_from("<IQ", data, p)
            p += 12
            assert et == 8
            meta[key] = [string() for _ in range(n)]
        else:
            raise AssertionError(f"unexpected value type {vt}")
    tensors = []
    for _ in range(n_tensors):
        name = string()
        (nd,) = struct.unpack_from("<I", data, p)
        dims = struct.unpack_from(f"<{nd}Q", data, p + 4)
        ttype, off = struct.unpack_from("<IQ", data, p + 4 + 8 * nd)
        p += 4 + 8 * nd + 12
        tensors.append((name, dims, ttype, off))
    data_start = (p + gen.ALIGN - 1) // gen.ALIGN * gen.ALIGN
    return meta, tensors, data_start, data


class GenSynthGgufTest(unittest.TestCase):
    def generate(self, d: str, name: str, args: list[str]) -> Path:
        out = Path(d) / name
        self.assertEqual(gen.main([str(out)] + args), 0)
        return out

    def test_structure_sizes_and_scales(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.generate(d, "tiny.gguf", TINY)
            meta, tensors, data_start, data = read_gguf(path)
            self.assertEqual(meta["general.architecture"], "llama")
            self.assertEqual(meta["llama.block_count"], 1)
            self.assertEqual(meta["llama.vocab_size"], 512)
            self.assertEqual(len(tensors), 2 + 9)
            by_type = {v[0]: (k, v) for k, v in gen.QTYPES.items()}
            end = data_start
            for name, dims, ttype, off in tensors:
                self.assertEqual(off % gen.ALIGN, 0, name)
                qt, (_, block_elems, block_bytes, scale_offs) = by_type[ttype]
                n = 1
                for x in dims:
                    n *= x
                nbytes = n // block_elems * block_bytes
                start = data_start + off
                self.assertLessEqual(start + nbytes, len(data), name)
                end = max(end, start + nbytes)
                # First and last block carry the fixed, valid fp16 scales.
                for block in (0, n // block_elems - 1):
                    for so, value in zip(scale_offs, gen.SCALES):
                        (h,) = struct.unpack_from("<e", data, start + block * block_bytes + so)
                        self.assertAlmostEqual(h, value, places=5, msg=f"{name} ({qt})")
            self.assertEqual(dict((t[0], t[1]) for t in tensors)["blk.0.ffn_down.weight"], (512, 256))
            self.assertLessEqual(len(data) - end, gen.ALIGN)

    def test_seed_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            h = [hashlib.sha256(self.generate(d, f"{i}.gguf", TINY + ["--seed", str(s)]).read_bytes()).digest()
                 for i, s in enumerate((7, 7, 8))]
            self.assertEqual(h[0], h[1])
            self.assertNotEqual(h[0], h[2])

    def test_spm_tokenizer_sets_vocab(self):
        with tempfile.TemporaryDirectory() as d:
            args = [a for a in TINY]
            del args[args.index("--vocab"):args.index("--vocab") + 2]
            meta, tensors, _, _ = read_gguf(self.generate(d, "tok.gguf", args + ["--spm-tokenizer"]))
            tokens = meta["tokenizer.ggml.tokens"]
            self.assertEqual(meta["tokenizer.ggml.model"], "llama")
            self.assertEqual(meta["llama.vocab_size"], len(tokens))
            self.assertEqual(dict((t[0], t[1]) for t in tensors)["token_embd.weight"], (256, len(tokens)))
            for merge in meta["tokenizer.ggml.merges"]:
                left, right = merge.split(" ")
                self.assertIn(left + right, tokens)

    def test_presets_are_complete(self):
        for name, cfg in gen.PRESETS.items():
            self.assertEqual(cfg["d_model"] % cfg["heads"], 0, name)
            self.assertEqual(cfg["heads"] % cfg["kv_heads"], 0, name)
            self.assertIn(cfg["wtype"], gen.QTYPES)
            self.assertIn(cfg["embd_type"], gen.QTYPES)


if __name__ == "__main__":
    unittest.main()
