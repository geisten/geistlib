#!/usr/bin/env python3
"""Hermetic checks for tools/gen_synth_gguf.py.

The generator exists so perf baselines need no download; a malformed file
would make every number taken with it meaningless, so the structure is pinned
here with an independent minimal GGUF reader: header, metadata, tensor table,
alignment, payload sizes, valid fp16 block scales, and seed determinism.
"""
import contextlib
import hashlib
import importlib.util
import io
import math
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
        if vt in (4, 5):
            (meta[key],) = struct.unpack_from("<I" if vt == 4 else "<i", data, p)
            p += 4
        elif vt == 6:
            (meta[key],) = struct.unpack_from("<f", data, p)
            p += 4
        elif vt == 7:
            meta[key] = data[p] != 0
            p += 1
        elif vt == 8:
            meta[key] = string()
        elif vt == 9:
            et, n = struct.unpack_from("<IQ", data, p)
            p += 12
            if et == 5:
                meta[key] = list(struct.unpack_from(f"<{n}i", data, p))
                p += 4 * n
            else:
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
            if cfg.get("arch") != "qwen35":  # qwen35 sets its head width apart
                self.assertEqual(cfg["d_model"] % cfg["heads"], 0, name)
            self.assertEqual(cfg["heads"] % cfg["kv_heads"], 0, name)
            self.assertIn(cfg["wtype"], gen.QTYPES)
            self.assertIn(cfg["embd_type"], gen.QTYPES)

    def test_bonsai_preset_is_the_real_file_to_the_byte(self):
        # Ternary-Bonsai-2-27B-PQ2_0.gguf as llama-bench reports it (model_n_params,
        # model_size in benchmark/results/raw/*bonsai2-27b-pq2*.jsonl): 851 tensors.
        _, tensors = gen.build_qwen35(gen.PRESETS["bonsai2-27b-pq2_0"], 1234)
        params = sum(math.prod(dims) for _, dims, _, _ in tensors)
        size = sum(math.prod(dims) // gen.QTYPES[qt][1] * gen.QTYPES[qt][2]
                   for _, dims, qt, _ in tensors)
        self.assertEqual((len(tensors), params, size), (851, 26_895_998_464, 7_195_047_936))

    def test_qwen35_preset_takes_only_its_depth(self):
        # A silently ignored --d-model would time another model than asked for.
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            out = Path(d) / "x.gguf"
            for extra in (["--d-model", "64"], ["--wtype", "q8_0"], ["--spm-tokenizer"]):
                with self.assertRaises(SystemExit):
                    gen.main([str(out), "--preset", "bonsai2-27b-pq2_0"] + extra)
            self.assertFalse(out.exists())

    def test_qwen35_structure_rotation_and_ternary_blocks(self):
        cfg = dict(arch="qwen35", layers=4, d_model=1024, heads=4, kv_heads=1, head_dim=256,
                   rope_dims=64, ffn=2048, vocab=1000, dn_k_heads=2, dn_v_heads=6,
                   dn_head_dim=128, conv=4, interval=4, wtype="pq2_0", embd_type="pq2_0",
                   ab_type="bf16", hadamard_block=256)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "qwen35.gguf"
            gen.write_gguf(str(path), *gen.build_qwen35(cfg, 7), 7)
            meta, tensors, data_start, data = read_gguf(path)
        self.assertEqual(meta["general.architecture"], "qwen35")
        self.assertEqual(len(meta["tokenizer.ggml.tokens"]), 1000)
        by_name = {name: (dims, ttype, off) for name, dims, ttype, off in tensors}
        # Layers 0-2 are DeltaNet (14 tensors), layer 3 attention (11), plus 3 globals.
        self.assertEqual(len(tensors), 3 + 3 * 14 + 11)
        self.assertEqual(by_name["blk.0.attn_qkv.weight"][0], (1024, 2 * 256 + 768))
        self.assertEqual(by_name["blk.0.ssm_alpha.weight"][:2], ((1024, 6), 30))
        self.assertEqual(by_name["blk.3.attn_q.weight"][0], (1024, 2 * 4 * 256))
        names = meta["prism.hadamard.weight_names"]
        self.assertEqual(len(names), 3 * 6 + 7 + 1)
        self.assertTrue(set(names) <= set(by_name))
        self.assertEqual(meta["prism.hadamard.inverse_weight_names"], ["token_embd.weight"])
        self.assertTrue(meta["prism.hadamard.gdn_v_grouped"])
        widths = meta["prism.hadamard.sign_widths"]
        self.assertEqual(widths, sorted({by_name[n][0][0] for n in names}))
        self.assertEqual(len(meta["prism.hadamard.sign_values"]), sum(widths))
        self.assertEqual(set(meta["prism.hadamard.sign_values"]), {-1, 1})
        for w in widths:
            self.assertEqual(w % meta["prism.hadamard.block_size"], 0)
        # Every PQ2_0 block: the fixed scale, and codes 0..2 only (no +2).
        for name, (dims, ttype, off) in by_name.items():
            if ttype != 142:
                continue
            start = data_start + off
            n_blocks = math.prod(dims) // 128
            for b in range(n_blocks):
                blk = data[start + 34 * b:start + 34 * b + 34]
                self.assertAlmostEqual(struct.unpack_from("<e", blk)[0], gen.SCALES[0], places=5)
                self.assertFalse(any((q >> s) & 3 == 3 for q in blk[2:] for s in (0, 2, 4, 6)),
                                 name)


if __name__ == "__main__":
    unittest.main()
