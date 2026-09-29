#!/usr/bin/env python3
"""gen_synth_gguf.py — hermetic synthetic Llama-family GGUF for perf baselines.

Writes a structurally valid `general.architecture = llama` GGUF with random
quantized weights of a chosen geometry, so the decode/prefill hot path can be
timed without a model download (the reference GGUFs are 0.4-3.1 GB, and a
container or CI runner whose egress policy blocks huggingface.co cannot fetch
them at all).

What the numbers mean, and what they do not:
  - GEMV/GEMM, attention and KV-cache timings are data-independent (integer
    kernels, fixed trip counts), so they transfer to a real model of the same
    geometry and quantization.
  - The weights are noise, so the outputs are meaningless and the activation
    distribution is not a real model's. Anything whose cost depends on the
    values — expf/tanhf slow paths, sparsity, early exits — is NOT
    representative. Never quote quality from these files.

Stdlib only (runs on a bare Pi). Block payloads come from random.randbytes and
each block's fp16 scale is patched in with extended-slice assignment — small,
valid scales, so no NaN/Inf reaches a kernel — which writes a 1B-parameter
file in seconds. Deterministic for a given --seed.

Usage:
  gen_synth_gguf.py OUT.gguf --preset llama32-1b-q4_k
  gen_synth_gguf.py OUT.gguf --layers 2 --d-model 256 --heads 4 --kv-heads 2 \\
      --ffn 17408 --vocab 4096 --wtype q4_k --embd-type q6_k
  gen_synth_gguf.py OUT.gguf --layers 1 --d-model 256 --heads 4 --kv-heads 2 \\
      --ffn 512 --wtype q8_0 --spm-tokenizer      # tokenizer benchmarks
"""
from __future__ import annotations

import argparse
import random
import struct
import sys

ALIGN = 32  # general.alignment, the GGUF default (src/io/gguf_reader.c)

# GGUF value types (src/io/gguf_reader.h)
VT_U32, VT_F32, VT_STRING, VT_ARRAY = 4, 6, 8, 9

# qtype -> (ggml type id, elements per block, bytes per block, byte offsets of
# the fp16 scales inside a block). Layouts follow src/quant/quant.h.
QTYPES = {
    "f32": (0, 1, 4, ()),
    "f16": (1, 1, 2, ()),
    "q4_0": (2, 32, 18, (0,)),
    "q8_0": (8, 32, 34, (0,)),
    "q4_k": (12, 256, 144, (0, 2)),  # d, dmin
    "q6_k": (14, 256, 210, (208,)),  # d sits at the tail
}
SCALES = (0.004, 0.002)  # d, then dmin where a format has one

PRESETS = {
    # Geometry of HuggingFaceTB/SmolLM2-360M-Instruct, the CI Llama fixture.
    "smollm2-360m-q8_0": dict(layers=32, d_model=960, heads=15, kv_heads=5, ffn=2560,
                              vocab=49152, wtype="q8_0", embd_type="q8_0"),
    # Geometry of Llama-3.2-1B, Q4_K_M-like: Q4_K body, Q6_K tied embedding.
    "llama32-1b-q4_k": dict(layers=16, d_model=2048, heads=32, kv_heads=8, ffn=8192,
                            vocab=128256, wtype="q4_k", embd_type="q6_k"),
}


def gstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k: str, v: int) -> bytes:
    return gstr(k) + struct.pack("<II", VT_U32, v)


def kv_f32(k: str, v: float) -> bytes:
    return gstr(k) + struct.pack("<If", VT_F32, v)


def kv_str(k: str, v: str) -> bytes:
    return gstr(k) + struct.pack("<I", VT_STRING) + gstr(v)


def kv_str_array(k: str, items: list[str]) -> bytes:
    return (gstr(k) + struct.pack("<IIQ", VT_ARRAY, VT_STRING, len(items))
            + b"".join(gstr(s) for s in items))


def spm_tokenizer() -> tuple[list[str], list[str]]:
    """A tiny SentencePiece-style vocab: specials, <0xXX> byte fallback, single
    characters and bigram merges. Enough to drive the SPM merge loop; the
    engine selects SPM for tokenizer.ggml.model != "gpt2" with merges."""
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    tokens += ["▁"] + [chr(c) for c in range(ord("a"), ord("z") + 1)] + [".", ","]
    merges = []
    for p in ("▁t", "he", "▁a", "in", "er", "an", "re", "on", "▁s", "en", "at",
              "or", "es", "▁o", "te", "ed", "is", "it", "ar", "al", "le", "▁w", "nd"):
        merges.append(f"{p[0]} {p[1]}")
        tokens.append(p)
    return tokens, merges


def tensor_bytes(rng: random.Random, qt: str, n_elems: int, fill: float | None) -> bytes:
    _, block_elems, block_bytes, scale_offs = QTYPES[qt]
    if n_elems % block_elems:
        raise SystemExit(f"{qt}: {n_elems} elements is not a multiple of the {block_elems}-element block")
    if qt in ("f32", "f16"):
        fmt, width = ("<f", 4) if qt == "f32" else ("<e", 2)
        if fill is not None:
            return struct.pack(fmt, fill) * n_elems
        # Small random values without numpy: tile a random 1024-element pattern.
        pattern = b"".join(struct.pack(fmt, rng.uniform(-0.05, 0.05)) for _ in range(1024))
        return (pattern * ((n_elems + 1023) // 1024))[: n_elems * width]
    n_blocks = n_elems // block_elems
    buf = bytearray(rng.randbytes(n_blocks * block_bytes))
    for off, value in zip(scale_offs, SCALES):
        lo, hi = struct.pack("<e", value)
        buf[off::block_bytes] = bytes([lo]) * n_blocks
        buf[off + 1::block_bytes] = bytes([hi]) * n_blocks
    return bytes(buf)


def build(cfg: dict, seed: int, spm: bool) -> tuple[list[bytes], list[tuple]]:
    """Returns (metadata KV records, [(name, dims fastest-first, qtype, fill)])."""
    n_layers, d, n_heads, n_kv, ffn = (cfg[k] for k in ("layers", "d_model", "heads", "kv_heads", "ffn"))
    if d % n_heads:
        raise SystemExit(f"d_model {d} is not a multiple of heads {n_heads}")
    hd = d // n_heads
    wt, et = cfg["wtype"], cfg["embd_type"]
    tok_meta: list[bytes] = []
    vocab = cfg["vocab"]
    if spm:
        tokens, merges = spm_tokenizer()
        vocab = len(tokens)
        tok_meta = [kv_str("tokenizer.ggml.model", "llama"),
                    kv_str_array("tokenizer.ggml.tokens", tokens),
                    kv_str_array("tokenizer.ggml.merges", merges),
                    kv_u32("tokenizer.ggml.bos_token_id", 1),
                    kv_u32("tokenizer.ggml.eos_token_id", 2),
                    kv_u32("tokenizer.ggml.unknown_token_id", 0)]

    tensors = [("token_embd.weight", (d, vocab), et, None),
               ("output_norm.weight", (d,), "f32", 1.0)]
    for i in range(n_layers):
        p = f"blk.{i}."
        tensors += [(p + "attn_norm.weight", (d,), "f32", 1.0),
                    (p + "attn_q.weight", (d, n_heads * hd), wt, None),
                    (p + "attn_k.weight", (d, n_kv * hd), wt, None),
                    (p + "attn_v.weight", (d, n_kv * hd), wt, None),
                    (p + "attn_output.weight", (n_heads * hd, d), wt, None),
                    (p + "ffn_norm.weight", (d,), "f32", 1.0),
                    (p + "ffn_gate.weight", (d, ffn), wt, None),
                    (p + "ffn_up.weight", (d, ffn), wt, None),
                    (p + "ffn_down.weight", (ffn, d), wt, None)]

    meta = [kv_str("general.architecture", "llama"),
            kv_str("general.name", f"synthetic-llama-{n_layers}x{d}-{wt}-seed{seed}"),
            kv_u32("general.alignment", ALIGN),
            kv_u32("llama.block_count", n_layers),
            kv_u32("llama.embedding_length", d),
            kv_u32("llama.feed_forward_length", ffn),
            kv_u32("llama.attention.head_count", n_heads),
            kv_u32("llama.attention.head_count_kv", n_kv),
            kv_u32("llama.vocab_size", vocab),
            kv_u32("llama.context_length", 8192),
            kv_u32("llama.rope.dimension_count", hd),
            kv_f32("llama.rope.freq_base", 10000.0),
            kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)] + tok_meta
    return meta, tensors


def write_gguf(path: str, meta: list[bytes], tensors: list[tuple], seed: int) -> int:
    rng = random.Random(seed)
    infos, sizes, offset = [], [], 0
    for name, dims, qt, _ in tensors:
        n = 1
        for x in dims:
            n *= x
        type_id, block_elems, block_bytes, _ = QTYPES[qt]
        nbytes = n // block_elems * block_bytes
        infos.append(gstr(name) + struct.pack("<I", len(dims))
                     + b"".join(struct.pack("<Q", x) for x in dims)
                     + struct.pack("<IQ", type_id, offset))
        sizes.append((n, nbytes))
        offset += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    header = (struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(meta))
              + b"".join(meta) + b"".join(infos))
    header += b"\0" * (-len(header) % ALIGN)
    total = 0
    with open(path, "wb") as f:
        f.write(header)
        for (name, _, qt, fill), (n, nbytes) in zip(tensors, sizes):
            data = tensor_bytes(rng, qt, n, fill)
            assert len(data) == nbytes, (name, len(data), nbytes)
            f.write(data)
            f.write(b"\0" * (-len(data) % ALIGN))
            total += n
    return total


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int)
    ap.add_argument("--d-model", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--ffn", type=int)
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--wtype", choices=sorted(QTYPES))
    ap.add_argument("--embd-type", choices=sorted(QTYPES))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--spm-tokenizer", action="store_true",
                    help="embed a tiny SentencePiece tokenizer (vocab = its size)")
    a = ap.parse_args(argv)

    cfg = dict(PRESETS[a.preset]) if a.preset else {}
    for k in ("layers", "d_model", "heads", "kv_heads", "ffn", "vocab", "wtype", "embd_type"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    required = ["layers", "d_model", "heads", "kv_heads", "ffn", "wtype"]
    if not a.spm_tokenizer:
        required.append("vocab")
    missing = [k for k in required if k not in cfg]
    if missing:
        ap.error(f"missing geometry (use --preset or pass): {', '.join(missing)}")
    cfg.setdefault("embd_type", cfg["wtype"])
    cfg.setdefault("vocab", 0)

    meta, tensors = build(cfg, a.seed, a.spm_tokenizer)
    total = write_gguf(a.out, meta, tensors, a.seed)
    print(f"wrote {a.out}: {len(tensors)} tensors, {total / 1e6:.1f} M params, "
          f"layers={cfg['layers']} d={cfg['d_model']} heads={cfg['heads']}/{cfg['kv_heads']} "
          f"ffn={cfg['ffn']} w={cfg['wtype']} embd={cfg['embd_type']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
