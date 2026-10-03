#!/usr/bin/env python3
"""gen_synth_gguf.py — hermetic synthetic GGUF for perf baselines.

Writes a structurally valid `general.architecture = llama` GGUF with random
quantized weights of a chosen geometry, so the decode/prefill hot path can be
timed without a model download (the reference GGUFs are 0.4-3.1 GB, and a
container or CI runner whose egress policy blocks huggingface.co cannot fetch
them at all). The preset bonsai2-27b-pq2_0 writes a `qwen35` file instead:
Ternary-Bonsai-2-27B's geometry, tensor formats (PQ2_0, BF16 DeltaNet
alpha/beta) and prism.hadamard keys, to the parameter and byte, 7.2 GB.

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
  gen_synth_gguf.py OUT.gguf --preset bonsai2-27b-pq2_0 [--layers 8]
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
VT_U32, VT_I32, VT_F32, VT_BOOL, VT_STRING, VT_ARRAY = 4, 5, 6, 7, 8, 9

# qtype -> (ggml type id, elements per block, bytes per block, byte offsets of
# the fp16 scales inside a block). Layouts follow src/quant/quant.h.
QTYPES = {
    "f32": (0, 1, 4, ()),
    "f16": (1, 1, 2, ()),
    "q4_0": (2, 32, 18, (0,)),
    "q8_0": (8, 32, 34, (0,)),
    "q3_k": (11, 256, 110, (108,)),  # d sits at the tail
    "q4_k": (12, 256, 144, (0, 2)),  # d, dmin
    "q5_k": (13, 256, 176, (0, 2)),  # d, dmin
    "q6_k": (14, 256, 210, (208,)),  # d sits at the tail
    "bf16": (30, 1, 2, ()),
    "pq2_0": (142, 128, 34, (0,)),  # PrismML's ternary format, d first
    "tq2_0": (35, 256, 66, (64,)),  # ternary BitNet, d at the tail
}
SCALES = (0.004, 0.002)  # d, then dmin where a format has one

PRESETS = {
    # Geometry of HuggingFaceTB/SmolLM2-360M-Instruct, the CI Llama fixture.
    "smollm2-360m-q8_0": dict(layers=32, d_model=960, heads=15, kv_heads=5, ffn=2560,
                              vocab=49152, wtype="q8_0", embd_type="q8_0"),
    # Geometry of Llama-3.2-1B, Q4_K_M-like: Q4_K body, Q6_K tied embedding.
    "llama32-1b-q4_k": dict(layers=16, d_model=2048, heads=32, kv_heads=8, ffn=8192,
                            vocab=128256, wtype="q4_k", embd_type="q6_k"),
    # Geometry of PrismML's Ternary-Bonsai-2-27B (Qwen3.8-27B): a qwen35 hybrid,
    # 48 gated-DeltaNet and 16 attention layers, PQ2_0 everywhere but the
    # norms (F32) and the DeltaNet alpha/beta projections (BF16), weights in a
    # blockwise Walsh-Hadamard basis. 26,895,998,464 parameters and
    # 7,195,047,936 tensor bytes, as in Ternary-Bonsai-2-27B-PQ2_0.gguf.
    "bonsai2-27b-pq2_0": dict(arch="qwen35", layers=64, d_model=5120, heads=24, kv_heads=4,
                              head_dim=256, rope_dims=64, ffn=17408, vocab=248320,
                              dn_k_heads=16, dn_v_heads=48, dn_head_dim=128, conv=4,
                              interval=4, wtype="pq2_0", embd_type="pq2_0",
                              ab_type="bf16", hadamard_block=1024),
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


def kv_bool(k: str, v: bool) -> bytes:
    return gstr(k) + struct.pack("<I?", VT_BOOL, v)


def kv_i32_array(k: str, items: list[int]) -> bytes:
    return (gstr(k) + struct.pack("<IIQ", VT_ARRAY, VT_I32, len(items))
            + struct.pack(f"<{len(items)}i", *items))


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


# PQ2_0 bytes whose four 2-bit codes are all 0..2 (-1, 0, +1): the real
# model is strictly ternary and never uses code 3 (+2).
PQ2_0_TERNARY = bytes(b for b in range(256) if all((b >> (2 * k)) & 3 != 3 for k in range(4)))
PQ2_0_TABLE = bytes(PQ2_0_TERNARY[i % len(PQ2_0_TERNARY)] for i in range(256))
CHUNK_BLOCKS = 1 << 18  # blocks drawn at a time: randbytes is limited to 2^31 bits


def tensor_bytes(rng: random.Random, qt: str, n_elems: int, fill: float | None):
    """The tensor's payload, in chunks of at most CHUNK_BLOCKS blocks."""
    _, block_elems, block_bytes, scale_offs = QTYPES[qt]
    if n_elems % block_elems:
        raise SystemExit(f"{qt}: {n_elems} elements is not a multiple of the {block_elems}-element block")
    if qt in ("f32", "f16", "bf16"):
        fmt, width = {"f32": ("<f", 4), "f16": ("<e", 2), "bf16": ("<f", 2)}[qt]
        # bf16 is a float32's upper half.
        pack = (lambda v: struct.pack("<f", v)[2:]) if qt == "bf16" else (lambda v: struct.pack(fmt, v))
        if fill is not None:
            yield pack(fill) * n_elems
            return
        # Small random values without numpy: tile a random 1024-element pattern.
        pattern = b"".join(pack(rng.uniform(-0.05, 0.05)) for _ in range(1024))
        yield (pattern * ((n_elems + 1023) // 1024))[: n_elems * width]
        return
    left = n_elems // block_elems
    while left:
        n_blocks = min(left, CHUNK_BLOCKS)
        raw = rng.randbytes(n_blocks * block_bytes)
        buf = bytearray(raw.translate(PQ2_0_TABLE) if qt == "pq2_0" else raw)
        for off, value in zip(scale_offs, SCALES):
            lo, hi = struct.pack("<e", value)
            buf[off::block_bytes] = bytes([lo]) * n_blocks
            buf[off + 1::block_bytes] = bytes([hi]) * n_blocks
        yield bytes(buf)
        left -= n_blocks


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


def build_qwen35(cfg: dict, seed: int) -> tuple[list[bytes], list[tuple]]:
    """A qwen35 (gated-DeltaNet hybrid) model, laid out as the loader reads it
    (src/archs/transformer/arch_family.c, weight_load/layer_wiring.c): layer i
    is attention when (i + 1) % interval == 0, a DeltaNet otherwise. With
    hadamard_block set, the prism.hadamard keys of a Ternary-Bonsai file
    (src/archs/transformer/rotation.c) with random signs, the grouped-value
    permutation on, and token_embd as the inverse-rotated tensor."""
    n_layers, d, ffn, vocab = cfg["layers"], cfg["d_model"], cfg["ffn"], cfg["vocab"]
    n_q, n_kv, hd, interval = cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["interval"]
    nk, nv, dh = cfg["dn_k_heads"], cfg["dn_v_heads"], cfg["dn_head_dim"]
    key_dim, val_dim, q_out = nk * dh, nv * dh, n_q * hd
    conv_dim = 2 * key_dim + val_dim
    wt, et, ab = cfg["wtype"], cfg["embd_type"], cfg.get("ab_type", cfg["wtype"])

    tensors = [("token_embd.weight", (d, vocab), et, None),
               ("output_norm.weight", (d,), "f32", 1.0),
               ("output.weight", (d, vocab), wt, None)]
    rotated = []
    for i in range(n_layers):
        p = f"blk.{i}."
        tensors += [(p + "attn_norm.weight", (d,), "f32", 1.0),
                    (p + "post_attention_norm.weight", (d,), "f32", 1.0)]
        if (i + 1) % interval == 0:
            mixer = [(p + "attn_q.weight", (d, 2 * q_out), wt, None),  # query | gate per head
                     (p + "attn_k.weight", (d, n_kv * hd), wt, None),
                     (p + "attn_v.weight", (d, n_kv * hd), wt, None),
                     (p + "attn_output.weight", (q_out, d), wt, None)]
            tensors += mixer + [(p + "attn_q_norm.weight", (hd,), "f32", 1.0),
                                (p + "attn_k_norm.weight", (hd,), "f32", 1.0)]
        else:
            mixer = [(p + "attn_qkv.weight", (d, conv_dim), wt, None),
                     (p + "attn_gate.weight", (d, val_dim), wt, None),
                     (p + "ssm_out.weight", (val_dim, d), wt, None)]
            tensors += mixer[:2] + [(p + "ssm_conv1d.weight", (cfg["conv"], conv_dim), "f32", None),
                                    (p + "ssm_beta.weight", (d, nv), ab, None),
                                    (p + "ssm_alpha.weight", (d, nv), ab, None),
                                    (p + "ssm_a", (nv,), "f32", -1.0),  # g = ssm_a * softplus(.) <= 0
                                    (p + "ssm_dt.bias", (nv,), "f32", 0.0),
                                    (p + "ssm_norm.weight", (dh,), "f32", 1.0)] + mixer[2:]
        ffn_ts = [(p + "ffn_gate.weight", (d, ffn), wt, None),
                  (p + "ffn_up.weight", (d, ffn), wt, None),
                  (p + "ffn_down.weight", (ffn, d), wt, None)]
        tensors += ffn_ts
        rotated += mixer + ffn_ts
    rotated.append(tensors[2])  # output.weight

    meta = [kv_str("general.architecture", "qwen35"),
            kv_str("general.name", f"synthetic-qwen35-{n_layers}x{d}-{wt}-seed{seed}"),
            kv_u32("general.alignment", ALIGN),
            kv_u32("qwen35.block_count", n_layers),
            kv_u32("qwen35.embedding_length", d),
            kv_u32("qwen35.feed_forward_length", ffn),
            kv_u32("qwen35.attention.head_count", n_q),
            kv_u32("qwen35.attention.head_count_kv", n_kv),
            kv_u32("qwen35.attention.key_length", hd),
            kv_u32("qwen35.rope.dimension_count", cfg["rope_dims"]),
            kv_u32("qwen35.full_attention_interval", interval),
            kv_u32("qwen35.ssm.group_count", nk),
            kv_u32("qwen35.ssm.time_step_rank", nv),
            kv_u32("qwen35.ssm.state_size", dh),
            kv_u32("qwen35.ssm.inner_size", val_dim),
            kv_u32("qwen35.ssm.conv_kernel", cfg["conv"]),
            kv_f32("qwen35.rope.freq_base", 1e7),
            kv_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
            # The loader takes the vocabulary size from the token count.
            kv_str("tokenizer.ggml.model", "gpt2"),
            kv_str("tokenizer.ggml.pre", "qwen35"),
            kv_str_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(vocab)])]
    block = cfg.get("hadamard_block")
    if block:
        # One sign vector per rotated input width, as rotation.c looks them up.
        widths = sorted({dims[0] for _, dims, _, _ in rotated})
        rng = random.Random(seed ^ 0x5157)
        meta += [kv_u32("prism.hadamard.version", 1),
                 kv_str("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard"),
                 kv_str("prism.hadamard.axis", "input-last-dimension"),
                 kv_str("prism.hadamard.sign_mode", "explicit"),
                 kv_u32("prism.hadamard.block_size", block),
                 kv_bool("prism.hadamard.gdn_v_grouped", True),
                 kv_str_array("prism.hadamard.weight_names", [name for name, _, _, _ in rotated]),
                 kv_str_array("prism.hadamard.inverse_weight_names", ["token_embd.weight"]),
                 kv_i32_array("prism.hadamard.sign_widths", widths),
                 kv_i32_array("prism.hadamard.sign_values",
                              [rng.choice((-1, 1)) for _ in range(sum(widths))])]
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
            written = 0
            for chunk in tensor_bytes(rng, qt, n, fill):
                f.write(chunk)
                written += len(chunk)
            assert written == nbytes, (name, written, nbytes)
            f.write(b"\0" * (-written % ALIGN))
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
    if cfg.get("arch") == "qwen35":
        # Its geometry is the preset's; only the depth may change.
        fixed = [k for k in ("d_model", "heads", "kv_heads", "ffn", "vocab", "wtype", "embd_type")
                 if getattr(a, k) is not None]
        if fixed or a.spm_tokenizer:
            ap.error(f"--preset {a.preset} takes only --layers and --seed")
        if a.layers is not None:
            cfg["layers"] = a.layers
        meta, tensors = build_qwen35(cfg, a.seed)
        total = write_gguf(a.out, meta, tensors, a.seed)
        print(f"wrote {a.out}: {len(tensors)} tensors, {total / 1e6:.1f} M params, "
              f"qwen35 layers={cfg['layers']} d={cfg['d_model']} w={cfg['wtype']}")
        return 0
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
