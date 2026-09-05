#!/usr/bin/env python3
"""convert_bitnet_embedding.py — BitNet embedding models (safetensors) -> GGUF.

Converts microsoft/BitNet-embedding-0.6B (Qwen3 backbone) and
microsoft/BitNet-embedding-270M (Gemma3 backbone) into a GGUF that
geistlib's reader accepts, with ternary weights packed as I2_S.

Why not Microsoft's converter: theirs (utils/convert-bitnet-embedding-to-gguf.py)
only runs against a pinned llama.cpp branch and pulls in torch. This one needs
numpy and the standard library, nothing else — the safetensors header is JSON
and the GGUF container is written here directly.

The I2_S layout is NOT the naive "four consecutive values per byte" the upstream
guide's prose suggests. It is strided, and the packing below mirrors, byte for
byte, geistlib's own reference implementations:

  - pack:   tests/test_i2_s_parity.c:30 (pack_i2_s)
  - unpack: src/backends/cpu_scalar/weight_resolve.c:109
  - extent: src/quant/quant.h:162 (I2_S_BLOCK_ELEMS/BYTES, i2_s_scale_offset)

Element b*256 + h*128 + g*32 + bb lives in byte b*64 + h*32 + bb at shift
6-2g, trits are stored as trit+1 (so -1,0,+1 -> 0,1,2), and ONE f32 scale for
the whole tensor sits immediately after the packed bytes, at offset n_elems/4.
That scale is a MULTIPLIER: dequant is trit * scale, so the value written is
mean(|w|), not its reciprocal.

Usage:
    python3 tools/convert_bitnet_embedding.py MODEL_DIR \\
        --outtype i2_s --outfile bitnet-embedding-0.6b-i2_s.gguf

Memory: tensors are streamed one at a time out of an mmapped safetensors file
and written straight through, so peak RSS tracks the largest single tensor,
not the model.
"""
from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

import numpy as np

# ---- GGUF constants ------------------------------------------------------
# Metadata value types and ggml tensor types, mirroring src/io/gguf_reader.c.

VT_U32, VT_F32, VT_BOOL, VT_STRING, VT_ARRAY = 4, 6, 7, 8, 9

GGML_F32, GGML_F16, GGML_I2_S = 0, 1, 36

GGUF_VERSION = 3
GGUF_ALIGNMENT = 32  # the spec default geistlib assumes (gguf_reader.c:367)

# I2_S blocking, from src/quant/quant.h:162.
I2S_BLOCK_ELEMS = 256
I2S_BLOCK_BYTES = 64


# ---- safetensors ---------------------------------------------------------

ST_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64}


class StTensor(NamedTuple):
    """One tensor located inside an mmapped safetensors file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


def read_st_index(model_dir: Path) -> list[Path]:
    """Shard files for a model dir, in index order (single-file models included)."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        # dict preserves insertion order; dict.fromkeys dedupes while keeping it.
        return [model_dir / s for s in dict.fromkeys(weight_map.values())]
    single = model_dir / "model.safetensors"
    if not single.is_file():
        raise SystemExit(f"no model.safetensors or shard index in {model_dir}")
    return [single]


def read_st_header(path: Path) -> tuple[list[StTensor], int]:
    """Parse a safetensors header. Returns (tensors, data_start)."""
    with path.open("rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    data_start = 8 + header_len
    tensors = [
        StTensor(name, meta["dtype"], tuple(meta["shape"]), *meta["data_offsets"])
        for name, meta in header.items()
        if name != "__metadata__"
    ]
    return tensors, data_start


def st_array(mm: mmap.mmap, data_start: int, t: StTensor) -> np.ndarray:
    """Zero-copy view of one tensor's bytes. BF16 comes back as uint16."""
    if t.dtype not in ST_DTYPES:
        raise SystemExit(f"{t.name}: unsupported safetensors dtype {t.dtype}")
    buf = memoryview(mm)[data_start + t.begin : data_start + t.end]
    return np.frombuffer(buf, dtype=ST_DTYPES[t.dtype]).reshape(t.shape)


def to_f32(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Widen to float32. BF16 arrives as uint16 and is shifted into place."""
    if dtype == "BF16":
        return (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.astype(np.float32, copy=False)


def to_f16(arr_f32: np.ndarray) -> np.ndarray:
    """Narrow to float16, refusing to lose a value silently (AGENT.md §5)."""
    out = arr_f32.astype(np.float16)
    if not np.isfinite(out[np.isfinite(arr_f32)]).all():
        raise SystemExit("f16 conversion overflowed: value outside float16 range")
    return out


# ---- I2_S ----------------------------------------------------------------


def quantize_i2s(w_f32: np.ndarray) -> tuple[np.ndarray, float]:
    """Pack a 2-D [n_out, n_in] ternary weight as I2_S.

    Returns (packed_bytes, scale) with scale = mean(|w|), the multiplier
    geistlib's decoder applies. Rows are packed independently, n_in/4 bytes
    each; the caller appends the f32 scale after the last row.
    """
    n_out, n_in = w_f32.shape
    if n_in % I2S_BLOCK_ELEMS:
        raise SystemExit(
            f"I2_S needs in-features divisible by {I2S_BLOCK_ELEMS}, got {n_in}. "
            "geistlib's decoder walks whole 256-element blocks "
            "(src/backends/cpu_scalar/weight_resolve.c:109), so a remainder "
            "would be silently dropped.\n"
            "This is what blocks BitNet-embedding-270M: its hidden size is 640 "
            "and 640 % 256 = 128. Until geistlib grows a 128-granular I2_S "
            "path, convert that model with --outtype f16 — it loads and runs, "
            "it just does not get the ternary win."
        )

    scale = float(np.abs(w_f32).mean())
    if not scale > 0.0:
        raise SystemExit("all-zero tensor: scale would be 0 and dequant undefined")

    # Ternarize, then store trit+1 so {-1,0,+1} -> {0,1,2}.
    codes = (np.rint(w_f32 / scale).clip(-1, 1) + 1).astype(np.uint8)

    # Element index within a block is h*128 + g*32 + bb, so viewing each row as
    # (blocks, h=2, g=4, bb=32) puts the four values that share a byte on axis
    # g -- exactly the strided grouping pack_i2_s() builds with its loops.
    grouped = codes.reshape(n_out, n_in // I2S_BLOCK_ELEMS, 2, 4, 32)
    packed = np.zeros(grouped.shape[:3] + (32,), dtype=np.uint8)
    for g in range(4):
        packed |= grouped[:, :, :, g, :] << (6 - 2 * g)
    return packed.reshape(n_out, n_in // 4), scale


def unpack_i2s(packed: np.ndarray, n_in: int, scale: float) -> np.ndarray:
    """Inverse of quantize_i2s -- a NumPy mirror of geistlib's scalar decoder.

    Only used by the self-test, but it is the thing that proves the packing:
    it is written from the C decoder, not from quantize_i2s.
    """
    n_out = packed.shape[0]
    qs = packed.reshape(n_out, n_in // I2S_BLOCK_ELEMS, 2, 32)
    trits = np.empty((n_out, n_in // I2S_BLOCK_ELEMS, 2, 4, 32), dtype=np.int8)
    for g in range(4):
        trits[:, :, :, g, :] = ((qs >> (6 - 2 * g)) & 3).astype(np.int8) - 1
    return trits.reshape(n_out, n_in).astype(np.float32) * scale


# ---- tensor name mapping -------------------------------------------------

# Per-layer HF suffix -> GGUF suffix. Shared by both backbones; the
# *_norm_in entries are the BitNet per-projection input norms that no
# standard architecture has (see docs/BITNET_EMBEDDINGS_PLAN.md §1).
LAYER_MAP_COMMON = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.q_proj.norm.weight": "attn_q_norm_in.weight",
    "self_attn.k_proj.norm.weight": "attn_k_norm_in.weight",
    "self_attn.v_proj.norm.weight": "attn_v_norm_in.weight",
    "self_attn.o_proj.norm.weight": "attn_output_norm_in.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate_proj.norm.weight": "ffn_gate_norm_in.weight",
    "mlp.up_proj.norm.weight": "ffn_up_norm_in.weight",
    "mlp.down_proj.norm.weight": "ffn_down_norm_in.weight",
}

# post_attention_layernorm means different things in the two backbones.
LAYER_MAP_QWEN3 = LAYER_MAP_COMMON | {
    "post_attention_layernorm.weight": "ffn_norm.weight",
}
LAYER_MAP_GEMMA3 = LAYER_MAP_COMMON | {
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "pre_feedforward_layernorm.weight": "ffn_norm.weight",
    "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
}

TOP_MAP = {
    "embed_tokens.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
}


def map_tensor_name(hf_name: str, layer_map: dict[str, str]) -> str | None:
    """HF tensor name -> GGUF name, or None for tensors we deliberately drop."""
    name = hf_name.removeprefix("model.")
    if name.startswith("lm_head.") or name == "output.weight":
        return None  # embedding models have no LM head
    if name in TOP_MAP:
        return TOP_MAP[name]
    if name.startswith("layers."):
        _, idx, suffix = name.split(".", 2)
        mapped = layer_map.get(suffix)
        return f"blk.{idx}.{mapped}" if mapped else None
    return None


# ---- GGUF writing --------------------------------------------------------


def _kv(key: str, vt: int, payload: bytes) -> bytes:
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", vt) + payload


def kv_u32(key: str, v: int) -> bytes:
    return _kv(key, VT_U32, struct.pack("<I", v))


def kv_f32(key: str, v: float) -> bytes:
    return _kv(key, VT_F32, struct.pack("<f", v))


def kv_bool(key: str, v: bool) -> bytes:
    return _kv(key, VT_BOOL, struct.pack("<B", 1 if v else 0))


def kv_str(key: str, v: str) -> bytes:
    b = v.encode()
    return _kv(key, VT_STRING, struct.pack("<Q", len(b)) + b)


def kv_str_array(key: str, items: list[str]) -> bytes:
    body = b"".join(struct.pack("<Q", len(b)) + b for b in (s.encode() for s in items))
    return _kv(key, VT_ARRAY, struct.pack("<IQ", VT_STRING, len(items)) + body)


def kv_i32_array(key: str, items: list[int]) -> bytes:
    body = struct.pack(f"<{len(items)}i", *items)
    return _kv(key, VT_ARRAY, struct.pack("<IQ", 5, len(items)) + body)


class OutTensor(NamedTuple):
    """A tensor planned for the output file. `emit` produces its bytes."""

    name: str
    dims: tuple[int, ...]  # GGUF order: fastest-varying first
    ggml_type: int
    nbytes: int
    emit: Callable[[], bytes]


def tensor_info(t: OutTensor, offset: int) -> bytes:
    name = t.name.encode()
    return (
        struct.pack("<Q", len(name))
        + name
        + struct.pack("<I", len(t.dims))
        + struct.pack(f"<{len(t.dims)}Q", *t.dims)
        + struct.pack("<IQ", t.ggml_type, offset)
    )


def write_gguf(path: Path, metadata: list[bytes], tensors: list[OutTensor]) -> None:
    """Write header + tensor infos, then stream each tensor's data."""
    header = struct.pack("<IIQQ", 0x46554747, GGUF_VERSION, len(tensors), len(metadata))
    body = header + b"".join(metadata)

    infos, offset = [], 0
    for t in tensors:
        infos.append(tensor_info(t, offset))
        offset += (t.nbytes + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT
    body += b"".join(infos)

    pad = -len(body) % GGUF_ALIGNMENT
    with path.open("wb") as out:
        out.write(body + b"\0" * pad)
        for t in tensors:
            data = t.emit()
            if len(data) != t.nbytes:
                raise SystemExit(f"{t.name}: planned {t.nbytes} bytes, got {len(data)}")
            out.write(data)
            out.write(b"\0" * (-len(data) % GGUF_ALIGNMENT))


# ---- conversion ----------------------------------------------------------


def plan_tensor(
    gguf_name: str,
    st: StTensor,
    load: Callable[[], np.ndarray],
    ternary: bool,
) -> OutTensor:
    """Decide a tensor's output dtype and how to produce its bytes.

    2-D projection weights go to I2_S in ternary mode; embeddings and all
    1-D norms stay F16, matching the upstream tensor-type table.
    """
    shape = st.shape
    dims = tuple(reversed(shape))  # GGUF stores fastest-varying first

    if ternary and len(shape) == 2:
        n_out, n_in = shape
        nbytes = n_out * n_in // 4 + 4  # packed trits + one f32 tensor scale

        def emit_i2s() -> bytes:
            packed, scale = quantize_i2s(load())
            return packed.tobytes() + struct.pack("<f", scale)

        return OutTensor(gguf_name, dims, GGML_I2_S, nbytes, emit_i2s)

    nbytes = int(np.prod(shape)) * 2
    return OutTensor(gguf_name, dims, GGML_F16, nbytes, lambda: to_f16(load()).tobytes())


def build_metadata(cfg: dict, arch: str, tok: dict, n_tensors: int) -> list[bytes]:
    """GGUF metadata, using the keys geistlib's populators actually read."""
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hidden = cfg["hidden_size"]
    # head_dim is NOT hidden/heads for either model -- the default derivation
    # would be wrong, so key_length/value_length are written explicitly.
    head_dim = cfg.get("head_dim") or hidden // n_heads

    meta = [
        kv_str("general.architecture", arch),
        kv_str("general.name", cfg.get("_name_or_path", "bitnet-embedding")),
        kv_u32(f"{arch}.block_count", cfg["num_hidden_layers"]),
        kv_u32(f"{arch}.context_length", cfg.get("max_position_embeddings", 32768)),
        kv_u32(f"{arch}.embedding_length", hidden),
        kv_u32(f"{arch}.feed_forward_length", cfg["intermediate_size"]),
        kv_u32(f"{arch}.attention.head_count", n_heads),
        kv_u32(f"{arch}.attention.head_count_kv", n_kv),
        kv_u32(f"{arch}.attention.key_length", head_dim),
        kv_u32(f"{arch}.attention.value_length", head_dim),
        kv_f32(f"{arch}.attention.layer_norm_rms_epsilon", cfg.get("rms_norm_eps", 1e-6)),
        kv_f32(f"{arch}.rope.freq_base", cfg.get("rope_theta", 10000.0)),
        kv_u32(f"{arch}.vocab_size", cfg["vocab_size"]),
        kv_bool("bitnet.embedding.projection_input_norms", True),
        kv_str("bitnet.embedding.pooling", "last_token"),
    ]
    if arch == "gemma3":
        meta.append(kv_u32("gemma3.attention.query_pre_attn_scalar",
                           cfg.get("query_pre_attn_scalar", head_dim)))
    meta += tokenizer_metadata(tok, cfg)
    return meta


def tokenizer_metadata(tok: dict, cfg: dict) -> list[bytes]:
    """Vocab, merges and special ids from a HF tokenizer.json (BPE)."""
    if not tok:
        return []
    model = tok.get("model", {})
    vocab = model.get("vocab", {})
    tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    merges = [" ".join(m) if isinstance(m, list) else m for m in model.get("merges", [])]

    # 1 = normal, 3 = control. Added tokens are the control ones.
    added = {a["id"] for a in tok.get("added_tokens", [])}
    types = [3 if i in added else 1 for i in range(len(tokens))]

    meta = [
        kv_str("tokenizer.ggml.model", "gpt2"),
        kv_str_array("tokenizer.ggml.tokens", tokens),
        kv_i32_array("tokenizer.ggml.token_type", types),
        kv_str_array("tokenizer.ggml.merges", merges),
    ]
    for key, cfg_key in (("bos_token_id", "bos_token_id"), ("eos_token_id", "eos_token_id")):
        if (tid := cfg.get(cfg_key)) is not None:
            meta.append(kv_u32(f"tokenizer.ggml.{key}", int(tid)))
    return meta


def convert(model_dir: Path, outfile: Path, outtype: str) -> None:
    cfg = json.loads((model_dir / "config.json").read_text())
    model_type = cfg.get("model_type", "")
    if model_type == "qwen3":
        arch, layer_map = "qwen3", LAYER_MAP_QWEN3
    elif model_type == "gemma3_text":
        arch, layer_map = "gemma3", LAYER_MAP_GEMMA3
    else:
        raise SystemExit(f"unsupported model_type {model_type!r} (need qwen3/gemma3_text)")

    tok_path = model_dir / "tokenizer.json"
    tok = json.loads(tok_path.read_text()) if tok_path.is_file() else {}
    ternary = outtype == "i2_s"

    shards = read_st_index(model_dir)
    planned: list[OutTensor] = []
    open_files = []  # kept alive for the duration of the write

    for shard in shards:
        fh = shard.open("rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        open_files.append((fh, mm))
        st_tensors, data_start = read_st_header(shard)
        for st in sorted(st_tensors, key=lambda t: t.name):
            gguf_name = map_tensor_name(st.name, layer_map)
            if gguf_name is None:
                continue
            # Late binding is the point: the array is materialised inside
            # emit(), one tensor at a time, not here.
            loader = (lambda m=mm, d=data_start, t=st: to_f32(st_array(m, d, t), t.dtype))
            is_proj = gguf_name.endswith(".weight") and len(st.shape) == 2
            is_embd = gguf_name == "token_embd.weight"
            planned.append(
                plan_tensor(gguf_name, st, loader, ternary and is_proj and not is_embd)
            )

    if not planned:
        raise SystemExit("no tensors mapped -- is this a BitNet embedding checkpoint?")

    meta = build_metadata(cfg, arch, tok, len(planned))
    write_gguf(outfile, meta, planned)
    for fh, mm in open_files:
        mm.close()
        fh.close()

    n_i2s = sum(t.ggml_type == GGML_I2_S for t in planned)
    print(f"wrote {outfile} — {len(planned)} tensors ({n_i2s} I2_S), arch={arch}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir", type=Path, help="HF checkpoint directory")
    ap.add_argument("--outtype", choices=("i2_s", "f16"), default="i2_s")
    ap.add_argument("--outfile", type=Path, required=True)
    args = ap.parse_args(argv)
    convert(args.model_dir, args.outfile, args.outtype)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
