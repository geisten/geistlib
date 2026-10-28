"""gguf_write.py — the GGUF v3 encoding shared by the tools that write one
(gen_synth_gguf.py, convert_bitnet_embedding.py): metadata key/values, tensor
infos and the aligned header. Little-endian, as src/io/gguf_reader.c reads it.
"""
from __future__ import annotations

import struct

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32  # general.alignment, the GGUF default (src/io/gguf_reader.c)
VT_U32, VT_I32, VT_F32, VT_BOOL, VT_STRING, VT_ARRAY = 4, 5, 6, 7, 8, 9


def gstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k: str, v: int) -> bytes:
    return gstr(k) + struct.pack("<II", VT_U32, v)


def kv_f32(k: str, v: float) -> bytes:
    return gstr(k) + struct.pack("<If", VT_F32, v)


def kv_bool(k: str, v: bool) -> bytes:
    return gstr(k) + struct.pack("<I?", VT_BOOL, v)


def kv_str(k: str, v: str) -> bytes:
    return gstr(k) + struct.pack("<I", VT_STRING) + gstr(v)


def kv_str_array(k: str, items: list[str]) -> bytes:
    return (gstr(k) + struct.pack("<IIQ", VT_ARRAY, VT_STRING, len(items))
            + b"".join(gstr(s) for s in items))


def kv_i32_array(k: str, items: list[int]) -> bytes:
    return (gstr(k) + struct.pack("<IIQ", VT_ARRAY, VT_I32, len(items))
            + struct.pack(f"<{len(items)}i", *items))


def tensor_info(name: str, dims, ggml_type: int, offset: int) -> bytes:
    """dims in GGUF order (fastest-varying first); offset relative to the data section."""
    return (gstr(name) + struct.pack(f"<I{len(dims)}Q", len(dims), *dims)
            + struct.pack("<IQ", ggml_type, offset))


def aligned(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def header(n_tensors: int, meta: list[bytes], infos: list[bytes]) -> bytes:
    """Magic, version, counts, metadata and tensor infos, zero-padded so the
    data section starts aligned."""
    out = (struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, n_tensors, len(meta))
           + b"".join(meta) + b"".join(infos))
    return out + b"\0" * (-len(out) % ALIGN)
