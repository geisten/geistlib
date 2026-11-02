#!/usr/bin/env python3
"""Tests of tools/gguf_merge_experts.py on small GGUFs written here: the
experts of a projection end up back to back without the padding between
them, other tensors and the metadata stay as they were. Stdlib only."""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gguf_merge_experts.py"
F32 = 0


def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors):
    """tensors: (name, dims, bytes), F32. One string key, 32-byte alignment."""
    kv = gstr("general.architecture") + struct.pack("<I", 8) + gstr("llama")
    out = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), 1) + kv
    off, blobs = 0, b""
    for name, dims, data in tensors:
        out += gstr(name) + struct.pack("<I", len(dims))
        out += struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", F32, off)
        pad = b"\0" * (-len(data) % 32)
        blobs += data + pad
        off += len(data) + len(pad)
    out += b"\0" * (-len(out) % 32)
    path.write_bytes(out + blobs)


def read_gguf(path):
    b = path.read_bytes()
    n_tensors, n_kv = struct.unpack_from("<QQ", b, 8)
    assert n_kv == 1
    p = 24
    p += 8 + struct.unpack_from("<Q", b, p)[0] + 4
    arch_len = struct.unpack_from("<Q", b, p)[0]
    arch = b[p + 8 : p + 8 + arch_len].decode()
    p += 8 + arch_len
    infos = []
    for _ in range(n_tensors):
        n = struct.unpack_from("<Q", b, p)[0]
        name = b[p + 8 : p + 8 + n].decode()
        p += 8 + n
        nd = struct.unpack_from("<I", b, p)[0]
        dims = struct.unpack_from(f"<{nd}Q", b, p + 4)
        p += 4 + 8 * nd
        dtype, off = struct.unpack_from("<IQ", b, p)
        p += 12
        infos.append((name, dims, dtype, off))
    p += -p % 32
    return arch, infos, b[p:]


def floats(*v):
    return struct.pack(f"<{len(v)}f", *v)


def merge(src, dst):
    return subprocess.run(
        [sys.executable, "-I", str(TOOL), str(src), str(dst)], capture_output=True, text=True
    )


def main():
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "a.gguf", Path(d) / "b.gguf"
        tail = floats(9, 9, 9, 9, 9)
        # 2x2 F32: 16 bytes, so 16 of padding after each expert in the source.
        write_gguf(src, [("token_embd.weight", (5,), tail),
                         ("blk.0.ffn_gate.0.weight", (2, 2), floats(1, 2, 3, 4)),
                         ("blk.0.ffn_up.0.weight", (2, 2), floats(5, 6, 7, 8)),
                         ("blk.0.ffn_gate.1.weight", (2, 2), floats(10, 20, 30, 40)),
                         ("blk.0.ffn_up.1.weight", (2, 2), floats(50, 60, 70, 80)),
                         ("blk.0.ffn_gate_inp.weight", (2, 2), floats(0, 1, 0, 1))])
        r = merge(src, dst)
        assert r.returncode == 0, r.stderr
        arch, infos, data = read_gguf(dst)
        assert arch == "llama"
        by = {n: (dims, t, off) for n, dims, t, off in infos}
        assert [n for n, *_ in infos] == ["token_embd.weight", "blk.0.ffn_gate_exps.weight",
                                          "blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_inp.weight"]
        assert by["blk.0.ffn_gate_exps.weight"][0] == (2, 2, 2)
        off = by["blk.0.ffn_gate_exps.weight"][2]
        assert data[off : off + 32] == floats(1, 2, 3, 4, 10, 20, 30, 40)
        off = by["blk.0.ffn_up_exps.weight"][2]
        assert data[off : off + 32] == floats(5, 6, 7, 8, 50, 60, 70, 80)
        off = by["blk.0.ffn_gate_inp.weight"][2]
        assert data[off : off + 16] == floats(0, 1, 0, 1)
        assert data[:20] == tail
        assert all(off % 32 == 0 for *_, off in infos)

        # A merged file has nothing to do, and a gap in the experts is refused.
        assert merge(dst, Path(d) / "c.gguf").returncode != 0
        write_gguf(src, [("blk.0.ffn_gate.0.weight", (2, 2), floats(1, 2, 3, 4)),
                         ("blk.0.ffn_gate.2.weight", (2, 2), floats(5, 6, 7, 8))])
        assert merge(src, dst).returncode != 0
    print("gguf_merge_experts: ok")


if __name__ == "__main__":
    main()
