#!/usr/bin/env python3
"""Rewrite a GGUF with one tensor per expert into the merged layout (#241).

    tools/gguf_merge_experts.py <in.gguf> <out.gguf>

Early Mixtral GGUFs (TheBloke/Mixtral-8x7B-v0.1-GGUF) store each expert as
blk.N.ffn_{gate,up,down}.E.weight. Current files have one 3-D tensor per
projection, blk.N.ffn_{gate,up,down}_exps.weight; llama.cpp reads only those
and so does geistlib, whose GPU path aliases a projection's experts as one
buffer. The rewrite only moves bytes: each merged tensor is its experts'
data back to back in expert order, quantization untouched. Metadata is
copied verbatim. Standard library only.
"""
import re
import struct
import sys

ALIGN_KEY = b"general.alignment"
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY = 8, 9
SPLIT = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)\.(\d+)\.weight$")
CHUNK = 1 << 24
# ggml type: (elements per block, bytes per block)
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 20), 3: (32, 24), 6: (32, 22), 7: (32, 26), 8: (32, 34),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210)}


def data_size(t):
    """Bytes of a 2-D tensor: whole blocks per row."""
    if t["dtype"] not in BLOCK:
        sys.exit(f"{t['name']}: no block size known for ggml type {t['dtype']}")
    elems, nbytes = BLOCK[t["dtype"]]
    cols, rows = t["dims"]
    return cols // elems * nbytes * rows


def skip_value(f, vtype):
    if vtype == STRING:
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif vtype == ARRAY:
        elem, count = struct.unpack("<IQ", f.read(12))
        if elem in SCALAR:
            f.seek(SCALAR[elem] * count, 1)
        else:
            for _ in range(count):
                skip_value(f, elem)
    else:
        f.seek(SCALAR[vtype], 1)


def read_metadata(f, n_kv):
    """Skip n_kv pairs; return the alignment the file declares."""
    align = 32
    for _ in range(n_kv):
        key = f.read(struct.unpack("<Q", f.read(8))[0])
        vtype = struct.unpack("<I", f.read(4))[0]
        if key == ALIGN_KEY and vtype in (4, 5):
            align = struct.unpack("<I", f.read(4))[0]
        else:
            skip_value(f, vtype)
    return align


def read_tensor(f):
    name = f.read(struct.unpack("<Q", f.read(8))[0]).decode()
    n_dims = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
    dtype, offset = struct.unpack("<IQ", f.read(12))
    return {"name": name, "dims": dims, "dtype": dtype, "offset": offset}


def write_info(o, t, offset):
    name = t["name"].encode()
    o.write(struct.pack("<Q", len(name)) + name)
    o.write(struct.pack("<I", len(t["dims"])))
    o.write(struct.pack(f"<{len(t['dims'])}Q", *t["dims"]))
    o.write(struct.pack("<IQ", t["dtype"], offset))


def pad_to(o, base, align):
    o.write(b"\0" * (-(o.tell() - base) % align))


def copy(f, o, offset, n):
    f.seek(offset)
    while n > 0:
        b = f.read(min(CHUNK, n))
        if not b:
            sys.exit("truncated input")
        o.write(b)
        n -= len(b)


def main(src, dst):
    with open(src, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF" or version < 2:
            sys.exit(f"{src}: not a GGUF v2+ file")
        align = read_metadata(f, n_kv)
        meta_end = f.tell()
        tensors = [read_tensor(f) for _ in range(n_tensors)]
        data = f.tell()
        data += -data % align
        size = f.seek(0, 2)

        # A tensor's size is the distance to the next offset, padding included.
        ends = sorted(t["offset"] for t in tensors) + [size - data]
        end_of = dict(zip(ends, ends[1:]))

        def stored_size(t):
            return end_of[t["offset"]] - t["offset"]

        groups = {}
        for t in tensors:
            m = SPLIT.match(t["name"])
            if m:
                groups.setdefault((int(m[1]), m[2]), {})[int(m[3])] = t
        if not groups:
            sys.exit(f"{src}: no per-expert tensors, nothing to do")

        # In file order, a merged tensor at its first expert's place.
        out, seen = [], set()
        for t in tensors:
            m = SPLIT.match(t["name"])
            if not m:
                out.append({**t, "parts": [t]})
                continue
            key = (int(m[1]), m[2])
            if key in seen:
                continue
            seen.add(key)
            parts = [groups[key][e] for e in sorted(groups[key])]
            if sorted(groups[key]) != list(range(len(parts))):
                sys.exit(f"{t['name']}: experts are not 0..{len(parts) - 1}")
            if any(p["dims"] != t["dims"] or p["dtype"] != t["dtype"] for p in parts):
                sys.exit(f"{t['name']}: experts differ in shape or type")
            name = f"blk.{key[0]}.ffn_{key[1]}_exps.weight"
            out.append({"name": name, "dims": (*t["dims"], len(parts)),
                        "dtype": t["dtype"], "parts": parts})

        # An expert's exact size, not the distance to the next tensor: the
        # padding between experts must not end up inside the merged tensor.
        for t in out:
            if len(t["parts"]) > 1:
                t["slice"] = data_size(t["parts"][0])

        with open(dst, "wb") as o:
            f.seek(0)
            o.write(struct.pack("<4sIQQ", magic, version, len(out), n_kv))
            f.seek(24)
            o.write(f.read(meta_end - 24))
            off = 0
            for t in out:
                write_info(o, t, off)
                n = sum(t.get("slice", stored_size(p)) for p in t["parts"])
                off += n + (-n % align)
            base = o.tell()
            base += -base % align
            o.write(b"\0" * (base - o.tell()))
            for t in out:
                for p in t["parts"]:
                    copy(f, o, data + p["offset"], t.get("slice", stored_size(p)))
                pad_to(o, base, align)
    print(f"{dst}: {n_tensors} tensors -> {len(out)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
