#!/usr/bin/env python3
"""Requantise a MoE GGUF's least-used experts to a lower precision (#247).

    tools/moe_cold_requant.py <in.gguf> <out.gguf> --usage <raw> --model <tag>
        [--cold 0.25] [--cold-type Q2_K] [--out-type Q8_0] [--pick least|random]
        [--threads 8] [--libggml <libggml-base.dylib|.so>]

A research tool, not a production path. GGUF stores each projection's experts
as one 3-D tensor of one type, so a file cannot mix precisions per expert.
The rewrite therefore carries every expert tensor in a fine container type
(`--out-type`, default Q8_0):

  hot expert:   out = quant(dequant(orig), out-type)
  cold expert:  out = quant(dequant(quant(dequant(orig), cold-type)), out-type)

The cold experts hold exactly the values a cold-type expert would hold, up to
the container's rounding, which the hot experts and a `--cold 0` baseline
share. Perplexity against that baseline is the cost of the cold precision.
When `--out-type` equals `--cold-type` and `--cold 1`, the experts are
quantised once, straight to that type: a uniform low-precision file for the
speed endpoints.

Cold experts are, per layer, the `--cold` share of experts with the fewest
router selections on the `profile` set of a #48 usage file
(benchmark/results/raw/2026-10-15_moe_expert_usage*.txt, one `USAGE` line
per model, set, file and layer). Ties break by expert id. The script also
prints the share of the `heldout` set's selections that hit cold experts.
`--pick random` draws the same number of cold experts per layer at random
(fixed seed), the control that shows what the profile is worth.

Quantisation runs in ggml's reference code through ctypes
(`ggml_quantize_chunk`, no importance matrix), the same arithmetic as
`llama-quantize` without `--imatrix`. Metadata is copied verbatim.
Standard library only.
"""
import argparse
import ctypes
import random
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY = 8, 9
TYPES = {"F32": 0, "F16": 1, "Q4_0": 2, "Q8_0": 8, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12,
         "Q5_K": 13, "Q6_K": 14, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23}
DEQUANT = {2: "q4_0", 8: "q8_0", 10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K",
           21: "iq3_s", 22: "iq2_s", 23: "iq4_xs"}
EXPS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
CHUNK = 1 << 24


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


def read_header(f):
    magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
    if magic != b"GGUF" or version < 2:
        sys.exit("not a GGUF v2+ file")
    align = 32
    for _ in range(n_kv):
        key = f.read(struct.unpack("<Q", f.read(8))[0])
        vtype = struct.unpack("<I", f.read(4))[0]
        if key == b"general.alignment" and vtype in (4, 5):
            align = struct.unpack("<I", f.read(4))[0]
        else:
            skip_value(f, vtype)
    meta_end = f.tell()
    tensors = []
    for _ in range(n_tensors):
        name = f.read(struct.unpack("<Q", f.read(8))[0]).decode()
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
        dtype, offset = struct.unpack("<IQ", f.read(12))
        tensors.append({"name": name, "dims": dims, "dtype": dtype, "offset": offset})
    data = f.tell()
    data += -data % align
    return version, n_kv, meta_end, tensors, data, align


def load_usage(path, model, n_layers, n_expert):
    """Per layer, the summed selection counts of each set."""
    sums = {s: [[0] * n_expert for _ in range(n_layers)] for s in ("profile", "heldout")}
    for line in open(path):
        if not line.startswith("USAGE "):
            continue
        _, m, s, _file, _ntok, layer, counts = line.split()
        if m != model or s not in sums:
            continue
        c = [int(x) for x in counts.split(",")]
        # moe_usage stops a list at the highest expert it saw (analyze.py pads too).
        c += [0] * (n_expert - len(c))
        if len(c) != n_expert:
            sys.exit(f"{path}: layer {layer} has {len(c)} experts, the model {n_expert}")
        row = sums[s][int(layer)]
        for e, v in enumerate(c):
            row[e] += v
    if not any(any(r) for r in sums["profile"]):
        sys.exit(f"{path}: no profile counts for model {model!r}")
    return sums


class Ggml:
    def __init__(self, path):
        lib = ctypes.CDLL(path)
        lib.ggml_row_size.restype = ctypes.c_size_t
        lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                            ctypes.c_void_p]
        lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib = lib

    def row_size(self, t, n):
        return self.lib.ggml_row_size(t, n)

    def dequant(self, t, src, n):
        out = (ctypes.c_float * n)()
        if t == 0:
            ctypes.memmove(out, src, 4 * n)
        else:
            fn = getattr(self.lib, f"dequantize_row_{DEQUANT[t]}")
            fn(ctypes.c_char_p(src), out, ctypes.c_int64(n))
        return out

    def quant(self, t, f32, rows, n_per_row):
        out = ctypes.create_string_buffer(self.row_size(t, n_per_row) * rows)
        self.lib.ggml_quantize_chunk(t, f32, out, 0, rows, n_per_row, None)
        return out.raw


def pick_cold(usage, n_layers, n_expert, share, pick):
    n_cold = round(share * n_expert)
    cold = []
    for layer in range(n_layers):
        order = sorted(range(n_expert), key=lambda e: (usage["profile"][layer][e], e))
        if pick == "random":
            random.Random(247 * 1000 + layer).shuffle(order)
        cold.append(set(order[:n_cold]))
    return n_cold, cold


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--usage", required=True)
    ap.add_argument("--model", required=True, help="model tag of the USAGE lines")
    ap.add_argument("--cold", type=float, default=0.0, help="share of experts per layer")
    ap.add_argument("--cold-type", default="Q2_K", choices=TYPES)
    ap.add_argument("--out-type", default="Q8_0", choices=TYPES)
    ap.add_argument("--pick", default="least", choices=("least", "random"),
                    help="least: the least used on the profile set; random: a control")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--libggml", default="libggml-base.dylib")
    a = ap.parse_args()
    g = Ggml(a.libggml)
    cold_t, out_t = TYPES[a.cold_type], TYPES[a.out_type]
    for t in {cold_t, out_t}:
        g.lib.ggml_quantize_init(t)

    with open(a.src, "rb") as f:
        version, n_kv, meta_end, tensors, data, align = read_header(f)
        exps = [t for t in tensors if t["name"].endswith(EXPS)]
        if not exps:
            sys.exit(f"{a.src}: no merged expert tensors")
        n_expert = exps[0]["dims"][2]
        n_layers = 1 + max(int(t["name"].split(".")[1]) for t in exps)
        usage = load_usage(a.usage, a.model, n_layers, n_expert)
        n_cold, cold = pick_cold(usage, n_layers, n_expert, a.cold, a.pick)
        hit = sum(usage["heldout"][li][e] for li in range(n_layers) for e in cold[li])
        tot = sum(map(sum, usage["heldout"]))
        prof_hit = sum(usage["profile"][li][e] for li in range(n_layers) for e in cold[li])
        prof_tot = sum(map(sum, usage["profile"]))
        print(f"cold: {n_cold} of {n_expert} experts per layer ({a.cold_type}, {a.pick}); "
              f"their share of the selections: profile {100 * prof_hit / prof_tot:.2f} %, "
              f"held-out {100 * hit / tot:.2f} %" if tot else "no held-out counts")

        out = []
        for t in tensors:
            ne0, rows = t["dims"][0], 1
            for d in t["dims"][1:]:
                rows *= d
            src_size = g.row_size(t["dtype"], ne0) * rows
            is_exps = t in exps
            dtype = out_t if is_exps else t["dtype"]
            out.append({**t, "out_dtype": dtype, "src_size": src_size,
                        "out_size": g.row_size(dtype, ne0) * rows, "exps": is_exps})

        with open(a.dst, "wb") as o:
            f.seek(0)
            o.write(struct.pack("<4sIQQ", b"GGUF", version, len(out), n_kv))
            f.seek(24)
            o.write(f.read(meta_end - 24))
            off = 0
            for t in out:
                name = t["name"].encode()
                o.write(struct.pack("<Q", len(name)) + name + struct.pack("<I", len(t["dims"])))
                o.write(struct.pack(f"<{len(t['dims'])}Q", *t["dims"]))
                o.write(struct.pack("<IQ", t["out_dtype"], off))
                off += t["out_size"] + (-t["out_size"] % align)
            base = o.tell()
            base += -base % align
            o.write(b"\0" * (base - o.tell()))
            pool = ThreadPoolExecutor(a.threads)
            for t in out:
                f.seek(data + t["offset"])
                if not t["exps"] or (t["out_dtype"] == t["dtype"] and n_cold == 0):
                    n = t["src_size"]
                    while n > 0:
                        b = f.read(min(CHUNK, n))
                        if not b:
                            sys.exit("truncated input")
                        o.write(b)
                        n -= len(b)
                else:
                    ne0, ne1, ne2 = t["dims"]
                    layer = int(t["name"].split(".")[1])
                    per = t["src_size"] // ne2
                    raw = f.read(t["src_size"])

                    def one(e, raw=raw, t=t, ne0=ne0, ne1=ne1, per=per, layer=layer):
                        src = raw[e * per:(e + 1) * per]
                        is_cold = e in cold[layer]
                        if t["dtype"] == t["out_dtype"] and not is_cold:
                            return src
                        x = g.dequant(t["dtype"], src, ne0 * ne1)
                        if is_cold and cold_t != out_t:
                            x = g.dequant(cold_t, g.quant(cold_t, x, ne1, ne0), ne0 * ne1)
                        return g.quant(out_t, x, ne1, ne0)

                    for b in pool.map(one, range(ne2)):
                        o.write(b)
                    print(f"  {t['name']}: {DEQUANT.get(t['dtype'], t['dtype'])} -> "
                          f"{DEQUANT[t['out_dtype']]}", file=sys.stderr, flush=True)
                o.write(b"\0" * (-(o.tell() - base) % align))
    print(f"{a.dst}: {len(exps)} expert tensors rewritten, {off / 1e9:.2f} GB of tensor data")


if __name__ == "__main__":
    main()
