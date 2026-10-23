"""Minimal GGUF metadata reader: tokenizer.ggml.tokens / token_type -> byte-level detokenizer."""
import struct

def read_meta(path, want=("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.model")):
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, = struct.unpack("<I", f.read(4)); nt, nkv = struct.unpack("<QQ", f.read(16))
        def s():
            n, = struct.unpack("<Q", f.read(8)); return f.read(n)
        SZ = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
        def val(t, keep):
            if t == 8: x = s(); return x.decode("utf-8", "surrogateescape") if keep else None
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et in SZ and not keep:
                    f.seek(n * struct.calcsize("<" + SZ[et]), 1); return None
                if et in SZ:
                    fmt = "<" + SZ[et]; sz = struct.calcsize(fmt); raw = f.read(n * sz)
                    return list(struct.unpack("<%d%s" % (n, SZ[et]), raw))
                return [val(et, keep) for _ in range(n)]
            fmt = "<" + SZ[t]; return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        for _ in range(nkv):
            k = s().decode(); t, = struct.unpack("<I", f.read(4))
            v = val(t, k in want)
            if k in want: out[k] = v
            if len(out) == len(want): break
    return out

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

class Detok:
    def __init__(self, path):
        m = read_meta(path)
        self.model = m.get("tokenizer.ggml.model")
        self.tokens = m["tokenizer.ggml.tokens"]; self.types = m["tokenizer.ggml.token_type"]
        u = bytes_to_unicode()
        self.bytes = []
        for tok, ty in zip(self.tokens, self.types):
            if ty == 1:  # normal: byte-level
                self.bytes.append(bytes(u[c] for c in tok) if all(c in u for c in tok) else tok.encode("utf-8", "surrogateescape"))
            else:
                self.bytes.append(tok.encode("utf-8", "surrogateescape"))
    def decode(self, ids):
        return b"".join(self.bytes[i] for i in ids)
