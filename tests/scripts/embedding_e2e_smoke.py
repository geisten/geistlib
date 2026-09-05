#!/usr/bin/env python3
"""embedding_e2e_smoke.py — run the BitNet embedding chain end to end.

Everything from phases 0-2 is exercised in one pass on a synthetic model:

    safetensors -> convert_bitnet_embedding.py -> GGUF -> geistlib loader
    -> per-projection input norms -> last-token pooling -> peek_embedding
    -> dump_geist_embedding -> .gemb

There are no real weights here and no reference vectors, so this proves
nothing about numerical correctness -- that is what
tools/eval_embedding_fidelity.py is for, and it needs the actual checkpoint.
What this proves is that the chain RUNS, which nothing else checked: the unit
tests stop at scratch sizing, and the converter test stops at the GGUF header.

That gap was not theoretical. The first time this ran it found two defects
that made every real conversion unloadable:

  1. The converter wrote 1-D norms as F16, following upstream's tensor-type
     table. geistlib's loader requires F32 and refuses the file outright
     (layer_wiring.c:65).
  2. The per-layer owning-buffer list held 16 entries. A BitNet embedding
     layer needs 18 -- 11 ordinary tensors plus the 7 per-projection input
     norms -- so the load died on the seventh norm.

Both now have narrow regression tests of their own. This stays as the broad
one: it catches the next thing that only shows up when the parts run together.

The weights are random ternary values, so the embeddings are meaningless.
Only structural invariants are asserted -- finite, unit-norm, deterministic,
input-dependent, and actually reading all seven norms.

Usage: embedding_e2e_smoke.py [path/to/dump_geist_embedding]
Without an argument the newest built tool is used. `make test-embedding` passes
$(BIN_DIR) explicitly, so MODE=asan runs the chain under AddressSanitizer —
which is how the scratch_proj_in buffer leak was found.

Exit codes follow tests/README.md: 0 PASS, 77 SKIPPED, 99 harness error.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SKIP, ERROR = 77, 99

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

# Small but structurally faithful: every projection input is a multiple of the
# 256-element I2_S block, and head_dim is deliberately NOT hidden/heads.
H, HEADS, KV, HD, INTER, LAYERS = 256, 2, 1, 128, 512, 2
VOCAB, BOS, EOS = 512, 510, 511

# The seven projections and the width of each one's input norm.
PROJECTIONS = (
    ("self_attn.q_proj", (HEADS * HD, H)),
    ("self_attn.k_proj", (KV * HD, H)),
    ("self_attn.v_proj", (KV * HD, H)),
    ("self_attn.o_proj", (H, HEADS * HD)),
    ("mlp.gate_proj", (INTER, H)),
    ("mlp.up_proj", (INTER, H)),
    ("mlp.down_proj", (H, INTER)),
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else f' — {detail}'}")
    if not ok:
        FAILURES.append(name)


def byte_to_codepoint(b: int) -> int:
    """gpt2 byte-level BPE mapping, mirroring gguf_tokenizer.c:51."""
    if (33 <= b <= 126) or (161 <= b <= 172) or b >= 174:
        return b
    if b <= 32:
        return 256 + b
    if 127 <= b <= 160:
        return 289 + (b - 127)
    return 323


def write_checkpoint(root: Path, np) -> None:
    """A synthetic HF checkpoint in the BitNet embedding tensor layout."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(dict(
        model_type="qwen3", hidden_size=H, num_attention_heads=HEADS,
        num_key_value_heads=KV, head_dim=HD, intermediate_size=INTER,
        num_hidden_layers=LAYERS, vocab_size=VOCAB, rms_norm_eps=1e-6,
        rope_theta=1e6, max_position_embeddings=4096,
        bos_token_id=BOS, eos_token_id=EOS,
    )))

    vocab = {chr(byte_to_codepoint(b)): b for b in range(256)}
    vocab.update({f"<unused{i}>": i for i in range(256, VOCAB)})
    (root / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [{"id": BOS, "content": "<bos>"},
                         {"id": EOS, "content": "<eos>"}],
    }))

    rng = np.random.default_rng(11)
    tensors: dict[str, "np.ndarray"] = {
        "model.embed_tokens.weight": rng.standard_normal((VOCAB, H), dtype=np.float32) * 0.02,
        "model.norm.weight": np.ones(H, np.float32),
        "lm_head.weight": rng.standard_normal((VOCAB, H), dtype=np.float32),
    }
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = np.ones(H, np.float32)
        tensors[p + "post_attention_layernorm.weight"] = np.ones(H, np.float32)
        tensors[p + "self_attn.q_norm.weight"] = np.ones(HD, np.float32)
        tensors[p + "self_attn.k_norm.weight"] = np.ones(HD, np.float32)
        for name, shape in PROJECTIONS:
            tensors[p + name + ".weight"] = (
                rng.integers(-1, 2, shape).astype(np.float32) * 0.05)
            tensors[p + name + ".norm.weight"] = np.ones(shape[1], np.float32)

    write_safetensors(root / "model.safetensors", tensors, np)


def write_safetensors(path: Path, tensors: dict, np) -> None:
    header, offset, blobs = {}, 0, []
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    hj = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))


def scale_one_norm_half(src: Path, dst: Path, tensor: str, np) -> None:
    """Copy a checkpoint, doubling the FIRST HALF of one norm vector.

    Half, not all of it: a uniform scale on a projection's input norm is
    cancelled exactly by the next RMSNorm downstream (q_norm after q_proj,
    the o_proj input norm after the attention output), so scaling the whole
    vector leaves the embedding bit-identical whether the norm is read or
    silently ignored. A non-uniform change cannot be cancelled, so it
    actually distinguishes the two.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "tokenizer.json"):
        (dst / f).write_text((src / f).read_text())

    blob = (src / "model.safetensors").read_bytes()
    (hlen,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8:8 + hlen])
    data = blob[8 + hlen:]
    if tensor not in header:
        raise KeyError(tensor)

    out = {}
    for name, meta in header.items():
        lo, hi = meta["data_offsets"]
        arr = np.frombuffer(data[lo:hi], dtype=np.float32).copy()
        if name == tensor:
            arr[: len(arr) // 2] *= 2.0
        out[name] = arr.reshape(meta["shape"])
    write_safetensors(dst / "model.safetensors", out, np)


def read_gemb(path: Path, np):
    blob = path.read_bytes()
    magic, version, count, dim = struct.unpack("<4sIII", blob[:16])
    if magic != b"GEMB" or version != 1:
        raise ValueError(f"not a v1 .gemb: {magic!r} v{version}")
    return np.frombuffer(blob[16:], dtype="<f4").reshape(count, dim)


def run_dump(binary: Path, gguf: Path, prompts: Path, out: Path) -> bool:
    r = subprocess.run([str(binary), str(gguf), str(prompts), str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    dump_geist_embedding failed: {r.stdout.strip()} {r.stderr.strip()}")
    return r.returncode == 0


def main() -> int:
    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy not installed")
        return SKIP
    try:
        from convert_bitnet_embedding import convert
    except ImportError as e:  # pragma: no cover - the tool is in-tree
        print(f"ERROR: cannot import the converter: {e}")
        return ERROR

    # The Makefile passes $(BIN_DIR)/tools/dump_geist_embedding so MODE=asan
    # exercises the ASan build; the glob is the fallback for a bare run.
    if len(sys.argv) > 1:
        binary = Path(sys.argv[1])
        if not binary.is_file():
            print(f"SKIP: {binary} not built — run `make bin` first")
            return SKIP
    else:
        matches = sorted(REPO.glob("bin/*/*/tools/dump_geist_embedding"))
        if not matches:
            print("SKIP: dump_geist_embedding not built — run `make bin` first")
            return SKIP
        binary = matches[-1]

    print("=== BitNet embedding: converter -> GGUF -> forward -> .gemb ===")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ckpt = root / "ckpt"
        write_checkpoint(ckpt, np)

        gguf = root / "mini-i2_s.gguf"
        convert(ckpt, gguf, "i2_s")

        prompts = root / "prompts.txt"
        prompts.write_text("hello world\nhello world\nhello worlds\nhello\n")

        base_out = root / "base.gemb"
        if not run_dump(binary, gguf, prompts, base_out):
            check("the model loads and embeds", False, "see above")
            print("FAILED: the chain does not run; later checks are meaningless")
            return 1
        check("the model loads and embeds", True)

        v = read_gemb(base_out, np)
        check("four rows at the model's width", v.shape == (4, H), str(v.shape))
        check("every value is finite", bool(np.isfinite(v).all()))

        norms = np.linalg.norm(v, axis=1)
        check("rows are L2-normalised", bool(np.abs(norms - 1.0).max() < 1e-5),
              f"max |‖v‖-1| = {float(np.abs(norms - 1.0).max()):.3e}")
        check("no row collapsed to zero", bool((np.abs(v).sum(axis=1) > 0).all()))

        # Rows 0 and 1 are the same prompt: the forward path must be
        # deterministic, or a fidelity number measured against it means nothing.
        check("the same prompt gives a bit-identical vector",
              bool(np.array_equal(v[0], v[1])))

        # Last-token pooling reads the final position of a causal stack, so a
        # different prompt must land somewhere else.
        check("different prompts give different vectors",
              len({r.tobytes() for r in v}) == 3, f"{len({r.tobytes() for r in v})} distinct")

        # Each of the seven per-projection norms must reach the output. A norm
        # that is loaded but never applied is invisible everywhere else --
        # the model still runs and still produces plausible unit vectors.
        print("  -- each per-projection input norm must move the output --")
        for tag, (proj, _) in zip("qkvogud", PROJECTIONS):
            variant = root / f"var_{tag}"
            scale_one_norm_half(ckpt, variant, f"model.layers.0.{proj}.norm.weight", np)
            vg = root / f"var_{tag}.gguf"
            convert(variant, vg, "i2_s")
            vout = root / f"var_{tag}.gemb"
            if not run_dump(binary, vg, prompts, vout):
                check(f"{proj} input norm is applied", False, "dump failed")
                continue
            w = read_gemb(vout, np)
            worst = min(float(v[i] @ w[i]) for i in range(len(v)))
            check(f"{proj} input norm is applied", worst < 0.999999,
                  f"cosine {worst:.6f} — output did not move")

    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1
    print("PASS: the embedding chain runs and every norm reaches the output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
