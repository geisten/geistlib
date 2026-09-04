#!/usr/bin/env python3
"""test_bitnet_embedding_convert_py.py — I2_S packing parity for the BitNet
embedding converter (tools/convert_bitnet_embedding.py).

Hermetic by contract: no GGUF, no network, no model weights.

The point of this test is that the converter's I2_S packing agrees with the C
side byte for byte. The golden vector below was produced by compiling the
`pack_i2_s` helper from tests/test_i2_s_parity.c:30 verbatim and dumping its
output for a deterministic trit pattern — so a drift in either implementation
fails here rather than in a silently wrong embedding.

The layout is strided (element b*256 + h*128 + g*32 + bb sits in byte
b*64 + h*32 + bb at shift 6-2g), which is exactly the detail the upstream
prose gets wrong, so it is worth pinning.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from convert_bitnet_embedding import (  # noqa: E402
    I2S_BLOCK_ELEMS,
    convert,
    map_tensor_name,
    LAYER_MAP_GEMMA3,
    LAYER_MAP_QWEN3,
    quantize_i2s,
    to_f16,
    unpack_i2s,
)

# pack_i2_s() from tests/test_i2_s_parity.c over trits[i] = ((i*7 + i//13) % 3) - 1
# for n_in=256, n_out=1, with the f32 tensor scale 0.125 appended.
GOLDEN_HEX = (
    "196688116688114699224499228411668811668821449922449962881166881199228411"
    "66881166882144992244996288116688116698224499224419668811" + "0000003e"
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else f' — {detail}'}")
    if not ok:
        FAILURES.append(name)


def trit_pattern(n: int) -> np.ndarray:
    return np.array([((i * 7 + i // 13) % 3) - 1 for i in range(n)], dtype=np.int8)


def test_golden_matches_c() -> None:
    trits = trit_pattern(I2S_BLOCK_ELEMS).reshape(1, I2S_BLOCK_ELEMS)
    # Weights whose ternarisation reproduces `trits` and whose mean(|w|) is 0.125.
    packed, _ = quantize_i2s(trits.astype(np.float32) * 0.125)
    got = (packed.tobytes() + struct.pack("<f", 0.125)).hex()
    check("packing is byte-identical to the C pack_i2_s", got == GOLDEN_HEX, got)


def test_round_trip() -> None:
    rng = np.random.default_rng(20260904)
    trits = rng.integers(-1, 2, size=(5, 512)).astype(np.int8)
    packed, scale = quantize_i2s(trits.astype(np.float32) * 0.25)
    back = np.rint(unpack_i2s(packed, 512, scale) / scale).astype(np.int8)
    check("pack -> unpack recovers every trit", np.array_equal(back, trits))


def test_scale_is_a_multiplier() -> None:
    """geistlib dequants as trit * scale, so the stored f32 is mean(|w|)."""
    w = (trit_pattern(256).reshape(1, 256) * 0.3).astype(np.float32)
    _, scale = quantize_i2s(w)
    check(
        "stored scale is mean(|w|), not its reciprocal",
        np.isclose(scale, np.abs(w).mean()),
        f"{scale} vs {np.abs(w).mean()}",
    )


def test_rejects_unrepresentable_width() -> None:
    """Gemma3-270M's hidden size (640) is not a whole number of 256-blocks."""
    try:
        quantize_i2s(np.ones((4, 640), dtype=np.float32))
    except SystemExit as exc:
        check("in-features not divisible by 256 is refused", "256" in str(exc))
    else:
        check("in-features not divisible by 256 is refused", False, "no error raised")


def test_name_mapping() -> None:
    cases_qwen3 = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "model.layers.3.self_attn.q_proj.norm.weight": "blk.3.attn_q_norm_in.weight",
        "model.layers.0.mlp.down_proj.norm.weight": "blk.0.ffn_down_norm_in.weight",
        "model.layers.7.post_attention_layernorm.weight": "blk.7.ffn_norm.weight",
        "lm_head.weight": None,
    }
    ok = all(map_tensor_name(k, LAYER_MAP_QWEN3) == v for k, v in cases_qwen3.items())
    check("qwen3 tensor names map correctly (LM head dropped)", ok)

    gemma = map_tensor_name("model.layers.2.post_attention_layernorm.weight", LAYER_MAP_GEMMA3)
    check(
        "gemma3 post_attention_layernorm keeps its own semantics",
        gemma == "blk.2.post_attention_norm.weight",
        str(gemma),
    )


def test_f16_overflow_is_loud() -> None:
    # 1e38 is finite in float32 and well past float16's 65504, so the guard
    # is what has to catch it. (1e39 would already be inf as a float32 and
    # would exercise nothing.)
    with np.errstate(over="ignore"):
        try:
            to_f16(np.array([1e38], dtype=np.float32))
        except SystemExit:
            check("f16 overflow fails instead of writing inf", True)
        else:
            check("f16 overflow fails instead of writing inf", False, "silently converted")


def write_mini_checkpoint(root: Path) -> None:
    """A 2-layer Qwen3-shaped checkpoint, small but structurally faithful."""
    h, heads, kv, hd, inter, vocab, layers = 256, 2, 1, 128, 512, 512, 2
    (root / "config.json").write_text(
        json.dumps(
            dict(
                model_type="qwen3", hidden_size=h, num_attention_heads=heads,
                num_key_value_heads=kv, head_dim=hd, intermediate_size=inter,
                num_hidden_layers=layers, vocab_size=vocab, rms_norm_eps=1e-6,
                rope_theta=1e6, max_position_embeddings=32768, eos_token_id=151643,
            )
        )
    )
    rng = np.random.default_rng(11)
    t: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.standard_normal((vocab, h), dtype=np.float32),
        "model.norm.weight": np.ones(h, np.float32),
        "lm_head.weight": rng.standard_normal((vocab, h), dtype=np.float32),
    }
    for i in range(layers):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = np.ones(h, np.float32)
        t[p + "post_attention_layernorm.weight"] = np.ones(h, np.float32)
        t[p + "self_attn.q_norm.weight"] = np.ones(hd, np.float32)
        t[p + "self_attn.k_norm.weight"] = np.ones(hd, np.float32)
        projections = (
            ("self_attn.q_proj", (heads * hd, h)), ("self_attn.k_proj", (kv * hd, h)),
            ("self_attn.v_proj", (kv * hd, h)), ("self_attn.o_proj", (h, heads * hd)),
            ("mlp.gate_proj", (inter, h)), ("mlp.up_proj", (inter, h)),
            ("mlp.down_proj", (h, inter)),
        )
        for name, shape in projections:
            t[p + name + ".weight"] = rng.integers(-1, 2, shape).astype(np.float32) * 0.05
            t[p + name + ".norm.weight"] = np.ones(shape[1], np.float32)

    header, offset, blobs = {}, 0, []
    for name, arr in t.items():
        raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        header[name] = {
            "dtype": "F32", "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)
    hj = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))


def test_end_to_end() -> None:
    """Convert a synthetic checkpoint and re-read the GGUF header we wrote."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_mini_checkpoint(root)
        out = root / "mini-i2_s.gguf"
        convert(root, out, "i2_s")

        blob = out.read_bytes()
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", blob[:24])
        check("GGUF magic and version are v3", magic == 0x46554747 and version == 3)
        # 2 layers x (7 projections + 7 input norms + 4 norms) + embd + output_norm,
        # with lm_head dropped.
        check("lm_head is dropped, everything else kept", n_tensors == 38, str(n_tensors))
        check("metadata is non-empty", n_kv > 10, str(n_kv))
        check("file is larger than its header", len(blob) > 24 + n_kv)


def main() -> int:
    print("=== BitNet embedding converter: I2_S packing ===")
    for fn in (
        test_golden_matches_c,
        test_round_trip,
        test_scale_is_a_multiplier,
        test_rejects_unrepresentable_width,
        test_name_mapping,
        test_f16_overflow_is_loud,
        test_end_to_end,
    ):
        fn()
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
