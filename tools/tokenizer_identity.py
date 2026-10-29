#!/usr/bin/env python3
"""Token-for-token identity of the tokenizers between two git revisions.

Builds tools/dump_geist_tokens.c (public API only) against the library of a
base revision, built in a git worktree of its own, and against the working
tree, runs both on a generated corpus and diffs what they print: the ids
geist_session_tokenize returns, the length and logits geist_session_set_prompt
leaves, token_to_str for every id, bos/eos/add_bos/add_eos, token_by_text and
whether a draft with the same vocabulary is accepted.

Models:
  - each --tokenizer-bin, beside a synthetic GGUF without a tokenizer
    (tools/gen_synth_gguf.py, F32, one layer), so the engine loads it;
  - an adversarial synthetic tokenizer.bin written here: no ▁ piece, some
    byte tokens missing, duplicate pieces and merges, merges whose result is
    no piece, specials that prefix each other, an empty special and one whose
    id is past the vocabulary;
  - a synthetic GGUF with an embedded SentencePiece tokenizer;
  - each --gguf as it is (embedded tokenizer; --no-prefill, no inference).

The corpus is deterministic for --seed: the strings of the tokenizer tests
plus --n random strings (multilingual text, emoji, whitespace runs, control
characters, \\r\\n, long words, special-token literals, ▁ and <0xXX> literals,
invalid UTF-8). Exit status 0 when every model's dumps are identical.

Example:
  tools/tokenizer_identity.py --base origin/main \\
      --tokenizer-bin ../gemma-4-E2B-it/tokenizer.bin \\
      --gguf gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = "tools/dump_geist_tokens.c"

FIXED = [
    "Hello, world!", "The quick brown fox jumps over the lazy dog.", "a", " ",
    "  leading double space", "trailing space ", "multiple    spaces between",
    "123 4567 89", "x = 42 + 3.14", "for (int i = 0; i < 10; i++) { sum += arr[i]; }",
    "def f(x): return x ** 2", "Hallo Welt — wie geht es dir?", "中文测试: 你好世界",
    "日本語のテスト", "Привет мир", "🎉 Emoji works! 🚀✨", "\n\nDouble newline\n",
    "tab\there", "Mixed\nline\rbreaks\r\n", "quote\"with'apostrophes", "<bos>", "<|image|>",
    "Just text with <|audio|> in middle",
    "This is a longer paragraph with multiple sentences. " * 5,
    "▁", "word▁with▁markers", "  multiple  ▁  marker  mix",
    "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n",
    "<|im_start|>user\nGrüße aus München — 東京 🚀<|im_end|>\n<|im_start|>assistant\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
    "<|turn>user\nhi<turn|>\n<|turn>model\n",
    "<<bos>>", "<bo", "<bos", "s>", "<0x41>",
]

SPECIALS = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<start_of_turn>", "<end_of_turn>",
            "<|image|>", "<|audio|>", "<|turn>", "<turn|>", "<|tool>", "<tool|>", "<|\"|>",
            "<|im_start|>", "<|im_end|>", "<|begin_of_text|>", "<|eot_id|>", "<s>", "</s>",
            "<image_soft_token>", "<|channel>", "<channel|>"]
WORDS = ["the", "of", "and", "token", "merge", "heap", "Straße", "naïve", "café", "日本",
         "東京", "Москва", "привет", "مرحبا", "שלום", "नमस्ते", "สวัสดี", "안녕하세요",
         "Ελληνικά", "ǅemal", "é", "ﬁ", "Ⅻ", "１２３", "x²", "½"]
EMOJI = ["🎉", "🚀", "✨", "👍🏽", "👨‍👩‍👧‍👦", "🇩🇪", "❤️", "🤖", "🙂", "\U0001F9EA"]
WS = [" ", "  ", "   ", "\t", "\n", "\r\n", "\r", "\n\n", " \n", " ", "　", " "]
INVALID = [b"\xff", b"\xfe", b"\x80", b"\xbf", b"\xc3", b"\xe2\x96", b"\xf0\x9f\x98",
           b"\xc0\xaf", b"\xe0\x80\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80",
           b"\xf8\x88\x80\x80"]


def piece(rng: random.Random) -> bytes:
    k = rng.randrange(12)
    if k == 0:
        return rng.choice(SPECIALS).encode()
    if k == 1:
        return rng.choice(EMOJI).encode()
    if k == 2:
        return rng.choice(WS).encode()
    if k == 3:
        return rng.choice(INVALID)
    if k == 4:
        return bytes([rng.randrange(1, 32)])
    if k == 5:
        n = rng.randrange(15, 80)  # a long word
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n)).encode()
    if k == 6:
        return rng.choice(["▁", "▁▁", "<0x%02X>" % rng.randrange(256), "<", ">", "|"]).encode()
    if k == 7:
        cps = (chr(rng.randrange(0x20, 0x2FFF)) for _ in range(rng.randrange(1, 6)))
        return "".join(cps).encode()
    if k == 8:
        return str(rng.randrange(10 ** rng.randrange(1, 12))).encode()
    if k == 9:
        return rng.choice("!@#$%^&*()[]{};:'\",.<>/?\\`~-_=+").encode()
    return rng.choice(WORDS).encode()


def corpus(seed: int, n: int) -> list[bytes]:
    rng = random.Random(seed)
    out = [s.encode() for s in FIXED]
    for _ in range(n):
        s = b"".join(piece(rng) + (b" " if rng.random() < 0.5 else b"")
                     for _ in range(rng.randrange(1, 24)))
        out.append(s.replace(b"\0", b""))
    return [s for s in out if s]


def adversarial_gmtk(path: Path) -> int:
    """A tokenizer.bin that exercises the encoder's corner cases."""
    rng = random.Random(7)
    toks = [b"<unk>", b"<s>", b"</s>", b"<pad>"]
    toks += [b"<0x%02X>" % b for b in range(256) if b % 7 != 3]   # some bytes missing
    toks += [bytes([c]) for c in range(0x21, 0x7f)]
    toks += [w.encode() for w in WORDS] + [e.encode() for e in EMOJI[:5]]
    merges = []
    for w in ["the", "token", "merge", "hello", "world", "abc", "ab", "xyz"]:
        for end in range(2, len(w) + 1):
            merges.append((w[:end - 1].encode(), w[end - 1].encode()))
            toks.append(w[:end].encode())
    merges.append((b"q", b"q"))                 # result "qq" is no piece
    merges.append((b"t", b"h"))                 # duplicate of a merge above
    merges.append((b"\t", b" "))
    merges.append((b"", b"z"))                  # empty left side
    toks += [b"th", b"the", b"", b"abc"]        # duplicates and an empty piece
    rng.shuffle(merges)
    specials = [(1, b"<s>"), (2, b"</s>"), (3, b"<pad>"), (len(toks) - 1, b"<s"),
                (len(toks) + 5, b"<oob>"), (0, b""), (4, b"<0x41>"), (5, b"<s><s>")]
    o = bytearray(struct.pack("<9I", 0x4B544D47, 1, len(toks), len(merges), len(specials),
                              1, 2, 3, 0))
    for t in toks:
        o += struct.pack("<H", len(t)) + t
    for left, right in merges:
        o += struct.pack("<H", len(left)) + left + struct.pack("<H", len(right)) + right
    for i, t in specials:
        o += struct.pack("<IH", i, len(t)) + t
    path.write_bytes(bytes(o))
    return len(toks)


def run(cmd: list[str], cwd: Path, **kw) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, **kw)


def build_tool(tree: Path, jobs: int) -> Path:
    """Build tree's library and TOOL (copied from the working tree) against it."""
    if tree != ROOT:
        shutil.copy(ROOT / TOOL, tree / TOOL)
        shutil.copy(ROOT / "tools/tool_common.h", tree / "tools/tool_common.h")
    run(["make", f"-j{jobs}", "lib"], tree, stdout=subprocess.DEVNULL)
    bins = glob.glob(str(tree / "lib/*/release*/libgeist.a"))
    rel = Path(max(bins, key=os.path.getmtime)).parent.relative_to(tree / "lib")
    target = f"bin/{rel}/tools/dump_geist_tokens"
    run(["make", f"-j{jobs}", f"DEMO_SOURCES={TOOL}", target], tree, stdout=subprocess.DEVNULL)
    return tree / target


def synth_gguf(path: Path, vocab: int | None) -> None:
    args = [sys.executable, str(ROOT / "tools/gen_synth_gguf.py"), str(path), "--layers", "1",
            "--d-model", "64", "--heads", "1", "--kv-heads", "1", "--ffn", "64",
            "--wtype", "f32", "--embd-type", "f32"]
    args += ["--vocab", str(vocab)] if vocab else ["--spm-tokenizer"]
    run(args, ROOT, stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="origin/main", help="revision to compare against")
    ap.add_argument("--tokenizer-bin", action="append", default=[], type=Path)
    ap.add_argument("--gguf", action="append", default=[], type=Path)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--n", type=int, default=2500, help="random corpus strings")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", type=Path, help="keep the dumps here")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="tokid-"))
    out = args.out or work
    out.mkdir(parents=True, exist_ok=True)
    texts = corpus(args.seed, args.n)
    (work / "corpus.hex").write_text("".join(t.hex() + "\n" for t in texts))
    print(f"corpus: {len(texts)} strings")

    wt = work / "base"
    run(["git", "worktree", "add", "--detach", str(wt), args.base], ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        tools = {"base": build_tool(wt, args.jobs), "new": build_tool(ROOT, args.jobs)}

        cases = []  # (name, gguf, tokenizer.bin or None, extra args)
        bins = list(args.tokenizer_bin) + [work / "adversarial.bin"]
        adv_vocab = adversarial_gmtk(work / "adversarial.bin")
        for i, tb in enumerate(bins):
            vocab = adv_vocab if tb.name == "adversarial.bin" else struct.unpack_from(
                "<I", tb.read_bytes(), 8)[0]
            g = work / f"notok{i}.gguf"
            synth_gguf(g, vocab)
            cases.append((f"bin:{tb}", g, tb, ["--ids", str(vocab + 8)]))
        g = work / "spm.gguf"
        synth_gguf(g, None)
        cases.append(("gguf:synthetic-spm", g, None, ["--ids", "600"]))
        for g in args.gguf:
            cases.append((f"gguf:{g}", g, None, ["--no-prefill"]))

        failed = 0
        for k, (name, g, tb, extra) in enumerate(cases):
            env = dict(os.environ)
            env.pop("GEIST_TOKENIZER_PATH", None)
            if tb is not None:
                env["GEIST_TOKENIZER_PATH"] = str(tb.resolve())
            dumps = {}
            for side, tool in tools.items():
                dumps[side] = out / f"case{k}_{side}.txt"
                with open(dumps[side], "wb") as f:
                    subprocess.run([str(tool), str(g), str(work / "corpus.hex"), *extra],
                                   env=env, stdout=f, stderr=subprocess.DEVNULL, check=True)
            a, b = dumps["base"].read_bytes(), dumps["new"].read_bytes()
            n_entries = a.count(b"\nentry ")
            if a == b:
                print(f"IDENTICAL  {name} ({n_entries} inputs, {len(a)} bytes)")
            else:
                failed += 1
                print(f"DIFFERENT  {name}: diff {dumps['base']} {dumps['new']}")
        return 1 if failed else 0
    finally:
        run(["git", "worktree", "remove", "--force", str(wt)], ROOT)
        if args.out is None:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
