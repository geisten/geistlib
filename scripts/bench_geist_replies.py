#!/usr/bin/env python3
"""bench_geist_replies.py — capture geist multimodal decoded text for the
five test samples (uses test_chat_audio_stream, parses 'step N: got=ID'
from stderr, decodes via the Gemma tokenizer)."""
import re, subprocess, time, wave
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

SCRIPT = Path(__file__).parent.resolve()
GGUF = SCRIPT / "gguf_artifacts" / "gemma4-e2b-Q8_0.gguf"
SAFE = SCRIPT.parent / "gemma-4-E2B-it" / "model.safetensors"
MEL = SCRIPT / "audio_test_data" / "mel_constants.bin"

tok = AutoTokenizer.from_pretrained(SCRIPT.parent / "gemma-4-E2B-it")


def parse_time(stderr):
    rss = None
    for line in stderr.splitlines():
        s = line.strip()
        if "maximum resident set size" in s:
            rss = int(s.split()[0])
    return rss


def fmt_bytes(b):
    if b is None: return "?"
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"


def make_bin(arr):
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    f.write(np.asarray(arr, dtype=np.int32).tobytes()); f.close()
    return Path(f.name)


def run_geist(wav, ref, n_steps=20):
    """Run test_chat_audio_stream for n_steps, capture decoded tokens.
    The expected_tokens.bin is filled with HF tokens so first 5 are checked,
    additional ones are just decoded by geist. We pad expected with 0 so the
    binary doesn't fail past step 5."""
    ids_bin = make_bin(ref["input_ids"].tolist())
    # Pad expected with zeros to at least n_steps so binary doesn't crash.
    exp = list(ref["hf_greedy"]) + [0] * max(0, n_steps - len(ref["hf_greedy"]))
    expected_bin = make_bin(exp[:n_steps])
    t0 = time.time()
    proc = subprocess.run(["/usr/bin/time", "-l",
        str(SCRIPT / "test_chat_audio_stream"),
        str(GGUF), str(SAFE), str(wav), str(MEL),
        str(ids_bin), str(ref["n_pre"]), str(ref["n_audio"]), str(ref["n_post"]),
        str(expected_bin), str(n_steps),
    ], capture_output=True, timeout=300)
    wall = time.time() - t0
    err = proc.stderr.decode("utf-8", "replace")
    rss = parse_time(err)
    decoded = []
    for m in re.finditer(r"step\s+(\d+): got=\s*(-?\d+)", err):
        decoded.append(int(m.group(2)))

    # encoder breakdown
    em = re.search(r"first-batch wait ([\d.]+) ms", err)
    encode_ms = float(em.group(1)) if em else None

    ids_bin.unlink(); expected_bin.unlink()
    return {"wall_s": wall, "rss": rss, "decoded": decoded,
            "encode_ms": encode_ms,
            "text": tok.decode(decoded[:n_steps], skip_special_tokens=False)}


def main():
    samples = [
        "hello_world.wav",
        "en_question.wav",
        "en_long.wav",
        "de_hello.wav",
        "de_question.wav",
    ]
    rows = ["| Sample | dur | encode | wall | RSS | decoded reply |",
            "|---|---:|---:|---:|---:|---|"]
    for stem in samples:
        wav = ROOT_DIR / "audio_test_data" / stem
        npz = ROOT_DIR / "dumps" / "audio" / f"{stem.replace('.wav','')}_template.npz"
        d = np.load(npz)
        ref = {"input_ids": d["input_ids"], "n_pre": int(d["n_pre"]),
               "n_audio": int(d["n_audio"]), "n_post": int(d["n_post"]),
               "hf_greedy": d["greedy_tokens"].tolist()}
        with wave.open(str(wav), "rb") as w:
            dur = w.getnframes() / w.getframerate()
        r = run_geist(wav, ref, n_steps=20)
        text_clean = r["text"].replace("|", "/").replace("\n", " ⏎ ")[:120]
        rows.append(
            f"| {stem} | {dur:.2f}s | {r['encode_ms']:.0f}ms | "
            f"{r['wall_s']:.2f}s | {fmt_bytes(r['rss'])} | `{text_clean}…` |"
        )
        print(f"  {stem}: wall={r['wall_s']:.1f}s encode={r['encode_ms']:.0f}ms text={r['text'][:60]!r}")

    out = ROOT_DIR / "geist_replies_report.md"
    out.write_text("# geist multimodal replies (Q8_0, 20 greedy tokens)\n\n" + "\n".join(rows) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
main__":
    main()
