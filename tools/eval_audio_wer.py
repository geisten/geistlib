#!/usr/bin/env python3
"""eval_audio_wer.py — score bench_audio_wer transcripts against reference
text. Stdlib only (runs on a bare Pi).

Usage:
    bench_audio_wer wavlist.txt > transcripts.tsv
    python3 tools/eval_audio_wer.py transcripts.tsv references.tsv

references.tsv: <wav-path>\t<reference text> per line. For LibriSpeech,
build it from the *.trans.txt files (see benchmark/results/PI5-audio.md).

Reports per-clip and aggregate WER (word-level Levenshtein over
lowercased, punctuation-stripped text — standard ASR normalization),
plus the speed columns bench_audio_wer emitted.
"""
import re
import sys


def norm_words(text: str) -> list[str]:
    """Lowercase, strip punctuation, split. Keeps Unicode letters (umlauts,
    accents) — the ASCII-only class silently split German words ("schön" ->
    "sch n"), wrecking non-English WER."""
    text = text.lower()
    text = re.sub(r"[^\w' ]+", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return text.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    refs = {}
    with open(sys.argv[2]) as f:
        for line in f:
            if "\t" in line:
                path, text = line.rstrip("\n").split("\t", 1)
                refs[path] = text

    total_err, total_words, n_clips = 0, 0, 0
    sum_attach, sum_decode, sum_tok = 0.0, 0.0, 0
    print(f"{'clip':<44} {'WER':>6}  {'attach':>7} {'decode':>7} {'tok':>4}")
    with open(sys.argv[1]) as f:
        for line in f:
            if not line.startswith("WER\t"):
                continue
            _, path, attach_ms, decode_ms, n_tok, hyp = line.rstrip("\n").split("\t", 5)
            if path not in refs:
                print(f"{path:<44} (no reference)", file=sys.stderr)
                continue
            ref_w = norm_words(refs[path])
            hyp_w = norm_words(hyp)
            err = edit_distance(ref_w, hyp_w)
            wer = err / max(1, len(ref_w))
            total_err += err
            total_words += len(ref_w)
            n_clips += 1
            sum_attach += float(attach_ms)
            sum_decode += float(decode_ms)
            sum_tok += int(n_tok)
            name = path.rsplit("/", 1)[-1]
            print(f"{name:<44} {wer:6.1%}  {attach_ms:>5}ms {decode_ms:>5}ms {n_tok:>4}")

    if n_clips == 0:
        print("no scored clips", file=sys.stderr)
        return 1
    print(f"\nclips: {n_clips}   aggregate WER: {total_err / max(1, total_words):.1%} "
          f"({total_err} errors / {total_words} ref words)")
    print(f"mean attach: {sum_attach / n_clips:.0f} ms   "
          f"decode: {sum_decode / n_clips:.0f} ms   "
          f"{sum_tok / (sum_decode / 1000.0):.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
