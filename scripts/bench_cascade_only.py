#!/usr/bin/env python3
"""bench_cascade_only.py — clean cascade (whisper + llama-completion) re-run
to fill the broken cascade-reply column from bench_suite.py v1.

Outputs cascade_report.md with: per-sample whisper transcript, llama-completion
reply, wall time + RSS for each stage.
"""
import argparse, re, subprocess, time, wave
from pathlib import Path

SCRIPT = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT.parent.parent
GGUF = ROOT_DIR / "gguf_artifacts" / "gemma4-e2b-Q8_0.gguf"
WHISPER = Path.home() / "whisper-models" / "ggml-base.bin"


def parse_time(stderr):
    rss = None
    for line in stderr.splitlines():
        s = line.strip()
        if "maximum resident set size" in s:
            rss = int(s.split()[0])
    return rss


def run_t(cmd, timeout=180):
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, timeout=timeout)
    return time.time() - t0, parse_time(p.stderr.decode("utf-8", "replace")), p.stdout, p.stderr


def fmt_bytes(b):
    if b is None: return "?"
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"


def cascade(wav, question):
    w_wall, w_rss, w_out, _ = run_t(
        ["whisper-cli", "-m", str(WHISPER), "-f", str(wav), "-nt", "-np"])
    transcript = " ".join(
        line.strip() for line in w_out.decode("utf-8", "replace").splitlines() if line.strip()
    )

    prompt = (f"<bos><start_of_turn>user\n"
              f"{question}\nTranscript: {transcript}\n"
              f"<end_of_turn>\n<start_of_turn>model\n")
    l_wall, l_rss, l_out, _ = run_t([
        "llama-completion", "-m", str(GGUF), "-p", prompt, "-n", "30",
        "-no-cnv", "--jinja", "--no-warmup", "--simple-io", "--temp", "0",
    ])
    raw = l_out.decode("utf-8", "replace")
    # Strip everything up to (and including) the LAST `<start_of_turn>model\n`,
    # plus trailing end-of-text markers.
    marker = "<start_of_turn>model\n"
    idx = raw.rfind(marker)
    reply = raw[idx + len(marker):] if idx >= 0 else raw
    reply = re.sub(r"\[end of text\].*$", "", reply, flags=re.S).strip()
    reply = re.sub(r"<end_of_turn>.*$", "", reply, flags=re.S).strip()
    return {
        "transcript": transcript, "reply": reply,
        "whisper_s": w_wall, "whisper_rss": w_rss,
        "llama_s": l_wall, "llama_rss": l_rss,
    }


def main():
    samples = [
        ("hello_world.wav",  "What did the speaker say? Answer briefly."),
        ("en_question.wav",  "Answer briefly."),
        ("en_long.wav",      "Repeat the sentence the speaker said."),
        ("de_hello.wav",     "Was hat der Sprecher gesagt? Kurz antworten."),
        ("de_question.wav",  "Beantworte die Frage des Sprechers."),
    ]
    rows = ["| Sample | dur | whisper | llama-cpl | total | RSS (max) | transcript | reply (first 100 chars) |",
            "|---|---:|---:|---:|---:|---:|---|---|"]
    for stem, q in samples:
        wav = SCRIPT / "audio_test_data" / stem
        with wave.open(str(wav), "rb") as w:
            dur = w.getnframes() / w.getframerate()
        c = cascade(wav, q)
        rss = max(c["whisper_rss"] or 0, c["llama_rss"] or 0)
        reply_clean = c["reply"][:100].replace("|", "/").replace("\n", " ⏎ ")
        rows.append(
            f"| {stem} | {dur:.2f}s | {c['whisper_s']:.2f}s | {c['llama_s']:.2f}s | "
            f"{c['whisper_s'] + c['llama_s']:.2f}s | {fmt_bytes(rss)} | "
            f"`{c['transcript'][:60]}` | `{reply_clean}…` |"
        )
        print(f"  {stem}: w={c['whisper_s']:.1f}s l={c['llama_s']:.1f}s reply='{c['reply'][:60]}'")

    out = SCRIPT / "cascade_report.md"
    out.write_text("# Cascade (whisper.cpp + llama-completion Q8_0)\n\n" + "\n".join(rows) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
