#!/usr/bin/env python3
"""bench_suite.py — comprehensive benchmark suite for geist.

Compares:
  • Text LM:    geist test_greedy_quant  vs  llama-cli   (Q4_K_M, Q8_0, F16)
  • Audio:      geist test_chat_audio    vs  whisper-cli + llama-cli (cascade)

Captures:
  • Wall-clock latency (model load, prefill, decode)
  • Peak resident set size (via /usr/bin/time -l)
  • Energy approximation (joules ≈ wall_time × P_active for the executable)
  • Audio quality: transcript / response

Outputs a markdown report with side-by-side tables.

Run:
  source ../gemma-4-E2B-it/.venv/bin/activate
  python bench_suite.py --report bench_report.md
"""
import argparse, json, re, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).parent.resolve()
GGUF_DIR = SCRIPT / "gguf_artifacts"
SAFETENSORS = SCRIPT.parent / "gemma-4-E2B-it" / "model.safetensors"
WHISPER_MODEL = Path.home() / "whisper-models" / "ggml-base.bin"
MEL_CONST = SCRIPT / "audio_test_data" / "mel_constants.bin"

# M1 single-core typical active power; rough but consistent baseline.
M1_ACTIVE_W = 4.0   # W
M1_IDLE_W   = 0.4   # W (single core idle)

TIME_CMD = "/usr/bin/time"   # BSD time on macOS, supports -l

# ----------------------------- helpers ------------------------------------ #

def parse_time_l(stderr_text: str):
    """Pull peak_rss (bytes), instructions, cycles from /usr/bin/time -l."""
    rss = inst = cyc = None
    for line in stderr_text.splitlines():
        line = line.strip()
        if "maximum resident set size" in line:
            rss = int(line.split()[0])
        elif "instructions retired" in line:
            inst = int(line.split()[0])
        elif "cycles elapsed" in line:
            cyc = int(line.split()[0])
    return rss, inst, cyc


def run_timed(cmd, env=None, input_bytes=None):
    """Run `cmd` under /usr/bin/time -l. Returns (wall_s, rss_bytes,
    instructions, cycles, stdout, stderr)."""
    full = [TIME_CMD, "-l"] + cmd
    t0 = time.time()
    proc = subprocess.run(full, env=env, input=input_bytes,
                          capture_output=True, timeout=600)
    wall = time.time() - t0
    rss, inst, cyc = parse_time_l(proc.stderr.decode("utf-8", "replace"))
    return wall, rss, inst, cyc, proc.stdout, proc.stderr


def joules(wall_s, cycles, instructions):
    """Crude energy estimate: wall_time × M1 active power.
    A more refined version using cycles ÷ clock × P_per_active_cycle would need
    per-core duty cycle data we don't have without `powermetrics` (sudo)."""
    return wall_s * M1_ACTIVE_W


def fmt_bytes(b):
    if b is None: return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024: return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def fmt_ms(s):
    if s is None: return "?"
    return f"{s*1000:.0f} ms" if s < 1 else f"{s:.2f} s"

# ----------------------------- text LM ------------------------------------ #

def make_prompt_bin(token_ids):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    f.write(np.asarray(token_ids, dtype=np.int32).tobytes()); f.close()
    return Path(f.name)


def bench_text_lm(quant: str, n_prompt: int, n_decode: int):
    """Returns dict: geist & llama-cli timings on the same prompt size + decode."""
    gguf = GGUF_DIR / f"gemma4-e2b-{quant}.gguf"
    if not gguf.exists():
        return None

    rng = np.random.default_rng(42)
    prompt_ids = [2] + rng.integers(low=2000, high=200000, size=n_prompt - 1, dtype=np.int32).tolist()
    pb = make_prompt_bin(prompt_ids)
    out_bin = Path(tempfile.NamedTemporaryFile(suffix=".bin", delete=False).name)

    # geist
    g_wall, g_rss, g_inst, g_cyc, _, _ = run_timed(
        [str(SCRIPT / "test_greedy_quant"), str(gguf), str(pb), str(n_decode), str(out_bin)])

    # llama-cli for the same effective work: --no-conversation, prompt-only,
    # n_predict = n_decode. Use simple single-line prompt via --prompt via stdin
    # of token ids isn't supported; we use llama-bench instead for fair perf
    # numbers.
    lb = subprocess.run(
        ["llama-bench", "-m", str(gguf), "-p", str(n_prompt), "-n", str(n_decode),
         "-r", "1", "-t", "1", "-ngl", "0"],
        capture_output=True, text=True, timeout=600)
    pp = tg = None
    for line in lb.stdout.splitlines():
        if "|" in line and ("pp" in line or "tg" in line):
            parts = [p.strip() for p in line.split("|")]
            try:
                t = float(parts[-2].split("±")[0].strip())
                if " pp" in line or "pp\t" in line or re.search(r"pp\d+", line):
                    pp = t
                if " tg" in line or "tg\t" in line or re.search(r"tg\d+", line):
                    tg = t
            except Exception:
                pass

    pb.unlink(missing_ok=True); out_bin.unlink(missing_ok=True)

    return {
        "quant": quant,
        "geist_wall_s": g_wall,
        "geist_rss": g_rss,
        "geist_joules": joules(g_wall, g_cyc, g_inst),
        "llama_pp_tps": pp,
        "llama_tg_tps": tg,
        "llama_bench_raw": lb.stdout,
    }

# ----------------------------- audio chat --------------------------------- #

def build_template_dump(wav: Path, n_steps=5,
                        question="Beantworte kurz: Was sagt der Sprecher?"):
    """Run dump_audio_chat_template.py — generates HF reference + extracts
    n_pre/n_audio/n_post + greedy_tokens for the given WAV+question."""
    dump = SCRIPT / "dump_audio_chat_template.py"
    proc = subprocess.run(
        [sys.executable, str(dump), str(wav), "--n_steps", str(n_steps),
         "--question", question],
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print("dump failed:", proc.stderr, file=sys.stderr)
        return None
    npz = SCRIPT / "dumps" / "audio" / f"{wav.stem}_template.npz"
    d = np.load(npz)
    return {
        "input_ids": d["input_ids"],
        "n_pre": int(d["n_pre"]),
        "n_audio": int(d["n_audio"]),
        "n_post": int(d["n_post"]),
        "hf_greedy": d["greedy_tokens"].tolist(),
    }


def bench_geist_audio(wav: Path, ref: dict, quant="Q8_0", n_steps=5):
    gguf = GGUF_DIR / f"gemma4-e2b-{quant}.gguf"
    ids_bin = make_prompt_bin(ref["input_ids"].tolist())
    expected_bin = make_prompt_bin(ref["hf_greedy"][:n_steps])

    wall, rss, inst, cyc, out, err = run_timed([
        str(SCRIPT / "test_chat_audio_stream"),
        str(gguf), str(SAFETENSORS), str(wav), str(MEL_CONST),
        str(ids_bin), str(ref["n_pre"]), str(ref["n_audio"]), str(ref["n_post"]),
        str(expected_bin), str(n_steps),
    ])
    matched = err.decode("utf-8", "replace").count("✓")
    ids_bin.unlink(); expected_bin.unlink()

    return {
        "wall_s": wall,
        "rss": rss,
        "joules": joules(wall, cyc, inst),
        "matched": matched,
        "n_steps": n_steps,
        "stderr_tail": err.decode("utf-8", "replace").splitlines()[-12:],
    }


def bench_cascade(wav: Path, question="Beantworte kurz: Was sagt der Sprecher?",
                   quant="Q8_0", n_decode=10):
    """Whisper transcribes WAV → llama-cli generates response from
    transcription + question prompt."""
    gguf = GGUF_DIR / f"gemma4-e2b-{quant}.gguf"

    # Step 1: whisper transcribe
    w_wall, w_rss, w_inst, w_cyc, w_stdout, w_stderr = run_timed([
        "whisper-cli", "-m", str(WHISPER_MODEL), "-f", str(wav), "-nt", "-np",
    ])
    transcript = w_stdout.decode("utf-8", "replace").strip().splitlines()
    transcript = " ".join(t.strip() for t in transcript if t.strip())

    # Step 2: llama-cli generate response
    prompt = (
        f"<bos><start_of_turn>user\n"
        f"{question}\nTranskript: {transcript}\n"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )
    l_wall, l_rss, l_inst, l_cyc, l_stdout, l_stderr = run_timed([
        "llama-cli", "-m", str(gguf), "-p", prompt, "-n", str(n_decode),
        "--no-conversation", "-no-cnv", "-st", "-t", "1", "-ngl", "0",
        "--temp", "0",
    ])
    reply = l_stdout.decode("utf-8", "replace")
    # Strip the echoed prompt from the reply if present.
    if prompt in reply:
        reply = reply.split(prompt, 1)[1]

    return {
        "transcript": transcript,
        "reply": reply.strip()[:200],
        "whisper_wall_s": w_wall,
        "whisper_rss": w_rss,
        "whisper_joules": joules(w_wall, w_cyc, w_inst),
        "llama_wall_s": l_wall,
        "llama_rss": l_rss,
        "llama_joules": joules(l_wall, l_cyc, l_inst),
        "total_wall_s": w_wall + l_wall,
        "total_rss": max(w_rss or 0, l_rss or 0),
        "total_joules": joules(w_wall, w_cyc, w_inst) + joules(l_wall, l_cyc, l_inst),
    }

# ----------------------------- driver ------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="bench_report.md")
    ap.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q8_0", "f16"])
    args = ap.parse_args()

    report = ["# geist Benchmark Report",
              f"_{time.strftime('%Y-%m-%d %H:%M')}_  · Apple M1 Max, 64 GB · single CPU thread (geist), GPU+8 threads (llama-bench)",
              ""]

    # ---- text LM ----
    print("== Text LM ==", file=sys.stderr)
    report += ["## Text LM (greedy decode, t/s)", ""]
    report += ["| Quant | geist wall (s) | geist RSS | geist J | llama-bench pp64 (t/s) | llama-bench tg16 (t/s) |",
               "|---|---:|---:|---:|---:|---:|"]
    for q in args.quants:
        r = bench_text_lm(q, n_prompt=64, n_decode=16)
        if not r:
            report.append(f"| {q} | (gguf missing) |||||")
            continue
        print(f"  {q}: wall={r['geist_wall_s']:.2f}s, rss={fmt_bytes(r['geist_rss'])}, "
              f"llama pp={r['llama_pp_tps']}, tg={r['llama_tg_tps']}", file=sys.stderr)
        report.append(
            f"| {q} | {r['geist_wall_s']:.2f} | {fmt_bytes(r['geist_rss'])} | "
            f"{r['geist_joules']:.2f} | {r['llama_pp_tps'] or '?'} | {r['llama_tg_tps'] or '?'} |"
        )
    report.append("")

    # ---- audio chat ----
    print("== Audio Chat ==", file=sys.stderr)
    samples = [
        ("hello_world.wav",  "What did the speaker say? Answer briefly."),
        ("en_question.wav",  "Answer briefly."),
        ("en_long.wav",      "Repeat the sentence the speaker said."),
        ("de_hello.wav",     "Was hat der Sprecher gesagt? Kurz antworten."),
        ("de_question.wav",  "Beantworte die Frage des Sprechers."),
    ]
    report += ["## Audio chat: geist multimodal (Q8_0) vs cascade (whisper-base + llama Q8_0)", ""]
    report += ["| Sample | geist wall | geist RSS | geist J | match | cascade wall | cascade RSS | cascade J |",
               "|---|---:|---:|---:|:-:|---:|---:|---:|"]
    detail_blocks = []
    for stem, question in samples:
        wav = SCRIPT / "audio_test_data" / stem
        if not wav.exists():
            report.append(f"| {stem} | (missing wav) |"+("|"*7))
            continue
        ref = build_template_dump(wav, question=question, n_steps=5)
        if not ref:
            report.append(f"| {stem} | (dump failed) |"+("|"*7))
            continue
        gz = bench_geist_audio(wav, ref)
        cz = bench_cascade(wav, question=question)
        report.append(
            f"| {stem} | {gz['wall_s']:.2f}s | {fmt_bytes(gz['rss'])} | "
            f"{gz['joules']:.2f} | {gz['matched']}/{gz['n_steps']} | "
            f"{cz['total_wall_s']:.2f}s | {fmt_bytes(cz['total_rss'])} | "
            f"{cz['total_joules']:.2f} |"
        )
        detail_blocks.append(
            f"### {stem} — `{question}`\n"
            f"- whisper transcript: `{cz['transcript'][:120]}`\n"
            f"- cascade reply: `{cz['reply'][:200]}`\n"
            f"- HF greedy ref tokens: `{ref['hf_greedy']}`\n"
        )
    report += ["", "## Audio detail", ""] + detail_blocks

    # ---- summary footnote ----
    report += [
        "",
        "## Notes",
        f"- **Energy** is `wall_time × {M1_ACTIVE_W} W` (rough M1 single-core CPU active power). "
        "True per-query joules need `powermetrics` (sudo).",
        "- **geist** runs CPU-only, single-thread NEON. **whisper-cli** uses Metal/GPU. **llama-bench** uses CPU with `-t 1 -ngl 0` (single thread). **llama-cli** in cascade uses defaults (Metal).",
        "- geist text-LM wall-time includes model load (~1s); llama-bench amortises load.",
        "- geist audio wall-time includes model load (~3s safetensors + ~1s GGUF), audio encode (~150ms), LM prefill of 30+ tokens, and 5 greedy tokens.",
        "- Memory comparison: geist holds full audio_tower in fp32 (~1.2 GB) PLUS LM. Cascade holds whisper (~150 MB) + LM separately, peak is the larger of the two.",
    ]

    out = ROOT_DIR / args.report
    out.write_text("\n".join(report))
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
)
 == "__main__":
    main()
