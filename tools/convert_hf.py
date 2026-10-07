#!/usr/bin/env python3
"""Reproducible Hugging Face -> GGUF conversion with a quality gate (#623).

Three subcommands:

  convert   a pinned HF revision -> GGUF with llama.cpp's converter at a
            pinned commit, optionally quantized with llama-quantize. Writes
            <out>.manifest.json: source repo + revision + the sha256 of every
            input file, the converter commit and arguments, the output's
            sha256 and size. --check-determinism converts twice into separate
            directories and fails unless the two outputs are byte-identical.

  gate      perplexity of a candidate GGUF against the source-precision GGUF
            (both run through geist's eval_geist on the same tokens) on a
            fixed public-domain text (tools/data/ppl_alice.txt). Fails when
            the relative perplexity increase exceeds the bound for the
            candidate's quantization (PPL_BOUNDS below), and writes the
            result into the candidate's manifest.

  manifest  a geist-runtime models/catalog.json entry from a manifest, so the
            entry is generated, not hand-written (geisten/geist-runtime#5).

Example (Qwen3 0.6B, Q8_0):

  tools/convert_hf.py convert --repo Qwen/Qwen3-0.6B --revision <40-hex sha> \\
      --quant q8_0 --llama-cpp ~/llama.cpp-cross-engine --out-dir out --check-determinism
  tools/convert_hf.py convert ... --quant bf16 ...        # the reference
  tools/convert_hf.py gate --ref out/qwen3-0.6b-bf16.gguf --cand out/qwen3-0.6b-q8_0.gguf
  tools/convert_hf.py manifest out/qwen3-0.6b-q8_0.gguf.manifest.json --id qwen3-0.6b

The reference-suite gate of geist-serve is not run here (the suite lives in
that repository); the manifest keeps a slot for its verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The llama.cpp commit the converter runs at: the one the cross-engine
# benchmark protocol pins (benchmark/cross_engine_*_protocol.json users).
LLAMA_PIN = "2d8d612e4c68d3801e556a1b4a028f55ec33ecbb"

# Output types the converter writes itself; everything else goes through
# llama-quantize from the BF16 conversion.
CONVERT_TYPES = {"f32", "f16", "bf16", "q8_0"}

# Allowed relative perplexity increase over the source-precision model on
# tools/data/ppl_alice.txt, by quantization. Generous where the literature
# and our own measurements put the usual loss (Q8_0 ~0.1 %, Q6_K ~0.5 %,
# Q4_K_M ~2-5 % on small models), so a bound trips on a broken conversion,
# not on ordinary quantization noise.
PPL_BOUNDS = {
    "f16": 0.005,
    "bf16": 0.005,
    "q8_0": 0.01,
    "q6_k": 0.02,
    "q5_k_m": 0.04,
    "q4_k_m": 0.08,
    "q4_0": 0.12,
}

# Small models lose more to the same quantization: Qwen3 0.6B Q4_K_M is
# +16.3 % here and +13.4 % with llama.cpp's own llama-perplexity on the same
# files. Below 2 B parameters (a source-precision file under 4 GB) these
# apply instead.
PPL_BOUNDS_SMALL = {
    "f16": 0.005,
    "bf16": 0.005,
    "q8_0": 0.02,  # SmolLM2 135M: +0.84 % on AVX-512; other kernels move it
    "q6_k": 0.03,
    "q5_k_m": 0.06,
    "q4_k_m": 0.20,
    "q4_0": 0.30,
}
SMALL_MODEL_BYTES = 4_000_000_000

# Files a conversion reads: weights, configs, tokenizer.
ALLOW = ["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "*.tiktoken"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           check=True, capture_output=True, text=True).stdout.strip()


def fetch(repo: str, revision: str, cache: Path | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=ALLOW,
                                  cache_dir=str(cache) if cache else None))


def llama_quantize(llama: Path) -> Path:
    """The pinned tree's llama-quantize, built on first use."""
    build = llama / "build-convert"
    exe = build / "bin" / "llama-quantize"
    if not exe.is_file():
        subprocess.run(["cmake", "-S", str(llama), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                        "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=OFF"], check=True, capture_output=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "llama-quantize", "-j"],
                       check=True, capture_output=True)
    return exe


def run(cmd: list[str]) -> None:
    """subprocess.run that shows the tool's own output when it fails."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        sys.exit(f"failed ({r.returncode}): {' '.join(cmd[:3])} ...")


def python_deps() -> dict[str, str]:
    """Versions of the converter's Python dependencies (they shape the output:
    Gemma 4's tokenizer needs transformers >= 5)."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("transformers", "torch", "numpy", "safetensors", "sentencepiece", "gguf"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def convert_once(src: Path, llama: Path, quant: str, out: Path) -> list[list[str]]:
    """One conversion into `out`; returns the commands it ran."""
    convert = [sys.executable, str(llama / "convert_hf_to_gguf.py"), str(src)]
    if quant in CONVERT_TYPES:
        cmd = convert + ["--outtype", quant, "--outfile", str(out)]
        run(cmd)
        return [cmd]
    tmp = out.with_suffix(".bf16.tmp.gguf")
    first = convert + ["--outtype", "bf16", "--outfile", str(tmp)]
    run(first)
    # one thread: llama-quantize's output does not depend on it, but say so
    second = [str(llama_quantize(llama)), str(tmp), str(out), quant.upper(), "1"]
    run(second)
    tmp.unlink()
    return [first, second]


def portable(cmd: list[str], src: Path, llama: Path, out_dir: Path) -> list[str]:
    """A command with this machine's paths replaced, so manifests compare."""
    subs = [(sys.executable, "python"), (str(llama), "<llama.cpp>"), (str(src), "<snapshot>"),
            (str(out_dir), "<out>")]
    out = []
    for c in cmd:
        for old, new in subs:
            c = c.replace(old, new)
        out.append(c)
    return out


def cmd_convert(a: argparse.Namespace) -> int:
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        sys.exit("--revision must be a full 40-hex commit sha (a branch or tag can move)")
    llama = Path(a.llama_cpp).expanduser().resolve()
    head = git_head(llama)
    if head != a.llama_pin:
        sys.exit(f"{llama} is at {head}, not the pinned {a.llama_pin} (git -C {llama} checkout it)")
    quant = a.quant.lower()
    src = fetch(a.repo, a.revision, Path(a.cache) if a.cache else None)
    inputs = {str(p.relative_to(src)): sha256_file(p) for p in sorted(src.rglob("*")) if p.is_file()}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = a.name or a.repo.split("/")[-1].lower()
    out = out_dir / f"{stem}-{quant}.gguf"
    cmds = convert_once(src, llama, quant, out)
    digest = sha256_file(out)
    determinism = None
    if a.check_determinism:
        with tempfile.TemporaryDirectory(dir=out_dir) as td:
            again = Path(td) / out.name
            convert_once(src, llama, quant, again)
            second = sha256_file(again)
        determinism = {"runs": 2, "identical": second == digest, "second_sha256": second}
        if second != digest:
            print(f"NOT REPRODUCIBLE: {digest} vs {second}", file=sys.stderr)
    manifest = {
        "schema": "geist.convert.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"repo": a.repo, "revision": a.revision, "files": inputs},
        "converter": {"llama_cpp": head, "python": python_deps(), "commands": [portable(cmd, src, llama, out_dir)
                                                      for cmd in cmds]},
        "output": {"file": out.name, "sha256": digest, "bytes": out.stat().st_size,
                   "quantization": quant},
        "determinism": determinism,
        "gate": None,
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{out}  sha256 {digest}" + (f"  reproducible: {determinism['identical']}" if determinism else ""))
    return 0 if determinism is None or determinism["identical"] else 1


class EvalGeist:
    """eval_geist as a subprocess (see tools/eval_geist.c for the protocol)."""

    def __init__(self, exe: Path, gguf: Path):
        self.p = subprocess.Popen([str(exe), str(gguf)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.p.stdout.readline().strip()  # announced once the model is loaded
        if ready != "READY":
            raise RuntimeError(f"eval_geist {gguf}: {ready!r} instead of READY")

    def ask(self, line: str) -> list[str]:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        reply = self.p.stdout.readline().split()
        if not reply or reply[0] != "OK":
            raise RuntimeError(f"eval_geist: {line[:40]}... -> {' '.join(reply)[:200]}")
        return reply[1:]

    def tokens(self, text: str) -> list[int]:
        esc = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
        r = self.ask(f"TOK {esc}")
        return [int(t) for t in r[1:1 + int(r[0])]]

    def close(self) -> None:
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
        except BrokenPipeError:
            pass
        self.p.wait(timeout=60)


def perplexity(exe: Path, gguf: Path, ids: list[int], window: int) -> float:
    """exp(-mean log p) over `ids` in windows of `window` tokens, each window's
    first token given as the prompt (it is not scored)."""
    ev = EvalGeist(exe, gguf)
    total, n = 0.0, 0
    try:
        # The model's own BOS policy (Gemma needs <bos>; llama-perplexity adds
        # it too): with BOS the whole window is scored after it, without it
        # the window's first token is the prompt.
        add_bos, bos = (int(x) for x in ev.ask("BOS")[:2])
        for w in range(0, len(ids) - 1, window):
            chunk = ids[w:w + window]
            if len(chunk) < 2:
                break
            prompt, cont = ([bos], chunk) if add_bos else ([chunk[0]], chunk[1:])
            ev.ask("RESET")
            r = ev.ask(f"SCORE {len(prompt)} " + " ".join(map(str, prompt)) +
                       f" {len(cont)} " + " ".join(map(str, cont)))
            lps = [float(x) for x in r[1:]]
            total += sum(lps)
            n += len(lps)
    finally:
        ev.close()
    return math.exp(-total / n)


def cmd_gate(a: argparse.Namespace) -> int:
    exe = Path(a.eval_geist)
    if not exe.is_file():
        sys.exit(f"{exe} not built: make bin")
    cand_manifest = Path(str(a.cand) + ".manifest.json")
    quant = a.quant
    if quant is None and cand_manifest.is_file():
        quant = json.loads(cand_manifest.read_text())["output"]["quantization"]
    if quant not in PPL_BOUNDS:
        sys.exit(f"no perplexity bound for quantization {quant!r}; pass --bound")
    small = Path(a.ref).stat().st_size < SMALL_MODEL_BYTES
    bound = a.bound if a.bound is not None else (PPL_BOUNDS_SMALL if small else PPL_BOUNDS)[quant]
    text = Path(a.text).read_text()
    # tokenize with the reference: both files must share the tokenizer
    ev = EvalGeist(exe, Path(a.ref))
    ids = ev.tokens(text)
    ev.close()
    ref_ppl = perplexity(exe, Path(a.ref), ids, a.window)
    cand_ppl = perplexity(exe, Path(a.cand), ids, a.window)
    rel = cand_ppl / ref_ppl - 1.0
    ok = rel <= bound
    result = {
        "text": Path(a.text).name, "text_sha256": sha256_file(Path(a.text)), "tokens": len(ids),
        "window": a.window, "ref": Path(a.ref).name, "ref_sha256": sha256_file(Path(a.ref)),
        "ppl_ref": round(ref_ppl, 4), "ppl_cand": round(cand_ppl, 4),
        "ppl_rel_increase": round(rel, 5), "bound": bound, "small_model_bounds": small, "pass": ok,
        "engine": git_head(ROOT), "reference_suite": None,
    }
    if cand_manifest.is_file():
        m = json.loads(cand_manifest.read_text())
        m["gate"] = result
        cand_manifest.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(("PASS" if ok else "FAIL") + f": perplexity {ref_ppl:.3f} -> {cand_ppl:.3f} "
          f"({rel * 100:+.2f} %, bound {bound * 100:.1f} %)")
    return 0 if ok else 1


def cmd_manifest(a: argparse.Namespace) -> int:
    m = json.loads(Path(a.manifest).read_text())
    if m.get("gate") is None or not m["gate"]["pass"]:
        sys.exit("the manifest has no passed quality gate: run `gate` first")
    out = m["output"]
    entry = {
        "id": a.id,
        "name": a.display_name or a.id,
        "file": out["file"],
        "url": a.url or "",
        "sha256": out["sha256"],
        "bytes": out["bytes"],
        "quantization": out["quantization"].upper(),
        "provenance": {
            "repo": m["source"]["repo"], "revision": m["source"]["revision"],
            "llama_cpp": m["converter"]["llama_cpp"],
            "reproducible": bool(m.get("determinism") and m["determinism"]["identical"]),
        },
        "gate": {k: m["gate"][k] for k in ("ppl_ref", "ppl_cand", "ppl_rel_increase", "bound",
                                            "engine", "text_sha256")},
    }
    print(json.dumps(entry, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="pinned HF revision -> GGUF + manifest")
    c.add_argument("--repo", required=True)
    c.add_argument("--revision", required=True, help="full commit sha of the HF repo")
    c.add_argument("--quant", required=True, help="f16, bf16, q8_0 or a llama-quantize type (q4_k_m ...)")
    c.add_argument("--llama-cpp", required=True, help="llama.cpp checkout at the pinned commit")
    c.add_argument("--llama-pin", default=LLAMA_PIN)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--name", help="output file stem (default: the repo name, lower case)")
    c.add_argument("--cache", help="huggingface_hub cache directory")
    c.add_argument("--check-determinism", action="store_true")
    g = sub.add_parser("gate", help="perplexity of a candidate against the source precision")
    g.add_argument("--ref", required=True)
    g.add_argument("--cand", required=True)
    g.add_argument("--quant", help="bound key (default: from the candidate's manifest)")
    g.add_argument("--bound", type=float, help="override the relative perplexity bound")
    g.add_argument("--text", default=str(ROOT / "tools" / "data" / "ppl_alice.txt"))
    g.add_argument("--window", type=int, default=512)
    g.add_argument("--eval-geist", default=str(ROOT / "bin" / "linux" / "release" / "tools" / "eval_geist"))
    m = sub.add_parser("manifest", help="geist-runtime catalog entry from a gated manifest")
    m.add_argument("manifest")
    m.add_argument("--id", required=True)
    m.add_argument("--display-name")
    m.add_argument("--url")
    a = p.parse_args()
    return {"convert": cmd_convert, "gate": cmd_gate, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
