#!/usr/bin/env python3
"""mlx_vs_geist.py — driver of tools/probes/mlx_vs_geist.sh: geist (Metal),
mlx-lm and llama.cpp (Metal) on the same models, interleaved, on one Mac.

    mlx_vs_geist.py --cache DIR --gguf-cache DIR --out DIR
                    [--models a,b] [--engines geist,mlx,llama] [--cycles 3] [--smoke]
    mlx_vs_geist.py --report results.jsonl [more.jsonl ...]     the table, on stdout

Per model and cycle the engines run in rotating order. An engine's turn is:
gate (AC power, idle CPU, cool chip), time to ready with the weights evicted
from the page cache ("cold"), time to ready again ("warm"), then the
throughput run: 512 prompt tokens of benchmark/prompts/moe-long.txt in the
engine's own tokenizer, 128 greedy tokens, one discarded warmup, 3 repeats.
Every process runs under /usr/bin/time -l (peak RSS, peak memory footprint).
Each finished cell is appended to <out>/results.jsonl at once.

Runs on the host's system Python (3.9 on the Mac mini): stdlib only, no
syntax newer than that at run time. The MLX side is mlx_vs_geist_mlx.py in
the probe's own virtual environment under --cache.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import bench_host  # noqa: E402

PINS = json.loads((HERE / "mlx_vs_geist_models.json").read_text())
PROTOCOL = ROOT / "benchmark" / "cross_engine_gpu_protocol.json"
PROMPT_TOKENS = 512
DECODE_N = 128
GEIST_WARMUP = 64  # tools/bench_cross_engine.py run_geist
# Dispatch-side threads of the apple_m6_mini_metal profile; the CPU cell uses
# the apple_m6_mini profile of the CPU protocol.
METAL_THREADS = (4, 4)
CPU_THREADS = (6, 5)
COOL = {"max_gpu_c": 50, "max_cpu_c": 50, "fallback_seconds": 45}
MIN_IDLE_PCT = 80
GATE_TIMEOUT = 900


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def clean_env(extra: dict | None = None) -> dict:
    """The environment of every child: no Hugging Face token, no telemetry."""
    env = dict(os.environ)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        env.pop(key, None)
    # LC_ALL=C: /usr/bin/time prints its seconds in the locale's number format.
    env.update({"HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1", "LC_ALL": "C", "PYTHONUTF8": "1"})
    env.update(extra or {})
    return env


# --- page cache ---------------------------------------------------------

def page_cache(path: Path, evict: bool) -> dict:
    """Share of the file's pages in the page cache (mincore), after dropping
    them with msync(MS_INVALIDATE) when evict is set. No root needed; pages
    another process still maps or wires stay."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_longlong]
    libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    size = path.stat().st_size
    page = os.sysconf("SC_PAGESIZE")
    fd = os.open(str(path), os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
        if addr is None or addr == ctypes.c_void_p(-1).value:
            return {"error": f"mmap errno {ctypes.get_errno()}"}
        vec = ctypes.create_string_buffer((size + page - 1) // page)

        def resident() -> float:
            if libc.mincore(addr, size, vec) != 0:
                return -1.0
            raw = vec.raw
            return round(sum(1 for b in raw if b & 1) / len(raw), 4)

        out = {"resident_before": resident()}
        if evict:
            out["msync_rc"] = libc.msync(addr, size, 2)  # MS_INVALIDATE
            out["resident_after"] = resident()
        libc.munmap(addr, size)
        return out
    finally:
        os.close(fd)


# --- GGUF header: parameter count --------------------------------------

def gguf_params(path: Path) -> dict:
    """Tensor count and total elements from the GGUF header (v2/v3)."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q",
              12: "d"}
    with path.open("rb") as fh:
        def rd(fmt: str):
            return struct.unpack("<" + fmt, fh.read(struct.calcsize("<" + fmt)))[0]

        def rd_str() -> bytes:
            return fh.read(rd("Q"))

        def skip(kind: int) -> None:
            if kind == 8:
                rd_str()
            elif kind == 9:
                inner, count = rd("I"), rd("Q")
                if inner in scalar:
                    fh.seek(struct.calcsize(scalar[inner]) * count, 1)
                else:
                    for _ in range(count):
                        skip(inner)
            else:
                rd(scalar[kind])

        if fh.read(4) != b"GGUF":
            return {"error": "not a GGUF file"}
        rd("I")
        n_tensors, n_kv = rd("Q"), rd("Q")
        arch = None
        for _ in range(n_kv):
            key, kind = rd_str(), rd("I")
            if key == b"general.architecture" and kind == 8:
                arch = rd_str().decode()
            else:
                skip(kind)
        total, types = 0, {}
        for _ in range(n_tensors):
            rd_str()
            dims = [rd("Q") for _ in range(rd("I"))]
            dtype = rd("I")
            rd("Q")
            count = 1
            for d in dims:
                count *= d
            total += count
            types[dtype] = types.get(dtype, 0) + count
    size = path.stat().st_size
    return {"architecture": arch, "tensors": n_tensors, "weights": total, "file_bytes": size,
            "bits_per_weight": round(size * 8 / total, 3) if total else None,
            "weights_by_ggml_type": {str(k): v for k, v in sorted(types.items())}}


# --- setup --------------------------------------------------------------

def run_checked(command: list, env: dict | None = None, label: str = "command",
                timeout: int | None = None) -> str:
    proc = subprocess.run([str(c) for c in command], capture_output=True, text=True, env=env,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} exited {proc.returncode}: "
                           f"{(proc.stderr or proc.stdout).strip()[-2000:]}")
    return proc.stdout


def uv_env(cache: Path) -> dict:
    return clean_env({"UV_CACHE_DIR": str(cache / "uv-cache"),
                      "UV_PYTHON_INSTALL_DIR": str(cache / "python"),
                      "UV_PYTHON_PREFERENCE": "only-managed", "UV_NO_CONFIG": "1",
                      "UV_NO_PROGRESS": "1", "HF_HOME": str(cache / "hf")})


def ensure_venv(cache: Path) -> dict:
    """uv, a standalone CPython and the pinned packages, all under cache."""
    pin = PINS["pins"]
    uv = cache / "bin" / "uv"
    if not uv.exists():
        cache.mkdir(parents=True, exist_ok=True)
        tarball = cache / "uv.tar.gz"
        log(f"fetching uv {pin['uv']['version']}")
        with urllib.request.urlopen(pin["uv"]["url"]) as resp, tarball.open("wb") as out:
            out.write(resp.read())
        got = bench_host.sha256_file(tarball)
        if got != pin["uv"]["sha256"]:
            tarball.unlink()
            raise RuntimeError(f"uv tarball sha256 {got} != pinned {pin['uv']['sha256']}")
        uv.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/uv") and m.isfile())
            with tar.extractfile(member) as src, uv.open("wb") as dst:
                dst.write(src.read())
        uv.chmod(0o755)
        tarball.unlink()
    env = uv_env(cache)
    python = cache / "venv" / "bin" / "python"
    stamp = cache / "venv" / ".geist-packages"
    want = " ".join(pin["packages"])
    if not python.exists() or not stamp.exists() or stamp.read_text() != want:
        log(f"creating venv (python {pin['python']}; {want})")
        run_checked([uv, "venv", "--clear", "--python", pin["python"], cache / "venv"], env,
                    "uv venv", 1800)
        run_checked([uv, "pip", "install", "--python", python, *pin["packages"]], env,
                    "uv pip install", 1800)
        stamp.write_text(want)
    return {
        "uv": run_checked([uv, "--version"], env).strip(),
        "python": run_checked([python, "--version"]).strip(),
        "packages": run_checked([uv, "pip", "freeze", "--python", python], env).split(),
        "pinned": pin["packages"],
    }


def worker(cache: Path, offline: bool = True) -> tuple:
    env = clean_env({"HF_HOME": str(cache / "hf"), "TOKENIZERS_PARALLELISM": "false"})
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    return [str(cache / "venv" / "bin" / "python"), str(HERE / "mlx_vs_geist_mlx.py")], env


def result_of(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError("worker printed no RESULT line")


def ensure_model(key: str, cache: Path, gguf_cache: Path) -> dict:
    """Both files of a pair in their caches, hashes checked; what was fetched."""
    spec = PINS["models"][key]
    fetch = [sys.executable, str(ROOT / "tools" / "fetch_bench_model.py"), "--cache",
             str(gguf_cache)]
    gguf = spec["gguf"]
    if "protocol_key" in gguf:
        fetch += ["--model", gguf["protocol_key"], "--protocol", str(PROTOCOL)]
        sha = json.loads(PROTOCOL.read_text())["models"][gguf["protocol_key"]]["sha256"]
        sources = json.loads((ROOT / "benchmark" / "model_sources.json").read_text())["sources"]
        url = sources[sha]["url"]
    else:
        fetch += ["--url", gguf["url"], "--sha256", gguf["sha256"]]
        sha, url = gguf["sha256"], gguf["url"]
    log(f"{key}: GGUF")
    gguf_path = Path(run_checked(fetch, clean_env(), "fetch_bench_model").strip().splitlines()[-1])
    log(f"{key}: MLX snapshot {spec['mlx']['repo']}")
    command, env = worker(cache, offline=False)
    snap = result_of(run_checked(command + ["download", "--repo", spec["mlx"]["repo"],
                                            "--revision", spec["mlx"]["revision"]], env,
                                 "mlx download", 7200))
    if snap["weights"] != spec["mlx"]["weights"]:
        raise RuntimeError(f"{key}: MLX weights {snap['weights']} != pinned")
    return {
        "key": key, "label": spec["label"],
        "gguf": {"path": str(gguf_path), "url": url, "sha256": sha, "quant": gguf["quant"],
                 **gguf_params(gguf_path)},
        "mlx": {"repo": spec["mlx"]["repo"], "revision": spec["mlx"]["revision"], **snap},
    }


def ensure_llama() -> dict:
    proc = subprocess.run(["sh", str(ROOT / "scripts" / "ensure-llama-bench.sh"), "metal"],
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"ensure-llama-bench failed: {proc.stderr.strip()[-1500:]}")
    binary = Path(proc.stdout.strip().splitlines()[-1])
    commit = run_checked(["git", "-C", binary.parent.parent.parent, "rev-parse", "HEAD"]).strip()
    return {"binary": str(binary), "commit": commit}


# --- gates and cells ----------------------------------------------------

def gate(smoke: bool, limits: dict = COOL) -> dict:
    """AC power, >= 80 % idle CPU and a cool chip before a cell; what it saw."""
    if smoke:
        return {"smoke": True, "load_1m": os.getloadavg()[0], "power": bench_host.apple_power()}
    deadline = time.monotonic() + GATE_TIMEOUT
    state = dict(bench_host.wait_for_apple_power(deadline))
    while True:
        idle = bench_host.cpu_idle_pct()
        if idle is None or idle >= MIN_IDLE_PCT:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"host busy: {idle:.1f} % idle")
        log(f"  waiting for idle host: {idle:.1f} % idle")
        time.sleep(10)
    state["cpu_idle_pct"] = idle
    state.update(bench_host.wait_for_apple_cool(limits, deadline))
    state["load_1m"] = round(os.getloadavg()[0], 2)
    return state


TIME_RE = {
    "real_s": re.compile(r"^\s*([\d.]+) real\b", re.MULTILINE),
    "max_rss_bytes": re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE),
    "peak_footprint_bytes": re.compile(r"^\s*(\d+)\s+peak memory footprint", re.MULTILINE),
}


def timed(command: list, env: dict, timeout: int = 1800) -> dict:
    """Run under /usr/bin/time -l: stdout, stderr without time's block, and
    time's wall seconds, peak RSS and peak memory footprint."""
    started = time.monotonic()
    proc = subprocess.run(["/usr/bin/time", "-l", *[str(c) for c in command]],
                          capture_output=True, text=True, env=env, timeout=timeout)
    out = {"wall_s": round(time.monotonic() - started, 3), "returncode": proc.returncode,
           "stdout": proc.stdout}
    for name, pattern in TIME_RE.items():
        match = pattern.search(proc.stderr)
        out[name] = float(match.group(1)) if match else None
    cut = TIME_RE["real_s"].search(proc.stderr)
    out["stderr"] = proc.stderr[:cut.start()] if cut else proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {out['stderr'].strip()[-1500:]}")
    return out


def proc_stats(run: dict) -> dict:
    return {k: run[k] for k in ("wall_s", "real_s", "max_rss_bytes", "peak_footprint_bytes")}


def geist_env(backend: str) -> dict:
    prefill, decode = METAL_THREADS if backend == "metal" else CPU_THREADS
    return clean_env({"OMP_WAIT_POLICY": "active", "OMP_NUM_THREADS": str(prefill),
                      "GEIST_PREFILL_THREADS": str(prefill), "GEIST_DECODE_THREADS": str(decode),
                      "GEIST_TEXT_ONLY": "1", "GEIST_BENCH_BACKEND": backend})


def geist_rows(run: dict, backend: str) -> list:
    if f"[bench] backend: {backend}" not in run["stderr"]:
        raise RuntimeError(f"geist did not select the {backend} backend")
    return [json.loads(line) for line in run["stdout"].splitlines() if line.startswith("{")]


def cell_geist(ctx: dict, model: dict, backend: str, repeats: int) -> dict:
    binary, gguf, env = ctx["geist"], model["gguf"]["path"], geist_env(backend)
    threads = env["GEIST_PREFILL_THREADS"]
    ready = [str(binary), "--gguf", gguf, "--seq-lens", "1", "--decode-n", "1", "--warmup", "0",
             "--repeats", "1", "--threads", threads]
    out = {"load": {}}
    for kind in ("cold", "warm"):
        cache = page_cache(Path(gguf), evict=kind == "cold")
        run = timed(ready, env)
        geist_rows(run, backend)
        out["load"][kind] = {"page_cache": cache, **proc_stats(run)}
    bench = [str(binary), "--gguf", gguf, "--seq-lens", str(PROMPT_TOKENS), "--decode-n",
             str(DECODE_N), "--warmup", str(GEIST_WARMUP), "--repeats", str(repeats),
             "--prompt-file", str(ROOT / PINS["prompt_file"]), "--emit-tokens", "--threads",
             threads]
    run = timed(bench, env)
    row = geist_rows(run, backend)[0]
    prompt = re.search(r"\[bench\] prompt: (\d+) tokens", run["stderr"])
    out.update({
        "prefill_tps": [PROMPT_TOKENS * 1000.0 / ms for ms in row["samples"]["prefill_ms"]],
        "decode_tps": [DECODE_N * 1000.0 / ms for ms in row["samples"]["decode_ms"]],
        "ttft_ms": row["samples"]["ttft_ms"],
        "prompt_file_tokens": int(prompt.group(1)) if prompt else None,
        "tokens": row.get("tokens", []),
        "native": {k: v for k, v in row.items() if k != "tokens"},
        "process": proc_stats(run),
        "stderr_tail": run["stderr"].strip().splitlines()[-6:],
    })
    return out


def cell_mlx(ctx: dict, model: dict, repeats: int) -> dict:
    command, env = worker(ctx["cache"])
    path = model["mlx"]["path"]
    prompt = str(ROOT / PINS["prompt_file"])
    out = {"load": {}}
    for kind in ("cold", "warm"):
        cache = [page_cache(Path(f), evict=kind == "cold") for f in model["mlx"]["weight_files"]]
        run = timed(command + ["load", "--model", path, "--prompt-file", prompt], env)
        out["load"][kind] = {"page_cache": cache, **proc_stats(run), "native": result_of(run["stdout"])}
    run = timed(command + ["bench", "--model", path, "--prompt-file", prompt, "--prompt-tokens",
                           str(PROMPT_TOKENS), "--decode-n", str(DECODE_N), "--warmup", "1",
                           "--repeats", str(repeats)], env)
    native = result_of(run["stdout"])
    runs = native["runs"]
    out.update({
        "prefill_tps": [r["lib_prompt_tps"] for r in runs],
        "decode_tps": [r["lib_generation_tps"] for r in runs],
        "wall_decode_tps": [r["wall_decode_tps"] for r in runs],
        "ttft_ms": [r["ttft_ms"] for r in runs],
        "prompt_file_tokens": native["prompt_file_tokens"],
        "tokens": runs[0]["tokens"],
        "native": native,
        "process": proc_stats(run),
        "stderr_tail": run["stderr"].strip().splitlines()[-6:],
    })
    return out


def cell_llama(ctx: dict, model: dict, repeats: int) -> dict:
    """llama-bench as tools/bench_cross_engine.py run_llama calls it on a GPU
    column: its own random prompt tokens, prefill and decode in two calls."""
    common = [ctx["llama"]["binary"], "-m", model["gguf"]["path"], "-ngl", "99", "-r",
              str(repeats), "-o", "jsonl", "-t", str(METAL_THREADS[0]), "--poll", "100"]
    prefill = timed(common + ["-p", str(PROMPT_TOKENS), "-n", "0"], clean_env())
    decode = timed(common + ["-p", "0", "-n", str(DECODE_N), "-d", str(PROMPT_TOKENS)],
                   clean_env())
    rows = [[json.loads(line) for line in run["stdout"].splitlines() if line.startswith("{")][0]
            for run in (prefill, decode)]
    if "MTL" not in rows[0].get("backends", "") or rows[0].get("n_gpu_layers") != 99:
        raise RuntimeError("llama-bench did not run fully offloaded on Metal")
    return {
        "prefill_tps": rows[0]["samples_ts"], "decode_tps": rows[1]["samples_ts"],
        "native": {"prefill": rows[0], "decode": rows[1]},
        "process": {"prefill": proc_stats(prefill), "decode": proc_stats(decode)},
    }


def run_cell(ctx: dict, model: dict, engine: str, cycle: int, position: int) -> dict:
    record = {"kind": "cell", "model": model["key"], "engine": engine, "cycle": cycle,
              "position": position, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    repeats = 1 if ctx["smoke"] else 3
    try:
        limits = dict(COOL, max_cpu_c=45) if engine == "geist-cpu" else COOL
        record["gate"] = gate(ctx["smoke"], limits)
        if engine == "geist":
            record.update(cell_geist(ctx, model, "metal", repeats))
        elif engine == "geist-cpu":
            record.update(cell_geist(ctx, model, "cpu_neon", repeats))
        elif engine == "mlx":
            record.update(cell_mlx(ctx, model, repeats))
        else:
            record.update(cell_llama(ctx, model, repeats))
        log(f"  {engine}: prefill {statistics.median(record['prefill_tps']):.1f} tok/s, "
            f"decode {statistics.median(record['decode_tps']):.1f} tok/s")
    except Exception as error:  # a failed cell must not lose the others
        record["error"] = f"{type(error).__name__}: {error}"
        log(f"  {engine}: FAILED {record['error'][:400]}")
    return record


def sanity(ctx: dict, model: dict, cells: list) -> dict:
    """The first 40 generated tokens of each engine as text, and how long the
    two greedy continuations agree."""
    geist = next((c for c in cells if c["engine"] == "geist" and c.get("tokens")), None)
    mlx = next((c for c in cells if c["engine"] == "mlx" and c.get("tokens")), None)
    record = {"kind": "sanity", "model": model["key"]}
    if not geist or not mlx:
        record["error"] = "one engine has no tokens"
        return record
    command, env = worker(ctx["cache"])
    try:
        run = timed(command + ["detok", "--model", model["mlx"]["path"], "--prompt-file",
                               str(ROOT / PINS["prompt_file"]), "--ids",
                               ",".join(map(str, geist["tokens"][:40]))], env)
        detok = result_of(run["stdout"])
        common = 0
        for a, b in zip(geist["tokens"], mlx["tokens"]):
            if a != b:
                break
            common += 1
        record.update({
            "geist_text_first_40": detok["text"],
            "mlx_text_first_40": mlx["native"]["text_first_40"],
            "common_prefix_tokens": common,
            "of_tokens": min(len(geist["tokens"]), len(mlx["tokens"])),
            "geist_prompt_file_tokens": geist["prompt_file_tokens"],
            "mlx_prompt_file_tokens": mlx["prompt_file_tokens"],
        })
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


# --- report -------------------------------------------------------------

def med_range(values: list, digits: int = 1) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    return (f"{statistics.median(values):.{digits}f} "
            f"({min(values):.{digits}f}–{max(values):.{digits}f})")


def gib(values: list) -> str:
    values = [v for v in values if v]
    return f"{max(values) / 2**30:.2f}" if values else "—"


def report(records: list) -> str:
    models = [r for r in records if r.get("kind") == "model"]
    cells = [r for r in records if r.get("kind") == "cell" and "error" not in r]
    failed = [r for r in records if r.get("kind") == "cell" and "error" in r]
    names = {"geist": "geist Metal", "mlx": "mlx-lm", "llama": "llama.cpp Metal",
             "geist-cpu": "geist CPU"}
    lines = ["| model | engine | prefill tok/s @512 | decode tok/s (128) | TTFT ms @512 | "
             "ready cold s | ready warm s | peak footprint GiB | peak RSS GiB | n |",
             "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |"]
    for model in models:
        for engine in ("geist", "mlx", "llama", "geist-cpu"):
            mine = [c for c in cells if c["model"] == model["key"] and c["engine"] == engine]
            if not mine:
                continue
            flat = lambda key: [v for c in mine for v in c.get(key, [])]  # noqa: E731
            if engine == "llama":
                procs = [p for c in mine for p in c["process"].values()]
            else:
                procs = [c["process"] for c in mine]
            load = lambda kind: [c["load"][kind]["real_s"] for c in mine if "load" in c]  # noqa: E731
            lines.append(
                f"| {model['label']} | {names[engine]} | {med_range(flat('prefill_tps'))} | "
                f"{med_range(flat('decode_tps'))} | {med_range(flat('ttft_ms'))} | "
                f"{med_range(load('cold'), 2)} | {med_range(load('warm'), 2)} | "
                f"{gib([p['peak_footprint_bytes'] for p in procs])} | "
                f"{gib([p['max_rss_bytes'] for p in procs])} | {len(flat('prefill_tps'))} |")
    for cell in failed:
        lines.append(f"\nFAILED {cell['model']} / {cell['engine']} cycle {cell['cycle']}: "
                     f"{cell['error'][:300]}")
    return "\n".join(lines) + "\n"


# --- main ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", nargs="+", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--gguf-cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--geist", type=Path,
                        default=ROOT / "bin" / "mac-omp" / "release" / "tests" / "bench_perf_sweep")
    parser.add_argument("--models", default=",".join(PINS["models"]))
    parser.add_argument("--engines", default="geist,mlx,llama")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--smoke", action="store_true",
                        help="one cycle, one repeat, no gates: plumbing only")
    args = parser.parse_args()
    if args.report:
        records = [json.loads(line) for path in args.report
                   for line in path.read_text().splitlines() if line.strip()]
        print(report(records), end="")
        return 0

    cache = args.cache.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.jsonl"
    records: list = []

    def emit(record: dict) -> None:
        records.append(record)
        with results.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    engines = [e for e in args.engines.split(",") if e]
    keys = [k for k in args.models.split(",") if k]
    ctx = {"cache": cache, "geist": args.geist, "smoke": args.smoke}
    meta = {
        "kind": "metadata",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geistlib_commit": bench_host.command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "geist_binary_sha256": bench_host.sha256_file(args.geist),
        "host": {k: bench_host.command_output(["sysctl", "-n", v]) for k, v in {
            "cpu": "machdep.cpu.brand_string", "model": "hw.model", "memory_bytes": "hw.memsize",
            "ncpu": "hw.ncpu"}.items()},
        "os": bench_host.command_output(["sw_vers", "-productVersion"]),
        "macmon": bench_host.find_macmon() is not None,
        "power": bench_host.apple_power(),
        "workload": {"prompt_tokens": PROMPT_TOKENS, "decode_n": DECODE_N,
                     "prompt_file": PINS["prompt_file"],
                     "prompt_file_sha256": bench_host.sha256_file(ROOT / PINS["prompt_file"]),
                     "geist_warmup": GEIST_WARMUP, "metal_threads": METAL_THREADS,
                     "cpu_threads": CPU_THREADS, "cool_gate": COOL, "min_idle_pct": MIN_IDLE_PCT,
                     "cycles": args.cycles, "smoke": args.smoke, "engines": engines},
        "venv": ensure_venv(cache),
    }
    if "llama" in engines:
        ctx["llama"] = meta["llama"] = ensure_llama()
    emit(meta)
    log(f"venv: {meta['venv']['python']}, {meta['venv']['pinned']}; macmon {meta['macmon']}")

    for key in keys:
        try:
            model = ensure_model(key, cache, args.gguf_cache.expanduser())
        except Exception as error:
            emit({"kind": "model_error", "key": key, "error": f"{type(error).__name__}: {error}"})
            log(f"{key}: FAILED {error}")
            continue
        emit({"kind": "model", **model})
        mine = list(engines)
        if PINS["models"][key].get("cpu") and "geist" in engines:
            mine.append("geist-cpu")
        variants = [{"name": name} for name in mine]
        cycles = 1 if args.smoke else args.cycles
        cells = []
        for cycle, position, variant in bench_host.build_schedule_rotating(variants, cycles):
            log(f"{key} cycle {cycle + 1}/{cycles}: {variant['name']}")
            cell = run_cell(ctx, model, variant["name"], cycle, position)
            cells.append(cell)
            emit(cell)
        emit(sanity(ctx, model, cells))

    hf = cache / "hf"
    sizes = {str(p): int(run_checked(["du", "-sk", p]).split()[0]) * 1024
             for p in (cache, hf, cache / "venv") if p.exists()}
    emit({"kind": "left_on_host", "bytes": sizes, "gguf_cache": str(args.gguf_cache)})
    (args.out / "summary.md").write_text(
        f"## mlx-lm vs geist vs llama.cpp ({'smoke' if args.smoke else 'measured'})\n\n"
        "median (min–max) over all repeats of all cycles; ready = process start to first token\n\n"
        + report(records)
        + "\n```\n" + "\n".join(json.dumps(r) for r in records if r["kind"] == "sanity")
        + "\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
