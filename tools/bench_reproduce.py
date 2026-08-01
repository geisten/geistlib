#!/usr/bin/env python3
"""bench_reproduce.py — the reproducible benchmark behind `make bench`.

Run one command, get a report you can paste into an issue or a comment:

    make bench

It measures geist, and — if a baseline engine's binary is reachable — measures
that too, in the same run under the same protocol. It never asks for
privileges, never refuses to run, and never writes into the repository.

Design notes, because each one is load-bearing:

* The protocol is FROZEN (see PROTOCOL). Every row in reference_runs.json was
  produced by it. Change it and every stored row becomes incomparable, which is
  exactly how the old headline table stopped meaning anything.

* An absolute t/s number cannot be checked by a stranger on unknown hardware.
  What can be checked is a RATIO against a baseline engine on the same box, so
  the baseline is measured whenever its binary is present and the report says
  plainly when it is not.

* The machine is measured, not commanded. We cannot quiesce someone else's
  laptop, so instead the run-to-run spread, load and temperature go into the
  report and the reader can judge the number for themselves. A wide spread is
  information, not a failure.

Exit codes follow tests/README.md: 0 PASS, 77 SKIPPED, 99 harness error.
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP, ERROR = 77, 99

# --- The frozen protocol ----------------------------------------------------
# Rows in reference_runs.json are only comparable because they all used this.
# Treat a change here as a dataset migration, not a tweak.
PROTOCOL = {
    "seq_lens": "32,128,512",
    "decode_n": 64,
    "warmup": 64,
    "repeats": 10,
    "agg": "mean",
}

# Spread above this reads as "the box was busy", from measurement: a quiesced
# Pi 5 lands at 0.4-1.3 %, a live desktop at ~20 %.
CLEAN_SPREAD_PCT = 2.0

# Load per core above this means the machine had other work. 0.7 is the usual
# rule of thumb; it exists because a starved run can post a *tight* spread while
# being uniformly slow, which reads as clean and is not.
BUSY_LOAD_PER_CORE = 0.7

# Both engines must start from the same thermal state. A passively cooled board
# reaches its soft limit during one sweep -- measured here, 48 -> 74 C -- and
# whichever engine runs second is throttled. Measured cost of ignoring it:
# bitnet.cpp's prefill read 37.5 t/s straight after geist against 44.5 on a
# cool board, a 16 % handicap invented by the running order and pointing the
# wrong way, in our favour.
COOL_C = 56.0
COOL_TIMEOUT_S = 600

# Weight bytes actually read per decode token for the default model. Derived in
# benchmark/results/TERNARY.md from the GGUF tensor table, not measured here.
# ponytail: one model only. Other models report no throughput rather than a
# number nobody can check — extend by doing the accounting, not by guessing.
PER_TOKEN_BYTES = {"bitnet-2b4t-i2_s.gguf": 582.0 * 1024 * 1024}


# --- System identity --------------------------------------------------------


def read_cpu_name() -> str:
    if sys.platform == "darwin":
        return sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    for path, pattern in (
        ("/proc/device-tree/model", None),  # Raspberry Pi names itself here
        ("/proc/cpuinfo", r"^model name\s*:\s*(.+)$"),
        ("/proc/cpuinfo", r"^Model\s*:\s*(.+)$"),
    ):
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        if pattern is None:
            return text.strip("\x00\n ")
        m = re.search(pattern, text, re.M)
        if m:
            return m.group(1).strip()
    return platform.processor() or "unknown"


def read_ram_gb() -> float:
    if sys.platform == "darwin":
        out = sh(["sysctl", "-n", "hw.memsize"])
        return round(int(out) / 2**30, 1) if out else 0.0
    try:
        m = re.search(r"MemTotal:\s+(\d+) kB", Path("/proc/meminfo").read_text())
        return round(int(m.group(1)) / 2**20, 1) if m else 0.0
    except OSError:
        return 0.0


def system_info() -> dict:
    return {
        "cpu": read_cpu_name(),
        "cores": os.cpu_count() or 0,
        "ram_gb": read_ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
    }


def sh(cmd) -> str:
    """Run a command, return stripped stdout, or "" if it fails in any way."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# --- Machine state: measured, never enforced --------------------------------


def load_avg() -> float:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return -1.0


def temp_c() -> float:
    """Board temperature where it is readable without privileges."""
    out = sh(["vcgencmd", "measure_temp"])  # Raspberry Pi
    if out:
        m = re.search(r"([\d.]+)", out)
        if m:
            return float(m.group(1))
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            return int(p.read_text()) / 1000.0
        except (OSError, ValueError):
            continue
    return -1.0


class EnergySampler:
    """Board power via the Pi's PMIC, sampled in the background.

    Unprivileged on Raspberry Pi OS (verified: `vcgencmd pmic_read_adc` needs
    no sudo). Everywhere else energy is root-only — x86 RAPL has been 0400
    since CVE-2020-8694, macOS powermetrics wants sudo — and a benchmark that
    asks a stranger for root has already lost the argument it came to win. So
    on other platforms this reports nothing and the report says why.
    """

    RE = re.compile(r"(\S+)_(?:A|V)\s+(current|volt)\(\d+\)=([\d.]+)")

    def __init__(self):
        self.available = shutil.which("vcgencmd") is not None and bool(
            sh(["vcgencmd", "pmic_read_adc"])
        )
        self._stop = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._thread: threading.Thread | None = None

    def _read_watts(self) -> float | None:
        out = sh(["vcgencmd", "pmic_read_adc"])
        if not out:
            return None
        amps, volts = {}, {}
        for rail, kind, value in self.RE.findall(out):
            (amps if kind == "current" else volts)[rail] = float(value)
        # Power is the sum over rails that report both a current and a voltage.
        return sum(a * volts[r] for r, a in amps.items() if r in volts) or None

    def _loop(self):
        while not self._stop.wait(0.2):
            w = self._read_watts()
            if w is not None:
                self._samples.append((time.monotonic(), w))

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False

    def summary(self) -> dict | None:
        """Mean watts and total joules over the sampled window."""
        if len(self._samples) < 2:
            return None
        span = self._samples[-1][0] - self._samples[0][0]
        mean_w = sum(w for _, w in self._samples) / len(self._samples)
        return {"mean_w": mean_w, "joules": mean_w * span, "samples": len(self._samples),
                "seconds": span}


# --- Measurement ------------------------------------------------------------


def find_geist_bench(target: str, mode: str) -> Path | None:
    exact = ROOT / "bin" / target / mode / "tests" / "bench_perf_sweep"
    if exact.exists():
        return exact
    found = sorted(ROOT.glob("bin/*/*/tests/bench_perf_sweep"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def run_geist(binary: Path, gguf: Path) -> list[dict]:
    env = dict(os.environ, OMP_WAIT_POLICY=os.environ.get("OMP_WAIT_POLICY", "active"))
    cmd = [str(binary), "--gguf", str(gguf),
           "--seq-lens", PROTOCOL["seq_lens"],
           "--decode-n", str(PROTOCOL["decode_n"]),
           "--warmup", str(PROTOCOL["warmup"]),
           "--repeats", str(PROTOCOL["repeats"]),
           "--emit-jsonl"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if p.returncode != 0:
        tail = (p.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"bench_perf_sweep failed: {tail}")
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    if not rows:
        raise RuntimeError("bench_perf_sweep produced no JSON rows")
    return rows


def cool_down() -> bool:
    """Wait for the board to return to the temperature geist started from.

    Without this the baseline runs on hardware the first engine just heated,
    and the ratio measures the running order as much as the engines. Returns
    True if we know the board was cool when the baseline started; False when
    temperature is not readable here, which the report then says out loud.
    """
    if temp_c() < 0:
        return False
    deadline = time.monotonic() + COOL_TIMEOUT_S
    while temp_c() > COOL_C and time.monotonic() < deadline:
        print(f"  cooling to {COOL_C:.0f} C before the baseline "
              f"(now {temp_c():.1f} C)", file=sys.stderr)
        time.sleep(20)
    return temp_c() <= COOL_C


def find_baseline() -> tuple[str, Path] | None:
    """A baseline engine's llama-bench, if one is reachable.

    Explicit env wins; otherwise the conventional build paths. Absent is the
    normal case for someone who just cloned this, and is not an error.
    """
    for var, name in (("BITNET_BENCH", "bitnet.cpp"), ("LLAMA_BENCH", "llama.cpp")):
        p = os.environ.get(var)
        if p and Path(p).exists():
            return name, Path(p)
    home = Path.home()
    for cand, name in ((home / "BitNet/build/bin/llama-bench", "bitnet.cpp"),
                       (home / "llama.cpp/build/bin/llama-bench", "llama.cpp")):
        if cand.exists():
            return name, cand
    return None


def run_baseline(binary: Path, gguf: Path) -> dict | None:
    """llama-bench at the same points, parsed from its own table."""
    seq = PROTOCOL["seq_lens"].split(",")[-1]
    cmd = [str(binary), "-m", str(gguf), "-p", seq, "-n", str(PROTOCOL["decode_n"]),
           "-r", str(PROTOCOL["repeats"])]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    out = {}
    for line in p.stdout.splitlines():
        m = re.search(r"\b(pp|tg)(\d+)\b.*?([\d.]+)\s*±", line)
        if m:
            out["prefill_tps" if m.group(1) == "pp" else "decode_tps"] = float(m.group(3))
    return out or None


# --- Derived figures --------------------------------------------------------


def spread_pct(row: dict) -> float:
    best, worst, mean = row.get("decode_ms_best"), row.get("decode_ms_worst"), row.get("decode_tps")
    if not best or not worst or not mean:
        return 0.0
    n = row["decode_n"]
    return (n * 1000 / best - n * 1000 / worst) / mean * 100 / 2


def weight_gbps(row: dict, model_name: str) -> float | None:
    """Weight bytes streamed per second — derived, not measured."""
    per_token = PER_TOKEN_BYTES.get(model_name)
    return per_token * row["decode_tps"] / 1e9 if per_token else None


# --- Report -----------------------------------------------------------------


def render(run: dict, refs: list[dict]) -> str:
    sysinfo, rows = run["system"], run["rows"]
    worst_spread = max((r["spread_pct"] for r in rows), default=0.0)
    quiet = worst_spread <= CLEAN_SPREAD_PCT
    L = [
        "## geist benchmark",
        "",
        f"| | |",
        f"| :-- | :-- |",
        f"| Model | `{run['model']}` |",
        f"| CPU | {sysinfo['cpu']} ({sysinfo['cores']} cores, {sysinfo['ram_gb']} GB) |",
        f"| OS | {sysinfo['os']} |",
        f"| geist | `{run['commit']}` |",
        f"| Protocol | seq {PROTOCOL['seq_lens']}, decode {PROTOCOL['decode_n']}, "
        f"{PROTOCOL['repeats']}× mean |",
        "",
        "| prompt tokens | decode t/s | prefill t/s | spread | weight GB/s |",
        "| --: | --: | --: | --: | --: |",
    ]
    for r in rows:
        gbps = r.get("weight_gbps")
        L.append(f"| {r['seq_len']} | **{r['decode_tps']:.2f}** | {r['prefill_tps']:.2f} "
                 f"| ±{r['spread_pct']:.1f} % | {f'{gbps:.1f}' if gbps else '—'} |")
    L += ["", f"Load {run['load_before']:.2f} → {run['load_after']:.2f}"
              + (f", {run['temp_before']:.1f} → {run['temp_after']:.1f} °C"
                 if run["temp_before"] > 0 else "")]

    # Two independent signals that must not be conflated. Load says whether
    # anything else was competing; spread says whether the run varied. Wide
    # spread on an idle box is usually the board heating up, not contention —
    # reporting that as "something else was using the machine" is a false alarm
    # that discredits a perfectly good measurement.
    per_core = run["load_before"] / max(sysinfo["cores"], 1)
    busy = per_core > BUSY_LOAD_PER_CORE
    warmed = run["temp_after"] - run["temp_before"] if run["temp_before"] > 0 else 0.0
    if busy:
        verdict = (f"**Contended.** Load was {run['load_before']:.1f} across "
                   f"{sysinfo['cores']} cores ({per_core:.1f} per core) — something "
                   "else was competing for the machine. Re-run on an idle box; "
                   "these numbers say more about the other workload than about "
                   "this engine.")
    elif quiet:
        verdict = (f"**Clean.** Load {run['load_before']:.2f} on {sysinfo['cores']} "
                   f"cores, spread at or under {CLEAN_SPREAD_PCT:.0f} % throughout.")
    else:
        cause = (f"the board warmed {warmed:.0f} °C during the run"
                 if warmed >= 15 else "the cause is not visible from load alone")
        verdict = (f"**Idle but variable.** Load was only {run['load_before']:.2f} on "
                   f"{sysinfo['cores']} cores, so nothing was competing, yet the widest "
                   f"spread reached ±{worst_spread:.1f} % against the "
                   f"{CLEAN_SPREAD_PCT:.0f} % a settled box holds — {cause}. The "
                   "shorter-context rows are the more trustworthy ones.")
    L += ["", "**Machine state:** " + verdict]

    if run.get("energy"):
        e = run["energy"]
        # Deliberately NOT divided into a per-token figure. The sampling window
        # covers prefill and decode together, and at a 512-token prompt prefill
        # dominates it -- dividing by decoded tokens alone charges prefill's
        # energy to them and inflates the number ~4.5x. A trustworthy J/token
        # needs the sampler phase-separated against the sweep's own timings,
        # which this does not do yet.
        L += ["", f"**Energy:** {e['mean_w']:.2f} W mean board power over "
                  f"{e['seconds']:.0f} s ({e['samples']} PMIC samples), "
                  f"{e['joules']:.0f} J for the whole sweep. Not divided per token: "
                  "the window covers prefill and decode together and they cost "
                  "very differently, so any single figure would be a fiction."]
    else:
        L += ["", "**Energy:** not readable without privileges on this platform "
                  "(Raspberry Pi reports it unprivileged; x86 RAPL and macOS "
                  "`powermetrics` need root, and this benchmark does not ask for it)."]

    if run.get("baseline"):
        b = run["baseline"]
        # llama-bench runs pp and tg as SEPARATE benchmarks: tg generates from
        # an empty context, it does not decode after the pp prompt. So its tg
        # number belongs beside our SHORTEST-context row, not our longest.
        # Comparing our seq-512 decode (full KV) against its empty-context tg
        # pits our hardest case against its easiest and understates us by ~20 %
        # — measured here at 1.67x where the like-for-like figure is 2.00x.
        # pp512 does line up with our 512-token prefill.
        short, long_ = rows[0], rows[-1]
        L += ["", f"### vs {b['engine']}", "",
              "| | geist | " + b["engine"] + " | ratio |", "| :-- | --: | --: | --: |"]
        if b.get("decode_tps"):
            L.append(f"| decode, {short['seq_len']}-token context "
                     f"| {short['decode_tps']:.2f} | {b['decode_tps']:.2f} "
                     f"| **{short['decode_tps'] / b['decode_tps']:.2f}×** |")
        if b.get("prefill_tps"):
            L.append(f"| prefill, {long_['seq_len']} tokens "
                     f"| {long_['prefill_tps']:.2f} | {b['prefill_tps']:.2f} "
                     f"| **{long_['prefill_tps'] / b['prefill_tps']:.2f}×** |")
        L += ["", ("Both engines started from the same thermal baseline "
                   f"(≤{COOL_C:.0f} °C); the board was allowed to cool between them."
                   if run.get("baseline_cooled") else
                   "**Caveat:** this platform does not expose a temperature this "
                   "tool can read, so the baseline may have run on hardware the "
                   "first engine had already warmed. On a passively cooled board "
                   "that favours whichever engine ran first — here, geist."),
              "", "Same box, same GGUF, same run — this is the falsifiable claim. "
                  "The rows are matched by *what was measured*, not by position: "
                  f"{b['engine']}'s generation benchmark starts from an empty "
                  "context, so it belongs against our shortest-context decode, "
                  "while its prompt benchmark lines up with our longest prefill. "
                  f"Our decode at {long_['seq_len']} tokens of context "
                  f"({long_['decode_tps']:.2f} t/s) has no counterpart here — "
                  "that engine was not measured under load."]
    else:
        L += ["", "**No baseline engine found**, so the numbers above stand alone: "
                  "throughput depends on your hardware and cannot be compared "
                  "against a result measured elsewhere. Point `LLAMA_BENCH` or "
                  "`BITNET_BENCH` at a built `llama-bench` and re-run to get a "
                  "ratio on *your* machine, which is the number that means "
                  "something."]

    if refs:
        L += ["", "### Reference runs (same command, same protocol, other machines)", "",
              "| system | model | decode t/s @512 | vs baseline | date |",
              "| :-- | :-- | --: | :-- | :-- |"]
        for r in refs:
            ratio = f"{r['ratio']:.2f}× {r['baseline_engine']}" if r.get("ratio") else "—"
            L.append(f"| {r['system']} | {r['model']} | {r['decode_tps_512']:.2f} "
                     f"| {ratio} | {r['date']} |")
        L += ["", "Different hardware, so the throughput columns are context rather "
                  "than a comparison — the ratio column is the part that carries "
                  "across machines."]

    L += ["", "---", "",
          "Reproduce: `make bench` (protocol frozen; see "
          "`benchmark/METHODOLOGY.md`). Weight GB/s is derived from the "
          "per-token byte budget in `benchmark/results/TERNARY.md`, not measured."]
    return "\n".join(L)


# --- Main -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--target", default="")
    ap.add_argument("--mode", default="release")
    ap.add_argument("--out-dir", default=str(Path.home() / "bench-geistlib"))
    ap.add_argument("--dataset", default=str(ROOT / "benchmark" / "reference_runs.json"))
    ap.add_argument("--record", action="store_true",
                    help="append this run to the dataset (maintainer use; off by "
                         "default so a reproducer never dirties its checkout)")
    args = ap.parse_args()

    gguf = Path(args.gguf)
    if not gguf.is_file():
        print(f"SKIP: model not found at {gguf} — run `make fetch-bench-model`")
        return SKIP
    binary = find_geist_bench(args.target, args.mode)
    if binary is None:
        print("SKIP: bench_perf_sweep not built — run `make bin` first")
        return SKIP

    run = {"model": gguf.name, "system": system_info(),
           "commit": sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]) or "unknown",
           "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
           "load_before": load_avg(), "temp_before": temp_c()}

    print(f"geist benchmark — {run['model']} on {run['system']['cpu']}", file=sys.stderr)
    print(f"  protocol: seq {PROTOCOL['seq_lens']}, {PROTOCOL['repeats']} repeats "
          f"(a few minutes)", file=sys.stderr)

    try:
        with EnergySampler() as energy:
            rows = run_geist(binary, gguf)
        run["energy"] = energy.summary()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return ERROR

    for r in rows:
        r["spread_pct"] = spread_pct(r)
        r["weight_gbps"] = weight_gbps(r, run["model"])
    run["rows"] = rows
    run["load_after"], run["temp_after"] = load_avg(), temp_c()

    baseline = find_baseline()
    if baseline:
        name, path = baseline
        print(f"  baseline found: {name} ({path})", file=sys.stderr)
        run["baseline_cooled"] = cool_down()
        res = run_baseline(path, gguf)
        if res:
            run["baseline"] = {"engine": name, "path": str(path), **res}

    refs = []
    ds = Path(args.dataset)
    if ds.is_file():
        try:
            refs = json.loads(ds.read_text()).get("runs", [])
        except (OSError, json.JSONDecodeError):
            refs = []

    report = render(run, refs)
    print(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run['date']}_{run['system']['cpu'].split()[0].lower()}_{gguf.stem}"
    (out_dir / f"{stem}_report.md").write_text(report + "\n")
    (out_dir / f"{stem}_raw.json").write_text(json.dumps(run, indent=1) + "\n")
    print(f"\nreport: {out_dir / (stem + '_report.md')}", file=sys.stderr)

    if args.record:
        entry = {
            "date": run["date"], "system": run["system"]["cpu"],
            "cores": run["system"]["cores"], "ram_gb": run["system"]["ram_gb"],
            "os": run["system"]["os"], "model": run["model"], "commit": run["commit"],
            "protocol": PROTOCOL,
            "decode_tps_512": rows[-1]["decode_tps"],
            "prefill_tps_512": rows[-1]["prefill_tps"],
            "spread_pct": rows[-1]["spread_pct"],
        }
        if run.get("baseline") and run["baseline"].get("decode_tps"):
            # Matched against the SHORT-context row: llama-bench's tg starts
            # from an empty context. See the note in render().
            entry["baseline_engine"] = run["baseline"]["engine"]
            entry["baseline_decode_tps"] = run["baseline"]["decode_tps"]
            entry["decode_tps_short"] = rows[0]["decode_tps"]
            entry["decode_ratio_short_ctx"] = rows[0]["decode_tps"] / run["baseline"]["decode_tps"]
        if run.get("baseline") and run["baseline"].get("prefill_tps"):
            entry["prefill_ratio_512"] = rows[-1]["prefill_tps"] / run["baseline"]["prefill_tps"]
        data = json.loads(ds.read_text()) if ds.is_file() else {"runs": []}
        data["runs"].append(entry)
        ds.write_text(json.dumps(data, indent=1) + "\n")
        print(f"recorded to {ds}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
