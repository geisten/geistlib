#!/usr/bin/env python3
"""metal_trace_report.py — tables from traces of tools/probes/metal_trace.m (#590).

    metal_trace_report.py LABEL=trace.jsonl [LABEL=trace.jsonl ...] [--kernels N]

Per trace and phase (prefill, decode; the mean over the measured repeats, the
warmup dropped): wall time, command buffers, dispatches, GPU time (the sum of
the buffers' GPUEndTime - GPUStartTime, and the union of those intervals),
the wall time with no buffer on the GPU, and the GPU time per kernel class.

Phases come from the trace's marks (`rN.start`, `rN.first`, `rN.end`), which
metal_profile_mlx.py and bench_perf_sweep write: prefill is start to first
(geist: the prefill call; mlx-lm: up to the first yielded token), decode is
first to end. With more than one sequence length in a bench_perf_sweep run
the marks repeat, so trace one length per run.

GPU time per kernel is read from a trace with one op per command buffer
(GEIST_METAL_PIPELINE=1 GEIST_METAL_PIPELINE_PREFILL=1: one geist op, one to
seven dispatches; MLX_MAX_OPS_PER_BUFFER=1: two dispatches). One-op buffers
run slower than the default ones, so the shares are what it gives. In other
traces a buffer's time is split over its kernels by dispatch count, which
says nothing about a kernel's time; the table header says which it is.
"""
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict

# First match wins. geist names: src/backends/metal/metal_shaders.h; mlx names:
# the kernel library of mlx 0.32.4 (the long generated names are compiled
# elementwise graphs).
CLASSES = [
    ("lm head / sampling", r"argmax|logsumexp|vsn_Subtract"),
    ("embedding", r"embed|gather_front|affine_dequantize"),
    ("quantized matmul",
     r"^(matvec_|matmul_|linear_|gate_up|mv\d_|mm_q|mm_tc|affine_qm|steel_gemm_splitk)"),
    ("attention", r"attention|sdpa|steel_gemm_fused|softmax|rope|kv_append|g\d_Select|GreaterEqual|arange"),
    ("norm", r"rms"),
    ("DeltaNet mixer", r"^(dn_|dnf_|deltanet)|gated_delta|depthwise_conv|LogAddExp"),
    ("elementwise / copies", r"."),
]


def classify(name):
    for label, pattern in CLASSES:
        if re.search(pattern, name):
            return label
    return "?"


def load(path):
    kernels, marks, records = {}, [], []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if "kernel" in row:
                kernels[row["kernel"]] = row["name"]
            elif "mark" in row:
                marks.append(row)
            else:
                records.append(row)
    records = [r for r in records if r["ge"] > 0 and r["c"] > 0]
    records.sort(key=lambda r: r["gs"])
    return kernels, marks, records


def phases_from_marks(marks):
    stamp = {m["mark"]: m for m in marks}
    out = []
    for name in sorted({m["mark"].split(".")[0] for m in marks if m["mark"].startswith("r")}):
        a, b, c = (stamp[f"{name}.{part}"] for part in ("start", "first", "end"))
        out.append(("prefill", a["t"], b["t"], b["cpu"] - a["cpu"]))
        out.append(("decode", b["t"], c["t"], c["cpu"] - b["cpu"]))
    return out


def union(intervals):
    total, last = 0.0, 0.0
    for a, b in sorted(intervals):
        a = max(a, last)
        if b > a:
            total += b - a
            last = b
    return total


def summarize(kernels, records, phases):
    out = {}
    for kind in ("prefill", "decode"):
        rows = []
        for name, t0, t1, cpu in phases:
            if name != kind:
                continue
            inside = [r for r in records if t0 <= r["gs"] < t1]
            work = [r for r in inside if r["n"]]
            per_kernel = defaultdict(lambda: [0, 0.0])
            for r in work:
                for k, n in r["k"]:
                    per_kernel[kernels[k]][0] += n
                    per_kernel[kernels[k]][1] += (r["ge"] - r["gs"]) * n / r["n"]
            # The widest quantized matmul of the phase: in decode the lm head
            # (one per token), which no kernel name tells from the others.
            wide = [r for r in work
                    if any(classify(kernels[k]) == "quantized matmul" for k, _ in r["k"])]
            top = max((r["tg"] for r in wide), default=0)
            head = [r for r in wide if r["tg"] >= 0.9 * top]
            waits = [r for r in inside if r["w1"] > 0]
            rows.append({
                "wall": t1 - t0,
                "cpu": cpu,
                "buffers": len(work),
                "empty": len(inside) - len(work),
                "dispatches": sum(r["n"] for r in work),
                "gpu_sum": sum(r["ge"] - r["gs"] for r in inside),
                "gpu_union": union([(r["gs"], min(r["ge"], t1)) for r in inside]),
                "sched": statistics.median([r["gs"] - r["c"] for r in work]) if work else 0.0,
                "head_n": len(head),
                "head_s": sum(r["ge"] - r["gs"] for r in head),
                "barriers": sum(r.get("mb", 0) for r in work),
                "concurrent": sum(r.get("cc", 0) for r in work) / max(1, sum(r["e"] for r in work)),
                "tokens": token_gaps(work),
                "waits": len(waits),
                "wait_s": sum(r["w1"] - r["w0"] for r in waits),
                "kernels": per_kernel,
            })
        if rows:
            out[kind] = rows
    return out


def token_gaps(work):
    """Decode of an engine that waits once per token (geist): per token, the
    GPU time and where the rest of the period goes. None when the phase does
    not have that shape."""
    work = sorted(work, key=lambda r: r["c"])
    tokens, current = [], []
    for record in work:
        current.append(record)
        if record["w1"] > 0:
            tokens.append(current)
            current = []
    if len(tokens) < 100 or statistics.median(sum(r["n"] for r in t) for t in tokens) < 50:
        return None
    out = defaultdict(list)
    for before, token in zip(tokens, tokens[1:]):
        last = before[-1]
        out["period"].append(token[-1]["w1"] - last["w1"])
        out["gpu"].append(sum(r["ge"] - r["gs"] for r in token))
        out["GPU end to wait returns"].append(last["w1"] - last["ge"])
        out["wait returns to first commit"].append(token[0]["c"] - last["w1"])
        out["first commit to GPU start"].append(token[0]["gs"] - token[0]["c"])
        out["GPU idle between the token's buffers"].append(
            sum(max(0.0, b["gs"] - a["ge"]) for a, b in zip(token, token[1:])))
    return {key: statistics.median(values) for key, values in out.items()}


def mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return statistics.fmean(values) if values else float("nan")


def main():
    args = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    top = int(sys.argv[sys.argv.index("--kernels") + 1]) if "--kernels" in sys.argv else 12
    data = {}
    for arg in args:
        label, path = arg.split("=", 1)
        kernels, marks, records = load(path)
        phases = phases_from_marks(marks)
        data[label] = summarize(kernels, records, phases)
    for kind in ("prefill", "decode"):
        print(f"\n### {kind}\n")
        print("| trace | repeats | wall ms | GPU busy ms (union) | GPU sum ms | no buffer on GPU ms | "
              "CPU ms | buffers | empty | dispatches | concurrent encoders | memory barriers | "
              "commit to GPU start, median us | waitUntilCompleted calls | ms in them |")
        print("| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | "
              "--: | --: |")
        for label, phases in data.items():
            rows = phases.get(kind)
            if not rows:
                continue
            wall, busy = mean(rows, "wall"), mean(rows, "gpu_union")
            print(f"| {label} | {len(rows)} | {wall * 1e3:.2f} | {busy * 1e3:.2f} | "
                  f"{mean(rows, 'gpu_sum') * 1e3:.2f} | {(wall - busy) * 1e3:.2f} | "
                  f"{mean(rows, 'cpu') * 1e3:.1f} | {mean(rows, 'buffers'):.0f} | "
                  f"{mean(rows, 'empty'):.0f} | {mean(rows, 'dispatches'):.0f} | "
                  f"{mean(rows, 'concurrent'):.0%} | {mean(rows, 'barriers'):.0f} | "
                  f"{mean(rows, 'sched') * 1e6:.0f} | {mean(rows, 'waits'):.0f} | "
                  f"{mean(rows, 'wait_s') * 1e3:.2f} |")
        for label, phases in data.items():
            gaps = [row["tokens"] for row in phases.get(kind, []) if row["tokens"]]
            if not gaps:
                continue
            print(f"\n{label}, {kind}, one token (median of the per-repeat medians, us): " +
                  ", ".join(f"{key} {statistics.median(g[key] for g in gaps) * 1e6:.0f}"
                            for key in gaps[0]))
        for label, phases in data.items():
            rows = phases.get(kind)
            if not rows:
                continue
            exact = all(row["dispatches"] < 2.5 * row["buffers"] for row in rows)
            total = mean(rows, "gpu_sum")
            per_kernel, per_class = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
            for row in rows:
                for name, (n, seconds) in row["kernels"].items():
                    for table, key in ((per_kernel, name), (per_class, classify(name))):
                        table[key][0] += n / len(rows)
                        table[key][1] += seconds / len(rows)
            how = ("one op per buffer; the time of a buffer with several kernels split by dispatch count"
                   if exact else "buffer time split by dispatch count: counts only")
            print(f"\n{label}, {kind}, per repeat ({how}):\n")
            print("| class | dispatches | GPU ms | share |")
            print("| :-- | --: | --: | --: |")
            for name, (n, seconds) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
                print(f"| {name} | {n:.0f} | {seconds * 1e3:.2f} | {100 * seconds / total:.1f} % |")
            if exact and kind == "decode":
                print(f"\nOf the quantized matmul, the buffers within 10 % of the most threadgroups "
                      f"(the lm head): {mean(rows, 'head_n'):.0f} per repeat, "
                      f"{mean(rows, 'head_s') * 1e3:.2f} ms, "
                      f"{100 * mean(rows, 'head_s') / total:.1f} %.")
            print("\n| kernel | dispatches | GPU ms | share | us per dispatch |")
            print("| :-- | --: | --: | --: | --: |")
            for name, (n, seconds) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:top]:
                print(f"| `{name[:70]}` | {n:.0f} | {seconds * 1e3:.2f} | "
                      f"{100 * seconds / total:.1f} % | {seconds / n * 1e6:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
