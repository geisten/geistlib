#!/usr/bin/env python3
"""Shape-matched per-op comparison of geist (GEIST_VK_PROFILE_SHAPES) vs
llama.cpp (GGML_VK_PERF_LOGGER), pp512 on the 2080 Ti. Markdown on stdout."""
import collections, json, os, re, sys

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prof")
MODELS = ["gemma4-e2b", "qwen35-4b", "qwen3-0.6b", "bonsai2-27b"]
WARM, TOK = 64, 512


def geist(model, m):
    err = open(f"{P}/geist_{model}_m{m}.err").read().splitlines()
    names = {int(l.split()[1]): l.split()[2] for l in err if l.startswith("PIPEIDX")}
    shapes = [tuple(map(float, l.split()[1:])) for l in err if l.startswith("SHAPE")]
    gemm = collections.defaultdict(list)  # (n_out, n_in) -> [(rows, ns, pipe)]
    other = collections.Counter()
    for pipe, n_in, n_out, rows, ns in shapes:
        name = names.get(int(pipe), f"pipe{int(pipe)}")
        if (name.startswith("matmul_") or name.startswith("mm_")) and rows > 1:
            gemm[(int(n_out), int(n_in))].append((rows, ns, name))
        elif not name.startswith("matvec") and name != "argmax":
            other[name] += ns * TOK / (TOK + WARM)  # ponytail: warmup share removed pro rata
    out = {}
    for key, ds in gemm.items():
        total = sum(r for r, _, _ in ds)
        skip, acc, keep = total * WARM / (TOK + WARM), 0.0, []
        for r, ns, name in ds:  # drop the leading warmup chunk(s) exactly
            if acc + r <= skip + 0.5:
                acc += r
                continue
            keep.append((ns, name, r))
        out[key] = (sum(n for n, _, _ in keep) / 1e3,
                    "/".join(sorted({n for _, n, _ in keep})), sum(r for _, _, r in keep))
    return out, {k: v / 1e3 for k, v in other.items()}


def llama(model, ub):
    blocks = open(f"{P}/llama_{model}_ub{ub}.err").read().split("Vulkan Timings:")[1:]
    gemm, other = collections.Counter(), collections.Counter()
    dtype = {}
    for b in blocks[-(TOK // ub):]:
        for line in b.splitlines():
            m = re.match(r"MUL_MAT (\S+) m=(\d+) n=(\d+) k=(\d+): .*= ([\d.]+) us", line)
            if m and int(m.group(3)) > 1:
                key = (int(m.group(2)), int(m.group(4)))
                gemm[key] += float(m.group(5)); dtype.setdefault(key, set()).add(m.group(1))
                continue
            m = re.match(r"(\S+).*= ([\d.]+) us", line)
            if m and not line.startswith("MUL_MAT_VEC") and not line.startswith("Total"):
                other[m.group(1)] += float(m.group(2))
    return gemm, other, dtype


def fmt(x):
    return f"{x/1e3:,.1f}"


for model in MODELS:
    g64, go64 = geist(model, 64)
    g128, _ = geist(model, 128)
    l64, lo64, dt = llama(model, 64)
    l512, lo512, _ = llama(model, 512)
    gt = sum(v[0] for v in g64.values()) + sum(go64.values())
    print(f"\n### {model} — pp512, ms GPU-Zeit (geist M=64 Summe {fmt(gt)} ms; llama ub64 {fmt(sum(l64.values())+sum(lo64.values()))} ms, ub512 {fmt(sum(l512.values())+sum(lo512.values()))} ms)\n")
    print("| N×K | Typ | geist-Kernel | geist M64 | geist M128 | llama ub64 | llama ub512 | Faktor @64 | TFLOPS geist / llama @64 | Anteil geist |")
    print("| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |")
    for key in sorted(g64, key=lambda k: -g64[k][0]):
        n, k = key
        a, pipe, rows = g64[key]
        b = g128.get(key, (None,))[0]
        c, d = l64.get(key), l512.get(key)
        fl = 2 * n * k * rows
        tf = lambda us: f"{fl/us/1e6:.1f}" if us else "—"
        print(f"| {n}×{k} | {'/'.join(sorted(dt.get(key, {'?'})))} | {pipe} | {fmt(a)} | {fmt(b) if b else '—'} | {fmt(c) if c else '—'} | {fmt(d) if d else '—'} | "
              f"{a/c:.2f}× | {tf(a)} / {tf(c)} | {a/gt*100:.0f} % |" if c else
              f"| {n}×{k} | ? | {pipe} | {fmt(a)} | {fmt(b) if b else '—'} | — | — | — | — | {a/gt*100:.0f} % |")
    unm = [k for k in l64 if k not in g64]
    if unm:
        print(f"\nllama-GEMMs ohne geist-Gegenstück: " + ", ".join(f"{'/'.join(sorted(dt[k]))} {k[0]}×{k[1]} ({fmt(l64[k])} ms)" for k in unm))
    print("\nNicht-GEMM (geist M64 → llama ub64): " + "; ".join(f"{k} {fmt(v)}" for k, v in sorted(go64.items(), key=lambda x: -x[1]) if v > 500)
          + "  ‖  " + "; ".join(f"{k} {fmt(v)}" for k, v in sorted(lo64.items(), key=lambda x: -x[1]) if v > 500))
