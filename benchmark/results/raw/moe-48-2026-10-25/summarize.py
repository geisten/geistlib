#!/usr/bin/env python3
"""Tables for the #48 research note from the *_moe_bench.sh / desktop JSONL.

Throughput: per (model, cfg) the median of the per-cycle means and the
range. Energy: per (model, cfg) the decode energy per token from the two
runs (short and long decode): (E_long - E_short) / (n_long - n_short),
E = mean power x wall time of the run (macmon / PMIC / nvidia-smi samples
give the mean power; their count x period can drift from the wall time).
"""
import json
import re
import statistics
import sys
from collections import defaultdict


def mean_power(e: dict) -> dict:
    if "e_board_j" in e:  # Pi: PMIC, trapezoid over the sampled span
        return {"all": e["mean_w"]}
    if "e_gpu_j" in e and "e_all_j" not in e:  # desktop: nvidia-smi, 100 ms
        return {"gpu": e["mean_w"]}
    n = max(e["samples"], 1) * 0.1  # Mac: macmon, 100 ms
    return {"all": e["e_all_j"] / n, "cpu": e["e_cpu_j"] / n, "gpu": e["e_gpu_j"] / n}


def main(paths: list[str]) -> None:
    thr = defaultdict(list)
    en = defaultdict(dict)
    for p in paths:
        for line in open(p):
            if not line.strip().startswith("{"):
                continue
            # the scripts' bc printed wall times below 1 s as ".87"
            # the Pi has no bc: an empty wall time
            line = line.replace('"wall_s": ,', '"wall_s": null,')
            try:
                d = json.loads(re.sub(r'": (-?)\.(\d)', r'": \g<1>0.\2', line))
            except json.JSONDecodeError:  # a run that failed: "result": }
                print(f"failed run: {line[:110]}...", file=sys.stderr)
                continue
            r = d["result"]
            if d["kind"] == "throughput" and r and "decode_tps" in r:
                thr[(d["model"], d["cfg"])].append((r["prefill_tps"], r["decode_tps"],
                                                    r.get("maxrss_mb"), r.get("major_faults")))
            elif d["kind"] == "energy" and d.get("energy"):
                en[(d["model"], d["cfg"])][d["decode_n"]] = (d, r)
    print("| model | engine | prefill 512 tok/s (median, range) | decode tok/s (median, range) | cycles |")
    print("| :-- | :-- | --: | --: | --: |")
    for (m, c), v in thr.items():
        pf = [x[0] for x in v]
        dc = [x[1] for x in v]
        extra = ""
        if v[0][2] is not None:
            extra = f" (max RSS {max(x[2] for x in v):.0f} MB, major faults {max(x[3] for x in v)})"
        print(f"| {m} | {c}{extra} | {statistics.median(pf):.1f} ({min(pf):.1f}-{max(pf):.1f}) "
              f"| {statistics.median(dc):.2f} ({min(dc):.2f}-{max(dc):.2f}) | {len(v)} |")
    print()
    print("| model | engine | decode tok/s | J per decoded token | tokens per J | split |")
    print("| :-- | :-- | --: | --: | --: | :-- |")
    for (m, c), runs in en.items():
        if len(runs) < 2:
            continue
        (ns, (ds, rs)), (nl, (dl, rl)) = sorted(runs.items())
        ps, pl = mean_power(ds["energy"]), mean_power(dl["energy"])
        parts = {}
        if "e_board_j" in dl["energy"]:  # Pi: the PMIC integral over the run itself
            parts["all"] = (dl["energy"]["e_board_j"] - ds["energy"]["e_board_j"]) / (nl - ns)
        for k in pl if not parts else []:
            parts[k] = (pl[k] * dl["wall_s"] - ps[k] * ds["wall_s"]) / (nl - ns)
        tot = parts.get("all", parts.get("gpu"))
        split = ", ".join(f"{k} {v:.3f}" for k, v in parts.items() if k != "all")
        print(f"| {m} | {c} | {rl['decode_tps']:.2f} | {tot:.3f} | {1 / tot:.2f} | {split} |")


if __name__ == "__main__":
    main(sys.argv[1:])
