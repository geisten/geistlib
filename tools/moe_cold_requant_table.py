#!/usr/bin/env python3
"""Quality table from the llama-perplexity logs of a #247 campaign.

    tools/moe_cold_requant_table.py <log dir>

Each <variant>.log holds the rewrite's summary line (`cold: ...`) and, per
held-out domain (`=== en|de|c|py`), llama-perplexity's KL-divergence block
against the `base` variant. Prints one Markdown row per variant: the
held-out selection share of its cold experts, Delta PPL (chunk-weighted
over the domains), mean KLD, same top-1 and Delta PPL per domain.

Delta PPL is the variant's PPL(Q) against base.log's own `Final estimate`,
not against the KL block's PPL(base): that one is computed from the stored,
compressed base logits and reads 0.04-1.9 % low, the same for every variant.
The +- is the KL block's paired standard error. Standard library only.
"""
import math
import re
import sys
from pathlib import Path

ORDER = ["orig", "q3k-0.10", "q3k-0.25", "q3k-0.50", "rnd-q3k-0.50", "q2k-0.10", "q2k-0.25",
         "q2k-0.50", "rnd-q2k-0.25", "iq2s-0.25"]


def blocks(text):
    return [(b.split()[0], b) for b in re.split(r"^=== ", text, flags=re.M)[1:]]


def parse(path):
    text = path.read_text()
    cold = re.search(r"held-out ([\d.]+) %", text)
    doms = {}
    for d, block in blocks(text):
        n = re.search(r"over (\d+) chunks", block)
        se = re.search(r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*[-\d.]+ ±\s*([\d.]+)", block)
        ppl = re.search(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", block)
        kld = re.search(r"Mean\s+KLD:\s*([-\d.]+)", block)
        top = re.search(r"Same top p:\s*([\d.]+)", block)
        if n and se and kld and top and ppl:
            doms[d] = {"n": int(n[1]), "ppl": float(ppl[1]), "se": float(se[1]),
                       "kld": float(kld[1]), "top": float(top[1])}
    return (float(cold[1]) if cold else None), doms


def main(d):
    base = {}
    for k, block in blocks((Path(d) / "base.log").read_text()):
        m = re.search(r"Final estimate: PPL = ([\d.]+)", block)
        if m:
            base[k] = float(m[1])
    print("| variant | cold selections (held-out) | ΔPPL all | KLD | same top-1 "
          "| ΔPPL en | de | c | py |")
    print("| :-- | --: | --: | --: | --: | --: | --: | --: | --: |")
    logs = {p.stem: p for p in Path(d).glob("*.log") if p.stem != "base"}
    for name in [v for v in ORDER if v in logs] + sorted(set(logs) - set(ORDER)):
        cold, doms = parse(logs[name])
        if not doms:
            continue
        for k, x in doms.items():
            x["ln"] = math.log(x["ppl"] / base[k])
        n = sum(x["n"] for x in doms.values())
        ln = sum(x["n"] * x["ln"] for x in doms.values()) / n
        se = math.sqrt(sum((x["n"] * x["se"]) ** 2 for x in doms.values())) / n
        kld = sum(x["n"] * x["kld"] for x in doms.values()) / n
        top = sum(x["n"] * x["top"] for x in doms.values()) / n
        per = " | ".join(f"{100 * math.expm1(doms[k]['ln']):+.1f} %" if k in doms else "–"
                         for k in ("en", "de", "c", "py"))
        c = f"{cold:.1f} %" if cold is not None and name != "orig" else "–"
        print(f"| {name} | {c} | {100 * math.expm1(ln):+.2f} % ± {100 * se:.2f} | {kld:.4f} | "
              f"{top:.1f} % | {per} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
