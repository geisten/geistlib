"""#48 phase 0: how skewed is expert usage? Reads "USAGE <model> <set> <file> <tokens>
<layer> <counts>" lines (benchmark/results/raw/2026-10-15_moe_expert_usage.txt). Per layer the hot set is the top K % of experts by
profile-corpus selections; reported is the share of selections it serves on the
profile set (in-sample) and on the held-out set (the number that matters), per
domain. A uniform router would give exactly K %."""
import json, math, re, sys
from collections import defaultdict

runs = {}
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) == 7 and p[0] == "USAGE":
        _, model, sset, fname, toks, layer, counts = p
        files = runs.setdefault((model, sset), {"files": {}})["files"]
        f = files.setdefault(fname, {"file": fname, "tokens": int(toks), "counts": {}})
        f["counts"][layer] = [int(x) for x in counts.split(",")]
for r in runs.values():
    r["files"] = list(r["files"].values())
# moe_usage stops a layer's list at the highest expert it saw; pad to the model's count.
for model in {k[0] for k in runs}:
    n = max(len(c) for k, r in runs.items() if k[0] == model for f in r["files"] for c in f["counts"].values())
    for k, r in runs.items():
        if k[0] == model:
            for f in r["files"]:
                for l, c in f["counts"].items():
                    f["counts"][l] = c + [0] * (n - len(c))

def totals(run, domain=None):
    t = defaultdict(lambda: None)
    for f in run["files"]:
        d = re.sub(r"\d+\.txt$", "", f["file"].rsplit("/", 1)[-1])
        if domain and d != domain:
            continue
        for l, c in f["counts"].items():
            t[int(l)] = c if t[int(l)] is None else [a + b for a, b in zip(t[int(l)], c)]
    return dict(t)

for model in sorted({k[0] for k in runs}):
    prof, held = totals(runs[(model, "profile")]), totals(runs[(model, "heldout")])
    n_exp = len(next(iter(prof.values())))
    toks = sum(f["tokens"] for f in runs[(model, "heldout")]["files"])
    print(f"\n## {model}: {len(prof)} layers x {n_exp} experts, held-out {toks} tokens")
    ent = [-sum(c / sum(cs) * math.log2(c / sum(cs)) for c in cs if c) / math.log2(n_exp) for cs in held.values()]
    print(f"normalised entropy of held-out usage per layer: min {min(ent):.3f} mean {sum(ent)/len(ent):.3f} (1.0 = uniform)")
    per_tok = {sum(c) / f["tokens"] for f in runs[(model, "heldout")]["files"] for c in f["counts"].values()}
    print(f"selections per token and layer: {sorted(per_tok)} (must be one value, the model's top-k)")
    never = sum(1 for l in prof for e in range(n_exp) if prof[l][e] == 0)
    print(f"experts never selected on the profile corpus: {never} of {len(prof)*n_exp}")
    print("| hot K % | profile (in-sample) | held-out all | " + " | ".join(f"held-out {d}" for d in ("en", "de", "c", "py")) + " | worst layer, held-out |")
    print("|" + " --: |" * 8)
    # 18.3 %: Strata's static profile cache, 4,500 of 24,576 experts, serves 50 % of the
    # routed experts on held-out prompts (Strata paper, section 3.4 and finding 4).
    for K in (10, 18.3, 25, 50, 75):
        k = max(1, round(n_exp * K / 100))
        hot = {l: set(sorted(range(n_exp), key=lambda e: -prof[l][e])[:k]) for l in prof}
        def cover(t):
            per = [sum(t[l][e] for e in hot[l]) / sum(t[l]) for l in t]
            return sum(sum(t[l][e] for e in hot[l]) for l in t) / sum(sum(t[l]) for l in t), min(per)
        row = [cover(prof)[0], cover(held)[0]] + [cover(totals(runs[(model, "heldout")], d))[0] for d in ("en", "de", "c", "py")]
        print(f"| {K} % ({k}) | " + " | ".join(f"{v*100:.1f} %" for v in row) + f" | {cover(held)[1]*100:.1f} % |")
