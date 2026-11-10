#!/usr/bin/env python3
"""Offload arithmetic for #247 (derived, not measured).

    tools/moe_cold_offload_model.py <Qwen3-30B-A3B-Q4_K_M.gguf> <libggml-base>

For a VRAM budget for experts, the share of the held-out router selections
the GPU-resident experts serve and the expert bytes per token the CPU branch
reads, with every expert in Q4_K_M or with cold experts in Q3_K/IQ2_S.
The resident set is the hottest experts of each layer on the profile set of
the #48 usage file (an equal share of the budget per layer); the held-out
set evaluates it. Standard library only.
"""
import os
import sys

from moe_cold_requant import EXPS, Ggml, load_usage, read_header

USAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark", "results",
                     "raw", "2026-10-15_moe_expert_usage_vulkan_run1225.txt")
N_LAYERS, N_EXPERT, TOP_K = 48, 128, 8


def per_layer_bytes(g, tensors, dtype=None):
    out = [0] * N_LAYERS
    for t in tensors:
        if t["name"].endswith(EXPS):
            ne0, ne1, _ = t["dims"]
            out[int(t["name"].split(".")[1])] += g.row_size(dtype or t["dtype"], ne0) * ne1
    return out


def main(gguf, lib):
    g = Ggml(lib)
    with open(gguf, "rb") as f:
        tensors = read_header(f)[3]
    s4 = per_layer_bytes(g, tensors)
    low = {"Q3_K": per_layer_bytes(g, tensors, 11), "IQ2_S": per_layer_bytes(g, tensors, 22)}
    u = load_usage(USAGE, "Qwen3-30B-A3B-Q4_K_M", N_LAYERS, N_EXPERT)
    held = u["heldout"]
    total = sum(map(sum, held))
    tokens = total / N_LAYERS / TOP_K
    rank = [sorted(range(N_EXPERT), key=lambda e: (-u["profile"][li][e], e))
            for li in range(N_LAYERS)]
    print(f"experts in Q4_K_M {sum(s4) * N_EXPERT / 1e9:.2f} GB; MB per expert: "
          f"Q4_K_M {sum(s4) / N_LAYERS / 1e6:.2f}, Q3_K {sum(low['Q3_K']) / N_LAYERS / 1e6:.2f}, "
          f"IQ2_S {sum(low['IQ2_S']) / N_LAYERS / 1e6:.2f}")
    print("| VRAM for experts | cold experts | resident | GPU share of selections "
          "| CPU expert MB / token |")
    print("| --: | :-- | --: | --: | --: |")
    schemes = (("none (all Q4_K_M)", None, None), ("non-resident in Q3_K", "Q3_K", None),
               ("non-resident in IQ2_S", "IQ2_S", None),
               ("least-used 50 % in Q3_K", "Q3_K", 0.5), ("least-used 50 % in IQ2_S", "IQ2_S", 0.5))
    for budget in (6e9, 8e9, 9.5e9):
        for label, cold_t, share in schemes:
            resident_n = hit = cpu = 0
            for li in range(N_LAYERS):
                size = [s4[li]] * N_EXPERT
                if share:
                    for e in rank[li][N_EXPERT - int(share * N_EXPERT):]:
                        size[e] = low[cold_t][li]
                used, resident = 0, set()
                for e in rank[li]:
                    if used + size[e] > budget / N_LAYERS:
                        break
                    used += size[e]
                    resident.add(e)
                for e in range(N_EXPERT):
                    if e in resident:
                        hit += held[li][e]
                    else:
                        miss = size[e] if share or not cold_t else low[cold_t][li]
                        cpu += held[li][e] * miss
                resident_n += len(resident)
            print(f"| {budget / 1e9:.1f} GB | {label} | {100 * resident_n / N_LAYERS / N_EXPERT:.1f} % "
                  f"| {100 * hit / total:.1f} % | {cpu / tokens / 1e6:.0f} |")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
