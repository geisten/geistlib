# #444 cause search, part 2: CPU qwen cell (geist) and the llama.cpp side of the Metal 27B cell, with macmon trace.
import json, os, subprocess, sys, time, threading
D = "/Users/germar/workspace/geistlib-campaigns/444"
sys.path.insert(0, D + "/tools/tools")
import bench_cross_engine as b, bench_host as h
from pathlib import Path
trace = []
def sampler():
    p = subprocess.Popen(["macmon", "pipe", "-i", "100"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in p.stdout:
        d = json.loads(line)
        trace.append((time.time(), d["temp"]["cpu_temp_avg"], d["temp"]["gpu_temp_avg"], d["pcpu_freq_mhz"], d["pcpu_active_ratio"],
                      d["ecpu_active_ratio"], d["gpu_freq_mhz"], d["gpu_active_ratio"]))
threading.Thread(target=sampler, daemon=True).start()
def setup(proto_name, profile, geist_dir):
    proto = json.load(open(f"{D}/tools/benchmark/cross_engine_{proto_name}_protocol.json"))
    prof = proto["host_profiles"][profile]
    return {**proto["workload"], **{k: v for k, v in prof.items() if k != "host_gate"}}, prof["host_gate"]
def run(eng, model, wl):
    return b.run_geist(eng, model, wl) if eng["kind"] == "geist" else b.run_llama(eng, model, wl, 0)
def call(tag, eng, model, wl, gate, gpu):
    h.wait_for_apple_cool(gate["apple_temp_gate"], time.monotonic() + 900)
    t0 = time.time(); r = run(eng, model, wl); t1 = time.time()
    ai, fi, ti = (7, 6, 2) if gpu else (4, 3, 1)
    w = [x for x in trace if t0 <= x[0] <= t1 and x[ai] > 0.5]
    print(f"{tag} load {os.getloadavg()[0]:.1f} spread {b.repeat_spread_pct(r):.1f} %", flush=True)
    for row in r["rows"]:
        print("   ", row["seq_len"], "pp", [round(v, 1) for v in row["prefill_tps"]], "tg", [round(v, 1) for v in row["decode_tps"]], flush=True)
    step = max(1, len(w) // 28)
    print("    C/MHz" + ("" if gpu else "/E-core-active") + ":", " ".join(("%.0f/%d" % (x[ti], x[fi])) + ("" if gpu else "/%.1f" % x[5]) for x in w[::step]), flush=True)
part = sys.argv[1]
if part == "qwen":
    wl, gate = setup("cpu", "apple_m1max", "cpu")
    eng = b.engine_metadata("geist", "geist", Path(D + "/cpu/bin/mac-omp/release/tests/bench_perf_sweep"))
    model = Path("/Users/germar/geist-models/qwen3.5-0.8b-q8_0.gguf")
    for i in range(2): call(f"Q-A{i} protocol call back to back", eng, model, wl, gate, False)
    for i in range(2): call(f"Q-B{i} 512 x12 in one call", eng, model, {**wl, "seq_lens": [512], "repeats": 12}, gate, False)
    for i in range(8): call(f"Q-C{i} 512 single cooled", eng, model, {**wl, "seq_lens": [512], "repeats": 1}, gate, False)
else:
    wl, gate = setup("gpu", "apple_m1max_metal", "metal")
    eng = b.engine_metadata("llama.cpp", "llama", Path("/Users/germar/llama.cpp-cross-engine-PrismML-Eng_llama.cpp/build-metal/bin/llama-bench"))
    model = Path("/Users/germar/geist-models/Ternary-Bonsai-2-27B-PQ2_0.gguf")
    for i in range(2): call(f"L-B{i} 512 x6 in one call", eng, model, {**wl, "seq_lens": [512], "repeats": 6}, gate, True)
    for s in (128, 1024):
        for i in range(3): call(f"L-C{s}/{i} single cooled", eng, model, {**wl, "seq_lens": [s], "repeats": 1}, gate, True)
