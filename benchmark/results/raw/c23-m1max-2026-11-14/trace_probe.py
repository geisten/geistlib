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
# part 4: 256/512 from 55 C scattered 2-3x in part 3 with the GPU clock at 388 MHz at 61-75 C. Who else is busy during a call?
others = {}
def ps_sampler():
    while True:
        out = subprocess.run(["ps", "-Aro", "%cpu,comm"], capture_output=True, text=True).stdout.splitlines()[1:6]
        for l in out:
            c, n = l.strip().split(None, 1); n = n.rsplit("/", 1)[-1]
            if "llama-bench" not in n: others[n] = max(others.get(n, 0), float(c.replace(",", ".")))
        time.sleep(1)
threading.Thread(target=ps_sampler, daemon=True).start()
wl, gate = setup("gpu", "apple_m1max_metal", "metal")
eng = b.engine_metadata("llama.cpp", "llama", Path("/Users/germar/llama.cpp-cross-engine-PrismML-Eng_llama.cpp/build-metal/bin/llama-bench"))
model = Path("/Users/germar/geist-models/Ternary-Bonsai-2-27B-PQ2_0.gguf")
g = {**gate, "apple_temp_gate": {**gate["apple_temp_gate"], "max_gpu_c": 55}}
for s in (256, 512, 1024):
    for i in range(4):
        others.clear()
        call(f"L-E55C/{s}/{i} single", eng, model, {**wl, "seq_lens": [s], "repeats": 1}, g, True)
        print("    others max %cpu:", sorted(others.items(), key=lambda x: -x[1])[:4], subprocess.run("pmset -g batt | tail -1 | cut -f2", shell=True, capture_output=True, text=True).stdout.strip()[:40], flush=True)
