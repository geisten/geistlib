#!/usr/bin/env python3
"""strata_compare.py - Strata, llama.cpp and geistlib on one model (#169).

Run by .gitea/workflows/strata-compare.yml, which sets MODEL, GBIN,
LLAMA_CUDA, LLAMA_VULKAN, STRATA_DIR, STRATA_CFG, STRATA_PY, ROUNDS,
ENGINES and PROFILE. Engines:

  strata, strata-mtp2   Strata's server (serve/server.py, its setup's config),
                        --mtp 0 (plain decode) or --mtp 2 (its MTP draft)
  llama-cuda-<k>        llama-server, CUDA, -ngl 99 --n-cpu-moe k
  llama-vk-<k>          llama-server, Vulkan, the same
  geist-gpu             bench_perf_sweep on Vulkan (experts on the CPU, the
                        hottest copied to the card), GEIST_MOE_PROFILE
  geist-cpu             bench_perf_sweep on cpu_x86, the model's default chunk

The servers get one chat request: the first PROMPT_CHARS characters of
docs/OFFLOAD.md, 128 tokens, greedy; prefill and decode tok/s are their
llama.cpp-style timings (Strata's server emits the same fields). geistlib
prefills the number of prompt tokens the servers reported, from the same
file, and decodes 128. One warm-up request per start; rounds alternate the
engine order; the summary is the median over rounds. Prints one JSON line
per measurement and a table at the end.
"""
import json, os, statistics, subprocess, sys, time, urllib.request

PORT = 8191
PROMPT_CHARS = 8000  # about 2,000 tokens of this file
GEN = 128
TEXT = open("docs/OFFLOAD.md", encoding="utf-8").read()[:PROMPT_CHARS]
OUT = "out"


def http(path, body=None, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def server_cmd(engine):
    if engine.startswith("strata"):
        cfg = json.load(open(os.environ["STRATA_CFG"], encoding="utf-8-sig"))
        cfg["port"] = PORT
        cfg.pop("vision", None)
        args = list(cfg["args"])
        mtp = "2" if engine == "strata-mtp2" else "0"
        if "--mtp" in args:
            args[args.index("--mtp") + 1] = mtp
        else:
            args += ["--mtp", mtp]
        cfg["args"] = args
        path = f"{OUT}/strata-cfg-{engine}.json"
        json.dump(cfg, open(path, "w"), indent=1)
        return [os.environ["STRATA_PY"], f"{os.environ['STRATA_DIR']}/serve/server.py", "--engine",
                "strata", "--config", os.path.abspath(path), "--port", str(PORT)], os.environ["STRATA_DIR"]
    kind, k = engine.rsplit("-", 1)
    exe, dev = (os.environ["LLAMA_CUDA"], "CUDA0") if kind == "llama-cuda" else (os.environ["LLAMA_VULKAN"], "Vulkan0")
    return [exe, "-m", os.environ["MODEL"], "--device", dev, "-ngl", "99", "--n-cpu-moe", k, "-c", "8192",
            "-np", "1", "--port", str(PORT), "--host", "127.0.0.1", "--jinja", "--no-webui"], None


def run_server(engine):
    cmd, cwd = server_cmd(engine)
    log = open(f"{OUT}/server-{engine}.log", "a")
    log.write(f"\n== {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd)
    try:
        t0 = time.time()
        while True:
            if p.poll() is not None:
                return {"error": f"server exited rc={p.returncode}"}
            if time.time() - t0 > 1500:
                return {"error": "not ready in 1500 s"}
            try:
                if http("/health", timeout=5).get("status", "ok") == "ok":
                    break
            except Exception:
                pass
            time.sleep(2)
        body = {"messages": [{"role": "user", "content": "Summarise this document:\n\n" + TEXT}],
                "max_tokens": GEN, "temperature": 0, "stream": False,
                # llama-server would keep the warm-up's prompt and prefill only
                # the last tokens of the measured request (run 1838: 4 of 2,726)
                "cache_prompt": False}
        http("/v1/chat/completions", body)  # warm-up
        t = http("/v1/chat/completions", body)["timings"]
        return {"prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
                "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
                "load_s": round(time.time() - t0, 1)}
    finally:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def run_geist(engine, prompt_n):
    gpu = engine.startswith("geist-gpu")
    env = dict(os.environ, GEIST_BENCH_BACKEND="vulkan" if gpu else "cpu_x86", GEIST_VERBOSE="1")
    if engine == "geist-gpu-nopad":
        env["GEIST_MOE_PAD"] = "0"  # an A/B of #167's padding in one build
    if os.environ.get("GEIST_DIAG"):  # per-stage times and the Vulkan memory plan, per run
        env.update(GEIST_PROFILE_FORWARD="1", GEIST_VK_VERBOSE="1")
    if gpu and os.environ.get("PROFILE"):
        env["GEIST_MOE_PROFILE"] = os.environ["PROFILE"]
    cmd = [f"{os.environ['GBIN']}/bench_perf_sweep", "--gguf", os.environ["MODEL"], "--seq-lens", str(prompt_n),
           "--decode-n", str(GEN), "--warmup", "8", "--repeats", "1", "--prompt-file", "docs/OFFLOAD.md"]
    if gpu:
        cmd += ["--m-max", "512"]
    run_geist.n = getattr(run_geist, "n", 0) + 1
    with open(f"{OUT}/geist-{run_geist.n:02d}-{engine}.log", "w") as log:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            return {"prompt_n": prompt_n, "predicted_n": GEN, "prefill_tps": d["prefill_tps"],
                    "decode_tps": d["decode_tps"]}
    return {"error": f"bench_perf_sweep rc={r.returncode}"}


def main():
    engines = [e for e in os.environ["ENGINES"].split(";") if e]
    rounds = int(os.environ.get("ROUNDS", "3"))
    geists = [e for e in engines if e.startswith("geist-")]
    servers = [e for e in engines if e not in geists]
    prompt_n = None
    res = {e: [] for e in engines}
    for r in range(rounds):
        order = servers if r % 2 == 0 else servers[::-1]
        for e in order:
            m = run_server(e)
            prompt_n = prompt_n or m.get("prompt_n")
            print(json.dumps({"engine": e, "round": r + 1, **m}), flush=True)
            res[e].append(m)
        for e in geists:  # needs the prompt length the servers reported
            m = run_geist(e, prompt_n or 2000)
            print(json.dumps({"engine": e, "round": r + 1, **m}), flush=True)
            res[e].append(m)
    print(f"\n| engine | prompt tokens | prefill tok/s (median) | decode tok/s (median) | rounds |")
    print("|:--|--:|--:|--:|--:|")
    for e in engines:
        ok = [m for m in res[e] if "error" not in m and m.get("decode_tps")]
        if not ok:
            print(f"| {e} | | failed: {res[e][-1].get('error') if res[e] else '-'} | | 0 |")
            continue
        med = lambda k: statistics.median(m[k] for m in ok)
        print(f"| {e} | {ok[0]['prompt_n']} | {med('prefill_tps'):.0f} | {med('decode_tps'):.1f} | {len(ok)} |")


if __name__ == "__main__":
    sys.exit(main())
