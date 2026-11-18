#!/usr/bin/env python3
"""#286 morning progress for the issue: counts, tokens, lease, throughput; never accuracy."""
import glob, json, sys
D = "/Users/germar/workspace/geistlib-campaigns/286"
def job(name):
    try:
        st = json.load(open(f"{D}/{name}/status.json"))
    except Exception:
        return None
    tokens = lease = 0.0
    stopped = 0
    for c in sorted(glob.glob(f"{D}/{name}/chunks/*")):
        try:
            ch = json.load(open(c + "/chunk.json"))
        except Exception:
            continue
        if ch.get("ended_unix"):
            lease += ch["ended_unix"] - ch["started_unix"]
            stopped += bool(ch.get("run", {}).get("stopped_at_deadline"))
        for line in open(c + "/batched.driver.jsonl"):
            r = json.loads(line)
            if r.get("kind") == "sample":
                tokens += len(r["generated_ids"])
    return st, tokens, lease, stopped
lines = []
for name, total in (("main", 3500), ("latency", 300), ("latency-full", 300)):
    r = job(name)
    if r is None:
        lines.append(f"- **{name}**: not started")
        continue
    st, tokens, lease, stopped = r
    rate = tokens / lease if lease else 0
    per_q = lease / st["complete"] if st["complete"] else 0
    left = per_q * st["pending"] / 3600
    lines.append(f"- **{name}**: {st['complete']}/{st['questions']} questions complete, {st['chunks']} chunks "
                 f"({stopped} cut at the deadline), {tokens:,.0f} generated tokens, {lease/3600:.1f} h of lease, "
                 f"{rate:.1f} tok/s over the lease; at {per_q:.0f} s of lease per question about {left:.1f} h left")
import subprocess, time
since = time.time() - 14 * 3600
for name in ("main", "latency", "latency-full"):
    try:
        rows = [l.split() for l in open(f"{D}/power-{name}.log") if l.strip()]
    except FileNotFoundError:
        continue
    rows = [r for r in rows if float(r[0]) >= since]
    if rows:
        ac = sum(r[1] == "AC" for r in rows); lpm = sum(r[2] == "lpm=1" for r in rows)
        flag = "" if ac == len(rows) and lpm == 0 else " -- **not all on AC with Low Power Mode off**"
        lines.append(f"- **power during the night's {name} chunks** (sampled every 30 s): "
                     f"{ac}/{len(rows)} samples on AC, {lpm}/{len(rows)} with Low Power Mode on{flag}")
now = subprocess.run([D + "/power.sh", "state"], capture_output=True, text=True).stdout.split()
lines.append(f"- power now: {now[1]}, Low Power Mode {now[2][4:]} (the latency phase waits for AC with it off)")
print("\n".join(lines))
