import json, sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
for p in dict.fromkeys(r['prompt'] for r in rows):
    rs=[r for r in rows if r['prompt']==p]
    ref=next(r for r in rs if r['bin']=='base_bench' and r['v']=='OFF')['r']
    rt, mg = ref['tokens'], ref['margins']
    small=sorted(range(len(mg)), key=lambda i: mg[i][1])[:3]
    print(f"{p}: seq {ref['seq_len']}, closest reference margins: " + ", ".join(f"step {i} {mg[i][1]:.3f}" for i in small))
    for r in rs:
        if r is not None and r['bin']=='base_bench' and r['v']=='OFF': continue
        t=r['r']['tokens']; d=next((i for i,(a,b) in enumerate(zip(t,rt)) if a!=b),None)
        sp=r['r'].get('spec')
        s=f"  {r['bin']:10} {r['v']:5} dec {r['r']['decode_tps']:6.2f} "
        if d is None: s+="identical over %d" % len(t)
        else: s+=f"first difference at {d}: ref {rt[d]} got {t[d]}; ref runner-up {mg[d][0]} gap {mg[d][1]:.3f}" + (" (= runner-up)" if mg[d][0]==t[d] else " (NOT runner-up)")
        if sp: s+=f"  | {sp['decoded']/sp['calls']:.2f} tok/call, accept " + "/".join(f"{a/sp['calls']:.2f}" for a in sp['accept'])
        print(s)
