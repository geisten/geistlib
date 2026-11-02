import json,sys,statistics,collections
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    if not l.startswith('{'): continue
    j=json.loads(l); r=j['r']
    d[j['v']].append((r['prefill_tps'],r['decode_tps'],j['th'],j['staged']))
for v,xs in d.items():
    p=[x[0] for x in xs]; q=[x[1] for x in xs]
    print(f"{v:32s} prefill {' / '.join(f'{a:.0f}' for a in p):24s} med {statistics.median(p):.0f}  decode {' / '.join(f'{a:.1f}' for a in q)} med {statistics.median(q):.1f}  {xs[0][2]} {xs[0][3]}")
