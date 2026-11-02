import json,sys,statistics,collections
d=collections.defaultdict(list); meta={}
for l in open(sys.argv[1]):
    if not l.startswith('{'): continue
    j=json.loads(l); r=j['r']
    d[j['v']].append((r['prefill_tps'],r['decode_tps']))
    meta.setdefault(j['v'],set()).add((j['th'],j['staged'],j['hot']))
med={v:statistics.median([x[0] for x in xs]) for v,xs in d.items()}
off=med.get('OFF'); al=med.get('ALL')
for v,xs in d.items():
    p=[x[0] for x in xs]; q=[x[1] for x in xs]
    sh=f"{100*(med[v]-off)/(al-off):.0f} %" if off and al and al!=off else ''
    print(f"{v:7s} prefill {' / '.join(f'{a:.0f}' for a in p):20s} med {med[v]:.0f} share {sh:6s} decode {' / '.join(f'{a:.1f}' for a in q)} med {statistics.median(q):.1f}")
    for m in sorted(meta[v]): print('        ',m)
