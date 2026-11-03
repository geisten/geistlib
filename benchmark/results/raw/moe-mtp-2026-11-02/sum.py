import sys, json, collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
ref={}
for d in rows:
    if d.get('error'): continue
    if d['v']=='OFF' and d['bin']=='base_bench': ref.setdefault(d['prompt'], d['r']['tokens'])
for d in rows:
    if d.get('error'): print('ERR', d); continue
    r=d['r']; sp=r.get('spec')
    tk=r.get('tokens',[]); rf=ref.get(d['prompt'])
    if rf is None:
        rf = next((x['r']['tokens'] for x in rows if not x.get('error') and x['v']=='OFF' and x['prompt']==d['prompt']), None)
    div = next((i for i,(a,b) in enumerate(zip(tk,rf)) if a!=b), None) if rf else None
    acc=''
    if sp: acc='calls %d tok/call %.2f acc %s' % (sp['calls'], sp['decoded']/sp['calls'], [round(a/sp['calls'],2) for a in sp['accept']])
    print('%d %-10s %-5s %-10s %-22s load %5.1f pre %6.1f dec %6.2f div %s %s' % (d['cycle'], d['bin'], d['v'], d['prompt'], d['env'], d['load'], r['prefill_tps'], r['decode_tps'], div, acc))
