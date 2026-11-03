import json,sys,collections,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
ref={}
for r in rows:
    if r['r'] and r['bin']=='base' and r['v']=='OFF': ref[(r['cycle'],r['prompt'])]=r['r']['tokens']
g=collections.defaultdict(list)
for r in rows:
    if not r['r']: continue
    g[(r['prompt'],r['bin'],r['v'])].append(r)
for k in sorted(g):
    rs=g[k]; tps=[x['r']['decode_tps'] for x in rs]
    same=sum(1 for x in rs if x['r']['tokens']==ref.get((x['cycle'],x['prompt'])))
    sp=rs[0]['r'].get('spec')
    acc=''
    if sp:
        calls=sum(x['r']['spec']['calls'] for x in rs); a=[sum(x['r']['spec']['accept'][i] for x in rs) for i in range(len(sp['accept']))]
        acc=' tok/call %.2f accept %s'%(sum(x['r']['spec']['decoded'] for x in rs)/calls,'/'.join('%.2f'%(ai/calls) for ai in a))
    print('%-16s %-4s %-5s tps %s median %.2f prefill %s same %d/%d%s'%(k[0],k[1],k[2],' '.join('%.2f'%t for t in tps),st.median(tps),' '.join('%.1f'%x['r']['prefill_tps'] for x in rs),same,len(rs),acc))
