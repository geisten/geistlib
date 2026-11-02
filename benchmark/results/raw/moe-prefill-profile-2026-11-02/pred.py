import sys,glob,collections
R='/tmp/geistlib-moe-prefill-profile/benchmark/results/raw/'
def usage_file(path,L,E):
    c=[0]*(L*E)
    for line in open(path):
        p=line.split()
        if len(p)>=7 and p[0]=='USAGE' and p[2]=='profile':
            l=int(p[5]); 
            if l>=L: continue
            for e,x in enumerate(p[6].split(',')[:E]): c[l*E+e]+=int(x)
    return c
def cache(path):
    L=E=0; hist={}; use=None
    for line in open(path):
        p=line.split()
        if line.startswith('#'): L=int(p[7+1]); E=int(p[9+1]); use=[0]*(L*E)
        elif p[0]=='USAGE':
            l=int(p[5]); 
            for e,x in enumerate(p[6].split(',')): use[l*E+e]=int(x)
        elif p[0]=='PREFILL':
            lo=int(p[3]); l=int(p[4]); b=lo.bit_length()-1
            for e,x in enumerate(p[5].split(',')): hist[(l*E+e,b)]=int(x)
    return L,E,use,hist
def rows(hist,L,E,lo):
    b0=lo.bit_length()-1
    return [sum(hist.get((i,b),0) for b in range(b0,10)) for i in range(L*E)]
for model,P,nhot,ncold in [('Qwen3.6-35B-A3B-UD-Q4_K_XL','2026-10-23_qwen36_expert_usage_run1830.txt',3917,2242),('Qwen3-30B-A3B-Q4_K_M','2026-10-15_moe_expert_usage_vulkan_run1225.txt',2832,1340)]:
    base=f'{sys.argv[1]}/'
    L,E,_,hs=cache(glob.glob(base+f'cache-same-{model}/*.txt')[0])
    _,_,_,hc=cache(glob.glob(base+f'cache-cross-{model}/*.txt')[0])
    dec=usage_file(R+P,L,E)
    rank=sorted(range(L*E),key=lambda i:(-dec[i],(i%E)*L+i//E))
    hot=set(rank[:nhot]); cold=[i for i in rank if i not in hot]
    tgt=rows(hs,L,E,32); tot=sum(tgt[i] for i in cold)
    def cov(key):
        sel=sorted(cold,key=key)[:ncold]; return 100*sum(tgt[i] for i in sel)/tot
    print(model,'cold',len(cold),'target staged rows',tot)
    print('  decode order (main)      %.1f'%cov(lambda i:(rank.index(i))) if False else '  decode order (main)      %.1f'%cov(lambda i:(-dec[i],(i%E)*L+i//E)))
    for lo in (32,16,8,4,1):
        r=rows(hc,L,E,lo)
        print(f'  cross rows>={lo:<3d} +decode  %.1f'%cov(lambda i:(-r[i],-dec[i],(i%E)*L+i//E)))
    r32=rows(hc,L,E,32); r1=rows(hc,L,E,1)
    print('  cross r32, then r1, dec  %.1f'%cov(lambda i:(-r32[i],-r1[i],-dec[i])))
    s32=rows(hs,L,E,32)
    print('  same rows>=32 (oracle)   %.1f'%cov(lambda i:(-s32[i],-dec[i])))
    for lo in (16,8,4):
        r=rows(hs,L,E,lo)
        print(f'  same rows>={lo:<3d} +decode   %.1f'%cov(lambda i:(-r[i],-dec[i])))
    # reverse: cross target, same predictor
    tgt2=rows(hc,L,E,32); tot2=sum(tgt2[i] for i in cold)
    for lo in (32,16,8):
        r=rows(hs,L,E,lo); sel=sorted(cold,key=lambda i:(-r[i],-dec[i]))[:ncold]
        print(f'  [arch target] same rows>={lo} %.1f'%(100*sum(tgt2[i] for i in sel)/tot2))
    sel=sorted(cold,key=lambda i:(-dec[i],(i%E)*L+i//E))[:ncold]; print('  [arch target] decode %.1f'%(100*sum(tgt2[i] for i in sel)/tot2))
    r32=rows(hc,L,E,32); r8=rows(hc,L,E,8); r16=rows(hc,L,E,16)
    print('  cross r32,r8,dec          %.1f'%cov(lambda i:(-r32[i],-r8[i],-dec[i])))
    print('  cross r32+r8              %.1f'%cov(lambda i:(-(r32[i]+r8[i]),-dec[i])))
    print('  cross r32+r16+r8          %.1f'%cov(lambda i:(-(r32[i]+r16[i]+r8[i]),-dec[i])))
    # double-profile: same+cross accumulated, target same
    hsum={k:hs.get(k,0)+hc.get(k,0) for k in set(hs)|set(hc)}
    for lo in (32,16,8):
        r=rows(hsum,L,E,lo); print(f'  same+cross rows>={lo}       %.1f'%cov(lambda i:(-r[i],-dec[i])))
