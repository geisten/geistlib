#!/usr/bin/env python3
"""Explore M6 PQ2 kernels under an exclusive host lease. Stdlib only.

Same binary, deterministic matrix fixture and actual production linear path.
Each subprocess discards 64 warmups; timing includes preparation and wait.
Two rotated outer cycles are exploratory, not a statistical no-regression gate.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from bench_host import command_output, sha256_file, wait_for_apple_cool, wait_for_apple_power
ROOT=Path(__file__).resolve().parents[1]
MODES=('int8','fp8','int2','half2','coop16','coop8','staged_int8','staged_fp8')
SHAPES=[(128,17408,5120),(256,5120,17408),(512,17408,5120)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cycles',type=int,default=2)
    ap.add_argument('--modes',default=','.join(MODES))
    ap.add_argument('--shape',action='append',help='M,N,K')
    args=ap.parse_args()
    if not 1<=args.cycles<=8:ap.error('cycles must be 1..8')
    modes=args.modes.split(',')
    if any(m not in MODES for m in modes):ap.error('unknown mode')
    shapes=[tuple(map(int,s.split(','))) for s in args.shape] if args.shape else SHAPES
    binary=ROOT/'bin/mac-omp/release/tests/bench_metal_pq2'
    configs=[dict(mode='off',tile=128,simdgroups=8,shared=False)]
    for mode in modes:
        configs += [dict(mode=mode,tile=tile,simdgroups=sg,shared=shared)
                    for shared in ((False,True) if mode in MODES[:4] else (False,))
                    for sg in (4,8) for tile in ((32,64) if shared else (32,64,128))]
    args.out.mkdir(parents=True,exist_ok=True)
    meta=dict(cpu=command_output(['sysctl','-n','machdep.cpu.brand_string']),os=command_output(['sw_vers']),
              revision=command_output(['git','rev-parse','HEAD']),binary_sha256=sha256_file(binary),
              configs=configs,shapes=shapes,outer_cycles=args.cycles,inner_cycles=2,repeats=8,warmup=64,
              notes='Exploratory rotated-order sweep; wall timing includes actual preparation, GEMM and host wait. Independent CPU oracle per point.')
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    rows=[]
    with (args.out/'runs.jsonl').open('w') as output:
        for cycle in range(args.cycles):
            for shape in shapes:
                order=configs if cycle%2==0 else list(reversed(configs))
                for index,cfg in enumerate(order):
                    label=f"{cfg['mode']}-m{cfg['tile']}-s{cfg['simdgroups']}-a{int(cfg['shared'])}"
                    print(f'{cycle+1}/{args.cycles} {shape} {index+1}/{len(configs)} {label}',flush=True)
                    deadline=time.monotonic()+1200
                    power=wait_for_apple_power(deadline)
                    thermal=wait_for_apple_cool(dict(max_cpu_c=65,max_gpu_c=65,fallback_seconds=45),deadline)
                    env=dict(os.environ,GEIST_METAL_TENSOR='1',GEIST_METAL_PQ2_MODE=cfg['mode'],
                             GEIST_METAL_PQ2_TILE=str(cfg['tile']),GEIST_METAL_PQ2_SIMDGROUPS=str(cfg['simdgroups']),
                             GEIST_METAL_PQ2_SHARED_ACC=str(int(cfg['shared'])))
                    with (args.out/f'{cycle}-{shape[0]}-{shape[1]}-{shape[2]}-{label}.err').open('w') as err:
                        proc=subprocess.run([str(binary),*map(str,shape),'8','2'],env=env,stdout=subprocess.PIPE,stderr=err,text=True,timeout=120)
                    if proc.returncode!=0:raise RuntimeError(f'{label} failed: {proc.returncode}')
                    parsed=[json.loads(x) for x in proc.stdout.splitlines()]
                    if len(parsed)!=2:raise RuntimeError('incomplete matrix sweep')
                    for r in parsed:
                        r.update(config=cfg,outer_cycle=cycle,power=power,thermal=thermal)
                        rows.append(r);output.write(json.dumps(r)+'\n')
                    output.flush()
    summary=[]
    for shape in shapes:
        base=statistics.median(r['wall_us'] for r in rows if (r['m'],r['n'],r['k'])==shape and r['mode']==0)
        for cfg in configs:
            selected=[r for r in rows if (r['m'],r['n'],r['k'])==shape and r['config']==cfg]
            us=statistics.median(r['wall_us'] for r in selected)
            summary.append(dict(m=shape[0],n=shape[1],k=shape[2],config=cfg,wall_us=us,speedup=base/us))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for shape in shapes:
        ranked=sorted((r for r in summary if (r['m'],r['n'],r['k'])==shape),key=lambda r:r['wall_us'])
        print(json.dumps(ranked[:8],indent=2),flush=True)


if __name__=='__main__':main()
