#!/usr/bin/env python3
"""Compare opt-in PQ2 modes on identical Bonsai weights and one executable.

Run under an exclusive M6 lease after building bench_perf_sweep, eval_geist,
dump_geist_logits and test_metal_pq2_lowp_unit. No downloads/dependencies.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from bench_host import command_output,sha256_file,wait_for_apple_cool,wait_for_apple_power
from eval_repl import EvalRepl
from pq2_quality_cases import CASES
ROOT=Path(__file__).resolve().parents[1]


def config(text):
    label,body=text.split('=',1)
    mode,tile,sg,shared=body.split(',')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',label) or label=='baseline':raise ValueError('invalid config label')
    if mode not in ('int8','fp8','int2','half2','coop16','coop8','staged_int8','staged_fp8'):raise ValueError('unknown mode')
    if int(tile) not in (32,64,128) or int(sg) not in (4,8) or int(shared) not in (0,1):raise ValueError('invalid geometry')
    return dict(label=label,mode=mode,tile=int(tile),simdgroups=int(sg),shared=bool(int(shared)))


def env_for(cfg,profile=False):
    return dict(os.environ,GEIST_BACKEND='metal',GEIST_BENCH_BACKEND='metal',GEIST_DUMP_BACKEND='metal',
                GEIST_METAL_TENSOR='1',GEIST_METAL_PQ2_MODE=cfg['mode'],GEIST_METAL_PQ2_TILE=str(cfg['tile']),
                GEIST_METAL_PQ2_SIMDGROUPS=str(cfg['simdgroups']),GEIST_METAL_PQ2_SHARED_ACC=str(int(cfg['shared'])),
                GEIST_METAL_PROFILE=str(int(profile)),OMP_NUM_THREADS='4')


def gate():
    deadline=time.monotonic()+1200
    return dict(power=wait_for_apple_power(deadline),thermal=wait_for_apple_cool(
        dict(max_cpu_c=65,max_gpu_c=65,fallback_seconds=45),deadline))


def quality(binary,model,cfg,out):
    label=cfg['label'];log=out/f'quality-{label}.err'
    repl=EvalRepl(binary,model,log=log);rows=[]
    try:
        for name,prompt,continuation in CASES:
            ids=repl.encode(prompt)
            if repl.bos is not None:ids.insert(0,repl.bos)
            if len(ids)<512:raise RuntimeError('quality prompt does not exercise large prefill')
            cont=repl.encode(continuation);prefix=f'{len(ids)} '+' '.join(map(str,ids))
            score=repl.command(f'SCORE {prefix} {len(cont)} '+' '.join(map(str,cont)))
            probs=list(map(float,score[1:]))
            decoded=repl.command(f'DECODE {prefix} 24');tokens=list(map(int,decoded[1:]))
            escaped=prompt.replace('\\','\\\\').replace('\n','\\n').replace('\t','\\t').replace('\r','\\r')
            text=' '.join(repl.command('GEN 24 '+escaped))
            if len(probs)!=len(cont) or not all(math.isfinite(v) for v in probs):raise RuntimeError('invalid quality scores')
            if len(tokens)!=24 or int(decoded[0])!=24 or any(t<0 for t in tokens):raise RuntimeError('invalid decode stream')
            rows.append(dict(arm=label,case=name,prompt=prompt,continuation=continuation,prompt_ids=ids,
                             continuation_ids=cont,logprobs=probs,generated_ids=tokens,generated_text=text))
    finally:repl.close()
    match=re.search(r'dispatch\.pq2_lowp_x\s+(\d+) dispatches',log.read_text())
    calls=int(match.group(1)) if match else 0
    if (calls>0)!=(label!='baseline' and cfg['mode']!='coop16'):raise RuntimeError('quality did not exercise requested mode')
    for row in rows:row['run_quant_dispatches']=calls
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--config',action='append',default=[])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bin-dir',type=Path,default=ROOT/'bin/mac-omp/release')
    ap.add_argument('--cycles',type=int,default=6)
    ap.add_argument('--seq-lens',default='128,512,2048')
    ap.add_argument('--m-max',type=int,default=0)
    ap.add_argument('--skip-quality',action='store_true')
    args=ap.parse_args()
    if args.cycles<6:ap.error('at least six rotated cycles required')
    seqs=[int(s) for s in args.seq_lens.split(',')]
    if not seqs or min(seqs)<64 or max(seqs)>8192:ap.error('sequence lengths must be 64..8192')
    configs=[dict(label='baseline',mode='off',tile=128,simdgroups=8,shared=False)]
    configs += [config(s) for s in (args.config or ['int8=staged_int8,128,4,0','fp8=staged_fp8,128,4,0'])]
    if len({c['label'] for c in configs})!=len(configs):ap.error('config labels must be unique')
    args.out.mkdir(parents=True,exist_ok=True)
    bench=args.bin_dir/'tests/bench_perf_sweep';evaluate=args.bin_dir/'tools/eval_geist';dump=args.bin_dir/'tools/dump_geist_logits'
    meta=dict(cpu=command_output(['sysctl','-n','machdep.cpu.brand_string']),os=command_output(['sw_vers']),
              revision=command_output(['git','rev-parse','HEAD']),model=dict(path=str(args.model),sha256=sha256_file(args.model)),
              binary_sha256={str(b.relative_to(args.bin_dir)):sha256_file(b) for b in (bench,evaluate,dump)},
              configs=configs,cycles=args.cycles,protocol=dict(seq_lens=seqs,decode_n=32,warmup=max(seqs),repeats=1,m_max=args.m_max,threads=4))
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    records=[]
    with (args.out/'runs.jsonl').open('w') as out:
        for cycle in range(args.cycles):
            order=configs[cycle%len(configs):]+configs[:cycle%len(configs)]
            for cfg in order:
                print(f'cycle {cycle+1}/{args.cycles} {cfg["label"]}',flush=True)
                state=gate()
                command=[str(bench),'--gguf',str(args.model),'--seq-lens',args.seq_lens,'--decode-n','32','--warmup',str(max(seqs)),'--repeats','1']
                if args.m_max:command+=['--m-max',str(args.m_max)]
                with (args.out/f'perf-{cycle}-{cfg["label"]}.err').open('w') as err:
                    run=subprocess.run(command,env=env_for(cfg),stdout=subprocess.PIPE,stderr=err,text=True,timeout=1200,check=True)
                rows=[json.loads(s) for s in run.stdout.splitlines()]
                if len(rows)!=len(seqs):raise RuntimeError('incomplete performance sweep')
                for row in rows:
                    active=row['pq2_native_calls']>0
                    if active!=(cfg['label']!='baseline'):raise RuntimeError('performance arm silently fell back')
                    if cfg['mode']!='coop16' and row['metal_pq2_quant_dispatches']!=row['pq2_native_calls']:raise RuntimeError('unexpected preparation reuse/count')
                    row.update(arm=cfg['label'],cycle=cycle,**state)
                    out.write(json.dumps(row)+'\n');out.flush();records.append(row)
    qrows=[];old_env=dict(os.environ)
    if not args.skip_quality:
        try:
            for cfg in configs:
                os.environ.update(env_for(cfg,True));print('quality '+cfg['label'],flush=True)
                qrows += quality(evaluate,args.model,cfg,args.out)
                for case in (0,3,4):
                    prompt=args.out/f'prompt-{case}.txt';prompt.write_text(CASES[case][1])
                    with (args.out/f'logits-{case}-{cfg["label"]}.err').open('w') as err:
                        subprocess.run([str(dump),str(args.model),str(prompt),str(args.out/f'ids-{case}-{cfg["label"]}.bin'),str(args.out/f'logits-{case}-{cfg["label"]}.bin')],stderr=err,timeout=300,check=True)
        finally:os.environ.clear();os.environ.update(old_env)
        (args.out/'quality.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in qrows))
    summary=dict(performance=[],quality=[],logits=[])
    for cfg in configs[1:]:
        for seq in seqs:
            arms={label:[r for r in records if r['arm']==label and r['seq_len']==seq] for label in ('baseline',cfg['label'])}
            row=dict(arm=cfg['label'],seq_len=seq)
            for label,rs in arms.items():row[label]={k:statistics.median(r[k] for r in rs) for k in ('prefill_tps','decode_tps','rss_mb','backend_allocated_mb')}
            ratios=[b['prefill_tps']/a['prefill_tps'] for a,b in zip(arms['baseline'],arms[cfg['label']])]
            row.update(paired_speedup=statistics.median(ratios),paired_range=[min(ratios),max(ratios)],wins=sum(v>1 for v in ratios))
            summary['performance'].append(row)
        if args.skip_quality:continue
        arms={label:[r for r in qrows if r['arm']==label] for label in ('baseline',cfg['label'])}
        for a,b in zip(arms['baseline'],arms[cfg['label']]):
            if a['prompt_ids']!=b['prompt_ids'] or a['continuation_ids']!=b['continuation_ids']:raise RuntimeError('quality token IDs differ')
        nll={label:-sum(sum(r['logprobs']) for r in rs)/sum(len(r['logprobs']) for r in rs) for label,rs in arms.items()}
        summary['quality'].append(dict(arm=cfg['label'],mean_nll=nll,conditional_ppl_ratio=math.exp(nll[cfg['label']]-nll['baseline']),
            cases=len(CASES),continuation_tokens=sum(len(r['logprobs']) for r in arms['baseline']),
            exact_generations=sum(a['generated_ids']==b['generated_ids'] for a,b in zip(arms['baseline'],arms[cfg['label']])),
            first_token_matches=sum(a['generated_ids'][0]==b['generated_ids'][0] for a,b in zip(arms['baseline'],arms[cfg['label']]))))
        for case in (0,3,4):
            if (args.out/f'ids-{case}-baseline.bin').read_bytes()!=(args.out/f'ids-{case}-{cfg["label"]}.bin').read_bytes():raise RuntimeError('logit input IDs differ')
            metrics=json.loads(subprocess.check_output(['python3',str(ROOT/'tools/compare_logits.py'),str(args.out/f'logits-{case}-baseline.bin'),str(args.out/f'logits-{case}-{cfg["label"]}.bin')],text=True))
            if not all(math.isfinite(v) for v in metrics.values() if isinstance(v,float)):raise RuntimeError('invalid logit comparison')
            summary['logits'].append(dict(arm=cfg['label'],case=CASES[case][0],**metrics))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256_file(p)) for p in sorted(args.out.glob('*.bin'))]
    (args.out/'logit-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
