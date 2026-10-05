"""Serial, frozen eight-question CPU/MPS development matrix."""
from pathlib import Path
import json
import os
import subprocess
import time

root = Path(__file__).resolve().parent
worktree = Path('/Users/germar/.codex/worktrees/decision-inference/geistlib')
python = '/private/tmp/decision-encoder-env/bin/python'
env = {**os.environ, 'OMP_WAIT_POLICY': 'PASSIVE', 'OMP_NUM_THREADS': '6',
       'VECLIB_MAXIMUM_THREADS': '6', 'PYTORCH_ENABLE_MPS_FALLBACK': '0',
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_TF': '0'}
models = {
    'modernbert': ('/private/tmp/decision-encoder-models/modernbert',
                   'answerdotai/ModernBERT-Large-Instruct', '9943452941e79c8c35ede72e78a38a8175a79bb5'),
    'laya': ('/private/tmp/decision-encoder-models/laya/typed-decisions',
             'convaiinnovations/laya', '7b928d828b7b0e022f929d9bd2e44165aa270148'),
}
plan = [{'device': device, 'adapter': adapter, 'shots': shots}
        for device in ('mps', 'cpu') for adapter in ('modernbert', 'laya') for shots in (0, 5)]
(root / 'matrix_plan.json').write_text(json.dumps({'scope': 'development pilot; no quality acceptance',
    'questions': 8, 'policies': plan, 'dtype': 'float32', 'warmup': 1, 'repeats': 1,
    'margin_pp': 2, 'threads': 6, 'concurrency': 1}, indent=2) + '\n')
start = time.perf_counter()
for policy in plan:
    device, adapter, shots = policy['device'], policy['adapter'], policy['shots']
    path, repo, revision = models[adapter]
    name = f'{device}-{adapter}-shots{shots}'
    out = root / name
    command = [python, str(worktree/'tools/eval_decision_encoders.py'), '--enable-encoder-backends',
               '--cases-dir', str(root/'cases'), '--split', 'development', '--purpose', 'pilot',
               '--adapter', adapter, '--model-dir', path, '--model-repo', repo, '--revision', revision,
               '--device', device, '--dtype', 'float32', '--max-tokens', '1024', '--threads', '6',
               '--shots', str(shots), '--warmup', '1', '--repeats', '1', '--margin-pp', '2',
               '--out-dir', str(out)]
    print('START', name, flush=True)
    with (root/f'{name}.log').open('w') as log:
        result = subprocess.run(command, cwd=worktree, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print('FAILED', name, 'see', root/f'{name}.log', flush=True)
        raise SystemExit(result.returncode)
    report = json.loads((out/'report.json').read_text())
    print('DONE', name, json.dumps({k:report[k] for k in ('correct','questions','rejected','warm_request_p50_ms','predictions_stable')}), flush=True)
print('matrix_wall_seconds', time.perf_counter()-start, flush=True)
