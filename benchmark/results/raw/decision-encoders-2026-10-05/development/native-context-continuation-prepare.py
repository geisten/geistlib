"""Prepare a disjoint continuation; never turn the failed run into success."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root = Path('/private/tmp/decision-encoder-development-2026-10-05')
sys.path.insert(0, '/Users/germar/.codex/worktrees/decision-inference/geistlib/tools')
import decision_encoders as encoders
import eval_decision_encoders as evaluation

meta, rows = evaluation.load_cases(root/'cases', 'development', 'pilot')
failed = root/'mps-modernbert-shots5-native-context'
with (failed/'samples.jsonl').open(encoding='utf-8') as stream:
    samples = [json.loads(line) for line in stream]
grouped = {}
for sample in samples:
    grouped.setdefault(sample['case_id'], []).append(sample)
if any({s['trial'] for s in trials} != {0, 1} for trials in grouped.values()):
    raise ValueError('partially recorded question needs independent recovery policy')
pending = [row for row in rows if row['id'] not in grouped]
if len(pending) != 164 or len(grouped) != 614:
    raise ValueError('unexpected recovery population')
destination = root/'native-context-continuation-cases'
destination.mkdir(exist_ok=False)
shutil.copyfile(root/'cases'/'split_audit.json', destination/'split_audit.json')
path = destination/'development.jsonl'
path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in pending))
meta.update(questions=len(pending), records=len(pending), cases_sha256=encoders.sha256(path),
            recovery={'original_population_questions': len(rows), 'completed_questions': len(grouped),
                      'failed_manifest_sha256': encoders.sha256(failed/'manifest.json'),
                      'failed_samples_sha256': encoders.sha256(failed/'samples.jsonl'),
                      'selection': 'only questions with no completed trials, original order',
                      'reason': 'explicit fresh-process continuation after MPS OOM; no automatic fallback'})
(destination/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
evaluation.load_cases(destination, 'development', 'pilot')
print('Disjoint continuation:', len(pending), 'questions')
