"""Fresh-process diagnostic for the failed MPS request, not a quality run."""
import json
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0',
                  PYTORCH_ENABLE_MPS_FALLBACK='0')
root = Path('/private/tmp/decision-encoder-development-2026-10-05')
sys.path.insert(0, '/Users/germar/.codex/worktrees/decision-inference/geistlib/tools')
import decision_encoders as encoders
import eval_decision_encoders as evaluation
import torch
from transformers import AutoTokenizer

_, rows = evaluation.load_cases(root/'cases', 'development', 'pilot')
run = root/'mps-modernbert-shots5-native-context'
with (run/'samples.jsonl').open(encoding='utf-8') as stream:
    samples = [json.loads(line) for line in stream]
completed = {s['case_id'] for s in samples if s.get('trial') == 1}
row = next(row for row in rows if row['id'] not in completed)
directory = Path('/private/tmp/decision-encoder-models/modernbert')
tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
backend = encoders.ModernBertBackend.__new__(encoders.ModernBertBackend)
backend.tokenizer, backend.max_tokens = tokenizer, 8192
lengths = [len(backend.prepare(evaluation.mmlu_request(r, 5))['input_ids']) for r in rows]
report = {'protocol': 'geist-encoder-context-diagnostic-v1', 'quality_result': False,
          'context_cap': 8192, 'completed_questions_before_failure': len(completed),
          'failed_case_id': row['id'], 'failed_question_id': row['question_id'],
          'failed_prompt_tokens': lengths[rows.index(row)],
          'max_preceding_prompt_tokens': max(lengths[:rows.index(row)]),
          'max_population_prompt_tokens': max(lengths), 'population_questions': len(rows)}
torch.set_num_threads(6)
torch.set_num_interop_threads(1)
backend = encoders.create_backend('modernbert', directory, enabled=True, device='mps',
                                 dtype='float32', max_tokens=8192)
def memory():
    torch.mps.synchronize()
    return {'current_allocated_bytes': torch.mps.current_allocated_memory(),
            'driver_allocated_bytes': torch.mps.driver_allocated_memory(),
            'recommended_max_bytes': torch.mps.recommended_max_memory()}
report['fresh_process_memory_after_load'] = memory()
try:
    result = backend.score(evaluation.mmlu_request(row, 5))
    report.update(isolated_request_succeeded=True, isolated_request=result,
                  fresh_process_memory_after_request=memory())
except Exception as exc:
    report.update(isolated_request_succeeded=False, error=type(exc).__name__, message=str(exc))
finally:
    backend.close()
    (root/'native-context-diagnostic.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'isolated_request'}, indent=2))
