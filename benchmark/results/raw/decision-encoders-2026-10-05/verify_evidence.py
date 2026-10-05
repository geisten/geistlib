#!/usr/bin/env python3
"""Read-only verification of archived hashes, populations and derived metrics."""
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile

bundle = Path(__file__).resolve().parent
root = bundle.parents[3]
sys.path.insert(0, str(root/'tools'))
import bench_decision as bench
import decision_encoders as encoders
import eval_decision_encoders as evaluation

summary = encoders.read_json(root/'benchmark/results/DECISION_ENCODERS_2026-10-05.json')
index = bundle/'SHA256SUMS'
if encoders.sha256(index) != summary['raw_bundle_sha256sums_sha256']:
    raise ValueError('index/summary checksum mismatch')
listed = {}
for line in index.read_text().splitlines():
    digest, relative = line.split('  ',1)
    path = bundle/relative
    if relative in listed or not path.resolve().is_relative_to(bundle.resolve()):
        raise ValueError('duplicate or escaping index path')
    if encoders.sha256(path) != digest:
        raise ValueError('checksum mismatch: '+relative)
    listed[relative] = path.stat().st_size
actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file() and p != index}
if (set(listed) != actual or len(listed) != summary['raw_bundle_files_excluding_index']
        or sum(listed.values()) != summary['raw_bundle_bytes_excluding_index']):
    raise ValueError('bundle inventory/size mismatch')

def reject_constant(value):
    raise ValueError('nonfinite JSON: '+value)

def samples(run):
    with io.StringIO(gzip.decompress((run/'samples.jsonl.gz').read_bytes()).decode('utf-8')) as stream:
        return [json.loads(line, parse_constant=reject_constant) for line in stream if line.strip()]

def cases(directory):
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder)
        for name in ('metadata.json','split_audit.json'):
            (target/name).write_bytes((directory/name).read_bytes())
        (target/'development.jsonl').write_bytes(gzip.decompress((directory/'development.jsonl.gz').read_bytes()))
        return evaluation.load_cases(target,'development','pilot')

reports = 0
forward_calls = 0
for group in ('pilot','development'):
    for entry in summary[group]:
        run = bundle/group/entry['run']
        report = encoders.read_json(run/'report.json')
        manifest = encoders.read_json(run/'manifest.json')
        rows_dir = (bundle/group/'native-context-continuation-cases'
                    if entry['run'].endswith('-continuation') else bundle/group/'cases')
        meta, rows = cases(rows_dir)
        raw = gzip.decompress((run/'samples.jsonl.gz').read_bytes())
        import hashlib
        if (hashlib.sha256(raw).hexdigest() != report['samples_sha256']
                or encoders.sha256(run/'manifest.json') != report['manifest_sha256']
                or manifest['dataset'] != meta):
            raise ValueError('report/raw/dataset binding mismatch')
        trials = samples(run)
        grouped = {}
        for trial in trials:
            grouped.setdefault(trial['case_id'],[]).append(trial)
        if set(grouped) != {row['id'] for row in rows}:
            raise ValueError('unexpected or missing case identities')
        for values in grouped.values():
            rejected = [v for v in values if v['status'] == 'rejected']
            if rejected:
                if len(values) != 1:
                    raise ValueError('mixed rejection/scored trials')
            elif (len(values) != manifest['warmup']+manifest['repeats']
                  or {v['trial'] for v in values} != set(range(manifest['warmup']+manifest['repeats']))):
                raise ValueError('incomplete trials')
        result = evaluation.summarize(trials,rows,manifest['warmup'])
        for key in ('questions','scored','correct','rejected','coverage','accuracy_rejections_count_incorrect',
                    'quality_on_scored_only','predictions_stable','max_repeat_logit_abs_drift',
                    'warm_request_p50_ms','warm_forward_p50_ms','per_question_timings'):
            if result[key] != report[key]:
                raise ValueError('derived report mismatch: '+key)
        if (result['quality_on_scored_only'] != entry['quality_raw_on_scored_only']
                or result['quality_checkpoint_scaled_on_scored_only'] != entry['quality_checkpoint_scaled_on_scored_only']):
            raise ValueError('raw/checkpoint probability metric mismatch')
        warm = [s['request_ms'] for s in trials if s['status']=='scored' and s['trial']>=manifest['warmup']]
        if (entry['warm_request_p50_ms'] != bench.percentile(warm,.5)
                or entry['warm_request_p95_ms'] != bench.percentile(warm,.95)):
            raise ValueError('summary timing mismatch')
        if not entry['run'].endswith('-continuation'):
            forward_calls += sum(s['status']=='scored' for s in trials)
        reports += 1

_, original_rows = cases(bundle/'development'/'cases')
failed = bundle/'development'/'mps-modernbert-shots5-native-context'
continuation = bundle/'development'/'mps-modernbert-shots5-native-context-continuation'
if (failed/'report.json').exists():
    raise ValueError('failed continuous run presented as successful')
failure = encoders.read_json(failed/'failure.json')
first = samples(failed)
second = samples(continuation)
if (failure['completed_sample_records'] != len(first)
        or {s['case_id'] for s in first} & {s['case_id'] for s in second}
        or {s['case_id'] for s in first+second} != {r['id'] for r in original_rows}):
    raise ValueError('failure/recovery population mismatch')
recovered = evaluation.summarize(first+second,original_rows,1)
expected = summary['recovered_geometry'][0]
saved = expected['quality_result_after_explicit_restart']
for key,value in saved.items():
    if recovered[key] != value:
        raise ValueError('recovered metric mismatch: '+key)
for field, fraction in (('warm_request_p50_ms',.5),('warm_request_p95_ms',.95)):
    if expected[field] != bench.percentile([s['request_ms'] for s in first+second if s['trial']==1],fraction):
        raise ValueError('recovered timing mismatch')
forward_calls += len(first)+len(second)
if forward_calls != summary['model_forward_calls_in_completed_policies_and_recovered_geometry']:
    raise ValueError('forward-call inventory mismatch')
print(f'PASS: {len(listed)} indexed files, {reports} complete reports, disjoint 614+164 recovery; '
      f'{forward_calls} forward calls; raw/temperature metrics and p50/p95 regenerated')
