"""#467's declared analysis (docs/DECISION_CAMPAIGN_PLAN.json early_exit_tau09_467), from this bundle only.

  python3 benchmark/results/raw/decision-early-exit-tau09-2026-11-18/analysis.py > \
      benchmark/results/DECISION_EARLY_EXIT_TAU09_2026-11-18.json

Primary: the declared policy (the first probe at position >= 128 whose top softmax
probability over the options is >= 0.9 answers with its argmax; otherwise the full
answer) against full reasoning, decision_metrics.paired_quality at 2 pp. Secondary:
the numbers the plan lists, and the same numbers for tau 0.8 as a description (not a
test). Nothing else is computed: no subsets, no other thresholds. The declared hashes
and settings are asserted before any outcome is read. It is #286's analysis.py with
tau 0.9 and without its latency check, subsets and exploratory tables. No inference;
standard library and tools/ only."""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / 'tools'))
import decision_metrics as dm  # noqa: E402

TAU, TAU_DESCRIPTIVE, INTERVAL, MARGIN, CAP, N = 0.9, 0.8, 128, 2, 8192, 3000
DECLARED = {  # docs/DECISION_CAMPAIGN_PLAN.json early_exit_tau09_467 (PR #480, 3c81cf93)
    'question_ids_sha256': 'b40cb1ab04303efb39f982ea30e6f0eae732d099739bae31f4eca6b7c63d5354',
    'cases_sha256': 'b8cfe3f37824241f0514e7b01b2010f81e4540a9aa552f990fdcfe7ee127413e',
    'model_sha256_prefix': '3907dc16',
    'source_revision': 'e5e4aaf8a5d91120fda389b65966030ad4045051',
    'settings': {'profile': 'chat_reasoning', 'split': 'test', 'backend': 'metal', 'threads': 6, 'align': 320,
                 'cap': CAP, 'batch': 8, 'pad': False, 'probe': [198, 248069, 271], 'probe_interval': INTERVAL,
                 'exit_tau': 0.0, 'question_ids': None,
                 'environment': {'OMP_NUM_THREADS': '6', 'OMP_WAIT_POLICY': 'passive'}}}


def lines(name):
    return [line for line in gzip.decompress((HERE / name).read_bytes()).decode().split('\n') if line]


def top(logits):
    best = max(range(len(logits)), key=lambda i: logits[i])
    return 1 / math.fsum(math.exp(x - logits[best]) for x in logits), best


def exit_at(q, tau):
    """(answer, position) of the first confident probe before the cap, or None."""
    for p in q['probes']:
        if INTERVAL <= p['pos'] < CAP:
            prob, best = top(p['logits'])
            if prob >= tau:
                return best, p['pos']
    return None


def arms(main, target, tau):
    """Both arms of every question from its one recorded generation."""
    full, early, tokens_full, tokens_early, exits = {}, {}, 0, 0, {}
    for k, q in main.items():
        full[k] = bool(q['correct']) and q['stop'] == 'eos'
        tokens_full += q['generated_tokens']
        e = exit_at(q, tau)
        if e:
            exits[k] = e
            early[k] = e[0] == target[k]
            tokens_early += e[1]
        else:
            early[k] = full[k]
            tokens_early += q['generated_tokens']
    return full, early, tokens_full, tokens_early, exits


def paired(early, full):
    r = dm.paired_quality(early, full, MARGIN)
    return {k: r[k] for k in ('n', 'gains', 'losses', 'direct_accuracy', 'baseline_accuracy', 'difference_pp',
                              'interval_pp')}


def verify(targets, report):
    """The declared sample, build and settings; every check must hold before an outcome is read."""
    manifest, per_question = report['manifest'], report['summary']['per_question']
    ids = [t['question_id'] for t in targets]
    excluded = lines('excluded-question-ids.txt.gz')
    earlier = [json.loads(line)['question_id']
               for line in gzip.decompress((HERE.parent / 'decision-early-exit-default-2026-11-05' /
                                            'targets.jsonl.gz').read_bytes()).decode().split('\n') if line]
    checks = {
        'question_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'question_ids_sha256_as_declared': None,
        'cases_sha256': manifest['cases_sha256'], 'cases_sha256_as_declared': None,
        'model_sha256': manifest['model_sha256'], 'model_sha256_as_declared': None,
        'source_revision': manifest['source_revision'], 'source_revision_as_declared': None,
        'binary_sha256': manifest['binary_sha256'],
        'manifest_file_equals_report_manifest': json.loads((HERE / 'main/manifest.json').read_text()) == manifest,
        'settings': {k: manifest[k] for k in DECLARED['settings']},
        'settings_as_declared': None,
        'questions': len(ids), 'unique_questions': len(set(ids)),
        'questions_with_one_recorded_generation': len(set(per_question) & set(ids)),
        'recorded_order_is_the_prepared_order': manifest['order'] == [t['id'] for t in targets],
        'stop_reasons': {s: sum(q['stop'] == s for q in per_question.values())
                         for s in sorted({q['stop'] for q in per_question.values()})},
        'recorded_correct_equals_valid_answer_on_target': sum(
            (q['valid'] and q['predicted_index'] == t['target_index']) == bool(q['correct'])
            for t in targets for q in [per_question[t['question_id']]]),
        'probes_at_every_128_tokens_before_the_end': sum(
            [p['pos'] for p in q['probes']] == list(range(0, q['generated_tokens'], INTERVAL))
            for q in per_question.values()),
        'excluded_question_ids': len(excluded), 'excluded_unique': len(set(excluded)),
        'excluded_tail_is_the_286_sample': excluded[500:] == earlier,
        'overlap_with_excluded': len(set(excluded) & set(ids))}
    checks['question_ids_sha256_as_declared'] = checks['question_ids_sha256'] == DECLARED['question_ids_sha256']
    checks['cases_sha256_as_declared'] = checks['cases_sha256'] == DECLARED['cases_sha256']
    checks['model_sha256_as_declared'] = checks['model_sha256'].startswith(DECLARED['model_sha256_prefix'])
    checks['source_revision_as_declared'] = checks['source_revision'] == DECLARED['source_revision']
    checks['settings_as_declared'] = checks['settings'] == DECLARED['settings']
    assert all(v for k, v in checks.items() if k.endswith('_as_declared')), checks
    assert checks['manifest_file_equals_report_manifest'] and checks['recorded_order_is_the_prepared_order'], checks
    assert set(per_question) == set(ids) and len(ids) == len(set(ids)) == N, checks
    assert checks['stop_reasons'].keys() <= {'eos', 'limit'}, checks
    assert checks['recorded_correct_equals_valid_answer_on_target'] == N, checks
    assert checks['probes_at_every_128_tokens_before_the_end'] == N, checks
    assert all(q['generated_tokens'] <= CAP for q in per_question.values()), checks
    assert len(excluded) == len(set(excluded)) == 4000 and checks['excluded_tail_is_the_286_sample'], checks
    assert checks['overlap_with_excluded'] == 0, checks
    return checks


def describe(main_q, target, tau):
    full, early, tf, te, exits = arms(main_q, target, tau)
    capped = {k for k, q in main_q.items() if q['stop'] == 'limit'}
    p = paired(early, full)
    return p, {'full_reasoning': {'accuracy': p['baseline_accuracy'], 'correct': sum(full.values()),
                                  'generated_tokens': tf, 'capped': len(capped)},
               'early_exit': {'accuracy': p['direct_accuracy'], 'correct': sum(early.values()),
                              'gains': p['gains'], 'losses': p['losses'], 'early_exits': len(exits),
                              'still_capped': len(capped - set(exits)), 'generated_tokens': te,
                              'token_ratio': te / tf}}


def main():
    targets = [json.loads(line) for line in lines('targets.jsonl.gz')]
    target = {t['question_id']: t['target_index'] for t in targets}
    r_main = json.loads(gzip.decompress((HERE / 'main/report.json.gz').read_bytes()))
    checks = verify(targets, r_main)
    main_q = r_main['summary']['per_question']
    primary, secondary = describe(main_q, target, TAU)
    _, descriptive = describe(main_q, target, TAU_DESCRIPTIVE)
    # The plan's rule, literally: the lower bound is above -2 pp.
    primary['noninferior'] = primary['interval_pp'][0] > -MARGIN
    out = {
        'tracking_issue': 'geistlib#467', 'date': '2026-11-18',
        'plan': 'early_exit_tau09_467 (declared 3c81cf93, PR #480)',
        'checks': checks,
        'settings': {'cap': CAP, 'batch': 8, 'pad': False, 'probe_interval': INTERVAL, 'tau': TAU, 'margin_pp': MARGIN},
        'primary': {**primary, 'verdict': 'PASS' if primary['noninferior'] else 'FAIL'},
        'secondary': secondary,
        'descriptive_tau_0.8': {'note': 'declared as a description of this sample, not a test: no interval, no verdict',
                                'tau': TAU_DESCRIPTIVE, 'early_exit': descriptive['early_exit']}}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == '__main__':
    main()
