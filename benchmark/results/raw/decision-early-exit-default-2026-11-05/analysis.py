"""#286's declared analysis (docs/DECISION_CAMPAIGN_PLAN.json early_exit_default_286), from this bundle only.

  python3 benchmark/results/raw/decision-early-exit-default-2026-11-05/analysis.py > \
      benchmark/results/DECISION_EARLY_EXIT_DEFAULT_2026-11-05.json

Primary: the frozen policy (the first probe at position >= 128 whose top softmax
probability over the options is >= 0.8 answers with its argmax; otherwise the full
answer) against full reasoning, decision_metrics.paired_quality at 2 pp. The rest is
secondary or, where marked, exploratory (chosen after seeing the test data: no
selection may use it). No inference; standard library and tools/ only."""
import collections
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / 'tools'))
import decision_metrics as dm  # noqa: E402

TAU, INTERVAL, MARGIN, CAP = 0.8, 128, 2, 8192


def report(name):
    return json.loads(gzip.decompress((HERE / name / 'report.json.gz').read_bytes()))


def top(logits):
    best = max(range(len(logits)), key=lambda i: logits[i])
    return 1 / math.fsum(math.exp(x - logits[best]) for x in logits), best


def exit_at(q, tau, cap=CAP):
    """(answer, position) of the first confident probe before cap, or None."""
    for p in q['probes']:
        if INTERVAL <= p['pos'] < cap:
            prob, best = top(p['logits'])
            if prob >= tau:
                return best, p['pos']
    return None


def pct(values, q):
    xs = sorted(values)
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def arms(main, target, tau, cap=CAP):
    """Both arms at (tau, cap); cap < 8192 truncates the same greedy trajectories."""
    full, early, tokens_full, tokens_early, exits = {}, {}, 0, 0, {}
    for k, q in main.items():
        finished = q['stop'] == 'eos' and q['generated_tokens'] <= cap
        full[k] = bool(q['correct']) and finished
        tokens_full += min(q['generated_tokens'], cap)
        e = exit_at(q, tau, cap)
        if e:
            exits[k] = e
            early[k] = e[0] == target[k]
            tokens_early += e[1]
        else:
            early[k] = full[k]
            tokens_early += min(q['generated_tokens'], cap)
    return full, early, tokens_full, tokens_early, exits


def paired(early, full):
    r = dm.paired_quality(early, full, MARGIN)
    return {k: r[k] for k in ('n', 'gains', 'losses', 'direct_accuracy', 'baseline_accuracy', 'difference_pp',
                              'interval_pp', 'noninferior')}


def main():
    targets = [json.loads(line) for line in gzip.decompress((HERE / 'targets.jsonl.gz').read_bytes()).decode().split('\n')
               if line]
    target = {t['question_id']: t['target_index'] for t in targets}
    subject = {t['question_id']: t['subject'] for t in targets}
    ids_sha = hashlib.sha256('\n'.join(t['question_id'] for t in targets).encode()).hexdigest()
    r_main = report('main')
    main_q = r_main['summary']['per_question']
    assert set(main_q) == set(target) and len(target) == 3500
    full, early, tf, te, exits = arms(main_q, target, TAU)
    primary = paired(early, full)
    capped = {k for k, q in main_q.items() if q['stop'] == 'limit'}
    gains = {k for k in full if early[k] and not full[k]}
    losses = {k for k in full if full[k] and not early[k]}
    finished = set(main_q) - capped

    def exit_prob(k):
        pos = exits[k][1]
        return top(next(p['logits'] for p in main_q[k]['probes'] if p['pos'] == pos))[0]

    by_subject = collections.defaultdict(lambda: collections.Counter())
    for k in main_q:
        c = by_subject[subject[k]]
        c['n'] += 1
        c['full'] += full[k]
        c['early_exit'] += early[k]
        c['gains'] += k in gains
        c['losses'] += k in losses
    exit_positions = collections.Counter(p for _, p in exits.values())

    # latency check (amendment 2026-10-30): both arms live on the same 300
    lat_ids = json.loads((HERE / 'campaign/latency_ids.json').read_text())
    live = {arm: report(name)['summary']['per_question'] for arm, name in (('early_exit', 'latency'),
                                                                           ('full', 'latency-full'))}
    assert all(set(v) == set(lat_ids) for v in live.values())

    def offline(k):
        e = exits.get(k)
        q = main_q[k]
        return (e[0], e[1], 'early_exit') if e else (q['predicted_index'] if q['valid'] else None,
                                                     q['generated_tokens'], q['stop'])

    def answer(q):
        return q['predicted_index'] if q['valid'] else None

    agreement = {
        'answers': sum(answer(live['early_exit'][k]) == offline(k)[0] for k in lat_ids),
        'answer_stop_and_position': sum((answer(live['early_exit'][k]), live['early_exit'][k]['generated_tokens'],
                                         live['early_exit'][k]['stop']) == offline(k) for k in lat_ids),
        'live_full_vs_main_run_answers_and_tokens': sum(
            answer(live['full'][k]) == answer(main_q[k]) and live['full'][k]['generated_tokens'] ==
            main_q[k]['generated_tokens'] for k in lat_ids)}
    latency = {}
    for arm, rows in (*live.items(), ('main_run_full_reference', {k: main_q[k] for k in lat_ids})):
        s = [rows[k]['latency_ms'] / 1000 for k in lat_ids]
        s_np = [(rows[k]['latency_ms'] - sum(rows[k]['probe_ms'])) / 1000 for k in lat_ids]
        latency[arm] = {'accuracy': sum(answer(rows[k]) == target[k] for k in lat_ids) / len(lat_ids),
                        'generated_tokens': sum(rows[k]['generated_tokens'] for k in lat_ids),
                        'capped': sum(rows[k]['stop'] == 'limit' for k in lat_ids),
                        'early_exits': sum(rows[k]['stop'] == 'early_exit' for k in lat_ids),
                        'latency_s': {'p50': pct(s, .5), 'p95': pct(s, .95), 'mean': statistics.mean(s)},
                        'latency_without_probes_s': {'p50': pct(s_np, .5), 'p95': pct(s_np, .95)}}
    for arm, name in (('early_exit', 'latency'), ('full', 'latency-full')):
        r = report(name)
        chunks = r['chunks']
        wall = sum(c['run']['wall_seconds'] for c in chunks)
        latency[arm]['throughput'] = {
            'chunks': len(chunks), 'cut_at_deadline': sum(bool(c['run'].get('stopped_at_deadline')) for c in chunks),
            'run_wall_s': wall, 'tokens_per_s_all_chunk_walls': r['summary']['generated_tokens'] / wall,
            'answers_per_hour': len(lat_ids) / (wall / 3600),
            'complete_driver_runs': r['summary']['complete_driver_runs']}
    live_paired = paired({k: answer(live['early_exit'][k]) == target[k] for k in lat_ids},
                         {k: answer(live['full'][k]) == target[k] for k in lat_ids})

    exploratory_tau = []
    for tau in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        f, e, a, b, x = arms(main_q, target, tau)
        p = paired(e, f)
        exploratory_tau.append({'tau': tau, 'accuracy': p['direct_accuracy'], 'gains': p['gains'],
                                'losses': p['losses'], 'interval_pp': p['interval_pp'], 'token_ratio': b / a,
                                'early_exits': len(x)})
    exploratory_cap = []
    for cap in (2048, 4096, 8192):
        f, e, a, b, _ = arms(main_q, target, TAU, cap)
        p = paired(e, f)
        exploratory_cap.append({'cap': cap, 'full_accuracy': p['baseline_accuracy'],
                                'early_exit_accuracy': p['direct_accuracy'], 'gains': p['gains'],
                                'losses': p['losses'], 'interval_pp': p['interval_pp'], 'token_ratio': b / a})

    out = {
        'tracking_issue': 'geistlib#286', 'date': '2026-11-05',
        'plan': 'early_exit_default_286 (declared d358ec3a, amendment 127451eb)',
        'question_ids_sha256': ids_sha, 'latency_question_ids_sha256':
            hashlib.sha256('\n'.join(lat_ids).encode()).hexdigest(),
        'model_sha256': r_main['manifest']['model_sha256'], 'binary_sha256': r_main['manifest']['binary_sha256'],
        'cases_sha256': r_main['manifest']['cases_sha256'],
        'settings': {'cap': CAP, 'batch': 8, 'pad': False, 'probe_interval': INTERVAL, 'tau': TAU, 'margin_pp': MARGIN},
        'primary': {**primary, 'verdict': 'PASS' if primary['noninferior'] else 'FAIL'},
        'secondary': {
            'full_reasoning': {'accuracy': primary['baseline_accuracy'], 'generated_tokens': tf,
                               'capped': len(capped)},
            'early_exit': {'accuracy': primary['direct_accuracy'], 'generated_tokens': te, 'token_ratio': te / tf,
                           'early_exits': len(exits), 'exits_at_first_probe': exit_positions[INTERVAL],
                           'still_capped': len(capped - set(exits))},
            'probes': {'recorded': r_main['summary']['probes'], 'probe_ms_total': r_main['summary']['probe_ms_total'],
                       'after_prompt_and_up_to_exit_or_end': sum(
                           sum(1 for p in q['probes'] if 0 < p['pos'] <= exits.get(k, (0, 10 ** 9))[1])
                           for k, q in main_q.items())},
            'capped_questions': {'n': len(capped), 'exit': len(capped & set(exits)),
                                 'exit_correct': sum(early[k] for k in capped & set(exits))},
            'gains': {'n': len(gains), 'on_capped': len(gains & capped)},
            'losses': {'n': len(losses), 'on_capped': len(losses & capped),
                       'exit_at_first_probe': sum(exits[k][1] == INTERVAL for k in losses),
                       'exit_probability_below_0.9': sum(exit_prob(k) < 0.9 for k in losses)},
            'finished_by_full_reasoning_post_hoc': paired({k: early[k] for k in finished}, {k: full[k] for k in finished}),
            'exit_accuracy_by_position': {str(pos): {'n': n, 'early_exit': sum(early[k] for k in exits if exits[k][1] == pos) / n,
                                                     'full_same_questions': sum(full[k] for k in exits if exits[k][1] == pos) / n}
                                          for pos, n in sorted(exit_positions.items()) if pos <= 512},
            'by_subject': {s: dict(c) for s, c in sorted(by_subject.items())}},
        'latency_check': {'agreement': agreement, 'arms': latency, 'live_paired': live_paired},
        'exploratory': {'note': 'chosen after seeing the test data; not usable for any selection',
                        'tau_sweep': exploratory_tau, 'cap_truncation': exploratory_cap}}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == '__main__':
    main()
