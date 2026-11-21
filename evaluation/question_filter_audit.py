"""Independent post-commit checks, using target references only for metric audits."""
import argparse
from pathlib import Path
import statistics

import numpy as np
import torch
from safetensors import safe_open

from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.question_filter_evaluate import paired
from evaluation.question_filter_math import rank, rank_consensus


def audit(root):
    plan = read_json(root / 'plan.json')
    summary = read_json(root / 'summary.json')
    barrier = read_json(root / 'evaluation-barrier.json')
    if not summary['all_scores_before_truth'] or not barrier['all_scores_and_candidates_verified_before_private_read']:
        raise ValueError('Missing commitment barrier')
    for item in barrier['files']:
        path = root / item['path']
        if file_hash(path) != item['sha256'] or path.stat().st_mtime > barrier['time']:
            raise ValueError('Score changed or committed after evaluation began')
    reference = read_json(Path('docs/experiments/data/federated-text-pilot/last-b-sweep-aggregate.json'))
    old = {(r['condition'], r['image'], r['method']): r for r in reference['aggregate']}
    checks = 0
    for row in summary['aggregate']:
        if row['method'] not in ('baseline', 'b_anchor'):
            continue
        original = old[row['condition'], row['image'],
                       'baseline' if row['method'] == 'baseline' else 'mix_l31_down']
        for part in ('question', 'answer', 'joint'):
            for suffix in ('recall', 'full_coverage'):
                key = part + '_' + suffix
                if abs(row[key] - original[key]) > 1e-12:
                    raise ValueError('Historical candidate metric changed')
                checks += 1
    rows = {(r['condition'], r['image'], r['index'], r['method']): r for r in summary['rows']}
    metric_checks, candidates, first_bundles = 0, 0, 0
    curves = []
    for condition in ('fedsgd-b1', 'noise', 'sparsify'):
        for image in ('random', 'gt'):
            for index in range(30):
                job = f'{condition}-{index:02d}'
                folder = root / 'candidates' / image
                meta = read_json(folder / (job + '.json'))
                forbidden = set(meta['forbidden_ids'])
                truth = read_tensors(Path(plan['source']) / 'runs' / job / 'captures' / image /
                                     'private/text_tokens.safetensors')
                q = set(truth['questions'].flatten().tolist()) - forbidden
                a = set(truth['targets'].flatten().tolist()) - forbidden
                bundle = read_tensors(folder / (job + '.safetensors'))
                for method, value in bundle.items():
                    chosen = set(value.tolist())
                    if len(chosen) != len(value) or len(chosen) != 109 or chosen & forbidden:
                        raise ValueError('Invalid candidate budget or special ID')
                    row = rows[condition, image, index, method]
                    for part, reference_tokens in (('question', q), ('answer', a), ('joint', q | a), ('exclusive', q - a)):
                        expected = (sum(t in chosen for t in reference_tokens) / len(reference_tokens)
                                    if reference_tokens else None)
                        if row[part + '_recall'] != expected:
                            raise ValueError('Independent candidate recall differs')
                        complete = all(t in chosen for t in reference_tokens) if reference_tokens else None
                        if row[part + '_full_coverage'] != complete:
                            raise ValueError('Independent full coverage differs')
                        metric_checks += 2
                    candidates += 1
                if not torch.equal(bundle['quota_single_20'], bundle['b_anchor']):
                    raise ValueError('Fixed quota failed to reproduce B anchor')
                original = read_tensors(Path(plan['source']) / 'runs' / job / 'attacks' /
                                        image / 'token_candidates.safetensors')
                flags = original['ambiguous']
                ambiguous = set(torch.nonzero(flags).flatten().tolist()) - forbidden
                with safe_open(str(Path(plan['previous']) / 'scores' / job / 'scores.safetensors'),
                               framework='pt', device='cpu') as stream:
                    b1, b2 = (stream.get_tensor(key) for key in ('l31_down', 'l32_o'))
                for kind, scores in (('single', b1), ('consensus', rank_consensus(b1, b2, flags, forbidden))):
                    order = rank(scores, flags, forbidden)
                    for k in (5, 10, 20, 40):
                        chosen = set(order[:k]) | ambiguous
                        curves.append({'condition': condition, 'image': image, 'kind': kind, 'k': k,
                            'candidate_count': len(chosen), 'answer_recall': len(chosen & a)/len(a),
                            'answer_full_coverage': a <= chosen})
                first = read_tensors(root / 'first' / image / (job + '.safetensors'))
                expected_methods = {'l1_o', 'l1_gate', 'l1_up', 'l1_down', 'l2_k', 'l2_v'}
                if set(first) != expected_methods | {'wrong_' + s for s in expected_methods} | {'pool'}:
                    raise ValueError('Incomplete shallow module surface')
                if first['pool'].shape != (256,):
                    raise ValueError('Changed first-token proposal budget')
                first_bundles += 1
    aggregate_checks = 0
    grouped = {}
    for row in summary['rows']:
        grouped.setdefault((row['condition'], row['image'], row['method']), []).append(row)
    for row in summary['aggregate']:
        values = grouped[row['condition'], row['image'], row['method']]
        for part in ('question', 'answer', 'joint', 'exclusive'):
            for suffix in ('recall', 'full_coverage'):
                key = part + '_' + suffix
                valid = [v[key] for v in values if v[key] is not None]
                expected = statistics.fmean(valid) if valid else None
                if row[key] != expected or row[key + '_n'] != len(valid):
                    raise ValueError('Independent aggregate differs')
                aggregate_checks += 1
    source = read_json(root / 'setup/source.json')
    for name, expected in source['code_files'].items():
        if file_hash(Path(source['snapshot']) / name) != expected:
            raise ValueError('Frozen scoring code changed')
    grouped_curves = {}
    for row in curves:
        key = (row['condition'], row['image'], row['kind'], row['k'])
        grouped_curves.setdefault(key, []).append(row)
    aggregated_curves = []
    for (condition, image, kind, k), group in grouped_curves.items():
        aggregated_curves.append({'condition': condition, 'image': image, 'kind': kind, 'k': k,
            'n': len(group), 'candidate_count': group[0]['candidate_count'],
            'answer_recall': statistics.fmean(r['answer_recall'] for r in group),
            'answer_full_coverage_count': sum(r['answer_full_coverage'] for r in group)})
    generated = []
    for condition in ('fedsgd-b1', 'noise', 'sparsify'):
        traces = [trace for index in range(30) for trace in read_json(root / 'priors' /
            f'{condition}-{index:02d}.json')['traces']]
        lengths = [len(t['generated_ids']) for t in traces]
        generated.append({'condition': condition, 'traces': len(traces),
            'eos_first_count': sum(n == 1 for n in lengths), 'max_steps_count': sum(n == 16 for n in lengths),
            'mean_steps': statistics.fmean(lengths)})
    write_json(root / 'budget-and-generation-audit.json', {'budget_curves': aggregated_curves,
        'generation': generated, 'contains_private_tokens': False,
        'budget_scope': 'Post-hoc fixed-K coverage of committed full ranks; K informative plus 9 ambiguous, not main 109 budget',
        'parameters_selected_by_these_curves': False})
    contrasts = []
    for condition in ('noise', 'sparsify'):
        for image in ('random', 'gt'):
            contrasts.append(paired(summary['rows'], 'bridge_q_matched', 'bridge_q_wrong', condition, image))
            selected = [r for r in summary['first_rows'] if (r['condition'], r['image'], r['method']) ==
                        (condition, image, 'l1_up')]
            wrong = {r['index']: r for r in summary['first_rows'] if (r['condition'], r['image'], r['method']) ==
                     (condition, image, 'wrong_l1_up')}
            difference = np.array([int(r['hit_5']) - int(wrong[r['index']]['hit_5']) for r in selected])
            rng = np.random.default_rng(20261105)
            boot = difference[rng.integers(len(difference), size=(10000, len(difference)))].mean(1)
            contrasts.append({'condition': condition, 'image': image, 'method': 'l1_up_first_top5',
                'control': 'wrong_l1_up', 'n': len(difference), 'gain': float(difference.mean()),
                'ci95': np.quantile(boot, [.025, .975]).tolist()})
    rank_stats = []
    for image in ('random', 'gt'):
        restricted = [row for path in (root / 'scores' / image).glob('sparsify-*.json')
                      for block in read_json(path)['diagnostics']['ridge'].values() for row in block]
        valid = [r for r in restricted if r['valid']]
        rank_stats.append({'image': image, 'rows': len(restricted), 'valid': len(valid),
            'full_observed_row_rank': sum(r['rank'] == r['support'] for r in valid),
            'mean_support': statistics.fmean(r['support'] for r in valid),
            'mean_rank': statistics.fmean(r['rank'] for r in valid)})
    metadata = [read_json(path) for path in (root / 'scores').glob('*/*.json')]
    write_json(root / 'supplemental-audit.json', {'paired': contrasts, 'restricted_public_rank': rank_stats,
        'baseline_max_score_delta': max(m['diagnostics']['baseline_max_delta'] for m in metadata),
        'contains_private_tokens': False, 'all_predictions_already_committed': True})
    write_json(root / 'finalization-audit.json', {'status': 'passed',
        'historical_metric_checks': checks, 'candidate_sets_checked': candidates,
        'independent_metric_checks': metric_checks, 'aggregate_checks': aggregate_checks,
        'first_bundles_checked': first_bundles, 'commitment_files_checked': len(barrier['files']),
        'all_budgets_109': True, 'fixed20_equals_historical_anchor': True,
        'all_six_context_modules_present': True, 'frozen_code_unchanged': True,
        'contains_private_tokens': False, 'private_reference_used_only_for_metric_audit': True,
        'summary_sha256': file_hash(root / 'summary.json'), 'code_sha256': file_hash(Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)
