"""Post-commit target evaluation and conservative six-family advancement gates."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.question_filter_math import metrics
from evaluation.question_filter_study import (CONDITIONS, IMAGES, baseline, load_verified)
from evaluation.soft_residual_filter_study import verify_plan


FIELDS = tuple(p + '_' + m for p in ('question', 'answer', 'joint', 'exclusive')
               for m in ('recall', 'full_coverage')) + ('informative_ap', 'selected_count')


def aggregate(rows, keys, fields=FIELDS):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in buckets.items():
        row = dict(zip(keys, key, strict=True), n=len(group))
        for field in fields:
            values = [r[field] for r in group if r.get(field) is not None]
            row[field] = statistics.fmean(values) if values else None
            row[field + '_n'] = len(values)
            if field.endswith('full_coverage'):
                row[field + '_count'] = sum(values)
        result.append(row)
    return result


def paired(rows, method, control, condition, image):
    current = {r['index']: r for r in rows if (r['condition'], r['image'], r['method']) ==
               (condition, image, method)}
    anchor = {r['index']: r for r in rows if (r['condition'], r['image'], r['method']) ==
              (condition, image, control)}
    if not current or current.keys() != anchor.keys():
        return None
    result = {'condition': condition, 'image': image, 'method': method, 'control': control,
              'n': len(current)}
    rng = np.random.default_rng(20261105)
    for field in ('exclusive_recall', 'question_recall', 'answer_recall', 'joint_full_coverage'):
        differences = np.asarray([current[i][field] - anchor[i][field] for i in sorted(current)
                                  if current[i][field] is not None and anchor[i][field] is not None])
        if not len(differences):
            result[field + '_delta'] = None
            continue
        samples = differences[rng.integers(len(differences), size=(10000, len(differences)))].mean(1)
        result.update({field + '_delta': float(differences.mean()),
            field + '_ci95': np.quantile(samples, [.025, .975]).tolist(),
            field + '_wins': int((differences > 1e-12).sum()),
            field + '_losses': int((differences < -1e-12).sum())})
    return result


def first_metrics(order, pool, truth, ambiguous):
    return {'pool_hit': truth in pool, 'ambiguous': truth in ambiguous,
            **{f'hit_{k}': truth in order[:k] for k in (1, 5, 20)}}


def gates(overall, first, calibration):
    lookup = {(r['condition'], r['image'], r['method']): r for r in overall}
    outcomes = []
    for condition in ('noise', 'sparsify'):
        anchor = lookup[condition, 'random', 'b_anchor']
        # Each family has a primary method fixed before target scores. Other scores remain descriptive.
        choices = [('E1', 'bridge_q_matched', 'bridge_q_blind', 'bridge_q_wrong'),
                   ('E3', 'mix_energy_kv', 'b_anchor', 'mix_wrong_energy_kv'),
                   ('E6', 'mix_b_forward', 'mix_b_permuted', 'mix_wrong_b_forward'),
                   ('E6-input', 'mix_b_input', 'b_anchor', 'mix_wrong_b_input')]
        if condition == 'sparsify':
            choices.append(('E2', 'mix_ridge', 'mix_row_kv', 'mix_wrong_ridge'))
        for family, name, control, wrong in choices:
            row = lookup[condition, 'random', name]
            compare = lookup[condition, 'random', control]
            donor = lookup[condition, 'random', wrong]
            exclusive_gain = row['exclusive_recall'] - compare['exclusive_recall']
            question_coverage = row['question_full_coverage'] >= compare['question_full_coverage']
            specific = row['exclusive_recall'] > donor['exclusive_recall']
            answer_safe = row['answer_recall'] >= anchor['answer_recall'] - .02
            joint_safe = row['joint_full_coverage'] >= anchor['joint_full_coverage']
            public_passed = calibration['audit_passed'] if family == 'E2' else True
            passed = exclusive_gain >= .05 and question_coverage and specific and answer_safe and joint_safe and public_passed
            outcomes.append({'family': family, 'condition': condition, 'method': name,
                'control': control, 'wrong': wrong, 'exclusive_gain': exclusive_gain,
                'question_coverage_nondecreasing': question_coverage, 'upload_specific': specific,
                'answer_nondecreasing_with_tolerance': answer_safe, 'joint_nondecreasing': joint_safe,
                'public_audit_passed': public_passed, 'passed': bool(passed)})
    # E4 is a separate first-position mechanism gate, never an end-to-end filter promotion.
    for condition in ('noise', 'sparsify'):
        for method in ('l1_o', 'l1_gate', 'l1_up', 'l1_down', 'l2_k', 'l2_v'):
            selected = [r for r in first if r['condition'] == condition and r['image'] == 'random'
                        and r['method'] == method and not r['ambiguous']]
            by_index = {r['index']: r for r in first if r['condition'] == condition and r['image'] == 'random'
                        and r['method'] == 'energy_kv' and not r['ambiguous']}
            wrong_rows = {r['index']: r for r in first if r['condition'] == condition and r['image'] == 'random'
                          and r['method'] == 'wrong_' + method and not r['ambiguous']}
            gains = sum(r['hit_5'] - by_index[r['index']]['hit_5'] for r in selected)
            specific = sum(r['hit_5'] - wrong_rows[r['index']]['hit_5'] for r in selected)
            outcomes.append({'family': 'E4', 'condition': condition, 'method': method,
                'informative_n': len(selected), 'first_top5_net_gain': gains,
                'wrong_net_gain': specific, 'passed': gains >= 3 and specific > 0,
                'scope': 'first token only; cannot promote to full question filter'})
    eligible = [r for r in outcomes if r['passed'] and r['family'] != 'E4']
    return {'outcomes': outcomes, 'any_full_filter_passed': bool(eligible),
            'eligible': eligible, 'selection_is_development_only': True}


def evaluate(root):
    plan = verify_plan(root)
    complete = read_json(root / 'all-candidates-complete.json')
    if complete['status'] != 'completed' or complete['groups'] != 180:
        raise ValueError('Every candidate bundle must commit before private evaluation')
    files = []
    for pattern in ('scores/*/*.safetensors', 'first/*/*.safetensors', 'priors/*.safetensors',
                    'candidates/*/*.safetensors', 'ranks/*/*.safetensors'):
        for path in sorted(root.glob(pattern)):
            load_verified(path)
            files.append({'path': str(path.relative_to(root)), 'sha256': file_hash(path)})
    # Assert the expected surface explicitly; a matching controller label alone is insufficient.
    for condition in CONDITIONS:
        for index in range(30):
            for image in IMAGES:
                for folder in ('scores', 'first', 'candidates', 'ranks'):
                    load_verified(root / folder / image / f'{condition}-{index:02d}.safetensors')
    write_json(root / 'evaluation-barrier.json', {'time': time.time(), 'files': files,
        'all_scores_and_candidates_verified_before_private_read': True})
    rows, first = [], []
    for condition in CONDITIONS:
        for index in range(30):
            job = f'{condition}-{index:02d}'
            for image in IMAGES:
                path = root / 'candidates' / image / (job + '.safetensors')
                candidates = load_verified(path)
                forbidden = set(read_json(path.with_suffix('.json'))['forbidden_ids'])
                original = baseline(root, plan, condition, index, image)
                ambiguous = set(torch.nonzero(original['ambiguous']).flatten().tolist()) - forbidden
                # This is the first target-reference read in this entry point.
                truth = read_tensors(Path(plan['source']) / 'runs' / job / 'captures' / image /
                                     'private/text_tokens.safetensors')
                q = set(truth['questions'].flatten().tolist()) - forbidden
                a = set(truth['targets'].flatten().tolist()) - forbidden
                ranks = load_verified(root / 'ranks' / image / (job + '.safetensors'))
                for method, tokens in candidates.items():
                    selected = set(tokens.tolist())
                    if len(selected) != 109:
                        raise ValueError('Candidate budget violated')
                    row = dict(condition=condition, image=image, index=index, method=method,
                        length_bin=plan['length_bins_by_index'][f'{index:02d}'], **metrics(selected, q, a))
                    informative = sorted((q | a) - ambiguous)
                    positions = sorted(ranks[method][informative].tolist()) if method in ranks else []
                    if positions and min(positions) <= 0:
                        raise ValueError('Informative reference excluded from ranking')
                    row['informative_ap'] = statistics.fmean((i+1)/r for i, r in enumerate(positions)) if positions else None
                    rows.append(row)
                shallow = load_verified(root / 'first' / image / (job + '.safetensors'))
                pool = shallow.pop('pool').tolist()
                matrix = load_verified(root / 'scores' / image / (job + '.safetensors'))
                shallow['energy_kv'] = matrix['energy_kv'][pool]
                shallow['baseline'] = original['scores'][pool]
                for method, value in shallow.items():
                    eligible = [i for i, token in enumerate(pool) if token not in forbidden and token not in ambiguous]
                    order = [pool[i] for i in sorted(eligible, key=lambda i: (float(value[i]), pool[i]))]
                    true_first = int(truth['questions'][0, 0])
                    first.append(dict(condition=condition, image=image, index=index, method=method,
                        **first_metrics(order, pool, true_first, ambiguous)))
    overall = aggregate(rows, ('condition', 'image', 'method'))
    gate = gates(overall, first, read_json(root / 'calibration.json'))
    pairs = []
    for condition in CONDITIONS:
        for image in IMAGES:
            methods = {r['method'] for r in rows if r['condition'] == condition and r['image'] == image}
            for method in sorted(methods):
                if method.startswith('wrong_') or method == 'baseline':
                    continue
                control = ('bridge_q_blind' if method == 'bridge_q_matched' else
                           'row_kv' if method == 'ridge' else 'baseline')
                result = paired(rows, method, control, condition, image)
                if result:
                    pairs.append(result)
    summary = {'status': 'completed', 'aggregate': overall,
        'by_length': aggregate(rows, ('condition', 'image', 'method', 'length_bin')),
        'first': aggregate(first, ('condition', 'image', 'method'), ('pool_hit', 'hit_1', 'hit_5', 'hit_20')),
        'first_informative': aggregate([r for r in first if not r['ambiguous']],
            ('condition', 'image', 'method'), ('pool_hit', 'hit_1', 'hit_5', 'hit_20')),
        'gates': gate, 'paired': pairs, 'rows': rows, 'first_rows': first,
        'all_scores_before_truth': True, 'contains_private_tokens': False,
        'independent_validation': False, 'source_sha256': plan['source_sha256']}
    write_json(root / 'summary.json', summary)
    write_json(root / 'phase-gates.json', gate)
    # A failed gate closes optional phases without silently changing the method.
    if not gate['any_full_filter_passed']:
        write_json(root / 'next-phases.json', {'P2': 'not_run_no_full_filter_gate',
            'P3': 'not_run_no_full_filter_gate', 'P4': 'not_run_no_full_filter_gate',
            'six_stage1_directions_evaluated': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    state = args.output / 'state-evaluate.json'
    if state.exists():
        raise FileExistsError('No implicit evaluation retry')
    torch.set_num_threads(4)
    write_json(state, {'status': 'running', 'time': time.time()})
    try:
        evaluate(args.output)
        write_json(state, {'status': 'completed', 'time': time.time()})
    except BaseException:
        write_json(state, {'status': 'failed', 'traceback': traceback.format_exc(), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
